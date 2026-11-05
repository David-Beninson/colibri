"""LangGraph & zero-dependency StateGraph orchestration engine for Colibri (`src/ai/graph.py`).

Targets Qwen3.6-35B-A3B (`qwen36-35b`) served via `coli serve` with:
- Unified deep `ColibriAgent` runner with StateGraph compatibility (`.invoke()`, `.run()`)
- Pluggable `ColibriTransport` network port seam with default `HttpTransport` (SSE streaming + JSON)
- Stateful KV + DeltaNet prefix caching (`cache_slot`, static system prompt & tool definitions)
- Thinking-block preservation across tool turns (`preserve_thinking: True`)
- Automatic server-side context compaction (`context_compaction: "auto"`)
- Parallel SSRF-safe tool execution delegating to `TOOL_REGISTRY_OBJ` with output truncation
- Step-budget & malformed-tool reflection guard
- Zero-roundtrip speculative local RAG memory prefetching
- Native persistent SQLite session checkpointing (`SqliteSessionStore` / `ColibriSqliteCheckpointer`)
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, TypedDict, Union

from .config import AgentConfig
from .context import ConversationContext
from .gateway import (
    ColibriGatewayPort,
    HttpGatewayAdapter,
    InMemoryGatewayAdapter,
)
from .session import (
    BaseCheckpointSaver,
    CheckpointTuple,
    InMemorySessionStore,
    SessionStore,
    SqliteSessionStore,
    get_session_history,
)
from .tools import TOOL_SUITE_OBJ, ToolSuite


# Default fallback persona if `src/ai/persona.py` is not yet present
_FALLBACK_PERSONA_PROMPT = (
    "You are Jimmy bot (ג'ימי בוט), a warm, intelligent, bilingual (Hebrew & English) "
    "AI companion serving the Jimmies family (משפחת ג'ימי), running locally on Qwen3.6-35B-A3B.\n"
    "Guidelines:\n"
    "1. Strict Spoken Brevity: Keep default spoken replies to 1–2 short, natural sentences.\n"
    "2. Internet & Research: Use `web_search`, `fetch_webpage_content`, `deep_web_research`, "
    "`wikipedia_search`, `arxiv_search`, `hackernews_search`, `github_repo_search`, `weather_lookup`, "
    "or `fetch_json_or_rss` for live facts, news, weather, and citations.\n"
    "3. Voice & Bilingual Speech: Respond naturally in the user's language (Hebrew or English). "
    "Use `transcribe_audio` for speech input and `text_to_speech` / `speak_text` when requested.\n"
    "4. Personal Memory (RAG): Use `query_memory`, `save_fact`, `ingest_url_or_note`, and `scale_recipe` "
    "to remember and recall personal facts, dates, preferences, recipes, and Jimmies family memories.\n"
    "5. Exact Computation: Use `calculate` or `run_python_calculation` for math and date calculations."
)


def _resolve_system_prompt() -> str:
    try:
        from .persona import build_system_prompt

        return build_system_prompt()
    except Exception:  # noqa: BLE001
        try:
            from .persona import COMPANION_PERSONA_PROMPT

            return str(COMPANION_PERSONA_PROMPT)
        except Exception:  # noqa: BLE001
            return _FALLBACK_PERSONA_PROMPT


# Static system prompt: kept byte-identical at index 0 of `messages` on every turn
# so Colibri's `cache_slot` KV + DeltaNet prefix cache is never invalidated.
SYSTEM_PROMPT = _resolve_system_prompt()

_SYNTHESIS_NUDGE = (
    "You have reached the tool-calling step limit or emitted an incomplete tool call. "
    "Do NOT call any more tools. Synthesize a comprehensive, well-cited final answer "
    "now using all evidence gathered so far."
)

_MALFORMED_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL)


def _colibri_extra_body(config: AgentConfig, active_slot: Optional[int] = None) -> Dict[str, Any]:
    """Build Colibri `/v1/chat/completions` extension fields for Qwen3.6 prefix reuse."""
    if getattr(config, "is_ollama_or_metal", False):
        return {}
    body = config.extra_body(slot=active_slot)
    # Explicitly ensure preserve_thinking is True when not overridden so Qwen3.6
    # reuses the KV & DeltaNet prefix across multi-step tool turns.
    if config.preserve_thinking is None:
        body["preserve_thinking"] = True
    else:
        body["preserve_thinking"] = bool(config.preserve_thinking)
    body["context_compaction"] = config.compact
    body["cache_slot"] = config.cache_slot if active_slot is None else active_slot
    return body


_RESEARCH_KEYWORDS_RE = re.compile(
    r"\b(arxiv|wikipedia|wiki|hackernews|hacker news|github|repo|rss|feed|ocr|image|photo|camera|python|script|deep research|article|http://|https://)\b",
    re.IGNORECASE,
)

_URL_INGEST_RE = re.compile(
    r"(https?://\S+|\b(?:ingest|scrape|bookmark|save\s+(?:this\s+)?(?:url|link|recipe|article|note)|"
    r"remember\s+this\s+(?:recipe|url|link|note)|שמור\s+(?:את\s+)?(?:המתכון|הקישור|הלינק))\b)",
    re.IGNORECASE,
)

_RECIPE_SCALING_RE = re.compile(
    r"\b(scale\s+(?:the\s+)?recipe|scale\s+to\s+\d+|for\s+\d+\s+(?:servings?|people|guests|portions?)|"
    r"double\s+the\s+(?:recipe|ingredients?)|halve\s+the\s+(?:recipe|ingredients?)|"
    r"convert\s+(?:servings?|ingredients?)|הכפל\s+את\s+המתכון|ל-\d+\s+מנות)\b",
    re.IGNORECASE,
)

_MEMORY_RECALL_RE = re.compile(
    r"\b(what\s+(?:is|are|was|were)\s+(?:her|his|my|our|the\s+jimmies|jimmy(?:'s)?)\s+(?:favorite|saved|preferred)|"
    r"do\s+you\s+remember|recall|from\s+(?:my|our)\s+(?:memory|vault|notes|bookmarks|recipes)|"
    r"what\s+saved\s+(?:recipes?|bookmarks?|facts?|notes?)|what\s+.*\s+does\s+(?:she|he)\s+love|"
    r"מה\s+(?:היא|הוא|אנחנו)\s+אוהב|זוכר)\b",
    re.IGNORECASE,
)

_FACTUAL_EXTRACTION_RE = re.compile(
    r"\b(weather|forecast|temperature|calculate|compute|sqrt|sum|solve|search|find|latest|news|"
    r"arxiv|wikipedia|wiki|hackernews|github|rss|price|stock|who\s+is|what\s+is\s+the\s+capital|"
    r"מזג\s+האוויר|חשב|חפש)\b|[0-9]+\s*[\+\-\*\/\^]\s*[0-9]+",
    re.IGNORECASE,
)


def classify_user_intent(prompt_text: str, config: Optional[AgentConfig] = None) -> str:
    """Classify user request into one of 5 canonical intents:
    - `"url_ingestion"`
    - `"recipe_scaling"`
    - `"memory_recall"`
    - `"factual_extraction"`
    - `"conversation"`
    """
    clean = ConversationContext.clean_user_text(prompt_text)
    if not clean:
        return "conversation"

    if _RECIPE_SCALING_RE.search(clean):
        return "recipe_scaling"
    if _URL_INGEST_RE.search(clean):
        return "url_ingestion"
    if _MEMORY_RECALL_RE.search(clean):
        return "memory_recall"
    if _FACTUAL_EXTRACTION_RE.search(clean):
        return "factual_extraction"

    if config is not None and getattr(config, "use_systemone_routing", False) and not getattr(config, "is_ollama_or_metal", False):
        try:
            from .systemone import SystemOneClient

            decision = SystemOneClient(config=config).route_query(clean)
            if decision.choice in ("web_research", "weather_or_math"):
                return "factual_extraction"
        except Exception:  # noqa: BLE001
            pass

    return "conversation"


# Alias for 100% backwards compatibility with existing callers
ColibriSqliteCheckpointer = SqliteSessionStore


def _prefetch_local_memory_context(prompt_text: str, config: AgentConfig) -> str:
    """Zero-roundtrip speculative local RAG lookup (<10ms).

    Pre-fetches matching personal facts from `data/memory_db/` when facts exist
    so the model can answer personal questions in 1 LLM pass instead of 2.
    Matches are sorted deterministically by `(key, category)` and stripped of
    volatile fields (`distance`, `updated_at`) so the prompt prefix stays
    100% byte-identical across multi-step tool turns.
    """
    if not getattr(config, "prefetch_memory", True) or not prompt_text:
        return ""
    clean_query = ConversationContext.clean_user_text(prompt_text)
    if not clean_query:
        return ""
    try:
        from .tools.rag import _canonical_match_sort_key, _load_facts, query_memory

        facts = _load_facts()
        if not facts:
            return ""
        raw = query_memory(query=clean_query, k=2)
        parsed = json.loads(raw)
        matches = [
            m
            for m in (parsed.get("matches") or [])
            if m.get("source") in ("chromadb", "json_vault") and m.get("value")
        ]
        if not matches:
            return ""
        matches.sort(key=_canonical_match_sort_key)
        valid = [f"- {m.get('key')}: {m.get('value')}" for m in matches]
        return "\n\n[LOCAL MEMORY CONTEXT]\n" + "\n".join(valid)
    except Exception:  # noqa: BLE001
        return ""


def get_langchain_tools(fast_mode: bool = False) -> List[Any]:
    """Return all bound internet, vision, voice, RAG, code, and device tools for LangChain `bind_tools()`."""
    from .tools import get_langchain_tools as _get_lc_tools

    return _get_lc_tools(fast_mode=fast_mode)


def _load_tools_registry(
    fast_mode: bool = False,
) -> Tuple[Dict[str, Callable[..., Any]], List[Dict[str, Any]], Callable[[], List[Any]]]:
    """Load cached tool registry, OpenAI JSON schemas (fast or full), and LangChain tool wrappers."""
    from .tools import (
        DEDUPLICATED_TOOLS_SCHEMA,
        FAST_COMPANION_TOOLS_SCHEMA,
        load_all_tool_suites,
    )

    registry, full_schemas, _ = load_all_tool_suites(use_cache=True)
    if fast_mode and FAST_COMPANION_TOOLS_SCHEMA:
        return registry, FAST_COMPANION_TOOLS_SCHEMA, lambda: get_langchain_tools(fast_mode=True)
    return registry, DEDUPLICATED_TOOLS_SCHEMA or full_schemas, get_langchain_tools


class AgentRunResult(dict):
    """Result object returned by `run_agent` supporting dict keys, attributes, and `str()`."""

    @property
    def answer(self) -> str:
        return str(self.get("answer", ""))

    @property
    def output(self) -> str:
        return str(self.get("answer", ""))

    @property
    def messages(self) -> List[Any]:
        return list(self.get("messages", []))

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        return list(self.get("tool_calls", []))

    @property
    def steps(self) -> int:
        return int(self.get("steps", 0))

    @property
    def runner(self) -> str:
        return str(self.get("runner", "stdlib"))

    @property
    def model(self) -> str:
        return str(self.get("model", ""))

    @property
    def intent(self) -> str:
        return str(self.get("intent", "conversation"))

    @property
    def thread_id(self) -> str:
        return str(self.get("thread_id", "default"))

    @property
    def context(self) -> Any:
        return self.get("context")

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'AgentRunResult' object has no attribute {name!r}")

    def __contains__(self, item: object) -> bool:
        if super().__contains__(item):
            return True
        if isinstance(item, str) and item in str(self.get("answer", "")):
            return True
        return False

    def __str__(self) -> str:
        return str(self.get("answer", ""))


# ==============================================================================
# Network Transport Seam (Transport Port & Adapters)
# ==============================================================================

class ColibriTransport(ColibriGatewayPort):
    """Network transport port for Colibri / OpenAI-compatible chat completions.

    Inherits from ColibriGatewayPort, retaining full backwards compatibility with
    legacy `.complete()` callers and subclasses while participating in the unified gateway seam.
    """

    def complete(
        self,
        config: AgentConfig,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        active_slot: Optional[int] = None,
        stream_tokens: bool = False,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a chat completion request and return the parsed response dictionary."""
        raise NotImplementedError

    def complete_chat(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        slot: int = 0,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        class _LegacyConfig:
            def __init__(self) -> None:
                self.cache_slot = slot
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.model = model or "qwen36-35b"
                self.preserve_thinking = True
                self.compact = "auto"
                self.is_ollama_or_metal = False

            def extra_body(self, slot: Optional[int] = None) -> Dict[str, Any]:
                return dict(extra_body or {})

        return self.complete(
            config=_LegacyConfig(),  # type: ignore[arg-type]
            messages=messages,
            tools=tools,
            active_slot=slot,
            stream_tokens=stream,
            token_callback=on_token,
        )

    def decide_systemone(
        self,
        state: Any = None,
        questions: Optional[Mapping[str, Any]] = None,
        slot: int = 2,
        normalize: Union[bool, str] = True,
        extra_body: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {}

    def probe_models(self) -> List[Dict[str, Any]]:
        return []


class HttpTransport(HttpGatewayAdapter, ColibriTransport):
    """Production transport executing chat completions via `urllib.request` (SSE streaming + JSON)."""

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        super().__init__(config=config)

    def complete(
        self,
        config: AgentConfig,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        active_slot: Optional[int] = None,
        stream_tokens: bool = False,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        if config is not None:
            self.base_url = config.base_url
            self.api_key = config.api_key
            self.model = config.model
            self.timeout = config.request_timeout

        slot = config.cache_slot if active_slot is None else active_slot
        extra = _colibri_extra_body(config, active_slot=active_slot)
        return self.complete_chat(
            messages=messages,
            tools=tools,
            slot=slot,
            stream=stream_tokens,
            on_token=token_callback,
            extra_body=extra,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            model=config.model,
        )


# ==============================================================================
# Unified Graph Runner: ColibriAgent
# ==============================================================================

class ColibriAgent:
    """Unified StateGraph & ReAct runner for the Colibri AI Companion (`src/ai/graph.py`).

    Encapsulates:
    - Speculative zero-roundtrip local RAG memory prefetching (`data/memory_db/`)
    - Two-slot KV cache prefix split (fast companion slot 0 vs full research slot 1)
    - Intent routing (deterministic heuristics + zero-token SystemOne endpoint)
    - Reflection guards against step budget exhaustion and malformed XML `<tool_call>` syntax
    - Parallel tool execution delegating directly to `TOOL_REGISTRY_OBJ`
    - Pluggable network transport (`ColibriTransport`)
    - Native SQLite checkpointing and history continuity across thread sessions
    """

    runner_name = "stdlib"

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        system_prompt: str = SYSTEM_PROMPT,
        transport: Optional[ColibriGatewayPort] = None,
        on_tool_call: Optional[Callable[[Dict[str, Any], str, float], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        stream: bool = False,
        checkpointer: Optional[Union[SqliteSessionStore, SessionStore]] = None,
        tool_suite: Optional[ToolSuite] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt
        self.transport: ColibriGatewayPort = transport or HttpGatewayAdapter(config=self.config)
        self.on_tool_call = on_tool_call
        self.on_token = on_token
        self.stream = stream
        self.checkpointer = checkpointer or SqliteSessionStore()
        self.tool_suite: ToolSuite = tool_suite or TOOL_SUITE_OBJ

    @property
    def tool_registry(self) -> ToolSuite:
        return self.tool_suite

    def prefetch_memory(self, prompt_text: str) -> str:
        """Prefetch personal facts from data/memory_db/ for zero-roundtrip local RAG context."""
        return _prefetch_local_memory_context(prompt_text, self.config)

    def classify_intent(self, prompt_text: str) -> str:
        """Classify user intent into one of 5 canonical categories."""
        return classify_user_intent(prompt_text, self.config)

    def execute_tools(
        self,
        tool_calls: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute one or more tool calls in parallel delegating directly to `ToolSuite`."""
        if not tool_calls:
            return []
        return self.tool_suite.execute(
            tool_calls,
            parallel=True,
            max_chars=self.config.max_tool_chars,
            on_tool_call=self.on_tool_call,
        )

    def _chat_completions(
        self,
        messages: List[Dict[str, Any]],
        include_tools: bool = True,
        fast_mode: bool = False,
        active_slot: Optional[int] = None,
        stream_tokens: bool = False,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Dispatch chat completions request via the configured gateway/transport port."""
        tools = None
        if include_tools:
            if fast_mode and self.tool_suite.fast_companion_schemas:
                tools = self.tool_suite.fast_companion_schemas
            else:
                tools = self.tool_suite.deduplicated_schemas or self.tool_suite.schemas

        slot = self.config.cache_slot if active_slot is None else active_slot
        extra = _colibri_extra_body(self.config, active_slot=active_slot)

        if hasattr(self.transport, "complete_chat"):
            return self.transport.complete_chat(
                messages=messages,
                tools=tools,
                slot=slot,
                stream=stream_tokens,
                on_token=token_callback,
                extra_body=extra,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                model=self.config.model,
            )

        return self.transport.complete(
            config=self.config,
            messages=messages,
            tools=tools,
            active_slot=active_slot,
            stream_tokens=stream_tokens,
            token_callback=token_callback,
        )

    def run_turn(
        self,
        prompt: str,
        history: Optional[Sequence[Dict[str, Any]]] = None,
        thread_id: str = "default",
        memory_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        intent: Optional[str] = None,
        slot: Optional[int] = None,
    ) -> AgentRunResult:
        """Execute a conversational turn using ConversationContext."""
        tid = (thread_id or "default").strip() or "default"
        clean_prompt = ConversationContext.clean_user_text(prompt)
        resolved_intent = intent or self.classify_intent(clean_prompt)
        needs_research = self.tool_suite.needs_full_research_suite(clean_prompt)
        mem_ctx = memory_context
        if mem_ctx is None and "[LOCAL MEMORY CONTEXT]" not in prompt:
            mem_ctx = self.prefetch_memory(clean_prompt) or None

        resolved_slot = slot if slot is not None else ConversationContext.resolve_slot(
            resolved_intent, requires_research=needs_research
        )
        ctx = ConversationContext.from_turn(
            prompt=clean_prompt,
            history=history,
            system_prompt=system_prompt or self.system_prompt,
            memory_context=mem_ctx,
            intent=resolved_intent,
            requires_research=needs_research,
            slot=resolved_slot,
        )
        return self.invoke(ctx, thread_id=tid)

    def run(
        self,
        prompt: str,
        thread_id: str = "default",
        history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> AgentRunResult:
        """Execute a prompt turn within the specified session thread."""
        return self.run_turn(
            prompt=prompt,
            history=history,
            thread_id=thread_id,
        )

    def invoke(
        self,
        state_or_prompt: Union[str, Dict[str, Any], ConversationContext],
        thread_id: Optional[str] = None,
    ) -> AgentRunResult:
        """Execute the `intent_router -> agent -> tools -> reflect_and_guard -> end` state machine."""
        resolved_thread_id = (thread_id or "").strip()
        ctx: Optional[ConversationContext] = None

        if isinstance(state_or_prompt, ConversationContext):
            ctx = state_or_prompt
            input_messages: List[Dict[str, Any]] = list(ctx.messages)
            max_steps = self.config.max_iterations
        elif isinstance(state_or_prompt, str):
            input_messages = [
                {"role": "user", "content": state_or_prompt}
            ]
            max_steps = self.config.max_iterations
        else:
            if not resolved_thread_id:
                if state_or_prompt.get("thread_id"):
                    resolved_thread_id = str(state_or_prompt.get("thread_id")).strip()
                elif isinstance(state_or_prompt.get("configurable"), dict) and state_or_prompt["configurable"].get("thread_id"):
                    resolved_thread_id = str(state_or_prompt["configurable"]["thread_id"]).strip()
            raw_msgs = state_or_prompt.get("messages", [])
            input_messages = []
            for m in raw_msgs:
                if isinstance(m, dict):
                    input_messages.append(dict(m))
                elif hasattr(m, "content"):
                    role = getattr(m, "type", "user")
                    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
                    entry: Dict[str, Any] = {"role": role_map.get(role, role), "content": str(m.content)}
                    if hasattr(m, "tool_call_id"):
                        entry["tool_call_id"] = str(getattr(m, "tool_call_id", ""))
                    if hasattr(m, "name"):
                        entry["name"] = str(getattr(m, "name", ""))
                    input_messages.append(entry)
            max_steps = int(state_or_prompt.get("max_steps") or self.config.max_iterations)

        checkpointer = self.checkpointer
        if resolved_thread_id and len(input_messages) <= 1:
            tup = checkpointer.get_tuple({"configurable": {"thread_id": resolved_thread_id}})
            if tup and tup.checkpoint.get("channel_values", {}).get("messages"):
                input_messages = [*tup.checkpoint["channel_values"]["messages"], *input_messages]
            else:
                prior_cp = checkpointer.load_checkpoint(resolved_thread_id)
                if prior_cp and prior_cp.get("messages"):
                    input_messages = [*prior_cp["messages"], *input_messages]

        # Find latest user query for intent classification, speculative RAG memory prefetch & two-slot tool routing
        last_user_idx = -1
        last_user_text = ""
        for idx in range(len(input_messages) - 1, -1, -1):
            if input_messages[idx].get("role") == "user":
                last_user_idx = idx
                last_user_text = str(input_messages[idx].get("content") or "")
                break

        clean_user_text = ConversationContext.clean_user_text(last_user_text)
        intent = self.classify_intent(clean_user_text)

        # Optimization: Zero-roundtrip speculative local RAG lookup (<10ms)
        mem_ctx = ""
        if last_user_idx >= 0 and "[LOCAL MEMORY CONTEXT]" not in last_user_text:
            mem_ctx = self.prefetch_memory(clean_user_text)

        # Optimization: Two-Slot Prefix Split (Slot 0 = Fast Companion tools, Slot 1 = Full tools)
        needs_research = self.tool_suite.needs_full_research_suite(clean_user_text)
        use_fast_tools = bool(
            getattr(self.config, "fast_tool_pruning", False)
            and not needs_research
        )
        research_slot = getattr(self.config, "research_cache_slot", 1)
        active_slot = research_slot if needs_research else self.config.cache_slot

        # Build or synchronize ConversationContext
        if ctx is None:
            ctx = ConversationContext.from_turn(
                prompt=clean_user_text,
                history=input_messages[:last_user_idx] if last_user_idx >= 0 else input_messages,
                system_prompt=self.system_prompt,
                memory_context=mem_ctx or None,
                intent=intent,
                requires_research=needs_research,
                slot=active_slot,
            )
            if last_user_idx >= 0 and last_user_idx + 1 < len(input_messages):
                ctx.messages.extend(input_messages[last_user_idx + 1:])
        else:
            if not ctx.memory_context and mem_ctx:
                ctx.memory_context = mem_ctx
            ctx.intent = intent
            ctx.slot = active_slot
            if len(input_messages) > len(ctx.messages):
                ctx.messages = input_messages

        messages: List[Dict[str, Any]] = ctx.to_llm_payload(slot=active_slot)

        # Optimization: Optional SystemOne zero-token routing (`use_systemone_routing`)
        skip_tools_via_sys1 = bool(
            getattr(self.config, "use_systemone_routing", False)
            and last_user_text
            and not getattr(self.config, "is_ollama_or_metal", False)
            and intent == "conversation"
        )

        step_count = 0
        tool_traces: List[Dict[str, Any]] = []
        compacted_messages = 0
        last_reasoning = ""
        force_finalize = skip_tools_via_sys1

        tts_streamer = None
        if getattr(self.config, "stream_tts", False):
            try:
                from .tools.audio import SentenceTTSStreamer

                tts_streamer = SentenceTTSStreamer(voice="auto", play_audio=True)
            except Exception:  # noqa: BLE001
                tts_streamer = None

        def _handle_token(tok: str) -> None:
            if self.on_token is not None:
                self.on_token(tok)
            if tts_streamer is not None:
                tts_streamer.feed(tok)

        use_streaming = bool(self.stream or self.on_token is not None or tts_streamer is not None)

        while True:
            # --- Node: `agent` ---
            response = self._chat_completions(
                messages,
                include_tools=not force_finalize,
                fast_mode=use_fast_tools,
                active_slot=active_slot,
                stream_tokens=use_streaming,
                token_callback=_handle_token if use_streaming else None,
            )
            if isinstance(response.get("compaction"), dict):
                compacted_messages += int(
                    response["compaction"].get("summarized_messages", 0)
                )

            choices = response.get("choices") or []
            message = (choices[0].get("message") if choices else None) or {}
            content = str(message.get("content") or "")
            reasoning = str(message.get("reasoning_content") or "")
            if reasoning:
                last_reasoning = reasoning

            raw_tool_calls = message.get("tool_calls") or []
            normalized_calls: List[Dict[str, Any]] = []
            for idx, tc in enumerate(raw_tool_calls):
                fn = tc.get("function") or {}
                normalized_calls.append(
                    {
                        "id": str(tc.get("id") or f"call_{step_count}_{idx}"),
                        "name": str(fn.get("name") or tc.get("name") or ""),
                        "args": fn.get("arguments")
                        if "arguments" in fn
                        else tc.get("args", "{}"),
                    }
                )

            # Check for inline <tool_call> tags if gateway returned raw text
            if not normalized_calls and not force_finalize:
                salvaged = self.tool_suite.salvage_inline_calls(content)
                if salvaged:
                    normalized_calls = salvaged

            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": content,
            }
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            if raw_tool_calls:
                assistant_msg["tool_calls"] = raw_tool_calls
            elif normalized_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["args"]
                            if isinstance(tc["args"], str)
                            else json.dumps(tc["args"], ensure_ascii=False),
                        },
                    }
                    for tc in normalized_calls
                ]
            messages.append(assistant_msg)

            # If no tool calls, check `reflect_and_guard` for malformed `<tool_call>` syntax
            if not normalized_calls:
                if (
                    "<tool_call>" in content
                    and not force_finalize
                    and step_count < max_steps
                ):
                    # --- Node: `reflect_and_guard` (malformed tool call recovery) ---
                    step_count += 1
                    force_finalize = True
                    messages.append({"role": "user", "content": _SYNTHESIS_NUDGE})
                    continue
                break

            # --- Node: `tools` (parallel tool execution delegating to TOOL_REGISTRY_OBJ) ---
            step_count += 1
            exec_results = self.execute_tools(normalized_calls)
            for res in exec_results:
                tool_traces.append(res)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": res["tool_call_id"],
                        "name": res["name"],
                        "content": res["content"],
                    }
                )

            # --- Node: `reflect_and_guard` (step budget check) ---
            if step_count >= max_steps and not force_finalize:
                force_finalize = True
                messages.append({"role": "user", "content": _SYNTHESIS_NUDGE})

        final_answer = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                final_answer = str(msg["content"])
                break

        if tts_streamer is not None:
            tts_streamer.flush()

        effective_tid = resolved_thread_id or "default"
        if ctx is not None:
            ctx.messages = messages
            storage_messages = ctx.to_storage_messages()
        else:
            storage_messages = messages

        try:
            checkpointer.save_checkpoint(
                thread_id=effective_tid,
                messages=storage_messages,
                intent=intent,
                step_count=step_count,
            )
            checkpointer.put(
                {"configurable": {"thread_id": effective_tid}},
                {
                    "v": 1,
                    "id": f"chk_{effective_tid}",
                    "channel_values": {
                        "messages": storage_messages,
                        "intent": intent,
                        "step_count": step_count,
                    },
                },
                {"source": "agent", "step": step_count},
            )
        except Exception:  # noqa: BLE001
            pass

        return AgentRunResult(
            answer=final_answer,
            output=final_answer,
            messages=messages,
            tool_calls=tool_traces,
            steps=step_count,
            runner=self.runner_name,
            model=self.config.model,
            compacted_messages=compacted_messages,
            reasoning=last_reasoning,
            intent=intent,
            slot=active_slot,
            thread_id=effective_tid,
            context=ctx,
        )


# Backward-compatible alias
StdlibColibriAgent = ColibriAgent


# Re-export LangGraph adapter interface for 100% backwards compatibility
from .langgraph_adapter import (
    InternetAgentState,
    build_internet_agent_graph,
    has_langgraph,
)


def run_agent(
    prompt: str,
    config: Optional[AgentConfig] = None,
    stream: bool = False,
    history: Optional[List[Dict[str, Any]]] = None,
    on_tool_call: Optional[Callable[[Dict[str, Any], str, float], None]] = None,
    on_token: Optional[Callable[[str], None]] = None,
    force_stdlib: bool = False,
    thread_id: Optional[str] = None,
    checkpointer: Optional[Union[SqliteSessionStore, SessionStore]] = None,
    transport: Optional[ColibriGatewayPort] = None,
) -> AgentRunResult:
    """Run the Colibri agent on `prompt` via the unified `ColibriAgent` runner."""
    cfg = config or AgentConfig()
    printed_via_stream = False

    def _stream_token_cb(tok: str) -> None:
        nonlocal printed_via_stream
        if on_token is not None:
            on_token(tok)
        if stream:
            printed_via_stream = True
            print(tok, end="", flush=True)

    token_cb = _stream_token_cb if (stream or on_token is not None) else None

    agent = ColibriAgent(
        config=cfg,
        on_tool_call=on_tool_call,
        on_token=token_cb,
        stream=stream,
        checkpointer=checkpointer,
        transport=transport,
    )

    resolved_tid = (thread_id or "").strip()
    result = agent.run(prompt=prompt, thread_id=resolved_tid or "default", history=history)

    if stream:
        if printed_via_stream:
            print(flush=True)
        elif result.answer:
            print(result.answer, flush=True)

    return result


__all__ = [
    "SYSTEM_PROMPT",
    "AgentRunResult",
    "ColibriAgent",
    "ColibriGatewayPort",
    "ColibriSqliteCheckpointer",
    "ColibriTransport",
    "HttpGatewayAdapter",
    "HttpTransport",
    "InMemoryGatewayAdapter",
    "InMemorySessionStore",
    "InternetAgentState",
    "SessionStore",
    "SqliteSessionStore",
    "StdlibColibriAgent",
    "build_internet_agent_graph",
    "classify_user_intent",
    "get_langchain_tools",
    "get_session_history",
    "has_langgraph",
    "run_agent",
]
