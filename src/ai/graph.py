"""LangGraph & zero-dependency StateGraph orchestration engine for Colibri (`src/ai/graph.py`).

Targets Qwen3.6-35B-A3B (`qwen36-35b`) served via `coli serve` with:
- Stateful KV + DeltaNet prefix caching (`cache_slot`, static system prompt & tool definitions)
- Thinking-block preservation across tool turns (`preserve_thinking: True`)
- Automatic server-side context compaction (`context_compaction: "auto"`)
- Parallel SSRF-safe internet tool execution (`ThreadPoolExecutor`) with output truncation
- Step-budget & malformed-tool reflection guard (`reflect_and_guard` node)
- Automatic fallback to `StdlibColibriAgent` (`urllib.request`) when `langgraph` is not installed
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, TypedDict, Union

from .config import AgentConfig

try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False

try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )

    _LANGGRAPH_CP_AVAILABLE = True
except ImportError:
    class BaseCheckpointSaver:  # type: ignore
        pass

    class CheckpointTuple(NamedTuple):  # type: ignore
        config: Dict[str, Any]
        checkpoint: Dict[str, Any]
        metadata: Dict[str, Any]
        parent_config: Optional[Dict[str, Any]] = None

    _LANGGRAPH_CP_AVAILABLE = False


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


def has_langgraph() -> bool:
    """Return True if `langgraph` and `langchain_openai` are installed."""
    return _LANGGRAPH_AVAILABLE


def _colibri_extra_body(config: AgentConfig, active_slot: Optional[int] = None) -> Dict[str, Any]:
    """Build Colibri `/v1/chat/completions` extension fields for Qwen3.6 prefix reuse."""
    if getattr(config, "is_ollama_or_metal", False):
        return {}
    body = config.extra_body()
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
    clean = (prompt_text or "").split("\n\n[LOCAL MEMORY CONTEXT]", 1)[0].strip()
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


class ColibriSqliteCheckpointer(BaseCheckpointSaver):
    """Persistent zero-dependency SQLite checkpointer (`data/memory_db/checkpoints.sqlite`).

    Saves and restores multi-turn session history (`thread_id`, `messages`, `intent`,
    `step_count`, `updated_at`) across process restarts natively for both `StdlibColibriAgent`
    and LangGraph `StateGraph`.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._explicit_db_path = db_path

    @property
    def db_path(self) -> str:
        if self._explicit_db_path:
            parent = os.path.dirname(os.path.abspath(self._explicit_db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            return os.path.abspath(self._explicit_db_path)
        from .tools.rag import _memory_dir

        return os.path.join(_memory_dir(), "checkpoints.sqlite")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_checkpoints (
                thread_id TEXT PRIMARY KEY,
                messages_json TEXT NOT NULL,
                intent TEXT NOT NULL DEFAULT 'conversation',
                step_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        return conn

    @staticmethod
    def _clean_messages_for_storage(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip static system prompt and transient `[LOCAL MEMORY CONTEXT]` suffixes before persisting."""
        cleaned: List[Dict[str, Any]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "user")
            if role == "system":
                continue
            item = dict(m)
            if role == "user" and isinstance(item.get("content"), str):
                item["content"] = item["content"].split("\n\n[LOCAL MEMORY CONTEXT]", 1)[0]
            cleaned.append(item)
        return cleaned

    def save_checkpoint(
        self,
        thread_id: str,
        messages: Sequence[Dict[str, Any]],
        intent: str = "conversation",
        step_count: int = 0,
    ) -> None:
        tid = (thread_id or "default").strip() or "default"
        clean_msgs = self._clean_messages_for_storage(messages)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO session_checkpoints (thread_id, messages_json, intent, step_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    messages_json = excluded.messages_json,
                    intent = excluded.intent,
                    step_count = excluded.step_count,
                    updated_at = excluded.updated_at
                """,
                (tid, json.dumps(clean_msgs, ensure_ascii=False), intent, int(step_count), now_iso),
            )
            conn.commit()

    def load_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        tid = (thread_id or "default").strip() or "default"
        if not os.path.isfile(self.db_path):
            return None
        try:
            with contextlib.closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT thread_id, messages_json, intent, step_count, updated_at FROM session_checkpoints WHERE thread_id = ?",
                    (tid,),
                ).fetchone()
            if not row:
                return None
            return {
                "thread_id": row[0],
                "messages": json.loads(row[1]) if row[1] else [],
                "intent": row[2],
                "step_count": int(row[3] or 0),
                "updated_at": row[4],
            }
        except Exception:  # noqa: BLE001
            return None

    def clear_checkpoint(self, thread_id: str) -> None:
        tid = (thread_id or "default").strip() or "default"
        if not os.path.isfile(self.db_path):
            return
        with contextlib.closing(self._connect()) as conn:
            conn.execute("DELETE FROM session_checkpoints WHERE thread_id = ?", (tid,))
            conn.commit()

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """LangGraph native checkpoint persistence protocol."""
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        tid = str(configurable.get("thread_id") or "default")
        checkpoint_id = str(checkpoint.get("id") or f"chk_{int(time.time()*1000)}")
        channel_values = checkpoint.get("channel_values", {})
        if not isinstance(channel_values, dict):
            channel_values = {}
        messages = channel_values.get("messages", [])
        intent = str(channel_values.get("intent") or "conversation")
        step_count = int(channel_values.get("step_count") or 0)
        self.save_checkpoint(thread_id=tid, messages=messages, intent=intent, step_count=step_count)
        return {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """LangGraph task write staging."""
        pass

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """LangGraph native checkpoint restoration protocol."""
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        tid = str(configurable.get("thread_id") or "default")
        data = self.load_checkpoint(tid)
        if not data:
            return None
        checkpoint_id = f"chk_{tid}"
        ckpt = {
            "v": 1,
            "id": checkpoint_id,
            "ts": data.get("updated_at") or datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "channel_values": {
                "messages": data.get("messages") or [],
                "intent": data.get("intent") or "conversation",
                "step_count": int(data.get("step_count") or 0),
            },
            "channel_versions": {},
            "versions_seen": {},
        }
        meta = {
            "source": "input",
            "step": int(data.get("step_count") or 0),
            "writes": {},
            "parents": {},
        }
        ret_cfg = {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint_id,
            }
        }
        return CheckpointTuple(config=ret_cfg, checkpoint=ckpt, metadata=meta, parent_config=None)

    def list(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ):
        """List checkpoint tuples matching thread_id."""
        tid = None
        if isinstance(config, dict):
            tid = config.get("configurable", {}).get("thread_id")
        with contextlib.closing(self._connect()) as conn:
            if tid:
                rows = conn.execute(
                    "SELECT thread_id, messages_json, intent, step_count, updated_at FROM session_checkpoints WHERE thread_id = ? ORDER BY updated_at DESC",
                    (tid,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT thread_id, messages_json, intent, step_count, updated_at FROM session_checkpoints ORDER BY updated_at DESC",
                ).fetchall()
        count = 0
        for row in rows:
            if limit and count >= limit:
                break
            rtid = row[0]
            r_msgs = json.loads(row[1]) if row[1] else []
            r_intent = row[2] or "conversation"
            r_step = int(row[3] or 0)
            r_upd = row[4]
            chk_id = f"chk_{rtid}"
            ckpt = {
                "v": 1,
                "id": chk_id,
                "ts": r_upd,
                "channel_values": {"messages": r_msgs, "intent": r_intent, "step_count": r_step},
                "channel_versions": {},
                "versions_seen": {},
            }
            meta = {"source": "input", "step": r_step, "writes": {}, "parents": {}}
            ret_cfg = {"configurable": {"thread_id": rtid, "checkpoint_ns": "", "checkpoint_id": chk_id}}
            yield CheckpointTuple(config=ret_cfg, checkpoint=ckpt, metadata=meta, parent_config=None)
            count += 1


def get_session_history(thread_id: str = "default", db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return persisted conversational messages for `thread_id` from `checkpoints.sqlite`."""
    cp = ColibriSqliteCheckpointer(db_path=db_path).load_checkpoint(thread_id)
    if not cp:
        return []
    return list(cp.get("messages") or [])


def _needs_full_research_suite(prompt_text: str) -> bool:
    """Return True if the user query explicitly mentions research/vision/code domains outside the fast companion suite."""
    intent = classify_user_intent(prompt_text)
    if intent in ("url_ingestion", "recipe_scaling"):
        return True
    return bool(_RESEARCH_KEYWORDS_RE.search(prompt_text or ""))


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
    clean_query = prompt_text.split("\n\n[LOCAL MEMORY CONTEXT]", 1)[0].strip()
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


def _truncate_output(text: Any, max_chars: int) -> str:
    """Ensure tool output is a string capped at `max_chars` characters."""
    raw = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    if max_chars > 0 and len(raw) > max_chars:
        return raw[:max_chars] + f"\n...[truncated to {max_chars} chars]"
    return raw


def _execute_single_tool(
    name: str,
    args: Union[str, Dict[str, Any]],
    max_chars: int,
) -> str:
    """Execute one named tool safely and truncate its output."""
    registry, _, _ = _load_tools_registry()
    if isinstance(args, str):
        try:
            parsed_args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as exc:
            return f"Error: malformed JSON arguments for tool '{name}': {exc}"
    elif isinstance(args, dict):
        parsed_args = dict(args)
    else:
        parsed_args = {}

    if not isinstance(parsed_args, dict):
        return f"Error: tool '{name}' arguments must decode to a JSON object."

    # Pass max_chars down if the tool accepts it and caller didn't specify one
    fn = registry.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(sorted(registry))}"

    try:
        result = fn(**parsed_args)
    except TypeError:
        try:
            # Fallback if single positional arg was passed under 'input' or 'query'
            result = fn(**parsed_args)
        except Exception as exc:  # noqa: BLE001
            return f"Error executing tool '{name}': {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Error executing tool '{name}': {exc}"

    return _truncate_output(result, max_chars)


def _execute_tool_calls_parallel(
    tool_calls: Sequence[Dict[str, Any]],
    max_chars: int,
    on_tool_call: Optional[Callable[[Dict[str, Any], str, float], None]] = None,
) -> List[Dict[str, Any]]:
    """Execute one or more tool calls in parallel via `ThreadPoolExecutor`.

    Each item in `tool_calls` must provide `id`, `name`, and `args` (dict or JSON str).
    Returns a list of tool result dicts in the same order as `tool_calls`.
    """
    if not tool_calls:
        return []

    def _worker(tc: Dict[str, Any]) -> Tuple[Dict[str, Any], str, float]:
        t0 = time.monotonic()
        name = str(tc.get("name") or tc.get("function", {}).get("name") or "")
        args = tc.get("args")
        if args is None:
            args = tc.get("function", {}).get("arguments", "{}")
        output = _execute_single_tool(name, args, max_chars=max_chars)
        elapsed = time.monotonic() - t0
        return tc, output, elapsed

    max_workers = min(len(tool_calls), 6)
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_worker, tc) for tc in tool_calls]
        for fut in futures:
            tc, output, elapsed = fut.result()
            tc_id = str(tc.get("id") or f"call_{len(results)}")
            name = str(tc.get("name") or tc.get("function", {}).get("name") or "")
            args = tc.get("args")
            if args is None:
                args = tc.get("function", {}).get("arguments", "{}")
            if on_tool_call is not None:
                try:
                    on_tool_call({"id": tc_id, "name": name, "args": args}, output, elapsed)
                except Exception:  # noqa: BLE001
                    pass
            results.append(
                {
                    "tool_call_id": tc_id,
                    "name": name,
                    "args": args,
                    "content": output,
                    "elapsed_s": round(elapsed, 3),
                }
            )
    return results


def _salvage_inline_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Salvage `<tool_call>{...}</tool_call>` blocks from raw text if not parsed by gateway."""
    if not content or "<tool_call>" not in content:
        return []
    salvaged: List[Dict[str, Any]] = []
    for idx, match in enumerate(_MALFORMED_TOOL_RE.finditer(content)):
        raw_json = match.group(1).strip()
        try:
            obj = json.loads(raw_json)
            if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                args = obj.get("arguments", obj.get("args", {}))
                salvaged.append(
                    {
                        "id": f"salvaged_call_{idx}",
                        "name": obj["name"],
                        "args": args if isinstance(args, dict) else {},
                    }
                )
        except Exception:  # noqa: BLE001
            continue
    return salvaged


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


class StdlibColibriAgent:
    """Zero-dependency StateGraph runner (`agent -> tools -> reflect_and_guard -> end`).

    Communicates directly with Colibri's `/v1/chat/completions` via `urllib.request`
    using the exact same node transitions and Qwen3.6 KV-cache extensions as the
    LangGraph workflow.
    """

    runner_name = "stdlib"

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        system_prompt: str = SYSTEM_PROMPT,
        on_tool_call: Optional[Callable[[Dict[str, Any], str, float], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        stream: bool = False,
        checkpointer: Optional[ColibriSqliteCheckpointer] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt
        self.on_tool_call = on_tool_call
        self.on_token = on_token
        self.stream = stream
        self.checkpointer = checkpointer or ColibriSqliteCheckpointer()

    def _chat_completions(
        self,
        messages: List[Dict[str, Any]],
        include_tools: bool = True,
        fast_mode: bool = False,
        active_slot: Optional[int] = None,
        stream_tokens: bool = False,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        _, schemas, _ = _load_tools_registry(fast_mode=fast_mode)
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": bool(stream_tokens),
        }
        payload.update(_colibri_extra_body(self.config, active_slot=active_slot))
        if include_tools and schemas:
            payload["tools"] = schemas
            payload["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream_tokens else "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.config.request_timeout) as resp:
            if not stream_tokens:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)

            content_parts: List[str] = []
            reasoning_parts: List[str] = []
            tool_calls_acc: Dict[int, Dict[str, Any]] = {}
            compaction_info: Optional[Dict[str, Any]] = None

            raw_bytes = resp.read()
            raw_text = raw_bytes.decode("utf-8", errors="replace")
            # Support both standard JSON (e.g. non-SSE mock servers) and SSE `data: ...` streams
            if raw_text.lstrip().startswith("{"):
                parsed_json = json.loads(raw_text)
                choices = parsed_json.get("choices") or []
                msg = (choices[0].get("message") if choices else None) or {}
                c_str = str(msg.get("content") or "")
                if c_str and token_callback:
                    token_callback(c_str)
                return parsed_json

            for line in raw_text.splitlines():
                line_s = line.strip()
                if not line_s.startswith("data:"):
                    continue
                data_str = line_s[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data_str)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(chunk.get("compaction"), dict):
                    compaction_info = chunk["compaction"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or choices[0].get("message") or {}
                c_delta = delta.get("content")
                if c_delta:
                    content_parts.append(str(c_delta))
                    if token_callback is not None:
                        token_callback(str(c_delta))
                r_delta = delta.get("reasoning_content")
                if r_delta:
                    reasoning_parts.append(str(r_delta))
                for tc_delta in delta.get("tool_calls") or []:
                    idx = int(tc_delta.get("index", 0))
                    entry = tool_calls_acc.setdefault(
                        idx,
                        {
                            "id": tc_delta.get("id") or f"call_stream_{idx}",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tc_delta.get("id"):
                        entry["id"] = tc_delta["id"]
                    fn_d = tc_delta.get("function") or {}
                    if fn_d.get("name"):
                        entry["function"]["name"] += str(fn_d["name"])
                    if fn_d.get("arguments"):
                        entry["function"]["arguments"] += str(fn_d["arguments"])

            assembled_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": "".join(content_parts),
            }
            if reasoning_parts:
                assembled_msg["reasoning_content"] = "".join(reasoning_parts)
            if tool_calls_acc:
                assembled_msg["tool_calls"] = [
                    tool_calls_acc[k] for k in sorted(tool_calls_acc)
                ]
            out_resp: Dict[str, Any] = {
                "choices": [{"index": 0, "message": assembled_msg}],
            }
            if compaction_info is not None:
                out_resp["compaction"] = compaction_info
            return out_resp

    def invoke(
        self,
        state_or_prompt: Union[str, Dict[str, Any]],
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the `intent_router -> agent -> tools -> reflect_and_guard -> end` state machine."""
        resolved_thread_id = (thread_id or "").strip()
        if isinstance(state_or_prompt, str):
            input_messages: List[Dict[str, Any]] = [
                {"role": "user", "content": state_or_prompt}
            ]
            max_steps = self.config.max_iterations
        else:
            if not resolved_thread_id and state_or_prompt.get("thread_id"):
                resolved_thread_id = str(state_or_prompt.get("thread_id")).strip()
            raw_msgs = state_or_prompt.get("messages", [])
            input_messages = []
            for m in raw_msgs:
                if isinstance(m, dict):
                    input_messages.append(dict(m))
                elif hasattr(m, "content"):
                    role = getattr(m, "type", "user")
                    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
                    input_messages.append(
                        {"role": role_map.get(role, role), "content": str(m.content)}
                    )
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

        intent = classify_user_intent(last_user_text, self.config)

        # Optimization: Zero-roundtrip speculative local RAG lookup (<10ms)
        if last_user_idx >= 0 and "[LOCAL MEMORY CONTEXT]" not in last_user_text:
            mem_ctx = _prefetch_local_memory_context(last_user_text, self.config)
            if mem_ctx:
                input_messages[last_user_idx] = {
                    **input_messages[last_user_idx],
                    "content": last_user_text + mem_ctx,
                }

        # Optimization: Two-Slot Prefix Split (Slot 0 = Fast 9 Companion tools, Slot 1 = Full tools)
        use_fast_tools = bool(
            getattr(self.config, "fast_tool_pruning", False)
            and not _needs_full_research_suite(last_user_text)
        )
        active_slot = (
            self.config.cache_slot
            if not (getattr(self.config, "fast_tool_pruning", False) and not use_fast_tools)
            else getattr(self.config, "research_cache_slot", self.config.cache_slot)
        )

        # Optimization: Optional SystemOne zero-token routing (`use_systemone_routing`)
        skip_tools_via_sys1 = False
        if getattr(self.config, "use_systemone_routing", False) and last_user_text and not getattr(self.config, "is_ollama_or_metal", False):
            try:
                from .systemone import SystemOneClient

                decision = SystemOneClient(config=self.config).route_query(last_user_text)
                if decision.choice == "direct" and decision.confidence >= 0.65:
                    skip_tools_via_sys1 = True
                elif decision.choice in ("companion_fast", "weather_or_math"):
                    use_fast_tools = True
                    active_slot = self.config.cache_slot
            except Exception:  # noqa: BLE001
                pass

        # Ensure static system prompt is always byte-identical at index 0 for KV cache prefix reuse
        # (never inject dynamic RAG chunks into messages[0]; earlier turns are never mutated)
        if not input_messages or input_messages[0].get("role") != "system":
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": self.system_prompt},
                *input_messages,
            ]
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                *input_messages[1:],
            ]

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
            # --- Node 1: `agent` ---
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
                salvaged = _salvage_inline_tool_calls(content)
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
                    # --- Node 3: `reflect_and_guard` (malformed tool call recovery) ---
                    step_count += 1
                    force_finalize = True
                    messages.append({"role": "user", "content": _SYNTHESIS_NUDGE})
                    continue
                break

            # --- Node 2: `tools` (parallel tool execution) ---
            step_count += 1
            exec_results = _execute_tool_calls_parallel(
                normalized_calls,
                max_chars=self.config.max_tool_chars,
                on_tool_call=self.on_tool_call,
            )
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

            # --- Node 3: `reflect_and_guard` (step budget check) ---
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
        try:
            checkpointer.put(
                {"configurable": {"thread_id": effective_tid}},
                {
                    "v": 1,
                    "id": f"chk_{effective_tid}",
                    "channel_values": {
                        "messages": messages,
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
            thread_id=effective_tid,
        )


if _LANGGRAPH_AVAILABLE:
    from typing import Annotated
    from langgraph.graph.message import add_messages

    class InternetAgentState(TypedDict, total=False):
        """State schema for the LangGraph Qwen3.6 internet research agent."""

        messages: Annotated[List[BaseMessage], add_messages]
        step_count: int
        max_steps: int
        tool_calls_log: List[Dict[str, Any]]
        force_finalize: bool
        intent: str
        thread_id: str


def build_internet_agent_graph(
    config: Optional[AgentConfig] = None,
    on_tool_call: Optional[Callable[[Dict[str, Any], str, float], None]] = None,
    on_token: Optional[Callable[[str], None]] = None,
    stream: bool = False,
    checkpointer: Optional[Any] = None,
) -> Any:
    """Build the Colibri Qwen3.6-35B-A3B internet research StateGraph.

    Uses `langgraph.graph.StateGraph` and `langchain_openai.ChatOpenAI` when
    installed, or returns `StdlibColibriAgent` with a compatible `.invoke()`
    interface when running in a zero-dependency environment.
    """
    cfg = config or AgentConfig()
    active_cp = checkpointer or ColibriSqliteCheckpointer()
    if not _LANGGRAPH_AVAILABLE:
        return StdlibColibriAgent(
            config=cfg,
            on_tool_call=on_tool_call,
            on_token=on_token,
            stream=stream,
            checkpointer=active_cp,
        )

    use_fast = bool(getattr(cfg, "fast_tool_pruning", False))
    _, _, get_lc_tools = _load_tools_registry(fast_mode=use_fast)
    lc_tools = get_lc_tools()
    extra_body = _colibri_extra_body(cfg)

    llm = ChatOpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key or "colibri",
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.request_timeout,
        streaming=bool(stream or on_token is not None or getattr(cfg, "stream_tts", False)),
        extra_body=extra_body,
    )
    llm_with_tools = llm.bind_tools(lc_tools) if lc_tools else llm

    def agent_node(state: InternetAgentState) -> Dict[str, Any]:
        msgs = list(state.get("messages") or [])
        # Keep static SystemMessage at index 0 so Colibri cache_slot prefix stays valid
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT), *msgs]

        force_finalize = bool(state.get("force_finalize", False))
        active_llm = llm if force_finalize else llm_with_tools
        if stream or on_token is not None or getattr(cfg, "stream_tts", False):
            tts_streamer = None
            if getattr(cfg, "stream_tts", False):
                try:
                    from .tools.audio import SentenceTTSStreamer

                    tts_streamer = SentenceTTSStreamer(voice="auto", play_audio=True)
                except Exception:  # noqa: BLE001
                    tts_streamer = None
            acc_chunk = None
            for chunk in active_llm.stream(msgs):
                c_text = str(getattr(chunk, "content", "") or "")
                if c_text:
                    if on_token is not None:
                        on_token(c_text)
                    if tts_streamer is not None:
                        tts_streamer.feed(c_text)
                acc_chunk = chunk if acc_chunk is None else acc_chunk + chunk
            if tts_streamer is not None:
                tts_streamer.flush()
            if acc_chunk is not None:
                ai_msg = AIMessage(
                    content=str(getattr(acc_chunk, "content", "") or ""),
                    tool_calls=list(getattr(acc_chunk, "tool_calls", None) or []),
                )
            else:
                ai_msg = active_llm.invoke(msgs)
        else:
            ai_msg = active_llm.invoke(msgs)

        # Salvage inline <tool_call> blocks if model emitted raw XML without parsed tool_calls
        if (
            isinstance(ai_msg, AIMessage)
            and not getattr(ai_msg, "tool_calls", None)
            and not force_finalize
            and isinstance(ai_msg.content, str)
        ):
            salvaged = _salvage_inline_tool_calls(ai_msg.content)
            if salvaged:
                ai_msg.tool_calls = [
                    {
                        "id": tc["id"],
                        "name": tc["name"],
                        "args": tc["args"],
                        "type": "tool_call",
                    }
                    for tc in salvaged
                ]

        updates: List[BaseMessage] = []
        if not state.get("messages") or not isinstance(state["messages"][0], SystemMessage):
            updates.append(SystemMessage(content=SYSTEM_PROMPT))
        updates.append(ai_msg)
        return {"messages": updates}

    def tools_node(state: InternetAgentState) -> Dict[str, Any]:
        msgs = state.get("messages") or []
        last_msg = msgs[-1] if msgs else None
        raw_calls = getattr(last_msg, "tool_calls", None) or []

        normalized: List[Dict[str, Any]] = []
        for idx, tc in enumerate(raw_calls):
            normalized.append(
                {
                    "id": str(tc.get("id") or f"call_{idx}"),
                    "name": str(tc.get("name") or ""),
                    "args": tc.get("args", {}),
                }
            )

        exec_results = _execute_tool_calls_parallel(
            normalized,
            max_chars=cfg.max_tool_chars,
            on_tool_call=on_tool_call,
        )
        tool_messages: List[BaseMessage] = [
            ToolMessage(
                content=res["content"],
                tool_call_id=res["tool_call_id"],
                name=res["name"],
            )
            for res in exec_results
        ]
        prev_log = list(state.get("tool_calls_log") or [])
        prev_log.extend(exec_results)
        step_count = int(state.get("step_count") or 0) + 1
        return {
            "messages": tool_messages,
            "step_count": step_count,
            "tool_calls_log": prev_log,
        }

    def reflect_and_guard_node(state: InternetAgentState) -> Dict[str, Any]:
        step_count = int(state.get("step_count") or 0)
        max_steps = int(state.get("max_steps") or cfg.max_iterations)
        msgs = state.get("messages") or []
        last_msg = msgs[-1] if msgs else None
        last_content = str(getattr(last_msg, "content", "") or "")

        if step_count >= max_steps or "<tool_call>" in last_content:
            return {
                "force_finalize": True,
                "messages": [HumanMessage(content=_SYNTHESIS_NUDGE)],
            }
        return {"force_finalize": False}

    def route_after_agent(state: InternetAgentState) -> str:
        msgs = state.get("messages") or []
        last_msg = msgs[-1] if msgs else None
        force_finalize = bool(state.get("force_finalize", False))
        if force_finalize:
            return END
        if getattr(last_msg, "tool_calls", None):
            return "tools"
        last_content = str(getattr(last_msg, "content", "") or "")
        if "<tool_call>" in last_content:
            return "reflect_and_guard"
        return END

    def intent_router_node(state: InternetAgentState) -> Dict[str, Any]:
        msgs = list(state.get("messages") or [])
        last_user_content = ""
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                last_user_content = str(getattr(m, "content", "") or "")
                break
        detected_intent = state.get("intent") or classify_user_intent(last_user_content, cfg)
        return {"intent": detected_intent}

    graph = StateGraph(InternetAgentState)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("reflect_and_guard", reflect_and_guard_node)

    graph.add_edge(START, "intent_router")
    graph.add_edge("intent_router", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "reflect_and_guard": "reflect_and_guard",
            END: END,
        },
    )
    graph.add_edge("tools", "reflect_and_guard")
    graph.add_edge("reflect_and_guard", "agent")

    return graph.compile(checkpointer=active_cp)


def run_agent(
    prompt: str,
    config: Optional[AgentConfig] = None,
    stream: bool = False,
    history: Optional[List[Dict[str, Any]]] = None,
    on_tool_call: Optional[Callable[[Dict[str, Any], str, float], None]] = None,
    on_token: Optional[Callable[[str], None]] = None,
    force_stdlib: bool = False,
    thread_id: Optional[str] = None,
    checkpointer: Optional[ColibriSqliteCheckpointer] = None,
) -> AgentRunResult:
    """Run the Colibri internet research agent on `prompt`.

    Automatically uses the compiled LangGraph `StateGraph` with native SQLite
    checkpointing when `langgraph` and `langchain_openai` are installed, or
    falls back to `StdlibColibriAgent` (`urllib.request`) with identical
    state machine semantics and native persistent SQLite checkpointing.
    """
    cfg = config or AgentConfig()
    printed_via_stream = False
    resolved_tid = (thread_id or "").strip()
    active_cp = checkpointer or ColibriSqliteCheckpointer()

    def _stream_token_cb(tok: str) -> None:
        nonlocal printed_via_stream
        if on_token is not None:
            on_token(tok)
        if stream:
            printed_via_stream = True
            print(tok, end="", flush=True)

    token_cb = _stream_token_cb if (stream or on_token is not None) else None

    compiled = build_internet_agent_graph(
        config=cfg,
        on_tool_call=on_tool_call,
        on_token=token_cb,
        stream=stream,
        checkpointer=active_cp,
    )

    if force_stdlib or not _LANGGRAPH_AVAILABLE:
        msgs: List[Dict[str, Any]] = []
        if history:
            msgs.extend(history)
        elif resolved_tid:
            msgs.extend(get_session_history(resolved_tid))
        msgs.append({"role": "user", "content": prompt})
        invoke_payload: Dict[str, Any] = {"messages": msgs, "max_steps": cfg.max_iterations}
        if resolved_tid:
            invoke_payload["thread_id"] = resolved_tid
        res = compiled.invoke(
            invoke_payload,
            thread_id=resolved_tid or None,
        )
        if stream:
            if printed_via_stream:
                print(flush=True)
            elif res.answer:
                print(res.answer, flush=True)
        return res

    effective_history = list(history) if history else (get_session_history(resolved_tid) if resolved_tid else [])
    intent = classify_user_intent(prompt, cfg)

    # Zero-roundtrip speculative local RAG lookup (<10ms)
    enriched_prompt = prompt
    if "[LOCAL MEMORY CONTEXT]" not in prompt:
        mem_ctx = _prefetch_local_memory_context(prompt, cfg)
        if mem_ctx:
            enriched_prompt = prompt + mem_ctx

    # Optional SystemOne zero-token routing (`use_systemone_routing`)
    skip_tools_via_sys1 = False
    if getattr(cfg, "use_systemone_routing", False) and prompt and not getattr(cfg, "is_ollama_or_metal", False):
        try:
            from .systemone import SystemOneClient

            decision = SystemOneClient(config=cfg).route_query(prompt)
            if decision.choice == "direct" and decision.confidence >= 0.65:
                skip_tools_via_sys1 = True
        except Exception:  # noqa: BLE001
            pass

    lc_messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    if effective_history:
        for m in effective_history:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            if role == "system":
                continue
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                lc_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=str(m.get("tool_call_id", "call_hist")),
                        name=str(m.get("name", "tool")),
                    )
                )
            else:
                lc_messages.append(HumanMessage(content=content))
    lc_messages.append(HumanMessage(content=enriched_prompt))

    initial_state: InternetAgentState = {
        "messages": lc_messages,
        "step_count": 0,
        "max_steps": cfg.max_iterations,
        "tool_calls_log": [],
        "force_finalize": skip_tools_via_sys1,
        "intent": intent,
        "thread_id": resolved_tid or "default",
    }
    compile_config = {"configurable": {"thread_id": resolved_tid or "default"}}
    try:
        final_state = compiled.invoke(initial_state, config=compile_config)
    except TypeError:
        final_state = compiled.invoke(initial_state)

    out_messages = final_state.get("messages") or []
    final_answer = ""
    for m in reversed(out_messages):
        if isinstance(m, AIMessage) and m.content:
            final_answer = str(m.content)
            break

    if resolved_tid:
        serialized_msgs: List[Dict[str, Any]] = []
        for m in out_messages:
            role_map = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}
            role = role_map.get(getattr(m, "type", "user"), "user")
            entry: Dict[str, Any] = {"role": role, "content": str(getattr(m, "content", ""))}
            if role == "tool":
                entry["tool_call_id"] = str(getattr(m, "tool_call_id", ""))
                entry["name"] = str(getattr(m, "name", ""))
            serialized_msgs.append(entry)
        try:
            active_cp.save_checkpoint(
                thread_id=resolved_tid,
                messages=serialized_msgs,
                intent=str(final_state.get("intent") or intent),
                step_count=int(final_state.get("step_count") or 0),
            )
        except Exception:  # noqa: BLE001
            pass

    if stream:
        if printed_via_stream:
            print(flush=True)
        elif final_answer:
            print(final_answer, flush=True)

    return AgentRunResult(
        answer=final_answer,
        output=final_answer,
        messages=serialized_msgs,
        tool_calls=list(final_state.get("tool_calls_log") or []),
        steps=int(final_state.get("step_count") or 0),
        runner="langgraph",
        model=cfg.model,
        compacted_messages=0,
        reasoning="",
        intent=str(final_state.get("intent") or intent),
        thread_id=resolved_tid or "default",
    )
