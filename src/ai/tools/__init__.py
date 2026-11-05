"""Unified Multimodal, Internet, Memory & Code Tools (`src/ai/tools/__init__.py`).

Aggregates tools across all Desktop AI Companion domain modules:
- `internet.py`    : SSRF-safe web search, webpage content, deep research, ArXiv, Wiki, HN, GitHub, Weather, RSS
- `vision.py`      : OCR (`extract_text_from_image`), scene description (`describe_image_scene`)
- `audio.py`       : Speech-to-Text (`transcribe_audio`) and Hebrew/English neural TTS (`text_to_speech`, `speak_text`)
- `rag.py`         : Private local personal memory & RAG (`query_local_knowledge`, `save_conversation_fact`, `list_saved_facts`)
- `code_runner.py` : Sandboxed Python math/data execution (`run_python_calculation`)
"""
from __future__ import annotations

import importlib
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .internet import (
    INTERNET_TOOLS,
    OPENAI_TOOLS_SCHEMA as INTERNET_TOOLS_SCHEMA,
    TOOL_REGISTRY as INTERNET_TOOL_REGISTRY,
    arxiv_search,
    calculate,
    deep_web_research,
    fetch_json_or_rss,
    fetch_webpage,
    github_repo_search,
    hackernews_search,
    weather_lookup,
    web_search,
    wikipedia_search,
)

# Optional attribute from internet.py (`fetch_webpage_content`)
fetch_webpage_content = getattr(
    importlib.import_module(".internet", __package__),
    "fetch_webpage_content",
    fetch_webpage,
)

# Optional domain modules (loaded gracefully whether already present or added dynamically)
try:
    from .vision import (  # type: ignore
        VISION_TOOL_REGISTRY,
        VISION_TOOLS,
        VISION_TOOLS_SCHEMA,
        describe_image_scene,
        extract_text_from_image,
    )
except Exception:  # noqa: BLE001
    VISION_TOOLS = []
    VISION_TOOL_REGISTRY = {}
    VISION_TOOLS_SCHEMA = []
    extract_text_from_image = None  # type: ignore[assignment]
    describe_image_scene = None  # type: ignore[assignment]

try:
    from .audio import (  # type: ignore
        AUDIO_TOOL_REGISTRY,
        AUDIO_TOOLS,
        AUDIO_TOOLS_SCHEMA,
        SentenceTTSStreamer,
        speak_text,
        text_to_speech,
        transcribe_audio,
    )
except Exception:  # noqa: BLE001
    AUDIO_TOOLS = []
    AUDIO_TOOL_REGISTRY = {}
    AUDIO_TOOLS_SCHEMA = []
    transcribe_audio = None  # type: ignore[assignment]
    text_to_speech = None  # type: ignore[assignment]
    speak_text = None  # type: ignore[assignment]
    SentenceTTSStreamer = None  # type: ignore[assignment]

try:
    from .rag import (  # type: ignore
        CANONICAL_COLLECTIONS,
        RAG_TOOL_REGISTRY,
        RAG_TOOLS,
        RAG_TOOLS_SCHEMA,
        _resolve_collection,
        ingest_url_or_note,
        list_saved_facts,
        query_local_knowledge,
        query_memory,
        save_conversation_fact,
        save_fact,
        scale_recipe,
    )
except Exception:  # noqa: BLE001
    CANONICAL_COLLECTIONS = ()  # type: ignore[assignment]
    RAG_TOOLS = []
    RAG_TOOL_REGISTRY = {}
    RAG_TOOLS_SCHEMA = []
    _resolve_collection = None  # type: ignore[assignment]
    query_memory = None  # type: ignore[assignment]
    save_fact = None  # type: ignore[assignment]
    query_local_knowledge = None  # type: ignore[assignment]
    save_conversation_fact = None  # type: ignore[assignment]
    list_saved_facts = None  # type: ignore[assignment]
    ingest_url_or_note = None  # type: ignore[assignment]
    scale_recipe = None  # type: ignore[assignment]

try:
    from .code_runner import (  # type: ignore
        CODE_TOOL_REGISTRY,
        CODE_TOOLS,
        CODE_TOOLS_SCHEMA,
        run_python_calculation,
    )
except Exception:  # noqa: BLE001
    CODE_TOOLS = []
    CODE_TOOL_REGISTRY = {}
    CODE_TOOLS_SCHEMA = []
    run_python_calculation = None  # type: ignore[assignment]


# Redundant alias tool names that remain executable in `TOOL_REGISTRY`
# but can be omitted in deduplicated/fast schemas to save prefill tokens.
ALIAS_TOOL_NAMES = frozenset(
    {
        "fetch_webpage_content",
        "speak_text",
        "query_local_knowledge",
        "save_conversation_fact",
    }
)

# Core fast-path companion tools (Slot 0) covering memory, voice, math, weather & search
FAST_COMPANION_TOOL_NAMES = (
    "query_memory",
    "save_fact",
    "web_search",
    "weather_lookup",
    "calculate",
    "text_to_speech",
)

class ToolRegistry:
    """Consolidated deep tool registry for companion tools, OpenAI function schemas, and execution."""

    def __init__(
        self,
        registry: Optional[Dict[str, Callable[..., Any]]] = None,
        schemas: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Any]] = None,
    ) -> None:
        self._registry: Dict[str, Callable[..., Any]] = dict(registry) if registry else {}
        self._schemas: List[Dict[str, Any]] = list(schemas) if schemas else []
        self._tools: List[Any] = list(tools) if tools else []

    @classmethod
    def from_modules(
        cls,
        module_specs: Optional[List[Tuple[str, str, str, str]]] = None,
    ) -> "ToolRegistry":
        """Build and populate a ToolRegistry by scanning tool modules."""
        if module_specs is None:
            module_specs = [
                (".internet", "TOOL_REGISTRY", "OPENAI_TOOLS_SCHEMA", "INTERNET_TOOLS"),
                (".vision", "VISION_TOOL_REGISTRY", "VISION_TOOLS_SCHEMA", "VISION_TOOLS"),
                (".audio", "AUDIO_TOOL_REGISTRY", "AUDIO_TOOLS_SCHEMA", "AUDIO_TOOLS"),
                (".rag", "RAG_TOOL_REGISTRY", "RAG_TOOLS_SCHEMA", "RAG_TOOLS"),
                (".code_runner", "CODE_TOOL_REGISTRY", "CODE_TOOLS_SCHEMA", "CODE_TOOLS"),
            ]

        merged_registry: Dict[str, Callable[..., Any]] = {}
        merged_schemas: List[Dict[str, Any]] = []
        seen_schema_names = set()
        merged_tools: List[Any] = []
        seen_tool_names = set()

        for mod_name, reg_attr, schema_attr, tools_attr in module_specs:
            try:
                mod = importlib.import_module(mod_name, __package__)
            except Exception:  # noqa: BLE001
                continue

            reg = getattr(mod, reg_attr, None) or getattr(mod, "TOOL_REGISTRY", {})
            if isinstance(reg, dict):
                merged_registry.update(reg)

            schemas = getattr(mod, schema_attr, None) or getattr(mod, "OPENAI_TOOLS_SCHEMA", [])
            if isinstance(schemas, list):
                for s in schemas:
                    fn_name = (s.get("function") or {}).get("name") or s.get("name")
                    if fn_name and fn_name not in seen_schema_names:
                        seen_schema_names.add(fn_name)
                        merged_schemas.append(s)

            tools_list = getattr(mod, tools_attr, None) or getattr(mod, "TOOLS", [])
            if isinstance(tools_list, list):
                for t in tools_list:
                    t_name = getattr(t, "name", getattr(t, "__name__", None))
                    if t_name and t_name not in seen_tool_names:
                        seen_tool_names.add(t_name)
                        merged_tools.append(t)

        return cls(registry=merged_registry, schemas=merged_schemas, tools=merged_tools)

    @property
    def registry(self) -> Dict[str, Callable[..., Any]]:
        return self._registry

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

    @property
    def tools(self) -> List[Any]:
        return self._tools

    @property
    def deduplicated_schemas(self) -> List[Dict[str, Any]]:
        return [
            s
            for s in self._schemas
            if ((s.get("function") or {}).get("name") or s.get("name")) not in ALIAS_TOOL_NAMES
        ]

    @property
    def fast_companion_schemas(self) -> List[Dict[str, Any]]:
        return [
            s
            for s in self._schemas
            if ((s.get("function") or {}).get("name") or s.get("name")) in FAST_COMPANION_TOOL_NAMES
        ]

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        schema: Optional[Dict[str, Any]] = None,
        tool_obj: Optional[Any] = None,
    ) -> None:
        """Register or override a callable tool, schema, and tool object."""
        self._registry[name] = fn
        if schema is not None:
            s_name = (schema.get("function") or {}).get("name") or schema.get("name")
            self._schemas = [
                s
                for s in self._schemas
                if ((s.get("function") or {}).get("name") or s.get("name")) != s_name
            ]
            self._schemas.append(schema)
        if tool_obj is not None:
            t_name = getattr(tool_obj, "name", getattr(tool_obj, "__name__", None))
            self._tools = [
                t
                for t in self._tools
                if getattr(t, "name", getattr(t, "__name__", None)) != t_name
            ]
            self._tools.append(tool_obj)

    def get(self, name: str, default: Any = None) -> Any:
        return self._registry.get(name, default)

    def __getitem__(self, name: str) -> Callable[..., Any]:
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry)

    def keys(self):
        return self._registry.keys()

    def values(self):
        return self._registry.values()

    def items(self):
        return self._registry.items()

    def get_langchain_tools(self, fast_mode: bool = False, deduplicate: bool = False) -> List[Any]:
        """Return registered tools unwrapped for LangChain `bind_tools()`."""
        selected: List[Any] = []
        for t in self._tools:
            t_name = getattr(t, "name", getattr(t, "__name__", ""))
            if fast_mode and t_name not in FAST_COMPANION_TOOL_NAMES:
                continue
            if deduplicate and t_name in ALIAS_TOOL_NAMES:
                continue
            selected.append(getattr(t, "_lc", t))
        return selected

    def execute(
        self,
        name: str,
        args: Union[str, Dict[str, Any], Any] = None,
        max_chars: int = 4000,
    ) -> str:
        """Execute a registered tool by name with JSON string or dict arguments.

        Handles JSON parsing/dict normalization, argument dispatch, error handling,
        and character truncation.
        """
        fn = self._registry.get(name)
        if fn is None:
            return json.dumps(
                {"error": f"Unknown tool: {name}", "available_tools": sorted(self._registry.keys())}
            )

        if isinstance(args, str):
            clean_args = args.strip()
            if not clean_args:
                kwargs: Dict[str, Any] = {}
            else:
                try:
                    kwargs = json.loads(clean_args)
                except Exception as exc:  # noqa: BLE001
                    return json.dumps(
                        {"error": f"Invalid tool JSON arguments for {name}: {exc}"}
                    )
        elif isinstance(args, dict):
            kwargs = dict(args)
        elif args is None:
            kwargs = {}
        else:
            kwargs = {}

        if not isinstance(kwargs, dict):
            return json.dumps(
                {"error": f"Tool arguments for {name} must decode to a dictionary/object."}
            )

        try:
            out = str(fn(**kwargs))
        except Exception as exc:  # noqa: BLE001
            out = json.dumps({"error": f"Tool {name} raised {type(exc).__name__}: {exc}"})

        if max_chars > 0 and len(out) > max_chars:
            return out[:max_chars] + "...[truncated]"
        return out


TOOL_REGISTRY_OBJ: ToolRegistry = ToolRegistry.from_modules()


def load_all_tool_suites(
    use_cache: bool = True,
) -> Tuple[Dict[str, Callable[..., Any]], List[Dict[str, Any]], List[Any]]:
    """Dynamically merge registries, OpenAI schemas, and tool objects across domain modules."""
    global TOOL_REGISTRY_OBJ
    if not use_cache or not TOOL_REGISTRY_OBJ.registry:
        TOOL_REGISTRY_OBJ = ToolRegistry.from_modules()
    return TOOL_REGISTRY_OBJ.registry, TOOL_REGISTRY_OBJ.schemas, TOOL_REGISTRY_OBJ.tools


TOOL_REGISTRY: Dict[str, Callable[..., Any]] = TOOL_REGISTRY_OBJ.registry
OPENAI_TOOLS_SCHEMA: List[Dict[str, Any]] = TOOL_REGISTRY_OBJ.schemas
ALL_TOOLS: List[Any] = TOOL_REGISTRY_OBJ.tools
DEDUPLICATED_TOOLS_SCHEMA: List[Dict[str, Any]] = TOOL_REGISTRY_OBJ.deduplicated_schemas
FAST_COMPANION_TOOLS_SCHEMA: List[Dict[str, Any]] = TOOL_REGISTRY_OBJ.fast_companion_schemas


def get_langchain_tools(fast_mode: bool = False, deduplicate: bool = False) -> List[Any]:
    """Return registered tools unwrapped for LangChain `bind_tools()`."""
    return TOOL_REGISTRY_OBJ.get_langchain_tools(fast_mode=fast_mode, deduplicate=deduplicate)


def execute_tool_call(
    name: str,
    arguments: Any = None,
    max_chars: int = 4000,
) -> str:
    """Execute any registered tool across internet, vision, audio, RAG, and code suites."""
    return TOOL_REGISTRY_OBJ.execute(name, arguments, max_chars=max_chars)


__all__ = [
    "ALL_TOOLS",
    "INTERNET_TOOLS",
    "INTERNET_TOOL_REGISTRY",
    "INTERNET_TOOLS_SCHEMA",
    "VISION_TOOLS",
    "VISION_TOOL_REGISTRY",
    "VISION_TOOLS_SCHEMA",
    "AUDIO_TOOLS",
    "AUDIO_TOOL_REGISTRY",
    "AUDIO_TOOLS_SCHEMA",
    "RAG_TOOLS",
    "RAG_TOOL_REGISTRY",
    "RAG_TOOLS_SCHEMA",
    "CODE_TOOLS",
    "CODE_TOOL_REGISTRY",
    "CODE_TOOLS_SCHEMA",
    "OPENAI_TOOLS_SCHEMA",
    "DEDUPLICATED_TOOLS_SCHEMA",
    "FAST_COMPANION_TOOLS_SCHEMA",
    "FAST_COMPANION_TOOL_NAMES",
    "ALIAS_TOOL_NAMES",
    "TOOL_REGISTRY",
    "TOOL_REGISTRY_OBJ",
    "ToolRegistry",
    "SentenceTTSStreamer",
    "load_all_tool_suites",
    "get_langchain_tools",
    "execute_tool_call",
    "web_search",
    "fetch_webpage",
    "fetch_webpage_content",
    "deep_web_research",
    "wikipedia_search",
    "arxiv_search",
    "hackernews_search",
    "github_repo_search",
    "weather_lookup",
    "fetch_json_or_rss",
    "calculate",
    "extract_text_from_image",
    "describe_image_scene",
    "transcribe_audio",
    "text_to_speech",
    "speak_text",
    "query_memory",
    "save_fact",
    "query_local_knowledge",
    "save_conversation_fact",
    "list_saved_facts",
    "ingest_url_or_note",
    "scale_recipe",
    "CANONICAL_COLLECTIONS",
    "_resolve_collection",
    "run_python_calculation",
]
