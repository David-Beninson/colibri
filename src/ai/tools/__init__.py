"""Unified Multimodal, Internet, Memory, Code & Hardware Tools (`src/ai/tools/__init__.py`).

Aggregates tools across all 6 Desktop Holographic AI Companion domain modules:
- `internet.py`    : SSRF-safe web search, webpage content, deep research, ArXiv, Wiki, HN, GitHub, Weather, RSS
- `vision.py`      : OCR (`extract_text_from_image`), scene description (`describe_image_scene`), person angle tracking (`detect_person_angle`)
- `audio.py`/`voice.py`: Speech-to-Text (`transcribe_audio`) and Hebrew/English neural TTS (`text_to_speech`, `speak_text`)
- `rag.py`         : Private local personal memory & RAG (`query_local_knowledge`, `save_conversation_fact`, `list_saved_facts`)
- `code_runner.py` : Sandboxed Python math/data execution (`run_python_calculation`)
- `device.py`      : Raspberry Pi servo tracking & Pepper's Ghost 3D avatar states (`rotate_device`, `set_avatar_state`, `trigger_hologram_effect`)
"""
from __future__ import annotations

import importlib
import json
from typing import Any, Callable, Dict, List, Tuple

from .device import (
    DEVICE_TOOL_REGISTRY,
    DEVICE_TOOLS,
    DEVICE_TOOLS_SCHEMA,
    get_device_state,
    rotate_device,
    rotate_device_servo,
    set_avatar_state,
    trigger_avatar_state,
    trigger_hologram_effect,
)
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
        detect_person_angle,
        extract_text_from_image,
    )
except Exception:  # noqa: BLE001
    VISION_TOOLS = []
    VISION_TOOL_REGISTRY = {}
    VISION_TOOLS_SCHEMA = []
    extract_text_from_image = None  # type: ignore[assignment]
    describe_image_scene = None  # type: ignore[assignment]
    detect_person_angle = None  # type: ignore[assignment]

try:
    from .audio import (  # type: ignore
        AUDIO_TOOL_REGISTRY,
        AUDIO_TOOLS,
        AUDIO_TOOLS_SCHEMA,
        speak_text,
        text_to_speech,
        transcribe_audio,
    )
except Exception:  # noqa: BLE001
    try:
        from .voice import (  # type: ignore
            AUDIO_TOOL_REGISTRY,
            AUDIO_TOOLS,
            AUDIO_TOOLS_SCHEMA,
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

try:
    from .rag import (  # type: ignore
        RAG_TOOL_REGISTRY,
        RAG_TOOLS,
        RAG_TOOLS_SCHEMA,
        list_saved_facts,
        query_local_knowledge,
        query_memory,
        save_conversation_fact,
        save_fact,
    )
except Exception:  # noqa: BLE001
    RAG_TOOLS = []
    RAG_TOOL_REGISTRY = {}
    RAG_TOOLS_SCHEMA = []
    query_memory = None  # type: ignore[assignment]
    save_fact = None  # type: ignore[assignment]
    query_local_knowledge = None  # type: ignore[assignment]
    save_conversation_fact = None  # type: ignore[assignment]
    list_saved_facts = None  # type: ignore[assignment]

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
        "rotate_device_servo",
        "trigger_avatar_state",
    }
)

# Core fast-path companion tools (Slot 0) covering memory, voice, motor/avatar, math, weather & search
FAST_COMPANION_TOOL_NAMES = (
    "query_memory",
    "save_fact",
    "web_search",
    "weather_lookup",
    "calculate",
    "rotate_device",
    "set_avatar_state",
    "trigger_hologram_effect",
    "text_to_speech",
)

_CACHED_TOOL_SUITES: Tuple[Dict[str, Callable[..., Any]], List[Dict[str, Any]], List[Any]] | None = None


def load_all_tool_suites(
    use_cache: bool = True,
) -> Tuple[Dict[str, Callable[..., Any]], List[Dict[str, Any]], List[Any]]:
    """Dynamically merge registries, OpenAI schemas, and tool objects across all 6 domain modules."""
    global _CACHED_TOOL_SUITES
    if use_cache and _CACHED_TOOL_SUITES is not None:
        return _CACHED_TOOL_SUITES

    merged_registry: Dict[str, Callable[..., Any]] = {}
    merged_schemas: List[Dict[str, Any]] = []
    seen_schema_names = set()
    merged_tools: List[Any] = []
    seen_tool_names = set()

    module_specs = [
        (".internet", "TOOL_REGISTRY", "OPENAI_TOOLS_SCHEMA", "INTERNET_TOOLS"),
        (".vision", "VISION_TOOL_REGISTRY", "VISION_TOOLS_SCHEMA", "VISION_TOOLS"),
        (".audio", "AUDIO_TOOL_REGISTRY", "AUDIO_TOOLS_SCHEMA", "AUDIO_TOOLS"),
        (".voice", "VOICE_TOOL_REGISTRY", "VOICE_TOOLS_SCHEMA", "VOICE_TOOLS"),
        (".rag", "RAG_TOOL_REGISTRY", "RAG_TOOLS_SCHEMA", "RAG_TOOLS"),
        (".code_runner", "CODE_TOOL_REGISTRY", "CODE_TOOLS_SCHEMA", "CODE_TOOLS"),
        (".device", "DEVICE_TOOL_REGISTRY", "DEVICE_TOOLS_SCHEMA", "DEVICE_TOOLS"),
    ]

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

    _CACHED_TOOL_SUITES = (merged_registry, merged_schemas, merged_tools)
    return _CACHED_TOOL_SUITES


TOOL_REGISTRY, OPENAI_TOOLS_SCHEMA, ALL_TOOLS = load_all_tool_suites()

# Deduplicated schema list (drops 6 duplicate alias schemas while keeping all 22 unique tools)
DEDUPLICATED_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    s
    for s in OPENAI_TOOLS_SCHEMA
    if ((s.get("function") or {}).get("name") or s.get("name")) not in ALIAS_TOOL_NAMES
]

# Fast-path Companion schema list (Slot 0 fixed prefix for rapid conversational turns)
FAST_COMPANION_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    s
    for s in OPENAI_TOOLS_SCHEMA
    if ((s.get("function") or {}).get("name") or s.get("name")) in FAST_COMPANION_TOOL_NAMES
]


def get_langchain_tools(fast_mode: bool = False, deduplicate: bool = False) -> List[Any]:
    """Return registered tools unwrapped for LangChain `bind_tools()`."""
    _, _, tools_list = load_all_tool_suites()
    selected: List[Any] = []
    for t in tools_list:
        t_name = getattr(t, "name", getattr(t, "__name__", ""))
        if fast_mode and t_name not in FAST_COMPANION_TOOL_NAMES:
            continue
        if deduplicate and t_name in ALIAS_TOOL_NAMES:
            continue
        selected.append(getattr(t, "_lc", t))
    return selected


def execute_tool_call(
    name: str,
    arguments: Any,
    max_chars: int = 4000,
) -> str:
    """Execute any registered tool across internet, vision, voice, RAG, code, and device suites."""
    registry, _, _ = load_all_tool_suites()
    fn = registry.get(name)
    if fn is None:
        return json.dumps(
            {"error": f"Unknown tool: {name}", "available_tools": sorted(registry)}
        )

    if isinstance(arguments, str):
        arguments = arguments.strip()
        if not arguments:
            kwargs: Dict[str, Any] = {}
        else:
            try:
                kwargs = json.loads(arguments)
            except Exception as exc:  # noqa: BLE001
                return json.dumps(
                    {"error": f"Invalid tool JSON arguments for {name}: {exc}"}
                )
    elif isinstance(arguments, dict):
        kwargs = dict(arguments)
    else:
        kwargs = {}

    try:
        out = str(fn(**kwargs))
    except Exception as exc:  # noqa: BLE001
        out = json.dumps({"error": f"Tool {name} raised {type(exc).__name__}: {exc}"})

    if max_chars > 0 and len(out) > max_chars:
        return out[:max_chars] + "...[truncated]"
    return out


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
    "DEVICE_TOOLS",
    "DEVICE_TOOL_REGISTRY",
    "DEVICE_TOOLS_SCHEMA",
    "OPENAI_TOOLS_SCHEMA",
    "DEDUPLICATED_TOOLS_SCHEMA",
    "FAST_COMPANION_TOOLS_SCHEMA",
    "FAST_COMPANION_TOOL_NAMES",
    "ALIAS_TOOL_NAMES",
    "TOOL_REGISTRY",
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
    "detect_person_angle",
    "transcribe_audio",
    "text_to_speech",
    "speak_text",
    "query_memory",
    "save_fact",
    "query_local_knowledge",
    "save_conversation_fact",
    "list_saved_facts",
    "run_python_calculation",
    "rotate_device",
    "rotate_device_servo",
    "set_avatar_state",
    "trigger_avatar_state",
    "trigger_hologram_effect",
    "get_device_state",
]
