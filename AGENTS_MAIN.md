# Colibri Desktop AI Companion Architecture (`AGENTS_MAIN.md`)

This document defines the multi-agent architecture, KV-cache optimization rules, context handoff protocols, 5-domain tool suite, and pre-flight verification standards for the Colibri Desktop AI Companion (`src/ai/`), purpose-built to run on Colibri's local inference gateway (`coli serve` / `qwen36-35b`) on Apple Silicon / macOS.

---

## 1. Repository & Module Layout (`src/ai/`)

```text
src/ai/
├── __init__.py          # Public package exports (AgentConfig, persona, SystemOneClient, 23+ tools, graph runner)
├── config.py            # AgentConfig dataclass + auto-detection of served model & context window (/v1/models)
├── persona.py           # English-only Jimmy bot companion persona & deterministic system prompt builder
├── gateway.py           # Unified ColibriGatewayPort seam + HttpGatewayAdapter (urllib/SSE) & InMemoryGatewayAdapter
├── session.py           # Standalone SessionStore seam + SqliteSessionStore (dual Colibri/LangGraph checkpointer) & InMemorySessionStore
├── context.py           # ConversationContext: byte-deterministic KV prefix, slot coordinator, & storage sanitization
├── systemone.py         # Client for Colibri's zero-generation decision endpoint (POST /v1/systemone)
├── graph.py             # Unified ColibriAgent ReAct runner (agent -> tools -> reflect_and_guard -> end)
├── langgraph_adapter.py # Optional LangGraph StateGraph adapter & fallback delegator
├── cli.py               # Interactive & one-shot CLI entry point (/tools, /birthday, /memory, /voice, /speak, /ocr)
├── test_agent.py        # Self-contained unit test suite covering all companion domains, persona, SSRF, and ColibriAgent
└── tools/
    ├── __init__.py      # Unified ToolSuite / ToolRegistry loader, OPENAI_TOOLS_SCHEMA, and TOOL_REGISTRY
    ├── internet.py      # 11 SSRF-safe internet & utility tools (search, fetch, deep research, ArXiv, Wiki, HN, GitHub, Weather, RSS, math)
    ├── vision.py        # Multilingual OCR (`heb+eng`) and cascading scene description
    ├── audio.py         # Local OpenAI Whisper STT (`transcribe_audio`) + Edge-TTS synthesis
    ├── rag.py           # Private local RAG & fact vault persisted at `data/memory_db/` (`query_memory`, `save_fact`, ChromaDB + JSON)
    └── code_runner.py   # AST-sandboxed Python calculation & snippet runner (`run_python_calculation`)
```

---

## 2. Multi-Agent Collaboration & Context Handoffs

1. **Decentralized Node Responsibilities**:
   - **Router / Classifier (`systemone.py` / `agent` Node)**: Uses Colibri's `POST /v1/systemone` or structured tool-calling on `qwen38-flash-next` / `qwen36-35b` to decide whether a query requires live internet research, personal RAG memory lookup (`query_memory`), vision/OCR, voice synthesis, or fast companion direct response.
   - **Multimodal Executor (`tools/` / `tools` Node)**: Executes SSRF-safe internet queries, local OCR/vision, Whisper STT / Edge-TTS, local ChromaDB/JSON memory retrieval, and AST-sandboxed Python calculations in parallel (`ThreadPoolExecutor`), returning bounded payloads (`max_chars <= COLIBRI_MAX_TOOL_CHARS`).
   - **Verifier / Reflection Guard (`reflect_and_guard` Node in `graph.py`)**: Tracks tool step budget (`max_steps`, default `8`) and recovers from malformed `<tool_call>` syntax by nudging final cited synthesis.

2. **Minimal Context Handoffs**:
   - Never dump raw HTML or unbounded JSON into the conversation state.
   - Every tool output is stripped of boilerplate/navigation (`trafilatura` / stdlib HTML parser) and capped to `COLIBRI_MAX_TOOL_CHARS` (default `8000` chars, elevated via `kv_fp8` unified memory savings).
   - Prefer `deep_web_research` for multi-source questions: it searches and extracts the top $N$ pages in a **single tool turn**, saving multiple prefill round-trips on `qwen36-35b`.

---

## 3. Colibri Engine & KV-Cache Prefix Preservation (`qwen36-35b`)

Colibri (`coli serve`) maintains stateful KV + DeltaNet/recurrent slots across HTTP requests when the token prefix matches:

1. **Dedicated KV Cache Slot (`cache_slot`)**:
   - Always pass `cache_slot` (default `COLIBRI_CACHE_SLOT=0`) in `extra_body` on `/v1/chat/completions` and `/v1/systemone` so multi-turn agent loops reuse the warm KV slot.
2. **Immutable System & Tool Prefix (`persona.py`)**:
   - Keep `SYSTEM_PROMPT` (`build_system_prompt()`) and `OPENAI_TOOLS_SCHEMA` **identical and deterministic** at index `0` across every turn of a session so the cached KV prefix is never invalidated.
3. **Thinking & Prefix Reuse on Qwen3.6 (`preserve_thinking`)**:
   - Pass `preserve_thinking: True` and preserve `reasoning_content` across assistant tool-call turns so Qwen3.6's `<think>` blocks match the cached KV prefix byte-for-byte.
4. **Automatic Context Compaction (`context_compaction`)**:
   - Pass `"context_compaction": "auto"` in `extra_body` so long sessions automatically summarize older turns on the server without raising `context_length_exceeded`.
5. **Zero-Generation Decisions (`POST /v1/systemone`)**:
   - For closed-set routing or verification decisions (`noul`, `choice`, `score`), use `SystemOneClient` (`POST /v1/systemone`) to score options in a single forward pass with calibrated probabilities and `confidence`.

---

## 4. Complete 6-Domain Tool Suite (`src/ai/tools/`)

### 4.1 Internet & Research (`src/ai/tools/internet.py`)
All URL fetches enforce strict SSRF protection (`_public_url`: only `http`/`https` resolving to global public IPs; all redirects re-verified):

| Tool Name | Purpose |
|---|---|
| `web_search` | Live web search via `duckduckgo_search` (`DDGS`), SearXNG, or DuckDuckGo HTML fallback. |
| `fetch_webpage` / `fetch_webpage_content` | SSRF-safe readable article extraction (`trafilatura` + stdlib HTML parser) + outbound links. |
| `deep_web_research` | One-shot search + extraction of top $N$ pages to minimize LLM round-trips. |
| `wikipedia_search` | Authoritative Wikipedia REST/Action API summary and canonical URLs. |
| `arxiv_search` | ArXiv API (`export.arxiv.org`) paper search with authors, dates, abstracts, and PDF links. |
| `hackernews_search` | Algolia HN API search for tech news, stories, and discussion threads. |
| `github_repo_search` | GitHub REST API repository search with stars, language, and descriptions. |
| `weather_lookup` | Keyless live weather via Open-Meteo Geocoding + Forecast APIs. |
| `fetch_json_or_rss` | SSRF-safe reader for public JSON endpoints and RSS/Atom feeds. |
| `calculate` | Safe AST math evaluator (`sin`, `cos`, `sqrt`, `log`, `pi`, `e`, powers, percentages). |

### 4.2 Vision & Perception (`src/ai/tools/vision.py`)
| Tool Name | Purpose |
|---|---|
| `extract_text_from_image` | Multilingual OCR (`heb+eng` with fallback to `eng`) via `pytesseract` and `Pillow`. |
| `describe_image_scene` | Cascading visual scene analysis (Gemini Flash -> OpenRouter -> local Colibri vision -> OCR/PIL fallback). |

### 4.3 Voice & Audio (`src/ai/tools/audio.py`)
| Tool Name | Purpose |
|---|---|
| `transcribe_audio` | Local Speech-to-Text via `openai-whisper` with model caching and language detection. |
| `text_to_speech` / `speak_text` | Neural TTS via `edge-tts` for natural companion voice output. |

### 4.4 Private Local RAG Memory (`src/ai/tools/rag.py`)
Persists 100% locally in `data/memory_db/` (and `~/.colibri/memory/`) using ChromaDB (`all-MiniLM-L6-v2`) + JSON vault:

| Tool Name | Purpose |
|---|---|
| `query_memory` / `query_local_knowledge` | Semantic + keyword retrieval of personal preferences, recipes, bookmarks, and family notes. |
| `save_fact` / `save_conversation_fact` | Dynamically stores new facts and preferences into `data/memory_db/`. |
| `list_saved_facts` | Lists all stored memories in the local companion vault, optionally filtered by category. |
| `ingest_url_or_note` | Scrapes and segments ingested content into `recipes_and_food`, `web_bookmarks_and_reads`, or `personal_facts_and_notes`. |
| `scale_recipe` | Accurately scales recipe ingredients and quantities to `target_servings`. |

### 4.5 Safe Code & Math Runner (`src/ai/tools/code_runner.py`)
| Tool Name | Purpose |
|---|---|
| `run_python_calculation` | AST-validated sandboxed Python execution (`PythonAstREPLTool` + restricted builtins for `math`, `datetime`, `statistics`). |

---

## 5. Mandatory Pre-Flight Checks & Verification

Before completing any modification to `src/ai/`, run and pass all checks:

```bash
# 1. Syntax & bytecode compilation check
python3 -m py_compile src/ai/__init__.py src/ai/config.py src/ai/persona.py src/ai/gateway.py src/ai/session.py src/ai/context.py src/ai/systemone.py src/ai/graph.py src/ai/langgraph_adapter.py src/ai/cli.py src/ai/test_agent.py src/ai/tools/__init__.py src/ai/tools/internet.py src/ai/tools/vision.py src/ai/tools/audio.py src/ai/tools/rag.py src/ai/tools/code_runner.py

# 2. Run unit & integration test suite
python3 -m unittest src.ai.test_agent -v

# 3. Smoke-test CLI & tool registration
python3 -m src.ai.cli --check
```
