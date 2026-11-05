"""Private local RAG Memory & Fact Vault for the Desktop Holographic AI Companion (`src/ai/tools/rag.py`).

100% local storage with zero external leaks:
- Default storage directory: ``data/memory_db/`` inside the repository (or ``COLIBRI_MEMORY_DIR`` env override).
- Persists every personal fact/preference (`key`, `value`, `category`, `updated_at`, `collection`) to a local
  JSON vault (`data/memory_db/facts.json`).
- Automatically indexes into a persistent local ChromaDB vector store (`data/memory_db/chroma`) using
  `sentence-transformers/all-MiniLM-L6-v2` embeddings whenever `chromadb` and `sentence_transformers`
  are installed, and seamlessly combines vector search with keyword/token-overlap fallback.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .internet import _tool_decorator

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_MEMORY_DIR = os.path.join(_REPO_ROOT, "data", "memory_db")

_CHROMA_CLIENT_CACHE: Dict[str, Any] = {}
_EMBEDDING_FN_CACHE: Dict[str, Any] = {}
_FACTS_MEM_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_QUERY_RESULT_CACHE: Dict[Tuple[str, str, int, float], str] = {}


def _memory_dir() -> str:
    base = os.environ.get("COLIBRI_MEMORY_DIR", _DEFAULT_MEMORY_DIR)
    path = os.path.abspath(os.path.expanduser(base))
    os.makedirs(path, exist_ok=True)
    return path


def _facts_file() -> str:
    return os.path.join(_memory_dir(), "facts.json")


def _load_facts() -> List[Dict[str, Any]]:
    fpath = _facts_file()
    if not os.path.isfile(fpath):
        _FACTS_MEM_CACHE.pop(fpath, None)
        return []
    try:
        mtime = os.path.getmtime(fpath)
        cached = _FACTS_MEM_CACHE.get(fpath)
        if cached is not None and cached[0] == mtime:
            return [dict(item) for item in cached[1]]
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else []
        _FACTS_MEM_CACHE[fpath] = (mtime, [dict(item) for item in rows])
        return rows
    except Exception:  # noqa: BLE001
        return []


def _save_facts(facts: List[Dict[str, Any]]) -> None:
    fpath = _facts_file()
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)
    try:
        mtime = os.path.getmtime(fpath)
    except OSError:
        mtime = 0.0
    _FACTS_MEM_CACHE[fpath] = (mtime, [dict(item) for item in facts])
    _QUERY_RESULT_CACHE.clear()


def _get_chroma_collection(collection_name: str = "companion_memory") -> Optional[Any]:
    """Return a persistent ChromaDB collection backed by `all-MiniLM-L6-v2` if installed."""
    try:
        import chromadb  # type: ignore
        from chromadb.utils import embedding_functions  # type: ignore

        chroma_path = os.path.join(_memory_dir(), "chroma")
        if chroma_path not in _CHROMA_CLIENT_CACHE:
            _CHROMA_CLIENT_CACHE[chroma_path] = chromadb.PersistentClient(path=chroma_path)
        client = _CHROMA_CLIENT_CACHE[chroma_path]

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        if model_name not in _EMBEDDING_FN_CACHE:
            try:
                _EMBEDDING_FN_CACHE[model_name] = (
                    embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=model_name
                    )
                )
            except Exception:  # noqa: BLE001
                _EMBEDDING_FN_CACHE[model_name] = None

        emb_fn = _EMBEDDING_FN_CACHE.get(model_name)
        if emb_fn is not None:
            return client.get_or_create_collection(
                name=collection_name, embedding_function=emb_fn
            )
        return client.get_or_create_collection(name=collection_name)
    except Exception:  # noqa: BLE001
        return None


def _store_fact_entry(
    key: str,
    value: str,
    category: str = "general",
    collection_name: str = "companion_memory",
) -> str:
    k = (key or "").strip()
    v = (value or "").strip()
    cat = (category or "general").strip() or "general"
    if not k and v:
        k = f"{cat}:{hashlib.sha256(v.encode('utf-8')).hexdigest()[:10]}"
    if not k or not v:
        return json.dumps({"error": "Non-empty fact/value is required."})

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    facts = _load_facts()

    updated = False
    for entry in facts:
        if entry.get("key", "").lower() == k.lower() and entry.get("collection", "companion_memory") == collection_name:
            entry["key"] = k
            entry["value"] = v
            entry["fact"] = v
            entry["category"] = cat
            entry["updated_at"] = now_iso
            updated = True
            break
    if not updated:
        facts.append(
            {
                "key": k,
                "value": v,
                "fact": v,
                "category": cat,
                "collection": collection_name,
                "updated_at": now_iso,
            }
        )
    _save_facts(facts)

    chroma_indexed = False
    use_chroma = os.environ.get("COLIBRI_ENABLE_CHROMA", "0").strip().lower() in ("1", "true", "yes")
    collection = _get_chroma_collection(collection_name) if use_chroma else None
    if collection is not None:
        try:
            doc_id = hashlib.sha256(f"{collection_name}:{k.lower()}".encode("utf-8")).hexdigest()[:24]
            collection.upsert(
                ids=[doc_id],
                documents=[f"[{cat}] {k}: {v}"],
                metadatas=[
                    {
                        "key": k,
                        "value": v,
                        "category": cat,
                        "updated_at": now_iso,
                    }
                ],
            )
            chroma_indexed = True
        except Exception:  # noqa: BLE001
            chroma_indexed = False

    return json.dumps(
        {
            "status": "saved",
            "key": k,
            "value": v,
            "category": cat,
            "collection": collection_name,
            "memory_dir": _memory_dir(),
            "updated_at": now_iso,
            "vector_indexed": chroma_indexed,
        },
        ensure_ascii=False,
    )


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^\w\u0590-\u05FF]+", (text or "").lower()) if len(t) > 1]


def _search_memory_entries(
    query: str,
    collection_name: str = "companion_memory",
    k: int = 3,
) -> str:
    q = (query or "").strip()
    if not q:
        return json.dumps({"error": "Empty query"})
    top_k = max(1, min(int(k), 15))

    # Fast in-memory query cache keyed by (memory_dir, query_lower, collection, k, facts_mtime)
    fpath = _facts_file()
    try:
        mtime = os.path.getmtime(fpath) if os.path.isfile(fpath) else 0.0
    except OSError:
        mtime = 0.0
    cache_key = (f"{_memory_dir()}:{q.lower()}", collection_name, top_k, mtime)
    if cache_key in _QUERY_RESULT_CACHE:
        return _QUERY_RESULT_CACHE[cache_key]

    results: List[Dict[str, Any]] = []
    seen_keys = set()

    # 1. Fast keyword + token-overlap search over in-memory cached JSON vault (<0.2ms)
    facts = [
        f
        for f in _load_facts()
        if not collection_name or f.get("collection", "companion_memory") == collection_name
    ]
    q_lower = q.lower()
    q_tokens = set(_tokenize(q))

    scored: List[tuple[float, Dict[str, Any]]] = []
    for entry in facts:
        ekey = str(entry.get("key", ""))
        eval_str = str(entry.get("value", entry.get("fact", "")))
        ecat = str(entry.get("category", "general"))
        combined = f"{ecat} {ekey} {eval_str}".lower()
        score = 0.0
        if q_lower in combined:
            score += 3.0
        entry_tokens = set(_tokenize(combined))
        if q_tokens and entry_tokens:
            overlap = len(q_tokens & entry_tokens)
            score += overlap / float(len(q_tokens))
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    for score, entry in scored:
        if len(results) >= top_k:
            break
        ekey_str = str(entry.get("key", "")).lower()
        seen_keys.add(ekey_str)
        results.append(
            {
                "key": entry.get("key"),
                "value": entry.get("value", entry.get("fact")),
                "category": entry.get("category", "general"),
                "updated_at": entry.get("updated_at"),
                "score": round(score, 3),
                "source": "json_vault",
            }
        )

    # 2. Only invoke heavy ChromaDB semantic embedding search if keyword search didn't fill top_k
    # or if explicitly enabled via COLIBRI_ENABLE_CHROMA=1
    use_chroma = os.environ.get("COLIBRI_ENABLE_CHROMA", "0").strip().lower() in ("1", "true", "yes")
    if (len(results) < top_k and not scored) or use_chroma:
        collection = _get_chroma_collection(collection_name) if use_chroma else None
        if collection is not None:
            try:
                count = collection.count()
                if count > 0:
                    res = collection.query(query_texts=[q], n_results=min(top_k, count))
                    docs = (res.get("documents") or [[]])[0]
                    metas = (res.get("metadatas") or [[]])[0]
                    dists = (res.get("distances") or [[]])[0]
                    for idx, doc in enumerate(docs):
                        if len(results) >= top_k:
                            break
                        meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
                        key_str = str(meta.get("key") or doc).lower()
                        if key_str in seen_keys:
                            continue
                        seen_keys.add(key_str)
                        results.append(
                            {
                                "key": meta.get("key"),
                                "value": meta.get("value") or doc,
                                "category": meta.get("category", "general"),
                                "updated_at": meta.get("updated_at"),
                                "distance": dists[idx] if idx < len(dists) else None,
                                "source": "chromadb",
                            }
                        )
            except Exception:  # noqa: BLE001
                pass

    if not results and facts:
        for entry in reversed(facts[-top_k:]):
            results.append(
                {
                    "key": entry.get("key"),
                    "value": entry.get("value", entry.get("fact")),
                    "category": entry.get("category", "general"),
                    "updated_at": entry.get("updated_at"),
                    "source": "recent_vault",
                }
            )

    payload_str = json.dumps(
        {
            "query": q,
            "collection": collection_name,
            "memory_dir": _memory_dir(),
            "matches": results[:top_k],
        },
        ensure_ascii=False,
    )
    if len(_QUERY_RESULT_CACHE) > 256:
        _QUERY_RESULT_CACHE.clear()
    _QUERY_RESULT_CACHE[cache_key] = payload_str
    return payload_str


@_tool_decorator
def query_memory(query: str, k: int = 3) -> str:
    """Retrieve relevant personal facts, habits, favorite songs, and preferences from `data/memory_db/`."""
    return _search_memory_entries(query=query, collection_name="companion_memory", k=k)


@_tool_decorator
def save_fact(fact: str, category: str = "general") -> str:
    """Save a new user preference, habit, favorite song, special date, or note dynamically into `data/memory_db/`."""
    cleaned_fact = (fact or "").strip()
    if not cleaned_fact:
        return json.dumps({"error": "Empty fact cannot be saved."})
    # Derive a concise human-readable key from category + first words of the fact
    words = re.sub(r"\s+", "_", cleaned_fact[:40].strip()).strip("_")
    key = f"{(category or 'general').strip()}:{words}"
    return _store_fact_entry(
        key=key,
        value=cleaned_fact,
        category=category or "general",
        collection_name="companion_memory",
    )


@_tool_decorator
def save_conversation_fact(
    key: str,
    value: str,
    collection_name: str = "companion_memory",
) -> str:
    """Save a keyed personal preference, favorite song, special date, or fact into private local memory."""
    return _store_fact_entry(
        key=key,
        value=value,
        category=key,
        collection_name=collection_name,
    )


@_tool_decorator
def query_local_knowledge(
    query: str,
    collection_name: str = "companion_memory",
    k: int = 3,
) -> str:
    """Search private local memory for personal facts, favorite songs, dates, or notes matching a query."""
    return _search_memory_entries(query=query, collection_name=collection_name, k=k)


@_tool_decorator
def list_saved_facts(category: str = "") -> str:
    """List all personal facts, preferences, and memories saved in the local companion vault."""
    facts = _load_facts()
    cat = (category or "").strip().lower()
    if cat:
        facts = [f for f in facts if str(f.get("category", "")).lower() == cat]
    return json.dumps(
        {
            "memory_dir": _memory_dir(),
            "count": len(facts),
            "facts": facts,
        },
        ensure_ascii=False,
    )


RAG_TOOLS: List[Any] = [
    query_memory,
    save_fact,
    save_conversation_fact,
    query_local_knowledge,
    list_saved_facts,
]

RAG_TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "query_memory": query_memory,
    "save_fact": save_fact,
    "save_conversation_fact": save_conversation_fact,
    "query_local_knowledge": query_local_knowledge,
    "list_saved_facts": list_saved_facts,
}

RAG_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_memory",
            "description": "Retrieve relevant personal facts, habits, favorite songs, and preferences from local persistent memory (`data/memory_db/`).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing the fact, preference, or memory needed.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of top matching memories to return (default 3).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_fact",
            "description": "Write a new personal preference, habit, favorite song, or note dynamically into local persistent memory (`data/memory_db/`).",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The personal fact, preference, or note to remember.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category label such as 'music', 'preferences', 'dates', 'family', or 'general'.",
                        "default": "general",
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_conversation_fact",
            "description": "Save a keyed personal fact, favorite song, preference, or special date into private local memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short topic or key name (e.g. 'favorite_song', 'anniversary_date', 'coffee_order').",
                    },
                    "value": {
                        "type": "string",
                        "description": "Detailed fact or preference value to remember.",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Memory collection name (default 'companion_memory').",
                        "default": "companion_memory",
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_local_knowledge",
            "description": "Search private local memory (ChromaDB + JSON vault) for personal facts, preferences, or notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing the memory or fact needed.",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Memory collection name (default 'companion_memory').",
                        "default": "companion_memory",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of top matching memories to return (default 3).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_saved_facts",
            "description": "List all saved personal facts and preferences stored in the companion's local memory vault.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter.",
                        "default": "",
                    }
                },
            },
        },
    },
]
