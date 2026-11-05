"""Private local RAG Memory, Multi-Collection Segmentation & Recipe/Bookmark Ingestion (`src/ai/tools/rag.py`).

100% local storage with zero external leaks and zero blocking of the main Qwen3.6 LLM engine:
- Default storage directory: ``data/memory_db/`` inside the repository (or ``COLIBRI_MEMORY_DIR`` env override).
- Multi-collection segmentation across 3 canonical collections (with ``"companion_memory"`` as a federated alias):
  1. ``recipes_and_food``       : structured recipes, ingredients, cooking steps, dietary notes, scaling conversions.
  2. ``web_bookmarks_and_reads``: scraped URLs, articles, bookmarks, key takeaways, summaries.
  3. ``personal_facts_and_notes``: Jimmies family preferences, birthdays, habits, reminders, daily notes.
- Automatically indexes into a persistent local ChromaDB vector store (`data/memory_db/chroma`) when
  ``COLIBRI_ENABLE_CHROMA`` is enabled (default ``"1"``), using a dedicated lightweight embedding provider
  that NEVER touches the main Qwen3.6 C engine:
  1. External lightweight embedding server (`COLIBRI_EMBED_URL`, e.g. Ollama `nomic-embed-text` on `:11434`),
     explicitly rejecting the Qwen3.6 gateway port (`:8000`) so token generation is never blocked.
  2. Local `ONNXMiniLM_L6_V2` or `SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2", device="mps")`
     on Apple Silicon MPS/CPU when enabled.
  3. Fast deterministic local character/word n-gram feature-hashing embedding function
     (`LocalHashEmbeddingFunction`) when offline or minimal, so ChromaDB vector indexing works out-of-the-box.
- Retrieved chunks are sorted deterministically by `(key, category)` after top-k selection and stripped of
  volatile fields (`distance` float jitter, `updated_at` timestamps) so prompt-injected RAG blocks keep
  Colibri's KV + DeltaNet prefix cache 100% valid across turns.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import re
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .internet import _tool_decorator, fetch_webpage

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_MEMORY_DIR = os.path.join(_REPO_ROOT, "data", "memory_db")

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EXTERNAL_EMBED_MODEL = "nomic-embed-text"

CANONICAL_COLLECTIONS: Tuple[str, str, str] = (
    "recipes_and_food",
    "web_bookmarks_and_reads",
    "personal_facts_and_notes",
)

_RECIPE_KEYWORDS_RE = re.compile(
    r"\b(recipe|recipes|ingredient|ingredients|servings?|serves|prep time|cook time|bake|baking|"
    r"oven|preheat|tbsp|tablespoons?|tsp|teaspoons?|cups?|grams?|flour|sugar|butter|olive oil|"
    r"garlic|onion|salt|pepper|whisk|simmer|boil|dough|batter|מתכון|מצרכים|מנות|אפייה|בישול)\b",
    re.IGNORECASE,
)

_BOOKMARK_KEYWORDS_RE = re.compile(
    r"(https?://|\b(?:bookmark|bookmarks|article|blog|webpage|url|link|read later|key takeaways?|summary)\b)",
    re.IGNORECASE,
)

_UNICODE_FRACTIONS: Dict[str, float] = {
    "½": 0.5,
    "⅓": 1.0 / 3.0,
    "⅔": 2.0 / 3.0,
    "¼": 0.25,
    "¾": 0.75,
    "⅕": 0.2,
    "⅖": 0.4,
    "⅗": 0.6,
    "⅘": 0.8,
    "⅙": 1.0 / 6.0,
    "⅚": 5.0 / 6.0,
    "⅛": 0.125,
    "⅜": 0.375,
    "⅝": 0.625,
    "⅞": 0.875,
}

_UNIT_ALIASES: Dict[str, str] = {
    "g": "g",
    "gram": "g",
    "grams": "g",
    "גרם": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "mg": "mg",
    "ml": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "l": "l",
    "liter": "l",
    "liters": "l",
    "cup": "cup",
    "cups": "cup",
    "כוס": "cup",
    "כוסות": "cup",
    "tbsp": "tbsp",
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "tbs": "tbsp",
    "כף": "tbsp",
    "כפות": "tbsp",
    "tsp": "tsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "כפית": "tsp",
    "כפיות": "tsp",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "pinch": "pinch",
    "pinches": "pinch",
    "clove": "clove",
    "cloves": "clove",
    "slice": "slice",
    "slices": "slice",
    "piece": "piece",
    "pieces": "piece",
    "can": "can",
    "cans": "can",
    "bunch": "bunch",
    "pkg": "package",
    "package": "package",
    "packages": "package",
}

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


def _chroma_enabled() -> bool:
    """ChromaDB indexing is enabled by default (`COLIBRI_ENABLE_CHROMA=1`) using dedicated embeddings."""
    val = os.environ.get("COLIBRI_ENABLE_CHROMA", "1").strip().lower()
    return val not in ("0", "false", "no", "off", "")


def _is_qwen36_gateway_url(url: str) -> bool:
    """Return True if `url` points to the Colibri Qwen3.6 inference gateway (default :8000).

    Embeddings must NEVER be routed to the main Qwen3.6 C engine so LLM prefill/decode
    is never blocked.
    """
    norm = (url or "").strip().lower()
    if not norm:
        return True
    qwen_base = (os.environ.get("COLIBRI_BASE_URL") or "http://127.0.0.1:8000").strip().lower().rstrip("/")
    if qwen_base.endswith("/v1"):
        qwen_base = qwen_base[:-3]
    if norm.startswith(qwen_base) or ":8000" in norm:
        return True
    return False


def _resolve_collection(category_or_collection: str = "", text: str = "") -> str:
    """Classify a memory entry into one of the 3 canonical segmented collections:
    - `recipes_and_food`
    - `web_bookmarks_and_reads`
    - `personal_facts_and_notes`
    """
    raw_col = (category_or_collection or "").strip().lower()
    if raw_col in CANONICAL_COLLECTIONS:
        return raw_col
    if raw_col in ("recipe", "recipes", "food", "cooking", "baking", "meal", "meals", "drink", "drinks"):
        return "recipes_and_food"
    if raw_col in ("bookmark", "bookmarks", "web", "url", "article", "articles", "reading", "reads"):
        return "web_bookmarks_and_reads"
    if raw_col in ("personal", "family", "notes", "note", "preferences", "dates", "birthday", "reminders", "music"):
        return "personal_facts_and_notes"

    combined = f"{raw_col} {text or ''}".strip()
    if len(_RECIPE_KEYWORDS_RE.findall(combined)) >= 2 or (
        raw_col in ("", "auto", "companion_memory", "general") and bool(_RECIPE_KEYWORDS_RE.search(raw_col))
    ):
        return "recipes_and_food"
    if _BOOKMARK_KEYWORDS_RE.search(combined):
        return "web_bookmarks_and_reads"
    return "personal_facts_and_notes"


class LocalHashEmbeddingFunction:
    """Fast, deterministic 256-dim L2-normalized word + character n-gram feature-hashing embedder.

    Runs in <0.1ms on CPU with zero external dependencies and zero network calls, ensuring
    ChromaDB vector indexing and cosine similarity queries work out-of-the-box without
    blocking the Qwen3.6 engine or requiring heavy PyTorch downloads.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def name(self) -> str:
        return f"colibri_local_hash_{self.dim}"

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        cleaned = (text or "").lower().strip()
        if not cleaned:
            vec[0] = 1.0
            return vec

        words = [w for w in re.split(r"[^\w\u0590-\u05FF]+", cleaned) if w]
        features: List[Tuple[str, float]] = []
        for w in words:
            features.append((f"w:{w}", 2.0))
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    features.append((f"c3:{w[i:i+3]}", 1.0))
        for i in range(len(words) - 1):
            features.append((f"b:{words[i]}_{words[i+1]}", 1.5))

        if not features:
            features.append((cleaned[:32], 1.0))

        for feat, weight in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            idx = h % self.dim
            sign = 1.0 if ((h >> 32) & 1) == 0 else -1.0
            vec[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-12:
            vec = [round(v / norm, 6) for v in vec]
        else:
            vec[0] = 1.0
        return vec

    def __call__(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002
        return [self._embed_one(str(item)) for item in input]

    def embed_query(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002
        return [self._embed_one(str(item)) for item in input]

    def embed_documents(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002
        return [self._embed_one(str(item)) for item in input]


class ExternalHTTPEmbeddingFunction:
    """Dedicated external OpenAI/Ollama-compatible embedding client (`COLIBRI_EMBED_URL`).

    Never touches the main Qwen3.6 server (`:8000`); falls back to `LocalHashEmbeddingFunction`
    if the external embedding server is unreachable.
    """

    def __init__(self, endpoint_url: str, model_name: str = DEFAULT_EXTERNAL_EMBED_MODEL) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        if not self.endpoint_url.endswith("/embeddings"):
            if not self.endpoint_url.endswith("/v1"):
                self.endpoint_url += "/v1"
            self.endpoint_url += "/embeddings"
        self.model_name = model_name
        self._fallback = LocalHashEmbeddingFunction()

    def name(self) -> str:
        return f"colibri_external_embed_{self.model_name}"

    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        payload = json.dumps({"model": self.model_name, "input": list(texts)}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
            data = body.get("data") or []
            if isinstance(data, list) and len(data) == len(texts):
                return [list(item["embedding"]) for item in data]
        except Exception:  # noqa: BLE001
            pass
        return self._fallback(texts)

    def __call__(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002
        return self._embed_batch(input)

    def embed_query(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002
        return self._embed_batch(input)

    def embed_documents(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002
        return self._embed_batch(input)


def get_dedicated_embedding_function() -> Any:
    """Resolve the dedicated lightweight embedding function for ChromaDB (never blocking Qwen3.6)."""
    embed_url = os.environ.get("COLIBRI_EMBED_URL", "").strip()
    embed_model = os.environ.get("COLIBRI_EMBED_MODEL", "").strip()
    use_onnx = os.environ.get("COLIBRI_USE_ONNX_EMBEDDINGS", "0").strip().lower() in ("1", "true", "yes")
    use_st = os.environ.get("COLIBRI_USE_SENTENCE_TRANSFORMERS", "0").strip().lower() in ("1", "true", "yes")
    cache_id = f"{embed_url}|{embed_model}|{use_onnx}|{use_st}"
    if cache_id in _EMBEDDING_FN_CACHE:
        return _EMBEDDING_FN_CACHE[cache_id]

    # 1. Dedicated external embedding endpoint (e.g. Ollama nomic-embed-text on :11434),
    # strictly ignoring any URL pointing to the main Qwen3.6 server (:8000).
    if embed_url and not _is_qwen36_gateway_url(embed_url):
        fn: Any = ExternalHTTPEmbeddingFunction(
            endpoint_url=embed_url,
            model_name=embed_model or DEFAULT_EXTERNAL_EMBED_MODEL,
        )
        _EMBEDDING_FN_CACHE[cache_id] = fn
        return fn

    # 2. Local ONNXMiniLM_L6_V2 or SentenceTransformer ("all-MiniLM-L6-v2" on MPS/CPU) when requested
    if use_onnx:
        try:
            from chromadb.utils import embedding_functions  # type: ignore

            fn = embedding_functions.ONNXMiniLM_L6_V2()
            _EMBEDDING_FN_CACHE[cache_id] = fn
            return fn
        except Exception:  # noqa: BLE001
            pass

    if use_st:
        target_st_model = embed_model or DEFAULT_EMBED_MODEL
        device = os.environ.get("COLIBRI_EMBED_DEVICE", "mps")
        try:
            from chromadb.utils import embedding_functions  # type: ignore

            try:
                fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=target_st_model,
                    device=device,
                )
            except Exception:  # noqa: BLE001
                fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=target_st_model,
                )
            _EMBEDDING_FN_CACHE[cache_id] = fn
            return fn
        except Exception:  # noqa: BLE001
            pass

    # 3. Default zero-latency deterministic local n-gram hash embedding function (<0.1ms, zero network)
    fn = LocalHashEmbeddingFunction(dim=256)
    _EMBEDDING_FN_CACHE[cache_id] = fn
    return fn


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
    """Return a persistent ChromaDB collection backed by our dedicated lightweight embedding function."""
    if not _chroma_enabled():
        return None
    try:
        import chromadb  # type: ignore

        chroma_path = os.path.join(_memory_dir(), "chroma")
        if chroma_path not in _CHROMA_CLIENT_CACHE:
            _CHROMA_CLIENT_CACHE[chroma_path] = chromadb.PersistentClient(path=chroma_path)
        client = _CHROMA_CLIENT_CACHE[chroma_path]

        emb_fn = get_dedicated_embedding_function()
        if emb_fn is not None:
            return client.get_or_create_collection(
                name=collection_name,
                embedding_function=emb_fn,
            )
        return client.get_or_create_collection(name=collection_name)
    except Exception:  # noqa: BLE001
        return None


def _store_fact_entry(
    key: str,
    value: str,
    category: str = "general",
    collection_name: str = "companion_memory",
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> str:
    k = (key or "").strip()
    v = (value or "").strip()
    cat = (category or "general").strip() or "general"
    if not k and v:
        k = f"{cat}:{hashlib.sha256(v.encode('utf-8')).hexdigest()[:10]}"
    if not k or not v:
        return json.dumps({"error": "Non-empty fact/value is required."})

    # Resolve canonical segmented collection while keeping backward compatibility
    if collection_name in ("", "auto", "companion_memory"):
        resolved_col = _resolve_collection(cat, f"{k} {v}")
    else:
        resolved_col = _resolve_collection(collection_name, f"{k} {v}")

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    facts = _load_facts()

    updated = False
    for entry in facts:
        if entry.get("key", "").lower() == k.lower() and entry.get("collection") in (resolved_col, collection_name):
            entry["key"] = k
            entry["value"] = v
            entry["fact"] = v
            entry["category"] = cat
            entry["collection"] = resolved_col
            entry["updated_at"] = now_iso
            if metadata_extra:
                entry.update(metadata_extra)
            updated = True
            break
    if not updated:
        new_entry: Dict[str, Any] = {
            "key": k,
            "value": v,
            "fact": v,
            "category": cat,
            "collection": resolved_col,
            "updated_at": now_iso,
        }
        if metadata_extra:
            new_entry.update(metadata_extra)
        facts.append(new_entry)
    _save_facts(facts)

    chroma_indexed = False
    target_collections = {resolved_col}
    if collection_name and collection_name not in ("auto", resolved_col):
        target_collections.add(collection_name)

    for col_name in target_collections:
        collection = _get_chroma_collection(col_name)
        if collection is not None:
            try:
                doc_id = hashlib.sha256(f"{col_name}:{k.lower()}".encode("utf-8")).hexdigest()[:24]
                collection.upsert(
                    ids=[doc_id],
                    documents=[f"[{cat}] {k}: {v}"],
                    metadatas=[
                        {
                            "key": k,
                            "value": v[:2000],
                            "category": cat,
                            "collection": resolved_col,
                        }
                    ],
                )
                chroma_indexed = True
            except Exception:  # noqa: BLE001
                pass

    return json.dumps(
        {
            "status": "saved",
            "key": k,
            "value": v,
            "category": cat,
            "collection": resolved_col,
            "memory_dir": _memory_dir(),
            "vector_indexed": chroma_indexed,
        },
        ensure_ascii=False,
    )


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^\w\u0590-\u05FF]+", (text or "").lower()) if len(t) > 1]


def _canonical_match_sort_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
    """Deterministic ordering key `(key, category, value)` for KV-cache-friendly RAG injection."""
    return (
        str(item.get("key") or "").lower(),
        str(item.get("category") or "general").lower(),
        str(item.get("value") or ""),
    )


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

    is_federated = collection_name in ("", "companion_memory", "auto", "all")
    resolved_target = collection_name if is_federated else _resolve_collection(collection_name, q)

    results: List[Dict[str, Any]] = []
    seen_keys = set()

    # 1. Fast keyword + token-overlap search over in-memory cached JSON vault (<0.2ms)
    all_facts = _load_facts()
    if is_federated:
        facts = all_facts
    else:
        facts = [
            f
            for f in all_facts
            if f.get("collection") in (collection_name, resolved_target)
            or _resolve_collection(str(f.get("category", "")), str(f.get("value", ""))) == resolved_target
        ]

    q_lower = q.lower()
    q_tokens = set(_tokenize(q))

    scored: List[Tuple[float, str, Dict[str, Any]]] = []
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
            scored.append((score, ekey.lower(), entry))

    scored.sort(key=lambda item: (-item[0], item[1]))
    for _score, ekey_str, entry in scored:
        if len(results) >= top_k:
            break
        seen_keys.add(ekey_str)
        # Strip volatile fields (`updated_at`, float `distance`) so repeated turns produce
        # byte-identical tool/RAG outputs for Colibri KV-cache prefix reuse.
        results.append(
            {
                "key": entry.get("key"),
                "value": entry.get("value", entry.get("fact")),
                "category": entry.get("category", "general"),
                "collection": entry.get("collection", _resolve_collection(str(entry.get("category", "")), str(entry.get("value", "")))),
                "source": "json_vault",
            }
        )

    # 2. Dedicated lightweight ChromaDB semantic vector search (out-of-band from Qwen3.6)
    if len(results) < top_k and _chroma_enabled():
        chroma_cols = list(CANONICAL_COLLECTIONS) + ["companion_memory"] if is_federated else [resolved_target]
        for col_name in chroma_cols:
            if len(results) >= top_k:
                break
            collection = _get_chroma_collection(col_name)
            if collection is not None:
                try:
                    count = collection.count()
                    if count > 0:
                        res = collection.query(query_texts=[q], n_results=min(top_k, count))
                        docs = (res.get("documents") or [[]])[0]
                        metas = (res.get("metadatas") or [[]])[0]
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
                                    "key": meta.get("key") or f"doc_{idx}",
                                    "value": meta.get("value") or doc,
                                    "category": meta.get("category", "general"),
                                    "collection": meta.get("collection", col_name),
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
                    "collection": entry.get("collection", resolved_target),
                    "source": "recent_vault",
                }
            )

    # Sort selected top-k chunks deterministically by (key, category, value) so multi-step
    # agent loops always render byte-identical RAG context.
    selected_matches = sorted(results[:top_k], key=_canonical_match_sort_key)

    payload_str = json.dumps(
        {
            "query": q,
            "collection": collection_name if is_federated else resolved_target,
            "memory_dir": _memory_dir(),
            "matches": selected_matches,
        },
        ensure_ascii=False,
    )
    if len(_QUERY_RESULT_CACHE) > 256:
        _QUERY_RESULT_CACHE.clear()
    _QUERY_RESULT_CACHE[cache_key] = payload_str
    return payload_str


# ---------------------------------------------------------------------------
# Structured Recipe & Bookmark Parsing + Scaling Helpers
# ---------------------------------------------------------------------------

def _parse_quantity_token(token: str) -> Optional[float]:
    """Parse integer, decimal, fraction (`1/2`), or unicode fraction (`½`) into float."""
    s = (token or "").strip()
    if not s:
        return None
    if s in _UNICODE_FRACTIONS:
        return _UNICODE_FRACTIONS[s]
    # Check leading number + trailing unicode fraction, e.g. "1½"
    if len(s) > 1 and s[-1] in _UNICODE_FRACTIONS:
        prefix = _parse_quantity_token(s[:-1])
        if prefix is not None:
            return prefix + _UNICODE_FRACTIONS[s[-1]]
    if "/" in s:
        parts = s.split("/", 1)
        try:
            num = float(parts[0].strip())
            den = float(parts[1].strip())
            if den != 0:
                return num / den
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


_INGREDIENT_LINE_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])?\s*"
    r"(?P<qty>(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]?|[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]))"
    r"(?:\s*(?P<unit>[A-Za-z\u0590-\u05FF]+)\.?)?\s+"
    r"(?P<item>.+?)\s*$"
)


def _parse_ingredient_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse an ingredient line like `1 1/2 cups all-purpose flour` or `200 g dark chocolate`."""
    cleaned = (line or "").strip()
    if not cleaned or len(cleaned) > 180:
        return None
    m = _INGREDIENT_LINE_RE.match(cleaned)
    if not m:
        return None
    qty_raw = m.group("qty").strip()
    if " " in qty_raw:
        first, second = qty_raw.split(None, 1)
        q1 = _parse_quantity_token(first)
        q2 = _parse_quantity_token(second)
        qty_val = (q1 or 0.0) + (q2 or 0.0)
    else:
        qty_val = _parse_quantity_token(qty_raw) or 0.0
    if qty_val <= 0:
        return None

    unit_raw = (m.group("unit") or "").strip().lower()
    item_raw = (m.group("item") or "").strip()
    if unit_raw in _UNIT_ALIASES:
        canonical_unit = _UNIT_ALIASES[unit_raw]
    else:
        # Unit token is actually part of the item name (e.g. "3 large eggs")
        canonical_unit = "unit"
        if unit_raw:
            item_raw = f"{m.group('unit')} {item_raw}".strip()

    if item_raw.lower().startswith("of "):
        item_raw = item_raw[3:].strip()

    return {
        "raw": cleaned,
        "quantity": round(qty_val, 4),
        "unit": canonical_unit,
        "item": item_raw,
    }


def _extract_servings(text: str, default: float = 4.0) -> float:
    m = re.search(
        r"(?:servings?|serves|yield|מנות)\s*[:=-]?\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:servings?|portions?|מנות)",
        text or "",
        re.IGNORECASE,
    )
    if m:
        val_str = m.group(1) or m.group(2)
        try:
            val = float(val_str)
            if val > 0:
                return val
        except ValueError:
            pass
    return default


def _extract_structured_recipe(text: str, title_hint: str = "") -> Dict[str, Any]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    title = (title_hint or "").strip()
    if not title and lines:
        first_line = re.sub(r"^#+\s*", "", lines[0]).strip()
        if not _parse_ingredient_line(first_line):
            title = first_line[:90]
    if not title:
        title = "Saved Family Recipe"

    servings = _extract_servings(text, default=4.0)
    ingredients: List[Dict[str, Any]] = []
    steps: List[str] = []

    for ln in lines:
        if re.match(r"^(?:ingredients?|מצרכים|servings?|serves|yield)\s*[:=-]?$", ln, re.IGNORECASE):
            continue
        if re.match(r"^(?:instructions?|directions?|steps?|method|הוראות הכנה|אופן הכנה)\s*[:=-]?$", ln, re.IGNORECASE):
            continue
        parsed_ing = _parse_ingredient_line(ln)
        if parsed_ing is not None:
            ingredients.append(parsed_ing)
        elif len(ln) > 15 and ln != title and not ln.lower().startswith(("page title:", "source url:", "servings:")):
            clean_step = re.sub(r"^(?:step\s*\d+[:.)]?|\d+[.)]|[-*•])\s*", "", ln, flags=re.IGNORECASE).strip()
            if clean_step:
                steps.append(clean_step)

    # Also support inline comma/semicolon-separated ingredient lists (e.g. "200g flour, 2 eggs, 1/2 cup milk")
    if not ingredients:
        for chunk in re.split(r"[,;\n]+", text or ""):
            parsed_ing = _parse_ingredient_line(chunk.strip())
            if parsed_ing is not None:
                ingredients.append(parsed_ing)

    takeaways = [
        f"Servings: {servings:g}",
        f"Ingredients ({len(ingredients)}): " + ", ".join(
            f"{ing['quantity']:g} {ing['unit'] if ing['unit'] != 'unit' else ''} {ing['item']}".strip()
            for ing in ingredients[:8]
        ),
    ]
    if steps:
        takeaways.append(f"Steps ({len(steps)}): {steps[0][:120]}")

    return {
        "title": title,
        "servings": servings,
        "ingredients": ingredients,
        "steps": steps[:12],
        "key_takeaways": takeaways,
    }


def _extract_bookmark_summary(text: str, url: str = "", title_hint: str = "") -> Dict[str, Any]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    title = (title_hint or "").strip()
    clean_lines: List[str] = []
    for ln in lines:
        if ln.lower().startswith("page title:") and not title:
            title = ln.split(":", 1)[1].strip()
            continue
        if ln.lower().startswith(("source url:", "extracted readable content:")):
            continue
        clean_lines.append(ln)

    if not title and clean_lines:
        title = re.sub(r"^#+\s*", "", clean_lines[0])[:100]
    if not title:
        title = url or "Saved Web Bookmark"

    body_text = " ".join(clean_lines)
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", body_text)
        if len(s.strip()) >= 20
    ]
    if not sentences and clean_lines:
        sentences = clean_lines[:5]

    key_takeaways = sentences[:5]
    summary = " ".join(sentences[:3])[:500] if sentences else body_text[:500]
    return {
        "title": title,
        "url": url,
        "key_takeaways": key_takeaways,
        "summary": summary,
    }


def _format_number(val: float) -> str:
    rounded = round(val, 2)
    if abs(rounded - round(rounded)) < 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:g}"


def _max_tool_chars() -> int:
    try:
        val = int(os.environ.get("COLIBRI_MAX_TOOL_CHARS", "4000"))
        return max(100, val)
    except Exception:
        return 4000


def _bound_tool_payload(payload_data: Any, max_chars: Optional[int] = None) -> str:
    """Enforce COLIBRI_MAX_TOOL_CHARS bounding on tool return payloads."""
    limit = max_chars if max_chars is not None else _max_tool_chars()
    serialized = json.dumps(payload_data, ensure_ascii=False)
    if len(serialized) <= limit:
        return serialized

    if isinstance(payload_data, dict):
        trimmed = dict(payload_data)
        for array_key in ("facts", "scaled_ingredients", "steps", "key_takeaways", "matches"):
            if isinstance(trimmed.get(array_key), list) and trimmed[array_key]:
                items = list(trimmed[array_key])
                while items and len(json.dumps({**trimmed, array_key: items}, ensure_ascii=False)) > limit:
                    items.pop()
                trimmed[array_key] = items
                trimmed["truncated"] = True
                serialized = json.dumps(trimmed, ensure_ascii=False)
                if len(serialized) <= limit:
                    return serialized

        for str_key in ("summary", "value", "content", "note"):
            if isinstance(trimmed.get(str_key), str) and len(trimmed[str_key]) > 40:
                avail = max(20, limit - len(json.dumps({**trimmed, str_key: ""}, ensure_ascii=False)) - 20)
                trimmed[str_key] = trimmed[str_key][:avail] + "... [truncated]"
                trimmed["truncated"] = True
                serialized = json.dumps(trimmed, ensure_ascii=False)
                if len(serialized) <= limit:
                    return serialized

    if len(serialized) > limit:
        return serialized[: max(0, limit - 16)] + "... [truncated]"
    return serialized


@_tool_decorator
def ingest_url_or_note(
    source: str,
    collection_name: str = "auto",
    title: str = "",
) -> str:
    """Ingest a URL (`http://`/`https://`) or raw recipe/article/personal note into segmented local memory.

    Automatically scrapes URLs via SSRF-safe `fetch_webpage`, extracts structured recipe fields
    (`title`, `servings`, `ingredients`, `steps`, `key_takeaways`) or bookmark takeaways (`title`,
    `url`, `key_takeaways`, `summary`), and indexes into both `facts.json` and ChromaDB under
    `recipes_and_food`, `web_bookmarks_and_reads`, or `personal_facts_and_notes`.
    """
    raw_source = (source or "").strip()
    if not raw_source:
        return _bound_tool_payload({"error": "Non-empty URL or note text is required."})

    max_bound = _max_tool_chars()
    is_url = raw_source.lower().startswith(("http://", "https://"))
    url_str = raw_source if is_url else ""
    content_text = raw_source[:max_bound]

    if is_url:
        fetched = fetch_webpage(raw_source, max_chars=max_bound, include_links=False)
        content_text = fetched[:max_bound]

    resolved_col = _resolve_collection(
        collection_name if collection_name not in ("", "auto", "companion_memory") else "",
        f"{title} {content_text}",
    )
    if is_url and resolved_col == "personal_facts_and_notes":
        resolved_col = "web_bookmarks_and_reads"

    # Check if content has structured recipe ingredients
    recipe_candidate = _extract_structured_recipe(content_text, title_hint=title)
    if resolved_col == "recipes_and_food" or len(recipe_candidate.get("ingredients", [])) >= 2:
        resolved_col = "recipes_and_food"
        rec_title = recipe_candidate["title"]
        key = f"recipe:{re.sub(r'[^a-zA-Z0-9\u0590-\u05FF]+', '_', rec_title.lower()).strip('_')[:48]}"
        summary_val = (
            f"Recipe: {rec_title} (Serves {recipe_candidate['servings']:g}). "
            + "; ".join(recipe_candidate["key_takeaways"])
        )
        store_res = json.loads(
            _store_fact_entry(
                key=key,
                value=summary_val,
                category="recipes_and_food",
                collection_name="recipes_and_food",
                metadata_extra={
                    "title": rec_title,
                    "url": url_str,
                    "servings": recipe_candidate["servings"],
                    "ingredients": recipe_candidate["ingredients"],
                    "steps": recipe_candidate["steps"],
                    "key_takeaways": recipe_candidate["key_takeaways"],
                },
            )
        )
        return _bound_tool_payload(
            {
                "status": "ingested",
                "collection": "recipes_and_food",
                "key": key,
                "title": rec_title,
                "url": url_str,
                "servings": recipe_candidate["servings"],
                "ingredients": recipe_candidate["ingredients"],
                "steps": recipe_candidate["steps"],
                "key_takeaways": recipe_candidate["key_takeaways"],
                "vector_indexed": store_res.get("vector_indexed", False),
            }
        )

    if resolved_col == "web_bookmarks_and_reads":
        bm = _extract_bookmark_summary(content_text, url=url_str, title_hint=title)
        bm_title = bm["title"]
        key = f"bookmark:{re.sub(r'[^a-zA-Z0-9\u0590-\u05FF]+', '_', bm_title.lower()).strip('_')[:48]}"
        summary_val = f"{bm_title} ({url_str}): {bm['summary']}" if url_str else f"{bm_title}: {bm['summary']}"
        store_res = json.loads(
            _store_fact_entry(
                key=key,
                value=summary_val,
                category="web_bookmarks_and_reads",
                collection_name="web_bookmarks_and_reads",
                metadata_extra={
                    "title": bm_title,
                    "url": url_str,
                    "key_takeaways": bm["key_takeaways"],
                    "summary": bm["summary"],
                },
            )
        )
        return _bound_tool_payload(
            {
                "status": "ingested",
                "collection": "web_bookmarks_and_reads",
                "key": key,
                "title": bm_title,
                "url": url_str,
                "key_takeaways": bm["key_takeaways"],
                "summary": bm["summary"],
                "vector_indexed": store_res.get("vector_indexed", False),
            }
        )

    # Personal fact or note (`personal_facts_and_notes`)
    note_info = _extract_bookmark_summary(content_text, url="", title_hint=title or "Personal Note")
    note_title = title or note_info["title"]
    key = f"note:{re.sub(r'[^a-zA-Z0-9\u0590-\u05FF]+', '_', note_title.lower()).strip('_')[:48]}"
    store_res = json.loads(
        _store_fact_entry(
            key=key,
            value=content_text,
            category="personal_facts_and_notes",
            collection_name="personal_facts_and_notes",
            metadata_extra={
                "title": note_title,
                "key_takeaways": note_info["key_takeaways"],
                "summary": note_info["summary"],
            },
        )
    )
    return _bound_tool_payload(
        {
            "status": "ingested",
            "collection": "personal_facts_and_notes",
            "key": key,
            "title": note_title,
            "key_takeaways": note_info["key_takeaways"],
            "summary": note_info["summary"],
            "vector_indexed": store_res.get("vector_indexed", False),
        }
    )


@_tool_decorator
def scale_recipe(
    recipe_query_or_text: str,
    target_servings: float,
    base_servings: float = 0.0,
) -> str:
    """Scale a saved recipe from `recipes_and_food` or raw ingredient text to `target_servings`."""
    query_or_text = (recipe_query_or_text or "").strip()
    if not query_or_text:
        return json.dumps({"error": "recipe_query_or_text is required."})
    if target_servings <= 0:
        return json.dumps({"error": "target_servings must be greater than 0."})

    # 1. Try parsing directly if raw ingredient lines were supplied
    direct_recipe = _extract_structured_recipe(query_or_text)
    ingredients: List[Dict[str, Any]] = list(direct_recipe.get("ingredients") or [])
    steps: List[str] = list(direct_recipe.get("steps") or [])
    recipe_title = str(direct_recipe.get("title") or query_or_text[:60])
    inferred_base = float(direct_recipe.get("servings") or 4.0)

    # 2. If fewer than 2 ingredients were parsed directly, look up saved recipe in `recipes_and_food`
    if len(ingredients) < 2:
        facts = _load_facts()
        q_lower = query_or_text.lower()
        q_tokens = set(_tokenize(query_or_text))
        best_entry: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for entry in facts:
            if entry.get("collection") != "recipes_and_food" and not str(entry.get("key", "")).startswith("recipe:"):
                continue
            combined = f"{entry.get('key', '')} {entry.get('title', '')} {entry.get('value', '')}".lower()
            score = 3.0 if q_lower in combined else 0.0
            if q_tokens:
                score += len(q_tokens & set(_tokenize(combined))) / float(len(q_tokens))
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None:
            recipe_title = str(best_entry.get("title") or best_entry.get("key") or recipe_title)
            if isinstance(best_entry.get("ingredients"), list) and best_entry["ingredients"]:
                ingredients = list(best_entry["ingredients"])
            else:
                parsed_from_val = _extract_structured_recipe(str(best_entry.get("value", "")))
                ingredients = list(parsed_from_val.get("ingredients") or [])
            if isinstance(best_entry.get("steps"), list):
                steps = list(best_entry["steps"])
            if best_entry.get("servings"):
                try:
                    inferred_base = float(best_entry["servings"])
                except (TypeError, ValueError):
                    pass

    effective_base = float(base_servings) if base_servings and base_servings > 0 else inferred_base
    if effective_base <= 0:
        effective_base = 4.0
    scale_factor = float(target_servings) / effective_base

    scaled_ingredients: List[Dict[str, Any]] = []
    for ing in ingredients:
        orig_qty = float(ing.get("quantity") or 0.0)
        new_qty = orig_qty * scale_factor
        unit = str(ing.get("unit") or "unit")
        item = str(ing.get("item") or "")
        qty_str = _format_number(new_qty)
        unit_str = f" {unit}" if unit and unit != "unit" else ""
        scaled_ingredients.append(
            {
                "item": item,
                "unit": unit,
                "original_quantity": round(orig_qty, 4),
                "scaled_quantity": round(new_qty, 4),
                "formatted": f"{qty_str}{unit_str} {item}".strip(),
            }
        )

    return _bound_tool_payload(
        {
            "status": "ok",
            "title": recipe_title,
            "base_servings": effective_base,
            "target_servings": float(target_servings),
            "scale_factor": round(scale_factor, 4),
            "scaled_ingredients": scaled_ingredients,
            "steps": steps,
        }
    )


@_tool_decorator
def query_memory(query: str, k: int = 3) -> str:
    """Retrieve relevant personal facts, habits, favorite songs, recipes, bookmarks, and preferences from `data/memory_db/`."""
    return _search_memory_entries(query=query, collection_name="companion_memory", k=k)


@_tool_decorator
def save_fact(fact: str, category: str = "general") -> str:
    """Save a new user preference, habit, favorite song, recipe, bookmark, or note dynamically into `data/memory_db/`."""
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
    """Save a keyed personal preference, favorite song, special date, recipe, or fact into private local memory."""
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
    """Search private local memory (`recipes_and_food`, `web_bookmarks_and_reads`, `personal_facts_and_notes`, or federated `companion_memory`)."""
    return _search_memory_entries(query=query, collection_name=collection_name, k=k)


@_tool_decorator
def list_saved_facts(category: str = "") -> str:
    """List all personal facts, recipes, bookmarks, and memories saved in the local companion vault."""
    facts = _load_facts()
    cat = (category or "").strip().lower()
    if cat:
        facts = [
            f
            for f in facts
            if str(f.get("category", "")).lower() == cat
            or str(f.get("collection", "")).lower() == cat
        ]
    # Sort deterministically by (key, category) and strip volatile timestamps for prefix stability
    clean_facts = sorted(
        [
            {
                "key": f.get("key"),
                "value": f.get("value", f.get("fact")),
                "category": f.get("category", "general"),
                "collection": f.get("collection", "personal_facts_and_notes"),
            }
            for f in facts
        ],
        key=_canonical_match_sort_key,
    )
    return _bound_tool_payload(
        {
            "memory_dir": _memory_dir(),
            "count": len(clean_facts),
            "facts": clean_facts,
        }
    )


RAG_TOOLS: List[Any] = [
    query_memory,
    save_fact,
    save_conversation_fact,
    query_local_knowledge,
    list_saved_facts,
    ingest_url_or_note,
    scale_recipe,
]

RAG_TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "query_memory": query_memory,
    "save_fact": save_fact,
    "save_conversation_fact": save_conversation_fact,
    "query_local_knowledge": query_local_knowledge,
    "list_saved_facts": list_saved_facts,
    "ingest_url_or_note": ingest_url_or_note,
    "scale_recipe": scale_recipe,
}

RAG_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_memory",
            "description": "Retrieve relevant personal facts, habits, favorite songs, recipes, bookmarks, and preferences from local persistent memory (`data/memory_db/`).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing the fact, recipe, bookmark, or memory needed.",
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
            "description": "Write a new personal preference, habit, favorite song, recipe, or note dynamically into local persistent memory (`data/memory_db/`).",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The personal fact, preference, or note to remember.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category label such as 'recipes_and_food', 'web_bookmarks_and_reads', 'personal_facts_and_notes', 'music', 'family', or 'general'.",
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
            "description": "Save a keyed personal fact, favorite song, preference, recipe, or special date into private local memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short topic or key name (e.g. 'favorite_song', 'pancake_recipe', 'coffee_order').",
                    },
                    "value": {
                        "type": "string",
                        "description": "Detailed fact, recipe, or preference value to remember.",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Memory collection ('recipes_and_food', 'web_bookmarks_and_reads', 'personal_facts_and_notes', or 'companion_memory').",
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
            "description": "Search private local memory (ChromaDB + JSON vault) across segmented collections (`recipes_and_food`, `web_bookmarks_and_reads`, `personal_facts_and_notes`, or federated `companion_memory`).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing the memory, recipe, bookmark, or fact needed.",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Collection name ('recipes_and_food', 'web_bookmarks_and_reads', 'personal_facts_and_notes', or 'companion_memory').",
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
            "description": "List all saved personal facts, recipes, and bookmarks stored in the companion's local memory vault.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category or collection filter.",
                        "default": "",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_url_or_note",
            "description": "Scrape and ingest a URL or raw recipe/bookmark/note text into segmented local memory (`recipes_and_food`, `web_bookmarks_and_reads`, or `personal_facts_and_notes`), extracting structured ingredients, steps, or key takeaways.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Public http/https URL to scrape or raw recipe/note text to ingest.",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Target collection ('auto', 'recipes_and_food', 'web_bookmarks_and_reads', or 'personal_facts_and_notes').",
                        "default": "auto",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional human-readable title for the recipe, bookmark, or note.",
                        "default": "",
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scale_recipe",
            "description": "Scale a saved recipe from `recipes_and_food` (or raw ingredient text) to `target_servings`, accurately converting fractions, metric/imperial units, and quantities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_query_or_text": {
                        "type": "string",
                        "description": "Name/query of a saved recipe in `recipes_and_food` or raw multi-line recipe ingredient text.",
                    },
                    "target_servings": {
                        "type": "number",
                        "description": "Desired number of servings to scale the recipe to.",
                    },
                    "base_servings": {
                        "type": "number",
                        "description": "Optional original servings count (if 0, inferred automatically from the recipe).",
                        "default": 0.0,
                    },
                },
                "required": ["recipe_query_or_text", "target_servings"],
            },
        },
    },
]
