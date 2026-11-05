"""Configuration and model auto-detection for the Colibri AI Agent (`src/ai`).

Reads environment variables with sensible defaults targeting a local `coli serve`
gateway running `qwen36-35b` at `http://127.0.0.1:8000/v1`.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
MLX_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_API_KEY = "colibri"
DEFAULT_MODEL = "qwen36-35b"
DEFAULT_CONTEXT_LENGTH = 65536


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or not val.strip():
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _query_models_endpoint(
    base_url: str,
    api_key: str,
    timeout: float = 1.5,
) -> Optional[Tuple[str, int]]:
    """Probe ``GET /v1/models`` at ``base_url`` and return ``(model_id, context_length)`` if reachable."""
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    models_url = f"{url}/models"

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(models_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        data = payload.get("data")
        if isinstance(data, list) and data:
            # Prefer Qwen3-30B-A3B / Qwen36 / Qwen2.5 models if multiple are listed (e.g. Ollama)
            preferred = ("qwen36", "qwen3:30b-a3b", "qwen3", "qwen2.5:14b", "qwen2.5")
            chosen = data[0] if isinstance(data[0], dict) else {}
            for item in data:
                if isinstance(item, dict):
                    mid = str(item.get("id") or "").lower()
                    if any(p in mid for p in preferred):
                        chosen = item
                        break
            model_id = str(chosen.get("id") or DEFAULT_MODEL)
            ctx = (
                chosen.get("context_length")
                or chosen.get("max_model_len")
                or DEFAULT_CONTEXT_LENGTH
            )
            return model_id, int(ctx)
    except Exception:  # noqa: BLE001
        return None
    return None


def detect_colibri_model(
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    timeout: float = 3.0,
) -> Tuple[str, int]:
    """Query ``GET /v1/models`` on the Colibri gateway and return ``(model_id, context_length)``.

    Falls back cleanly to ``("qwen36-35b", 65536)`` if the server is unreachable or
    does not advertise a model list.
    """
    probed = _query_models_endpoint(base_url, api_key, timeout=timeout)
    if probed is not None:
        return probed
    return DEFAULT_MODEL, DEFAULT_CONTEXT_LENGTH


def detect_best_local_backend(
    preferred_url: Optional[str] = None,
    api_key: str = DEFAULT_API_KEY,
    timeout: float = 0.8,
) -> Tuple[str, str, int]:
    """Auto-detect the fastest reachable local OpenAI-compatible inference engine.

    Checks ``preferred_url`` (or ``COLIBRI_BASE_URL`` / ``OPENAI_BASE_URL``), then probes:
    1. Colibri C gateway (`http://127.0.0.1:8000/v1`)
    2. Apple Metal Ollama gateway (`http://127.0.0.1:11434/v1`)
    3. Apple Metal MLX gateway (`http://127.0.0.1:8080/v1`)
    """
    candidates = []
    if preferred_url:
        candidates.append(preferred_url.rstrip("/"))
    for fallback in (DEFAULT_BASE_URL, OLLAMA_BASE_URL, MLX_BASE_URL):
        if fallback not in candidates:
            candidates.append(fallback)

    for candidate in candidates:
        res = _query_models_endpoint(candidate, api_key=api_key, timeout=timeout)
        if res is not None:
            norm = candidate if candidate.endswith("/v1") else f"{candidate}/v1"
            return norm, res[0], res[1]

    base = (preferred_url or DEFAULT_BASE_URL).rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base, DEFAULT_MODEL, DEFAULT_CONTEXT_LENGTH


@dataclass
class AgentConfig:
    """Runtime configuration for Colibri LangGraph / ReAct agents."""

    base_url: str = field(
        default_factory=lambda: (
            os.environ.get("COLIBRI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
    )
    api_key: str = field(
        default_factory=lambda: (
            os.environ.get("COLIBRI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or DEFAULT_API_KEY
        )
    )
    model: str = field(default_factory=lambda: os.environ.get("COLIBRI_MODEL", ""))
    context_length: int = field(
        default_factory=lambda: _env_int("COLIBRI_CONTEXT_LENGTH", 0)
    )
    cache_slot: int = field(
        default_factory=lambda: _env_int("COLIBRI_CACHE_SLOT", 0)
    )
    research_cache_slot: int = field(
        default_factory=lambda: _env_int("COLIBRI_RESEARCH_CACHE_SLOT", 1)
    )
    enable_thinking: bool = field(
        default_factory=lambda: _env_bool("COLIBRI_ENABLE_THINKING", False)
    )
    preserve_thinking: Optional[bool] = field(default=None)
    compact: str = field(
        default_factory=lambda: os.environ.get("COLIBRI_COMPACT", "auto")
    )
    max_tool_chars: int = field(
        default_factory=lambda: _env_int("COLIBRI_MAX_TOOL_CHARS", 2500)
    )
    max_iterations: int = field(
        default_factory=lambda: _env_int("COLIBRI_MAX_ITERATIONS", 3)
    )
    temperature: float = field(
        default_factory=lambda: float(os.environ.get("COLIBRI_TEMPERATURE", "0.2"))
    )
    max_tokens: int = field(
        default_factory=lambda: _env_int("COLIBRI_MAX_TOKENS", 384)
    )
    request_timeout: float = field(
        default_factory=lambda: float(os.environ.get("COLIBRI_TIMEOUT", "120.0"))
    )
    use_systemone_routing: bool = field(
        default_factory=lambda: _env_bool("COLIBRI_USE_SYSTEMONE", False)
    )
    auto_avatar_state: bool = field(
        default_factory=lambda: _env_bool("COLIBRI_AUTO_AVATAR", True)
    )
    async_avatar_bridge: bool = field(
        default_factory=lambda: _env_bool("COLIBRI_ASYNC_AVATAR", False)
    )
    prefetch_memory: bool = field(
        default_factory=lambda: _env_bool("COLIBRI_PREFETCH_MEMORY", True)
    )
    fast_tool_pruning: bool = field(
        default_factory=lambda: _env_bool("COLIBRI_FAST_TOOL_PRUNING", False)
    )
    stream_tts: bool = field(
        default_factory=lambda: _env_bool("COLIBRI_STREAM_TTS", False)
    )
    rpi_url: str = field(
        default_factory=lambda: os.environ.get("COLIBRI_RPI_URL", "http://127.0.0.1:8081")
    )

    def __post_init__(self) -> None:
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url.rstrip("/") + "/v1"
        if not self.model or self.context_length <= 0:
            detected_model, detected_ctx = detect_colibri_model(
                self.base_url, self.api_key
            )
            if not self.model:
                self.model = detected_model
            if self.context_length <= 0:
                self.context_length = detected_ctx

    @property
    def is_ollama_or_metal(self) -> bool:
        """Return True when targeting an Apple Metal Ollama or MLX server instead of `coli serve`."""
        return ":11434" in self.base_url or ":8080" in self.base_url or "ollama" in self.base_url.lower()

    @property
    def root_url(self) -> str:
        """Gateway root URL without trailing ``/v1`` (e.g. ``http://127.0.0.1:8000``)."""
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3]
        return self.base_url

    def extra_body(self) -> Dict[str, Any]:
        """Colibri-specific OpenAI request extensions for KV reuse and context compaction."""
        if self.is_ollama_or_metal:
            # Ollama / MLX Metal backends ignore or reject custom extra_body fields
            return {}
        body: Dict[str, Any] = {
            "cache_slot": self.cache_slot,
            "enable_thinking": self.enable_thinking,
        }
        if self.preserve_thinking is not None:
            body["preserve_thinking"] = self.preserve_thinking
        else:
            # When thinking is disabled, preserve_thinking=True guarantees prefix reuse.
            body["preserve_thinking"] = not self.enable_thinking
        if self.compact and self.compact != "off":
            body["context_compaction"] = self.compact
        return body

    @classmethod
    def from_env(cls, **overrides: Any) -> "AgentConfig":
        """Create an ``AgentConfig`` from environment variables with optional overrides."""
        cfg = cls()
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
