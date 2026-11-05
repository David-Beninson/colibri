"""Unified Gateway Seam for Colibri Inference Gateway (`src/ai/gateway.py`).

Defines the core `ColibriGatewayPort` abstraction and production/testing adapters:
- `ColibriGatewayPort`: Abstract interface for chat completions, zero-generation SystemOne routing, and model probing.
- `HttpGatewayAdapter`: Concrete HTTP implementation via `urllib.request` with SSE streaming and robust error handling.
- `InMemoryGatewayAdapter`: In-memory test double supporting canned responses, custom handlers, and call inspection.
"""
from __future__ import annotations

import abc
import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

DEFAULT_GATEWAY_URL = "http://127.0.0.1:8000/v1"
DEFAULT_GATEWAY_API_KEY = "colibri"
DEFAULT_GATEWAY_MODEL = "qwen36-35b"
DEFAULT_GATEWAY_TIMEOUT = 60.0


class ColibriGatewayPort(abc.ABC):
    """Abstract port interface for Colibri inference gateway operations.

    Unifies:
    1. Chat completions with KV cache slot preservation and optional SSE streaming (`/v1/chat/completions`)
    2. Zero-generation SystemOne decision routing (`/v1/systemone`)
    3. Model probing and context length discovery (`/v1/models`)
    """

    @abc.abstractmethod
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
        """Execute a chat completion request with KV cache slot preservation and optional SSE streaming."""
        raise NotImplementedError

    @abc.abstractmethod
    def decide_systemone(
        self,
        state: Any = None,
        questions: Optional[Mapping[str, Any]] = None,
        slot: int = 2,
        normalize: Union[bool, str] = True,
        extra_body: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a zero-generation SystemOne decision routing query batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def probe_models(self) -> List[Dict[str, Any]]:
        """Query /v1/models and return available model descriptor objects."""
        raise NotImplementedError

    def complete(
        self,
        config: Any,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        active_slot: Optional[int] = None,
        stream_tokens: bool = False,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Backward-compatibility bridge for legacy ColibriTransport.complete callers."""
        slot = getattr(config, "cache_slot", 0) if active_slot is None else active_slot
        temp = getattr(config, "temperature", 0.7)
        max_tok = getattr(config, "max_tokens", None)
        model = getattr(config, "model", None)
        extra: Dict[str, Any] = {}
        if hasattr(config, "extra_body"):
            try:
                extra = config.extra_body(slot=active_slot)
            except Exception:  # noqa: BLE001
                pass

        return self.complete_chat(
            messages=messages,
            tools=tools,
            slot=slot,
            stream=stream_tokens,
            on_token=token_callback,
            extra_body=extra,
            temperature=temp,
            max_tokens=max_tok,
            model=model,
        )

    def __call__(
        self,
        config: Any,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        active_slot: Optional[int] = None,
        stream_tokens: bool = False,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        return self.complete(
            config=config,
            messages=messages,
            tools=tools,
            active_slot=active_slot,
            stream_tokens=stream_tokens,
            token_callback=token_callback,
        )


class HttpGatewayAdapter(ColibriGatewayPort):
    """Production HTTP gateway adapter using `urllib.request`.

    Handles:
    - POST `/v1/chat/completions` (JSON or SSE streaming)
    - POST `/v1/systemone`
    - GET `/v1/models`
    """

    def __init__(
        self,
        base_url: str = DEFAULT_GATEWAY_URL,
        api_key: str = DEFAULT_GATEWAY_API_KEY,
        model: str = DEFAULT_GATEWAY_MODEL,
        timeout: float = DEFAULT_GATEWAY_TIMEOUT,
        config: Optional[Any] = None,
    ) -> None:
        if config is not None:
            self.base_url = str(getattr(config, "base_url", base_url))
            self.api_key = str(getattr(config, "api_key", api_key))
            self.model = str(getattr(config, "model", model))
            self.timeout = float(getattr(config, "request_timeout", timeout))
        else:
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
            self.timeout = timeout

    def _resolve_url(self, endpoint: str) -> str:
        base = self.base_url.rstrip("/")
        clean_ep = endpoint.lstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/{clean_ep}"

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
        url = self._resolve_url("chat/completions")

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": bool(stream),
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if slot is not None:
            payload["cache_slot"] = slot
        if extra_body:
            payload.update(extra_body)
            payload["extra_body"] = dict(extra_body)
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if not stream:
                    raw = resp.read().decode("utf-8", errors="replace")
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed JSON response from {url}: {exc}") from exc

                content_parts: List[str] = []
                reasoning_parts: List[str] = []
                tool_calls_acc: Dict[int, Dict[str, Any]] = {}
                compaction_info: Optional[Dict[str, Any]] = None

                raw_bytes = resp.read()
                raw_text = raw_bytes.decode("utf-8", errors="replace")

                # Support both standard JSON responses (mock servers / fallback) and SSE data: lines
                if raw_text.lstrip().startswith("{"):
                    try:
                        parsed_json = json.loads(raw_text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed JSON response from {url}: {exc}") from exc
                    choices = parsed_json.get("choices") or []
                    msg = (choices[0].get("message") if choices else None) or {}
                    c_str = str(msg.get("content") or "")
                    if c_str and on_token:
                        on_token(c_str)
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
                        if on_token is not None:
                            on_token(str(c_delta))
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

        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                f"HTTP {exc.code} error from {url}: {err_body or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network connection failed for {url}: {exc.reason}") from exc

    def decide_systemone(
        self,
        state: Any = None,
        questions: Optional[Mapping[str, Any]] = None,
        slot: int = 2,
        normalize: Union[bool, str] = True,
        extra_body: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = self._resolve_url("systemone")
        norm_val = (
            "sum"
            if normalize is True
            else ("none" if normalize is False else str(normalize))
        )
        merged_extra: Dict[str, Any] = {"cache_slot": slot}
        if extra_body:
            merged_extra.update(extra_body)

        body: Dict[str, Any] = {
            "model": model or self.model,
            "state": state,
            "questions": dict(questions or {}),
            "normalize": norm_val,
            "cache_slot": slot,
            "extra_body": merged_extra,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON response from {url}: {exc}") from exc
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                f"HTTP {exc.code} error from {url}: {err_body or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network connection failed for {url}: {exc.reason}") from exc

    def probe_models(self) -> List[Dict[str, Any]]:
        url = self._resolve_url("models")
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            data = payload.get("data")
            if isinstance(data, list):
                return data
            if isinstance(payload, list):
                return payload
            return []
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} probing models from {url}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network connection failed probing models from {url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON response from {url}: {exc}") from exc


class InMemoryGatewayAdapter(ColibriGatewayPort):
    """Pure in-memory test double for Colibri inference gateway operations.

    Records all calls and supports canned responses, response queues, or custom handlers for:
    - Chat completions (`complete_chat`)
    - SystemOne decisions (`decide_systemone`)
    - Model probing (`probe_models`)
    """

    def __init__(
        self,
        canned_chat: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]], Callable[..., Dict[str, Any]]]] = None,
        canned_systemone: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]], Callable[..., Dict[str, Any]]]] = None,
        canned_models: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        self.chat_calls: List[Dict[str, Any]] = []
        self.systemone_calls: List[Dict[str, Any]] = []
        self.models_calls: int = 0

        self._chat_queue: List[Union[Dict[str, Any], Callable[..., Dict[str, Any]]]] = []
        if canned_chat is not None:
            if isinstance(canned_chat, (list, tuple)):
                self._chat_queue.extend(canned_chat)
            else:
                self._chat_queue.append(canned_chat)

        self._sys1_queue: List[Union[Dict[str, Any], Callable[..., Dict[str, Any]]]] = []
        if canned_systemone is not None:
            if isinstance(canned_systemone, (list, tuple)):
                self._sys1_queue.extend(canned_systemone)
            else:
                self._sys1_queue.append(canned_systemone)

        self._models: List[Dict[str, Any]] = (
            list(canned_models)
            if canned_models is not None
            else [{"id": "qwen36-35b", "context_length": 65536}]
        )

    def register_chat_response(
        self, response: Union[Dict[str, Any], Callable[..., Dict[str, Any]]]
    ) -> None:
        """Register a canned chat response or dynamic callable."""
        self._chat_queue.append(response)

    def register_systemone_response(
        self, response: Union[Dict[str, Any], Callable[..., Dict[str, Any]]]
    ) -> None:
        """Register a canned SystemOne decision response or dynamic callable."""
        self._sys1_queue.append(response)

    def register_models(self, models: Sequence[Dict[str, Any]]) -> None:
        """Register a list of model descriptor dictionaries."""
        self._models = list(models)

    def reset(self) -> None:
        """Clear recorded calls and queued responses."""
        self.chat_calls.clear()
        self.systemone_calls.clear()
        self.models_calls = 0
        self._chat_queue.clear()
        self._sys1_queue.clear()

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
        call_record = {
            "messages": list(messages),
            "tools": list(tools) if tools else None,
            "slot": slot,
            "stream": stream,
            "extra_body": dict(extra_body) if extra_body else None,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model": model,
        }
        self.chat_calls.append(call_record)

        if self._chat_queue:
            item = self._chat_queue.pop(0) if len(self._chat_queue) > 1 else self._chat_queue[0]
            if callable(item):
                resp = item(
                    messages=messages,
                    tools=tools,
                    slot=slot,
                    stream=stream,
                    extra_body=extra_body,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                )
            else:
                resp = dict(item)
        else:
            resp = {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! I am Colibri companion.",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

        if on_token is not None:
            choices = resp.get("choices") or []
            msg = (choices[0].get("message") if choices else None) or {}
            c_str = str(msg.get("content") or "")
            if c_str:
                on_token(c_str)

        return resp

    def decide_systemone(
        self,
        state: Any = None,
        questions: Optional[Mapping[str, Any]] = None,
        slot: int = 2,
        normalize: Union[bool, str] = True,
        extra_body: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        call_record = {
            "state": state,
            "questions": dict(questions or {}),
            "slot": slot,
            "normalize": normalize,
            "extra_body": dict(extra_body) if extra_body else None,
            "model": model,
        }
        self.systemone_calls.append(call_record)

        if self._sys1_queue:
            item = self._sys1_queue.pop(0) if len(self._sys1_queue) > 1 else self._sys1_queue[0]
            if callable(item):
                return item(
                    state=state,
                    questions=questions,
                    slot=slot,
                    normalize=normalize,
                    extra_body=extra_body,
                    model=model,
                )
            return dict(item)

        return {
            "model": model or "qwen36-35b",
            "answers": {
                "q": {
                    "choice": "direct",
                    "probabilities": {"direct": 1.0},
                    "confidence": 1.0,
                    "noul": 1.0,
                    "score": 1.0,
                }
            },
        }

    def probe_models(self) -> List[Dict[str, Any]]:
        self.models_calls += 1
        return list(self._models)


__all__ = [
    "DEFAULT_GATEWAY_API_KEY",
    "DEFAULT_GATEWAY_MODEL",
    "DEFAULT_GATEWAY_TIMEOUT",
    "DEFAULT_GATEWAY_URL",
    "ColibriGatewayPort",
    "HttpGatewayAdapter",
    "InMemoryGatewayAdapter",
]
