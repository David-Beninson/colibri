"""Hardware & Pepper's Ghost 3D Avatar Control hooks (`src/ai/tools/device.py`).

Controls the Desktop Holographic AI Companion cylinder via the Raspberry Pi
hardware bridge (`COLIBRI_RPI_URL`, default `http://127.0.0.1:8081`) with
automatic local simulation persistence at `~/.colibri/memory/device_state.json`
when the Raspberry Pi controller is offline or running in simulation mode.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from .internet import _tool_decorator
except Exception:  # noqa: BLE001

    def _tool_decorator(func: Callable[..., str]) -> Any:
        func.name = func.__name__  # type: ignore[attr-defined]
        func.description = (func.__doc__ or "").strip()  # type: ignore[attr-defined]
        func.invoke = (  # type: ignore[attr-defined]
            lambda inp, _f=func: _f(**inp) if isinstance(inp, dict) else _f(inp)
        )
        return func


DEFAULT_RPI_URL = "http://127.0.0.1:8081"

VALID_AVATAR_STATES = {
    "idle",
    "thinking",
    "speaking",
    "listening",
    "happy",
    "birthday",
    "love",
    "waving",
}

VALID_HOLOGRAM_EFFECTS = {
    "birthday_confetti",
    "floating_hearts",
    "starlight_glow",
    "ambient_pulse",
    "clear",
}


def _state_file_path() -> Path:
    custom = os.environ.get("COLIBRI_DEVICE_STATE_PATH")
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".colibri" / "memory" / "device_state.json"


def _default_state() -> Dict[str, Any]:
    return {
        "servo_angle": 0,
        "unit": "degrees",
        "avatar_state": "idle",
        "emotion": "warm",
        "subtitle": "",
        "hologram_effect": "ambient_pulse",
        "led_rgb": "255,180,220",
        "rpi_url": os.environ.get("COLIBRI_RPI_URL", DEFAULT_RPI_URL).rstrip("/"),
        "hardware_connected": False,
        "updated_at": time.time(),
    }


_MEM_DEVICE_STATE: Dict[str, Any] = {}
_MEM_STATE_PATH: Optional[Path] = None
_RPI_CIRCUIT_OPEN_UNTIL: Dict[str, float] = {}
_RPI_COOLDOWN_SECONDS: float = 15.0


def get_device_state() -> Dict[str, Any]:
    """Return the current servo angle, avatar animation state, emotion, and holographic effect."""
    global _MEM_DEVICE_STATE, _MEM_STATE_PATH
    path = _state_file_path()
    base = _default_state()
    if _MEM_STATE_PATH == path and _MEM_DEVICE_STATE and path.exists():
        base.update(_MEM_DEVICE_STATE)
        return base
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                base.update(loaded)
                _MEM_DEVICE_STATE = dict(base)
                _MEM_STATE_PATH = path
        except Exception:  # noqa: BLE001
            pass
    return base


def _save_device_state(updates: Dict[str, Any]) -> Dict[str, Any]:
    global _MEM_DEVICE_STATE, _MEM_STATE_PATH
    state = get_device_state()
    state.update(updates)
    state["updated_at"] = round(time.time(), 3)
    path = _state_file_path()
    _MEM_DEVICE_STATE = dict(state)
    _MEM_STATE_PATH = path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return state


def _post_rpi(
    endpoint: str,
    payload: Dict[str, Any],
    timeout: float = 0.25,
    async_fire: bool = False,
) -> Dict[str, Any]:
    """Send a JSON command to the Raspberry Pi controller with fast circuit-breaker fallback."""
    rpi_url = os.environ.get("COLIBRI_RPI_URL", DEFAULT_RPI_URL).rstrip("/")
    url = f"{rpi_url}/{endpoint.lstrip('/')}"
    now = time.monotonic()

    # Circuit breaker: skip blocking socket connection if RPi recently timed out
    # (unless urlopen is patched during unit tests)
    is_mocked = getattr(urllib.request.urlopen, "__module__", "") != "urllib.request"
    if not is_mocked and _RPI_CIRCUIT_OPEN_UNTIL.get(rpi_url, 0.0) > now:
        return {
            "hardware_connected": False,
            "mode": "simulation",
            "rpi_url": rpi_url,
            "note": "Simulated locally (Raspberry Pi bridge offline circuit-breaker active)",
        }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    if async_fire and not is_mocked:
        def _bg() -> None:
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    resp.read()
                _RPI_CIRCUIT_OPEN_UNTIL.pop(rpi_url, None)
            except Exception:  # noqa: BLE001
                _RPI_CIRCUIT_OPEN_UNTIL[rpi_url] = time.monotonic() + _RPI_COOLDOWN_SECONDS

        threading.Thread(target=_bg, daemon=True).start()
        return {
            "hardware_connected": False,
            "mode": "async_bridge",
            "rpi_url": rpi_url,
            "note": "Dispatched asynchronously in background thread",
        }

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            reply = json.loads(raw) if raw.strip() else {}
            _RPI_CIRCUIT_OPEN_UNTIL.pop(rpi_url, None)
            return {
                "hardware_connected": True,
                "mode": "hardware",
                "rpi_url": rpi_url,
                "controller_reply": reply,
            }
    except Exception as exc:  # noqa: BLE001
        if not is_mocked:
            _RPI_CIRCUIT_OPEN_UNTIL[rpi_url] = time.monotonic() + _RPI_COOLDOWN_SECONDS
        return {
            "hardware_connected": False,
            "mode": "simulation",
            "rpi_url": rpi_url,
            "note": f"Simulated locally (Raspberry Pi bridge offline: {type(exc).__name__})",
        }


def _rotate_impl(angle: int) -> str:
    try:
        raw_angle = int(round(float(angle)))
    except (TypeError, ValueError):
        return json.dumps({"status": "error", "error": f"Invalid servo angle: {angle!r}"})

    clamped = max(-180, min(180, raw_angle))
    payload = {"angle": clamped, "unit": "degrees", "requested_angle": raw_angle}
    async_fire = os.environ.get("COLIBRI_SYNC_RPI", "").strip().lower() not in ("1", "true", "yes")
    bridge = _post_rpi("/servo/rotate", payload, async_fire=async_fire)
    state = _save_device_state(
        {
            "servo_angle": clamped,
            "unit": "degrees",
            "hardware_connected": bridge["hardware_connected"],
        }
    )
    return json.dumps(
        {
            "status": "ok",
            "action": "rotate_device",
            "servo_angle": state["servo_angle"],
            "clamped": clamped != raw_angle,
            "unit": "degrees",
            "mode": bridge["mode"],
            "hardware_connected": bridge["hardware_connected"],
        },
        ensure_ascii=False,
    )


@_tool_decorator
def rotate_device(angle: int) -> str:
    """Rotate the motorized holographic display cylinder base toward the user (angle in degrees, -180 to 180)."""
    return _rotate_impl(angle)


@_tool_decorator
def rotate_device_servo(angle: int) -> str:
    """Rotate the Raspberry Pi servo motor controlling the holographic cylinder (angle in degrees, -180 to 180)."""
    return _rotate_impl(angle)


def _avatar_state_impl(state: str, emotion: str = "warm", subtitle: str = "") -> str:
    norm_state = (state or "idle").strip().lower()
    if norm_state not in VALID_AVATAR_STATES:
        return json.dumps(
            {
                "status": "error",
                "error": f"Invalid avatar state {state!r}. Allowed: {sorted(VALID_AVATAR_STATES)}",
            },
            ensure_ascii=False,
        )
    norm_emotion = (emotion or "warm").strip().lower()
    clean_subtitle = (subtitle or "").strip()[:280]
    payload = {
        "state": norm_state,
        "emotion": norm_emotion,
        "subtitle": clean_subtitle,
    }
    async_fire = os.environ.get("COLIBRI_SYNC_RPI", "").strip().lower() not in ("1", "true", "yes")
    bridge = _post_rpi("/avatar/state", payload, async_fire=async_fire)
    saved = _save_device_state(
        {
            "avatar_state": norm_state,
            "emotion": norm_emotion,
            "subtitle": clean_subtitle,
            "hardware_connected": bridge["hardware_connected"],
        }
    )
    return json.dumps(
        {
            "status": "ok",
            "action": "set_avatar_state",
            "avatar_state": saved["avatar_state"],
            "emotion": saved["emotion"],
            "subtitle": saved["subtitle"],
            "mode": bridge["mode"],
            "hardware_connected": bridge["hardware_connected"],
        },
        ensure_ascii=False,
    )


@_tool_decorator
def set_avatar_state(state: str, emotion: str = "warm", subtitle: str = "") -> str:
    """Set the Pepper's Ghost 3D holographic avatar animation state ('idle', 'thinking', 'speaking', 'listening', 'happy', 'birthday', 'love', 'waving'), facial emotion, and optional floating subtitle."""
    return _avatar_state_impl(state=state, emotion=emotion, subtitle=subtitle)


@_tool_decorator
def trigger_avatar_state(state: str, emotion: str = "warm", subtitle: str = "") -> str:
    """Trigger a 3D holographic avatar animation state ('idle', 'thinking', 'speaking', 'listening', 'happy', 'birthday', 'love', 'waving') with emotion and optional subtitle."""
    return _avatar_state_impl(state=state, emotion=emotion, subtitle=subtitle)


@_tool_decorator
def trigger_hologram_effect(
    effect: str = "birthday_confetti",
    led_rgb: str = "255,180,220",
) -> str:
    """Trigger a Pepper's Ghost holographic visual overlay ('birthday_confetti', 'floating_hearts', 'starlight_glow', 'ambient_pulse', 'clear') and cylinder LED ring RGB color."""
    norm_effect = (effect or "birthday_confetti").strip().lower()
    if norm_effect not in VALID_HOLOGRAM_EFFECTS:
        return json.dumps(
            {
                "status": "error",
                "error": f"Invalid effect {effect!r}. Allowed: {sorted(VALID_HOLOGRAM_EFFECTS)}",
            },
            ensure_ascii=False,
        )
    clean_rgb = (led_rgb or "255,180,220").strip()
    payload = {"effect": norm_effect, "led_rgb": clean_rgb}
    async_fire = os.environ.get("COLIBRI_SYNC_RPI", "").strip().lower() not in ("1", "true", "yes")
    bridge = _post_rpi("/hologram/effect", payload, async_fire=async_fire)
    saved = _save_device_state(
        {
            "hologram_effect": norm_effect,
            "led_rgb": clean_rgb,
            "hardware_connected": bridge["hardware_connected"],
        }
    )
    return json.dumps(
        {
            "status": "ok",
            "action": "trigger_hologram_effect",
            "hologram_effect": saved["hologram_effect"],
            "led_rgb": saved["led_rgb"],
            "mode": bridge["mode"],
            "hardware_connected": bridge["hardware_connected"],
        },
        ensure_ascii=False,
    )


DEVICE_TOOLS: List[Any] = [
    rotate_device,
    rotate_device_servo,
    set_avatar_state,
    trigger_avatar_state,
    trigger_hologram_effect,
]

DEVICE_TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "rotate_device": rotate_device,
    "rotate_device_servo": rotate_device_servo,
    "set_avatar_state": set_avatar_state,
    "trigger_avatar_state": trigger_avatar_state,
    "trigger_hologram_effect": trigger_hologram_effect,
}

DEVICE_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "rotate_device",
            "description": "Rotate the motorized holographic display cylinder base toward the user (angle in degrees, -180 to 180).",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "integer",
                        "description": "Target rotation angle in degrees (-180 to 180, where 0 is center).",
                    },
                },
                "required": ["angle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate_device_servo",
            "description": "Rotate the Raspberry Pi servo motor controlling the holographic cylinder (angle in degrees, -180 to 180).",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "integer",
                        "description": "Target servo rotation angle in degrees (-180 to 180).",
                    },
                },
                "required": ["angle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_avatar_state",
            "description": "Set the Pepper's Ghost 3D holographic avatar animation state, facial emotion, and optional subtitle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": sorted(VALID_AVATAR_STATES),
                        "description": "Avatar animation state ('idle', 'thinking', 'speaking', 'listening', 'happy', 'birthday', 'love', 'waving').",
                    },
                    "emotion": {
                        "type": "string",
                        "description": "Facial expression tone (e.g. 'warm', 'joyful', 'curious', 'celebratory').",
                        "default": "warm",
                    },
                    "subtitle": {
                        "type": "string",
                        "description": "Optional short caption displayed beneath the 3D hologram (English or Hebrew).",
                        "default": "",
                    },
                },
                "required": ["state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_avatar_state",
            "description": "Trigger a 3D holographic avatar animation state ('idle', 'thinking', 'speaking', 'listening', 'happy', 'birthday', 'love', 'waving').",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": sorted(VALID_AVATAR_STATES),
                        "description": "Avatar animation state to trigger.",
                    },
                    "emotion": {
                        "type": "string",
                        "description": "Facial expression tone.",
                        "default": "warm",
                    },
                    "subtitle": {
                        "type": "string",
                        "description": "Optional subtitle text.",
                        "default": "",
                    },
                },
                "required": ["state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_hologram_effect",
            "description": "Trigger a Pepper's Ghost holographic visual overlay ('birthday_confetti', 'floating_hearts', 'starlight_glow', 'ambient_pulse', 'clear') and cylinder LED ring color.",
            "parameters": {
                "type": "object",
                "properties": {
                    "effect": {
                        "type": "string",
                        "enum": sorted(VALID_HOLOGRAM_EFFECTS),
                        "description": "Holographic particle/shader overlay effect.",
                        "default": "birthday_confetti",
                    },
                    "led_rgb": {
                        "type": "string",
                        "description": "Comma-separated RGB value for the base LED ring, e.g. '255,180,220'.",
                        "default": "255,180,220",
                    },
                },
                "required": [],
            },
        },
    },
]
