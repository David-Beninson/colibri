"""Perception and Vision tool suite for the Colibri Companion (`src/ai/tools/vision.py`).

Provides:
- ``extract_text_from_image``: Multilingual OCR (`heb+eng` with fallback to `eng`) via `pytesseract` & `PIL`.
- ``describe_image_scene``: Scene understanding with cascading vision providers (Google Gemini Flash ->
  OpenRouter free vision -> local Colibri vision endpoint -> local PIL + OCR metadata fallback).
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from .internet import _tool_decorator


def _read_image_dimensions(image_path: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Return ``(width, height, format)`` using Pillow if installed, else ``(None, None, None)``."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as img:
            return img.width, img.height, img.format
    except Exception:  # noqa: BLE001
        return None, None, None


@_tool_decorator
def extract_text_from_image(image_path: str, lang: str = "heb+eng") -> str:
    """Extract text (OCR) from a local image file using Tesseract (`heb+eng` with fallback to `eng`)."""
    path = os.path.expanduser((image_path or "").strip())
    if not path or not os.path.isfile(path):
        return json.dumps({"error": f"Image file not found: {image_path}"})

    width, height, fmt = _read_image_dimensions(path)

    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore

        with Image.open(path) as img:
            used_lang = lang
            try:
                raw_text = pytesseract.image_to_string(img, lang=lang)
            except Exception:  # noqa: BLE001 - fallback to English if heb+eng tessdata is missing
                used_lang = "eng"
                raw_text = pytesseract.image_to_string(img, lang="eng")

        cleaned = (raw_text or "").strip()
        return json.dumps(
            {
                "image_path": path,
                "language": used_lang,
                "text": cleaned,
                "chars": len(cleaned),
                "dimensions": {"width": width, "height": height, "format": fmt},
            },
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "image_path": path,
                "text": "",
                "chars": 0,
                "dimensions": {"width": width, "height": height, "format": fmt},
                "error": f"OCR unavailable ({type(exc).__name__}: {exc}). Ensure pillow and pytesseract/tesseract-ocr are installed.",
            },
            ensure_ascii=False,
        )


def _encode_image_base64(image_path: str) -> Tuple[str, str]:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return b64, mime


def _query_gemini_vision(b64_data: str, mime: str, prompt: str, api_key: str) -> Optional[str]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": b64_data}},
                ]
            }
        ]
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    candidates = data.get("candidates") or []
    if candidates:
        parts = candidates[0].get("content", {}).get("parts") or []
        texts = [p.get("text", "") for p in parts if p.get("text")]
        if texts:
            return "\n".join(texts).strip()
    return None


def _query_openai_compat_vision(
    endpoint_url: str,
    api_key: str,
    model: str,
    b64_data: str,
    mime: str,
    prompt: str,
) -> Optional[str]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_data}"},
                    },
                ],
            }
        ],
        "max_tokens": 512,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        endpoint_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    choices = data.get("choices") or []
    if choices:
        content = choices[0].get("message", {}).get("content")
        if content:
            return str(content).strip()
    return None


@_tool_decorator
def describe_image_scene(
    image_path: str,
    prompt: str = "Describe this scene in detail, including any people, expressions, held objects, or text.",
) -> str:
    """Analyze a camera frame or image file and return a rich scene description.

    Tries in priority order:
    1. Google Gemini Flash (`GEMINI_API_KEY`)
    2. OpenRouter Vision (`OPENROUTER_API_KEY`, model `COLIBRI_VISION_MODEL` default `google/gemini-2.0-flash-exp:free`)
    3. Local Colibri multimodal endpoint (`COLIBRI_VISION_URL` or `COLIBRI_BASE_URL` when `COLIBRI_LOCAL_VISION=1`)
    4. Offline local fallback combining image dimensions/format and OCR text via `extract_text_from_image`.
    """
    path = os.path.expanduser((image_path or "").strip())
    if not path or not os.path.isfile(path):
        return json.dumps({"error": f"Image file not found: {image_path}"})

    try:
        b64_data, mime = _encode_image_base64(path)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Failed to read image {path}: {exc}"})

    # 1. Google Gemini Flash API
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            desc = _query_gemini_vision(b64_data, mime, prompt, gemini_key)
            if desc:
                return json.dumps(
                    {"image_path": path, "provider": "gemini-2.0-flash", "description": desc},
                    ensure_ascii=False,
                )
        except Exception:  # noqa: BLE001
            pass

    # 2. OpenRouter Vision API
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        vision_model = os.environ.get(
            "COLIBRI_VISION_MODEL", "google/gemini-2.0-flash-exp:free"
        ).strip()
        try:
            desc = _query_openai_compat_vision(
                "https://openrouter.ai/api/v1/chat/completions",
                openrouter_key,
                vision_model,
                b64_data,
                mime,
                prompt,
            )
            if desc:
                return json.dumps(
                    {"image_path": path, "provider": f"openrouter/{vision_model}", "description": desc},
                    ensure_ascii=False,
                )
        except Exception:  # noqa: BLE001
            pass

    # 3. Local Colibri multimodal endpoint (when configured)
    local_vision_url = os.environ.get("COLIBRI_VISION_URL", "").strip()
    if not local_vision_url and os.environ.get("COLIBRI_LOCAL_VISION", "").strip() in ("1", "true", "yes"):
        base = os.environ.get("COLIBRI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        local_vision_url = f"{base}/chat/completions"
    if local_vision_url:
        try:
            local_model = os.environ.get("COLIBRI_MODEL", "qwen38-flash-next")
            local_key = os.environ.get("COLIBRI_API_KEY", "colibri")
            desc = _query_openai_compat_vision(
                local_vision_url,
                local_key,
                local_model,
                b64_data,
                mime,
                prompt,
            )
            if desc:
                return json.dumps(
                    {"image_path": path, "provider": "colibri-local-vision", "description": desc},
                    ensure_ascii=False,
                )
        except Exception:  # noqa: BLE001
            pass

    # 4. Local PIL + OCR fallback
    width, height, fmt = _read_image_dimensions(path)
    ocr_raw = extract_text_from_image(path)
    try:
        ocr_obj = json.loads(ocr_raw)
        ocr_text = ocr_obj.get("text", "")
    except Exception:  # noqa: BLE001
        ocr_text = ""

    return json.dumps(
        {
            "image_path": path,
            "provider": "local-fallback",
            "dimensions": {"width": width, "height": height, "format": fmt},
            "ocr_text": ocr_text,
            "description": (
                f"Local image ({fmt or 'unknown'} {width}x{height}). "
                + (f"Detected text: {ocr_text[:400]}" if ocr_text else "No readable OCR text detected. ")
                + " (Set GEMINI_API_KEY or OPENROUTER_API_KEY for full neural scene descriptions.)"
            ),
        },
        ensure_ascii=False,
    )


VISION_TOOLS: List[Any] = [
    extract_text_from_image,
    describe_image_scene,
]

VISION_TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "extract_text_from_image": extract_text_from_image,
    "describe_image_scene": describe_image_scene,
}

VISION_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "extract_text_from_image",
            "description": "Extract text (OCR) in Hebrew and English from a local image or camera frame using Tesseract.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Local file path to the image."},
                    "lang": {
                        "type": "string",
                        "description": "Tesseract language code (default 'heb+eng').",
                        "default": "heb+eng",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_image_scene",
            "description": "Describe the visual scene, people, expressions, or held objects in a camera frame or image file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Local file path to the image."},
                    "prompt": {
                        "type": "string",
                        "description": "Specific visual question or instruction for describing the scene.",
                        "default": "Describe this scene in detail, including any people, expressions, held objects, or text.",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
]

__all__ = [
    "VISION_TOOLS",
    "VISION_TOOL_REGISTRY",
    "VISION_TOOLS_SCHEMA",
    "extract_text_from_image",
    "describe_image_scene",
]

