"""Voice tool re-exports (`src/ai/tools/voice.py`) for seamless compatibility with `audio.py`."""
from __future__ import annotations

from .audio import (
    AUDIO_TOOL_REGISTRY,
    AUDIO_TOOLS,
    AUDIO_TOOLS_SCHEMA,
    _WHISPER_CACHE,
    speak_text,
    text_to_speech,
    transcribe_audio,
)

VOICE_TOOLS = AUDIO_TOOLS
VOICE_TOOL_REGISTRY = AUDIO_TOOL_REGISTRY
VOICE_TOOLS_SCHEMA = AUDIO_TOOLS_SCHEMA

__all__ = [
    "transcribe_audio",
    "text_to_speech",
    "speak_text",
    "AUDIO_TOOLS",
    "AUDIO_TOOL_REGISTRY",
    "AUDIO_TOOLS_SCHEMA",
    "VOICE_TOOLS",
    "VOICE_TOOL_REGISTRY",
    "VOICE_TOOLS_SCHEMA",
    "_WHISPER_CACHE",
]
