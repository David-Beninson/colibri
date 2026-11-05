"""Voice and Audio (STT & TTS) tool suite for the Desktop Holographic AI Companion (`src/ai/tools/audio.py`).

Provides:
- ``transcribe_audio``: Local speech-to-text via `openai-whisper` (`whisper.load_model`) with module-level
  model caching (`_WHISPER_CACHE`) and automatic Hebrew (`he`) / English (`en`) language detection.
- ``text_to_speech`` / ``speak_text``: Expressive neural speech synthesis via `edge_tts` (`he-IL-HilaNeural`
  default, with smart bilingual voice selection and safe async execution from sync/async contexts).
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Dict, List, Optional

from .internet import _tool_decorator

_WHISPER_CACHE: Dict[str, Any] = {}
_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


def _select_voice(text: str, requested_voice: str) -> str:
    """Select the appropriate Edge-TTS neural voice for Hebrew/English text."""
    voice = (requested_voice or "he-IL-HilaNeural").strip()
    if voice.lower() == "auto":
        if _HEBREW_RE.search(text or ""):
            return os.environ.get("COLIBRI_HE_VOICE", "he-IL-HilaNeural")
        return os.environ.get("COLIBRI_EN_VOICE", "en-US-AriaNeural")
    if voice == "he-IL-HilaNeural" and not _HEBREW_RE.search(text or ""):
        env_en = os.environ.get("COLIBRI_EN_VOICE", "").strip()
        if env_en:
            return env_en
    return voice


def _run_coro_sync(coro: Any) -> Any:
    """Run an async coroutine safely whether or not an event loop is already running in this thread."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        result_holder: Dict[str, Any] = {}

        def _runner() -> None:
            try:
                result_holder["result"] = asyncio.run(coro)
            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = exc

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join(timeout=60.0)
        if "error" in result_holder:
            raise result_holder["error"]
        return result_holder.get("result")

    return asyncio.run(coro)


@_tool_decorator
def transcribe_audio(
    audio_path: str = "",
    audio_file_path: str = "",
    model_size: str = "base",
    language: Optional[str] = None,
) -> str:
    """Transcribe a spoken audio file (Hebrew or English) into text using local OpenAI Whisper."""
    target_path = os.path.expanduser((audio_path or audio_file_path or "").strip())
    if not target_path or not os.path.isfile(target_path):
        return json.dumps(
            {"error": f"Audio file not found: {audio_path or audio_file_path}"}
        )

    size = (model_size or os.environ.get("COLIBRI_WHISPER_MODEL", "base")).strip()
    try:
        import whisper  # type: ignore

        if size not in _WHISPER_CACHE:
            _WHISPER_CACHE[size] = whisper.load_model(size)
        model = _WHISPER_CACHE[size]
        kwargs: Dict[str, Any] = {}
        if language:
            kwargs["language"] = language
        result = model.transcribe(target_path, **kwargs)
        text = (result.get("text") or "").strip()
        detected_lang = result.get("language") or language or "auto"
        return json.dumps(
            {
                "audio_path": target_path,
                "text": text,
                "language": detected_lang,
                "model": size,
            },
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "audio_path": target_path,
                "text": "",
                "model": size,
                "error": f"Whisper transcription unavailable ({type(exc).__name__}: {exc}). Install openai-whisper.",
            },
            ensure_ascii=False,
        )


def _synthesize_edge_tts(
    text: str,
    output_path: str = "/tmp/colibri_tts.mp3",
    voice: str = "he-IL-HilaNeural",
) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return json.dumps({"error": "Empty text for speech synthesis"})

    out_file = os.path.expanduser((output_path or "/tmp/colibri_tts.mp3").strip())
    parent_dir = os.path.dirname(os.path.abspath(out_file))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    selected_voice = _select_voice(cleaned, voice)
    try:
        import edge_tts  # type: ignore

        async def _save() -> None:
            communicate = edge_tts.Communicate(cleaned, selected_voice)
            await communicate.save(out_file)

        _run_coro_sync(_save())
        size_bytes = os.path.getsize(out_file) if os.path.isfile(out_file) else 0
        return json.dumps(
            {
                "status": "ok",
                "output_path": out_file,
                "voice": selected_voice,
                "bytes": size_bytes,
                "text_preview": cleaned[:120],
            },
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "status": "error",
                "output_path": out_file,
                "voice": selected_voice,
                "error": f"edge-tts failed ({type(exc).__name__}: {exc}). Ensure edge-tts is installed.",
            },
            ensure_ascii=False,
        )


@_tool_decorator
def text_to_speech(
    text: str,
    output_path: str = "/tmp/colibri_tts.mp3",
    voice: str = "he-IL-HilaNeural",
) -> str:
    """Synthesize natural Hebrew or English speech audio (MP3) using Edge-TTS (`he-IL-HilaNeural` default)."""
    return _synthesize_edge_tts(text=text, output_path=output_path, voice=voice)


@_tool_decorator
def speak_text(
    text: str,
    output_path: str = "/tmp/colibri_tts.mp3",
    voice: str = "he-IL-HilaNeural",
) -> str:
    """Speak text aloud by generating an MP3 audio file with Edge-TTS (`he-IL-HilaNeural` default)."""
    return _synthesize_edge_tts(text=text, output_path=output_path, voice=voice)


_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.?!;:\n—])\s+|(?<=,)\s+")
_MARKDOWN_CLEAN_RE = re.compile(r"[*`#_~]+|\[([^\]]+)\]\([^)]+\)")


def _clean_for_speech(text: str) -> str:
    """Strip markdown formatting and raw URLs so Edge-TTS reads natural speech."""
    cleaned = _MARKDOWN_CLEAN_RE.sub(r"\1", text or "")
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _play_audio_file_blocking(filepath: str) -> None:
    """Play an synthesized MP3 file locally using macOS `afplay` or cross-platform `ffplay`/`mpg123`."""
    if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
        return
    players = [["afplay", filepath]] if sys.platform == "darwin" else []
    players.extend(
        [
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath],
            ["mpg123", "-q", filepath],
        ]
    )
    for cmd in players:
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30.0,
                check=False,
            )
            return
        except Exception:  # noqa: BLE001
            continue


class SentenceTTSStreamer:
    """Clause- & sentence-level streaming TTS queue for low-latency voice output.

    Chunks incoming LLM token deltas as soon as the first clause (4+ words ending in `,`, `—`, `:`)
    or sentence (`.`, `?`, `!`, `\\n`) completes, and synthesizes/plays each chunk in a background
    worker thread while the LLM continues generating subsequent tokens.
    """

    def __init__(
        self,
        voice: str = "auto",
        play_audio: bool = True,
        min_first_clause_words: int = 4,
    ) -> None:
        self.voice = voice
        self.play_audio = play_audio
        self.min_first_clause_words = min_first_clause_words
        self._buffer = ""
        self._first_chunk_emitted = False
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self.synthesized_chunks: List[str] = []
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def feed(self, token_delta: str) -> List[str]:
        """Feed an incoming token delta and enqueue any completed clause/sentence for immediate TTS."""
        if not token_delta:
            return []
        self._buffer += token_delta
        emitted: List[str] = []

        while True:
            # Before first audio starts, split eagerly on comma/colon/dash after >=4 words
            # After first audio starts, split on sentence boundaries (. ? ! ; \n)
            pattern = (
                r"(.+?[.?!;:\n—,]|\S.+?\n)\s+"
                if not self._first_chunk_emitted
                else r"(.+?[.?!;\n])\s+"
            )
            match = re.match(pattern, self._buffer, flags=re.DOTALL)
            if not match:
                break
            candidate = match.group(1).strip()
            cleaned = _clean_for_speech(candidate)
            word_count = len(cleaned.split())
            if not self._first_chunk_emitted and word_count < self.min_first_clause_words and not re.search(r"[.?!]$", candidate):
                break
            self._buffer = self._buffer[match.end() :]
            if cleaned:
                self._first_chunk_emitted = True
                emitted.append(cleaned)
                self._queue.put(cleaned)
        return emitted

    def flush(self) -> List[str]:
        """Flush any remaining buffered tokens to the TTS worker and wait for completion."""
        emitted: List[str] = []
        tail = _clean_for_speech(self._buffer)
        self._buffer = ""
        if tail:
            emitted.append(tail)
            self._queue.put(tail)
        self._queue.put(None)
        self._worker.join(timeout=30.0)
        return emitted

    def _run_worker(self) -> None:
        idx = 0
        while True:
            chunk = self._queue.get()
            if chunk is None:
                break
            idx += 1
            out_path = os.path.join(tempfile.gettempdir(), f"colibri_stream_{os.getpid()}_{idx}.mp3")
            try:
                res_raw = _synthesize_edge_tts(text=chunk, output_path=out_path, voice=self.voice)
                res = json.loads(res_raw)
                if res.get("status") == "ok":
                    self.synthesized_chunks.append(out_path)
                    if self.play_audio:
                        _play_audio_file_blocking(out_path)
            except Exception:  # noqa: BLE001
                pass
            finally:
                if self.play_audio and os.path.isfile(out_path):
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass


AUDIO_TOOLS: List[Any] = [
    transcribe_audio,
    text_to_speech,
    speak_text,
]

AUDIO_TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "transcribe_audio": transcribe_audio,
    "text_to_speech": text_to_speech,
    "speak_text": speak_text,
}

AUDIO_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe a spoken audio file (Hebrew or English) into text using local OpenAI Whisper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Local path to the recorded WAV/MP3/M4A audio file.",
                    },
                    "model_size": {
                        "type": "string",
                        "description": "Whisper model size ('tiny', 'base', 'small', 'medium', 'large').",
                        "default": "base",
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional ISO language hint such as 'he' or 'en'.",
                    },
                },
                "required": ["audio_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_to_speech",
            "description": "Synthesize Hebrew or English speech to an MP3 file using Edge-TTS neural voices (default 'he-IL-HilaNeural').",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text in Hebrew or English to speak."},
                    "output_path": {
                        "type": "string",
                        "description": "Target MP3 file path.",
                        "default": "/tmp/colibri_tts.mp3",
                    },
                    "voice": {
                        "type": "string",
                        "description": "Edge-TTS voice name (e.g. 'he-IL-HilaNeural', 'he-IL-AvriNeural', 'en-US-AriaNeural', or 'auto').",
                        "default": "he-IL-HilaNeural",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak_text",
            "description": "Speak text aloud by generating an MP3 audio file with Edge-TTS (`he-IL-HilaNeural` default).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text in Hebrew or English to speak."},
                    "output_path": {
                        "type": "string",
                        "description": "Target MP3 file path.",
                        "default": "/tmp/colibri_tts.mp3",
                    },
                    "voice": {
                        "type": "string",
                        "description": "Edge-TTS voice name (default 'he-IL-HilaNeural').",
                        "default": "he-IL-HilaNeural",
                    },
                },
                "required": ["text"],
            },
        },
    },
]

__all__ = [
    "transcribe_audio",
    "text_to_speech",
    "speak_text",
    "SentenceTTSStreamer",
    "AUDIO_TOOLS",
    "AUDIO_TOOL_REGISTRY",
    "AUDIO_TOOLS_SCHEMA",
    "_WHISPER_CACHE",
    "_select_voice",
]
