"""CLI entry point for the Colibri Qwen3.6-35B-A3B Desktop Holographic AI Companion (`src/ai/cli.py`).

Supports:
- Single-query research & companion mode:
    python3 -m src.ai.cli "What are the latest breakthroughs in..."
- Interactive multi-turn chat & holographic companion mode:
    python3 -m src.ai.cli
  with slash commands:
    /tools                             List all bound internet, vision, voice, memory, code & device tools
    /slot N                            Switch Colibri KV cache_slot (0..15)
    /think on|off                      Toggle Qwen3.6 reasoning mode
    /decide <question> | <opt1> | ...  Closed-set System One probability decision
    /avatar [state] [emotion]          Show or set Pepper's Ghost 3D avatar state
    /rotate <angle>                    Rotate holographic cylinder servo (-180..180 deg)
    /birthday                          Trigger Birthday celebration hologram, confetti & bilingual greeting
    /memory [query]                    List saved personal facts or search `data/memory_db/`
    /voice <audio_path>                Transcribe spoken Hebrew/English audio via Whisper
    /speak <text>                      Synthesize speech via Edge-TTS (`he-IL-HilaNeural` / English)
    /ocr <image_path>                  Extract Hebrew/English text from an image via Tesseract OCR
    /clear                             Clear conversation history
    /status                            Show active runner, model, context window, KV slot & hardware state
- Gateway & tool pre-flight check:
    python3 -m src.ai.cli --check
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .config import MLX_BASE_URL, OLLAMA_BASE_URL, AgentConfig, detect_best_local_backend
from .graph import _load_tools_registry, has_langgraph, run_agent
from .persona import BIRTHDAY_GREETING_EN, BIRTHDAY_GREETING_HE, COMPANION_NAME
from .tools.device import (
    get_device_state,
    rotate_device,
    set_avatar_state,
    trigger_hologram_effect,
)


def _runner_label(force_stdlib: bool = False) -> str:
    if not force_stdlib and has_langgraph():
        return "LangGraph (StateGraph + ChatOpenAI)"
    return "Stdlib (StdlibColibriAgent StateGraph)"


def _print_banner(config: AgentConfig, force_stdlib: bool = False, stream: Any = None) -> None:
    out = stream or sys.stderr
    runner = _runner_label(force_stdlib=force_stdlib)
    think_str = "on" if config.enable_thinking else "off"
    dev = get_device_state()
    print(
        f"[colibri-agent] Companion: {COMPANION_NAME} | Runner: {runner} | Model: {config.model} "
        f"| Context: {config.context_length} tokens | Slot: {config.cache_slot} "
        f"| Thinking: {think_str} | Avatar: {dev.get('avatar_state', 'idle')} ({dev.get('servo_angle', 0)}°)",
        file=out,
    )


def _format_tool_trace(tc: Dict[str, Any], output: str, elapsed_s: float) -> str:
    name = tc.get("name", "tool")
    args = tc.get("args", {})
    if isinstance(args, str):
        args_str = args
    else:
        args_str = json.dumps(args, ensure_ascii=False)
    if len(args_str) > 120:
        args_str = args_str[:117] + "..."
    preview = output.replace("\n", " ").strip()
    if len(preview) > 140:
        preview = preview[:137] + "..."
    return f"  -> [tool:{name}] ({elapsed_s:.2f}s) args={args_str}\n     result: {preview}"


def _run_decide_command(raw_arg: str, config: AgentConfig) -> str:
    """Execute `/decide <question> | <opt1> | <opt2> ...` via Colibri `/v1/systemone`."""
    parts = [p.strip() for p in raw_arg.split("|") if p.strip()]
    if not parts:
        return "Usage: /decide <question> | <option1> | <option2> [...]"

    question = parts[0]
    options = parts[1:]

    try:
        from .systemone import SystemOneClient

        client = SystemOneClient(config=config)
        if not options:
            res_noul = client.noul(state=question, instructions=question)
            return (
                f"[System One / noul] P(yes)={res_noul.probability_yes:.4f} "
                f"(is_yes={res_noul.is_yes})"
            )
        res_choice = client.choose(
            state=question,
            instructions=question,
            criteria={opt: None for opt in options},
        )
        return (
            f"[System One / choice] choice={res_choice.choice!r} "
            f"(confidence={res_choice.confidence:.4f})\n"
            f"  probabilities={json.dumps(res_choice.probabilities, ensure_ascii=False)}"
        )
    except Exception:  # noqa: BLE001
        pass

    url = f"{config.base_url.rstrip('/')}/systemone"
    if not options:
        q_spec: Dict[str, Any] = {"type": "noul", "instructions": question}
    else:
        q_spec = {
            "type": "choice",
            "instructions": question,
            "criteria": {opt: None for opt in options},
        }
    body: Dict[str, Any] = {
        "state": question,
        "model": config.model,
        "cache_slot": config.cache_slot,
        "questions": {"decision": q_spec},
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=config.request_timeout) as resp:
            reply = json.loads(resp.read().decode("utf-8", errors="replace"))
        ans = (reply.get("answers") or {}).get("decision") or reply
        return f"[System One] {json.dumps(ans, indent=2, ensure_ascii=False)}"
    except Exception as exc:  # noqa: BLE001
        return f"[System One error] Could not reach {url}: {exc}"


def _list_tools_text() -> str:
    registry, schemas, _ = _load_tools_registry()
    if schemas:
        lines = [f"Bound Companion & Research Tools ({len(schemas)}):"]
        for item in schemas:
            fn = item.get("function") or item
            name = fn.get("name", "unknown")
            desc = (
                (fn.get("description") or "").strip().splitlines()[0]
                if fn.get("description")
                else ""
            )
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)
    if registry:
        lines = [f"Bound Companion & Research Tools ({len(registry)}):"]
        for name, fn in sorted(registry.items()):
            doc = (getattr(fn, "__doc__", "") or "").strip().splitlines()
            summary = doc[0] if doc else ""
            lines.append(f"  - {name}: {summary}")
        return "\n".join(lines)
    return "No tools currently registered in src.ai.tools."


def _format_status(config: AgentConfig, history: List[Dict[str, Any]], force_stdlib: bool) -> str:
    registry, schemas, _ = _load_tools_registry(fast_mode=getattr(config, "fast_tool_pruning", False))
    tool_count = len(schemas) or len(registry)
    dev = get_device_state()
    return (
        f"Colibri Holographic Companion Status ({COMPANION_NAME}):\n"
        f"  Runner            : {_runner_label(force_stdlib=force_stdlib)}\n"
        f"  Gateway Base URL  : {config.base_url}\n"
        f"  Model             : {config.model}\n"
        f"  Context Window    : {config.context_length} tokens\n"
        f"  KV Cache Slot     : {config.cache_slot} (research_slot={getattr(config, 'research_cache_slot', 1)})\n"
        f"  Thinking Mode     : {'on' if config.enable_thinking else 'off'} "
        f"(preserve_thinking={config.preserve_thinking if config.preserve_thinking is not None else True})\n"
        f"  Context Compaction: {config.compact}\n"
        f"  Fast Tool Pruning : {'on' if getattr(config, 'fast_tool_pruning', False) else 'off'} ({tool_count} active schemas / {len(registry)} callable)\n"
        f"  Memory Prefetch   : {'on' if getattr(config, 'prefetch_memory', True) else 'off'}\n"
        f"  Streaming TTS     : {'on' if getattr(config, 'stream_tts', False) else 'off'}\n"
        f"  Max Tool Chars    : {config.max_tool_chars}\n"
        f"  Max Step Budget   : {config.max_iterations}\n"
        f"  Bound Tools       : {tool_count}\n"
        f"  Avatar State      : {dev.get('avatar_state', 'idle')} (emotion={dev.get('emotion', 'warm')})\n"
        f"  Hologram Effect   : {dev.get('hologram_effect', 'ambient_pulse')} (LED={dev.get('led_rgb', '255,180,220')})\n"
        f"  Servo Angle       : {dev.get('servo_angle', 0)}°\n"
        f"  History Messages  : {len(history)}"
    )


def _interactive_loop(config: AgentConfig, force_stdlib: bool = False) -> int:
    _print_banner(config, force_stdlib=force_stdlib, stream=sys.stdout)
    print(
        "Type a message or command (/tools, /slot N, /think on|off, /decide, "
        "/avatar, /rotate <deg>, /birthday, /memory, /voice <path>, /speak <text>, /ocr <path>, /clear, /status, /quit).",
        flush=True,
    )
    history: List[Dict[str, Any]] = []

    while True:
        try:
            line = input("\ncolibri> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not line:
            continue

        if line in ("/quit", "/exit"):
            return 0

        if line == "/help":
            print(
                "Commands:\n"
                "  /tools                             List all bound companion & internet tools\n"
                "  /slot N                            Switch Colibri KV cache slot (0..15)\n"
                "  /think on|off                      Toggle Qwen3.6 thinking mode\n"
                "  /decide <question> | <opt1> | ...  Run System One closed-set decision\n"
                "  /avatar [state] [emotion]          Inspect or set 3D hologram avatar state\n"
                "  /rotate <angle>                    Rotate cylinder servo (-180..180 degrees)\n"
                "  /birthday                          Trigger Birthday celebration mode & greeting\n"
                "  /memory [query]                    List saved facts or query local memory\n"
                "  /voice <audio_path>                Transcribe audio file via Whisper\n"
                "  /speak <text>                      Synthesize speech via Edge-TTS\n"
                "  /ocr <image_path>                  Extract text from image via Tesseract OCR\n"
                "  /clear                             Reset conversation history\n"
                "  /status                            Show runner, model, context & hardware status\n"
                "  /quit                              Exit interactive session"
            )
            continue

        if line == "/tools":
            print(_list_tools_text())
            continue

        if line == "/clear":
            history.clear()
            print("[colibri-agent] Conversation history cleared.")
            continue

        if line == "/status":
            print(_format_status(config, history, force_stdlib))
            continue

        if line.startswith("/slot"):
            arg = line[len("/slot") :].strip()
            try:
                slot = int(arg)
                if not 0 <= slot < 16:
                    raise ValueError("slot must be between 0 and 15")
                config.cache_slot = slot
                print(f"[colibri-agent] Switched KV cache_slot to {config.cache_slot}.")
            except ValueError:
                print("Usage: /slot N  (where N is an integer 0..15)")
            continue

        if line.startswith("/think"):
            arg = line[len("/think") :].strip().lower()
            if arg in ("on", "true", "1"):
                config.enable_thinking = True
                print("[colibri-agent] Thinking mode enabled (enable_thinking=True).")
            elif arg in ("off", "false", "0"):
                config.enable_thinking = False
                print("[colibri-agent] Thinking mode disabled (enable_thinking=False).")
            else:
                print("Usage: /think on|off")
            continue

        if line.startswith("/decide"):
            arg = line[len("/decide") :].strip()
            print(_run_decide_command(arg, config))
            continue

        if line.startswith("/avatar"):
            arg = line[len("/avatar") :].strip()
            if not arg:
                print(json.dumps(get_device_state(), indent=2, ensure_ascii=False))
            else:
                parts = arg.split(maxsplit=2)
                st = parts[0]
                em = parts[1] if len(parts) > 1 else "warm"
                sub = parts[2] if len(parts) > 2 else ""
                print(set_avatar_state(state=st, emotion=em, subtitle=sub))
            continue

        if line.startswith("/rotate"):
            arg = line[len("/rotate") :].strip()
            try:
                angle = int(round(float(arg)))
                print(rotate_device(angle))
            except ValueError:
                print("Usage: /rotate <angle_in_degrees> (e.g. /rotate -30 or /rotate 45)")
            continue

        if line == "/birthday":
            set_avatar_state(state="birthday", emotion="joyful", subtitle="יום הולדת שמח! 🎂")
            trigger_hologram_effect(effect="birthday_confetti", led_rgb="255,140,200")
            print(f"\n🎂 {BIRTHDAY_GREETING_HE}\n\n✨ {BIRTHDAY_GREETING_EN}")
            continue

        if line.startswith("/memory"):
            arg = line[len("/memory") :].strip()
            from .tools.rag import list_saved_facts, query_memory

            if not arg:
                print(list_saved_facts())
            else:
                print(query_memory(query=arg))
            continue

        if line.startswith("/voice"):
            arg = line[len("/voice") :].strip()
            if not arg:
                print("Usage: /voice <audio_file_path>")
            else:
                from .tools.audio import transcribe_audio

                print(transcribe_audio(audio_path=arg))
            continue

        if line.startswith("/speak"):
            arg = line[len("/speak") :].strip()
            if not arg:
                print("Usage: /speak <text_in_hebrew_or_english>")
            else:
                from .tools.audio import text_to_speech

                print(text_to_speech(text=arg))
            continue

        if line.startswith("/ocr"):
            arg = line[len("/ocr") :].strip()
            if not arg:
                print("Usage: /ocr <image_file_path>")
            else:
                from .tools.vision import extract_text_from_image

                print(extract_text_from_image(image_path=arg))
            continue

        if line.startswith("/fast"):
            arg = line[len("/fast") :].strip().lower()
            if arg in ("on", "true", "1", ""):
                config.fast_tool_pruning = True
                config.enable_thinking = False
                config.max_tokens = 256
                print(
                    "[colibri-agent] Fast Companion Mode enabled (9 core tools on slot 0, enable_thinking=False, max_tokens=256)."
                )
            elif arg in ("off", "false", "0"):
                config.fast_tool_pruning = False
                print("[colibri-agent] Fast Companion Mode disabled (all deduplicated tools bound).")
            else:
                print("Usage: /fast [on|off]")
            continue

        if line.startswith("/tts"):
            arg = line[len("/tts") :].strip().lower()
            if arg in ("on", "true", "1"):
                config.stream_tts = True
                print("[colibri-agent] Clause-level streaming TTS enabled (stream_tts=True).")
            elif arg in ("off", "false", "0"):
                config.stream_tts = False
                print("[colibri-agent] Clause-level streaming TTS disabled (stream_tts=False).")
            else:
                print("Usage: /tts on|off")
            continue

        def _trace_cb(tc: Dict[str, Any], output: str, elapsed: float) -> None:
            print(_format_tool_trace(tc, output, elapsed), flush=True)

        try:
            print()
            result = run_agent(
                line,
                config=config,
                stream=True,
                history=history,
                on_tool_call=_trace_cb,
                force_stdlib=force_stdlib,
            )
            history.append({"role": "user", "content": line})
            history.append({"role": "assistant", "content": result.answer})
        except Exception as exc:  # noqa: BLE001
            print(f"[colibri-agent error] {exc}", file=sys.stderr)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coli-agent",
        description="Colibri Qwen3.6-35B-A3B Desktop Holographic AI Companion & Research Agent CLI",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Question or instruction for single-query mode (omit for interactive chat)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Colibri gateway OpenAI base URL (default: COLIBRI_BASE_URL or http://127.0.0.1:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Colibri API key if COLI_API_KEY is enabled",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (default: auto-detected or qwen36-35b)",
    )
    parser.add_argument(
        "--slot",
        type=int,
        default=None,
        help="Colibri KV cache_slot (0..15, default: 0)",
    )
    parser.add_argument(
        "--think",
        choices=["on", "off"],
        default=None,
        help="Enable or disable Qwen3.6 thinking mode",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum tool-calling iterations before synthesis (default: 8)",
    )
    parser.add_argument(
        "--stdlib",
        action="store_true",
        help="Force zero-dependency StdlibColibriAgent runner even if LangGraph is installed",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run configuration and tool pre-flight check and exit",
    )
    parser.add_argument(
        "--birthday",
        action="store_true",
        help="Trigger Birthday celebration mode on the 3D hologram and print bilingual greetings",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable Fast Companion Mode (9 core tools on slot 0, thinking off, max_tokens=256)",
    )
    parser.add_argument(
        "--stream-tts",
        action="store_true",
        help="Stream generated sentences/clauses immediately to Edge-TTS (he-IL-HilaNeural)",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Target local Apple Metal Ollama server (http://127.0.0.1:11434/v1)",
    )
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Target local Apple Metal MLX server (http://127.0.0.1:8080/v1)",
    )
    parser.add_argument(
        "--auto-backend",
        action="store_true",
        help="Auto-probe Colibri C gateway, Ollama Metal, and MLX Metal for the best active server",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON result in single-query mode",
    )

    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.ollama:
        overrides["base_url"] = OLLAMA_BASE_URL
    elif args.mlx:
        overrides["base_url"] = MLX_BASE_URL
    elif args.auto_backend:
        best_url, best_model, best_ctx = detect_best_local_backend()
        overrides["base_url"] = best_url
        overrides["model"] = best_model
        overrides["context_length"] = best_ctx
    if args.base_url is not None:
        overrides["base_url"] = args.base_url
    if args.api_key is not None:
        overrides["api_key"] = args.api_key
    if args.model is not None:
        overrides["model"] = args.model
    if args.slot is not None:
        overrides["cache_slot"] = args.slot
    if args.think is not None:
        overrides["enable_thinking"] = args.think == "on"
    if args.max_steps is not None:
        overrides["max_iterations"] = args.max_steps
    if args.fast:
        overrides["fast_tool_pruning"] = True
        overrides["enable_thinking"] = False
        overrides["max_tokens"] = 256
    if args.stream_tts:
        overrides["stream_tts"] = True

    config = AgentConfig.from_env(**overrides)

    if args.check:
        print(_format_status(config, [], args.stdlib))
        print(_list_tools_text())
        return 0

    if args.birthday:
        set_avatar_state(state="birthday", emotion="joyful", subtitle="יום הולדת שמח! 🎂")
        trigger_hologram_effect(effect="birthday_confetti", led_rgb="255,140,200")
        print(f"🎂 {BIRTHDAY_GREETING_HE}\n\n✨ {BIRTHDAY_GREETING_EN}")
        return 0

    query_str = " ".join(args.query).strip()
    if not query_str:
        return _interactive_loop(config, force_stdlib=args.stdlib)

    _print_banner(config, force_stdlib=args.stdlib, stream=sys.stderr)

    def _trace_cb(tc: Dict[str, Any], output: str, elapsed: float) -> None:
        print(_format_tool_trace(tc, output, elapsed), file=sys.stderr, flush=True)

    try:
        result = run_agent(
            query_str,
            config=config,
            on_tool_call=_trace_cb,
            force_stdlib=args.stdlib,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[colibri-agent error] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "answer": result.answer,
                    "runner": result.runner,
                    "model": result.model,
                    "steps": result.steps,
                    "tool_calls": result.tool_calls,
                    "device_state": get_device_state(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
