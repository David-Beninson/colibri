"""Self-contained unit test suite for the Colibri Desktop Holographic AI Companion (`src/ai/test_agent.py`).

Verifies:
1. Unified 6-domain tool registration (`internet`, `vision`, `audio`/`voice`, `rag`, `code_runner`, `device`)
2. Bilingual Persona & deterministic system prompt (`persona.py`: `COMPANION_PERSONA_PROMPT`, `build_system_prompt`)
3. Local RAG Memory persistence (`save_fact`, `query_memory`, `save_conversation_fact`, `query_local_knowledge` in `data/memory_db/`)
4. Hardware & Pepper's Ghost 3D Avatar hooks (`rotate_device`, `rotate_device_servo`, `set_avatar_state`, `trigger_hologram_effect`)
5. Safe AST math (`calculate`) & sandboxed Python code execution (`run_python_calculation`)
6. Vision (`extract_text_from_image`, `detect_person_angle`) & Voice (`transcribe_audio`, `text_to_speech`, Hebrew/English voice routing)
7. Strict SSRF blocking (`127.0.0.1`, `169.254.169.254`, `192.168.1.1`) & HTML readable extraction
8. StateGraph transitions (`agent -> tools -> reflect_and_guard -> end`) with Qwen3.6 prefix flags & `SystemOneClient`
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.request
from typing import Any, Dict, List
from unittest.mock import patch

from src.ai.config import AgentConfig, detect_colibri_model
from src.ai.graph import (
    SYSTEM_PROMPT,
    StdlibColibriAgent,
    build_internet_agent_graph,
    get_langchain_tools,
    run_agent,
)
from src.ai.persona import (
    BIRTHDAY_GREETING_EN,
    BIRTHDAY_GREETING_HE,
    COMPANION_PERSONA_PROMPT,
    build_system_prompt,
)
from src.ai.systemone import SystemOneClient
from src.ai.tools import (
    OPENAI_TOOLS_SCHEMA,
    TOOL_REGISTRY,
    calculate,
    detect_person_angle,
    extract_text_from_image,
    fetch_json_or_rss,
    fetch_webpage,
    get_device_state,
    list_saved_facts,
    load_all_tool_suites,
    query_local_knowledge,
    query_memory,
    rotate_device,
    rotate_device_servo,
    run_python_calculation,
    save_conversation_fact,
    save_fact,
    set_avatar_state,
    speak_text,
    text_to_speech,
    transcribe_audio,
    trigger_avatar_state,
    trigger_hologram_effect,
)
from src.ai.tools.audio import _select_voice
from src.ai.tools.internet import _LinkAndTextExtractor, _public_url


class _MockHTTPResponse(io.BytesIO):
    """In-memory `urllib.request.urlopen` response object compatible with context managers."""

    def __init__(self, payload: Dict[str, Any], status: int = 200, url: str = "") -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        super().__init__(raw)
        self.status = status
        self._url = url
        self.headers = {"Content-Type": "application/json"}

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "_MockHTTPResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class _MockColibriGateway:
    """In-memory mock of Colibri `/v1/models`, `/v1/chat/completions`, and `/v1/systemone`."""

    def __init__(self) -> None:
        self.recorded_requests: List[Dict[str, Any]] = []
        self.mode: str = "tool_then_answer"

    def urlopen(self, req: Any, timeout: float = 10.0) -> _MockHTTPResponse:
        url = req.full_url if hasattr(req, "full_url") else str(req)
        method = req.get_method() if hasattr(req, "get_method") else "GET"
        body: Dict[str, Any] = {}
        if getattr(req, "data", None):
            body = json.loads(req.data.decode("utf-8", errors="replace"))
        self.recorded_requests.append({"method": method, "url": url, "body": body})

        if url.endswith("/v1/models") or url.endswith("/models"):
            return _MockHTTPResponse(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "qwen36-35b",
                            "object": "model",
                            "context_length": 65536,
                            "max_model_len": 65536,
                        }
                    ],
                },
                url=url,
            )

        if url.endswith("/systemone"):
            return _MockHTTPResponse(
                {
                    "id": "req_mock_sys1",
                    "model": "qwen36-35b",
                    "provider": "colibri",
                    "answers": {
                        "q": {
                            "type": "choice",
                            "choice": "web_research",
                            "probabilities": {
                                "web_research": 0.91,
                                "direct": 0.09,
                            },
                            "confidence": 0.85,
                            "noul": 0.88,
                        }
                    },
                },
                url=url,
            )

        if "/servo/rotate" in url or "/avatar/state" in url or "/hologram/effect" in url:
            return _MockHTTPResponse({"status": "ok", "bridge": "mock_rpi"}, url=url)

        if url.endswith("/chat/completions"):
            messages = body.get("messages") or []
            has_tool_msg = any(m.get("role") == "tool" for m in messages)

            if self.mode == "infinite_tools":
                if not body.get("tools"):
                    return _MockHTTPResponse(
                        {
                            "id": "chatcmpl-final-forced",
                            "object": "chat.completion",
                            "model": "qwen36-35b",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Synthesized answer after hitting max_steps guard: 144.",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                        },
                        url=url,
                    )
                return _MockHTTPResponse(
                    {
                        "id": "chatcmpl-loop",
                        "object": "chat.completion",
                        "model": "qwen36-35b",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "reasoning_content": "Computing intermediate step.",
                                    "tool_calls": [
                                        {
                                            "id": "call_loop_1",
                                            "type": "function",
                                            "function": {
                                                "name": "calculate",
                                                "arguments": json.dumps(
                                                    {"expression": "12 * 12"}
                                                ),
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    },
                    url=url,
                )

            if not has_tool_msg:
                return _MockHTTPResponse(
                    {
                        "id": "chatcmpl-turn1",
                        "object": "chat.completion",
                        "model": "qwen36-35b",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "reasoning_content": "Let me calculate sqrt(144) + 25 * 4 using calculate.",
                                    "tool_calls": [
                                        {
                                            "id": "call_calc_1",
                                            "type": "function",
                                            "function": {
                                                "name": "calculate",
                                                "arguments": json.dumps(
                                                    {"expression": "sqrt(144) + 25 * 4"}
                                                ),
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    },
                    url=url,
                )

            tool_outputs = [
                m.get("content", "") for m in messages if m.get("role") == "tool"
            ]
            return _MockHTTPResponse(
                {
                    "id": "chatcmpl-turn2",
                    "object": "chat.completion",
                    "model": "qwen36-35b",
                    "compaction": {"summarized_messages": 0, "summary": ""},
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"Based on the tool calculation ({tool_outputs[-1]}), the exact result is 112.",
                                "reasoning_content": "Verified tool output.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
                url=url,
            )

        return _MockHTTPResponse({"error": "not found"}, status=404, url=url)


class ColibriAgentTestSuite(unittest.TestCase):
    """Comprehensive unit tests for all 6 companion tool domains, persona, SSRF, and StateGraph."""

    def setUp(self) -> None:
        self.gateway = _MockColibriGateway()
        self.base_url = "http://127.0.0.1:8000/v1"

    def test_all_six_tool_suites_registered(self) -> None:
        registry, schemas, tools_list = load_all_tool_suites()
        expected_core_tools = {
            # Internet suite
            "web_search",
            "fetch_webpage",
            "fetch_webpage_content",
            "deep_web_research",
            "wikipedia_search",
            "arxiv_search",
            "hackernews_search",
            "github_repo_search",
            "weather_lookup",
            "fetch_json_or_rss",
            "calculate",
            # Vision suite
            "extract_text_from_image",
            "describe_image_scene",
            "detect_person_angle",
            # Voice / Audio suite
            "transcribe_audio",
            "text_to_speech",
            "speak_text",
            # Private Local RAG suite
            "query_memory",
            "save_fact",
            "query_local_knowledge",
            "save_conversation_fact",
            "list_saved_facts",
            # Code runner suite
            "run_python_calculation",
            # Device & Pepper's Ghost 3D Avatar suite
            "rotate_device",
            "rotate_device_servo",
            "set_avatar_state",
            "trigger_avatar_state",
            "trigger_hologram_effect",
        }
        self.assertTrue(expected_core_tools.issubset(set(registry.keys())))
        schema_names = {
            item["function"]["name"] for item in schemas if "function" in item
        }
        self.assertTrue(expected_core_tools.issubset(schema_names))
        lc_tools = get_langchain_tools()
        self.assertGreaterEqual(len(lc_tools), len(expected_core_tools))

    def test_persona_and_bilingual_greetings(self) -> None:
        prompt = build_system_prompt("companion")
        self.assertIn("Noga", prompt)
        self.assertIn("he-IL-HilaNeural", prompt)
        self.assertEqual(prompt, COMPANION_PERSONA_PROMPT)
        self.assertIn("יום הולדת שמח", BIRTHDAY_GREETING_HE)
        self.assertIn("Happy Birthday", BIRTHDAY_GREETING_EN)

    def test_rag_memory_persistence_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"COLIBRI_MEMORY_DIR": tmpdir}):
                save_res = json.loads(
                    save_fact(
                        "Her favorite birthday song is Halelu by Ehud Banai and she loves iced matcha latte.",
                        category="music_and_drinks",
                    )
                )
                self.assertEqual(save_res.get("status"), "saved")
                self.assertTrue(os.path.isfile(os.path.join(tmpdir, "facts.json")))

                query_res = json.loads(query_memory("favorite birthday song matcha", k=3))
                matches = query_res.get("matches") or []
                self.assertGreaterEqual(len(matches), 1)
                self.assertIn("Ehud Banai", matches[0].get("value", ""))

                # Also verify keyed save_conversation_fact + query_local_knowledge + list_saved_facts
                save_conversation_fact(
                    key="anniversary_flower",
                    value="Pink peonies and white orchids",
                )
                local_res = json.loads(query_local_knowledge("peonies orchids"))
                self.assertTrue(
                    any("peonies" in str(m.get("value", "")).lower() for m in local_res.get("matches", []))
                )
                listed = json.loads(list_saved_facts())
                self.assertEqual(listed.get("count"), 2)

    def test_device_servo_and_hologram_avatar_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "device_state.json")
            with patch.dict(os.environ, {"COLIBRI_DEVICE_STATE_PATH": state_file}):
                # Test angle clamping [-180, 180]
                rot = json.loads(rotate_device(240))
                self.assertEqual(rot["status"], "ok")
                self.assertEqual(rot["servo_angle"], 180)
                self.assertTrue(rot["clamped"])

                rot2 = json.loads(rotate_device_servo(-45))
                self.assertEqual(rot2["servo_angle"], -45)
                self.assertFalse(rot2["clamped"])

                # Test avatar state validation & update
                av = json.loads(
                    set_avatar_state(
                        state="birthday",
                        emotion="joyful",
                        subtitle="יום הולדת שמח אהובה!",
                    )
                )
                self.assertEqual(av["status"], "ok")
                self.assertEqual(av["avatar_state"], "birthday")

                av_alias = json.loads(trigger_avatar_state(state="love", emotion="affectionate"))
                self.assertEqual(av_alias["avatar_state"], "love")

                av_bad = json.loads(set_avatar_state(state="invalid_state_xyz"))
                self.assertEqual(av_bad["status"], "error")

                # Test hologram visual effect
                eff = json.loads(
                    trigger_hologram_effect(
                        effect="floating_hearts",
                        led_rgb="255,105,180",
                    )
                )
                self.assertEqual(eff["status"], "ok")
                self.assertEqual(eff["hologram_effect"], "floating_hearts")

                persisted = get_device_state()
                self.assertEqual(persisted["servo_angle"], -45)
                self.assertEqual(persisted["avatar_state"], "love")
                self.assertEqual(persisted["hologram_effect"], "floating_hearts")

    def test_code_runner_and_calculate_sandboxes(self) -> None:
        res = json.loads(calculate("sqrt(144) + 2 ** 5"))
        self.assertEqual(res["result"], 44)

        py_ok = json.loads(
            run_python_calculation("import math\nresult = sum(i * i for i in range(1, 6))\nprint(result)")
        )
        self.assertIn("55", str(py_ok))

        py_blocked = json.loads(
            run_python_calculation("import subprocess\nsubprocess.run(['ls'])")
        )
        self.assertIn("error", str(py_blocked).lower())

    def test_vision_and_voice_graceful_handling(self) -> None:
        # Bilingual voice selection
        self.assertEqual(_select_voice("שלום אהובה, יום הולדת שמח!", "auto"), "he-IL-HilaNeural")
        self.assertEqual(_select_voice("Happy birthday my love!", "auto"), "en-US-AriaNeural")

        # Missing file checks return structured JSON errors without raising unhandled exceptions
        ocr_missing = json.loads(extract_text_from_image("/nonexistent/image_12345.png"))
        self.assertIn("error", ocr_missing)

        angle_missing = json.loads(detect_person_angle("/nonexistent/frame_12345.jpg"))
        self.assertIn("error", angle_missing)

        stt_missing = json.loads(transcribe_audio("/nonexistent/audio_12345.wav"))
        self.assertIn("error", stt_missing)

        tts_empty = json.loads(text_to_speech(""))
        self.assertIn("error", tts_empty)
        self.assertTrue(callable(speak_text))

    def test_ssrf_blocking_private_and_loopback_ips(self) -> None:
        blocked_targets = [
            "http://127.0.0.1:8000/v1/models",
            "http://169.254.169.254/latest/meta-data/",
            "http://192.168.1.1/admin",
            "file:///etc/passwd",
            "ftp://example.com/resource",
        ]
        for target in blocked_targets:
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    _public_url(target)

        page_out = fetch_webpage("http://127.0.0.1:8000/v1/models")
        self.assertTrue("error" in page_out.lower() or "not a public" in page_out.lower())

        feed_out = fetch_json_or_rss("http://169.254.169.254/latest/meta-data/")
        self.assertTrue("error" in feed_out.lower() or "not a public" in feed_out.lower())

    def test_html_readable_and_link_extraction(self) -> None:
        sample_html = """
        <html>
          <head>
            <title>Qwen3.6 Hybrid DeltaNet Architecture</title>
            <script>console.log("ignore secret script");</script>
            <style>body { color: red; }</style>
          </head>
          <body>
            <nav><a href="/nav">Skip Nav</a></nav>
            <article>
              <h1>Qwen3.6-35B-A3B Benchmarks</h1>
              <p>75% Gated DeltaNet layers and 25% Gated Attention layers.</p>
              <a href="https://huggingface.co/Qwen/Qwen3.6-35B-A3B">Official Model Card</a>
              <a href="/docs/qwen36">Local Docs</a>
            </article>
          </body>
        </html>
        """
        parser = _LinkAndTextExtractor("https://example.org/research/post")
        parser.feed(sample_html)
        text = " ".join(parser.text_parts)
        self.assertEqual(parser.title.strip(), "Qwen3.6 Hybrid DeltaNet Architecture")
        self.assertIn("75% Gated DeltaNet layers", text)
        self.assertNotIn("ignore secret script", text)
        self.assertEqual(
            parser.links,
            [
                ("Official Model Card", "https://huggingface.co/Qwen/Qwen3.6-35B-A3B"),
                ("Local Docs", "https://example.org/docs/qwen36"),
            ],
        )

    def test_graph_state_transitions_and_qwen36_prefix_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "device_state.json")
            with patch.dict(os.environ, {"COLIBRI_DEVICE_STATE_PATH": state_file}):
                with patch.object(urllib.request, "urlopen", side_effect=self.gateway.urlopen):
                    model_id, ctx_len = detect_colibri_model(self.base_url, api_key="test-key")
                    self.assertEqual(model_id, "qwen36-35b")
                    self.assertEqual(ctx_len, 65536)

                    cfg = AgentConfig(
                        base_url=self.base_url,
                        api_key="test-key",
                        model="qwen36-35b",
                        cache_slot=3,
                        enable_thinking=True,
                        compact="auto",
                        max_tool_chars=500,
                        max_iterations=4,
                    )

                    traces: List[Dict[str, Any]] = []
                    result = run_agent(
                        "What is sqrt(144) + 25 * 4?",
                        config=cfg,
                        on_tool_call=lambda tc, out, el: traces.append({"tc": tc, "out": out}),
                        force_stdlib=True,
                    )

                    self.assertIn("112", result.answer)
                    self.assertEqual(result.steps, 1)
                    self.assertEqual(len(result.tool_calls), 1)
                    self.assertEqual(result.tool_calls[0]["name"], "calculate")
                    self.assertEqual(len(traces), 1)

                    chat_reqs = [
                        r["body"]
                        for r in self.gateway.recorded_requests
                        if r["url"].endswith("/chat/completions")
                    ]
                    self.assertEqual(len(chat_reqs), 2)
                    for req_body in chat_reqs:
                        self.assertEqual(req_body.get("cache_slot"), 3)
                        self.assertEqual(req_body.get("context_compaction"), "auto")
                        self.assertIs(req_body.get("preserve_thinking"), True)
                        self.assertEqual(req_body["messages"][0]["role"], "system")
                        self.assertEqual(req_body["messages"][0]["content"], SYSTEM_PROMPT)

                    second_turn_msgs = chat_reqs[1]["messages"]
                    assistant_turns = [
                        m for m in second_turn_msgs if m.get("role") == "assistant"
                    ]
                    self.assertEqual(
                        assistant_turns[0].get("reasoning_content"),
                        "Let me calculate sqrt(144) + 25 * 4 using calculate.",
                    )

                    # Verify avatar pulsed to "speaking" on completion
                    self.assertEqual(get_device_state()["avatar_state"], "speaking")

    def test_reflect_and_guard_max_steps_limit(self) -> None:
        self.gateway.mode = "infinite_tools"
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "device_state.json")
            with patch.dict(os.environ, {"COLIBRI_DEVICE_STATE_PATH": state_file}):
                with patch.object(urllib.request, "urlopen", side_effect=self.gateway.urlopen):
                    cfg = AgentConfig(
                        base_url=self.base_url,
                        api_key="test-key",
                        model="qwen36-35b",
                        max_iterations=2,
                    )
                    agent = build_internet_agent_graph(config=cfg)
                    if not isinstance(agent, StdlibColibriAgent):
                        agent = StdlibColibriAgent(config=cfg)
                    res = agent.invoke("Keep calculating 12 * 12")
                    self.assertEqual(res["steps"], 2)
                    self.assertIn("Synthesized answer after hitting max_steps guard", res["answer"])

    def test_system_one_client_integration(self) -> None:
        with patch.object(urllib.request, "urlopen", side_effect=self.gateway.urlopen):
            cfg = AgentConfig(
                base_url=self.base_url, api_key="test-key", model="qwen36-35b"
            )
            sys1 = SystemOneClient(config=cfg)
            decision = sys1.route_query("Find the latest Qwen3.6 benchmarks on the web")
            self.assertEqual(decision.choice, "web_research")
            self.assertAlmostEqual(decision.confidence, 0.85, places=2)

    def test_optimizations_dedup_two_slot_and_memory_prefetch(self) -> None:
        from src.ai.tools import (
            ALIAS_TOOL_NAMES,
            DEDUPLICATED_TOOLS_SCHEMA,
            FAST_COMPANION_TOOLS_SCHEMA,
        )
        from src.ai.tools.audio import SentenceTTSStreamer

        # 1. Verify default thinking is disabled & max_tokens is compact
        default_cfg = AgentConfig(base_url=self.base_url, model="qwen36-35b")
        self.assertFalse(default_cfg.enable_thinking)
        self.assertLessEqual(default_cfg.max_tokens, 512)

        # 2. Verify deduplicated schemas drop the 6 alias duplicates while keeping all 28 callable
        self.assertEqual(len(OPENAI_TOOLS_SCHEMA) - len(DEDUPLICATED_TOOLS_SCHEMA), len(ALIAS_TOOL_NAMES))
        self.assertEqual(len(FAST_COMPANION_TOOLS_SCHEMA), 9)

        # 3. Verify clause-level TTS streamer chunks first clause eagerly and flushes cleanly
        streamer = SentenceTTSStreamer(voice="auto", play_audio=False, min_first_clause_words=4)
        chunks_1 = streamer.feed("שלום אהובה שלי מה שלומך היום, ")
        self.assertEqual(len(chunks_1), 1)
        chunks_2 = streamer.feed("הכנתי לך הפתעה מיוחדת ליום ההולדת! ")
        self.assertEqual(len(chunks_2), 1)
        tail = streamer.flush()
        self.assertEqual(tail, [])

        # 4. Verify zero-roundtrip speculative RAG memory prefetch injects context automatically
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "device_state.json")
            with patch.dict(
                os.environ,
                {"COLIBRI_MEMORY_DIR": tmpdir, "COLIBRI_DEVICE_STATE_PATH": state_file},
            ):
                save_fact("She loves drinking iced matcha latte every morning.", category="drinks")
                with patch.object(urllib.request, "urlopen", side_effect=self.gateway.urlopen):
                    cfg = AgentConfig(
                        base_url=self.base_url,
                        api_key="test-key",
                        model="qwen36-35b",
                        fast_tool_pruning=True,
                        prefetch_memory=True,
                    )
                    tokens_seen: List[str] = []
                    run_agent(
                        "What iced matcha drink does she love?",
                        config=cfg,
                        on_token=lambda tok: tokens_seen.append(tok),
                        force_stdlib=True,
                    )
                    chat_reqs = [
                        r["body"]
                        for r in self.gateway.recorded_requests
                        if r["url"].endswith("/chat/completions")
                    ]
                    self.assertGreaterEqual(len(chat_reqs), 1)
                    first_req = chat_reqs[0]
                    # Fast companion mode passes only 9 tool schemas on slot 0
                    self.assertEqual(len(first_req.get("tools", [])), 9)
                    self.assertEqual(first_req.get("cache_slot"), 0)
                    user_msg = [m for m in first_req["messages"] if m["role"] == "user"][0]["content"]
                    self.assertIn("[LOCAL MEMORY CONTEXT]", user_msg)
                    self.assertIn("matcha latte", user_msg)
                    self.assertGreaterEqual(len(tokens_seen), 1)

        # 5. Verify SystemOne zero-token routing skips tool schemas when choice == 'direct'
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "device_state.json")
            with patch.dict(os.environ, {"COLIBRI_DEVICE_STATE_PATH": state_file}):
                self.gateway.recorded_requests.clear()

                def _mock_direct_sys1(req: Any, timeout: float = 10.0) -> _MockHTTPResponse:
                    url = req.full_url if hasattr(req, "full_url") else str(req)
                    if url.endswith("/systemone"):
                        return _MockHTTPResponse(
                            {
                                "id": "req_sys1_direct",
                                "model": "qwen36-35b",
                                "answers": {
                                    "q": {
                                        "type": "choice",
                                        "choice": "direct",
                                        "probabilities": {"direct": 0.94, "web_research": 0.06},
                                        "confidence": 0.92,
                                    }
                                },
                            },
                            url=url,
                        )
                    if url.endswith("/chat/completions"):
                        body = json.loads(req.data.decode("utf-8"))
                        self.gateway.recorded_requests.append({"url": url, "body": body})
                        return _MockHTTPResponse(
                            {
                                "id": "chatcmpl-direct",
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {"role": "assistant", "content": "היי אהובה, אני כאן איתך!"},
                                    }
                                ],
                            },
                            url=url,
                        )
                    return self.gateway.urlopen(req, timeout=timeout)

                with patch.object(urllib.request, "urlopen", side_effect=_mock_direct_sys1):
                    sys1_cfg = AgentConfig(
                        base_url=self.base_url,
                        api_key="test-key",
                        model="qwen36-35b",
                        use_systemone_routing=True,
                        prefetch_memory=False,
                    )
                    res_direct = run_agent("שלום נוגה!", config=sys1_cfg, force_stdlib=True)
                    self.assertIn("אהובה", res_direct.answer)
                    chat_reqs_direct = [
                        r["body"]
                        for r in self.gateway.recorded_requests
                        if r["url"].endswith("/chat/completions")
                    ]
                    self.assertEqual(len(chat_reqs_direct), 1)
                    # Verify tools were omitted because SystemOne routed to 'direct'
                    self.assertNotIn("tools", chat_reqs_direct[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
