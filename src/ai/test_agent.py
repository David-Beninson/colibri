"""Self-contained unit test suite for the Colibri AI Companion (`src/ai/test_agent.py`).

Verifies:
1. Unified companion tool registration (`internet`, `vision`, `audio`/`voice`, `rag`, `code_runner`)
2. Bilingual Persona & deterministic system prompt (`persona.py`: `COMPANION_PERSONA_PROMPT`, `build_system_prompt`)
3. Local RAG Memory persistence (`save_fact`, `query_memory`, `save_conversation_fact`, `query_local_knowledge` in `data/memory_db/`)
4. Safe AST math (`calculate`) & sandboxed Python code execution (`run_python_calculation`)
5. Vision (`extract_text_from_image`, `describe_image_scene`) & Voice (`transcribe_audio`, `text_to_speech`, Hebrew/English voice routing)
6. Strict SSRF blocking (`127.0.0.1`, `169.254.169.254`, `192.168.1.1`) & HTML readable extraction
7. StateGraph transitions (`agent -> tools -> reflect_and_guard -> end`) with Qwen3.6 prefix flags & `SystemOneClient`
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
from src.ai.context import ConversationContext
from src.ai.gateway import (
    ColibriGatewayPort,
    HttpGatewayAdapter,
    InMemoryGatewayAdapter,
)
from src.ai.graph import (
    SYSTEM_PROMPT,
    ColibriAgent,
    ColibriSqliteCheckpointer,
    ColibriTransport,
    HttpTransport,
    StdlibColibriAgent,
    build_internet_agent_graph,
    classify_user_intent,
    get_langchain_tools,
    get_session_history,
    run_agent,
)
from src.ai.persona import (
    COMPANION_NAME,
    COMPANION_PERSONA_PROMPT,
    build_system_prompt,
)
from src.ai.session import (
    InMemorySessionStore,
    SessionStore,
    SqliteSessionStore,
)
from src.ai.systemone import SystemOneClient
from src.ai.tools import (
    CANONICAL_COLLECTIONS,
    COLIBRI_MAX_TOOL_CHARS,
    OPENAI_TOOLS_SCHEMA,
    TOOL_REGISTRY,
    TOOL_REGISTRY_OBJ,
    TOOL_SUITE_OBJ,
    ToolRegistry,
    ToolSuite,
    _resolve_collection,
    calculate,
    describe_image_scene,
    extract_text_from_image,
    fetch_json_or_rss,
    fetch_webpage,
    ingest_url_or_note,
    list_saved_facts,
    load_all_tool_suites,
    query_local_knowledge,
    query_memory,
    run_python_calculation,
    save_conversation_fact,
    save_fact,
    scale_recipe,
    speak_text,
    text_to_speech,
    transcribe_audio,
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

    def test_companion_tool_suites_registered(self) -> None:
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
        }
        self.assertTrue(expected_core_tools.issubset(set(registry.keys())))
        schema_names = {
            item["function"]["name"] for item in schemas if "function" in item
        }
        self.assertTrue(expected_core_tools.issubset(schema_names))
        lc_tools = get_langchain_tools()
        self.assertGreaterEqual(len(lc_tools), len(expected_core_tools))

        # Scope creep checks: robotics & device tools purged
        purged_tools = {
            "detect_person_angle",
            "rotate_device",
            "rotate_device_servo",
            "set_avatar_state",
            "trigger_avatar_state",
            "trigger_hologram_effect",
        }
        for purged in purged_tools:
            self.assertNotIn(purged, registry)
            self.assertNotIn(purged, schema_names)

    def test_persona_and_bilingual_greetings(self) -> None:
        prompt = build_system_prompt("companion")
        self.assertIn("Jimmy bot", prompt)
        self.assertIn("Jimmies family", prompt)
        self.assertIn("STRICT BREVITY", prompt)
        self.assertIn("English", prompt)
        self.assertEqual(prompt, COMPANION_PERSONA_PROMPT)

        # Hebrew strings and scope creep are strictly purged
        purged = [
            "ג'ימי בוט",
            "משפחת ג'ימי",
            "he-IL-HilaNeural",
            "עברית",
            "Pepper's Ghost",
            "holographic cylinder",
            "motorized",
            "rotate_device",
            "set_avatar_state",
            "trigger_hologram_effect",
            "detect_person_angle",
        ]
        for term in purged:
            self.assertNotIn(term, prompt)

        # Birthday greetings removed from persona module exports
        import src.ai.persona as persona_mod
        self.assertFalse(hasattr(persona_mod, "BIRTHDAY_GREETING_HE"))
        self.assertFalse(hasattr(persona_mod, "BIRTHDAY_GREETING_EN"))

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

    def test_robotics_and_device_tools_purged(self) -> None:
        """Verify device.py and robotics/servo/avatar/person-angle tools are completely purged."""
        import src.ai.tools as tools_mod
        robotics_attrs = [
            "rotate_device",
            "rotate_device_servo",
            "set_avatar_state",
            "trigger_avatar_state",
            "trigger_hologram_effect",
            "get_device_state",
            "detect_person_angle",
            "DEVICE_TOOLS",
            "DEVICE_TOOL_REGISTRY",
            "DEVICE_TOOLS_SCHEMA",
        ]
        for attr in robotics_attrs:
            self.assertFalse(hasattr(tools_mod, attr), f"{attr} should not be exported from src.ai.tools")

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

        scene_missing = json.loads(describe_image_scene("/nonexistent/frame_12345.jpg"))
        self.assertIn("error", scene_missing)

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
            with patch.dict(os.environ, {"COLIBRI_MEMORY_DIR": tmpdir}):
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
                        self.assertIn("extra_body", req_body)
                        self.assertEqual(req_body["extra_body"].get("cache_slot"), 3)
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

    def test_reflect_and_guard_max_steps_limit(self) -> None:
        self.gateway.mode = "infinite_tools"
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"COLIBRI_MEMORY_DIR": tmpdir}):
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

        # 2. Verify deduplicated schemas drop alias duplicates while keeping unique companion tools
        self.assertEqual(len(OPENAI_TOOLS_SCHEMA) - len(DEDUPLICATED_TOOLS_SCHEMA), len(ALIAS_TOOL_NAMES))
        self.assertEqual(len(FAST_COMPANION_TOOLS_SCHEMA), 6)

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
            with patch.dict(
                os.environ,
                {"COLIBRI_MEMORY_DIR": tmpdir},
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
                    # Fast companion mode passes only 6 core tool schemas on slot 0
                    self.assertEqual(len(first_req.get("tools", [])), 6)
                    self.assertEqual(first_req.get("cache_slot"), 0)
                    user_msg = [m for m in first_req["messages"] if m["role"] == "user"][0]["content"]
                    self.assertIn("[LOCAL MEMORY CONTEXT]", user_msg)
                    self.assertIn("matcha latte", user_msg)
                    self.assertGreaterEqual(len(tokens_seen), 1)

                    # Research query switches to cache_slot=1 even when fast_tool_pruning is False
                    self.gateway.recorded_requests.clear()
                    cfg_default = AgentConfig(
                        base_url=self.base_url,
                        api_key="test-key",
                        model="qwen36-35b",
                        fast_tool_pruning=False,
                        prefetch_memory=False,
                    )
                    run_agent(
                        "Search arxiv for latest quantum computing papers",
                        config=cfg_default,
                        force_stdlib=True,
                    )
                    res_reqs = [
                        r["body"]
                        for r in self.gateway.recorded_requests
                        if r["url"].endswith("/chat/completions")
                    ]
                    self.assertGreaterEqual(len(res_reqs), 1)
                    self.assertEqual(res_reqs[0].get("cache_slot"), 1)
                    self.assertEqual(res_reqs[0].get("extra_body", {}).get("cache_slot"), 1)

        # 5. Verify SystemOne zero-token routing skips tool schemas when choice == 'direct'
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"COLIBRI_MEMORY_DIR": tmpdir}):
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

    def test_multi_collection_segmentation_ingestion_and_recipe_scaling(self) -> None:
        self.assertEqual(
            CANONICAL_COLLECTIONS,
            ("recipes_and_food", "web_bookmarks_and_reads", "personal_facts_and_notes"),
        )
        self.assertEqual(_resolve_collection("recipe", "2 cups flour"), "recipes_and_food")
        self.assertEqual(_resolve_collection("bookmark", "https://example.org/article"), "web_bookmarks_and_reads")
        self.assertEqual(_resolve_collection("family", "Mom's birthday is in May"), "personal_facts_and_notes")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"COLIBRI_MEMORY_DIR": tmpdir}):
                # 1. Ingest a structured family recipe into `recipes_and_food`
                recipe_text = (
                    "Jimmies Family Fluffy Pancakes\n"
                    "Servings: 4\n"
                    "Ingredients:\n"
                    "- 1 1/2 cups all-purpose flour\n"
                    "- 2 tbsp sugar\n"
                    "- 200 ml milk\n"
                    "- 1/2 tsp salt\n"
                    "Steps:\n"
                    "1. Whisk dry ingredients together in a bowl.\n"
                    "2. Stir in milk and cook on a buttered griddle until golden."
                )
                ingested_rec = json.loads(ingest_url_or_note(recipe_text))
                self.assertEqual(ingested_rec["status"], "ingested")
                self.assertEqual(ingested_rec["collection"], "recipes_and_food")
                self.assertEqual(ingested_rec["servings"], 4.0)
                self.assertEqual(len(ingested_rec["ingredients"]), 4)

                # 2. Scale the saved recipe from 4 servings to 8 servings (2x)
                scaled = json.loads(scale_recipe("Fluffy Pancakes", target_servings=8))
                self.assertEqual(scaled["status"], "ok")
                self.assertEqual(scaled["base_servings"], 4.0)
                self.assertEqual(scaled["target_servings"], 8.0)
                self.assertAlmostEqual(scaled["scale_factor"], 2.0)
                by_item = {row["item"]: row for row in scaled["scaled_ingredients"]}
                self.assertAlmostEqual(by_item["all-purpose flour"]["scaled_quantity"], 3.0)
                self.assertAlmostEqual(by_item["milk"]["scaled_quantity"], 400.0)
                self.assertAlmostEqual(by_item["salt"]["scaled_quantity"], 1.0)

                # 3. Ingest a web bookmark via mocked `fetch_webpage` into `web_bookmarks_and_reads`
                with patch(
                    "src.ai.tools.rag.fetch_webpage",
                    return_value=(
                        "Page Title: Apple Silicon Unified Memory Guide\n"
                        "Source URL: https://example.org/m1-guide\n"
                        "Extracted Readable Content:\n"
                        "Apple Silicon M1 shares unified low-latency memory between CPU and Metal GPU cores. "
                        "This eliminates PCIe copy overhead during Qwen3.6 MoE inference."
                    ),
                ):
                    ingested_bm = json.loads(ingest_url_or_note("https://example.org/m1-guide"))
                self.assertEqual(ingested_bm["status"], "ingested")
                self.assertEqual(ingested_bm["collection"], "web_bookmarks_and_reads")
                self.assertEqual(ingested_bm["title"], "Apple Silicon Unified Memory Guide")

                # 4. Ingest a personal note into `personal_facts_and_notes`
                ingested_note = json.loads(
                    ingest_url_or_note(
                        "The Jimmies family loves Sunday family dinners at 7pm.",
                        collection_name="personal_facts_and_notes",
                        title="Sunday Family Dinner",
                    )
                )
                self.assertEqual(ingested_note["collection"], "personal_facts_and_notes")

                # 5. Query segmented collections individually and via federated `companion_memory`
                rec_search = json.loads(
                    query_local_knowledge("Pancakes flour", collection_name="recipes_and_food")
                )
                self.assertGreaterEqual(len(rec_search["matches"]), 1)
                self.assertEqual(rec_search["matches"][0]["collection"], "recipes_and_food")

                bm_search = json.loads(
                    query_local_knowledge("Unified Memory Metal", collection_name="web_bookmarks_and_reads")
                )
                self.assertGreaterEqual(len(bm_search["matches"]), 1)
                self.assertEqual(bm_search["matches"][0]["collection"], "web_bookmarks_and_reads")

                fed_search = json.loads(query_memory("Sunday family dinners", k=3))
                self.assertTrue(
                    any("Sunday" in str(m.get("value", "")) for m in fed_search["matches"])
                )

    def test_rag_strict_context_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"COLIBRI_MEMORY_DIR": tmpdir, "COLIBRI_MAX_TOOL_CHARS": "600"}):
                # 1. Test list_saved_facts bounded to COLIBRI_MAX_TOOL_CHARS
                for i in range(20):
                    save_fact(f"Family fact number {i}: detail info about preference {i} with long description " * 3)
                facts_list = list_saved_facts()
                self.assertLessEqual(len(facts_list), 600)

                # 2. Test ingest_url_or_note bounded to COLIBRI_MAX_TOOL_CHARS on note
                huge_note = "Detail note content " * 100
                ingested = ingest_url_or_note(huge_note, title="Huge Note")
                self.assertLessEqual(len(ingested), 600)

                # 3. Test ingest_url_or_note bounded to COLIBRI_MAX_TOOL_CHARS on scraped URL
                with patch("src.ai.tools.rag.fetch_webpage", return_value="Scraped Web Content " * 200) as mock_fetch:
                    ingested_url = ingest_url_or_note("https://example.org/huge-article")
                    mock_fetch.assert_called_once()
                    self.assertLessEqual(mock_fetch.call_args[1].get("max_chars", 9999), 600)
                    self.assertLessEqual(len(ingested_url), 600)

                # 4. Test scale_recipe bounded to COLIBRI_MAX_TOOL_CHARS
                large_recipe = "\n".join([f"- {i+1} cups ingredient_{i}" for i in range(40)])
                scaled = scale_recipe(large_recipe, target_servings=10)
                self.assertLessEqual(len(scaled), 600)

                # 5. Test unified truncate_output strictly caps length including suffix
                from src.ai.tools import truncate_output, TOOL_REGISTRY_OBJ
                truncated = truncate_output("A" * 5000, max_chars=4000)
                self.assertEqual(len(truncated), 4000)
                self.assertTrue(truncated.endswith("...[truncated]"))
                reg_out = TOOL_REGISTRY_OBJ.execute("calculate", {"expression": "2 ** 10"}, max_chars=50)
                self.assertLessEqual(len(reg_out), 50)

    def test_native_langgraph_checkpointer_integration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "custom_checkpoints.sqlite")
            custom_cp = ColibriSqliteCheckpointer(db_path=db_path)
            runner = build_internet_agent_graph(checkpointer=custom_cp)
            self.assertIs(runner.checkpointer, custom_cp)
            self.assertEqual(runner.checkpointer.db_path, os.path.abspath(db_path))

            # Test LangGraph native checkpointer interface: put, get_tuple, list
            cfg = {"configurable": {"thread_id": "session_alpha"}}
            ckpt = {
                "v": 1,
                "id": "chk_001",
                "channel_values": {
                    "messages": [{"role": "user", "content": "Hello Jimmy bot"}, {"role": "assistant", "content": "Shalom!"}],
                    "intent": "conversation",
                    "step_count": 1,
                },
            }
            meta = {"source": "input", "step": 1}
            ret_cfg = custom_cp.put(cfg, ckpt, meta)
            self.assertEqual(ret_cfg.get("configurable", {}).get("thread_id"), "session_alpha")

            tup = custom_cp.get_tuple(cfg)
            self.assertIsNotNone(tup)
            self.assertEqual(tup.checkpoint["channel_values"]["intent"], "conversation")
            self.assertEqual(len(tup.checkpoint["channel_values"]["messages"]), 2)

            all_ckpts = list(custom_cp.list(cfg))
            self.assertEqual(len(all_ckpts), 1)
            self.assertEqual(all_ckpts[0].config["configurable"]["thread_id"], "session_alpha")

            # Verify LangGraph adapter module decoupling and delegation
            from src.ai.langgraph_adapter import (
                InternetAgentState as AdapterState,
                build_internet_agent_graph as adapter_build_graph,
                has_langgraph as adapter_has_lg,
            )
            from src.ai import (
                InternetAgentState as RootState,
                build_internet_agent_graph as root_build_graph,
            )
            self.assertIs(AdapterState, RootState)
            self.assertIs(adapter_build_graph, root_build_graph)
            self.assertIsInstance(adapter_has_lg(), bool)

            # Test force_stdlib delegation cleanly returns ColibriAgent
            stdlib_runner = adapter_build_graph(force_stdlib=True)
            self.assertIsInstance(stdlib_runner, ColibriAgent)

    def test_intent_classification_and_sqlite_checkpoint_persistence(self) -> None:
        self.assertEqual(
            classify_user_intent("Please ingest https://example.org/pancake-recipe into my vault"),
            "url_ingestion",
        )
        self.assertEqual(
            classify_user_intent("Scale the pancake recipe for 8 servings"),
            "recipe_scaling",
        )
        self.assertEqual(
            classify_user_intent("What is her favorite song from my memory vault?"),
            "memory_recall",
        )
        self.assertEqual(
            classify_user_intent("What is the weather in Tel Aviv and calculate 12 * 12?"),
            "factual_extraction",
        )
        self.assertEqual(
            classify_user_intent("Hey Jimmy bot, how are you doing today?"),
            "conversation",
        )
        self.assertEqual(COMPANION_NAME, "Jimmy bot")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"COLIBRI_MEMORY_DIR": tmpdir},
            ):
                with patch.object(urllib.request, "urlopen", side_effect=self.gateway.urlopen):
                    cfg = AgentConfig(
                        base_url=self.base_url,
                        api_key="test-key",
                        model="qwen36-35b",
                        prefetch_memory=False,
                    )
                    res1 = run_agent(
                        "Calculate sqrt(144) + 25 * 4",
                        config=cfg,
                        force_stdlib=True,
                        thread_id="jimmies_family_session_1",
                    )
                    self.assertEqual(res1.intent, "factual_extraction")
                    self.assertEqual(res1.thread_id, "jimmies_family_session_1")
                    self.assertTrue(os.path.isfile(os.path.join(tmpdir, "checkpoints.sqlite")))

                    persisted_hist = get_session_history("jimmies_family_session_1")
                    self.assertGreaterEqual(len(persisted_hist), 2)
                    self.assertEqual(persisted_hist[0]["role"], "user")
                    self.assertIn("sqrt(144)", persisted_hist[0]["content"])

                    # Second turn on the same thread_id automatically restores prior conversation history
                    self.gateway.recorded_requests.clear()
                    res2 = run_agent(
                        "Scale the pancake recipe for 6 servings",
                        config=cfg,
                        force_stdlib=True,
                        thread_id="jimmies_family_session_1",
                    )
                    self.assertEqual(res2.intent, "recipe_scaling")
                    chat_reqs = [
                        r["body"]
                        for r in self.gateway.recorded_requests
                        if r["url"].endswith("/chat/completions")
                    ]
                    self.assertGreaterEqual(len(chat_reqs), 1)
                    sent_msgs = chat_reqs[0]["messages"]
                    # System prompt at index 0 + restored Turn 1 user/assistant/tool messages + Turn 2 user message
                    self.assertEqual(sent_msgs[0]["role"], "system")
                    self.assertTrue(
                        any("sqrt(144)" in str(m.get("content", "")) for m in sent_msgs if m.get("role") == "user")
                    )

                    cp = ColibriSqliteCheckpointer().load_checkpoint("jimmies_family_session_1")
                    self.assertIsNotNone(cp)
                    self.assertEqual(cp["intent"], "recipe_scaling")

    def test_colibri_agent_and_custom_transport(self) -> None:
        """Verify unified ColibriAgent runner, pluggable ColibriTransport port, and HttpTransport."""
        self.assertTrue(issubclass(HttpTransport, ColibriTransport))
        self.assertIs(StdlibColibriAgent, ColibriAgent)

        class _CustomTestTransport(ColibriTransport):
            def __init__(self) -> None:
                self.calls: List[Dict[str, Any]] = []

            def complete(
                self,
                config: AgentConfig,
                messages: Any,
                tools: Any = None,
                active_slot: Any = None,
                stream_tokens: bool = False,
                token_callback: Any = None,
            ) -> Dict[str, Any]:
                self.calls.append(
                    {
                        "messages": list(messages),
                        "tools": list(tools) if tools else None,
                        "slot": active_slot,
                        "streaming": stream_tokens,
                    }
                )
                if token_callback is not None:
                    token_callback("Custom transport token.")

                # If first turn, return a tool call to calculate; otherwise return final answer
                has_tool_result = any(m.get("role") == "tool" for m in messages)
                if not has_tool_result:
                    return {
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "custom_call_001",
                                            "type": "function",
                                            "function": {
                                                "name": "calculate",
                                                "arguments": json.dumps({"expression": "100 / 4"}),
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                return {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The result is 25.",
                            },
                        }
                    ]
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_transport = _CustomTestTransport()
            cfg = AgentConfig(base_url=self.base_url, model="qwen36-35b")
            cp = ColibriSqliteCheckpointer(db_path=os.path.join(tmpdir, "transport_test.sqlite"))
            agent = ColibriAgent(config=cfg, transport=custom_transport, checkpointer=cp)

            # 1. Test .run() with prompt string
            run_res = agent.run("What is 100 / 4?", thread_id="thread_transport_test")
            self.assertEqual(run_res.answer, "The result is 25.")
            self.assertEqual(run_res.steps, 1)
            self.assertEqual(len(run_res.tool_calls), 1)
            self.assertEqual(run_res.tool_calls[0]["name"], "calculate")
            self.assertEqual(len(custom_transport.calls), 2)
            self.assertEqual(custom_transport.calls[0]["messages"][0]["role"], "system")

            # 2. Test .invoke() with dict input format
            dict_res = agent.invoke(
                {"messages": [{"role": "user", "content": "Direct question"}], "max_steps": 2},
                thread_id="thread_invoke_test",
            )
            self.assertIn("The result is 25.", dict_res["answer"])
            self.assertIsInstance(dict_res, dict)

            # 3. Test HttpTransport directly with JSON response and real SSE stream parsing
            http_transport = HttpTransport()
            with patch.object(urllib.request, "urlopen", side_effect=self.gateway.urlopen):
                # JSON completion (turn 2 with tool message produces answer tokens)
                streamed_tokens: List[str] = []
                resp = http_transport.complete(
                    config=cfg,
                    messages=[
                        {"role": "user", "content": "test"},
                        {"role": "assistant", "content": ""},
                        {"role": "tool", "content": "112"},
                    ],
                    stream_tokens=True,
                    token_callback=lambda t: streamed_tokens.append(t),
                )
                self.assertIn("choices", resp)
                self.assertGreaterEqual(len(streamed_tokens), 1)
                self.assertIn("112", streamed_tokens[0])

            # SSE line-by-line stream decoding
            class _MockSSEResponse(io.BytesIO):
                def __enter__(self) -> "_MockSSEResponse":
                    return self

                def __exit__(self, *args: Any) -> None:
                    self.close()

            sse_raw = (
                b'data: {"choices": [{"delta": {"content": "Hello ", "reasoning_content": "think "}}]}\n\n'
                b'data: {"choices": [{"delta": {"content": "world!"}}]}\n\n'
                b"data: [DONE]\n\n"
            )
            sse_tokens: List[str] = []
            with patch.object(urllib.request, "urlopen", return_value=_MockSSEResponse(sse_raw)):
                parsed_sse = http_transport.complete(
                    config=cfg,
                    messages=[{"role": "user", "content": "hi"}],
                    stream_tokens=True,
                    token_callback=lambda t: sse_tokens.append(t),
                )
                self.assertEqual(parsed_sse["choices"][0]["message"]["content"], "Hello world!")
                self.assertEqual(parsed_sse["choices"][0]["message"]["reasoning_content"], "think ")
                self.assertEqual(sse_tokens, ["Hello ", "world!"])

    def test_in_memory_gateway_adapter_and_colibri_agent_without_mocks(self) -> None:
        """Verify InMemoryGatewayAdapter port and test ColibriAgent with zero urllib monkey-patching."""
        # 1. Verify ColibriGatewayPort hierarchy
        self.assertTrue(issubclass(HttpGatewayAdapter, ColibriGatewayPort))
        self.assertTrue(issubclass(InMemoryGatewayAdapter, ColibriGatewayPort))
        self.assertTrue(issubclass(ColibriTransport, ColibriGatewayPort))

        # 2. InMemoryGatewayAdapter direct operation and registration
        gw = InMemoryGatewayAdapter()
        models = gw.probe_models()
        self.assertEqual(gw.models_calls, 1)
        self.assertEqual(models[0]["id"], "qwen36-35b")

        # Dynamic model detection using gateway seam
        detected_id, detected_ctx = detect_colibri_model(gateway=gw)
        self.assertEqual(detected_id, "qwen36-35b")
        self.assertEqual(detected_ctx, 65536)

        # 3. SystemOneClient using gateway seam directly
        gw.register_systemone_response(
            {
                "model": "qwen36-35b",
                "answers": {
                    "q": {
                        "choice": "companion_fast",
                        "probabilities": {"companion_fast": 0.95, "direct": 0.05},
                        "confidence": 0.95,
                    }
                },
            }
        )
        sys1 = SystemOneClient(gateway=gw)
        decision = sys1.route_query("What time is sunset tonight?")
        self.assertEqual(decision.choice, "companion_fast")
        self.assertAlmostEqual(decision.confidence, 0.95)
        self.assertEqual(len(gw.systemone_calls), 1)
        self.assertEqual(gw.systemone_calls[0]["slot"], 2)

        # 4. Multi-turn ColibriAgent workflow with InMemoryGatewayAdapter (NO urllib mocking!)
        gw.reset()
        t1_response = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_inmem_01",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": json.dumps({"expression": "7 * 6"}),
                                },
                            }
                        ],
                    },
                }
            ]
        }
        t2_response = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "7 multiplied by 6 is 42.",
                    },
                }
            ]
        }
        gw.register_chat_response(t1_response)
        gw.register_chat_response(t2_response)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AgentConfig(base_url="http://mock-colibri.internal/v1", model="qwen36-35b")
            cp = ColibriSqliteCheckpointer(db_path=os.path.join(tmpdir, "inmem_test.sqlite"))
            agent = ColibriAgent(config=cfg, transport=gw, checkpointer=cp)

            streamed_chunks: List[str] = []
            res = agent.run("What is 7 * 6?")

            self.assertEqual(res.answer, "7 multiplied by 6 is 42.")
            self.assertEqual(res.steps, 1)
            self.assertEqual(len(res.tool_calls), 1)
            self.assertEqual(res.tool_calls[0]["name"], "calculate")
            self.assertEqual(len(gw.chat_calls), 2)
            # Verify slot preservation
            self.assertEqual(gw.chat_calls[0]["slot"], 0)
            self.assertEqual(gw.chat_calls[1]["slot"], 0)

        # 5. Verify HttpGatewayAdapter error handling raises RuntimeError / ValueError
        bad_adapter = HttpGatewayAdapter(base_url="http://127.0.0.1:1/v1", timeout=0.1)
        with self.assertRaises(RuntimeError):
            bad_adapter.probe_models()
        with self.assertRaises(RuntimeError):
            bad_adapter.complete_chat(messages=[{"role": "user", "content": "hello"}])
        with self.assertRaises(RuntimeError):
            bad_adapter.decide_systemone(state="test", questions={"q": {"type": "noul"}})

    def test_session_store_standalone_and_inmemory(self) -> None:
        """Verify SessionStore ABC, SqliteSessionStore, and InMemorySessionStore."""
        self.assertTrue(issubclass(SqliteSessionStore, SessionStore))
        self.assertTrue(issubclass(InMemorySessionStore, SessionStore))
        self.assertIs(ColibriSqliteCheckpointer, SqliteSessionStore)

        # 1. InMemorySessionStore lifecycle
        mem = InMemorySessionStore()
        self.assertEqual(mem.list_threads(), [])
        self.assertIsNone(mem.load_checkpoint("t1"))
        self.assertEqual(mem.get_history("t1"), [])
        self.assertFalse(mem.clear("t1"))

        # Save via dict
        mem.save_checkpoint("t1", {
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "Hello\n\n[LOCAL MEMORY CONTEXT] facts"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "intent": "conversation",
            "step_count": 1,
        })
        self.assertEqual(mem.list_threads(), ["t1"])
        cp1 = mem.load_checkpoint("t1")
        self.assertIsNotNone(cp1)
        self.assertEqual(cp1["intent"], "conversation")
        self.assertEqual(cp1["step_count"], 1)
        # Message cleaning stripped system prompt and transient suffix
        self.assertEqual(len(cp1["messages"]), 2)
        self.assertEqual(cp1["messages"][0]["content"], "Hello")
        self.assertEqual(mem.get_history("t1")[1]["content"], "Hi!")

        # LangGraph put, get_tuple, put_writes, list
        cfg = {"configurable": {"thread_id": "t2"}}
        ckpt = {
            "v": 1,
            "id": "chk_t2",
            "channel_values": {
                "messages": [{"role": "user", "content": "Ping"}],
                "intent": "quick_query",
                "step_count": 2,
            },
        }
        mem.put(cfg, ckpt)
        mem.put_writes(cfg, [("channel_a", "val_a")], task_id="task_001")
        tup = mem.get_tuple(cfg)
        self.assertIsNotNone(tup)
        self.assertEqual(tup.checkpoint["channel_values"]["intent"], "quick_query")
        listed = list(mem.list())
        self.assertGreaterEqual(len(listed), 2)

        # Clear
        self.assertTrue(mem.clear("t1"))
        self.assertIsNone(mem.load_checkpoint("t1"))

        # 2. SqliteSessionStore lifecycle
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "session_test.sqlite")
            sql_store = SqliteSessionStore(db_path=db_path)
            self.assertEqual(sql_store.db_path, os.path.abspath(db_path))
            self.assertEqual(sql_store.list_threads(), [])

            # Save via sequence of messages
            sql_store.save_checkpoint(
                thread_id="sql_thread_1",
                checkpoint_data=[
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
                intent="calculation",
                step_count=1,
            )
            self.assertIn("sql_thread_1", sql_store.list_threads())
            hist = sql_store.get_history("sql_thread_1")
            self.assertEqual(len(hist), 2)
            self.assertEqual(hist[0]["content"], "What is 2+2?")

            # Verify get_session_history helper
            h2 = get_session_history("sql_thread_1", db_path=db_path)
            self.assertEqual(len(h2), 2)

            # Clear
            self.assertTrue(sql_store.clear("sql_thread_1"))
            self.assertFalse(sql_store.clear("sql_thread_1"))
            self.assertEqual(sql_store.list_threads(), [])

        # 3. Agent execution with InMemorySessionStore
        gw = InMemoryGatewayAdapter()
        gw.register_chat_response({
            "choices": [{"message": {"role": "assistant", "content": "Session store active."}}]
        })
        mem_store = InMemorySessionStore()
        agent = ColibriAgent(
            config=AgentConfig(base_url="http://mock.internal/v1"),
            transport=gw,
            checkpointer=mem_store,
        )
        res = agent.run("Test message", thread_id="agent_thread")
        self.assertEqual(res.answer, "Session store active.")
        self.assertIn("agent_thread", mem_store.list_threads())
        self.assertEqual(len(mem_store.get_history("agent_thread")), 2)

    def test_conversation_context_prefix_preservation_and_slot_coordination(self) -> None:
        """Verify ConversationContext KV-slot mapping, prefix preservation, and storage sanitization."""
        # 1. Slot resolution
        self.assertEqual(ConversationContext.resolve_slot("conversation"), 0)
        self.assertEqual(ConversationContext.resolve_slot("memory_recall"), 0)
        self.assertEqual(ConversationContext.resolve_slot("factual_extraction", requires_research=False), 0)
        self.assertEqual(ConversationContext.resolve_slot("factual_extraction", requires_research=True), 1)
        self.assertEqual(ConversationContext.resolve_slot("url_ingestion"), 1)
        self.assertEqual(ConversationContext.resolve_slot("recipe_scaling"), 1)
        self.assertEqual(ConversationContext.resolve_slot("research"), 1)
        self.assertEqual(ConversationContext.resolve_slot("classification"), 2)
        self.assertEqual(ConversationContext.resolve_slot("systemone"), 2)

        # 2. Text cleaning helper
        raw_text = "What is my mother's birthday?\n\n[LOCAL MEMORY CONTEXT]\n- mom_birthday: 1965-04-12"
        self.assertEqual(ConversationContext.clean_user_text(raw_text), "What is my mother's birthday?")
        self.assertEqual(ConversationContext.clean_user_text("Normal prompt"), "Normal prompt")
        self.assertEqual(ConversationContext.clean_user_text(""), "")

        # 3. from_turn and prefix preservation on wire payload
        ctx = ConversationContext.from_turn(
            prompt="What is my mother's birthday?",
            system_prompt="DETERMINISTIC_COMPANION_PROMPT",
            memory_context="- mom_birthday: 1965-04-12",
            intent="memory_recall",
        )
        self.assertEqual(ctx.slot, 0)
        wire_payload = ctx.to_llm_payload()
        self.assertEqual(len(wire_payload), 2)
        # Wire index 0 is always byte-identical system prompt
        self.assertEqual(wire_payload[0]["role"], "system")
        self.assertEqual(wire_payload[0]["content"], "DETERMINISTIC_COMPANION_PROMPT")
        # Wire user message contains speculative memory context
        self.assertEqual(wire_payload[1]["role"], "user")
        self.assertIn("What is my mother's birthday?", wire_payload[1]["content"])
        self.assertIn("[LOCAL MEMORY CONTEXT]", wire_payload[1]["content"])
        self.assertIn("mom_birthday: 1965-04-12", wire_payload[1]["content"])

        # 4. Storage messages sanitization (transient memory stripped, no system prompt)
        ctx.messages.append({"role": "assistant", "content": "Her birthday is April 12th."})
        storage = ctx.to_storage_messages()
        self.assertEqual(len(storage), 2)
        self.assertEqual(storage[0], {"role": "user", "content": "What is my mother's birthday?"})
        self.assertEqual(storage[1], {"role": "assistant", "content": "Her birthday is April 12th."})

        # 5. Multi-turn continuation with prior storage history
        ctx2 = ConversationContext.from_turn(
            prompt="What should I buy her?",
            history=storage,
            system_prompt="DETERMINISTIC_COMPANION_PROMPT",
            memory_context="- likes: gardening",
            intent="conversation",
        )
        wire2 = ctx2.to_llm_payload()
        self.assertEqual(len(wire2), 4)
        self.assertEqual(wire2[0]["role"], "system")
        # Prior turn in wire2 is clean (no stale memory context)
        self.assertEqual(wire2[1], {"role": "user", "content": "What is my mother's birthday?"})
        self.assertEqual(wire2[2], {"role": "assistant", "content": "Her birthday is April 12th."})
        # Active user turn has the new memory context
        self.assertEqual(wire2[3]["role"], "user")
        self.assertIn("What should I buy her?", wire2[3]["content"])
        self.assertIn("likes: gardening", wire2[3]["content"])

        # 6. ColibriAgent.run_turn execution & context binding
        gw = InMemoryGatewayAdapter()
        gw.register_chat_response({
            "choices": [{"message": {"role": "assistant", "content": "A nice gardening kit."}}]
        })
        agent = ColibriAgent(
            config=AgentConfig(base_url="http://mock.internal/v1"),
            transport=gw,
            checkpointer=InMemorySessionStore(),
        )
        turn_result = agent.run_turn(
            "What should I get her for her birthday?",
            history=storage,
            thread_id="turn_integration_thread",
        )
        self.assertEqual(turn_result.answer, "A nice gardening kit.")
        self.assertIsNotNone(turn_result.context)
        self.assertIsInstance(turn_result.context, ConversationContext)
        self.assertEqual(turn_result.context.slot, 0)
        self.assertEqual(gw.chat_calls[0]["slot"], 0)
        self.assertEqual(gw.chat_calls[0]["messages"][0]["role"], "system")

    def test_tool_suite_consolidation_execution_concurrency_and_pruning(self) -> None:
        """Verify deep ToolSuite evolution: direct execution, concurrency, XML salvaging, and pruning."""
        # 1. Backwards compatibility alias and global instances
        self.assertIs(ToolRegistry, ToolSuite)
        self.assertIs(TOOL_REGISTRY_OBJ, TOOL_SUITE_OBJ)
        self.assertIsInstance(TOOL_SUITE_OBJ, ToolSuite)
        self.assertEqual(COLIBRI_MAX_TOOL_CHARS, 4000)

        suite = ToolSuite.from_modules()
        self.assertGreater(len(suite), 15)

        # 2. Direct single execution and backwards-compatible calling signatures
        out_single = suite.execute_single("calculate", {"expression": "12 * 12"})
        self.assertEqual(json.loads(out_single)["result"], 144)

        # Legacy execute(name, args)
        out_legacy = suite.execute("calculate", {"expression": "9 * 9"})
        self.assertEqual(json.loads(out_legacy)["result"], 81)

        # Unified execute(calls_list) returning List[Dict[str, Any]]
        calls_result = suite.execute([{"name": "calculate", "args": {"expression": "25 * 4"}}])
        self.assertIsInstance(calls_result, list)
        self.assertEqual(len(calls_result), 1)
        self.assertEqual(calls_result[0]["name"], "calculate")
        self.assertEqual(json.loads(calls_result[0]["content"])["result"], 100)
        self.assertIn("call_", calls_result[0]["tool_call_id"])
        self.assertGreaterEqual(calls_result[0]["elapsed_s"], 0)

        # Truncation bounds
        truncated_res = suite.execute(
            [{"name": "calculate", "args": {"expression": "10**25"}}],
            max_chars=18,
        )
        self.assertLessEqual(len(truncated_res[0]["content"]), 18)
        self.assertTrue(truncated_res[0]["content"].endswith("...[truncated]"))

        # 3. Parallel concurrency using ThreadPoolExecutor
        parallel_calls = [
            {"id": f"c_{i}", "name": "calculate", "args": {"expression": f"{i} * 10"}}
            for i in range(5)
        ]
        parallel_results = suite.execute(parallel_calls, parallel=True)
        self.assertEqual(len(parallel_results), 5)
        self.assertEqual([r["tool_call_id"] for r in parallel_results], [f"c_{i}" for i in range(5)])
        self.assertEqual([json.loads(r["content"])["result"] for r in parallel_results], [0, 10, 20, 30, 40])

        # 4. Inline XML <tool_call> salvaging
        xml_content = (
            "I will evaluate this for you:\n"
            "<tool_call>{\"name\": \"calculate\", \"arguments\": {\"expression\": \"7 * 8\"}}</tool_call>\n"
            "And the second step:\n"
            "<tool_call>{\"name\": \"calculate\", \"args\": {\"expression\": \"100 - 35\"}}</tool_call>"
        )
        salvaged = ToolSuite.salvage_inline_calls(xml_content)
        self.assertEqual(len(salvaged), 2)
        self.assertEqual(salvaged[0]["name"], "calculate")
        self.assertEqual(salvaged[0]["args"], {"expression": "7 * 8"})
        self.assertEqual(salvaged[1]["args"], {"expression": "100 - 35"})

        # Salvage and execution via unified suite.execute(content)
        salvaged_exec = suite.execute(xml_content)
        self.assertEqual(len(salvaged_exec), 2)
        self.assertEqual(json.loads(salvaged_exec[0]["content"])["result"], 56)
        self.assertEqual(json.loads(salvaged_exec[1]["content"])["result"], 65)

        # 5. Keyword-based tool pruning heuristics
        fast_companion = suite.prune_for_prompt("What is the weather in Haifa today?")
        self.assertEqual(len(fast_companion), 6)

        research_schemas = suite.prune_for_prompt("Search arxiv for transformer architectures")
        self.assertEqual(len(research_schemas), 20)

        ingest_schemas = suite.prune_for_prompt("Bookmark this link https://google.com")
        self.assertEqual(len(ingest_schemas), 20)

        scaling_schemas = suite.prune_for_prompt("Scale the recipe to 12 servings")
        self.assertEqual(len(scaling_schemas), 20)

        # Classmethod invocation check
        cls_pruned = ToolSuite.prune_for_prompt("Hi Jimmy, how are you doing?")
        self.assertEqual(len(cls_pruned), 6)

        # 6. ColibriAgent delegation verification
        agent = ColibriAgent(
            config=AgentConfig(base_url="http://mock.internal/v1"),
            transport=InMemoryGatewayAdapter(),
            tool_suite=suite,
        )
        self.assertIs(agent.tool_suite, suite)
        self.assertIs(agent.tool_registry, suite)
        delegated_results = agent.execute_tools([{"name": "calculate", "args": {"expression": "3 + 7"}}])
        self.assertEqual(len(delegated_results), 1)
        self.assertEqual(json.loads(delegated_results[0]["content"])["result"], 10)

    def test_gateway_extra_body_kv_slot_and_cli_session_store(self) -> None:
        """Verify HttpGatewayAdapter extra_body KV slot compliance and CLI SessionStore wiring."""
        # 1. Verify decide_systemone payload extra_body KV slot
        adapter = HttpGatewayAdapter(base_url="http://mock-colibri.internal/v1")
        captured_reqs: List[urllib.request.Request] = []

        def fake_urlopen(req: urllib.request.Request, timeout: float = 30.0) -> Any:
            captured_reqs.append(req)
            resp_data = json.dumps({"questions": {"q": {"choice": "web_research", "confidence": 0.9}}}).encode("utf-8")
            from io import BytesIO
            bio = BytesIO(resp_data)
            setattr(bio, "read", bio.read)
            return bio

        with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            res = adapter.decide_systemone(
                state="query",
                questions={"q": {"type": "choice"}},
                slot=2,
            )
            self.assertEqual(res["questions"]["q"]["choice"], "web_research")
            self.assertEqual(len(captured_reqs), 1)
            sent_body = json.loads(captured_reqs[0].data.decode("utf-8"))
            # Standard AGENTS_MAIN.md §3.1 check
            self.assertEqual(sent_body["cache_slot"], 2)
            self.assertIn("extra_body", sent_body)
            self.assertEqual(sent_body["extra_body"]["cache_slot"], 2)

        # 2. Verify ConversationContext slot configuration overrides
        cfg = AgentConfig(cache_slot=4, research_cache_slot=5, routing_cache_slot=6)
        ctx_companion = ConversationContext.from_turn("hello", config=cfg)
        self.assertEqual(ctx_companion.slot, 4)
        ctx_research = ConversationContext.from_turn("research arxiv", requires_research=True, config=cfg)
        self.assertEqual(ctx_research.slot, 5)
        ctx_routing = ConversationContext.from_turn("classify query", intent="classification", config=cfg)
        self.assertEqual(ctx_routing.slot, 6)

        # 3. Verify CLI SessionStore clear and history persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "cli_test.sqlite")
            store = SqliteSessionStore(db_path=db_path)
            store.save_checkpoint("default", [{"role": "user", "content": "hello"}])
            self.assertEqual(len(store.get_history("default")), 1)
            store.clear("default")
            self.assertEqual(len(store.get_history("default")), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
