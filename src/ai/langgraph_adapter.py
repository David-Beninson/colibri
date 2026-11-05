"""LangGraph StateGraph orchestration adapter for Colibri (`src/ai/langgraph_adapter.py`).

Provides optional LangGraph StateGraph node wiring (agent, tools, reflect_and_guard,
intent_router) with graceful delegation to `ColibriAgent` when `langgraph` is not installed
or `force_stdlib=True`.
"""
from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TypedDict,
    Union,
)

from .config import AgentConfig
from .gateway import ColibriGatewayPort
from .session import SessionStore, SqliteSessionStore
from .tools import TOOL_SUITE_OBJ, ToolSuite

try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False


def has_langgraph() -> bool:
    """Return True if `langgraph` and `langchain_openai` are installed."""
    return _LANGGRAPH_AVAILABLE


class InternetAgentState(TypedDict, total=False):
    """State schema for the Colibri agent in LangGraph."""

    messages: List[Any]
    step_count: int
    max_steps: int
    tool_calls_log: List[Dict[str, Any]]
    force_finalize: bool
    intent: str
    thread_id: str


def build_internet_agent_graph(
    config: Optional[AgentConfig] = None,
    on_tool_call: Optional[Callable[[Dict[str, Any], str, float], None]] = None,
    on_token: Optional[Callable[[str], None]] = None,
    stream: bool = False,
    checkpointer: Optional[Union[SqliteSessionStore, SessionStore, Any]] = None,
    transport: Optional[ColibriGatewayPort] = None,
    force_stdlib: bool = False,
) -> Any:
    """Build or return the unified `ColibriAgent` runner or compiled StateGraph.

    Uses `langgraph.graph.StateGraph` compiled with `checkpointer` when
    installed, or returns `ColibriAgent` with native SQLite checkpointing
    and StateGraph-compatible execution interface.
    """
    from .graph import (
        SYSTEM_PROMPT,
        ColibriAgent,
        _SYNTHESIS_NUDGE,
        _colibri_extra_body,
        _load_tools_registry,
        classify_user_intent,
    )

    cfg = config or AgentConfig()
    active_cp = checkpointer or SqliteSessionStore()
    if not _LANGGRAPH_AVAILABLE or force_stdlib:
        return ColibriAgent(
            config=cfg,
            on_tool_call=on_tool_call,
            on_token=on_token,
            stream=stream,
            checkpointer=active_cp,
            transport=transport,
        )

    use_fast = bool(getattr(cfg, "fast_tool_pruning", False))
    _, _, get_lc_tools = _load_tools_registry(fast_mode=use_fast)
    lc_tools = get_lc_tools()
    extra_body = _colibri_extra_body(cfg)

    llm = ChatOpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key or "colibri",
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.request_timeout,
        streaming=bool(stream or on_token is not None or getattr(cfg, "stream_tts", False)),
        extra_body=extra_body,
    )
    llm_with_tools = llm.bind_tools(lc_tools) if lc_tools else llm

    def agent_node(state: InternetAgentState) -> Dict[str, Any]:
        msgs = list(state.get("messages") or [])
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT), *msgs]

        force_finalize = bool(state.get("force_finalize", False))
        active_llm = llm if force_finalize else llm_with_tools
        if stream or on_token is not None or getattr(cfg, "stream_tts", False):
            acc_chunk = None
            for chunk in active_llm.stream(msgs):
                c_text = str(getattr(chunk, "content", "") or "")
                if c_text and on_token is not None:
                    on_token(c_text)
                acc_chunk = chunk if acc_chunk is None else acc_chunk + chunk
            ai_msg = (
                AIMessage(
                    content=str(getattr(acc_chunk, "content", "") or ""),
                    tool_calls=list(getattr(acc_chunk, "tool_calls", None) or []),
                )
                if acc_chunk is not None
                else active_llm.invoke(msgs)
            )
        else:
            ai_msg = active_llm.invoke(msgs)

        if (
            isinstance(ai_msg, AIMessage)
            and not getattr(ai_msg, "tool_calls", None)
            and not force_finalize
            and isinstance(ai_msg.content, str)
        ):
            salvaged = ToolSuite.salvage_inline_calls(ai_msg.content)
            if salvaged:
                ai_msg.tool_calls = [
                    {
                        "id": tc["id"],
                        "name": tc["name"],
                        "args": tc["args"],
                        "type": "tool_call",
                    }
                    for tc in salvaged
                ]

        updates: List[BaseMessage] = []
        if not state.get("messages") or not isinstance(state["messages"][0], SystemMessage):
            updates.append(SystemMessage(content=SYSTEM_PROMPT))
        updates.append(ai_msg)
        return {"messages": updates}

    def tools_node(state: InternetAgentState) -> Dict[str, Any]:
        msgs = state.get("messages") or []
        last_msg = msgs[-1] if msgs else None
        raw_calls = getattr(last_msg, "tool_calls", None) or []

        normalized: List[Dict[str, Any]] = []
        for idx, tc in enumerate(raw_calls):
            normalized.append(
                {
                    "id": str(tc.get("id") or f"call_{idx}"),
                    "name": str(tc.get("name") or ""),
                    "args": tc.get("args", {}),
                }
            )

        exec_results = TOOL_SUITE_OBJ.execute(
            normalized,
            parallel=True,
            max_chars=cfg.max_tool_chars,
            on_tool_call=on_tool_call,
        )
        tool_messages: List[BaseMessage] = [
            ToolMessage(
                content=res["content"],
                tool_call_id=res["tool_call_id"],
                name=res["name"],
            )
            for res in exec_results
        ]
        prev_log = list(state.get("tool_calls_log") or [])
        prev_log.extend(exec_results)
        step_count = int(state.get("step_count") or 0) + 1
        return {
            "messages": tool_messages,
            "step_count": step_count,
            "tool_calls_log": prev_log,
        }

    def reflect_and_guard_node(state: InternetAgentState) -> Dict[str, Any]:
        step_count = int(state.get("step_count") or 0)
        max_steps = int(state.get("max_steps") or cfg.max_iterations)
        msgs = state.get("messages") or []
        last_msg = msgs[-1] if msgs else None
        last_content = str(getattr(last_msg, "content", "") or "")

        if step_count >= max_steps or "<tool_call>" in last_content:
            return {
                "force_finalize": True,
                "messages": [HumanMessage(content=_SYNTHESIS_NUDGE)],
            }
        return {"force_finalize": False}

    def route_after_agent(state: InternetAgentState) -> str:
        msgs = state.get("messages") or []
        last_msg = msgs[-1] if msgs else None
        force_finalize = bool(state.get("force_finalize", False))
        if force_finalize:
            return END
        if getattr(last_msg, "tool_calls", None):
            return "tools"
        last_content = str(getattr(last_msg, "content", "") or "")
        if "<tool_call>" in last_content:
            return "reflect_and_guard"
        return END

    def intent_router_node(state: InternetAgentState) -> Dict[str, Any]:
        msgs = list(state.get("messages") or [])
        last_user_content = ""
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                last_user_content = str(getattr(m, "content", "") or "")
                break
        detected_intent = state.get("intent") or classify_user_intent(last_user_content, cfg)
        return {"intent": detected_intent}

    graph = StateGraph(InternetAgentState)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("reflect_and_guard", reflect_and_guard_node)

    graph.add_edge(START, "intent_router")
    graph.add_edge("intent_router", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "reflect_and_guard": "reflect_and_guard",
            END: END,
        },
    )
    graph.add_edge("tools", "reflect_and_guard")
    graph.add_edge("reflect_and_guard", "agent")

    return graph.compile(checkpointer=active_cp)


__all__ = [
    "InternetAgentState",
    "build_internet_agent_graph",
    "has_langgraph",
]
