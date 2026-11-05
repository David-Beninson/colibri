"""Conversation Context & KV-Slot Coordinator (`src/ai/context.py`).

Provides deterministic conversation context encapsulation, LLM payload formatting
with KV-cache prefix preservation, storage message sanitization, and slot mapping:
- Slot 0: Fast companion turns (bilingual chit-chat, personal memory recall)
- Slot 1: Research & execution turns (web search, deep research, vision/OCR, code)
- Slot 2: Zero-generation routing & classification (SystemOne / fast classifier)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


def _resolve_default_system_prompt() -> str:
    """Retrieve the deterministic companion system prompt without circular imports."""
    try:
        from .persona import build_system_prompt

        return build_system_prompt()
    except Exception:  # noqa: BLE001
        try:
            from .persona import COMPANION_PERSONA_PROMPT

            return str(COMPANION_PERSONA_PROMPT)
        except Exception:  # noqa: BLE001
            return ""


@dataclass
class ConversationContext:
    """Encapsulates turn state, KV slot selection, and storage vs wire message formats.

    Attributes:
        system_prompt: Byte-identical system prompt kept at wire index 0 for KV prefix reuse.
        messages: Conversation turn messages (role, content, tool calls).
        intent: Classified intent string ("conversation", "url_ingestion", etc.).
        memory_context: Speculative local RAG facts attached only to active LLM wire payload.
        slot: Active Colibri KV cache slot (0 = companion, 1 = research/tools, 2 = classification).
    """

    system_prompt: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    intent: str = "conversation"
    memory_context: Optional[str] = None
    slot: int = 0

    def __post_init__(self) -> None:
        if not self.system_prompt:
            self.system_prompt = _resolve_default_system_prompt()

    @staticmethod
    def clean_user_text(raw_text: str) -> str:
        """Strip any legacy or transient delimiter if encountered (`\n\n[LOCAL MEMORY CONTEXT]`)."""
        if not raw_text:
            return ""
        text = str(raw_text)
        if "\n\n[LOCAL MEMORY CONTEXT]" in text:
            text = text.split("\n\n[LOCAL MEMORY CONTEXT]", 1)[0]
        elif "[LOCAL MEMORY CONTEXT]" in text:
            text = text.split("[LOCAL MEMORY CONTEXT]", 1)[0]
        return text.strip()

    @staticmethod
    def resolve_slot(
        intent: str = "conversation",
        requires_research: bool = False,
        config: Optional[Any] = None,
    ) -> int:
        """Centralize KV-cache slot mapping:
        - Companion turns: config.cache_slot (default 0)
        - Research / tools: config.research_cache_slot (default 1)
        - Classification: config.routing_cache_slot (default 2)
        """
        slot_companion = getattr(config, "cache_slot", 0) if config else 0
        slot_research = getattr(config, "research_cache_slot", 1) if config else 1
        slot_routing = getattr(config, "routing_cache_slot", 2) if config else 2

        norm = (intent or "").strip().lower()
        if norm in ("classification", "classifier", "systemone"):
            return int(slot_routing)
        if requires_research or norm in ("research", "url_ingestion", "recipe_scaling"):
            return int(slot_research)
        return int(slot_companion)

    @classmethod
    def from_turn(
        cls,
        prompt: str,
        history: Optional[Sequence[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        memory_context: Optional[str] = None,
        intent: Optional[str] = None,
        requires_research: bool = False,
        slot: Optional[int] = None,
        config: Optional[Any] = None,
    ) -> ConversationContext:
        """Construct a clean `ConversationContext` for a new turn.

        Sanitizes historical user messages, extracts or applies memory context,
        and assigns the appropriate KV cache slot.
        """
        resolved_sys = system_prompt if system_prompt is not None else _resolve_default_system_prompt()
        clean_prompt = cls.clean_user_text(prompt)

        extracted_mem = memory_context
        if extracted_mem is None and "[LOCAL MEMORY CONTEXT]" in (prompt or ""):
            parts = (prompt or "").split("[LOCAL MEMORY CONTEXT]", 1)
            if len(parts) > 1 and parts[1].strip():
                extracted_mem = parts[1].strip()

        resolved_intent = intent or "conversation"
        resolved_slot = (
            slot
            if slot is not None
            else cls.resolve_slot(
                resolved_intent,
                requires_research=requires_research,
                config=config,
            )
        )

        msgs: List[Dict[str, Any]] = []
        if history:
            for m in history:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "")
                if role == "system":
                    continue
                item = dict(m)
                if role == "user" and isinstance(item.get("content"), str):
                    item["content"] = cls.clean_user_text(item["content"])
                msgs.append(item)

        if clean_prompt:
            # Avoid duplicating prompt if history already ended with the same cleaned message
            if (
                not msgs
                or msgs[-1].get("role") != "user"
                or cls.clean_user_text(str(msgs[-1].get("content") or "")) != clean_prompt
            ):
                msgs.append({"role": "user", "content": clean_prompt})
        elif not msgs:
            msgs.append({"role": "user", "content": ""})

        return cls(
            system_prompt=resolved_sys,
            messages=msgs,
            intent=resolved_intent,
            memory_context=extracted_mem,
            slot=resolved_slot,
        )

    def to_llm_payload(self, slot: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return wire messages formatted for `/v1/chat/completions`.

        Guarantees:
        - System prompt is byte-identical at index 0 for KV-cache prefix reuse.
        - Active user turn contains speculative memory context if present.
        - Previous user turns have transient memory annotations stripped.
        """
        if slot is not None:
            self.slot = slot

        payload: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        last_user_idx = -1
        for idx in range(len(self.messages) - 1, -1, -1):
            if self.messages[idx].get("role") == "user":
                last_user_idx = idx
                break

        for idx, msg in enumerate(self.messages):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "")
            if role == "system":
                continue
            entry = dict(msg)
            if role == "user":
                clean_content = self.clean_user_text(str(entry.get("content") or ""))
                if idx == last_user_idx and self.memory_context:
                    mem = self.memory_context.strip()
                    if "[LOCAL MEMORY CONTEXT]" in mem:
                        entry["content"] = f"{clean_content}\n\n{mem}"
                    else:
                        entry["content"] = f"{clean_content}\n\n[LOCAL MEMORY CONTEXT]\n{mem}"
                else:
                    entry["content"] = clean_content
            payload.append(entry)

        return payload

    def to_storage_messages(self) -> List[Dict[str, Any]]:
        """Return clean, persistent messages without system prompt or transient memory context annotations."""
        storage_messages: List[Dict[str, Any]] = []
        for m in self.messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "")
            if role == "system":
                continue
            item = dict(m)
            if role == "user" and "content" in item:
                item["content"] = self.clean_user_text(str(item["content"] or ""))
            storage_messages.append(item)
        return storage_messages
