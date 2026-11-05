"""Client for Colibri's ``POST /v1/systemone`` zero-generation decision API.

Allows LangGraph nodes to perform calibrated probability routing (`choice`),
binary sufficiency checks (`noul`), and rubric grading (`score`) on ``qwen36-35b``
without generating output tokens.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .config import AgentConfig


@dataclass
class ChoiceDecision:
    """Parsed answer for a ``choice`` question on ``/v1/systemone``."""

    choice: str
    probabilities: Dict[str, float]
    confidence: float


@dataclass
class NoulDecision:
    """Parsed answer for a binary ``noul`` (yes/no) question on ``/v1/systemone``."""

    probability_yes: float

    @property
    def is_yes(self) -> bool:
        return self.probability_yes >= 0.5


@dataclass
class ScoreDecision:
    """Parsed answer for a multi-level ``score`` question on ``/v1/systemone``."""

    score: float
    probabilities: Dict[str, float]
    legend: Dict[str, str]
    confidence: float


class SystemOneClient:
    """Zero-dependency HTTP client for Colibri's ``POST /v1/systemone`` endpoint."""

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig.from_env()

    @property
    def endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/systemone"

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        prefix: Optional[str] = None,
        pin_state: Optional[bool] = None,
        cache_slot: Optional[int] = None,
        normalize: str = "sum",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send a batch of typed questions (`noul`, `choice`, `score`) over a shared `state`."""
        body: Dict[str, Any] = {
            "model": self.config.model,
            "state": state,
            "questions": dict(questions),
            "normalize": normalize,
            "cache_slot": self.config.routing_cache_slot if cache_slot is None else cache_slot,
        }
        if prefix is not None:
            body["prefix"] = prefix
        if pin_state is not None:
            body["pin_state"] = pin_state

        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        req = urllib.request.Request(
            self.endpoint, data=data, headers=headers, method="POST"
        )
        with urllib.request.urlopen(
            req, timeout=timeout or self.config.request_timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    def choose(
        self,
        state: Any,
        instructions: str,
        criteria: Mapping[str, Optional[str]],
        *,
        prefix: Optional[str] = None,
        cache_slot: Optional[int] = None,
    ) -> ChoiceDecision:
        """Ask a single ``choice`` question and return the top label + calibrated probabilities."""
        reply = self.decide(
            state=state,
            questions={
                "q": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": dict(criteria),
                }
            },
            prefix=prefix,
            cache_slot=cache_slot,
        )
        ans = reply.get("answers", {}).get("q", {})
        return ChoiceDecision(
            choice=str(ans.get("choice", "")),
            probabilities=dict(ans.get("probabilities") or {}),
            confidence=float(ans.get("confidence", 0.0)),
        )

    def noul(
        self,
        state: Any,
        instructions: str,
        criteria: Optional[Mapping[str, str]] = None,
        *,
        prefix: Optional[str] = None,
        cache_slot: Optional[int] = None,
    ) -> NoulDecision:
        """Ask a single binary ``noul`` question and return P(yes)."""
        q_spec: Dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            q_spec["criteria"] = dict(criteria)
        reply = self.decide(
            state=state,
            questions={"q": q_spec},
            prefix=prefix,
            cache_slot=cache_slot,
        )
        ans = reply.get("answers", {}).get("q", {})
        return NoulDecision(probability_yes=float(ans.get("noul", 0.0)))

    def score(
        self,
        state: Any,
        instructions: str,
        levels: Sequence[str],
        *,
        prefix: Optional[str] = None,
        cache_slot: Optional[int] = None,
    ) -> ScoreDecision:
        """Ask a single ``score`` question across ordered rubric levels."""
        reply = self.decide(
            state=state,
            questions={
                "q": {
                    "type": "score",
                    "instructions": instructions,
                    "criteria": list(levels),
                }
            },
            prefix=prefix,
            cache_slot=cache_slot,
        )
        ans = reply.get("answers", {}).get("q", {})
        return ScoreDecision(
            score=float(ans.get("score", 0.0)),
            probabilities=dict(ans.get("probabilities") or {}),
            legend=dict(ans.get("legend") or {}),
            confidence=float(ans.get("confidence", 0.0)),
        )

    def route_query(self, query: str) -> ChoiceDecision:
        """Classify a user query into the best execution route using System One mode."""
        return self.choose(
            state=query,
            instructions="Select the best primary execution strategy to answer this user request.",
            criteria={
                "direct": "Can be answered directly from general knowledge, casual conversation, or provided memory context without calling external tools.",
                "companion_fast": "Needs personal memory lookup/save, voice synthesis, weather, or quick math.",
                "web_research": "Needs current internet search, news, live websites, or multi-page synthesis.",
                "academic": "Needs scientific papers, preprints, or encyclopedic reference (ArXiv / Wikipedia).",
                "developer": "Needs open-source repositories, code, or HackerNews tech discussions.",
                "weather_or_math": "Needs live weather forecast or exact mathematical calculation.",
            },
        )

    def check_sufficiency(self, query: str, gathered_evidence: str) -> NoulDecision:
        """Evaluate whether gathered tool evidence is sufficient to answer the query."""
        return self.noul(
            state={"query": query, "evidence": gathered_evidence[:3500]},
            instructions="Is the gathered evidence sufficient and specific enough to accurately answer the query?",
            criteria={
                "true": "Evidence directly answers the user's question with facts or sources.",
                "false": "Evidence is missing key facts, empty, or requires another search.",
            },
        )
