from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TurnMetrics:
    relevance_score: float
    used_memory: bool
    token_total: int
    memory_hits: dict[str, int]


def relevance_heuristic(response: str, expected_substring: Optional[str]) -> float:
    if expected_substring is None:
        return 1.0
    if not response:
        return 0.0
    return 1.0 if expected_substring.lower() in response.lower() else 0.0


def context_utilization_heuristic(state: Dict[str, Any]) -> bool:
    hits = state.get("memory_hits") or {}
    return any(int(v) > 0 for v in hits.values())


def token_usage_from_state(state: Dict[str, Any]) -> int:
    usage = state.get("usage") or {}
    if "total_tokens" in usage:
        return int(usage["total_tokens"] or 0)
    # fallback to context manager token count
    meta = state.get("context_meta") or {}
    return int(meta.get("tokens", 0) or 0)


def compute_turn_metrics(state: Dict[str, Any], expected_substring: Optional[str]) -> TurnMetrics:
    response = state.get("response", "") or ""
    rel = relevance_heuristic(response, expected_substring)
    used_mem = context_utilization_heuristic(state)
    token_total = token_usage_from_state(state)
    hits = state.get("memory_hits") or {"short_term": 0, "long_term": 0, "episodic": 0, "semantic": 0}
    return TurnMetrics(relevance_score=rel, used_memory=used_mem, token_total=token_total, memory_hits=hits)

