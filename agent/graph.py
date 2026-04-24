from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent.context_manager import ContextManager, ContextPack
from agent.llm import OpenAICompatibleLLM, build_messages
from agent.router import RuleBasedIntentRouter
from config import Settings, get_settings
from memory.episodic_json import EpisodicJSONMemory
from memory.long_term import build_profile_memory
from memory.semantic_chroma import ChromaSemanticMemory
from memory.short_term import ConversationBufferMemory


log = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    user_id: str
    user_input: str
    intent: str
    intent_meta: dict
    retrieved: dict
    context_system: str
    context_meta: dict
    response: str
    usage: dict
    memory_hits: dict


@dataclass(frozen=True)
class AgentConfig:
    enable_memory: bool = True
    max_short_term_messages: int = 12


class MultiMemoryAgent:
    def __init__(self, *, settings: Settings, config: AgentConfig, data_dir: Optional[Path] = None) -> None:
        self.settings = settings
        self.config = config

        root = settings.repo_root
        self.data_dir = data_dir or (root / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.short_term = ConversationBufferMemory(max_messages=config.max_short_term_messages)
        self.long_term = build_profile_memory(settings)
        self.episodic = EpisodicJSONMemory(path=self.data_dir / "episodes.jsonl")
        self.semantic = ChromaSemanticMemory(
            settings=settings,
            corpus_dir=self.data_dir / "semantic_corpus",
        )

        self.intent_router = RuleBasedIntentRouter()
        self.llm = OpenAICompatibleLLM(settings)
        self.context_manager = ContextManager(model=settings.openai_model, max_tokens=settings.max_tokens)

        self.graph = self._build_graph()

    @staticmethod
    def default(settings: Settings | None = None, *, enable_memory: bool = True) -> "MultiMemoryAgent":
        st = settings or get_settings()
        return MultiMemoryAgent(settings=st, config=AgentConfig(enable_memory=enable_memory))

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("classify_intent", self.classify_intent)
        g.add_node("retrieve_memory", self.retrieve_memory)
        g.add_node("merge_context", self.merge_context)
        g.add_node("generate_response", self.generate_response)

        g.set_entry_point("classify_intent")
        g.add_edge("classify_intent", "retrieve_memory")
        g.add_edge("retrieve_memory", "merge_context")
        g.add_edge("merge_context", "generate_response")
        g.add_edge("generate_response", END)
        return g.compile()

    def reset(self, user_id: str) -> None:
        self.short_term.clear(user_id)

    def clear_all(self, user_id: str) -> None:
        self.short_term.clear(user_id)
        self.long_term.clear(user_id)
        self.episodic.clear(user_id)
        self.semantic.clear(user_id)

    def run(self, *, user_id: str, user_input: str) -> AgentState:
        if self.config.enable_memory:
            self.short_term.save(user_id, {"role": "user", "content": user_input})

        state: AgentState = {"user_id": user_id, "user_input": user_input}
        return self.graph.invoke(state)

    # --- Graph nodes ---
    def classify_intent(self, state: AgentState) -> AgentState:
        text = state.get("user_input", "")
        res = self.intent_router.classify(text)
        state["intent"] = res.intent
        state["intent_meta"] = {"reason": res.reason, "confidence": res.confidence}
        return state

    def retrieve_memory(self, state: AgentState) -> AgentState:
        user_id = state["user_id"]
        query = state.get("user_input", "")

        hits: dict[str, Any] = {"short_term": [], "long_term": {}, "episodic": [], "semantic": []}
        mem_hits = {"short_term": 0, "long_term": 0, "episodic": 0, "semantic": 0}

        if not self.config.enable_memory:
            state["retrieved"] = hits
            state["memory_hits"] = mem_hits
            return state

        # Always retrieve recent context
        recent = self.short_term.retrieve(user_id, query)
        hits["short_term"] = recent
        mem_hits["short_term"] = len(recent)

        intent = state.get("intent", "general")
        if intent == "preference":
            profile = self.long_term.retrieve(user_id, query)
            hits["long_term"] = profile
            mem_hits["long_term"] = len(profile)
        elif intent == "experience_recall":
            epis = self.episodic.retrieve(user_id, query)
            hits["episodic"] = epis
            mem_hits["episodic"] = len(epis)
        elif intent == "factual_recall":
            sem = self.semantic.retrieve(user_id, query)
            hits["semantic"] = sem
            mem_hits["semantic"] = len(sem)
        else:
            # General: retrieve lightly from all
            hits["long_term"] = self.long_term.load(user_id)
            mem_hits["long_term"] = len(hits["long_term"])
            hits["episodic"] = self.episodic.load(user_id)[:3]
            mem_hits["episodic"] = len(hits["episodic"])
            hits["semantic"] = self.semantic.retrieve(user_id, query)[:2]
            mem_hits["semantic"] = len(hits["semantic"])

        state["retrieved"] = hits
        state["memory_hits"] = mem_hits
        return state

    def merge_context(self, state: AgentState) -> AgentState:
        user_id = state["user_id"]
        retrieved = state.get("retrieved", {}) or {}

        pack = ContextPack(
            long_term_facts=retrieved.get("long_term", {}) or {},
            semantic_hits=retrieved.get("semantic", []) or [],
            episodic_events=retrieved.get("episodic", []) or [],
            short_term_messages=retrieved.get("short_term", []) or [],
        )
        system_prompt, meta = self.context_manager.render(pack)
        state["context_system"] = system_prompt
        state["context_meta"] = meta
        return state

    def generate_response(self, state: AgentState) -> AgentState:
        user_id = state["user_id"]
        user_input = state.get("user_input", "")
        system_prompt = state.get("context_system", "")

        msgs = build_messages(system_prompt, user_input)
        result = self.llm.generate(msgs)
        state["response"] = result.text
        state["usage"] = result.usage or {}

        if self.config.enable_memory:
            self.short_term.save(user_id, {"role": "assistant", "content": result.text})

            # Update long-term profile facts/preferences in Redis (conflict: last write wins).
            facts = _extract_profile_facts(user_input)
            if facts:
                self.long_term.save(user_id, facts)

            # Append episodic event (compact) every turn.
            self.episodic.save(
                user_id,
                {
                    "at": _utc_now_iso(),
                    "summary": _summarize_turn(user_input),
                    "outcome": _summarize_turn(result.text),
                },
            )

        return state


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).isoformat()


def _summarize_turn(text: str, max_len: int = 140) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _extract_profile_facts(user_input: str) -> dict[str, Any]:
    text = (user_input or "").strip()
    lowered = text.lower()
    facts: dict[str, Any] = {}

    m = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z .'-]{1,60})", text, flags=re.IGNORECASE)
    if m:
        facts["name"] = m.group(1).strip().strip(".")

    m = re.search(r"\b(tôi|mình)\s+tên\s+(là\s+)?(.+)$", text, flags=re.IGNORECASE)
    if m:
        name = re.sub(r"[.?!]+$", "", m.group(3)).strip()
        if 1 <= len(name) <= 60:
            facts["name"] = name

    # Allergy + correction patterns
    m = re.search(r"\ballergic to\s+(.+)$", text, flags=re.IGNORECASE)
    if m:
        facts["allergy"] = re.sub(r"[.?!]+$", "", m.group(1)).strip()

    m = re.search(r"\bdị\s*ứng\s+(.+)$", lowered, flags=re.IGNORECASE)
    if m:
        allergy = re.sub(r"[.?!]+$", "", m.group(1)).strip()
        allergy = re.sub(r"\b(chứ|chớ)\b.*$", "", allergy, flags=re.IGNORECASE).strip()
        if allergy:
            facts["allergy"] = allergy

    m = re.search(r"\bnot\s+(.+?)\s+but\s+(.+)$", text, flags=re.IGNORECASE)
    if m and ("allerg" in lowered or "dị ứng" in lowered):
        facts["allergy"] = re.sub(r"[.?!]+$", "", m.group(2)).strip()

    m = re.search(r"không\s+phải\s+(.+?)\s+(mà|chứ)\s+(.+)$", lowered, flags=re.IGNORECASE)
    if m and ("dị ứng" in lowered):
        facts["allergy"] = re.sub(r"[.?!]+$", "", m.group(3)).strip()

    # Preference
    m = re.search(r"\bi prefer\s+(.+)$", text, flags=re.IGNORECASE)
    if m:
        facts["preference"] = re.sub(r"[.?!]+$", "", m.group(1)).strip()

    m = re.search(r"\btôi\s+thích\s+(.+)$", lowered, flags=re.IGNORECASE)
    if m:
        facts["preference"] = re.sub(r"[.?!]+$", "", m.group(1)).strip()

    return facts
