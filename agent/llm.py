from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import Settings


ChatMessage = Dict[str, str]  # {"role": "...", "content": "..."}


@dataclass(frozen=True)
class LLMResult:
    text: str
    usage: dict[str, int]
    model: str


class OpenAICompatibleLLM:
    """
    OpenAI-compatible chat interface.
    - If OPENAI_API_KEY is set: calls OpenAI (or compatible base_url).
    - Otherwise: uses a deterministic local fallback for fully offline runs.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def generate(self, messages: List[ChatMessage], *, temperature: float = 0.2) -> LLMResult:
        if self._settings.openai_api_key:
            return self._openai_generate(messages, temperature=temperature)
        return self._fallback_generate(messages)

    def _openai_generate(self, messages: List[ChatMessage], *, temperature: float) -> LLMResult:
        from openai import OpenAI

        client = OpenAI(
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url or None,
        )
        resp = client.chat.completions.create(
            model=self._settings.openai_model,
            messages=messages,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
        }
        return LLMResult(text=text, usage=usage, model=self._settings.openai_model)

    def _fallback_generate(self, messages: List[ChatMessage]) -> LLMResult:
        system = ""
        user = ""
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "") or ""
            if m.get("role") == "user":
                user = m.get("content", "") or ""

        u = user.lower()
        facts = _parse_long_term_facts(system)
        recent_text = _extract_section(system, "RECENT CONTEXT")
        episodic_text = _extract_section(system, "EPISODIC EVENTS")
        semantic_text = _extract_section(system, "SEMANTIC HITS")

        if ("what is my name" in u) or ("tên tôi" in u) or ("my name" in u):
            name = facts.get("name")
            return LLMResult(text=(f"Your name is {name}." if name else "I don't know your name yet."), usage={}, model="fallback")

        if ("dị ứng" in u) or ("allerg" in u):
            allergy = facts.get("allergy")
            return LLMResult(text=(f"Your allergy is: {allergy}." if allergy else "I don't have an allergy saved yet."), usage={}, model="fallback")

        if ("preference" in u) or ("prefer" in u) or ("tôi thích gì" in u) or ("ưu tiên" in u):
            pref = facts.get("preference")
            return LLMResult(text=(f"Your preference is: {pref}." if pref else "I don't have a saved preference yet."), usage={}, model="fallback")

        if ("what code" in u) or ("code did i just" in u) or ("vừa nói" in u):
            # naive: pick last token-like substring from recent context
            token = _find_code_token(recent_text)
            return LLMResult(text=(token or "I couldn't find a recent code token."), usage={}, model="fallback")

        if ("last time" in u) or ("lần trước" in u) or ("what did i say" in u) or ("hôm trước" in u):
            line = _last_bullet_line(episodic_text)
            return LLMResult(text=(f"Last time: {line}" if line else "No episodic events saved yet."), usage={}, model="fallback")

        if ("faq" in u) or ("semantic" in u) or ("prompt injection" in u) or ("define" in u) or ("explain" in u):
            first = _first_bullet_line(semantic_text)
            if first:
                return LLMResult(text=f"From semantic memory: {first}", usage={}, model="fallback")

        # Default
        if facts:
            return LLMResult(
                text="Got it. I’ll use your saved facts and retrieved context to help with the next steps.",
                usage={},
                model="fallback",
            )
        return LLMResult(text="Got it. What would you like to do next?", usage={}, model="fallback")


def build_messages(system: str, user: str) -> list[ChatMessage]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _extract_section(system: str, header: str) -> str:
    # Sections are rendered by ContextManager with predictable headers.
    lines = (system or "").splitlines()
    start = None
    for i, line in enumerate(lines):
        if header in line:
            start = i + 1
            break
    if start is None:
        return ""
    out: list[str] = []
    stop_markers = [
        "LONG-TERM FACTS",
        "SEMANTIC HITS",
        "EPISODIC EVENTS",
        "RECENT CONTEXT",
    ]
    for line in lines[start:]:
        if any(m in line for m in stop_markers) and (header not in line):
            break
        out.append(line)
    return "\n".join(out).strip()


def _parse_long_term_facts(system: str) -> dict[str, str]:
    section = _extract_section(system, "LONG-TERM FACTS")
    facts: dict[str, str] = {}
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        body = line[1:].strip()
        if ":" not in body:
            continue
        k, v = body.split(":", 1)
        facts[k.strip().lower()] = v.strip()
    return facts


def _first_bullet_line(text: str) -> str | None:
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("-"):
            return line[1:].strip()
    return None


def _last_bullet_line(text: str) -> str | None:
    last = None
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("-"):
            last = line[1:].strip()
    return last


def _find_code_token(text: str) -> str | None:
    import re

    # Look for things like ABC123 or similar.
    m = re.findall(r"\b[A-Z]{2,}\d{2,}\b", text or "")
    if m:
        return m[-1]
    return None
