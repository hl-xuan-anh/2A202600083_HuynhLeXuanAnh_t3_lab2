from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


Intent = Literal["preference", "factual_recall", "experience_recall", "recent_context", "general"]


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    reason: str
    confidence: float


class RuleBasedIntentRouter:
    def classify(self, text: str) -> IntentResult:
        t = (text or "").strip().lower()

        # Direct profile/fact questions should route to long-term profile (Redis).
        if re.search(r"\b(my name|tên tôi|tôi tên|allergy|dị ứng|preference|tôi thích|ưu tiên)\b", t):
            return IntentResult(intent="preference", reason="profile fact keyword", confidence=0.95)

        if re.search(r"\b(prefer|preference|thích|ưu tiên)\b", t):
            return IntentResult(intent="preference", reason="preference keyword", confidence=0.9)

        if re.search(r"\b(last time|what did i say|before|hôm trước|lần trước|nhớ lần)\b", t):
            return IntentResult(intent="experience_recall", reason="experience recall keyword", confidence=0.9)

        if re.search(r"\b(remember|recent|context|we just|just told|just tell|just said|vừa nói|vừa bảo)\b", t):
            return IntentResult(intent="recent_context", reason="recent context keyword", confidence=0.7)

        if re.search(r"\b(who|what|when|where|define|explain|semantic|faq)\b", t) or "?" in t:
            return IntentResult(intent="factual_recall", reason="question-like / factual recall", confidence=0.6)

        return IntentResult(intent="general", reason="default", confidence=0.5)
