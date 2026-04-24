from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ConversationCase:
    case_id: str
    scenario: str
    turns: List[str]
    expected_substring: Optional[str]
    notes: str = ""


CASES: List[ConversationCase] = [
    ConversationCase(
        case_id="1",
        scenario="Profile recall: name after 6 turns",
        turns=[
            "My name is Linh.",
            "Let's talk about cooking.",
            "What's a quick dinner idea?",
            "Any tips for grocery shopping?",
            "How to store herbs?",
            "By the way, what is my name?",
        ],
        expected_substring="Linh",
        notes="Tests long-term persistence beyond short-term window.",
    ),
    ConversationCase(
        case_id="2",
        scenario="Conflict update: allergy correction (required)",
        turns=[
            "I'm allergic to cow's milk.",
            "I'm actually allergic to soy milk.",
            "What am I allergic to?",
        ],
        expected_substring="soy",
        notes="Must prioritize the corrected fact.",
    ),
    ConversationCase(
        case_id="3",
        scenario="Preference recall: concise answers",
        turns=[
            "I prefer concise answers.",
            "Explain memory systems in detail.",
            "What is my preference?",
        ],
        expected_substring="concise",
        notes="Routes preference intent to Redis profile.",
    ),
    ConversationCase(
        case_id="4",
        scenario="Episodic recall: last time experience",
        turns=[
            "Help me debug my Docker compose networking issue.",
            "Thanks, it worked!",
            "What did I say before / last time?",
        ],
        expected_substring="last",
        notes="Routes experience recall to episodic JSON.",
    ),
    ConversationCase(
        case_id="5",
        scenario="Semantic retrieval: prompt injection safety",
        turns=[
            "What does the FAQ say about prompt injection safety?",
        ],
        expected_substring="untrusted",
        notes="Routes factual recall to Chroma semantic memory.",
    ),
    ConversationCase(
        case_id="6",
        scenario="Semantic retrieval: memory type definitions",
        turns=[
            "Define short-term vs episodic memory (semantic).",
        ],
        expected_substring="Short-term",
        notes="Uses semantic corpus chunk.",
    ),
    ConversationCase(
        case_id="7",
        scenario="Context trimming: overflow keeps important facts",
        turns=[
            "My name is An.",
            " ".join(["filler"] * 2500),
            "What is my name?",
        ],
        expected_substring="An",
        notes="Forces token trimming; long-term facts should survive.",
    ),
    ConversationCase(
        case_id="8",
        scenario="Recent context: short-term window usage",
        turns=[
            "Remember this code: ABC123.",
            "What code did I just tell you?",
        ],
        expected_substring="ABC123",
        notes="Tests recent context retrieval (ConversationBuffer).",
    ),
    ConversationCase(
        case_id="9",
        scenario="Semantic + profile interplay",
        turns=[
            "My name is Minh.",
            "Explain semantic memory from the FAQ.",
            "What is my name?",
        ],
        expected_substring="Minh",
        notes="Uses both semantic and profile; name should come from Redis.",
    ),
    ConversationCase(
        case_id="10",
        scenario="Memory hit-rate diversity across intents",
        turns=[
            "I prefer Vietnamese examples.",
            "What does the FAQ say about episodic memory?",
            "What did we talk about last time?",
        ],
        expected_substring="episodic",
        notes="Preference -> Redis, factual -> Chroma, experience -> episodic JSON.",
    ),
]

