from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List

from memory.base import MemoryBackend


class ConversationBufferMemory(MemoryBackend[dict]):
    """
    In-memory short-term conversation buffer (ConversationBufferMemory-like).
    Stores a list of {"role": "...", "content": "..."} messages per user_id.
    """

    def __init__(self, max_messages: int = 12) -> None:
        self._max = max_messages
        self._buffers: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=self._max))

    def save(self, user_id: str, item: dict) -> None:
        self._buffers[user_id].append({"role": item.get("role", "user"), "content": item.get("content", "")})

    def load(self, user_id: str) -> List[dict]:
        return list(self._buffers[user_id])

    def retrieve(self, user_id: str, query: str) -> List[dict]:
        # For short-term, retrieval is simply the most recent window.
        return self.load(user_id)

    def clear(self, user_id: str) -> None:
        self._buffers[user_id].clear()


# Alias to match requirement naming in other modules
ConversationBuffer = ConversationBufferMemory

