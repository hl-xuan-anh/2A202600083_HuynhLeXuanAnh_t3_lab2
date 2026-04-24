from __future__ import annotations

from typing import Any, Dict

import redis

from memory.base import MemoryBackend


class RedisProfileMemory(MemoryBackend[dict]):
    """
    Long-term profile memory stored in Redis.
    Data model: hash at key "user:{user_id}:profile"
    Conflict handling: last write wins (overwrite the field).
    """

    def __init__(self, *, redis_url: str) -> None:
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def _key(self, user_id: str) -> str:
        return f"user:{user_id}:profile"

    def save(self, user_id: str, item: dict) -> None:
        if not item:
            return
        key = self._key(user_id)
        mapping = {str(k): "" if v is None else str(v) for k, v in item.items()}
        self._client.hset(key, mapping=mapping)

    def load(self, user_id: str) -> Dict[str, Any]:
        key = self._key(user_id)
        return dict(self._client.hgetall(key))

    def retrieve(self, user_id: str, query: str) -> Dict[str, Any]:
        profile = self.load(user_id)
        q = (query or "").lower()
        if not q:
            return profile
        # Preference intent usually: return preference-like keys first.
        preferred = {}
        for k, v in profile.items():
            if k.lower() in q or any(tok in k.lower() for tok in q.split()[:5]):
                preferred[k] = v
        return preferred or profile

    def clear(self, user_id: str) -> None:
        self._client.delete(self._key(user_id))

