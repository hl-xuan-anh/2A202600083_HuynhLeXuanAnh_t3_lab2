from __future__ import annotations

from typing import Protocol

from config import Settings
from memory.long_term_local import LocalJSONProfileMemory
from memory.long_term_redis import RedisProfileMemory


class ProfileMemory(Protocol):
    def save(self, user_id: str, item: dict) -> None: ...
    def load(self, user_id: str): ...
    def retrieve(self, user_id: str, query: str): ...
    def clear(self, user_id: str) -> None: ...


def build_profile_memory(settings: Settings) -> ProfileMemory:
    if settings.use_redis:
        return RedisProfileMemory(redis_url=settings.redis_url)
    return LocalJSONProfileMemory(path=settings.profile_json_path)

