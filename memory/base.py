from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class MemoryBackend(ABC, Generic[T]):
    @abstractmethod
    def save(self, user_id: str, item: T) -> None: ...

    @abstractmethod
    def load(self, user_id: str) -> Any: ...

    @abstractmethod
    def retrieve(self, user_id: str, query: str) -> Any: ...

    @abstractmethod
    def clear(self, user_id: str) -> None: ...

