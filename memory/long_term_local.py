from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from memory.base import MemoryBackend


class LocalJSONProfileMemory(MemoryBackend[dict]):
    """
    Local file-based long-term profile store (no Redis required).
    Data model: a JSON dict keyed by user_id -> facts dict.
    Conflict handling: last write wins per key (overwrite).
    """

    def __init__(self, *, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("{}", encoding="utf-8")

    def _read_all(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_all(self, data: dict[str, dict[str, Any]]) -> None:
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self, user_id: str, item: dict) -> None:
        if not item:
            return
        data = self._read_all()
        user = data.get(user_id) or {}
        for k, v in item.items():
            user[str(k)] = "" if v is None else str(v)
        data[user_id] = user
        self._write_all(data)

    def load(self, user_id: str) -> Dict[str, Any]:
        return dict((self._read_all().get(user_id) or {}))

    def retrieve(self, user_id: str, query: str) -> Dict[str, Any]:
        profile = self.load(user_id)
        q = (query or "").lower().strip()
        if not q:
            return profile
        filtered: Dict[str, Any] = {}
        for k, v in profile.items():
            if k.lower() in q or any(tok in k.lower() for tok in q.split()[:5]):
                filtered[k] = v
        return filtered or profile

    def clear(self, user_id: str) -> None:
        data = self._read_all()
        if user_id in data:
            del data[user_id]
            self._write_all(data)

