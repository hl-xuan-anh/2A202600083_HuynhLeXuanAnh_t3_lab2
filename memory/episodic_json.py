from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from memory.base import MemoryBackend


class EpisodicJSONMemory(MemoryBackend[dict]):
    """
    Episodic memory stored as JSONL. Each line is an event dict.
    Retrieval: keyword match over summary/outcome, plus recent fallback.
    """

    def __init__(self, path: Path, *, max_events_per_user: int = 5000) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_events_per_user = max_events_per_user
        if not self._path.exists():
            self._path.write_text("", encoding="utf-8")

    def save(self, user_id: str, item: dict) -> None:
        obj = dict(item)
        obj["user_id"] = user_id
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._compact_if_needed(user_id)

    def load(self, user_id: str) -> List[dict]:
        if not self._path.exists():
            return []
        out: List[dict] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("user_id") == user_id:
                out.append(obj)
        return out

    def retrieve(self, user_id: str, query: str) -> List[dict]:
        q = (query or "").lower().strip()
        events = self.load(user_id)
        if not events:
            return []
        if not q:
            return events[-5:]

        hits: List[dict] = []
        for e in reversed(events):
            hay = f"{e.get('summary','')} {e.get('outcome','')}".lower()
            if q in hay or any(tok in hay for tok in q.split()[:5]):
                hits.append(e)
            if len(hits) >= 5:
                break
        return list(reversed(hits)) if hits else events[-3:]

    def clear(self, user_id: str) -> None:
        if not self._path.exists():
            return
        kept: List[str] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("user_id") != user_id:
                kept.append(json.dumps(obj, ensure_ascii=False))
        self._path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    def _compact_if_needed(self, user_id: str) -> None:
        events = self.load(user_id)
        if len(events) <= self._max_events_per_user:
            return
        # Keep the newest N events for this user
        keep = events[-self._max_events_per_user :]
        # Rewrite file for all users (simple but OK for lab scale)
        all_lines = self._path.read_text(encoding="utf-8").splitlines()
        kept_other: List[dict] = []
        for line in all_lines:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("user_id") != user_id:
                kept_other.append(obj)

        merged = kept_other + keep
        merged.sort(key=lambda x: (x.get("user_id", ""), x.get("at", "")))
        self._path.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in merged) + "\n", encoding="utf-8")

