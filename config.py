from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    repo_root: Path

    openai_api_key: str | None
    openai_base_url: str | None
    openai_model: str

    use_redis: bool
    redis_url: str
    profile_json_path: Path

    chroma_collection: str
    chroma_persist_dir: Path

    use_openai_embeddings: bool
    openai_embedding_model: str

    max_tokens: int


def get_settings(repo_root: Path | None = None) -> Settings:
    root = repo_root or Path(__file__).resolve().parent

    chroma_collection = os.getenv("CHROMA_COLLECTION", "lab17_semantic")
    chroma_persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", str(root / "data" / "chroma")))
    profile_json_path = Path(os.getenv("PROFILE_JSON_PATH", str(root / "data" / "profile_facts.json")))

    return Settings(
        repo_root=root,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        use_redis=(os.getenv("USE_REDIS", "false").lower() == "true"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        profile_json_path=profile_json_path,
        chroma_collection=chroma_collection,
        chroma_persist_dir=chroma_persist_dir,
        use_openai_embeddings=(os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        max_tokens=int(os.getenv("MAX_TOKENS", "1800")),
    )
