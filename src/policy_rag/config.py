"""Settings loaded from the environment (.env), plus fixed project paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # PDFs you download (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"  # chunk files (git-ignored)
SOURCES_CSV = DATA_DIR / "sources.csv"  # manifest of every PDF (committed)
QDRANT_DIR = ROOT / ".qdrant"           # local Qdrant storage (git-ignored)
CACHE_DIR = ROOT / ".cache" / "llm"     # cached LLM responses (git-ignored)

# Hosted demo (Streamlit Community Cloud builds from the GitHub repo, where .qdrant/ is
# git-ignored): fall back to the committed copy in deploy/index/, which holds only the
# policy_chunks collection. A local checkout keeps using its own .qdrant/.
DEPLOY_INDEX = ROOT / "deploy" / "index"
if not QDRANT_DIR.exists() and DEPLOY_INDEX.exists():
    QDRANT_DIR = DEPLOY_INDEX


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gen_model: str = field(default_factory=lambda: _env("GEN_MODEL", "gemini-3.5-flash-lite"))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "gemini-3.5-flash-lite"))
    llm_max_rpm: int = field(default_factory=lambda: int(_env("LLM_MAX_RPM", "12")))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    # Groq's free tier caps gpt-oss-120b at 8K tokens/minute; a judge call is ~2K tokens.
    groq_max_rpm: int = field(default_factory=lambda: int(_env("GROQ_MAX_RPM", "3")))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))


settings = Settings()
