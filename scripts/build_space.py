"""Assemble the Hugging Face Space for the demo, and optionally push it.

    python scripts/build_space.py                       # build into build/space/ and stop
    python scripts/build_space.py --push --space <user>/lending-policy-assistant

What goes into the Space: app.py, the runtime package (src/policy_rag), the Docker files from
deploy/space/, a copy of the search index holding only the `policy_chunks` collection (the
probe collections stay behind), and the LLM response cache, so the example questions are
answered without spending quota. What never goes: .env, the PDFs, evaluation data.

--push needs HF_TOKEN (a write token) and GEMINI_API_KEY in .env. The Gemini key is stored as
a Space *secret* through the Hub API; it is never written into a file.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "build" / "space"
COLLECTION = "policy_chunks"
# Runtime modules only: ingestion and chunking need pypdf/langchain, which the Space skips.
RUNTIME_MODULES = [
    "__init__.py", "config.py", "demo.py", "graph_agent.py", "guardrails.py", "llm.py",
    "rag.py", "react_agent.py", "retrieve.py", "tools.py", "tracing.py",
]
NEVER = {".env", ".env.local", "secrets.toml"}


def assemble(root: Path = ROOT, out: Path = OUT, with_index: bool = True) -> list[Path]:
    """Write the Space's files into `out` (emptied first). Returns the files written."""
    if out.exists():
        shutil.rmtree(out)
    (out / "src" / "policy_rag").mkdir(parents=True)

    for name in ("Dockerfile", "README.md", "requirements.txt"):
        shutil.copy2(root / "deploy" / "space" / name, out / name)
    shutil.copy2(root / "app.py", out / "app.py")
    for name in RUNTIME_MODULES:
        shutil.copy2(root / "src" / "policy_rag" / name, out / "src" / "policy_rag" / name)

    cache = root / ".cache" / "llm"
    if cache.exists():
        shutil.copytree(cache, out / ".cache" / "llm")
    if with_index:
        copy_index(root / ".qdrant", out / ".qdrant")

    files = [p for p in out.rglob("*") if p.is_file()]
    leaked = [p for p in files if p.name in NEVER]
    if leaked:
        raise RuntimeError(f"refusing to build: secret files in the Space: {leaked}")
    return files


def copy_index(src: Path, dst: Path) -> int:
    """Copy only the policy_chunks collection into a fresh local Qdrant store."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    source = QdrantClient(path=str(src))
    info = source.get_collection(COLLECTION)
    target = QdrantClient(path=str(dst))
    target.create_collection(COLLECTION, vectors_config=info.config.params.vectors)
    n, offset = 0, None
    while True:
        batch, offset = source.scroll(COLLECTION, limit=256, offset=offset,
                                      with_payload=True, with_vectors=True)
        if batch:
            target.upsert(COLLECTION, points=[
                PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in batch])
            n += len(batch)
        if offset is None:
            break
    source.close()
    target.close()
    return n


def check_token(api, namespace: str) -> None:
    """Fail early with a plain reason instead of the Hub's 403 on create_repo."""
    me = api.whoami()
    auth = (me.get("auth") or {}).get("accessToken") or {}
    role = auth.get("role", "?")
    print(f"Hugging Face account: {me.get('name')} | token: {auth.get('displayName', '?')} ({role})")
    if me.get("name") != namespace and namespace not in {o.get("name") for o in me.get("orgs", [])}:
        raise SystemExit(f"This token belongs to {me.get('name')!r}, not {namespace!r}. "
                         f"Use --space {me.get('name')}/lending-policy-assistant, or that account's token.")
    if role == "read":
        raise SystemExit("This is a Read token. Create a Write token on huggingface.co/settings/tokens "
                         "and put it in .env as HF_TOKEN.")
    if not me.get("emailVerified", True):
        raise SystemExit("Your Hugging Face email is not confirmed yet; new accounts cannot create "
                         "Spaces until it is. Click the link in Hugging Face's email, then retry.")


def push(out: Path, space: str) -> str:
    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    # override=True: a terminal opened before .env was edited (VS Code's useEnvFile) still holds
    # the old values in its environment; the file is the source of truth here.
    load_dotenv(ROOT / ".env", override=True)
    token, gemini = os.getenv("HF_TOKEN", "").strip(), os.getenv("GEMINI_API_KEY", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is not set in .env (create a write token on huggingface.co).")
    if not gemini:
        raise SystemExit("GEMINI_API_KEY is not set in .env.")
    api = HfApi(token=token)
    check_token(api, space.split("/")[0])
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_secret(space, "GEMINI_API_KEY", gemini)
    for key, value in {"GEN_MODEL": os.getenv("GEN_MODEL", "gemini-3.5-flash-lite"),
                       "LLM_MAX_RPM": "12", "DEMO_SESSION_QUESTIONS": "10",
                       "DEMO_DAILY_CALLS": "200"}.items():
        api.add_space_variable(space, key, value)
    api.upload_folder(repo_id=space, repo_type="space", folder_path=str(out),
                      commit_message="Deploy demo from GitHub repo")
    return f"https://huggingface.co/spaces/{space}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="upload to the Space (needs HF_TOKEN)")
    ap.add_argument("--space", default="", help="<user>/<space-name> on Hugging Face")
    args = ap.parse_args()
    if args.push and "/" not in args.space:
        ap.error("--push needs --space <user>/<space-name>")

    files = assemble()
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"Built {OUT} ({len(files)} files, {size:.1f} MB)")
    if args.push:
        print("Pushing; the Space then builds its Docker image (5-10 minutes)...")
        print("Space:", push(OUT, args.space))
    return 0


if __name__ == "__main__":
    sys.exit(main())
