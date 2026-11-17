"""Refresh the search index the hosted demo reads from the repo: deploy/index/.

    python scripts/refresh_deploy_data.py

Streamlit Community Cloud builds the app from the GitHub repo, where .qdrant/ and .cache/ are
git-ignored, so the demo uses a committed copy instead (see config.py). This rewrites it from
your local index, keeping only the `policy_chunks` collection (the probe collections stay
behind: 12 MB instead of 46 MB). Run it after rebuilding the index, then commit deploy/.

The LLM response cache is not published: every prompt change invalidates it, so a committed
copy would be stale and only add weight.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "policy_chunks"


def copy_index(src: Path, dst: Path) -> int:
    """Copy only the policy_chunks collection into a fresh local Qdrant store at `dst`."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    if dst.exists():
        shutil.rmtree(dst)
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
    (dst / ".lock").unlink(missing_ok=True)  # a lock file is per process, never committed
    return n


def main() -> int:
    n = copy_index(ROOT / ".qdrant", ROOT / "deploy" / "index")
    print(f"deploy/index: {n} chunks. Now commit deploy/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
