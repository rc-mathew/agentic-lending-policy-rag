"""The committed demo index: only policy_chunks, usable, no lock file."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("refresh", ROOT / "scripts" / "refresh_deploy_data.py")
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


def test_index_copy_keeps_only_policy_chunks(tmp_path):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    src = QdrantClient(path=str(tmp_path / "src"))
    for name in ("policy_chunks", "probe_fixed"):
        src.create_collection(name, vectors_config=VectorParams(size=3, distance=Distance.COSINE))
        src.upsert(name, points=[PointStruct(id=i, vector=[1.0, i, 0.5], payload={"i": i})
                                 for i in range(300)])
    src.close()
    assert refresh.copy_index(tmp_path / "src", tmp_path / "dst") == 300
    assert not (tmp_path / "dst" / ".lock").exists()
    dst = QdrantClient(path=str(tmp_path / "dst"))
    assert [c.name for c in dst.get_collections().collections] == ["policy_chunks"]
    assert dst.count("policy_chunks").count == 300
    dst.close()


def test_committed_demo_data_is_usable():
    # The hosted demo reads these; a broken or missing copy would only show after deploying.
    meta = json.loads((ROOT / "deploy" / "index" / "meta.json").read_text())
    assert list(meta["collections"]) == ["policy_chunks"]
    assert (ROOT / "deploy" / "index" / "collection" / "policy_chunks" / "storage.sqlite").stat().st_size > 1e6
    assert not (ROOT / "deploy" / "index" / ".lock").exists()
