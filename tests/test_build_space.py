"""The Space bundle: right files in, secrets and probe collections out. Offline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_space", ROOT / "scripts" / "build_space.py")
build_space = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_space)


def test_bundle_has_the_app_and_no_secrets(tmp_path):
    files = build_space.assemble(out=tmp_path / "space", with_index=False)
    names = {p.relative_to(tmp_path / "space").as_posix() for p in files}
    assert {"Dockerfile", "README.md", "requirements.txt", "app.py",
            "src/policy_rag/graph_agent.py", "src/policy_rag/tools.py"} <= names
    assert not any(n.endswith(".env") or "ingest" in n or "evaluate" in n for n in names)
    readme = (tmp_path / "space" / "README.md").read_text(encoding="utf-8")
    assert "sdk: docker" in readme and "app_port: 8501" in readme


def test_runtime_modules_cover_every_import_of_the_app():
    # Every policy_rag module the app imports (directly or through others) must be shipped.
    import ast

    shipped = {m.removesuffix(".py") for m in build_space.RUNTIME_MODULES}
    pkg = ROOT / "src" / "policy_rag"
    for mod in shipped - {"__init__"}:
        tree = ast.parse((pkg / f"{mod}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                assert node.module.split(".")[0] in shipped, f"{mod} imports .{node.module}"


def test_index_copy_keeps_only_policy_chunks(tmp_path):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    src = QdrantClient(path=str(tmp_path / "src"))
    for name in ("policy_chunks", "probe_fixed"):
        src.create_collection(name, vectors_config=VectorParams(size=3, distance=Distance.COSINE))
        src.upsert(name, points=[PointStruct(id=i, vector=[1.0, i, 0.5], payload={"i": i})
                                 for i in range(300)])
    src.close()
    assert build_space.copy_index(tmp_path / "src", tmp_path / "dst") == 300
    dst = QdrantClient(path=str(tmp_path / "dst"))
    assert [c.name for c in dst.get_collections().collections] == ["policy_chunks"]
    assert dst.count("policy_chunks").count == 300
    dst.close()
