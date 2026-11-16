"""Day-1 gate: confirm the laptop can run every piece of week 1.

    python scripts/check_setup.py            # all checks, including 1 Gemini call
    python scripts/check_setup.py --offline  # skip the Gemini call

Each check prints PASS / FAIL with a hint. Exit code is non-zero if anything failed.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def wrap(fn):
        def run(*args):
            t0 = time.perf_counter()
            try:
                detail = fn(*args) or ""
                RESULTS.append((name, True, f"{detail} ({time.perf_counter() - t0:.1f}s)"))
            except Exception as e:  # noqa: BLE001 - report every failure, keep going
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                if "--debug" in sys.argv:
                    traceback.print_exc()

        return run

    return wrap


@check("Python version")
def python_version():
    v = sys.version_info
    if not ((3, 11) <= (v.major, v.minor) < (3, 14)):
        raise RuntimeError(f"Python {v.major}.{v.minor}; use 3.11, 3.12 or 3.13")
    return f"{platform.python_version()} on {platform.system()}"


@check("Memory")
def memory():
    try:
        import psutil  # optional

        total = psutil.virtual_memory().total / 2**30
        avail = psutil.virtual_memory().available / 2**30
        return f"{total:.1f} GB total, {avail:.1f} GB free"
    except ImportError:
        return "psutil not installed; skipped"


@check("Package imports")
def imports():
    import google.genai  # noqa: F401
    import langchain_text_splitters  # noqa: F401
    import pypdf  # noqa: F401
    import qdrant_client  # noqa: F401
    import sentence_transformers  # noqa: F401
    import torch

    cuda = torch.cuda.is_available()
    return f"torch {torch.__version__} ({'CUDA' if cuda else 'CPU only'})"


@check("Embeddings (BGE small on CPU)")
def embeddings():
    from sentence_transformers import SentenceTransformer

    from policy_rag.config import settings

    model = SentenceTransformer(settings.embed_model, device="cpu")
    vecs = model.encode(
        ["Maximum loan-to-value ratio for gold loans", "FOIR limit for personal loans"],
        normalize_embeddings=True,
    )
    return f"{settings.embed_model}: dim {vecs.shape[1]}"


@check("Qdrant local mode")
def qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(":memory:")
    client.create_collection("smoke", vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    client.upsert(
        "smoke",
        points=[PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"doc": "test"})],
    )
    hits = client.query_points("smoke", query=[0.1, 0.2, 0.3, 0.4], limit=1).points
    assert hits and hits[0].payload["doc"] == "test"
    return "in-memory collection: upsert + search OK"


@check("Gemini API (1 request)")
def gemini():
    from policy_rag.llm import generate

    r = generate("Reply with exactly: OK", use_cache=False)
    return f"{r.model} replied {r.text.strip()!r}"


def list_models_hint():
    """If the configured model name is wrong, show the Flash models this key can use."""
    try:
        from google import genai

        from policy_rag.config import settings

        client = genai.Client(api_key=settings.gemini_api_key)
        names = [m.name.removeprefix("models/") for m in client.models.list()]
        flash = sorted(n for n in names if "flash" in n)
        print("\n  Flash models available to your key:\n   ", "\n    ".join(flash))
    except Exception as e:  # noqa: BLE001
        print(f"\n  (Could not list models: {e})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip the Gemini call")
    ap.add_argument("--debug", action="store_true", help="print tracebacks")
    args = ap.parse_args()

    python_version()
    memory()
    imports()
    embeddings()
    qdrant()
    if not args.offline:
        gemini()

    print()
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:32} {detail}")
    failed = [r for r in RESULTS if not r[1]]
    if any(n.startswith("Gemini") and "not found" in d.lower() for n, ok, d in failed):
        list_models_hint()
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
