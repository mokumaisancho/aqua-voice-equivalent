from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence" / "model_identity.json"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
MODEL_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
EXPECTED_SHARDS = {
    "model-00001-of-00002.safetensors": "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6",
    "model-00002-of-00002.safetensors": "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc",
}


def version(name):
    try:
        return md.version(name)
    except Exception:
        return None


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def blocked(reason: str, **extra) -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"status":"BLOCKED","model_id":MODEL_ID,"revision":MODEL_REVISION,"reason":reason,**extra}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 2


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        return blocked(f"huggingface_hub unavailable: {e}")

    try:
        snap = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION))
    except Exception as e:
        return blocked(f"pinned model resolve failed: {e}")

    shard_checks = []
    for rel, expected in EXPECTED_SHARDS.items():
        p = snap / rel
        if not p.exists():
            return blocked(f"pinned shard missing: {rel}")
        actual = hash_file(p)
        shard_checks.append({"path":rel,"expected_sha256":expected,"actual_sha256":actual,"match":actual==expected})
        if actual != expected:
            return blocked(f"pinned shard hash mismatch: {rel}", shard_checks=shard_checks)

    files = []
    for p in sorted(x for x in snap.rglob("*") if x.is_file()):
        rel = p.relative_to(snap).as_posix()
        files.append({"path": rel, "size": p.stat().st_size, "sha256": hash_file(p)})

    tree = hashlib.sha256()
    for row in files:
        tree.update(f"{row['path']}\0{row['size']}\0{row['sha256']}\n".encode())

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "qwen-asr": version("qwen-asr"),
        "transformers": version("transformers"),
        "torch": version("torch"),
        "vllm": version("vllm"),
    }
    runtime_hash = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()

    obj = {
        "status": "READY",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot_path": str(snap),
        "expected_shards": EXPECTED_SHARDS,
        "shard_checks": shard_checks,
        "artifact_tree_sha256": tree.hexdigest(),
        "files": files,
        "runtime": runtime,
        "runtime_hash": runtime_hash,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status":"READY","revision":MODEL_REVISION,"artifact_tree_sha256":obj["artifact_tree_sha256"],"runtime_hash":runtime_hash,"file_count":len(files)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
