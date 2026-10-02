from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence" / "model_identity.json"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"


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


def main() -> int:
    try:
        from huggingface_hub import model_info, snapshot_download
    except Exception as e:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"status":"BLOCKED","reason":f"huggingface_hub unavailable: {e}"}, indent=2), encoding="utf-8")
        return 2

    try:
        info = model_info(MODEL_ID)
        revision = info.sha
        snap = Path(snapshot_download(MODEL_ID, revision=revision))
    except Exception as e:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"status":"BLOCKED","reason":f"model resolve failed: {e}"}, indent=2), encoding="utf-8")
        return 2

    files = []
    for p in sorted(x for x in snap.rglob("*") if x.is_file()):
        rel = p.relative_to(snap).as_posix()
        # Hash all resolved snapshot files once; this is intentionally expensive but reproducible.
        files.append({"path": rel, "size": p.stat().st_size, "sha256": hash_file(p)})

    tree = hashlib.sha256()
    for row in files:
        tree.update(f"{row['path']}\0{row['size']}\0{row['sha256']}\n".encode())

    obj = {
        "status": "READY",
        "model_id": MODEL_ID,
        "revision": revision,
        "snapshot_path": str(snap),
        "artifact_tree_sha256": tree.hexdigest(),
        "files": files,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "qwen-asr": version("qwen-asr"),
            "transformers": version("transformers"),
            "torch": version("torch"),
            "vllm": version("vllm")
        }
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status":"READY","revision":revision,"artifact_tree_sha256":obj["artifact_tree_sha256"],"file_count":len(files)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
