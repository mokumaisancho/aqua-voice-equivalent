from __future__ import annotations

import json
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
MODEL_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
EXPECTED_SHARDS = {
    "model-00001-of-00002.safetensors": "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6",
    "model-00002-of-00002.safetensors": "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc",
}


def resolve_pinned_model(identity_path: str = "evidence/model_identity.json") -> str:
    """Return one local directory containing both model and processor at the pinned revision."""
    p = Path(identity_path)
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            snap = Path(obj.get("snapshot_path", ""))
            if (
                obj.get("status") == "READY"
                and obj.get("model_id") == MODEL_ID
                and obj.get("revision") == MODEL_REVISION
                and snap.is_dir()
            ):
                return str(snap)
        except Exception:
            pass

    from huggingface_hub import snapshot_download
    return str(Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION)))
