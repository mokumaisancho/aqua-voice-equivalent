from __future__ import annotations

import json
import tempfile
from pathlib import Path

import qwen_model_pin as pin

ROOT = Path(__file__).resolve().parent
EXPECTED_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"

assert pin.MODEL_ID == "Qwen/Qwen3-ASR-1.7B"
assert pin.MODEL_REVISION == EXPECTED_REVISION
assert set(pin.EXPECTED_SHARDS) == {
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
}
for h in pin.EXPECTED_SHARDS.values():
    assert len(h) == 64
    int(h, 16)

with tempfile.TemporaryDirectory() as td:
    snap = Path(td) / "snapshot"
    snap.mkdir()
    identity = Path(td) / "model_identity.json"
    identity.write_text(json.dumps({
        "status": "READY",
        "model_id": pin.MODEL_ID,
        "revision": pin.MODEL_REVISION,
        "snapshot_path": str(snap),
    }), encoding="utf-8")
    assert pin.resolve_pinned_model(str(identity)) == str(snap)

for name in ["run_qwen_inference_only.py", "run_qwen_latency_v4.py", "run_context_dictionary_v4.py"]:
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "resolve_pinned_model" in text, name
    assert "model_path = resolve_pinned_model()" in text, name
    assert "Qwen/Qwen3-ASR-1.7B" not in text, f"duplicate uncentralized model id in {name}"

etext = (ROOT / "evaluate_tcc_v4.py").read_text(encoding="utf-8")
assert "from qwen_model_pin import MODEL_REVISION" in etext
assert "revision mismatch" in etext

print("PASS")
