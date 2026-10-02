from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    (d / "fixtures").mkdir()
    manifest = {
        "fixtures": [
            {"id":"en1","audio":"x.wav","reference":"hello world","language":"en","suite":"general"},
            {"id":"ja1","audio":"y.wav","reference":"こんにちは世界","language":"ja","suite":"general"},
            {"id":"tech1","audio":"z.wav","reference":"Qwen3-ASR","language":"en","suite":"technical"}
        ]
    }
    plan = {
        "acceptance_gates": {
            "english_wer_max":0.0623,
            "technical_key_term_accuracy_min":0.95,
            "release_to_final_p50_ms_max":450,
            "release_to_final_p95_ms_max":700
        }
    }
    (d / "fixtures" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (d / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    offline = [
        {"id":"en1","hypothesis":"hello world","mode":"offline"},
        {"id":"ja1","hypothesis":"こんにちは世界","mode":"offline"},
        {"id":"tech1","hypothesis":"Qwen3-ASR","mode":"offline"}
    ]
    streaming = [
        {"id":"en1","hypothesis":"hello world","mode":"streaming","first_partial_ms":100,"release_to_final_ms":200},
        {"id":"ja1","hypothesis":"こんにちは世界","mode":"streaming","first_partial_ms":120,"release_to_final_ms":220},
        {"id":"tech1","hypothesis":"Qwen3-ASR","mode":"streaming","first_partial_ms":110,"release_to_final_ms":210}
    ]
    for name, rows in [("offline.jsonl", offline), ("streaming.jsonl", streaming)]:
        with (d / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out = d / "evidence"
    p = subprocess.run([
        sys.executable, str(ROOT / "evaluate_external_results.py"),
        "--manifest", str(d / "fixtures" / "manifest.json"),
        "--results", str(d / "offline.jsonl"), str(d / "streaming.jsonl"),
        "--plan", str(d / "plan.json"),
        "--out", str(out),
    ], capture_output=True, text=True)

    validation = json.loads((out / "validation_result.json").read_text(encoding="utf-8"))
    accuracy = json.loads((out / "accuracy_metrics.json").read_text(encoding="utf-8"))
    latency = json.loads((out / "latency_metrics.json").read_text(encoding="utf-8"))

    # Japanese Aqua pair is intentionally absent, so fail-closed BLOCKED is correct.
    assert p.returncode == 2
    assert validation["status"] == "BLOCKED"
    assert validation["branch"] == "PARTIAL_EXTERNAL_EVIDENCE_BLOCKED"
    assert "japanese_cer_vs_aqua_missing_aqua_pair" in validation["blocked"]
    assert accuracy["english_wer"] == 0.0
    assert accuracy["technical_key_term_accuracy"] == 1.0
    assert latency["release_to_final_ms"]["p50"] == 210.0
    assert latency["release_to_final_ms"]["p95"] <= 220.0
    assert validation["offline_result_count"] == 3
    assert validation["streaming_result_count"] == 3

print("PASS")
