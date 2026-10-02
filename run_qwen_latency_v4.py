from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
MODEL_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"


def load_audio(path: Path):
    import numpy as np
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        dur = len(wav) / float(sr)
        n = max(1, int(round(dur * 16000)))
        xo = np.linspace(0, dur, num=len(wav), endpoint=False)
        xn = np.linspace(0, dur, num=n, endpoint=False)
        wav = np.interp(xn, xo, wav).astype(np.float32)
    return wav


def one_run(model, wav, chunk_ms=250):
    step = max(1, int(16000 * chunk_ms / 1000))
    state = model.init_streaming_state(unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=2.0)
    t_start = time.perf_counter()
    first = None
    pos = 0
    release_ts = None

    while pos < len(wav):
        end = min(pos + step, len(wav))
        seg = wav[pos:end]
        pos = end
        due = t_start + (pos / 16000.0)
        remain = due - time.perf_counter()
        if remain > 0:
            time.sleep(remain)
        if pos >= len(wav):
            release_ts = time.perf_counter()
        model.streaming_transcribe(seg, state)
        if first is None and getattr(state, "text", "").strip():
            first = (time.perf_counter() - t_start) * 1000.0

    if release_ts is None:
        raise RuntimeError("release timestamp not established")
    model.finish_streaming_transcribe(state)
    final_ms = (time.perf_counter() - release_ts) * 1000.0
    return first, final_ms, getattr(state, "text", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="fixtures/manifest.json")
    ap.add_argument("--out", default="evidence/streaming_latency_runs.jsonl")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()
    if args.runs < 100:
        raise SystemExit("v4 requires --runs >=100")

    from qwen_asr import Qwen3ASRModel
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fixtures = [x for x in manifest.get("fixtures", []) if x.get("suite") == "general"]
    if not fixtures:
        raise SystemExit("no general fixtures")

    model = Qwen3ASRModel.LLM(
        model=MODEL_ID,
        revision=MODEL_REVISION,
        gpu_memory_utilization=0.8,
        max_new_tokens=32,
    )
    cache = {x["id"]: load_audio(Path(args.manifest).parent / x["audio"]) for x in fixtures}

    for i in range(args.warmup):
        fx = fixtures[i % len(fixtures)]
        one_run(model, cache[fx["id"]])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in range(args.runs):
            fx = fixtures[i % len(fixtures)]
            first, final, text = one_run(model, cache[fx["id"]])
            row = {
                "run": i + 1,
                "id": fx["id"],
                "measured": True,
                "real_time_paced": True,
                "clock_contract": "last-frame-arrival-before-final-chunk-inference",
                "chunk_ms": 250,
                "first_partial_ms": first,
                "release_to_final_ms": final,
                "final_text": text,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"status": "READY", "warmup": args.warmup, "measured_runs": args.runs, "out": str(out), "model_revision": MODEL_REVISION}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
