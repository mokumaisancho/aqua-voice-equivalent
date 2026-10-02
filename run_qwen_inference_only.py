from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from qwen_model_pin import MODEL_ID, MODEL_REVISION, resolve_pinned_model


def language_arg(lang: str):
    if lang == "en":
        return "English"
    if lang == "ja":
        return "Japanese"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="fixtures/manifest.json")
    ap.add_argument("--out", default="predictions.jsonl")
    ap.add_argument("--streaming", action="store_true")
    args = ap.parse_args()

    from qwen_asr import Qwen3ASRModel
    model_path = resolve_pinned_model()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fixtures = manifest.get("fixtures", [])
    if not fixtures:
        raise SystemExit("no fixtures")

    rows = []
    if args.streaming:
        import numpy as np
        import soundfile as sf

        model = Qwen3ASRModel.LLM(
            model=model_path,
            gpu_memory_utilization=0.8,
            max_new_tokens=32,
        )

        for fx in fixtures:
            path = Path(args.manifest).parent / fx["audio"]
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

            state = model.init_streaming_state(
                unfixed_chunk_num=2,
                unfixed_token_num=5,
                chunk_size_sec=2.0,
            )
            step = 8000
            pos = 0
            first_partial_ms = None
            start = time.perf_counter()
            while pos < len(wav):
                seg = wav[pos:pos + step]
                pos += len(seg)
                model.streaming_transcribe(seg, state)
                if first_partial_ms is None and getattr(state, "text", "").strip():
                    first_partial_ms = (time.perf_counter() - start) * 1000.0
            release = time.perf_counter()
            model.finish_streaming_transcribe(state)
            release_to_final_ms = (time.perf_counter() - release) * 1000.0
            rows.append({
                "id": fx["id"],
                "hypothesis": getattr(state, "text", "").strip(),
                "first_partial_ms": first_partial_ms,
                "release_to_final_ms": release_to_final_ms,
                "mode": "streaming",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_path": model_path,
            })
    else:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        for fx in fixtures:
            path = Path(args.manifest).parent / fx["audio"]
            t0 = time.perf_counter()
            out = model.transcribe(
                audio=str(path),
                context=fx.get("context", ""),
                language=language_arg(fx["language"]),
                return_time_stamps=False,
            )
            elapsed = time.perf_counter() - t0
            if not isinstance(out, list) or len(out) != 1:
                raise RuntimeError(f"unexpected result shape for {fx['id']}")
            rows.append({
                "id": fx["id"],
                "hypothesis": str(out[0].text).strip(),
                "inference_s": elapsed,
                "mode": "offline",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_path": model_path,
            })

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "out": str(out_path), "streaming": args.streaming, "model_revision": MODEL_REVISION, "model_path": model_path}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
