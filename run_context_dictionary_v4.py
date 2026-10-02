from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from qwen_model_pin import MODEL_ID, MODEL_REVISION, resolve_pinned_model


def norm(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).lower().split())


def contains_term(text, term):
    t = norm(term)
    h = norm(text)
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", h) is not None


def units(s, lang):
    s = norm(s)
    if lang == "ja":
        s = "".join(ch for ch in s if not ch.isspace() and not unicodedata.category(ch).startswith("P"))
        return list(s)
    s = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in s)
    return s.split()


def edit_counts(ref, hyp, lang):
    r = units(ref, lang)
    h = units(hyp, lang)
    n, m = len(r), len(h)
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0)
    for i in range(1, n + 1):
        c, s, ins, d = dp[i - 1][0]
        dp[i][0] = (c + 1, s, ins, d + 1)
    for j in range(1, m + 1):
        c, s, ins, d = dp[0][j - 1]
        dp[0][j] = (c + 1, s, ins + 1, d)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cand = []
            c, s, ins, d = dp[i - 1][j]
            cand.append((c + 1, s, ins, d + 1))
            c, s, ins, d = dp[i][j - 1]
            cand.append((c + 1, s, ins + 1, d))
            c, s, ins, d = dp[i - 1][j - 1]
            if r[i - 1] == h[j - 1]:
                cand.append((c, s, ins, d))
            else:
                cand.append((c + 1, s + 1, ins, d))
            dp[i][j] = min(cand, key=lambda x: (x[0], x[1] + x[2] + x[3]))
    cost, sub, ins, dele = dp[n][m]
    denom = max(1, len(r))
    return {"error_rate": cost / denom, "substitution_rate": sub / denom, "substitutions": sub, "insertions": ins, "deletions": dele}


def lang_arg(x):
    return "English" if x == "en" else ("Japanese" if x == "ja" else None)


def tx(model, fx, context):
    out = model.transcribe(
        audio=str(Path("fixtures") / fx["audio"]),
        context=context,
        language=lang_arg(fx["language"]),
        return_time_stamps=False,
    )
    return str(out[0].text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="fixtures/manifest.json")
    ap.add_argument("--cases", default="fixtures/context_dictionary_manifest.json")
    ap.add_argument("--out", default="evidence/context_dictionary_results.jsonl")
    args = ap.parse_args()

    from qwen_asr import Qwen3ASRModel
    import torch

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    by_id = {x["id"]: x for x in man["fixtures"]}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_path = resolve_pinned_model()
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=256,
    )

    rows = []
    target_total = target_hit = 0
    forbidden_total = newly_induced = 0
    ctx_deltas = []

    for c in cases["cases"]:
        fx = by_id.get(c["source_fixture"])
        if not fx:
            continue
        base = tx(model, fx, "")

        if c["id"].startswith("dict-positive"):
            targets = c["targets"]
            prompt = "Preferred spellings and domain terms: " + ", ".join(targets)
            treated = tx(model, fx, prompt)
            for t in targets:
                target_total += 1
                target_hit += int(contains_term(treated, t))
            rows.append({"id": c["id"], "fixture_id": fx["id"], "mode": "dictionary", "base": base, "treated": treated, "targets": targets, "model_revision": MODEL_REVISION})

        elif c["id"].startswith("false-bias"):
            forbidden = c["forbidden_injection"]
            prompt = "Preferred spellings and domain terms: " + ", ".join(forbidden)
            treated = tx(model, fx, prompt)
            for t in forbidden:
                forbidden_total += 1
                base_has = contains_term(base, t)
                treated_has = contains_term(treated, t)
                newly_induced += int((not base_has) and treated_has)
            rows.append({"id": c["id"], "fixture_id": fx["id"], "mode": "false_bias", "base": base, "treated": treated, "forbidden": forbidden, "model_revision": MODEL_REVISION})

        else:
            treated = tx(model, fx, "Active application context: preserve the speaker's wording and language; do not invent facts.")
            b = edit_counts(fx["reference"], base, fx["language"])
            t = edit_counts(fx["reference"], treated, fx["language"])
            delta_pp = (t["substitution_rate"] - b["substitution_rate"]) * 100.0
            ctx_deltas.append(delta_pp)
            rows.append({
                "id": c["id"], "fixture_id": fx["id"], "mode": "context",
                "base": base, "treated": treated,
                "base_edit": b, "context_edit": t,
                "context_false_substitution_delta_pp": delta_pp,
                "model_revision": MODEL_REVISION,
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")

    metrics = {
        "status": "READY" if rows else "BLOCKED",
        "dictionary_target_recall": (target_hit / target_total if target_total else None),
        "false_bias_rate": (newly_induced / forbidden_total if forbidden_total else None),
        "context_false_substitution_delta_pp": (sum(ctx_deltas) / len(ctx_deltas) if ctx_deltas else None),
        "false_bias_definition": "newly induced forbidden target in treated output when absent from paired base output",
        "context_harm_definition": "paired substitution-rate delta, percentage points",
        "pairs": len(rows),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": model_path,
    }
    Path("evidence/context_dictionary_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0 if metrics["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
