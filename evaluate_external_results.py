from __future__ import annotations

import argparse
import json
from pathlib import Path


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref, hyp):
    r = ref.strip().split()
    h = hyp.strip().split()
    return 0.0 if not r and not h else (1.0 if not r else lev(r, h) / len(r))


def cer(ref, hyp):
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    return 0.0 if not r and not h else (1.0 if not r else lev(r, h) / len(r))


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="fixtures/manifest.json")
    ap.add_argument("--results", required=True)
    ap.add_argument("--plan", default="TCC_EXECUTION_PLAN_v3.json")
    ap.add_argument("--out", default="evidence/external_eval")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    gates = plan["acceptance_gates"]
    fixtures = {x["id"]: x for x in manifest.get("fixtures", [])}

    rows = []
    for line in Path(args.results).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    seen = set()
    samples = []
    latency_rows = []
    for row in rows:
        fid = row.get("id")
        if fid not in fixtures:
            raise SystemExit(f"unknown fixture id: {fid}")
        if fid in seen:
            raise SystemExit(f"duplicate fixture id: {fid}")
        seen.add(fid)
        fx = fixtures[fid]
        hyp = str(row.get("hypothesis", "")).strip()
        rec = {
            "id": fid,
            "suite": fx["suite"],
            "language": fx["language"],
            "reference": fx["reference"],
            "hypothesis": hyp,
        }
        if fx["language"] == "en":
            rec["wer"] = wer(fx["reference"], hyp)
        elif fx["language"] == "ja":
            rec["cer"] = cer(fx["reference"], hyp)
        samples.append(rec)

        if row.get("release_to_final_ms") is not None or row.get("first_partial_ms") is not None:
            latency_rows.append({
                "id": fid,
                "release_to_final_ms": row.get("release_to_final_ms"),
                "first_partial_ms": row.get("first_partial_ms"),
            })

    missing = sorted(set(fixtures) - seen)
    english = [x["wer"] for x in samples if x.get("suite") == "general" and "wer" in x]
    japanese = [x["cer"] for x in samples if x.get("suite") == "general" and x.get("language") == "ja" and "cer" in x]
    technical = [x for x in samples if x.get("suite") == "technical"]
    technical_acc = None
    if technical:
        technical_acc = sum(1 for x in technical if x["reference"].lower() in x["hypothesis"].lower()) / len(technical)

    release = [x["release_to_final_ms"] for x in latency_rows if x.get("release_to_final_ms") is not None]
    partial = [x["first_partial_ms"] for x in latency_rows if x.get("first_partial_ms") is not None]

    accuracy = {
        "status": "COMPLETE" if not missing else "PARTIAL",
        "english_wer": sum(english) / len(english) if english else None,
        "japanese_cer": sum(japanese) / len(japanese) if japanese else None,
        "technical_key_term_accuracy": technical_acc,
        "samples": samples,
        "missing_fixture_ids": missing,
    }
    latency = {
        "status": "COMPLETE" if release else "NOT_PROVIDED",
        "first_partial_ms": {"p50": pct(partial, 50), "p95": pct(partial, 95), "p99": pct(partial, 99)},
        "release_to_final_ms": {"p50": pct(release, 50), "p95": pct(release, 95), "p99": pct(release, 99)},
        "samples": latency_rows,
    }

    failures = []
    if missing:
        failures.append("missing_fixture_results")
    if accuracy["english_wer"] is not None and accuracy["english_wer"] > gates["english_wer_max"]:
        failures.append("english_wer")
    if technical_acc is not None and technical_acc < gates["technical_key_term_accuracy_min"]:
        failures.append("technical_key_term_accuracy")
    if latency["release_to_final_ms"]["p50"] is not None and latency["release_to_final_ms"]["p50"] > gates["release_to_final_p50_ms_max"]:
        failures.append("release_to_final_p50")
    if latency["release_to_final_ms"]["p95"] is not None and latency["release_to_final_ms"]["p95"] > gates["release_to_final_p95_ms_max"]:
        failures.append("release_to_final_p95")

    # AC2 requires paired Aqua Japanese evidence and cannot be inferred from Qwen-only results.
    blocked = []
    if accuracy["japanese_cer"] is not None:
        blocked.append("japanese_cer_vs_aqua_missing_aqua_pair")
    if not release:
        blocked.append("streaming_latency_missing")

    status = "REFRAME" if failures else ("BLOCKED" if blocked else "PASS")
    branch = "REFRAME_EXTERNAL_RESULT_AC_FAIL" if failures else ("PARTIAL_EXTERNAL_EVIDENCE_BLOCKED" if blocked else "PASS")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "accuracy_metrics.json").write_text(json.dumps(accuracy, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "latency_metrics.json").write_text(json.dumps(latency, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = {
        "status": status,
        "branch": branch,
        "failures": failures,
        "blocked": blocked,
        "source": str(Path(args.results)),
        "rule": "external inference results are evaluated against frozen fixtures and frozen v3 ACs; no threshold mutation",
    }
    (out / "validation_result.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
