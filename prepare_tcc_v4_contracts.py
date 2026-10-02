from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
FIXTURES = ROOT / "fixtures"


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    plan = json.loads((ROOT / "TCC_EXECUTION_PLAN_v4.json").read_text(encoding="utf-8"))
    catalog = json.loads((FIXTURES / "source_catalog.json").read_text(encoding="utf-8"))

    write_json(EVIDENCE / "scorer_contract.json", {
        "status":"READY","version":"scorer-v4.1","unicode_normalization":"NFKC",
        "local_smoke_metrics": {
            "english":{"case":"lowercase","punctuation":"strip","whitespace":"collapse","tokenization":"whitespace"},
            "japanese":{"case":"lowercase ASCII","punctuation":"strip","whitespace":"remove for CER","character_unit":"Unicode code point"}
        },
        "hard_ac_english_rule":"AC-ACC-EN uses pinned official OpenASR short-form scorer/protocol only; local FLEURS smoke WER cannot satisfy AC-ACC-EN.",
        "raw_asr_rule":"score raw ASR before downstream normalization"
    })

    write_json(EVIDENCE / "openasr_protocol.json", {
        "status":"READY",
        "repository":"huggingface/open_asr_leaderboard",
        "commit":"67e8bd6acea240819ad67080f6f31e15d4a90da5",
        "model_family":"qwen",
        "model":"Qwen/Qwen3-ASR-1.7B",
        "benchmark":"English short-form main public datasets",
        "dataset_aggregate":"hf-audio/open-asr-leaderboard",
        "special_dataset":"ArtificialAnalysis/Earnings22-Cleaned-AA-chunked",
        "official_entrypoints":["qwen/submit_jobs.sh","transformers/submit_jobs_qwen3asr.sh"],
        "local_evaluation_allowed":true,
        "hard_ac_input":"official macro-average English short-form WER",
        "note":"FLEURS fixtures in this repository remain functional smoke/EN-JA/domain fixtures and are not substituted for the official OpenASR English AC."
    })

    tech_cases=[]
    for src in catalog.get("sources",[]):
        if src.get("id")=="fleurs-en-tech-0193": tech_cases.append({"fixture_id":src["id"],"key_terms":["802.11a","802.11b","802.11g"]})
        if src.get("id")=="fleurs-ja-tech-0045": tech_cases.append({"fixture_id":src["id"],"key_terms":["asus","eee pc","2007","taipei it month"]})
    write_json(FIXTURES / "technical_term_annotations.json", {
        "status":"READY" if tech_cases else "BLOCKED","version":"tech-annotations-v4.1",
        "scoring":"exact normalized declared-key-term recall; whole-reference substring forbidden","cases":tech_cases
    })

    write_json(FIXTURES / "context_dictionary_manifest.json", {
        "status":"READY","version":"context-dictionary-v4.1",
        "rules":{"dictionary_target_recall_min":0.98,"false_bias_rate_max":0.01,"context_false_substitution_delta_pp_max":1.0,"paired":true},
        "cases":[
            {"id":"dict-positive-80211","targets":["802.11a","802.11b","802.11g"],"source_fixture":"fleurs-en-tech-0193","modes":["base","dictionary"]},
            {"id":"dict-positive-asus","targets":["asus","eee pc","taipei it month"],"source_fixture":"fleurs-ja-tech-0045","modes":["base","dictionary"]},
            {"id":"false-bias-negative-general-en","source_fixture":"fleurs-en-general-0000","forbidden_injection":["802.11a","eee pc"],"modes":["base","dictionary"]},
            {"id":"context-paired-general-ja","source_fixture":"fleurs-ja-general-0000","modes":["base","context"]}
        ]
    })

    write_json(FIXTURES / "semantic_oracle_manifest.json", {
        "status":"READY","version":"semantic-oracle-v4.1",
        "critical_mutations":["negation added_or_removed","number changed","named_entity replaced","technical_identifier changed","factual relation changed"],
        "allowed_changes":["punctuation","casing","explicit filler cleanup","whitespace"],
        "rule":"Any critical mutation => AC-SEM FAIL; missing semantic evaluation => BLOCKED"
    })

    write_json(EVIDENCE / "latency_contract.json", {
        "status":"READY","version":"latency-v4.1","clock":"time.perf_counter monotonic",
        "input_pacing":"wall-clock real-time; sleep to audio duration before each next chunk",
        "warmup_runs":5,"measured_warm_runs_min":100,
        "release_definition":"immediately after final audio frame delivery",
        "final_definition":"finish_streaming_transcribe returns stable final text",
        "metrics":["first_stable_partial_ms","release_to_final_ms"],"percentiles":[50,95,99],
        "thresholds_ms":{"release_to_final_p50":450,"release_to_final_p95":700}
    })

    write_json(EVIDENCE / "hard_ac_coverage_schema.json", {
        "status":"READY","version":"hard-ac-schema-v4.1","allowed_status":["PASS","FAIL","BLOCKED"],
        "forbidden_status":["NOT_RUN",None],"required_ac":list(plan["hard_ac"].keys()),
        "pass_rule":"every required_ac exactly once with status=PASS","missing_rule":"missing/NOT_RUN/null => BLOCKED_HARD_AC_NOT_RUN"
    })

    aqua=EVIDENCE / "aqua_pair_manifest.json"
    if not aqua.exists():
        write_json(aqua,{"status":"BLOCKED","reason":"paired Aqua Japanese measurements not yet supplied","required":["aqua_version","test_date","fixture_ids","references","aqua_hypotheses","aqua_cer"]})

    print(json.dumps({"status":"PASS","generated":[
        "evidence/scorer_contract.json","evidence/openasr_protocol.json","fixtures/technical_term_annotations.json",
        "fixtures/context_dictionary_manifest.json","fixtures/semantic_oracle_manifest.json","evidence/latency_contract.json",
        "evidence/hard_ac_coverage_schema.json","evidence/aqua_pair_manifest.json"]},ensure_ascii=False))
    return 0


if __name__=="__main__": raise SystemExit(main())
