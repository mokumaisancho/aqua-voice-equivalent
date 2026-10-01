import json, subprocess, sys, tempfile
from pathlib import Path

SRC=Path(__file__).resolve().parent

def test_fail_closed_fixture_block():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)
        (d/"TCC_EXECUTION_PLAN.json").write_text('{"plan_id":"TCC-QWEN-ASR-001"}',encoding="utf-8")
        (d/"fixtures").mkdir()
        (d/"fixtures"/"manifest.json").write_text(json.dumps({"fixtures":[{"id":"x","audio":"missing.wav","reference":"x","language":"en","suite":"general"}]}),encoding="utf-8")
        p=subprocess.run([sys.executable,str(SRC/"tcc_runner.py"),"--mode","preflight"],cwd=d,capture_output=True,text=True)
        val=json.loads((d/"evidence"/"validation_result.json").read_text(encoding="utf-8"))
        assert p.returncode==2
        assert val["status"]=="BLOCKED"
        assert val["branch"]=="BRANCH_FIXTURE_BLOCKED"
        required=["runtime_snapshot.json","fixture_manifest.json","accuracy_metrics.json","latency_metrics.json","resource_metrics.json","ablation_summary.json","semantic_integrity.json","validation_result.json","current_work_state.json"]
        assert all((d/"evidence"/x).exists() for x in required)

if __name__=="__main__":
    test_fail_closed_fixture_block()
    print("PASS")
