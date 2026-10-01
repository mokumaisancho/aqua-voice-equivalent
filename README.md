# Aqua Voice Qwen PoC

PoC v0.1 fixes `Qwen/Qwen3-ASR-1.7B` as the primary ASR model and keeps model execution behind an adapter.

## Current vertical slice

WAV -> 16 kHz mono -> ASR adapter -> raw transcript -> deterministic dictionary -> bounded normalization -> final transcript -> latency/evidence JSON.

## Why offline adapter first

The official Qwen3-ASR streaming path currently requires the vLLM backend. The present ChatGPT sandbox runs Python 3.13 and does not have `qwen_asr` or `vllm`, so model execution is fail-closed rather than silently substituted. Use Python 3.12 for the Qwen runtime.

## Sandbox verification

```bash
python check_runtime.py
PYTHONPATH=src pytest -q
```

## Smoke test without model weights

```bash
python run_poc.py sample.wav --backend static --static-text "qwen three asr latency" --dict "qwen three asr=Qwen3-ASR"
```

## Qwen runtime target

Create a Python 3.12 environment and install the official package. Offline inference can use the transformers backend; streaming requires the vLLM extra. No alternative ASR model is substituted by this PoC.
