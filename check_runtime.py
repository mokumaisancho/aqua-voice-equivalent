#!/usr/bin/env python3
import importlib.util, json, platform, sys
report = {
  "python": sys.version.split()[0],
  "platform": platform.platform(),
  "modules": {m: bool(importlib.util.find_spec(m)) for m in ["torch","numpy","soundfile","qwen_asr","vllm"]},
  "qwen_streaming_ready": bool(importlib.util.find_spec("qwen_asr")) and bool(importlib.util.find_spec("vllm")) and sys.version_info[:2] == (3,12),
}
print(json.dumps(report, indent=2))
