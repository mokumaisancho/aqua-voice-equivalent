# Mac M1 MLX Dictation Experimental Adapter

**Separate from the official Qwen3-ASR-1.7B hard-AC research runner.**
It deliberately does not turn the 4-bit backend into a PASS for the
original unquantized evaluation. No model fallback is allowed.

- Backend: mlx-qwen3-asr 0.4.4; quantized 1.7B (4-bit decoder / 8-bit encoder)
- Model: moona3k/mlx-qwen3-asr-1.7b-4bit
- Model revision: b2315a2537123353af98e37fc93eae078f1b7cf3
- Transport: local user-owned UNIX socket, chmod 600; no external API
- Startup: model loads once; multiple recognition calls reuse it
- Inputs: WAV path / optional English or Japanese / bounded domain context
- Output: JSON, latency in ms, model revision, raw transcript
- No NN post-correction or QSIV until a blinded holdout demonstrates improvement.

After preparing a Python 3.13 venv with `mlx-qwen3-asr`, download the
pinned model snapshot (revision above). Run the service with:

`python mac_mlx_dictation.py serve --model-path <absolute-snapshot-path>`

In another terminal:

`python mac_mlx_dictation.py client --audio <file.wav> --language Japanese --copy`

`--copy` copies the recognized text to the macOS clipboard. It does not
inject keystrokes. A Raycast binding can handle copy/paste UX.

To verify offline contract tests without model weights:

`python -m unittest discover -s tests -p 'test_mac_mlx_dictation.py' -v`

Known gaps: microphone-to-socket bridge, GUI binding, independent Japanese/
English holdout results, quantified 4-bit vs original-model quality,
100+ warm-run latency evidence, and real-user audio timing.

## Verified on Mac M1 (2026-10-11)

Pinned SHA256 integrity verified, persistent Session successfully loaded. English
and Japanese synthetic 3.7-second utterances transcribed in 0.48 to 0.60 seconds
once loaded. Domain context was necessary for exact Guidewire ClaimCenter
spelling in one test. Two non-domain synthetic utterances transcribed correctly
with domain context. Measured details are in
evidence/MLX_MAC_M1_2026-10-11.json.
No real speech quality inference or product-grade performance claim is made.

### Raycast integration (separate liveTranscript feature branch)
feat/use-aqua-mlx-service-20261011 adds explicit socket client routing to
the existing Raycast microphone recorder. To choose MLX, configure
~/Library/Caches/aqua_mlx_dictation/raycast.json:

{"backend":"mlx","python":"<aqua>/.venv/bin/python","client":"<aqua>/mac_mlx_dictation.py","context_file":"<optional-dictionary-terms-file>"}

The MLX service and Raycast extension must be running. The Raycast integration
retains Moonshine as the old explicitly unselected backend, not a silent
fallback. Full Raycast GUI paste has not been user-verified yet.
