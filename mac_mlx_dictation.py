"""Local, pinned MLX Qwen3-ASR dictation service for Apple Silicon.

Separate from the frozen official-Qwen research/evaluation pipeline.
Server loads one ASR model and accepts short UNIX-socket requests; no
cloud uploads, hidden model substitutions, or unverified text rewrites.
"""
from __future__ import annotations

import argparse
import hashlib
import wave
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time
from typing import Any

MODEL_ID = "moona3k/mlx-qwen3-asr-1.7b-4bit"
MODEL_REVISION = "b2315a2537123353af98e37fc93eae078f1b7cf3"
WEIGHTS_SHA256 = "5c9177b6fc67851d3c1af234ab8bd9d240faf07d08499bf3e751257d1ec88c5f"
SOCKET = Path.home() / "Library/Caches/aqua_mlx_dictation/asr.sock"
MAX_REQUEST = 8192
AUDIO_SUFFIXES = {".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a"}


def validate(payload: dict[str, Any]) -> tuple[Path, str, str | None]:
    if not isinstance(payload, dict):
        raise ValueError("request must be an object")
    file_value = payload.get("audio")
    if not isinstance(file_value, str) or not file_value:
        raise ValueError("audio path required")
    audio = Path(file_value).expanduser().resolve()
    if audio.suffix.lower() not in AUDIO_SUFFIXES or not audio.is_file():
        raise ValueError("existing audio file required")
    if audio.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("audio too large (>100 MB)")
    context = payload.get("context", "")
    language = payload.get("language")
    if not isinstance(context, str) or len(context) > 2048:
        raise ValueError("context must be <=2048 characters")
    if language not in (None, "Japanese", "English"):
        raise ValueError("language must be Japanese, English, or omitted")
    return audio, context, language


def verify_model_pin(model_path: str) -> None:
    p = Path(model_path).expanduser().resolve()
    if (p.name != MODEL_REVISION or p.parent.name != "snapshots"
            or p.parent.parent.name != "models--moona3k--mlx-qwen3-asr-1.7b-4bit"):
        raise ValueError("model ID/revision mismatch: fail closed")
    weights = p / "weights.safetensors"
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError("model weights SHA256 mismatch: fail closed")


def audio_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            return round(wav.getnframes() / wav.getframerate(), 3)
    except (OSError, wave.Error, ZeroDivisionError):
        return None


def evaluate(payload: dict[str, Any], model: Any) -> dict[str, Any]:
    audio, context, language = validate(payload)
    t0 = time.perf_counter()
    res = model.transcribe(str(audio), context=context, language=language)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "status": "PASS", "text": res.text, "language": res.language,
        "inference_ms": ms, "audio_sec": audio_duration(audio),
        "backend": "mlx_quantized_4bit", "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def recv_json(conn: socket.socket) -> dict[str, Any]:
    chunks = []
    length = 0
    while True:
        block = conn.recv(4096)
        if not block:
            break
        length += len(block)
        if length > MAX_REQUEST:
            raise ValueError("request exceeds size limit")
        chunks.append(block)
        if b"\n" in block:
            break
    raw = b"".join(chunks)
    return json.loads(raw.split(b"\n", 1)[0])


def serve(path: Path, model_path: str) -> None:
    verify_model_pin(model_path)
    from mlx_qwen3_asr import Session
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists():
        if not stat.S_ISSOCK(path.lstat().st_mode):
            raise RuntimeError("refuse to replace non-socket file")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.5)
                probe.connect(str(path))
            raise RuntimeError("dictation server already running")
        except (ConnectionRefusedError, FileNotFoundError, socket.timeout):
            path.unlink()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        os.chmod(path, 0o600)
        server.listen(2)
        print(json.dumps({"status": "LOADING", "model_revision": MODEL_REVISION}), flush=True)
        try:
            session = Session(model=model_path)
            print(json.dumps({"status": "READY", "socket": str(path)}), flush=True)
            while True:
                conn, _ = server.accept()
                with conn:
                    conn.settimeout(120)
                    try:
                        payload = recv_json(conn)
                        result = evaluate(payload, session)
                    except Exception as e:
                        result = {"status": "FAIL", "error": str(e), "type": type(e).__name__}
                    conn.sendall(json.dumps(result, ensure_ascii=False).encode("utf-8") + b"\n")
        finally:
            if path.exists() and stat.S_ISSOCK(path.lstat().st_mode):
                path.unlink()


def client(path: Path, audio: str, context: str, language: str | None, copy: bool) -> dict[str, Any]:
    if not path.exists():
        raise ConnectionError("MLX service not running: explicit failure; no ASR fallback")
    request = {"audio": audio, "context": context, "language": language}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(120)
        sock.connect(str(path))
        sock.sendall(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
        sock.shutdown(socket.SHUT_WR)
        payload = recv_json(sock)
    if copy and payload.get("status") == "PASS":
        subprocess.run(["pbcopy"], input=payload["text"], text=True, check=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    p = parser.add_subparsers(dest="command", required=True)
    s = p.add_parser("serve")
    s.add_argument("--socket", type=Path, default=SOCKET)
    s.add_argument("--model-path", default=os.environ.get("AQUA_MLX_SNAPSHOT", ""))
    c = p.add_parser("client")
    c.add_argument("--socket", type=Path, default=SOCKET)
    c.add_argument("--audio", required=True)
    c.add_argument("--context", default="")
    c.add_argument("--context-file", type=Path)
    c.add_argument("--language", choices=["Japanese", "English"])
    c.add_argument("--copy", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "serve":
            if not args.model_path or not Path(args.model_path).is_dir():
                raise FileNotFoundError("pinned local model required via --model-path or AQUA_MLX_SNAPSHOT")
            serve(args.socket, args.model_path)
        else:
            context = args.context
            if args.context_file:
                context = args.context_file.read_text(encoding="utf-8").strip()
            result = client(args.socket, args.audio, context, args.language, args.copy)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result.get("status") == "PASS" else 1
    except (OSError, RuntimeError, ValueError) as e:
        print(json.dumps({"status": "FAIL", "error": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
