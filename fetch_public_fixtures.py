from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "fixtures" / "source_catalog.json"
OUT = ROOT / "fixtures"
MANIFEST = OUT / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def download(url: str, dst: Path) -> None:
    data = fetch_bytes(url)
    if len(data) < 44:
        raise RuntimeError(f"download too small: {url}")
    dst.write_bytes(data)


def raw_reference_url(blob_url: str) -> str:
    if "/blob/" in blob_url:
        return blob_url.replace("/blob/", "/resolve/")
    return blob_url


def normalize_reference(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_transcript_table(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        rows[parts[0]] = parts[1].strip()
    return rows


def validate_reference(src: dict) -> tuple[str, str]:
    url = raw_reference_url(src["reference_source"])
    raw = fetch_bytes(url).decode("utf-8")
    rows = parse_transcript_table(raw)
    source_id = src["source_id"]
    if source_id not in rows:
        raise RuntimeError(f"reference id missing: {source_id} in {url}")
    authoritative = rows[source_id]
    if normalize_reference(authoritative) != normalize_reference(src["reference"]):
        raise RuntimeError(
            f"reference mismatch for {source_id}: catalog={src['reference']!r} authoritative={authoritative!r}"
        )
    return authoritative, url


def validate_wav(path: Path) -> None:
    header = path.read_bytes()[:12]
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise RuntimeError(f"not a RIFF/WAVE file: {path}")


def main() -> int:
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    fixtures = []
    deferred = []

    for src in data["sources"]:
        if "audio_url" not in src:
            deferred.append({
                "id": src["id"],
                "reason": "licensed archive requires local extraction; redistribution restricted",
                "archive_url": src.get("archive_url"),
                "reference": src["reference"],
                "selection_rule": src.get("selection_rule")
            })
            continue

        authoritative_reference, raw_ref_url = validate_reference(src)
        dst = OUT / src["audio_filename"]
        if not dst.exists():
            download(src["audio_url"], dst)
        validate_wav(dst)
        fixtures.append({
            "id": src["id"],
            "audio": dst.name,
            "reference": authoritative_reference,
            "language": src["language"],
            "suite": src["suite"],
            "sha256": sha256(dst),
            "corpus": src["corpus"],
            "license": src["license"],
            "source_url": src["audio_url"],
            "source_id": src["source_id"],
            "reference_source": raw_ref_url,
            "reference_verified": True
        })

    manifest = {
        "schema_version": 2,
        "frozen": True,
        "reference_verification": "authoritative corpus transcript checked at fetch time",
        "fixtures": fixtures,
        "deferred_external_fixtures": deferred
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "downloaded_or_present": len(fixtures),
        "verified_references": len(fixtures),
        "deferred": len(deferred),
        "manifest": str(MANIFEST)
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
