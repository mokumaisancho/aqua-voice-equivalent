from __future__ import annotations

import hashlib
import json
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


def download(url: str, dst: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, dst.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


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

        dst = OUT / src["audio_filename"]
        if not dst.exists():
            download(src["audio_url"], dst)
        fixtures.append({
            "id": src["id"],
            "audio": dst.name,
            "reference": src["reference"],
            "language": src["language"],
            "suite": src["suite"],
            "sha256": sha256(dst),
            "corpus": src["corpus"],
            "license": src["license"],
            "source_url": src["audio_url"],
            "reference_source": src["reference_source"]
        })

    manifest = {
        "schema_version": 1,
        "frozen": True,
        "fixtures": fixtures,
        "deferred_external_fixtures": deferred
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "downloaded_or_present": len(fixtures),
        "deferred": len(deferred),
        "manifest": str(MANIFEST)
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
