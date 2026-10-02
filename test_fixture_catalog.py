from pathlib import Path
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fetch_public_fixtures", ROOT / "fetch_public_fixtures.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_parser_and_normalizer():
    table = "en_us_0000 hello   world\nja_jp_0000 こんにちは 世界\n"
    rows = mod.parse_transcript_table(table)
    assert rows["en_us_0000"] == "hello   world"
    assert mod.normalize_reference(rows["en_us_0000"]) == "hello world"
    assert rows["ja_jp_0000"] == "こんにちは 世界"


def test_catalog_contract():
    data = json.loads((ROOT / "fixtures" / "source_catalog.json").read_text(encoding="utf-8"))
    ids = set()
    for src in data["sources"]:
        assert src["id"] not in ids
        ids.add(src["id"])
        for k in ("id", "suite", "language", "corpus", "license", "reference"):
            assert src.get(k), f"missing {k}: {src}"
        if "audio_url" in src:
            for k in ("audio_filename", "reference_source", "source_id"):
                assert src.get(k), f"missing {k}: {src['id']}"
            assert "/resolve/" in src["audio_url"]
            assert src["reference_source"].endswith(".trans.txt")


if __name__ == "__main__":
    test_parser_and_normalizer()
    test_catalog_contract()
    print("PASS")
