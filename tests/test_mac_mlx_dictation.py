import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mac_mlx_dictation as m


class FakeResult:
    text = "Guidewire ClaimCenter"
    language = "English"
    duration = 2.0


class FakeSession:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return FakeResult()


class DictationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.audio = Path(self.temp.name) / "sample.wav"
        self.audio.write_bytes(b"RIFF" + b"\0" * 80)

    def test_valid_and_provenance(self):
        s = FakeSession()
        got = m.evaluate({"audio": str(self.audio), "context": "ClaimCenter", "language": "English"}, s)
        self.assertEqual(got["text"], "Guidewire ClaimCenter")
        self.assertEqual(got["backend"], "mlx_quantized_4bit")
        self.assertEqual(got["model_revision"], m.MODEL_REVISION)
        self.assertEqual(s.calls[0][1], {"context": "ClaimCenter", "language": "English"})

    def test_invalid_language_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "language"):
            m.validate({"audio": str(self.audio), "language": "French"})

    def test_missing_audio_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "existing audio"):
            m.validate({"audio": "/not-an-audio.wav"})

    def test_oversized_context_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "2048"):
            m.validate({"audio": str(self.audio), "context": "x" * 2049})

    def test_unpinned_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "model ID/revision mismatch"):
            m.verify_model_pin(self.temp.name)

    def test_missing_socket_never_falls_back(self):
        with self.assertRaises(ConnectionError):
            m.client(Path(self.temp.name) / "absent.sock", str(self.audio), "", None, False)


if __name__ == "__main__":
    unittest.main()
