from __future__ import annotations

import re
import unicodedata

VERSION = "finalizer-v4.0"
EN_FILLERS = {"um", "uh", "erm", "hmm"}


def finalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text))
    # Only remove standalone English filler tokens. Japanese filler deletion is
    # intentionally disabled because no-space morphology can make it unsafe.
    tokens = s.split()
    kept = [t for t in tokens if t.lower().strip(".,!?;:") not in EN_FILLERS]
    return " ".join(kept)


def removed_tokens(raw: str, final: str) -> list[str]:
    raw_tokens = unicodedata.normalize("NFKC", str(raw)).split()
    final_tokens = unicodedata.normalize("NFKC", str(final)).split()
    out = []
    j = 0
    for tok in raw_tokens:
        if j < len(final_tokens) and tok == final_tokens[j]:
            j += 1
        else:
            out.append(tok)
    return out


def allowed_removed_token(token: str) -> bool:
    return token.lower().strip(".,!?;:") in EN_FILLERS
