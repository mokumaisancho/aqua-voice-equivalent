from finalizer_v4 import allowed_removed_token, finalize, removed_tokens
from run_context_dictionary_v4 import contains_term, edit_counts

# Technical term matching: exact ASCII boundaries, not substring leakage.
assert contains_term("supports 802.11a and 802.11b", "802.11a")
assert not contains_term("supports x802.11a9", "802.11a")
assert contains_term("ASUSの Eee PC が発売された", "asus")
assert contains_term("ASUSの Eee PC が発売された", "eee pc")

# Paired substitution accounting: insertion/deletion must not masquerade as substitution.
a = edit_counts("alpha beta", "alpha gamma", "en")
assert a["substitutions"] == 1
b = edit_counts("alpha beta", "alpha beta extra", "en")
assert b["substitutions"] == 0 and b["insertions"] == 1

# Bounded finalizer: only allowlisted standalone filler deletion.
raw = "um keep 123 Qwen3-ASR"
final = finalize(raw)
assert final == "keep 123 Qwen3-ASR"
removed = removed_tokens(raw, final)
assert removed == ["um"]
assert all(allowed_removed_token(x) for x in removed)

# No Japanese filler deletion: unsafe morphology stays untouched.
ja = "あの人は来ます"
assert finalize(ja) == ja

print("PASS")
