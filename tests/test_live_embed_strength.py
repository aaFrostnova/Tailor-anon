"""Unit test for LiveMeasurer._embed native-knob strength + order (Phase 2 Task 1).

Mocks the fragments so this runs CPU-only, no GPU: asserts _embed (1) respects the
explicit `order` when embedding fragments in sequence -- checked via a GLOBAL shared
recorder across fragments (a per-fragment call list alone is vacuous here: each mock
only sees its own calls, so an _embed that ignored `order` entirely and just walked
`frags` would still pass) -- and (2) passes each fragment's OWN native strength to
`embed_with_target(..., strength=s)`, not a shared alpha / post-hoc scale_resid.
"""
import os, sys, numpy as np
from PIL import Image
sys.path.insert(0, "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")


def test_embed_uses_native_strength_and_order(monkeypatch=None):
    import live_measure as LM

    seq = []   # shared across ALL fragments: proves cross-fragment embed ORDER, not just per-fragment calls

    class _FakeFrag:
        def __init__(self, name): self.name = name
        def embed_with_target(self, pil, t, strength=1.0):
            seq.append((self.name, strength)); return pil   # identity; records (who, strength) globally

    lm = LM.LiveMeasurer.__new__(LM.LiveMeasurer)         # bypass __init__ (no GPU)
    lm.NB = 8
    lm.FR = {"VINE": _FakeFrag("VINE"), "VideoSeal": _FakeFrag("VideoSeal")}
    lm.scale_resid = lambda a, b, s: b
    lm.nested_vine_embed = lambda *a, **k: a[1]
    cover = Image.fromarray((np.ones((16, 16, 3)) * 100).astype(np.uint8))
    emb, tgts = lm._embed(cover, frags=["VINE", "VideoSeal"], nested=False,
                           strengths={"VINE": 0.4, "VideoSeal": 1.3}, order=["VideoSeal", "VINE"])
    # global order respected: fails if _embed ignores `order` (e.g. iterates frags/dict order instead)
    assert [n for n, _ in seq] == ["VideoSeal", "VINE"]
    # each fragment got its OWN native strength passed to embed_with_target
    assert dict((n, round(s, 4)) for n, s in seq) == {"VideoSeal": 1.3, "VINE": 0.4}
