"""Unit test for LiveMeasurer._embed native-knob strength + order (Phase 2 Task 1).

Mocks the fragments so this runs CPU-only, no GPU: asserts _embed (1) respects the
explicit `order` when embedding fragments in sequence, and (2) passes each fragment's
OWN native strength to `embed_with_target(..., strength=s)` -- not a shared alpha /
post-hoc scale_resid.
"""
import os, sys, numpy as np
from PIL import Image
sys.path.insert(0, "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")


class _FakeFrag:
    def __init__(self, name): self.name = name; self.calls = []
    def embed_with_target(self, pil, t, strength=1.0):
        self.calls.append(("emb", strength)); return pil   # identity; record the strength it was given


def test_embed_uses_native_strength_and_order(monkeypatch=None):
    import live_measure as LM
    lm = LM.LiveMeasurer.__new__(LM.LiveMeasurer)         # bypass __init__ (no GPU)
    lm.NB = 8
    lm.FR = {"VINE": _FakeFrag("VINE"), "VideoSeal": _FakeFrag("VideoSeal")}
    lm.scale_resid = lambda a, b, s: b
    lm.nested_vine_embed = lambda *a, **k: a[1]
    cover = Image.fromarray((np.ones((16, 16, 3)) * 100).astype(np.uint8))
    emb, tgts = lm._embed(cover, frags=["VINE", "VideoSeal"], nested=False,
                           strengths={"VINE": 0.4, "VideoSeal": 1.3}, order=["VideoSeal", "VINE"])
    # order respected: VideoSeal embedded before VINE
    assert lm.FR["VideoSeal"].calls[0][0] == "emb" and lm.FR["VINE"].calls[0][0] == "emb"
    # each fragment got its OWN native strength passed to embed_with_target
    assert abs(lm.FR["VINE"].calls[0][1] - 0.4) < 1e-9
    assert abs(lm.FR["VideoSeal"].calls[0][1] - 1.3) < 1e-9
