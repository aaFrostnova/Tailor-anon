"""The deployed decoder runs its zero-bit presence tests at the REQUEST's false-positive budget.

Until 2026-09-09 `OursComposite.decode` always used the 1 percent default of `presence_detected`, so the
budget the solver reasoned about never reached the decoder (the certification re-thresholds recorded reads
per request, which kept the reported numbers right, but a deployment at 1e-4 or at 0.1 decoded at 1 percent).
The budget now comes from the configuration (`alpha`) or the call; the default is unchanged.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "defense"))
import numpy as np
import watermark_smt_v2 as W
from eval_matrix import OursComposite, presence_tau, _id_payload
from src.shortened_bch import ShortenedBCH


def _stub(agree=0.60):
    """A two-fragment decoder whose fragments both read the keyed codeword with `agree` of the bits right:
    below the identity line (no keyed pass) so the verdict rests on the presence tests alone."""
    c = OursComposite.__new__(OursComposite)
    c.sb = ShortenedBCH(); c.order = ["vine", "videoseal"]; c.geo = False; c.alpha = 0.01
    iid = "img_00001"; tx = c.sb.encode(_id_payload(iid, c.sb.data_bits))
    rng = np.random.default_rng(0); n = len(tx); flip = np.zeros(n, bool); flip[rng.choice(n, int(round((1 - agree) * n)), replace=False)] = True
    bits = np.where(flip, 1 - tx, tx); llr = np.where(bits == 1, 3.0, -3.0)
    c._frag_llr = lambda name, img, image_id: llr.copy()
    return c, (iid, tx)


def test_default_budget_is_one_percent():
    c, sec = _stub(0.60)
    assert presence_tau(2, 100, 0.01) > 0.60 > presence_tau(2, 100, 0.10) - 1e-9     # 0.65 and 0.60
    ba, det = c.decode(None, sec)
    assert abs(ba - 0.60) < 1e-9 and not det


def test_the_request_budget_reaches_the_presence_test():
    c, sec = _stub(0.60)
    assert c.decode(None, sec, alpha=0.10)[1]            # at a 10 percent budget the two-fragment threshold is 0.60
    assert not c.decode(None, sec, alpha=0.01)[1]
    c.alpha = 0.10
    assert c.decode(None, sec)[1]                        # or from the configuration


def test_frontend_config_carries_the_budget():
    cfg = W.frontend_config({"resync": True}, alpha=1e-4)
    assert cfg["alpha"] == 1e-4 and cfg["resync"] is True
    assert "alpha" not in W.frontend_config({"resync": True})
