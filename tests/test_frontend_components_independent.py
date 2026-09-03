"""Each geometric front-end is a separate decision, so turning one on must not run the others.

The cascade used to collapse resync and nested into a single `geo` flag and then run all four of
its stages whenever either was set. That made the solver's answer untrue of what got deployed: a
request that selected resync alone still paid the VINE scale search and the blind angle sweep at
decode time, and the effect of those stages was credited to resync. These tests pin the stages
apart so the flags mean what the solver charges for.
"""
import os, sys, types, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "defense"))


def _stub(cfg):
    """An OursComposite with the stages instrumented and every model call stubbed out."""
    import eval_matrix as em
    o = object.__new__(em.OursComposite)
    o.order = ["vine", "trustmark"]
    o.dev = "cpu"
    o.ran = []
    o.nested = bool(cfg.get("nested", False))
    o.resync = bool(cfg.get("resync", False))
    o.fe_scale = bool(cfg.get("scale_search", o.nested))
    o.fe_angle = bool(cfg.get("angle_sweep", o.resync))
    o.fe_tile = bool(cfg.get("tile", False))
    o.geo = o.nested or o.resync or o.fe_scale or o.fe_angle or o.fe_tile
    class _T:
        def crop_recover(self, att, iid, perm, M, return_view=False):
            o.ran.append("S5")
            return (False, None) if return_view else False
    o._tiled = _T() if o.fe_tile else None
    o.frag = {"trustmark": type("F", (), {"get_perm_M": staticmethod(lambda i: (None, None))})()}
    o._vss, o._rr, o._rs, o._bc = 0.005, 180.0, 3.0, 10.0
    o._sync = object() if o.resync else None
    o._sync_rectify = lambda s, a, d: (o.ran.append("S1") or a, None)
    o._rot = lambda a, d: (o.ran.append("rot") or a)
    o._frag_llr = lambda n, p, i: [0.0]
    o._cv = lambda rl, iid: False                       # never accepts, so every stage runs to the end
    return o


class _Img:
    size = (512, 512)
    def crop(self, box): return self


def _stages(cfg, monkeypatch):
    import eval_matrix as em
    o = _stub(cfg)
    probe = types.ModuleType("src.angle_probe")
    probe.candidate_angles = lambda img, search, step, refine: (o.ran.append("S4") or [1.0])
    monkeypatch.setitem(sys.modules, "src.angle_probe", probe)
    em.OursComposite.geo_cascade(o, _Img(), "img_00000", None)
    return set(o.ran)


def test_resync_alone_does_not_run_the_scale_search(monkeypatch):
    ran = _stages({"resync": True, "nested": False, "scale_search": False}, monkeypatch)
    assert "S1" in ran, "resync must run the SyncSeal rectify"
    assert "S3" not in ran, "resync must not silently run the nested ring's scale search"


def test_nested_alone_does_not_run_syncseal_or_the_angle_sweep(monkeypatch):
    ran = _stages({"nested": True, "resync": False, "angle_sweep": False}, monkeypatch)
    assert "S1" not in ran, "the nested ring must not silently run SyncSeal"
    assert "S4" not in ran, "the nested ring must not silently run the blind angle sweep"


def test_angle_sweep_is_selectable_on_its_own(monkeypatch):
    """The blind probe reads the image, needs no embed-side mark, and costs no PSNR -- so it is a
    front-end the solver can select by itself."""
    ran = _stages({"resync": False, "nested": False, "angle_sweep": True}, monkeypatch)
    assert ran == {"S4", "rot"}, f"expected the angle sweep alone, ran {ran}"


def test_no_frontend_runs_nothing(monkeypatch):
    assert _stages({}, monkeypatch) == set()


def test_tiled_trustmark_is_its_own_stage(monkeypatch):
    """Spatial redundancy closes the TRANSLATION-crop column that nothing else in the cascade
    addresses, so it is a front-end the solver decides on, not something bundled with the rest."""
    ran = _stages({"tile": True}, monkeypatch)
    assert ran == {"S5"}, f"expected the window search alone, ran {ran}"
    assert "S5" not in _stages({"resync": True, "angle_sweep": False}, monkeypatch), \
        "no other front-end may silently run the window search"


def test_syncseal_model_is_not_loaded_without_resync():
    """Loading SyncSeal costs a GPU model; a config that never rectifies must not pay for it."""
    o = _stub({"angle_sweep": True})
    assert o._sync is None
