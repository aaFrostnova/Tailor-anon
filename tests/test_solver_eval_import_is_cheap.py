"""Importing the scenario-evaluation module must not run the evaluation.

certify_solver_eval.py and solve_live_continuous.py import solver_eval_continuous for its scenario
distribution (SCEN, MEASURED, FIXED). The module used to run its 2000-scenario shard loop at import
time: every GPU job that imported it spent 2.7 h of CPU solving before doing its own work, and then
overwrote solver_eval_cont_shard00.json with a one-shard file on top of the array's shard 0.
"""
import os, sys, time, importlib, pytest

SC = "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k"
CF = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"


@pytest.mark.skipif(not os.path.exists(f"{SC}/solver_eval_continuous.py") or
                    not os.path.exists(f"{SC}/surrogate_canonical.json"),
                    reason="scratch evaluation scripts are not on this machine")
def test_import_does_not_solve_or_write(tmp_path):
    for p in (f"{CF}/scripts/defense", SC):
        if p not in sys.path: sys.path.insert(0, p)
    shard0 = f"{SC}/solver_eval_cont_shard00.json"
    before = os.stat(shard0).st_mtime if os.path.exists(shard0) else None
    argv = sys.argv; sys.argv = [argv[0]]              # the importers clear their argv the same way
    t0 = time.time()
    try:
        sys.modules.pop("solver_eval_continuous", None)
        mod = importlib.import_module("solver_eval_continuous")
    finally:
        sys.argv = argv
    assert time.time() - t0 < 60, "importing the module ran the evaluation"
    assert len(mod.SCEN) == 2000 and mod.MEASURED and mod.FIXED
    assert hasattr(mod, "main"), "the shard loop must live in main(), behind __main__"
    after = os.stat(shard0).st_mtime if os.path.exists(shard0) else None
    assert before == after, "importing the module rewrote shard 0"
