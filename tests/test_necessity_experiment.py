import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
sys.path.insert(0, "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
from surrogate_model import synthetic_surrogate
from smt_necessity_experiment import run_necessity

def test_z3_at_least_best_grid_and_grid_grows():
    sg = synthetic_surrogate(attacks=("jpeg25","crop75"))
    scen = dict(min_psnr=0.0, max_ms=1e9, attacks=["jpeg25","crop75"], min_ba=0.60,
                allow_resync=True, allow_nested=True, min_bits=0, resolution=512)
    r = run_necessity(sg, scen, g_list=[3,6,12], orders=(True,))
    # z3's exact optimum is >= any grid's best-feasible (grid only approximates from below)
    for g in [3,6,12]:
        assert r["z3_psnr"] + 1e-6 >= r["grid_psnr"][g]
    # grid cost grows with resolution
    assert r["grid_size"][12] > r["grid_size"][6] > r["grid_size"][3]
    # refining the grid closes the gap (monotone non-increasing gap)
    gap = {g: r["z3_psnr"] - r["grid_psnr"][g] for g in [3,6,12]}
    assert gap[12] <= gap[3] + 1e-9
