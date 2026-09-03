"""A measurement script must reach the dispatcher that covers every in-process attack.

`attack_pil` handles geometry, the signal family and neural compression; the diffusion family goes
through `attack_pil_any`, which wraps it. Importing the narrow one and then asking for `regen` raises at
run time -- after the job has been queued, scheduled and started. That happened twice, in two different
campaigns, for the same reason. The names are easy to confuse and the failure is late, so the invariant
is asserted here instead of being remembered.
"""
import ast, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k"
sys.path.insert(0, REPO)

DIFFUSION_NAMES = {"regen", "rinse2x", "rinse4x"}
SCRIPTS = ["ablate_cascade.py", "measure_prior_width.py", "solve_live_continuous.py",
           "certify_solver_eval.py", "make_signal_overlay.py", "make_geo_overlay.py"]


def _asks_for_diffusion(src):
    return any(f'"{n}"' in src or f"'{n}'" in src for n in DIFFUSION_NAMES)


def test_scripts_naming_a_diffusion_attack_use_the_full_dispatcher():
    from src.attacks import attack_pil, attack_pil_any, RINSE_PASSES, IN_PROCESS
    assert DIFFUSION_NAMES <= set(RINSE_PASSES), "the diffusion family is not in RINSE_PASSES"
    assert DIFFUSION_NAMES <= IN_PROCESS
    offenders = []
    for name in SCRIPTS:
        p = os.path.join(SCRATCH, name)
        if not os.path.exists(p):
            continue
        src = open(p).read()
        if not _asks_for_diffusion(src):
            continue
        if "attack_pil_any" not in src:
            offenders.append(name)
    assert not offenders, (
        "these ask for a diffusion attack but import only the narrow dispatcher: " + ", ".join(offenders))


def test_narrow_dispatcher_refuses_diffusion_loudly():
    """It must raise rather than silently return the unattacked image."""
    import numpy as np, pytest
    from PIL import Image
    from src.attacks import attack_pil
    im = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="unknown attack"):
        attack_pil("regen", im)
