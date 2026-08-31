"""No program that feeds the surrogate or the reported matrix may define its own attack.

This is a structural guard, not a style check. Each of these programs used to restate the operators,
and the restatements drifted apart without either program being able to notice, because each was
internally consistent: crop read its fraction as an area in one and a side in the other, rotation
filled corners with black in one and reflected in the other, and brightness, contrast and noise were
measurably milder in the campaigns. The consequence was that the solver judged feasibility against an
easier threat than the evaluation scored it on, so every satisfaction number built on the surrogate was
inflated. Patching the values fixes one round of measurements; asserting that there is one definition
is what stops it recurring.

Scope: the programs whose output reaches the canonical surrogate or Table~1. One-off probes are not
covered -- they answer a question and are discarded, and constraining them buys nothing.
"""
import ast, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k"

# Programs whose measurements reach the canonical surrogate or the reported matrix.
CANONICAL_PRODUCERS = [
    os.path.join(REPO, "scripts/defense/eval_matrix.py"),
    os.path.join(SCRATCH, "make_surrogate_ext9.py"),
    os.path.join(SCRATCH, "make_geo_overlay.py"),
    os.path.join(SCRATCH, "make_signal_overlay.py"),
    os.path.join(SCRATCH, "make_frontend_overlay.py"),
    os.path.join(SCRATCH, "make_mildregen_overlay_n100.py"),
    os.path.join(SCRATCH, "make_regen_overlay_ext.py"),
]

# Function names that would constitute a private re-implementation of an attack.
FORBIDDEN = {
    "crop_keep", "crop_resize", "center_crop_resize", "rotate_reflect", "resize_down_up",
    "crop_then_jpeg", "rot9", "rot", "crop", "crop09", "jpeg", "jpeg25", "blur", "noise",
    "bright", "contrast", "vae", "vae_att", "bm3d", "hflip", "rs256", "resize512",
}
# Thin adapters that only re-shape an argument are allowed, provided the body calls the shared module.
ALLOWED_IF_DELEGATES = {"_crop_then_jpeg"}


def _defs(path):
    tree = ast.parse(open(path).read())
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _delegates_to_shared(node):
    """True when the body only forwards to src.attacks (an adapter, not a re-implementation)."""
    src = ast.unparse(node)
    return any(k in src for k in ("attack_pil", "_crop_then_jpeg_pil", "GEO[", "regen_attacker",
                                  "center_crop_resize", "rotate_reflect", "resize_down_up"))


def test_no_canonical_producer_defines_its_own_attack():
    offenders = []
    for path in CANONICAL_PRODUCERS:
        if not os.path.exists(path):
            continue
        for name, node in _defs(path).items():
            if name not in FORBIDDEN and name not in ALLOWED_IF_DELEGATES:
                continue
            if _delegates_to_shared(node):
                continue
            offenders.append(f"{os.path.basename(path)}:{node.lineno} def {name}")
    assert not offenders, (
        "these re-implement an attack instead of importing src.attacks:\n  " + "\n  ".join(offenders))


def test_every_canonical_producer_reaches_the_shared_module():
    """Importing it is what makes the guard above meaningful; a program with no attacks at all is fine."""
    missing = []
    for path in CANONICAL_PRODUCERS:
        if not os.path.exists(path):
            continue
        src = open(path).read()
        if "src.attacks" not in src and "from eval_matrix import" not in src:
            missing.append(os.path.basename(path))
    assert not missing, f"these never reach the shared attack definitions: {missing}"


def test_in_process_set_covers_what_the_surrogate_stores():
    """Every attack the canonical surrogate carries must be one the shared module can actually run,
    or an explicitly cross-environment one -- never a name that exists only in a campaign."""
    import json
    sys.path.insert(0, REPO)
    from src.attacks import IN_PROCESS, ALIASES, CROSS_ENV
    canon = os.path.join(SCRATCH, "surrogate_canonical.json")
    if not os.path.exists(canon):
        return                                   # nothing measured yet in this checkout
    attacks = json.load(open(canon)).get("attacks", [])
    unknown = [a for a in attacks
               if ALIASES.get(a, a) not in IN_PROCESS
               and not any(a.startswith(c) for c in CROSS_ENV)]
    assert not unknown, f"surrogate carries attacks no shared definition can produce: {unknown}"
