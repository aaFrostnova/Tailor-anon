"""The single definition of every attack, shared by the measurement campaigns and the reported matrix.

An attack that is implemented twice is two attacks. This module exists because they had drifted: the
in-process surrogate campaigns restated the operators locally, and the restatements disagreed with the
ones the reported matrix uses -- crop read its fraction as an area rather than a side, rotation filled
the corners with black instead of reflecting, and the photometric and noise attacks were substantially
milder (brightness measured 23.9 dB against the cover where the reported operator measures 11.5 dB).
The solver was therefore planning against a weaker threat than the evaluation scored it on, and nothing
in either program could notice, because both were internally consistent.

Everything here is PIL-in / PIL-out so a campaign can call it in-process, while `attack_file` keeps the
path-based form the evaluation harness uses. The two share one implementation, so they cannot diverge.
"""
import io, os, tempfile
import numpy as np
from PIL import Image

# ---------------------------------------------------------------- geometry (pure PIL, no deps)
def center_crop_resize(img, frac):
    """Keep a centred `frac` of each SIDE and resize back -- so frac=0.75 keeps 56% of the area."""
    W, H = img.size; cw, ch = int(round(W * frac)), int(round(H * frac))
    l, t = (W - cw) // 2, (H - ch) // 2
    return img.crop((l, t, l + cw, t + ch)).resize((W, H), Image.BICUBIC)

def rotate_reflect(img, deg):
    """Rotate with reflection padding, then crop back: no black corners, and no content lost at the
    border -- but note the original frame's corners end up outside the returned view."""
    W, H = img.size; arr = np.array(img); pad = max(W, H) // 2
    refl = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    big = Image.fromarray(refl).rotate(deg, resample=Image.BICUBIC, expand=False)
    bw, bh = big.size; l, t = (bw - W) // 2, (bh - H) // 2
    return big.crop((l, t, l + W, t + H))

def resize_down_up(img, mid):
    W, H = img.size
    return img.resize((mid, mid), Image.BICUBIC).resize((W, H), Image.BICUBIC)

def crop_then_jpeg(img, frac=0.75, quality=25):
    out = center_crop_resize(img.convert("RGB"), frac)
    buf = io.BytesIO(); out.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

GEO = {
    "crop75": lambda im: center_crop_resize(im, 0.75),
    "crop50": lambda im: center_crop_resize(im, 0.50),
    "rot9":   lambda im: rotate_reflect(im, 9.0),
    "rs256":  lambda im: resize_down_up(im, 256),
    "hflip":  lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "crop_jpeg": crop_then_jpeg,
}

# ---------------------------------------------------------------- signal family (WAVES numerics)
# Parameters are the ones the reported matrix runs; they live here so a campaign cannot pick others.
SIGNAL_PARAMS = {"jpeg": dict(quality=25), "blur": dict(ksize=5, sigma=1),
                 "noise": dict(std=0.05), "bright": dict(factor=0.2),
                 "contrast": dict(factor=0.2), "bm3d": dict()}
_SIG_CACHE = {}

def _signal_attackers():
    if not _SIG_CACHE:
        from wmattacker import (GaussianBlurAttacker, GaussianNoiseAttacker, JPEGAttacker,
                                BrightnessAttacker, ContrastAttacker, BM3DAttacker)
        p = SIGNAL_PARAMS
        _SIG_CACHE.update({
            "jpeg":     JPEGAttacker(quality=p["jpeg"]["quality"]),
            "blur":     GaussianBlurAttacker(p["blur"]["ksize"], p["blur"]["sigma"]),
            "noise":    GaussianNoiseAttacker(std=p["noise"]["std"]),
            "bright":   BrightnessAttacker(p["bright"]["factor"]),
            "contrast": ContrastAttacker(p["contrast"]["factor"]),
            "bm3d":     BM3DAttacker(),
        })
    return _SIG_CACHE

ALIASES = {"jpeg25": "jpeg", "vaeB": "vae_b", "vaeC": "vae_c"}

def attack_pil(name, img, dev="cuda", _vae_cache={}):
    """Apply one attack to a PIL image and return a PIL image. Geometry is pure PIL; the signal family
    routes through the same attacker objects the harness uses, via a temp file, so there is exactly one
    parameterisation. VAE is lazy so a geometry-only campaign needs no compressai."""
    name = ALIASES.get(name, name)
    if name == "clean":
        return img.convert("RGB")
    if name in GEO:
        out = GEO[name](img.convert("RGB"))
        return out if out.size == img.size else out.resize(img.size, Image.BICUBIC)
    if name in ("vae_b", "vae_c"):
        if name not in _vae_cache:
            from wmattacker import VAEWMAttacker
            mn = {"vae_b": "bmshj2018-hyperprior", "vae_c": "cheng2020-anchor"}[name]
            _vae_cache[name] = VAEWMAttacker(mn, quality=3, metric="mse", device=dev)
        att = _vae_cache[name]
    else:
        att = _signal_attackers().get(name)
        if att is None:
            raise ValueError(f"unknown attack: {name}")
    d = tempfile.mkdtemp()
    try:
        ip, op = os.path.join(d, "i.png"), os.path.join(d, "o.png")
        img.convert("RGB").save(ip)
        att.attack([ip], [op])
        return Image.open(op).convert("RGB")
    finally:
        for f in ("i.png", "o.png"):
            try: os.remove(os.path.join(d, f))
            except OSError: pass
        try: os.rmdir(d)
        except OSError: pass
