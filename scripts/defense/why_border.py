"""WHY is VINE's energy on the border? Three discriminating experiments (all measured, no attacks):
E1 content-(in)dependence: embed the SAME message on covers {natural, gray128, white, black, uniform-noise}.
   If the border ring persists on flat/noise covers -> content-INDEPENDENT fixed template (architecture/training),
   not perceptual content-adaptive hiding.
E2 literal-edge vs learned-band: residual energy vs distance-to-nearest-edge (the ring is a square frame band).
   Padding/boundary CNN artifact -> spike in the outermost 1-3 px. Learned template -> a wider band a few % in.
   Report energy in outer 2px, in the 2-10% band, and the peak-energy distance.
E3 message dependence: on the natural cover embed messages A, B, and all-zeros. If the border PATTERN flips with
   the message (low spatial corr A-vs-B) it carries the payload there; compare border magnitude for zeros."""
import os, sys, glob, json, numpy as np
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"); sys.path.insert(0, "scripts/defense")
import torch
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
KEY = b"v5_key_encoder_master"; dev = "cuda"
sb = ShortenedBCH(); n = sb.n
V = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev)
OUT = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense"
FG = f"{OUT}/final_figs"; os.makedirs(FG, exist_ok=True)
print(f"[gpu] free={torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def target(iid):
    p, M = V.get_perm_M(iid); return apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)
def resid(cover_arr, tgt):
    c = Image.fromarray(cover_arr.astype(np.uint8))
    wm = to512(V.embed_with_target(c, tgt))
    return np.abs(np.asarray(wm, np.float64) - cover_arr).mean(axis=2)   # HxW residual magnitude
# distance-to-nearest-edge map (0 at edge .. 255 at center)
yy, xx = np.mgrid[0:512, 0:512]
D = np.minimum(np.minimum(yy, 511 - yy), np.minimum(xx, 511 - xx)).astype(int)
def edge_profile(res, nb=64):
    bins = np.linspace(0, 256, nb + 1); idx = np.clip(np.digitize(D.ravel(), bins) - 1, 0, nb - 1)
    prof = np.array([res.ravel()[idx == b].mean() if np.any(idx == b) else 0.0 for b in range(nb)])
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers.tolist(), prof.tolist()
def bandstats(res):
    tot = res.sum()
    outer2 = res[D < 2].sum() / tot                 # literal edge (0-2 px)
    band_2_10 = res[(D >= 2) & (D < 51)].sum() / tot  # 2px .. 10% (51px) frame band
    inner = res[D >= 51].sum() / tot                # central 80%
    # peak distance
    cc, pp = edge_profile(res); peak_d = cc[int(np.argmax(pp))]
    return {"frac_edge_0_2px": round(float(outer2), 4), "frac_band_2px_10pct": round(float(band_2_10), 4),
            "frac_center_beyond10pct": round(float(inner), 4), "peak_energy_dist_px": round(float(peak_d), 1)}

report = {}
# ---- E1: content dependence ----
nat = np.asarray(to512(Image.open(sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[0]).convert("RGB")), np.float64)
rng = np.random.RandomState(0)
COVERS = {"natural": nat, "gray128": np.full((512, 512, 3), 128.0), "white": np.full((512, 512, 3), 255.0),
          "black": np.full((512, 512, 3), 0.0), "noise": rng.randint(0, 256, (512, 512, 3)).astype(np.float64)}
tgtA = target("whyA")
report["E1_content"] = {}
resids = {}
for name, cov in COVERS.items():
    r = resid(cov, tgtA); resids[name] = r
    report["E1_content"][name] = bandstats(r)
    s = report["E1_content"][name]
    print(f"E1 {name:9}: edge0-2px={s['frac_edge_0_2px']:.3f}  band2px-10%={s['frac_band_2px_10pct']:.3f}  "
          f"center>10%={s['frac_center_beyond10pct']:.3f}  peak@{s['peak_energy_dist_px']:.0f}px", flush=True)
    np.save(f"{FG}/whyborder_resid_{name}.npy", r.astype(np.float32))

# ---- E2: fine edge profile (natural + gray, to show band vs literal edge) ----
report["E2_profile"] = {}
for name in ("natural", "gray128", "noise"):
    c, p = edge_profile(resids[name], nb=128); report["E2_profile"][name] = {"dist_px": c, "energy": p}

# ---- E3: message dependence (on natural cover) ----
rA = resids["natural"]
rB = resid(nat, target("whyB"))
r0 = resid(nat, apply_crypto(np.zeros(n, dtype=np.uint8), *V.get_perm_M("whyA")))  # embed all-zero codeword (A's key)
def bordercorr(x, y, mask):
    xv, yv = x[mask], y[mask]; xv = xv - xv.mean(); yv = yv - yv.mean()
    return float((xv * yv).sum() / (np.sqrt((xv**2).sum() * (yv**2).sum()) + 1e-9))
bm = D < 51  # border band mask
report["E3_message"] = {
    "border_mag_msgA": round(float(rA[bm].mean()), 4), "border_mag_msgB": round(float(rB[bm].mean()), 4),
    "border_mag_zeros": round(float(r0[bm].mean()), 4),
    "spatial_corr_A_vs_B_border": round(bordercorr(rA, rB, bm), 4),
    "spatial_corr_A_vs_zeros_border": round(bordercorr(rA, r0, bm), 4)}
print("E3 message-dependence:", report["E3_message"], flush=True)
json.dump(report, open(f"{OUT}/why_border.json", "w"), indent=2)

# ---- figure: residual heatmaps across covers + edge profiles ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig = plt.figure(figsize=(14, 4.4)); gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1.6], wspace=0.18)
for k, name in enumerate(["natural", "gray128", "white", "black", "noise"]):
    ax = fig.add_subplot(gs[k]); r = resids[name]
    ax.imshow(r, cmap="magma", vmin=0, vmax=np.percentile(r, 99.5)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(name, fontsize=10)
axp = fig.add_subplot(gs[5])
for name, col in [("natural", "#2a78d6"), ("gray128", "#eb6834"), ("noise", "#008300")]:
    c, p = report["E2_profile"][name]["dist_px"], np.array(report["E2_profile"][name]["energy"])
    axp.plot(c, p / (p.max() + 1e-9), color=col, lw=2, label=name)
axp.set_xlabel("distance to nearest edge (px)"); axp.set_ylabel("residual energy (norm)")
axp.set_title("energy vs distance-to-edge", fontsize=10, loc="left")
axp.legend(frameon=False, fontsize=9); axp.spines["top"].set_visible(False); axp.spines["right"].set_visible(False)
axp.axvspan(0, 51, color="#eee", zorder=0); axp.set_xlim(0, 256)
fig.suptitle("Why is VINE's energy on the border? Same message on 5 covers — the frame ring persists on flat/noise = content-INDEPENDENT template",
             fontsize=11.5, y=1.02)
fig.savefig(f"{FG}/fig_why_border.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("WHY_BORDER_DONE", flush=True)
