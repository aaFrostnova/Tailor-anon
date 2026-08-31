"""Render the composite watermark pipeline as a 3-layer flowchart (encode/channel/decode)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
fig, ax = plt.subplots(figsize=(15, 19))
ax.set_xlim(0, 15); ax.set_ylim(0, 24); ax.axis("off")

C_ENC = "#dbe9ff"; C_VINE = "#cfe8d6"; C_TM = "#ffe2c2"; C_CH = "#ffd6d6"; C_DEC = "#e7dcff"; C_OUT = "#fff3b0"

def box(x, y, w, h, text, fc, fs=11, bold=False, ec="#333"):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", zorder=5)

def arr(x1, y1, x2, y2, style="-|>", color="#333", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=2))

def band(y0, y1, label, color):
    ax.add_patch(plt.Rectangle((0.15, y0), 14.7, y1 - y0, fc=color, ec="none", alpha=0.18, zorder=0))
    ax.text(0.45, y1 - 0.35, label, ha="left", va="top", fontsize=15, fontweight="bold", color="#444", zorder=1)

# ---------------- bands ----------------
band(16.4, 23.6, "① ENCODE  (embed)", C_ENC)
band(13.1, 16.2, "② CHANNEL  (attack / transmission)", C_CH)
band(0.4, 12.9, "③ DECODE  (detect + identify)", C_DEC)

# ---------------- ENCODE ----------------
box(7.5, 22.8, 7.6, 0.9, "image_id  --SHA-256-->  37-bit payload", C_ENC, 12)
arr(7.5, 22.35, 7.5, 21.85)
box(7.5, 21.4, 8.8, 0.95, "ShortenedBCH(100, 37, t=10) encode  ->  100-bit codeword  cw\n(37 data + 63 ECC; shortened BCH(127,64))", C_ENC, 11, bold=True)
# split
arr(7.5, 20.9, 4.0, 20.1); arr(7.5, 20.9, 11.0, 20.1)
box(4.0, 19.2, 6.2, 1.7,
    "VINE-R  branch\n(s_v, M_v) = HKDF-SHA512(key,\n\"vine\", image_id)\ntarget_v = apply_crypto(cw, s_v, M_v)\nSD-Turbo latent enc @256 -> residual",
    C_VINE, 10)
box(11.0, 19.2, 6.2, 1.7,
    "TrustMark-B  branch\n(s_t, M_t) = HKDF-SHA512(key,\n\"trustmark\", image_id)\ntarget_t = apply_crypto(cw, s_t, M_t)\nAdobe TrustMark enc -> pixel residual",
    C_TM, 10)
# alpha scale
box(4.0, 17.65, 6.2, 0.7, "x  alpha = 0.70   (residual scale)", C_VINE, 10)
box(11.0, 17.65, 6.2, 0.7, "x  alpha = 0.70   (residual scale)", C_TM, 10)
arr(4.0, 18.35, 4.0, 18.0); arr(11.0, 18.35, 11.0, 18.0)
# merge -> watermarked
arr(4.0, 17.3, 7.0, 16.85); arr(11.0, 17.3, 8.0, 16.85)
box(7.5, 16.6, 8.0, 0.7, "watermarked image   (sequential: VINE then TM, both @ alpha 0.70)", C_ENC, 11, bold=True)
ax.text(7.5, 16.0, "both fragments are LOW-freq & mutually non-interfering; complementary by EMBEDDING DOMAIN + training aug, not band",
        ha="center", va="center", fontsize=8.5, style="italic", color="#555")

# ---------------- CHANNEL ----------------
arr(7.5, 16.25, 7.5, 15.55)
box(7.5, 15.0, 11.5, 1.05,
    "noisy channel / adversary:  JPEG · blur · noise · brightness · contrast · erasing\nresize · CROP · ROTATE · REGEN · rinse · VAE-b/c · UnMarker · overwrite",
    C_CH, 10.5, bold=True)
ax.text(7.5, 13.7, "REGEN/rinse erase TrustMark (VINE survives)   |   CROP/ROTATE break VINE (TrustMark survives)  ->  complementary",
        ha="center", va="center", fontsize=9, style="italic", color="#a33")

# ---------------- DECODE ----------------
arr(7.5, 13.45, 7.5, 12.75)
box(7.5, 12.3, 9.6, 0.95,
    "RESYNC (optional, if base decode fails):  rotation search  arange(-30,30,3deg) = 21 cands\n-> accept a candidate ONLY if crypto-verify passes  (per-cand FPR ~ 2^-37  ->  zero false-accept)",
    C_DEC, 10, bold=True)
# split decoders
arr(7.5, 11.82, 4.0, 11.15); arr(7.5, 11.82, 11.0, 11.15)
box(4.0, 10.5, 6.2, 1.15, "VINE decoder (ConvNeXt)\n-> sigmoid probs  -> prob_to_llr\n-> align(s_v, M_v)  =>  LLR_v", C_VINE, 10)
box(11.0, 10.5, 6.2, 1.15, "TrustMark decoder\n-> logits (passthrough)\n-> align(s_t, M_t)  =>  LLR_t", C_TM, 10)
# fuse
arr(4.0, 9.92, 7.0, 9.25); arr(11.0, 9.92, 8.0, 9.25)
box(7.5, 8.85, 9.2, 1.0,
    "FUSE:  fuse_llrs  =  maximal-ratio combining  L = w_v.LLR_v + w_t.LLR_t\ncurrent: equal weight (w=1)   ->   [LEARNED context-gated head: in training]",
    C_DEC, 10.5, bold=True)
arr(7.5, 8.35, 7.5, 7.75)
box(7.5, 7.45, 5.0, 0.6, "fused codeword LLR  (length 100)", C_DEC, 10)
# dual detect split
arr(7.5, 7.15, 4.0, 6.5); arr(7.5, 7.15, 11.0, 6.5)
box(4.0, 5.85, 6.3, 1.2,
    "ZERO-BIT  (presence)\nfused bit-acc >= tau = 0.63\n(binom 1% FPR null)\n=> \"is it watermarked?\"",
    C_OUT, 10)
box(11.0, 5.85, 6.3, 1.2,
    "CRYPTO-ID  (identity)\nChase-II BCH decode (p=8, 256 pat)\n-> exact 37-bit match (FPR ~ 2^-37)\n=> \"which image_id?\"",
    C_OUT, 10)
# merge -> decision
arr(4.0, 5.25, 7.0, 4.55); arr(11.0, 5.25, 8.0, 4.55)
box(7.5, 4.15, 7.2, 0.85, "DECISION:  composite_or  =  ZERO-BIT  OR  CRYPTO-ID", C_DEC, 12, bold=True)
arr(7.5, 3.72, 7.5, 3.15)
box(7.5, 2.7, 6.4, 0.8, "detected (yes/no)  +  recovered image_id", C_OUT, 12, bold=True)

ax.text(7.5, 1.4, "Composite watermark pipeline:  VINE-R (regen-robust) + TrustMark-B (geometry-robust), shared BCH codeword, per-image crypto whitening, dual detection",
        ha="center", va="center", fontsize=10, color="#333")
ax.text(7.5, 0.95, "Crypto roles:  (encode) per-image perm+mask whitening = no systematic weak bit  |  (decode) align/undo + resync zero-false-accept gate + 2^-37 identity",
        ha="center", va="center", fontsize=9, style="italic", color="#555")

plt.tight_layout()
out = os.path.join(REPO, "results/defense/pipeline_flowchart.png")
plt.savefig(out, dpi=130, bbox_inches="tight"); print(f"[saved] {out}\nFLOW_DONE")
