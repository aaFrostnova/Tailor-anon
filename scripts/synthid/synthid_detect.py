"""Step 2 — SynthID detection wrapper + SELF-CALIBRATION.

Detection paths (pick one in detect()):
  A. Gemini API Q&A   -> scriptable, but UNOFFICIAL/black-box. MUST be calibrated first.
  B. Vertex AI verify -> authoritative + batchable, but access-gated (stub below).
  C. SynthID Detector portal (https://... ) -> authoritative, MANUAL, small-N spot-check only.

--calibrate is a HARD GATE: it measures the detector's own TPR (on clean SynthID images,
should be ~1) and FPR (on non-SynthID COCO, should be ~0). If TPR<~0.9 or FPR>~0.1, the
Gemini-QA detector is NOT usable for a quantitative robustness sweep -> use portal/Vertex.

  python synthid_detect.py --calibrate --pos results/synthid/wm --neg <coco_dir>
  # then it is imported by synthid_eval.py as detect(client, pil)->{0,1}
"""
import os, glob, time, argparse
from PIL import Image

DET_MODEL = "gemini-flash-latest"
DET_PROMPT = ("Was this image generated or edited by Google AI (does it carry a Google "
              "SynthID watermark)? Consider only the SynthID signal, not visual style. "
              "Answer with exactly one word: YES or NO.")

def make_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key: raise SystemExit("set GEMINI_API_KEY")
    return genai.Client(api_key=key)

def detect(client, pil, retries=3):
    """Path A: Gemini Q&A. Returns 1 (SynthID present) / 0 / -1 (undecided/error)."""
    for t in range(retries):
        try:
            r = client.models.generate_content(model=DET_MODEL, contents=[pil, DET_PROMPT])
            s = (r.text or "").strip().upper()
            if s.startswith("Y"): return 1
            if s.startswith("N"): return 0
            return -1
        except Exception as e:
            time.sleep(4 * (t + 1))
    return -1

# --- Path B (authoritative, if you have Vertex AI access): stub ---
# from google.cloud import aiplatform ; use the Imagen model's `.verify()` / media watermark
# verification endpoint. Left as a stub because access is gated as of 2025.

def rate(client, files):
    d = [detect(client, Image.open(f).convert("RGB")) for f in files]
    valid = [x for x in d if x >= 0]
    pos = sum(1 for x in valid if x == 1)
    return (pos / len(valid) if valid else float("nan")), len(valid), len(d) - len(valid)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--pos", help="dir of clean SynthID images (expect detection ~1.0)")
    ap.add_argument("--neg", help="dir of non-SynthID COCO images (expect ~0.0)")
    ap.add_argument("--glob", default="*")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    if not a.calibrate: raise SystemExit("use --calibrate (this module is imported by synthid_eval.py)")
    client = make_client()
    posf = sorted(glob.glob(os.path.join(a.pos, "*.png")))[:a.n]
    negf = sorted(glob.glob(os.path.join(a.neg, a.glob)))[:a.n]
    tpr, nv1, err1 = rate(client, posf)
    fpr, nv2, err2 = rate(client, negf)
    print(f"=== DETECTOR SELF-CALIBRATION (Gemini-QA) ===")
    print(f"  TPR on clean SynthID: {tpr:.3f}  (n={nv1}, undecided={err1})  -> want ~1.0")
    print(f"  FPR on non-SynthID  : {fpr:.3f}  (n={nv2}, undecided={err2})  -> want ~0.0")
    usable = (tpr >= 0.9 and fpr <= 0.1)
    print(f"  VERDICT: {'USABLE for the sweep' if usable else 'NOT usable -> use SynthID Detector portal or Vertex AI verify'}")
    print("CALIBRATE_DONE")

if __name__ == "__main__":
    main()
