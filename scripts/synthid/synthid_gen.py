"""Step 1 — generate SynthID-watermarked images by feeding COCO images through a
near-identity Gemini 2.5 Flash Image edit (the output carries an invisible SynthID
watermark per Google policy, while the pixels stay close to the input).

TEMPLATE: set GEMINI_API_KEY. Start with --n 8 to validate before scaling.
  pip install google-genai pillow
  export GEMINI_API_KEY=...          # from Google AI Studio
  python synthid_gen.py --src <coco_dir> --out results/synthid/wm --n 64
"""
import os, sys, io, glob, time, argparse
from PIL import Image

MODEL = "gemini-2.5-flash-image"     # aka "Nano Banana"; edit output carries SynthID
EDIT_PROMPT = ("Return this exact image unchanged. Do not add, remove, or alter any "
               "content, colors, or composition. Output the same picture.")

def make_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        # Vertex AI alternative (enterprise): genai.Client(vertexai=True, project=..., location="us-central1")
        raise SystemExit("set GEMINI_API_KEY (Google AI Studio) — see SYNTHID_TEST_DESIGN.md")
    return genai.Client(api_key=key)

def gemini_edit(client, pil, retries=4):
    """Near-identity edit -> SynthID-watermarked PNG (PIL). Returns None on failure."""
    for t in range(retries):
        try:
            resp = client.models.generate_content(model=MODEL, contents=[pil, EDIT_PROMPT])
            for part in resp.candidates[0].content.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
        except Exception as e:
            print(f"    retry {t+1}: {type(e).__name__} {str(e)[:100]}", flush=True)
            time.sleep(5 * (t + 1))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of COCO source images")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--out", default="results/synthid/wm")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--size", type=int, default=512, help="resize source to this square before edit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    client = make_client()
    imgs = sorted(glob.glob(os.path.join(a.src, a.glob)))[:a.n]
    print(f"generating {len(imgs)} SynthID images via {MODEL} -> {a.out}", flush=True)
    ok = 0
    for j, fp in enumerate(imgs):
        dst = os.path.join(a.out, f"sid_{j:05d}.png")
        if os.path.exists(dst):
            ok += 1; continue
        src = Image.open(fp).convert("RGB")
        if a.size: src = src.resize((a.size, a.size))
        wm = gemini_edit(client, src)
        if wm is None:
            print(f"  [{j}] FAILED (skipped)", flush=True); continue
        wm.resize((a.size, a.size)).save(dst); ok += 1
        if (j + 1) % 10 == 0: print(f"  [{j+1}/{len(imgs)}] ok={ok}", flush=True)
    print(f"GEN_DONE ok={ok}/{len(imgs)} -> {a.out}")
    print("NEXT: python synthid_detect.py --calibrate  (verify these are detectable BEFORE attacking)")

if __name__ == "__main__":
    main()
