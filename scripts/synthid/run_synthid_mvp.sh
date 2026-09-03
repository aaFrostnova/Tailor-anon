#!/bin/bash
# SynthID robustness MVP: generate -> CALIBRATE detector (hard gate) -> attack+detect.
# Prereqs: pip install google-genai pillow ; export GEMINI_API_KEY=...
# Cost for N=64: ~$2 (or ~$0 within GCP free credits). Start small (N=8) to validate.
set -e
cd "$(dirname "$0")"
COCO=${1:-/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017}
N=${2:-64}
OUT=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/synthid
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python   # env must have google-genai

echo "== [1/3] generate $N SynthID images from COCO =="
$PY synthid_gen.py --src "$COCO" --glob '*.jpg' --out $OUT/wm --n $N

echo "== [2/3] CALIBRATE detector (TPR on SynthID vs FPR on raw COCO) — HARD GATE =="
$PY synthid_detect.py --calibrate --pos $OUT/wm --neg "$COCO" --n $((N<32?N:32))
echo "   -> if 'NOT usable', switch detection to the SynthID Detector portal / Vertex AI verify before continuing."

echo "== [3/3] attack + detect (add --regen for the regeneration axis) =="
$PY synthid_eval.py --wm $OUT/wm --n $N --out $OUT/synthid_robustness.json

echo "== DONE. Compare $OUT/synthid_robustness.json to our composite (results/defense/official_vine_tm_videoseal.json). =="
echo "Optional head-to-head: embed our composite on the SAME SynthID images:"
echo "  $PY ../defense/composite_external_eval.py --mode embed --fragments vine trustmark videoseal \\"
echo "     --image_dir $OUT/wm --image_glob '*.png' --n_images $N --embed_dir $OUT/wm_plus_ours"
