#!/bin/bash
# Build the unmarker conda env for a STANDALONE UnMarker attack (no 30GB watermarker weights,
# no CUDA-11.8 toolkit -- we use torch cu121's bundled CUDA). Editable-install the local pkgs.
set -e
source $(conda info --base)/etc/profile.d/conda.sh
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO/external/ai-watermark

echo "=== [1/4] conda create unmarker (py3.10) ==="
conda create -n unmarker python=3.10 -y
conda activate unmarker

echo "=== [2/4] pip install -r requirements.txt (torch2.2.2 cu121 + tf2.9 + kornia/piqa/torch_dct...) ==="
pip install -r requirements.txt 2>&1 | tail -8

echo "=== [3/4] editable installs (systems, ptw, stable_sig, unmark) ==="
cd systems && pip install -e . 2>&1 | tail -2 && cd ..
cd systems/watermarkers/networks/ptw && pip install -e . 2>&1 | tail -2 && cd $REPO/external/ai-watermark
cd systems/watermarkers/networks/stable_sig && pip install -e . 2>&1 | tail -2 && cd $REPO/external/ai-watermark
cd modules/attack/unmark && pip install -e . 2>&1 | tail -2 && cd $REPO/external/ai-watermark

echo "=== [4/4] import smoke test ==="
python - <<'PY'
import sys
try:
    from modules.attack import UnMark, BaseAttack
    from modules.attack.unmark.losses import get_loss
    print("IMPORT_OK UnMark + BaseAttack + get_loss")
except Exception as e:
    import traceback; traceback.print_exc(); print("IMPORT_FAIL", type(e).__name__, e)
PY
echo "UNMARKER_ENV_DONE"
