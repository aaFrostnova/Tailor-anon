#!/bin/bash
# Set up CtrlRegen+ (RAVEN "CtrlGen+", arXiv:2410.05470, ICLR'25) in an ISOLATED conda env.
# Its pins (diffusers==0.27.2, transformers==4.37.2, numpy<2) conflict with the fingerprint
# env, so it MUST be separate. Invoked as a subprocess from our attack harness.
# Stage all HF weights on /work to avoid any /home quota.
set -e
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
mkdir -p $HF_HOME
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO

echo "=== [1/5] conda create ctrlregen ==="
source ~/.conda/etc/profile.d/conda.sh 2>/dev/null || source $(conda info --base)/etc/profile.d/conda.sh
conda create -n ctrlregen python=3.10 -y
conda activate ctrlregen

echo "=== [2/5] torch ==="
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo "=== [3/5] clone CtrlRegen ==="
[ -d external/CtrlRegen ] || git clone https://github.com/yepengliu/CtrlRegen.git external/CtrlRegen
cd external/CtrlRegen
pip install -q -r requirements.txt
pip install -q accelerate safetensors "numpy<2" huggingface_hub

echo "=== [4/5] download ctrlregen checkpoints (HF yepengliu/ctrlregen) ==="
huggingface-cli download yepengliu/ctrlregen --local-dir . 2>&1 | tail -5

echo "=== [5/5] pre-pull base weights ==="
python - <<'PY'
from huggingface_hub import snapshot_download
for r in ["SG161222/Realistic_Vision_V4.0_noVAE", "stabilityai/sd-vae-ft-mse", "facebook/dinov2-giant"]:
    try:
        snapshot_download(r); print("pulled", r, flush=True)
    except Exception as e:
        print("WARN pull", r, type(e).__name__, str(e)[:80], flush=True)
PY
echo "CTRLREGEN_SETUP_DONE"
