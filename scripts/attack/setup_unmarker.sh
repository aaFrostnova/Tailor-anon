#!/bin/bash
# Set up UnMarker (Kassis & Hengartner, IEEE S&P'25, arXiv:2405.08363) in an ISOLATED env.
# Repo: github.com/andrekassis/ai-watermark (branch master). Its install.sh conda-installs
# CUDA 11.8 -> MUST be a separate env. ~30GB weights staged on /work (not /home).
set -e
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO

echo "=== [1/4] clone ai-watermark ==="
[ -d external/ai-watermark ] || git clone https://github.com/andrekassis/ai-watermark.git external/ai-watermark
cd external/ai-watermark
echo "HEAD: $(git rev-parse --short HEAD)"

echo "=== [2/4] inspect install.sh + requirements (record exact pins before running) ==="
echo "----- install.sh -----"; sed -n '1,80p' install.sh
echo "----- requirements.txt -----"; cat requirements.txt 2>/dev/null | head -60
echo "----- attack.py usage / entrypoint -----"; sed -n '1,60p' attack.py
echo "----- Vine.yaml (the config we care about) -----"; cat attack_configs/Vine.yaml 2>/dev/null

echo "=== [3/4] modules/attack/unmark core (the standalone optimizer signature) ==="
sed -n '1,120p' modules/attack/unmark/cw.py 2>/dev/null | head -120

echo "=== [4/4] NOTE: not auto-running install.sh yet (it conda-installs CUDA 11.8 + downloads ~30GB)."
echo "Review the above, then run install.sh + download_data_and_models.sh in a fresh env."
echo "UNMARKER_RECON_DONE"
