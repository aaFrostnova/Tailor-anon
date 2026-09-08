set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
CRPY=${WM_PY_CTRLREGEN:-python}
UMPY=${WM_PY_UNMARKER:-python}
FPPY=${WM_PY:-python}
cd $CF          # src.dft_kred_modules imports scripts.prototype_dft_carrier, a namespace package that resolves from the repo root only
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CF:$CF/scripts:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC"
export TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
QUIET='Warning|warn|weights_only|Loading|Migrating|exists, skipping|Model loaded|^\s*$'
