#!/bin/bash
# Local worker for the margin run: solves the given class's shards in FORWARD order on this node while the
# SLURM array works in reverse order; class_scenarios.py skips a shard the other side already completed.
# Usage: bash local_worker_v3.sh <class index> <parallel workers>
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
REPO=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
export PYTHONPATH=$REPO:${PYTHONPATH:-}
CLS=$1; NW=${2:-8}
seq 0 39 | xargs -P $NW -I{} sh -c "${WM_PY:-python} -u $SC/class_scenarios.py $CLS {} 40 2000 > $SC/logs/classes_local_C$((CLS+1))_{}.log 2>&1"
echo "LOCAL_WORKER_DONE class $CLS $(date)"
