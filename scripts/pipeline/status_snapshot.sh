#!/bin/bash
# One-screen status of the whole pipeline, written to logs/STATUS.txt (read it from any session).
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
{
echo "STATUS $(date -u)"
echo "class shards written: $(ls $SC/class_eval_C*_shard*.json 2>/dev/null | wc -l)/100"
for n in classes orch cert_full cf_xenv; do echo -n "$n: "; sacct -u mingzhel_umass_edu -S 2026-09-08T03:30 --name=$n -X -n -o State 2>/dev/null | sort | uniq -c | tr '\n' ' '; echo; done
echo "certify groups measured: indomain $(ls $SC/certify_full/C*/g*.json 2>/dev/null | wc -l)  ood_content $(ls $SC/certify_full_ood_content/C*/g*.json 2>/dev/null | wc -l)  ood_generator $(ls $SC/certify_full_ood_generator/C*/g*.json 2>/dev/null | wc -l)"
echo "chain milestones:"; grep -E "^=== |CHAIN_CLASSES_DONE|NOT ALL" $SC/logs/chain_after_classes.out 2>/dev/null | tail -6
} > $SC/logs/STATUS.txt 2>&1
