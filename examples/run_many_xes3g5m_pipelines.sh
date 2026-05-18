#!/bin/bash
set -euo pipefail

BASE_DIR="/home/remote/u8016457/my_code/ukulele-win-fm"
EXAMPLES_DIR="${BASE_DIR}/examples"

cd "${EXAMPLES_DIR}"

mkdir -p logs

echo "=== Many pipelines start at $(date) ==="

# 格式：
# dataset model emb_type folds project
EXPERIMENTS=(
  # "xes3g5m dkt_plus qid 0,1,2,3,4 ukulele-win-fm-examples"
  "xes3g5m sakt qid 0,1,2,3,4 ukulele-win-fm-examples"
  "xes3g5m simplekt qid 0,1,2,3,4 ukulele-win-fm-examples"
)

for EXP in "${EXPERIMENTS[@]}"; do
  echo ""
  echo "============================================================"
  echo "Running experiment: ${EXP}"
  echo "Start time: $(date)"
  echo "============================================================"

  bash run_full_pipeline.sh ${EXP}

  echo "============================================================"
  echo "Finished experiment: ${EXP}"
  echo "End time: $(date)"
  echo "============================================================"
done

echo "=== All pipelines done at $(date) ==="