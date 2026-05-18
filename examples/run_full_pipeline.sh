#!/bin/bash
set -euo pipefail

# Robust end-to-end automation:
# 1) generate W&B train sweep yaml files
# 2) register train sweeps and save unique log
# 3) generate serial single-GPU train sbatch by exact filename matching
# 4) submit train agents and wait
# 5) merge train results and generate predict sweep script
# 6) register predict sweep, submit predict agent sbatch, wait
# 7) summarize testauc/testacc
#
# Usage:
#   bash run_full_pipeline.sh <dataset> <model> [emb_type] [folds] [project]
#
# Example:
#   bash run_full_pipeline.sh nips_task34 dkt qid 0,1,2,3,4 ukulele-win-fm-examples
#   bash run_full_pipeline.sh nips_task34 dkt_plus qid_tree 0,1,2,3,4 ukulele-win-fm-examples

DATASET_NAME="${1:-nips_task34}"
MODEL_NAME="${2:-dkt}"
EMB_TYPE="${3:-qid}"
FOLDS="${4:-0,1,2,3,4}"
PROJECT_NAME="${5:-ukulele-win-fm-examples}"

BASE_DIR="/home/remote/u8016457/my_code/ukulele-win-fm"
EXAMPLES_DIR="${BASE_DIR}/examples"

cd "${EXAMPLES_DIR}"

mkdir -p logs all_wandbs results pipeline_runs

sanitize() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-nojob}_$$"
RUN_TAG="$(sanitize "${DATASET_NAME}__${MODEL_NAME}__${EMB_TYPE}__${FOLDS}__${RUN_STAMP}")"
RUN_DIR="${EXAMPLES_DIR}/pipeline_runs/${RUN_TAG}"

RUN_LOG_DIR="${RUN_DIR}/logs"
TRAIN_ALL_DIR="${RUN_DIR}/all_wandbs"
TRAIN_LAUNCH_FILE="${RUN_DIR}/all_start.sh"
TRAIN_LOG="${RUN_DIR}/log.train.all"
TRAIN_SBATCH="${RUN_DIR}/run_agents.sbatch"

PRED_START_FILE="${RUN_DIR}/start_predict.sh"
PRED_LOG="${RUN_DIR}/log.predict.all"
PRED_SBATCH="${RUN_DIR}/run_wandb_predict_auto.sbatch"

LOCK_FILE="${EXAMPLES_DIR}/.merge_predict.lock"
CACHE_DIR="${RUN_DIR}/wandb_result_cache"

mkdir -p "${RUN_DIR}" "${RUN_LOG_DIR}" "${TRAIN_ALL_DIR}" "${CACHE_DIR}"

echo "=== Pipeline start ==="
echo "dataset=${DATASET_NAME}"
echo "model=${MODEL_NAME}"
echo "emb=${EMB_TYPE}"
echo "folds=${FOLDS}"
echo "project=${PROJECT_NAME}"
echo "run_dir=${RUN_DIR}"

FOLD_COUNT="$(echo "${FOLDS}" | awk -F',' '{print NF}')"
echo "fold_count=${FOLD_COUNT}"

wait_for_job() {
  local job_id="$1"
  echo "[wait] job ${job_id}"

  while true; do
    local queue_state
    queue_state="$(squeue -j "${job_id}" -h -o "%T" 2>/dev/null | head -n 1 || true)"

    if [[ -n "${queue_state}" ]]; then
      echo "[wait] ${job_id} queue_state=${queue_state}"
      sleep 30
      continue
    fi

    local state
    state="$(sacct -j "${job_id}" -X --format=State --noheader 2>/dev/null | awk 'NF{print $1; exit}' || true)"

    if [[ -z "${state}" ]]; then
      echo "[wait] ${job_id} not in queue, waiting for sacct..."
      sleep 15
      continue
    fi

    echo "[wait] ${job_id} final_state=${state}"

    case "${state}" in
      COMPLETED|COMPLETED+)
        return 0
        ;;
      FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*)
        echo "[error] job ${job_id} finished with state ${state}"
        return 1
        ;;
      *)
        sleep 30
        ;;
    esac
  done
}

extract_sweep_target() {
  local log_file="$1"

  sed -n 's/^wandb: Run sweep agent with: wandb agent //p' "${log_file}" | tail -n 1
}

echo "=== Step 1: generate train sweep yaml files ==="

python generate_wandb.py \
  --dataset_names "${DATASET_NAME}" \
  --project_name "${PROJECT_NAME}" \
  --model_names "${MODEL_NAME}" \
  --emb_types "${EMB_TYPE}" \
  --folds "${FOLDS}" \
  --all_dir "${TRAIN_ALL_DIR}" \
  --launch_file "${TRAIN_LAUNCH_FILE}"

if [[ ! -s "${TRAIN_LAUNCH_FILE}" ]]; then
  echo "[error] train launch file was not generated or is empty: ${TRAIN_LAUNCH_FILE}"
  exit 1
fi

echo "=== Step 2: register train sweeps and save train log ==="

sh "${TRAIN_LAUNCH_FILE}" > "${TRAIN_LOG}" 2>&1

if ! grep -q "^wandb: Run sweep agent with:" "${TRAIN_LOG}"; then
  echo "[error] no wandb agent command found in train log"
  echo "Please check:"
  echo "${TRAIN_LOG}"
  tail -n 50 "${TRAIN_LOG}"
  exit 1
fi

echo "=== Step 3: generate train sbatch by exact filename matching ==="

PIPE_LOG="${TRAIN_LOG}" \
PIPE_SBATCH="${TRAIN_SBATCH}" \
PIPE_DATASET="${DATASET_NAME}" \
PIPE_MODEL="${MODEL_NAME}" \
PIPE_EMB="${EMB_TYPE}" \
PIPE_FOLDS="${FOLDS}" \
PIPE_BASE_DIR="${BASE_DIR}" \
PIPE_RUN_DIR="${RUN_DIR}" \
python - <<'PY'
import os
import sys

log_path = os.environ["PIPE_LOG"]
sbatch_path = os.environ["PIPE_SBATCH"]

dataset = os.environ["PIPE_DATASET"]
model = os.environ["PIPE_MODEL"]
emb = os.environ["PIPE_EMB"]
folds = [x.strip() for x in os.environ["PIPE_FOLDS"].split(",") if x.strip()]

base_dir = os.environ["PIPE_BASE_DIR"]
run_dir = os.environ["PIPE_RUN_DIR"]
log_dir = os.path.join(run_dir, "logs")

with open(log_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f]

# Map:
#   yaml basename -> sweep target
#
# Example:
#   nips_task34_dkt_plus_qid_tree_0.yaml
#       -> chilltse808-anu/ukulele-win-fm-examples/xxxx
#
# Important:
#   We do NOT split dataset/model/emb by underscore.
#   We match the full expected basename exactly.
sweep_map = {}

i = 0
while i < len(lines):
    line = lines[i]

    if not line.startswith("wandb: Creating sweep from: "):
        i += 1
        continue

    yaml_path = line.split("wandb: Creating sweep from: ", 1)[1]
    yaml_basename = os.path.basename(yaml_path)

    sweep_target = None
    for j in range(i + 1, min(i + 8, len(lines))):
        if lines[j].startswith("wandb: Run sweep agent with: "):
            raw = lines[j].split("wandb: Run sweep agent with: ", 1)[1]
            if raw.startswith("wandb agent "):
                sweep_target = raw.replace("wandb agent ", "", 1)
            else:
                sweep_target = raw
            break

    if sweep_target is not None:
        sweep_map[yaml_basename] = sweep_target

    i += 1

commands = []
missing = []

for fold in folds:
    expected = f"{dataset}_{model}_{emb}_{fold}.yaml"

    if expected not in sweep_map:
        missing.append(expected)
        continue

    target = sweep_map[expected]
    commands.append(f"CUDA_VISIBLE_DEVICES=0 wandb agent --count 1 {target}")

if missing:
    print("[error] Missing expected sweep yaml entries in train log:")
    for x in missing:
        print("  ", x)

    print("\nAvailable yaml basenames in log:")
    for x in sorted(sweep_map.keys()):
        print("  ", x)

    sys.exit(1)

if not commands:
    print("[error] No train commands generated.")
    sys.exit(1)

content = f"""#!/bin/bash
#SBATCH --job-name=train_{dataset}_{model}_{emb}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output={log_dir}/train-%j.out
#SBATCH --error={log_dir}/train-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u8016457@anu.edu.au

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/.bashrc
source /home/remote/u8016457/miniconda3/etc/profile.d/conda.sh
conda activate pykt

BASE_DIR="{base_dir}"
cd "${{BASE_DIR}}/examples"

DATA_DIR="{dataset}"
echo "Deleting pkl for ${{DATA_DIR}}"
rm -f "${{DATA_DIR}}"/*.pkl

"""

for cmd in commands:
    content += cmd + "\n"

content += "\nwait\n"

with open(sbatch_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"[info] generated train sbatch: {sbatch_path}")
print(f"[info] commands: {len(commands)}")
for cmd in commands:
    print("  " + cmd)
PY

if [[ ! -s "${TRAIN_SBATCH}" ]]; then
  echo "[error] train sbatch was not generated or is empty: ${TRAIN_SBATCH}"
  exit 1
fi

if ! grep -q "wandb agent --count 1" "${TRAIN_SBATCH}"; then
  echo "[error] train sbatch does not contain expected wandb agent commands"
  cat "${TRAIN_SBATCH}"
  exit 1
fi

echo "=== Step 4: submit train agents ==="

TRAIN_SUBMIT_MSG="$(sbatch "${TRAIN_SBATCH}")"
echo "${TRAIN_SUBMIT_MSG}"
TRAIN_JOB_ID="$(echo "${TRAIN_SUBMIT_MSG}" | awk '{print $4}')"

if [[ -z "${TRAIN_JOB_ID}" ]]; then
  echo "[error] failed to parse train job id"
  exit 1
fi

wait_for_job "${TRAIN_JOB_ID}"

echo "=== Step 5: merge results and prepare predict sweep ==="
echo "[info] this step uses a lock because merge_wandb_results.py writes global start_predict.sh"

(
  flock -x 200

  python merge_wandb_results.py \
    --project_name "${PROJECT_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --model_names "${MODEL_NAME}" \
    --emb_types "${EMB_TYPE}" \
    --update True \
    --save_dir "results" \
    --extract_best_model True

  if [[ ! -s start_predict.sh ]]; then
    echo "[error] global start_predict.sh was not generated or is empty"
    exit 1
  fi

  cp start_predict.sh "${PRED_START_FILE}"
) 200>"${LOCK_FILE}"

if [[ ! -s "${PRED_START_FILE}" ]]; then
  echo "[error] unique predict start file was not created: ${PRED_START_FILE}"
  exit 1
fi

echo "=== Step 6: register predict sweep ==="

sh "${PRED_START_FILE}" > "${PRED_LOG}" 2>&1

PRED_SWEEP_TARGET="$(extract_sweep_target "${PRED_LOG}")"

if [[ -z "${PRED_SWEEP_TARGET}" ]]; then
  echo "[error] failed to extract predict sweep target from:"
  echo "${PRED_LOG}"
  tail -n 50 "${PRED_LOG}"
  exit 1
fi

echo "[info] predict sweep target: ${PRED_SWEEP_TARGET}"

echo "=== Step 7: generate predict sbatch ==="

cat > "${PRED_SBATCH}" <<EOF
#!/bin/bash
#SBATCH --job-name=pred_${DATASET_NAME}_${MODEL_NAME}_${EMB_TYPE}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=${RUN_LOG_DIR}/predict-%j.out
#SBATCH --error=${RUN_LOG_DIR}/predict-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u8016457@anu.edu.au

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/.bashrc
source /home/remote/u8016457/miniconda3/etc/profile.d/conda.sh
conda activate pykt

BASE_DIR=/home/remote/u8016457/my_code/ukulele-win-fm
cd "\${BASE_DIR}/examples"

CUDA_VISIBLE_DEVICES=0 wandb agent --count ${FOLD_COUNT} ${PRED_SWEEP_TARGET}
EOF

echo "=== Step 8: submit predict agent job ==="

PRED_SUBMIT_MSG="$(sbatch "${PRED_SBATCH}")"
echo "${PRED_SUBMIT_MSG}"
PRED_JOB_ID="$(echo "${PRED_SUBMIT_MSG}" | awk '{print $4}')"

if [[ -z "${PRED_JOB_ID}" ]]; then
  echo "[error] failed to parse predict job id"
  exit 1
fi

wait_for_job "${PRED_JOB_ID}"

echo "=== Step 9: summarize prediction metrics ==="

python run_sum_predict.py \
  --user "chilltse808-anu" \
  --project_name "${PROJECT_NAME}" \
  --dataset_name "${DATASET_NAME}" \
  --model_name "${MODEL_NAME}" \
  --emb_type "${EMB_TYPE}" \
  --print_std True \
  --use_cache False \
  --cache_dir "${CACHE_DIR}" \
  --only_finished True \
  --min_finished_runs "${FOLD_COUNT}"

echo "=== Pipeline done ==="
echo "run_dir=${RUN_DIR}"