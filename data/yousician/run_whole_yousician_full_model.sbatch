#!/bin/bash
#SBATCH -J pykt_all_models_qid_fmkc_a100
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# --- 邮件配置 ---
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u8016457@anu.edu.au

set -euo pipefail

# ============================================================
# 0) 只选择一个 dataset 跑
# 可选：
#   RUN_DATASET="yousician"       -> emb_type=qid
#   RUN_DATASET="yousician_fmkc"  -> emb_type=qid_fmkc
# ============================================================
RUN_DATASET="yousician_fmkc"

case "${RUN_DATASET}" in
    "yousician")
        RUN_EMB_TYPE="qid"
        RUN_CONFIG_LABEL="qid"
        ;;
    "yousician_fmkc")
        RUN_EMB_TYPE="qid_fmkc"
        RUN_CONFIG_LABEL="qid_fmkc"
        ;;
    *)
        echo "ERROR: Unknown RUN_DATASET=${RUN_DATASET}"
        echo "Allowed values: yousician, yousician_fmkc"
        exit 1
        ;;
esac

log_ts() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]" "$@"
}

log_rule() {
    echo "--------------------------------------------------------------------------------"
}

log_banner() {
    echo ""
    echo "================================================================================"
    echo "  $*"
    echo "================================================================================"
    log_ts "BEGIN: $*"
}

log_banner_end() {
    log_ts "END:   $*"
    echo "================================================================================"
    echo ""
}

log_sub() {
    echo ""
    log_rule
    log_ts "$*"
    log_rule
}

# ---------- 1) 环境配置 ----------
source ~/.bashrc
source /home/remote/u8016457/miniconda3/etc/profile.d/conda.sh
conda activate pykt

# ---------- 2) 路径设置 ----------
BASE_DIR=/home/remote/u8016457/my_code/ukulele-win-fm
cd "${BASE_DIR}/examples"

mkdir -p logs

echo "运行: python -u data_preprocess.py --dataset_name=${RUN_DATASET}"
python -u data_preprocess.py --dataset_name="${RUN_DATASET}" --maxlen=150
echo "STEP A preprocess 完成。"

# ---------- 3) predict 脚本 ----------
PREDICT_SCRIPT="wandb_predict.py"

# ---------- 4) 模型配置 ----------
# 格式：
# model_label:train_script
#
# 不再传 --model_name
MODELS=(
    "DKT:wandb_dkt_train.py"
    "DKT_plus:wandb_dkt_plus_train.py"
    "AKT:wandb_akt_train.py"
    "simpleKT:wandb_simplekt_train.py"
    "SAKT:wandb_sakt_train.py"
)

# ---------- 5) 数据配置 ----------
# 这里只放一个 config，由最开头的 RUN_DATASET 控制
# 格式：
# dataset_name:emb_type:config_label
CONFIGS=(
    "${RUN_DATASET}:${RUN_EMB_TYPE}:${RUN_CONFIG_LABEL}"
)

# ---------- 6) 从训练日志中提取 save_dir ----------
extract_save_dir() {
    local log_file="$1"

    grep 'save_dir:' "$log_file" \
        | tail -n 1 \
        | sed -E 's/.*save_dir:[[:space:]]*//; s/[[:space:]]*,[[:space:]]*dataset_name:.*//; s/[[:space:]]+$//' \
        || true
}

# ---------- 7) 单组实验：train + predict ----------
run_train_predict() {
    local dataset_name="$1"
    local emb_type="$2"
    local config_label="$3"
    local model_label="$4"
    local train_script="$5"

    local tag="${model_label}_${config_label}"
    local train_log="logs/train_${tag}_${SLURM_JOB_ID}.log"
    local pred_log="logs/predict_${tag}_${SLURM_JOB_ID}.log"

    local t_all=$SECONDS

    log_sub "TRAIN | label=${model_label} | dataset=${dataset_name} | emb_type=${emb_type}"
    echo "  -> train script: ${train_script}"
    echo "  -> train log   : ${train_log}"
    echo "  -> command:"
    echo "     python -u ${train_script} --dataset_name=${dataset_name} --emb_type=${emb_type}"

    local t_train=$SECONDS

    python -u "${train_script}" \
        --dataset_name="${dataset_name}" \
        --emb_type="${emb_type}" \
        2>&1 | tee "${train_log}"

    echo "  -> train time: $((SECONDS - t_train)) seconds"

    log_ts "Parsing save_dir from training log..."

    local ckpt
    ckpt="$(extract_save_dir "${train_log}")"

    if [ -z "${ckpt}" ]; then
        log_ts "ERROR: Cannot parse save_dir from ${train_log}"
        log_ts "Please check whether the training script prints a line containing 'save_dir:'"
        exit 1
    fi

    echo "  -> save_dir = ${ckpt}"

    log_sub "PREDICT | label=${model_label} | dataset=${dataset_name} | emb_type=${emb_type}"
    echo "  -> predict log: ${pred_log}"
    echo "  -> command:"
    echo "     python -u ${PREDICT_SCRIPT} --save_dir=${ckpt}"

    local t_pred=$SECONDS

    python -u "${PREDICT_SCRIPT}" \
        --save_dir="${ckpt}" \
        2>&1 | tee "${pred_log}"

    echo "  -> predict time: $((SECONDS - t_pred)) seconds"
    echo "  -> total time for ${tag}: $((SECONDS - t_all)) seconds"

    log_ts "DONE: label=${model_label} | dataset=${dataset_name} | emb_type=${emb_type}"
}

# ---------- 8) Job header ----------
JOB_T0=$SECONDS

log_banner "JOB HEADER"
echo "  SLURM_JOB_ID     : ${SLURM_JOB_ID:-N/A}"
echo "  SLURM_JOB_NAME   : ${SLURM_JOB_NAME:-N/A}"
echo "  SLURM_SUBMIT_DIR : ${SLURM_SUBMIT_DIR:-N/A}"
echo "  HOSTNAME         : $(hostname)"
echo "  PWD              : $(pwd)"
echo "  BASE_DIR         : ${BASE_DIR}"
echo "  PREDICT_SCRIPT   : ${PREDICT_SCRIPT}"
echo "  GPU REQUEST      : --gres=gpu:a100:1"
echo "  RUN_DATASET      : ${RUN_DATASET}"
echo "  RUN_EMB_TYPE     : ${RUN_EMB_TYPE}"
echo "  RUN_CONFIG_LABEL : ${RUN_CONFIG_LABEL}"
echo "  MODELS           : ${MODELS[*]}"
echo "  CONFIGS          : ${CONFIGS[*]}"
log_banner_end "JOB HEADER"

log_banner "PIPELINE: Train + Predict selected dataset only"
echo "  This job will sequentially run:"
echo "    1) DKT        -> wandb_dkt_train.py"
echo "    2) DKT_plus   -> wandb_dkt_plus_train.py"
echo "    3) AKT        -> wandb_akt_train.py"
echo "    4) simpleKT   -> wandb_simplekt_train.py"
echo "    5) SAKT       -> wandb_sakt_train.py"
echo ""
echo "  Selected config:"
echo "    --dataset_name=${RUN_DATASET} --emb_type=${RUN_EMB_TYPE}"
echo ""
echo "  No --model_name argument."
echo "  No JSON copy."
echo "  Only selected data_preprocess, train, and predict."
log_banner_end "PIPELINE: Train + Predict selected dataset only"

# ---------- 9) 顺次执行 ----------
model_idx=0
total_models=${#MODELS[@]}
total_configs=${#CONFIGS[@]}

for model_item in "${MODELS[@]}"; do
    model_idx=$((model_idx + 1))

    model_label="$(echo "${model_item}" | cut -d':' -f1)"
    train_script="$(echo "${model_item}" | cut -d':' -f2)"

    log_banner "MODEL [${model_idx}/${total_models}] ${model_label}"
    echo "  model_label  : ${model_label}"
    echo "  train_script : ${train_script}"

    if [ ! -f "${train_script}" ]; then
        log_ts "ERROR: train script not found: ${train_script}"
        exit 1
    fi

    config_idx=0

    for config_item in "${CONFIGS[@]}"; do
        config_idx=$((config_idx + 1))

        dataset_name="$(echo "${config_item}" | cut -d':' -f1)"
        emb_type="$(echo "${config_item}" | cut -d':' -f2)"
        config_label="$(echo "${config_item}" | cut -d':' -f3)"

        log_banner "CONFIG [${config_idx}/${total_configs}] model=${model_label} | config=${config_label}"
        echo "  dataset_name : ${dataset_name}"
        echo "  emb_type     : ${emb_type}"
        echo "  config_label : ${config_label}"

        run_train_predict \
            "${dataset_name}" \
            "${emb_type}" \
            "${config_label}" \
            "${model_label}" \
            "${train_script}"

        log_banner_end "CONFIG [${config_idx}/${total_configs}] model=${model_label} | config=${config_label}"
    done

    log_ts "Completed MODEL [${model_idx}/${total_models}] ${model_label}"
    log_banner_end "MODEL [${model_idx}/${total_models}] ${model_label}"
done

# ---------- 10) Done ----------
log_banner "ALL DONE"
log_ts "All models and configs finished."
log_ts "Total job time: $((SECONDS - JOB_T0)) seconds"
log_banner_end "ALL DONE"