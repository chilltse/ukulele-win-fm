#!/bin/bash
#SBATCH -J pykt_split_json_akt_qid_fm4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# --- 邮件配置 ---
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u8016457@anu.edu.au

set -euo pipefail

# ---------- 日志格式（主输出 .out / .err 里更易扫读）----------
log_ts() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]" "$@"
}

log_rule() {
    # 单行分隔
    echo "--------------------------------------------------------------------------------"
}

log_banner() {
    # 大段开始
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
    # 子步骤
    echo ""
    log_rule
    log_ts "$*"
    log_rule
}

# 1) 环境配置
source ~/.bashrc
source /home/remote/u8016457/miniconda3/etc/profile.d/conda.sh
conda activate pykt

# 2) 路径设置
BASE_DIR=/home/remote/u8016457/my_code/ukulele-win-fm
cd "${BASE_DIR}/examples"

DATA_DIR="${BASE_DIR}/data/yousician"
SPLIT_JSON_DIR="${DATA_DIR}/split_json"
TARGET_JSON="${DATA_DIR}/yousician_ukulele.json"
FM4_DATA_DIR="${DATA_DIR}/yousician_fm4"

mkdir -p logs
mkdir -p "${FM4_DATA_DIR}"

# 3) 从训练日志中提取 save_dir
extract_save_dir() {
    local log_file="$1"

    grep 'save_dir:' "$log_file" \
        | tail -n 1 \
        | sed -E 's/.*save_dir:[[:space:]]*//; s/[[:space:]]*,[[:space:]]*dataset_name:.*//; s/[[:space:]]+$//' \
        || true
}

# 4) 单次：训练 AKT + 预测（dataset / emb_type 由参数决定）
run_akt_train_predict() {
    local dataset_name="$1"
    local emb_type="$2"
    local tag="$3"
    local variant_label="$4"

    local train_log="logs/train_${tag}_${variant_label}_${SLURM_JOB_ID}.log"
    local pred_log="logs/predict_${tag}_${variant_label}_${SLURM_JOB_ID}.log"
    local _t_all=$SECONDS

    log_sub "TRAIN  | tag=${tag} | ${variant_label} | dataset=${dataset_name} | emb_type=${emb_type}"
    echo "  -> 完整训练日志（tee）: ${train_log}"
    echo "  -> 命令: python -u wandb_akt_train.py --dataset_name=${dataset_name} --model_name=akt --emb_type=${emb_type}"

    local _t_train=$SECONDS
    python -u wandb_akt_train.py \
        --dataset_name="${dataset_name}" \
        --model_name=akt \
        --emb_type="${emb_type}" \
        2>&1 | tee "${train_log}"
    echo "  -> 训练阶段耗时: $((SECONDS - _t_train)) 秒"

    echo "  -> 从训练日志解析 checkpoint 路径 (save_dir) ..."

    local ckpt
    ckpt="$(extract_save_dir "${train_log}")"

    if [ -z "${ckpt}" ]; then
        log_ts "ERROR: 无法从 ${train_log} 解析 save_dir（请确认训练脚本会打印含 save_dir: 的行）"
        exit 1
    fi

    echo "  -> save_dir = ${ckpt}"

    log_sub "PREDICT | tag=${tag} | ${variant_label} | 使用上述 save_dir"
    echo "  -> 完整预测日志（tee）: ${pred_log}"
    echo "  -> 命令: python -u wandb_predict.py --save_dir=${ckpt}"

    local _t_pred=$SECONDS
    python -u wandb_predict.py \
        --save_dir="${ckpt}" \
        2>&1 | tee "${pred_log}"
    echo "  -> 预测阶段耗时: $((SECONDS - _t_pred)) 秒"

    echo "  -> 本段 TRAIN+PREDICT 合计: $((SECONDS - _t_all)) 秒"
    log_ts "完成: ${variant_label} | tag=${tag}"
}

# ---------- 作业头信息 ----------
log_banner "JOB HEADER"
echo "  SLURM_JOB_ID     : ${SLURM_JOB_ID:-N/A}"
echo "  SLURM_JOB_NAME   : ${SLURM_JOB_NAME:-N/A}"
echo "  SLURM_SUBMIT_DIR : ${SLURM_SUBMIT_DIR:-N/A}"
echo "  HOSTNAME         : $(hostname)"
echo "  PWD              : $(pwd)"
echo "  BASE_DIR         : ${BASE_DIR}"
echo "  SPLIT_JSON_DIR   : ${SPLIT_JSON_DIR}"
echo "  TARGET_JSON      : ${TARGET_JSON}"
echo "  FM4_DATA_DIR     : ${FM4_DATA_DIR}"
log_banner_end "JOB HEADER"

log_banner "PIPELINE: split_json 中每个 JSON -> yousician (qid) + yousician_fm4 (qid_fm4)"
log_ts "流程说明:"
echo "  对每个 split_json/*.json : 复制为 yousician_ukulele.json"
echo "    A) preprocess yousician  -> AKT emb_type=qid    -> predict"
echo "    B) preprocess yousician_fm4 -> AKT emb_type=qid_fm4 -> predict"
log_banner_end "PIPELINE: split_json 中每个 JSON -> yousician (qid) + yousician_fm4 (qid_fm4)"

JOB_T0=$SECONDS

shopt -s nullglob
json_files=("${SPLIT_JSON_DIR}"/*.json)
if [ "${#json_files[@]}" -eq 0 ]; then
    log_ts "ERROR: 目录下没有 *.json : ${SPLIT_JSON_DIR}"
    exit 1
fi

total_json=${#json_files[@]}
log_ts "共发现 ${total_json} 个 split JSON，将依次处理。"

json_idx=0
for SRC_JSON in "${json_files[@]}"; do
    json_idx=$((json_idx + 1))

    stem="$(basename "${SRC_JSON}" .json)"
    tag="${stem}"
    SPLIT_T0=$SECONDS

    log_banner "SPLIT JSON  [${json_idx}/${total_json}]  tag=${tag}"
    echo "  SRC_JSON : ${SRC_JSON}"
    echo "  tag      : ${tag}"

    if [ ! -f "${SRC_JSON}" ]; then
        log_ts "ERROR: 文件不存在，跳过: ${SRC_JSON}"
        continue
    fi

    log_sub "[${json_idx}/${total_json}] COPY 源 JSON -> TARGET_JSON"
    echo "  cp -f \"${SRC_JSON}\" \"${TARGET_JSON}\""
    cp -f "${SRC_JSON}" "${TARGET_JSON}"
    log_ts "COPY 完成。"

    # ---------- 普通 Yousician + AKT (qid) ----------
    log_sub "[${json_idx}/${total_json}] STEP A | preprocess | dataset=yousician | standard KC | tag=${tag}"
    echo "  -> 清理 pkl: ${DATA_DIR}/*.pkl"
    rm -f "${DATA_DIR}"/*.pkl
    echo "  -> 运行: python -u data_preprocess.py --dataset_name=yousician"
    python -u data_preprocess.py --dataset_name=yousician
    log_ts "STEP A preprocess 完成。"

    run_akt_train_predict "yousician" "qid" "${tag}" "akt_qid"

    # ---------- FM4 KC + AKT (qid_fm4) ----------
    log_sub "[${json_idx}/${total_json}] STEP B | preprocess | dataset=yousician_fm4 | FM4 KC | tag=${tag}"
    mkdir -p "${FM4_DATA_DIR}"
    echo "  -> 清理 pkl: ${FM4_DATA_DIR}/*.pkl"
    rm -f "${FM4_DATA_DIR}"/*.pkl
    echo "  -> 运行: python -u data_preprocess.py --dataset_name=yousician_fm4"
    python -u data_preprocess.py --dataset_name=yousician_fm4
    log_ts "STEP B preprocess 完成。"

    run_akt_train_predict "yousician_fm4" "qid_fm4" "${tag}" "akt_qid_fm4"

    echo ""
    log_ts "本 split JSON 全部完成 [${json_idx}/${total_json}] tag=${tag} | 本文件累计耗时: $((SECONDS - SPLIT_T0)) 秒"
    log_banner_end "SPLIT JSON  [${json_idx}/${total_json}]  tag=${tag}"
    log_rule
done

log_banner "ALL DONE"
log_ts "所有 split_json 已跑完。作业总耗时: $((SECONDS - JOB_T0)) 秒 (${total_json} 个 JSON)。"
log_banner_end "ALL DONE"
