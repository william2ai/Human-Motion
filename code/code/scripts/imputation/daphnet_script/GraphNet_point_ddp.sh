#!/bin/bash
#SBATCH --job-name=freqtemp_daph_p01_ddp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yiting.lu@kaust.edu.sa

set -eo pipefail

source "${CONDA_PROFILE:-/ibex/user/luy0f/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-freqtemp}"

cd "${PROJECT_DIR:-/ibex/user/luy0f/work/FreqTempNet}"
mkdir -p checkpoints results test_results classification/Daphnet/0.1_point Records/Daphnet/mask_0.1

if [[ -n "${DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${DEVICES}"
fi

if [[ -z "${NUM_GPUS:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
        NUM_GPUS=$(python - <<'PY'
import os
devices = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
print(len(devices) if devices else 1)
PY
)
    else
        NUM_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)
    fi
fi

if [[ -z "${DEVICES:-}" ]]; then
    DEVICES=$(python - <<PY
n = int("${NUM_GPUS}")
print(",".join(str(i) for i in range(n)))
PY
)
fi

VISIBLE_GPU_COUNT=$(python - <<'PY'
import os
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if visible and visible != "NoDevFiles":
    print(len([d for d in visible.split(",") if d.strip()]))
else:
    print("")
PY
)
if [[ -n "${VISIBLE_GPU_COUNT}" && "${VISIBLE_GPU_COUNT}" -lt "${NUM_GPUS}" ]]; then
    echo "ERROR: NUM_GPUS=${NUM_GPUS}, but CUDA_VISIBLE_DEVICES exposes only ${VISIBLE_GPU_COUNT} device(s): ${CUDA_VISIBLE_DEVICES}"
    exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

RUN_TAG="${RUN_TAG:-${NUM_GPUS}gpu_ddp}"
MODEL_ID="${MODEL_ID:-GraphNet_Daphnet_point_mask_0.1_${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-Records/Daphnet/mask_0.1/${MODEL_ID}.txt}"
TIME_FILE="${TIME_FILE:-Records/Daphnet/mask_0.1/${MODEL_ID}_time.txt}"

echo "Job started on $(hostname) at $(date)"
echo "cwd=$(pwd)"
echo "NUM_GPUS=${NUM_GPUS}"
echo "DEVICES=${DEVICES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"

python - <<'PY'
import os, torch
from pathlib import Path
print("cwd:", os.getcwd())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i}:", torch.cuda.get_device_name(i))
print("tools.py line:", Path("utils/tools.py").read_text().splitlines()[38])
PY

if grep -R "np\.Inf" -n --include="*.py" . ; then
    echo "ERROR: np.Inf still exists in source code"
    exit 2
fi

START_TS=$(date +%s)
echo "Training command started at $(date)" | tee "$TIME_FILE"

CMD=(
  torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" run.py
  --mask "${MASK_TYPE:-0}"
  --patience "${PATIENCE:-3}"
  --num_edges "${NUM_EDGES:-5}"
  --num_sensors "${NUM_SENSORS:-9}"
  --train_epochs "${TRAIN_EPOCHS:-100}"
  --task_name imputation_graph
  --is_training 1
  --root_path "${ROOT_PATH:-./dataset/daphnet_data/}"
  --model_id "${MODEL_ID}"
  --mask_rate "${MASK_RATE:-0.1}"
  --model "${MODEL_NAME:-GraphNet}"
  --data "${DATA_NAME:-Daphnet}"
  --features M
  --seq_len "${SEQ_LEN:-96}"
  --label_len 0
  --pred_len 0
  --e_layers 2
  --d_layers 1
  --factor 3
  --enc_in "${ENC_IN:-9}"
  --dec_in "${DEC_IN:-9}"
  --c_out "${C_OUT:-9}"
  --batch_size "${BATCH_SIZE:-64}"
  --num_workers "${NUM_WORKERS:-2}"
  --d_model "${D_MODEL:-64}"
  --des "${DESCRIPTION:-Exp}"
  --itr 1
  --top_k "${TOP_K:-3}"
  --learning_rate "${LEARNING_RATE:-0.0001}"
  --use_multi_gpu
  --devices "${DEVICES}"
)

if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v "${CMD[@]}" > "$LOG_FILE" 2>&1
else
    "${CMD[@]}" > "$LOG_FILE" 2>&1
fi

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
{
    echo "Training command finished at $(date)"
    echo "Elapsed seconds: ${ELAPSED}"
    printf "Elapsed HH:MM:SS: %02d:%02d:%02d\n" "$((ELAPSED / 3600))" "$(((ELAPSED % 3600) / 60))" "$((ELAPSED % 60))"
} | tee -a "$TIME_FILE"

echo "Job completed at $(date)"
