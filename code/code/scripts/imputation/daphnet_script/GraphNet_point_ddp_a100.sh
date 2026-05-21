#!/bin/bash
#SBATCH --job-name=freqtemp_daph_p01_full_4a100
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100
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

source /ibex/user/luy0f/miniconda3/etc/profile.d/conda.sh
conda activate freqtemp

cd /ibex/user/luy0f/work/FreqTempNet
mkdir -p checkpoints results test_results classification/Daphnet/0.1_point Records/Daphnet/mask_0.1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=2

LOG_FILE="Records/Daphnet/mask_0.1/GraphNet_Daphnet_point_mask_0.1_4a100_ddp.txt"
TIME_FILE="Records/Daphnet/mask_0.1/GraphNet_Daphnet_point_mask_0.1_4a100_ddp_time.txt"

echo "Job started on $(hostname) at $(date)"
echo "cwd=$(pwd)"
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
  torchrun --standalone --nnodes=1 --nproc_per_node=4 run.py
  --mask 0
  --patience 3
  --num_edges 5
  --num_sensors 9
  --train_epochs 100
  --task_name imputation_graph
  --is_training 1
  --root_path ./dataset/daphnet_data/
  --model_id GraphNet_Daphnet_point_mask_0.1_4a100_ddp
  --mask_rate 0.1
  --model GraphNet
  --data Daphnet
  --features M
  --seq_len 96
  --label_len 0
  --pred_len 0
  --e_layers 2
  --d_layers 1
  --factor 3
  --enc_in 9
  --dec_in 9
  --c_out 9
  --batch_size 64
  --num_workers 2
  --d_model 64
  --des Exp
  --itr 1
  --top_k 3
  --learning_rate 0.0001
  --use_multi_gpu
  --devices 0,1,2,3
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
