# FreqTempNet / GraphNet

This repository contains the PyTorch implementation for GraphNet-based multivariate time-series imputation experiments. The current training path supports single-GPU training and multi-GPU DistributedDataParallel (DDP) training with `torchrun`.

## What This Code Does

- Runs imputation experiments for multivariate sensor time series.
- Supports Daphnet and Realdisp loaders.
- Uses frequency-aware dynamic graph construction and temporal convolution blocks.
- Supports point, block, and combined missing-value masks.
- Supports DDP multi-GPU training for faster large-scale experiments.

## Environment

Create or activate an environment with PyTorch, PyTorch Geometric, NumPy, pandas, scikit-learn, matplotlib, and the other project dependencies installed.

Example:

```bash
conda activate freqtemp
cd /ibex/user/luy0f/work/FreqTempNet
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

## Dataset Layout

The Daphnet example expects the dataset here:

```text
dataset/daphnet_data/
```

The command below uses:

```bash
--root_path ./dataset/daphnet_data/
--data Daphnet
--num_sensors 9
--enc_in 9
--dec_in 9
--c_out 9
```

## DDP Multi-GPU Example

Use `torchrun` for multi-GPU training. Do not launch multi-GPU training with plain `python run.py`; DDP needs one process per GPU.

The provided Slurm script is:

```bash
scripts/imputation/daphnet_script/GraphNet_point_ddp.sh
```

GPU type and GPU count are controlled by the Slurm submission command and environment variables, not hardcoded in the script. Use the GPU type and count available on your cluster:

```bash
NUM_GPUS=<N> DEVICES=<comma-separated-local-ids> sbatch --gres=gpu:<gpu_type>:<N> scripts/imputation/daphnet_script/GraphNet_point_ddp.sh

# Example with four visible GPUs:
NUM_GPUS=4 DEVICES=0,1,2,3 sbatch --gres=gpu:<gpu_type>:4 scripts/imputation/daphnet_script/GraphNet_point_ddp.sh
```

The core launch command is:

```bash
NUM_GPUS=4 DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" run.py \
  --mask 0 \
  --patience 3 \
  --num_edges 5 \
  --num_sensors 9 \
  --train_epochs 100 \
  --task_name imputation_graph \
  --is_training 1 \
  --root_path ./dataset/daphnet_data/ \
  --model_id GraphNet_Daphnet_point_mask_0.1_4gpu_ddp \
  --mask_rate 0.1 \
  --model GraphNet \
  --data Daphnet \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --batch_size 64 \
  --num_workers 2 \
  --d_model 64 \
  --des Exp \
  --itr 1 \
  --top_k 3 \
  --learning_rate 0.0001 \
  --use_multi_gpu \
  --devices "${DEVICES}"
```

In DDP mode, `--batch_size` is treated as the global batch size. For example, with `--batch_size 64` and `NUM_GPUS=4`, each process receives a per-GPU batch size of 16. This keeps the optimization setup comparable when you change the GPU count.

## Outputs

Training and evaluation artifacts are written to:

```text
checkpoints/
results/
test_results/
classification/
Records/
```

The DDP script writes logs to:

```text
Records/Daphnet/mask_0.1/<MODEL_ID>.txt
Records/Daphnet/mask_0.1/<MODEL_ID>_time.txt
```

## Notes For Reproducible Experiments

- Multi-GPU training uses DDP, not `nn.DataParallel`.
- Graph convolution layers are registered before optimizer creation, so trainable parameters are not missed by Adam.
- `--num_edges` is passed into dynamic graph construction.
- DDP validation losses are reduced across ranks.
- Only rank 0 saves checkpoints and final test outputs.
- Keep `--learning_rate 0.0001` unless you intentionally change the global batch size.

## Single-GPU Fallback

For single-GPU runs, use plain Python without `--use_multi_gpu`:

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --mask 0 \
  --patience 3 \
  --num_edges 5 \
  --num_sensors 9 \
  --train_epochs 100 \
  --task_name imputation_graph \
  --is_training 1 \
  --root_path ./dataset/daphnet_data/ \
  --model_id GraphNet_Daphnet_point_mask_0.1 \
  --mask_rate 0.1 \
  --model GraphNet \
  --data Daphnet \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --batch_size 64 \
  --d_model 64 \
  --des Exp \
  --itr 1 \
  --top_k 3 \
  --learning_rate 0.0001
```
