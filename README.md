# FreqTempNet / GraphNet

This repository contains the PyTorch implementation for GraphNet-based multivariate time-series imputation experiments. The main source tree is under `code/code/`, and the shared neural network layers are under `layers/`.

The current training path supports:

- Single-GPU training with `python run.py`.
- Multi-GPU DistributedDataParallel (DDP) training with `torchrun`.
- Daphnet and Realdisp imputation experiments.
- Point, block, and combined missing-value masks.

## Repository Layout

```text
code/code/
  run.py
  data_provider/
  exp/
  models/
  scripts/
  utils/
layers/
```

For normal training, run commands from `code/code/`:

```bash
cd code/code
```

On the cluster, this is typically the project root, for example:

```bash
cd /ibex/user/luy0f/work/FreqTempNet
```

## Environment

Use an environment with PyTorch, PyTorch Geometric, NumPy, pandas, scikit-learn, matplotlib, and the other project dependencies installed.

Example:

```bash
conda activate freqtemp
```

Check CUDA:

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

The Daphnet example expects:

```text
dataset/daphnet_data/
```

The Daphnet DDP example uses:

```bash
--root_path ./dataset/daphnet_data/
--data Daphnet
--num_sensors 9
--enc_in 9
--dec_in 9
--c_out 9
```

## DDP Multi-GPU Example

Use `torchrun` for multi-GPU training. Do not use plain `python run.py` for multi-GPU DDP runs.

Provided Slurm script:

```bash
code/code/scripts/imputation/daphnet_script/GraphNet_point_ddp.sh
```

GPU type and GPU count are controlled by your Slurm submission command and environment variables, not hardcoded in the script. Use the GPU type and count available on your cluster:

```bash
cd code/code
NUM_GPUS=<N> DEVICES=<comma-separated-local-ids> sbatch --gres=gpu:<gpu_type>:<N> scripts/imputation/daphnet_script/GraphNet_point_ddp.sh

# Example with four visible GPUs:
NUM_GPUS=4 DEVICES=0,1,2,3 sbatch --gres=gpu:<gpu_type>:4 scripts/imputation/daphnet_script/GraphNet_point_ddp.sh
```

The core launch command is:

```bash
CUDA_VISIBLE_DEVICES=<comma-separated-gpu-ids> torchrun --standalone --nnodes=1 --nproc_per_node=<N> run.py \
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
  --devices <comma-separated-gpu-ids>
```

In DDP mode, `--batch_size` is treated as the global batch size and must be divisible by the number of processes. With `--batch_size 64` and 4 GPUs, each process receives a per-GPU batch size of 16. This keeps the optimization setup comparable when you change the GPU count.

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

## Reproducibility Notes

- Multi-GPU training uses DDP, not `nn.DataParallel`.
- Graph convolution layers are registered before optimizer creation, so Adam sees all trainable parameters.
- `--num_edges` is passed into dynamic graph construction.
- DDP validation losses are reduced across ranks.
- Only rank 0 saves checkpoints and final test outputs.
- Keep `--learning_rate 0.0001` unless changing the global batch size intentionally.
