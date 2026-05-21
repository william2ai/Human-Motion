export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name imputation_graph \
  --mask 1 \
  --is_training 0 \
  --patience 3 \
  --train_epochs 100 \
  --root_path ./dataset/Realdisp/ \
  --model_id TimesNet_Realdisp_block_mask_0.1 \
  --mask_rate 0.1 \
  --model $model_name \
  --data Realdisp \
  --features M \
  --seq_len 96\
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 81 \
  --dec_in 81 \
  --c_out 81 \
  --batch_size 64 \
  --d_model 64 \
  --d_ff 64 \
  --des 'Exp' \
  --itr 1 \
  --top_k 3 \
  --learning_rate 0.001
  #  &> ./Records/Realdisp/mask_0.1/TimesNet_Realdisp_block_mask_0.1.txt
