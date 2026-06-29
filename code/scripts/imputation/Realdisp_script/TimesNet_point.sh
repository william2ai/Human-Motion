export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name imputation_graph \
  --mask 0 \
  --is_training 1 \
  --patience 3 \
  --train_epochs 1 \
  --root_path ./dataset/Realdisp/ \
  --model_id TimesNet_Realdisp_point_mask_0.7 \
  --mask_rate 0.7 \
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
  --learning_rate 0.001 \
  --batch_size 32 \
  --d_model 64 \
  --d_ff 64 \
  --des 'Exp' \
  --itr 1 \
  --top_k 3 
  # &> ./Records/Realdisp/mask_0.5/TimesNet_Realdisp_point_mask_0.5.txt
