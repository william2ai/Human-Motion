export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name imputation_graph \
  --mask 2 \
  --patience 3 \
  --is_training 0 \
  --train_epochs 100 \
  --root_path ./dataset/daphnet_data/ \
  --model_id TimesNet_Daphnet_general_mask_0.7 \
  --mask_rate 0.7 \
  --model $model_name \
  --data Daphnet \
  --features M \
  --seq_len 96\
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --batch_size 32 \
  --d_model 64 \
  --d_ff 64 \
  --des 'Exp' \
  --itr 1 \
  --top_k 3 \
  --learning_rate 0.001 
  # &> ./Records/Daphnet/mask_0.7/TimesNet_Daphnet_general_mask_0.7.txt
