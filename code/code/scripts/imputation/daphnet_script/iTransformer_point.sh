export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer


  python -u run.py \
  --task_name imputation_graph \
  --is_training 0 \
  --root_path ./dataset/Realdisp/ \
  --model_id iTransformer_Realdisp_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --data Realdisp \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 81 \
  --dec_in 81 \
  --c_out 81 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001