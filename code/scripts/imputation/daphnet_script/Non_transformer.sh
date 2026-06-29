export CUDA_VISIBLE_DEVICES=0

model_name=Nonstationary_Transformer

python -u run.py \
  --mask 2 \
  --train_epochs 20 \
  --task_name imputation_graph \
  --is_training 1 \
  --root_path ./dataset/daphnet_data/ \
  --model_id Non_trans_Daphnet_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
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
  --d_model 256 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001 \
  --p_hidden_dims 196 196 \
  --p_hidden_layers 2
  # --p_hidden_layers 2 &> ./Records/Daphnet/Non_trans.txt
    