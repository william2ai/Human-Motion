export CUDA_VISIBLE_DEVICES=1

model_name=GraphNet

python -u run.py \
  --mask 1 \
  --task_name imputation_graph \
  --patience 3 \
  --num_sensors 81 \
  --train_epochs 20 \
  --num_edges 5 \
  --is_training 1 \
  --root_path ./dataset/Realdisp/ \
  --model_id GraphNet_Realdisp_block_mask_0.7 \
  --mask_rate 0.7 \
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
  --batch_size 32 \
  --d_model 64 \
  --des 'Exp' \
  --itr 1 \
  --top_k 3 \
  --learning_rate 0.001 
  # &> ./Records/Realdisp/mask_0.1/GraphNet_Realdisp_block_mask_0.1.txt
