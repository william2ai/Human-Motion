export CUDA_VISIBLE_DEVICES=0

model_name=GraphNet

python -u run.py \
  --mask 2 \
  --task_name imputation_graph \
  --patience 3 \
  --num_sensors 81 \
  --num_edges 5 \
  --train_epochs 100 \
  --is_training 0 \
  --root_path ./dataset/Realdisp/ \
  --model_id GraphNet_Realdisp_general_mask_0.1 \
  --mask_rate 0.1 \
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
  --batch_size 64 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 3 \
  --learning_rate 0.001 &> ./Records/Realdisp/mask_0.1/GraphNet_Realdisp_general_mask_0.1.txt
