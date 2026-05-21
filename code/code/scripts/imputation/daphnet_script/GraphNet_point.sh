export CUDA_VISIBLE_DEVICES=0

model_name=GraphNet

python -u run.py \
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
  --d_model 64 \
  --des 'Exp' \
  --itr 1 \
  --top_k 3 \
  --learning_rate 0.001 &> ./Records/Daphnet/mask_0.1/Daphnet_point_mask_0.1.txt
  # &> menkong_point.txt
  # &> ./Records/Daphnet/mask_0.3/wo_Temporal_Daphnet_point_mask_0.3.txt

  

  
  
  
  
  
    

  
