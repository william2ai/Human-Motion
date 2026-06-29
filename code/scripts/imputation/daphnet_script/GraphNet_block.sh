export CUDA_VISIBLE_DEVICES=0

model_name=GraphNet

python -u run.py \
  --mask 1 \
  --patience 3 \
  --num_sensors 9 \
  --num_edges 5 \
  --train_epochs 100 \
  --task_name imputation_graph \
  --is_training 0 \
  --root_path ./dataset/daphnet_data/ \
  --model_id GraphNet_Daphnet_block_mask_0.5 \
  --mask_rate 0.5 \
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
  --learning_rate 0.001 
  # > 0.5_mask_block.txt
  # &> block_attention.txt
  # &> ./Records/Daphnet/mask_0.3/wo_Temporal_Daphnet_block_mask_0.3.txt

  

  
  
  
  
  
    

  
