export CUDA_VISIBLE_DEVICES=0

model_name=GraphNet

python -u run.py \
  --mask 1 \
  --patience 3 \
  --num_edges 5 \
  --num_sensors 9 \
  --train_epochs 1 \
  --task_name imputation_graph \
  --is_training 1 \
  --root_path ./dataset/daphnet_data/ \
  --model_id GraphNet_Daphnet_general_mask_0.5 \
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
  --top_k 9 \
  --learning_rate 0.001  
  # &> ./Records/Daphnet/mask_0.5/GraphNet_Daphnet_general_mask_0.5.txt
  # &> general_attention.txt


  

  
  
  
  
  
    

  
