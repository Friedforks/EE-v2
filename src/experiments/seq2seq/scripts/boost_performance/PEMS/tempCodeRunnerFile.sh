python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/PEMS/ \
#   --data_path PEMS03.npz \
#   --model_id PEMS03_48_12 \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 48 \
#   --pred_len 12 \
#   --e_layers 4 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --des 'Exp' \
#   --train_epochs 5\
#   --learning_rate 0.001 \
#   --itr 1