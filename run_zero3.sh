export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_P2P_LEVEL="NVL" 
export MODEL_PATH=''
export SAVE_PATH=''
export DS_SKIP_CUDA_CHECK=1

config_file="/accelerator_config_zero3.yaml"

accelerate launch --num_processes 4 --main_process_port 12345 --config_file $config_file train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "/PersonaMathQA.json" \
    --data_length 10000000 \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
