CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 --master_port=34323 train_dapt.py \
    --model_name_or_path EleutherAI/polyglot-ko-5.8b \
    --data_path data/unified_DAPT_data.json \
    --optim adafactor \
    --fp16 \
    --torch_dtype float16 \
    --output_dir experiments/poly5.8b-DAPT \
    --run_name poly5.8b-DAPT_b3_ga8_g8_l1k_lr4e5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 5 \
    --preprocessing_num_workers 6 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --logging_steps 2 \
    --block_size 1024 \
    --model_max_length 1024 \
    --deepspeed ds_zero3-nooffload.json \
    --low_cpu_mem_usage

# --evaluation_strategy "epoch" \