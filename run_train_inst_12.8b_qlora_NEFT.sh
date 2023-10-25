CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=34321 train_instruct_supervision_NEFT.py \
    --model_name_or_path  experiments/poly12.8b-DAPT \
    --data_path data/inst_edu_v4.json \
    --optim paged_adamw_32bit \
    --fp16 \
    --torch_dtype float16 \
    --output_dir experiments/poly12.8b-DAPT2INST_v4 \
    --run_name poly12.8b-DAPT2INST_v4_qlora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --preprocessing_num_workers 6 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --lr_scheduler_type "linear" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 2 \
    --block_size 1024 \
    --model_max_length 1024 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --bits 4