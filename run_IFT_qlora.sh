MODEL_PATH=experiments/poly12.8b-DAPT2INST_wo_edu
DATA_PATH=data/inst_edu_only.json

python3 caching.py --data_path $DATA_PATH --model_name $MODEL_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=34321 train_qlora.py \
    --model_name_or_path  $MODEL_PATH \
    --data_path $DATA_PATH \
    --optim paged_adamw_32bit \
    --fp16 \
    --torch_dtype float16 \
    --output_dir experiments/poly12.8b-DAPT2INST_third \
    --run_name poly12.8b-DAPT2INST_qlora \
    --neftune_noise_alpha 0 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 6 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "linear" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 2 \
    --block_size 512 \
    --model_max_length 512 \
    --bits 8 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05