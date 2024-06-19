#!/bin/bash

# deepspeed --master_port=29500 --include="localhost:0,1" llava/train/train_mem.py \
deepspeed llava/train/train_mem_demon.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./path/LLaVA/llava1.5-7b \
    --version v1 \
    --data_path ./data_mmsd4llava/train_mmsd2.json \
    --image_folder put_path_of_dataset_image_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-mmsd-lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none
    # --report_to wandb