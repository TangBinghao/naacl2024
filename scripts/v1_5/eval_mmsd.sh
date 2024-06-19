#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_mmsd \
    --model-path ./checkpoints_path \
    --model-base ./base_model_path/LLaVA/llava1.5-7b \
    --data-file ./data_mmsd4llava/test_mmsd2.json \
    --image-folder path_to_dataset_image \
    --answers-file ./data_mmsd4llava/out_prediction_mmsd2.json \
    --temperature 0 \
    --num_beams 1 \
    --conv-mode llava_v1
