# Leveraging Generative Large Language Models with Visual Instruction and Demonstration Retrieval for Multimodal Sarcasm Detection
The official repository of the paper ''Leveraging Generative Large Language Models with Visual Instruction
and Demonstration Retrieval for Multimodal Sarcasm Detection'' published at the conference NAACL 2024.
# RedEval
The out-of-distribution (OOD) dataset RedEval is available at https://drive.google.com/drive/folders/1On1IFLNRLWlh--YqTwSlxm56Io-L2EO2?usp=drive_link.

# Code

## Data Preparation
```
cd data_mmsd4llava
python save_features.py
python prepare_data.py
```

## Fintune
```
sh ./scripts/v1_5/finetune_task_lora.sh
```
## Infer
```
sh ./scripts/v1_5/eval_mmsd.sh
```
