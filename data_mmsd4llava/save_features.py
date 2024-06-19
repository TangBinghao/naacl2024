import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import json
import utils
import numpy as np
import time
import pickle

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("/openai/clip-vit-large-patch14-336").to(device)
processor = CLIPProcessor.from_pretrained("/openai/clip-vit-large-patch14-336")
image_folder = "dataset_image"

data_folder = "/MMSD2.0" # MMSD2.0
max_len = 77
for data_set in ["valid","test","train"]:
    data = json.load(open(os.path.join(data_folder,data_set+'.json'),'r'))
    results = {}
    total = len(data)
    for i, sample in enumerate(data):
        image_id = str(sample["image_id"])
        image = Image.open(os.path.join(image_folder,image_id+'.jpg')).convert("RGB")
        text = sample["text"]
        tokenized_text = processor.tokenizer.tokenize(text)
        tokenized_ids = processor.tokenizer.convert_tokens_to_ids(tokenized_text)
        input_ids = [processor.tokenizer.bos_token_id] + tokenized_ids + [processor.tokenizer.eos_token_id]
        if len(input_ids) > max_len:
            input_ids  = input_ids[:max_len-1] + [processor.tokenizer.eos_token_id]
        
        attention_mask = [[1 for _ in range(len(input_ids))]]
        input_ids = [input_ids]
        # print(tokenized_text)
        # print(tokenized_ids)
        # print(processor.tokenizer.bos_token_id)
        # print(processor.tokenizer.eos_token_id)
        pixel_values = processor.image_processor(image)
        # print(pixel_values)
        inputs = {
            "input_ids": torch.LongTensor(input_ids).to(device),
            "attention_mask": torch.LongTensor(attention_mask).to(device),
            # "pixel_values": torch.LongTensor(pixel_values["pixel_values"]).to(device),
        }
        
        inputs_image = processor(images=image, return_tensors="pt", padding=True).to(device)
        # # print(inputs.keys())
        inputs["pixel_values"] = inputs_image["pixel_values"]
        # print(inputs)
        # print(processor(text=[text],images=image, return_tensors="pt", padding=True))
        outputs = model(**inputs)

        # text_features = outputs['text_model_output']['last_hidden_state'] # torch.Size([1, 26, 768])
        # image_features = outputs['vision_model_output']['last_hidden_state'] # torch.Size([1, 577, 1024])
        text_feature = outputs['text_model_output']['pooler_output'] # torch.Size([1, 768])
        image_feature = outputs['vision_model_output']['pooler_output'] # torch.Size([1, 1024])
        # print(image_feature)
        results[image_id] = {
            "text":sample["text"],
            "text_feature": text_feature.cpu().detach().numpy(),
            "image_feature": image_feature.cpu().detach().numpy(),
            "label": sample["label"],
            }
        # print(outputs[image_id])
        if i % 1000 == 0:
            print(f"processed {i}/{total}= {i/total}")
    with open(f"{data_folder}/{data_set}_features", 'wb') as handle:
        # print(results)
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # # cosine_scores = utils.cos_sim(text_feature, image_feature)
        # st = time.time()
        # cosine_scores = utils.cos_sim(torch.randn(1,1792), torch.randn(1,1792))

        # print(cosine_scores)
        # print(time.time() - st)

