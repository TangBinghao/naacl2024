import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import random

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score, accuracy_score
import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image
import math
import requests
from io import BytesIO
import re


def image_parser(image_file):
    out = image_file.split(args.sep)
    return out


def load_image(image_folder,image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(os.path.join(image_folder,image_file)).convert("RGB")
    return image


def load_images(image_folder,image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_folder,image_file)
        out.append(image)
    return out



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = json.load(open(args.data_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    mmsd_preds = []
    mmsd_golds = []
    
    for line in tqdm(questions):
        idx = line["id"]
        image_files = line["image"]
        qs = line["conversations"][0]["value"]
        gold_label = line["conversations"][1]["value"]
        
            
        cur_prompt = qs
        # if model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # prompt = cur_prompt
        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_files = image_parser(image_files)
        images = load_images(args.image_folder,image_files)
        # print(images)
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        )
        if type(images_tensor) is list:
            # print(len(images_tensor))
            images_tensor = [image.to(model.device, dtype=torch.float16) for image in images_tensor]
        else:
            # print(images_tensor.shape)
            images_tensor = images_tensor.to(model.device, dtype=torch.float16)
        
        # image = Image.open(os.path.join(args.image_folder, image_files))
        # images_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().cuda()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        
        def prefix_allowed_tokens_fn(batch_id, batch_tokens):
            return tokenizer.convert_tokens_to_ids(['0', '1'])
        with torch.inference_mode():
            import time
            # st = time.time()
            # print("input_ids",input_ids)
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, # constraint greedy docoding
                use_cache=True)
            # print("time:", time.time()-st)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        # TODO: fix
        # some errors without constraint greedy docoding
        if "1" in outputs and "0" not in outputs:
            outputs = "1"
        elif "0" in outputs:
            outputs = "0"

        # print()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "outputs": outputs,
                                   "gold_label": gold_label,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        mmsd_preds.append(outputs)
        mmsd_golds.append(gold_label)
        ans_file.flush()
    
    res = {}
    res['Acc'] = accuracy_score(mmsd_preds, mmsd_golds)
    # res['F1_SA'] = f1_score(golds, preds, average='weighted')
    res['F1_macro'] = f1_score(mmsd_preds, mmsd_golds, average='macro')
    res['P_macro'] = precision_score(mmsd_preds, mmsd_golds, average='macro')
    res['R_macro'] = recall_score(mmsd_preds, mmsd_golds, average='macro')
    ans_file.write(json.dumps(res) + "\n")
    ans_file.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./dataset_image")
    parser.add_argument("--data-file", type=str, default="./data_mmsd4llava/test_mmsd2.json")
    parser.add_argument("--answers-file", type=str, default="answer.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=5)
    args = parser.parse_args()

    eval_model(args)
