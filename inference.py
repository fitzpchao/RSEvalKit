import os
import re
import math
import requests
import json
import argparse
from io import BytesIO

import torch
from PIL import Image
from tqdm import tqdm
# import shortuuid

from vhm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from vhm.conversation import conv_templates, SeparatorStyle
from vhm.model.builder import load_pretrained_model
from vhm.utils import disable_torch_init
from vhm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from vhm import VHM


# rs_v7.2
model_path_map = {
    'VHM':'~/cks/vhm-7b_sft'
    
}


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def convt_qa(conversations):
   values = [conversation['value'] for conversation in conversations]
   query = values[0]
   answer = values[1]
   return query,answer


def infer_single(model, args):
    
    fn_full = args.image_path

    q= args.question
    
    try:
        with torch.inference_mode():
            outputs = model.generate(fn_full, q)
        outputs = outputs
    except Exception as e:
        print(f'An error occurred: {e}')
        outputs = str(e)
    return outputs


def infer_model(args):
    # Model
    disable_torch_init()
    model = VHM(name=args.model_name, model_path_map=model_path_map)

    out = infer_single(model, args)
    print(out)

    

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="h2rsvlm_pretrain")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    args = parser.parse_args()
    args.image_path = ''
    args.question = 'Please describe the remote sensing image.'
    if args.image_path is None or args.question is None:
        print("Please provide image_path and question")
        exit(0)
    
    infer_model(args)

