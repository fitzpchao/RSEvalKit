import torch
from PIL import Image
from abc import abstractproperty
import os
import os.path as osp
from smp import *
from utils.data_util import DATASET_TYPE


class VHM:

    INSTALL_REQ = True

    def __init__(self, 
                 name,
                 model_path_map,
                 **kwargs): 

        from vhm.model.builder import load_pretrained_model
        from vhm.mm_utils import get_model_name_from_path
        self.model_path_map = model_path_map
        assert name in self.model_path_map or osp.exists(name)
        if name in self.model_path_map:
            model_path = self.model_path_map[name]
        else:
            model_path = name
        print(model_path)
        assert osp.exists(model_path) or splitlen(model_path) == 2
        print(f"Loading model from {model_path}")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path, 
            model_base=None, 
            model_name=get_model_name_from_path(model_path), 
            device='cpu', 
            device_map='cpu'
        )
        # print(f"{self.image_processor} loaded.")
        self.model = self.model.cuda()
       
        self.conv_mode = 'v1'

        kwargs_default = dict(do_sample=False, temperature=1.0, max_new_tokens=512, top_p=1.0, num_beams=1)
        
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def build_prompt(self, line, dataset=None):
        from utils import img_root_map
        assert dataset is None or isinstance(dataset, str)
        img_root = osp.join('images', img_root_map[dataset])

        os.makedirs(img_root, exist_ok=True)
        idx = line['index']
        img = line['image']

        tgt_path = osp.join(img_root, f'{idx}.jpg')
        decode_base64_to_image_file(img, tgt_path)

        if dataset is not None and DATASET_TYPE(dataset) == 'multi-choice':
            question = line['question']
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            if hint is not None:
                question + hint + '\n' + question

            option_candidate = ['A', 'B', 'C', 'D', 'E']
            options = {
                cand: line[cand]
                for cand in option_candidate
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                question += f'\n{key}. {item}'
            prompt = question

            if not cn_string(prompt):
                prompt = prompt + "\n" + "Answer with the option's letter from the given choices directly."
            else:
                prompt = prompt + "\n" + "请直接回答选项字母。"
        else:
            prompt = line['question']

        return {'image': tgt_path, 'text': prompt}

    def generate(self, image_path, prompt, dataset=None):
        # print(prompt)
        from vhm.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from vhm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from vhm.conversation import conv_templates, SeparatorStyle
        # print(image_path)
        image = Image.open(image_path).convert('RGB')
        # print(image.shape)
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images([image], self.image_processor, args).to('cuda', dtype=torch.float16)

        # === just for test ===
        # image_tensor = torch.zeros_like(image_tensor)

        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # print(input_ids.shape)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        try:
            # print("Generating...")
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
            output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip().split("</s>")[0]
        except Exception as e:
            output = e
        return output