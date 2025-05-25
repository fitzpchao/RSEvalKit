# %%

# Summary:
# ------------------------------------------------------------
# A dumbed down version of the inference.py file with only the relevant imports and functions, abstructions were simplified,
# parameters were removed/hard coded, and explanations were added.
# Conversations are not supported, only single image and question are supported.
# 
# Requirements:



# %%
# IMPORTS
# ------------------------------------------------------------

## vhm_model.py
import sys
sys.path.append('/path/to/RSEvalKit')

import torch
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

## inference.py
from vhm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from vhm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from vhm.conversation import conv_templates, SeparatorStyle
from vhm.model.builder import load_pretrained_model

from vhm.utils import disable_torch_init

# %%
# MANUAL UTILS

## inference.py
def expand2square(pil_img, background_color):
    """
    Expands a PIL image to a square by padding with a background color while maintaining the original image centered.
    
    Args:
        pil_img (PIL.Image): Input PIL image
        background_color: Color to use for padding (RGB tuple or single value depending on image mode)
    
    Returns:
        PIL.Image: Square image with original content centered and padded
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


## vhm_model.py
class VHM_Inference:
    def __init__(self, model_path, **kwargs): 
        """
        VHM_Inference class for generating responses to prompts given an image.
        Args:
            model_path (str): Path to the model checkpoint, downloaded from huggingface at: https://huggingface.co/FitzPC/vhm_7B/tree/main
            **kwargs: Additional keyword arguments for the model.
        """
        # init
        # ------
        # setting device and conversation mode
        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        self.conv_mode = 'v1' # default conversation mode

        # setting generation config
        kwargs_default = dict(do_sample=False, temperature=1.0, max_new_tokens=512, top_p=1.0, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default # update with any additional kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config.")
        
        
        # load model
        print(f"Loading model from {model_path}")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None, 
            model_name="vhm-7b-v1.0", 
            device = self.device, 
            device_map=self.device
        )

    def process_image(self, image_path, image_constant_mean = [0.48145466, 0.4578275, 0.40821073]):
        # expand image to square, fill in color with the constant mean of the image as default
        fill_in_color = tuple(int(x*255) for x in image_constant_mean)
        
        # open image
        image = Image.open(image_path).convert('RGB')
        image = expand2square(image, fill_in_color)

        # processing image
        image = self.image_processor.preprocess(image, return_tensors='pt')[
                        'pixel_values'][0]
        
        # stack image into a tensor
        new_images = []
        new_images.append(image)
        new_images = torch.stack(new_images, dim=0)
        image_tensor = new_images.to('cuda', dtype=torch.float16)

        return image_tensor
        
        

    def generate(self, image_path, prompt):
        # process image
        image_tensor = self.process_image(image_path)

        # Process prompt by adding constant tokens
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conversation = conv_templates[self.conv_mode].copy()
        conversation.append_message(conversation.roles[0], inp)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()

        # tokenize processed prompt
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
        keywords = [stop_str]
        
        # set stopping criteria for generation
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # generate response
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
            output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip().split("</s>")[0]
        except Exception as e:
            output = e
        return output


# %%
# Manual Inference
default_config = {
    "model_name": "h2rsvlm_pretrain",
    "image_path": "/path/to/image.png",
    "question": "Please describe the remote sensing image."
}

disable_torch_init()
model = VHM_Inference(
    model_path="/path/to/model"
    )

with torch.inference_mode():
    outputs = model.generate(default_config['image_path'], default_config['question'])

print(f"Output: {outputs}")

# %%