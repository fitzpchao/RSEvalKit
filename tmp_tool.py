import os
import sys
import csv
csv.field_size_limit(sys.maxsize)
import json
import base64
from io import BytesIO

from PIL import Image
from tqdm import tqdm


# 这里简单写点临时用的数据处理脚本

def convert_rel_to_abs_path():
    dataset_base = r'datasets'
    json_list = [name for name in os.listdir(dataset_base) if os.path.splitext(name)[1] == '.json']
    
    for json_name in json_list:
        json_file = os.path.join(dataset_base, json_name)
        with open(json_file, 'r') as f:
            json_dict = json.load(f)

        new_json_dict = []
        for o in json_dict:
            o['image_path'] = os.path.basename(o['image_path'])
            new_json_dict.append(o)
        
        with open(json_file, 'w') as f:
            json_dict = json.dump(new_json_dict, f)
    pass


if __name__ == '__main__':
    convert_rel_to_abs_path()
