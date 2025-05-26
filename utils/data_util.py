import sys
import os
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from RSEvalKit.smp import load, dump
LAST_MODIFIED = 231126000000

dataset_URLs = {
    'MMBench_DEV_EN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv", 
    'MMBench_TEST_EN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv", 
    'MMBench_DEV_CN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_CN.tsv", 
    'MMBench_TEST_CN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_CN.tsv", 
    "MMBench": "https://opencompass.openxlab.space/utils/VLMEval/MMBench.tsv",  # Link Invalid, Internal Only
    "MMBench_CN": "https://opencompass.openxlab.space/utils/VLMEval/MMBench_CN.tsv",    # Link Invalid, Internal Only
    'CCBench': "https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv", 
    'MME': "https://opencompass.openxlab.space/utils/VLMEval/MME.tsv", 
    'SEEDBench_IMG': "https://opencompass.openxlab.space/utils/VLMEval/SEEDBench_IMG.tsv", 
    'MMRS': "/mnt/petrelfs/pangchao/project/rsbench/rs_lr_hallucination_en.tsv",
    'MMRS_reasoning':"/mnt/petrelfs/pangchao/project/rsbench/reason_dota_en_with_image.tsv",
    'MMRS_grounding':'/mnt/petrelfs/pangchao/project/rsbench/reason_dota_en_with_image_grounding.tsv',
    'MMRS_caption':'/mnt/petrelfs/pangchao/project/rsbench/genmini/geolocation/sat_caption_list.tsv',
    'MMRS_caption_dota':'/mnt/petrelfs/pangchao/project/rsbench/genmini/vis/dota_caption_list.tsv',
    'MMRS_caption_cvusa':'/mnt/petrelfs/pangchao/project/rsbench/genmini/vis/cvusa_caption_list.tsv',
    'MMRS_pope':'/mnt/petrelfs/pangchao/project/rsbench/genmini/vis/data_test_polling.tsv',

    "rs_visibility_v0.0": "/mnt/petrelfs/share_data/wj/rs_repo/LMUData/rs_visibility_v0.0__small_things.tsv",
    "rs_ResEstimate_v0.0_336_mcq": "/mnt/petrelfs/share_data/wj/rs_repo/LMUData/rs_ResEstimate_v0.0_336_mcq.tsv",
    "MMRS_ResVisibility_v0_gsd10": "/mnt/petrelfs/share_data/wj/rs_repo/LMUData/rs_ResVisibility_v0_gsd10_a.tsv",
    "MMRS_ResVisibility_v0_gsd24": "/mnt/petrelfs/share_data/wj/rs_repo/LMUData/rs_ResVisibility_v0_gsd24_a.tsv",

    'rs_list_samples_target_azimuth_v0_mcq':'list_samples_v0_target_azimuth_mcq.tsv',
    'rs_list_samples_target_azimuth_v1_mcq':'list_samples_v1_target_azimuth_mcq.tsv',
    'rs_list_samples_imgType_mcq':'list_samples_imgType_mcq.tsv',
    "MMRS_dota-test-dev_pan_honesty_list":'dota-test-dev_pan_honesty_list.tsv',
    "MMRS_resolution":'/mnt/petrelfs/pangchao/project/rsbench/bench/resolution/list/list_samples_resolution_val.tsv',


    ###### Final Tsv ######
    # 'rs_position_mo_mc':'dota_test_position_single_s_mcq.tsv',
    # 'rs_presence_mo_judge':'/mnt/petrelfs/pangchao/ljy_files/datasets/dota_test_polling_s.tsv',
    # 'rs_counting_mo_open':'/mnt/petrelfs/pangchao/ljy_files/LMUData/dota_test_counting_balance_range_s.tsv',
    # 'rs_relative_mo_mc':'dota_test_relative_single_s_mc.tsv',
    # 'rs_imagetype_mc':'/mnt/petrelfs/pangchao/LMUData/list_samples_imgType_mcq.tsv',
    # 'rs_azimuth_mc':'/mnt/petrelfs/pangchao/LMUData/list_samples_v1_target_azimuth_mcq.tsv',
    # 'rs_relative_mo_fo_mc':'dota_se_test_object_relative_single_s_mc.tsv',
    # 'rs_fasle_premise_gpt_open':'/mnt/petrelfs/pangchao/ljy_files/datasets/dota_test_false_premise_s.tsv',
    # 'rs_res_estimate_mc':'/mnt/petrelfs/share_data/liuyi2/tsv3/rs_ResEstimate_v1.0_512_mcq.tsv'
    ##############################
    #location
    # 'rs_position_mo_mc':'dota_test_position_single_s_mcq.tsv',
    # 'rs_relative_position_mo_mc':'dota_test_relative_single_s_mc.tsv',
    # 'rs_relative_position_mo_fo_mc':'dota_se_test_object_relative_single_s_mc.tsv',
    'rs_position_mo_mc': 'dota_test_position_single_s_mcq_v1.tsv',
    'rs_relative_position_mo_mc': 'dota_test_relative_single_s_mc_v1.tsv',
    'rs_relative_position_mo_fo_mc':
    'dota_se_test_object_relative_single_s_mc_v1.tsv',
    'rs_relative_position_mo_fo_v2_mc':'/mnt/petrelfs/pangchao/ljy_files/experiments/LLMAPP/outputs/dota_se_test_object_relative_single_sample_manual/dota_se_test_object_relative_single_manual_s.tsv',
    # presece
    'rs_presence_mo_judge':'/mnt/petrelfs/pangchao/ljy_files/datasets/dota_test_polling_s.tsv',
    'rs_presence_sar_judge':'/mnt/petrelfs/share_data/liuyi2/objdet_train/msar_exist_test.tsv',
    # counting
    'rs_counting_mo_open':'/mnt/petrelfs/pangchao/ljy_files/LMUData/dota_test_counting_balance_range_s.tsv',
    'rs_counting_fo_open':'/mnt/petrelfs/pangchao/ljy_files/datasets/crowdAI_val_counting_s.tsv',
    'rs_counting_sar_open':'/mnt/petrelfs/share_data/liuyi2/objdet_train/msar_counting_test.tsv',
    # imagetype
    'rs_imagetype_mc':'/mnt/petrelfs/pangchao/LMUData/list_samples_imgType_mcq.tsv',
    'rs_recognizable_pan_open':'/mnt/petrelfs/share_data/liuyi2/tsv2/dota-test-dev_pan_honesty_list_v1.tsv',
    'rs_recognizable_sar_open':'/mnt/petrelfs/share_data/liuyi2/tsv2/msar-test_sar_honesty_list_v1.tsv',
    # resolution
    'rs_res_estimate_mc':'/mnt/petrelfs/share_data/liuyi2/tsv3/rs_ResEstimate_v1.0_512_mcq.tsv',
    'rs_visbility_gsd24_open':'/mnt/petrelfs/share_data/liuyi2/tsv4/rs_ResVisibility_v1_gsd24_a.tsv',
    'rs_visbility_gsd10_open':'/mnt/petrelfs/share_data/liuyi2/tsv4/rs_ResVisibility_v1_gsd10_modify.tsv',
    'rs_measurement_open':'/mnt/petrelfs/share_data/liuyi2/actual_obj/dota_test.tsv',
    # segmentation label
    'rs_landcover_fbp_open':'/mnt/petrelfs/share_data/liuyi2/sem_train/fbp_test.tsv',
    'rs_landcover_gid_open':'/mnt/petrelfs/share_data/liuyi2/sem_train/gid_test.tsv',
    'rs_landcover_potsdam_open':'/mnt/petrelfs/share_data/liuyi2/sem_train/potsdam_test.tsv',
    'rs_landcover_deepglobe_open':'/mnt/petrelfs/share_data/liuyi2/sem_train/deepglobe_lc_test.tsv',
    # color
    'rs_color_open':'/mnt/petrelfs/pangchao/LMUData/color.tsv',
    # others
    'rs_azimuth_mc':'/mnt/petrelfs/pangchao/LMUData/list_samples_v1_target_azimuth_mcq.tsv',
    'rs_false_premise_gpt_open':'/mnt/petrelfs/pangchao/ljy_files/datasets/dota_test_false_premise_s.tsv',
    'rs_false_premise_postive_gpt_open':'dota_test_false_premise_right_question_s.tsv',
    # cls
    'cls_aid':'/mnt/petrelfs/pangchao/project/rsbench/bench/tsv_list/AID.tsv',
    'cls_euroast':'/mnt/petrelfs/pangchao/project/rsbench/bench/tsv_list/eurosat.tsv',
    'cls_METER_ML':'/mnt/petrelfs/pangchao/project/rsbench/bench/tsv_list/METER_ML.tsv',
    'cls_NWPU_RESISC45':'/mnt/petrelfs/pangchao/project/rsbench/bench/tsv_list/NWPU_RESISC45.tsv',
    'cls_SIRI':'/mnt/petrelfs/pangchao/project/rsbench/bench/tsv_list/SIRI_WHU.tsv',
    'WHU_RS19':'/mnt/petrelfs/pangchao/project/rsbench/bench/tsv_list/WHU_RS19.tsv',
    }
# dataset_URLs = bench_datasets

img_root_map = {
    'MMBench_DEV_EN': "MMBench", 
    'MMBench_TEST_EN': "MMBench", 
    'MMBench_DEV_CN': "MMBench", 
    'MMBench_TEST_CN': "MMBench", 
    "MMBench": "MMBench",  # Link Invalid, Internal Only
    "MMBench_CN": "MMBench",    # Link Invalid, Internal Only
    'CCBench': "CCBench", 
    'MME': "MME", 
    'SEEDBench_IMG': "SEEDBench_IMG", 
}
rs_root_map = {_:_ for _ in dataset_URLs.keys() if _.startswith(('rs_', 'MMRS_','cls_'))}
img_root_map.update(rs_root_map)

def DATASET_TYPE(dataset):
    if 'mmbench' in dataset.lower() or 'seedbench' in dataset.lower():
        return 'multi-choice'
    elif dataset.lower().endswith('_mc'):
        return 'multi-choice'
    elif dataset.lower().endswith('_judge'):
        return 'Y/N'
    elif dataset.lower().endswith('_open'):
        return 'QA'
    elif 'MME' in dataset:
        return 'Y/N'
    return 'QA'

class TSVDataset:
    
    def __init__(self, dataset='MMBench', img_root=None):

        self.data_root = LMUDataRoot()
        assert osp.exists(self.data_root)

        self.dataset = dataset
        self.dataset_type = DATASET_TYPE(dataset)
        
        url = dataset_URLs[dataset]
        file_name = url.split('/')[-1]
        data_path = osp.join(self.data_root, file_name)

        if osp.exists(data_path) and int(last_modified(data_path)) > LAST_MODIFIED:
            # if '/' in url:
            #     if osp.exists(url):
            #         shutil.copy(url, data_path,)
            #     else:
            #         warnings.warn("The dataset tsv is not downloaded")
            #         download_file(url, data_path)
            pass
        else:
            if osp.exists(url):
                shutil.copy(url, data_path)
            else:
                warnings.warn("The dataset tsv is not downloaded")
                
                download_file(url, data_path)

        data = smp.load(data_path)

        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        for k, v in image_map.items():
            if k >= 1000000 and self.dataset in ['MMBench', 'MMBench_CN', 'CCBench']:
                image_map[k] = image_map[k % 1000000]
            elif k % 2 == 1 and self.dataset in ['MME']:
                image_map[k] = image_map[k - 1]
        data['image'] = [image_map[k] for k in data['index']]

        
        self.data = data

        img_root = img_root if img_root is not None else osp.join('images', img_root_map[dataset])
        os.makedirs(img_root, exist_ok=True)
        self.img_root = img_root

    def __len__(self):
        return len(self.data)

    def build_prompt(self, line, dataset=None):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
        if not osp.exists(tgt_path):
            decode_base64_to_image_file(line['image'], tgt_path)
        
        if dataset in ['MMBench', 'MMBench_CN', 'CCBench', 'SEEDBench_IMG'] \
            or "mc" in dataset.lower():
            question = line['question']
            option_candidate = ['A', 'B', 'C', 'D', 'E']
            options = {
                cand: line[cand]
                for cand in option_candidate
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = '\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += question
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'

        elif dataset == 'MME' or 'MMRS' in dataset or dataset.startswith("rs_") :
            prompt = line['question']
        else:
            raise NotImplementedError

        return dict(image=tgt_path, text=prompt)
    
    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        mmqa_display(line)
    