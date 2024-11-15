# Official evaluation toolkit of [VHM: Versatile and Honest Vision Language Model for Remote Sensing Image Analysis](https://fitzpchao.github.io/vhm_page/)



## Setup
Create a new conda env, and Install the necessary dependencies:
```sh
conda create -n rseval
conda activate rseval
pip install -r requirements.txt
```

## Dataset
1. Please refer to the evaluation [data description](https://github.com/opendatalab/VHM/tree/main/docs/Data.md#vhm_eval-dataset) and download the [vhm_eval](https://huggingface.co/datasets/FitzPC/VHM_eval_dataset) dataset.
2. Prepare the datasets following the file structure below:

```
{dataset_base}/
    # image dirs
    abspos_c1f4_dota-test_mc/
        image0.jpg
        image1.jpg
        ...
    abspos_dota-test_mc/
    ...

    # json files
    abspos_c1f4_dota-test_mc.json
    abspos_dota-test_mc.json
    ...
```

## Evaluation
### Download VHM weights
Please refer to this [guide](https://github.com/opendatalab/VHM/blob/main/README.md#models) to download the corresponding VHM model weights.

### Single GPU:
```sh
$ CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 52302 ./model_eval_mp.py --task all --batch-per-gpu 1 --dataset-base ${dataset_base} --save-path ${your_save_path}
```

### Multiple GPUs:
If you want to evaluate our model on multiple GPUs, you can tweak the arguments ```--nproc_per_node``` and ```--batch-per-gpu```, then make sure that the value of these arguments follow the equation:
```
${nproc_per_node} = ${batch-per-gpu} × ${the number of your GPUs}
```

For example, to perform an evaluation on 4 GPUs, each of which has a batchsize of 3, you should run:
```sh
$ CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=12 --master_port 52302 ./model_eval_mp.py --task all --batch-per-gpu 3 --dataset-base ${dataset_base} --save-path ${your_save_path}
```
## Citation
Please refer to our paper for more technical details:

If this code is helpful to your research, please consider citing [our paper](https://arxiv.org/abs/2403.20213) by:

```
@misc{pang2024vhmversatilehonestvision,
      title={VHM: Versatile and Honest Vision Language Model for Remote Sensing Image Analysis}, 
      author={Chao Pang and Xingxing Weng and Jiang Wu and Jiayu Li and Yi Liu and Jiaxing Sun and Weijia Li and Shuai Wang and Litong Feng and Gui-Song Xia and Conghui He},
      year={2024},
      eprint={2403.20213},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.20213}, 
}
```
## Acknowledgements
We gratefully acknowledge the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) works.



