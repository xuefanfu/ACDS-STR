# Attention Guidance by Cross-Domain Supervision Signals for Scene Text Recognition

![data_demo](/assets/images/networks.png)


### Install requirements

```
pip3 install -r requirements.txt
```

### Dataset

The lmdb dataset which contains MJ and reconstituted ST will be released.

- Training datasets

    1. [MJSynth]() (MJ): 
    2. [SynthText]() (ST):

- Evaluation datasets
  LMDB datasets can be downloaded from [BaiduNetdisk(passwd:1dbv)](https://pan.baidu.com/s/1RUg3Akwp7n8kZYJ55rU5LQ), [GoogleDrive](https://drive.google.com/file/d/1dTI0ipu14Q1uuK4s4z32DqbqF3dJPdkk/view?usp=sharing).<br>
    1. ICDAR 2013 (IC13(857))
    2. ICDAR 2013 (IC13(1015))
    3. ICDAR 2015 (IC15(1811))
    4. ICDAR 2015 (IC15(2077))
    5. IIIT5K Words (IIIT)
    6. Street View Text (SVT)
    7. Street View Text-Perspective (SVTP)
    8. CUTE80 (CUTE)

- The structure of data folder as below.
```
data
├── evaluation
│   ├── CUTE80
│   ├── IC13_857
|   ├── IC13_1015
│   ├── IC15_1811
│   ├── IC15_2077
│   ├── IIIT5k_3000
│   ├── SVT
│   └── SVTP
├── training
│   ├── MJ
│   │   ├── MJ_test
│   │   ├── MJ_train
│   │   └── MJ_valid
│   └── ST
```



### Pretrained Models 

Available model weights:

| Tiny | Small  | Base |
| :---: | :---: | :---: |
|[ACDS-STR-Base-STD]([https://github.com/AlibabaResearch/AdvancedLiterateMachinery/releases/download/V1.0.1-ECCV2022-model/mgp_str_tiny_patch4_32_128.pth](https://pan.baidu.com/s/17dR48P12pP2bOUX7_B8cWg?pwd=253a))|[ACDS-STR-Base-RD]([https://github.com/AlibabaResearch/AdvancedLiterateMachinery/releases/download/V1.0.1-ECCV2022-model/mgp_str_small_patch4_32_128.pth](https://pan.baidu.com/s/17dR48P12pP2bOUX7_B8cWg?pwd=253a))


### experiments result
![experiments reuslts](/assets/images/experiments_result.png)



### Run demo with pretrained model
1. Download pretrained model 
2. Add image files to test into `demo_imgs/`
3. Run demo.py
```
mkdir demo_imgs/attens
CUDA_VISIBLE_DEVICES=0 python3 demo.py --Transformer mgp-str \
--TransformerModel=mgp_str_base_patch4_3_32_128 --model_dir mgp_str_base.pth --demo_imgs demo_imgs/
```


### Train

MGP-STR-base

```
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --master_port 29501 train_final_dist.py --train_data data/training \
--valid_data data/evaluation  --select_data MJ-ST  \
--batch_ratio 0.5-0.5  --Transformer mgp-str \
--TransformerModel=mgp_str_base_patch4_3_32_128 --imgH 32 --imgW 128 \
--manualSeed=226 --workers=12 --isrand_aug --scheduler --batch_size=100 --rgb \
--saved_path <path/to/save/dir> --exp_name mgp_str_patch4_3_32_128 --valInterval 5000 --num_iter 2000000 --lr 1
```

### Multi-GPU training

MGP-STR-base on a 2-GPU machine

It is recommended to train larger networks like MGP-STR-Small and MGP-STR-Base on a multi-GPU machine. To keep a fixed batch size at `100`, use the `--batch_size` option. Divide `100` by the number of GPUs. For example, to train MGP-STR-Small on a 2-GPU machine, this would be `--batch_size=50`.

```
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 29501 train_final_dist.py --train_data data/training \
--valid_data data/evaluation  --select_data MJ-ST  \
--batch_ratio 0.5-0.5  --Transformer mgp-str \
--TransformerModel=mgp_str_base_patch4_3_32_128 --imgH 32 --imgW 128 \
--manualSeed=226 --workers=12 --isrand_aug --scheduler --batch_size=50 --rgb \
--saved_path <path/to/save/dir> --exp_name mgp_str_patch4_3_32_128 --valInterval 5000 --num_iter 2000000 --lr 1
```


### Test

Find the path to `best_accuracy.pth` checkpoint file (usually in `saved_path` folder).

```
CUDA_VISIBLE_DEVICES=0 python3 test_final.py --eval_data data/evaluation --benchmark_all_eval --Transformer mgp-str  --data_filtering_off --rgb --fast_acc --TransformerModel=mgp_str_base_patch4_3_32_128 --model_dir <path_to/best_accuracy.pth>
```

## Visualization
The illustration of spatial attention masks on Character A3 module, BPE A3 module and WordPiece A3 module, respectively.

![cases](./figures/attens.png)


## Acknowledgements
This implementation has been based on these repository [ViTSTR](https://github.com/roatienza/deep-text-recognition-benchmark), [CLOVA AI Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark), [TokenLearner](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner).


## Citation
If you find this work useful, please cite:

```
@inproceedings{ECCV2022mgp_str,
  title={Multi-Granularity Prediction for Scene Text Recognition},
  author={Peng Wang, Cheng Da, and Cong Yao},
  booktitle = {ECCV},
  year={2022}
}
```

## *License*

MGP-STR is released under the terms of the [Apache License, Version 2.0](LICENSE).

```
MGP-STR is an algorithm for scene text recognition and the code and models herein created by the authors from Alibaba can only be used for research purpose.
Copyright (C) 1999-2022 Alibaba Group Holding Ltd. 

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
