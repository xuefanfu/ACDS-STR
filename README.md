# Attention Guidance by Cross-Domain Supervision Signals for Scene Text Recognition

![data_demo](/assets/images/networks.png)


### Install requirements

```
pip install -r requirements.txt
```

We will release the following contents for **ACDS-STR**:exclamation:
- [x] code
- [x] Traing Detail
- [x] Training Dataset
- [x] Evaluation Dataset

### Dataset

The training including MJ and reconstituted ST and evaluation dataset will be released.

- Training datasets

    1. [MJSynth]() (MJ): 
    2. [SynthText]() (ST):

- Evaluation datasets
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
you can attain the pre-train model weights from (https://pan.baidu.com/s/17dR48P12pP2bOUX7_B8cWg). The password is 253a.


### experiments result
![experiments reuslts](/assets/images/experiments_result.png)

### infer
```
sh test.sh
```

### eval
```
sh test_val.sh
```

### Train
```
sh train.sh
```

## Acknowledgements
This implementation has been based on these repositories [ViTSTR](https://github.com/roatienza/deep-text-recognition-benchmark), [MGP-STR](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/MGP-STR), [TokenLearner](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner).


## *License*

ACDS-STR is released under the terms of the [Apache License, Version 2.0](LICENSE).

```
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
