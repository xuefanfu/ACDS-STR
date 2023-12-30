"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn

# from modules.char_str_ori import create_char_str
from modules.acds_str_base import create_acds_str_base
from modules.acds_str_small import create_acds_str_small
# from modules.mgp_str import create_mgp_str

import math

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        
        if opt.Transformer == "ACDS-STR-Base":
            print("USE ACDS-STR-Base")
            self.acds_str= create_acds_str_base(batch_max_length=opt.batch_max_length+2, num_tokens=opt.num_class,freeze=opt.freeze)
        elif opt.Transformer == "ACDS-STR-Small":
            print("USE ACDS-STR-Small")
            self.acds_str= create_acds_str_small(batch_max_length=opt.batch_max_length+2, num_tokens=opt.num_class,freeze=opt.freeze)

    def forward(self, input, is_eval=False):
        prediction = self.acds_str(input, is_eval=is_eval)
        return prediction


