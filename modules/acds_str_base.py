'''
Implementation of ACDS-STR based on MGP-STR and ViTSTR.

Copyright 2023 xuefanfu
'''
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from .token_learner import TokenLearner
from .transformer_vit import VisionTransformer
from .resnet import resnet45
import numpy as np
from PIL import Image


class CHARSTR(VisionTransformer):

    def __init__(self, batch_max_length, freeze, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.batch_max_length = batch_max_length
        self.char_tokenLearner = TokenLearner(self.embed_dim, self.batch_max_length)
        self.cnnBackbone = resnet45(freeze)
        self.conv1 = nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1,
                  bias=False)
        self.conv_fpn = nn.Sequential(nn.Conv2d(352, 352, kernel_size=1, stride=1,bias=False),
                                    nn.Conv2d(352, 1, kernel_size=1, stride=1,bias=False))
        self.conv_class = torch.nn.Parameter(torch.zeros(1, 1, 1), requires_grad=True)
        self.gated = torch.nn.Parameter(torch.zeros(1, 1536, 768), requires_grad=True)
        self.split_blk = [8,9]
        
        self.number_img=0
        
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.char_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    

    def forward_features(self, x):
        self.number_img=self.number_img+1
        B = x.shape[0]
        x,fpn_features = self.cnnBackbone(x) # x bs,512,8,32
        fpn_features = self.conv_fpn(fpn_features) # bs,1,32,128
        conv_atten = F.interpolate(fpn_features, scale_factor=0.25, mode='nearest')
        conv_atten = conv_atten.flatten(2).transpose(1,2) # bs,256,1
        x = self.conv1(x) # x bs,768,8,32
        x = x.flatten(2).transpose(1,2)# x bs,256,768
        # x = self.patch_embed(x) # x torch.Size([20, 256, 768])

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        conv_class = self.conv_class.expand(B,-1,-1)
        conv_atten = torch.cat((conv_class,conv_atten),dim=1)

        for i,blk in enumerate(self.blocks):
            x,atten = blk(x)
            if i==7: # 
                atten = torch.sum(atten,dim=1)/12
                atten_com = torch.sigmoid(atten + conv_atten)
                x_atten = torch.matmul(atten_com, x) 
                x_atten = self.norm(x_atten)
            if i in self.split_blk:
                x_atten,atten = blk(x_atten)
                atten = torch.sum(atten,dim=1)/12
                atten_com = torch.sigmoid(atten + conv_atten)
                x_atten = torch.matmul(atten_com, x_atten) 
                x_atten = self.norm(x_atten)
        
        # gate 
        fuse_x_atten = self.norm(torch.matmul(atten_com, x)+x_atten)
        gate = torch.sigmoid(torch.matmul(torch.cat((x,fuse_x_atten),dim=2),self.gated))  
        fuse_x =  gate*x+(1-gate)*fuse_x_atten      
        attens = []
        char_attn, char_x = self.char_tokenLearner(fuse_x)
        char_out = char_x
        attens = [char_attn] 
        
        return attens, char_out,fpn_features

    def forward(self, x, is_eval=False):
        attn_scores, char_out,fpn_features = self.forward_features(x)
        if is_eval:
            return [attn_scores, char_out,fpn_features]
        else:
            return [char_out,fpn_features]

def create_acds_str_base(batch_max_length, num_tokens,freeze):
    char_str =  CHARSTR(
        batch_max_length,freeze,img_size=(32,128),num_classes=num_tokens,patch_size=4, embed_dim=768, depth=10, num_heads=12, mlp_ratio=4, qkv_bias=True)
    char_str.reset_classifier(num_classes=num_tokens)
    return char_str




