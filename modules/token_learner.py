'''
Implementation of ATD based on TokenLearner and A3.

Copyright 2023 xuefanfu
'''

import torch.nn as nn


# update decoder
class TokenLearner(nn.Module):

    def __init__(self, input_embed_dim, out_token=30):
        super().__init__()
        self.token_norm = nn.LayerNorm(input_embed_dim)
        self.tokenLearner1 = nn.Sequential(nn.Conv2d(257, 257, kernel_size = (1,1), stride=1,  bias=False),
                                          nn.Conv2d(257, 27, kernel_size = (1,1), stride=1, bias=False))
        self.tokenLearner2 = nn.Sequential(nn.Conv2d(768, 768, kernel_size = (1,1), stride=1, groups=8, bias=False),
                                          nn.Conv2d(768, 38, kernel_size = (1,1), stride=1, bias=False))
        self.norm_1 = nn.LayerNorm(38)


    def forward(self, x):
        B = x.shape[0]
        x = self.token_norm(x) # [bs, 257, 768]

        x = x.unsqueeze(-1) # [bs, 257, 768, 1]
        x  = self.tokenLearner1(x) # [bs, 27, 768, 1].
        x = x.transpose(1, 2)# [bs, 768, 27, 1].
        x= self.tokenLearner2(x)# [bs, 38, 27, 1].
        x = x.transpose(1, 2)
        x = x.squeeze(-1)
        x = self.norm_1(x)
        return x, x