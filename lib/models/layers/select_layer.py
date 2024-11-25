from functools import partial
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import  SelectBlock_sapd
    
class SelectLayer_sapd(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, divide=False):
        super().__init__()
        self.block = SelectBlock_sapd(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
                divide=True)
        
    def forward(self, x_v, x_i, attn_list_v, attn_list_i, template_mask_v, template_mask_i, search_feat_len, threshold, tgt_type):
        
        x_v, x_i, decision, loss = self.block(x_v, x_i, attn_list_v, attn_list_i, template_mask_v, template_mask_i, search_feat_len,
                                   threshold=threshold, tgt_type=tgt_type)
        return x_v, x_i, decision, loss


