import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
import numpy as np
from lib.models.layers.attn import Attention, Attention_st, Attention_o_policy, decision


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        topk = global_index.type(torch.int64)
        # print(topk.type())
        return tokens, tokens, global_index, None, topk

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)
    # removed_index = None

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)
    
    return tokens_new, tokens, keep_index, removed_index, topk_idx


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, x_other, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)
        lens_t = global_index_template.shape[1]
        removed_index_search = None

        topk_idx = global_index_search.type(torch.int64)
        x_ori = x
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            global_index_search_ori = global_index_search.clone().detach()
            x, x_ori, global_index_search, removed_index_search, topk_idx = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            # import ipdb; ipdb.set_trace()
            # x_other, _, _ = candidate_elimination(attn, x_other, lens_t, keep_ratio_search, global_index_search_ori, ce_template_mask)
        x_ori = x_ori + self.drop_path2(self.mlp2(self.norm3(x_ori)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, x_ori, x_other, global_index_template, global_index_search, removed_index_search, attn, topk_idx


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CASTBlock(nn.Module):

    def __init__(self, dim, num_heads, mode, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_reshape = Attention_st(dim, mode, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn_reshape(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
class SelectBlock_sapd(nn.Module):###单独计算
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, divide=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_decision = decision()
        self.attn_policy = Attention_o_policy(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, divide=divide)
        # Note: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.max_pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x_v, x_i, attn_list_v, attn_list_i, template_mask_v, template_mask_i, search_feat_len, threshold, tgt_type):
        
        mask = self.attn_decision(attn_list_v, attn_list_i)
        x = torch.cat((x_v[:, :64, :], x_i[:, :64, :], x_v[:, 64:, :], x_i[:, 64:, :]), dim=1)
        
        x_v = self.attn_policy(self.norm1(x_v), mask, search_feat_len, attn_masking=True)
        x_i = self.attn_policy(self.norm1(x_i), mask, search_feat_len, attn_masking=True)
        
        feat = torch.cat((x_v[:, :64, :], x_i[:, :64, :], x_v[:, 64:, :], x_i[:, 64:, :]), dim=1)
      
        x = x + self.drop_path(feat)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        x_v_output = torch.cat((x[:, :64, :], x[:, (2 * 64): (320 + 64), :]), dim=1)
        x_i_output = torch.cat((x[:, 64: (64 + 64), :], x[:, (320 + 64):, :]), dim=1)
               
        if self.training:
            boolean_mask = (mask[:, :, 0] == 1) & (mask[:, :, 1] == 0)
            boolean_mask_expanded = boolean_mask.unsqueeze(-1).type_as(x_v_output)
            
            x_v_sel = x_v_output[:,64:,:]* boolean_mask_expanded
            x_i_sel = x_i_output[:,64:,:]* boolean_mask_expanded            
            
            loss = nn.MSELoss()(x_v_sel, x_i_sel) 
        
            return x_v_output, x_i_output, mask, loss
        else:
            return x_v_output, x_i_output, mask, None
        

class SelectBlock_sapd_toge(nn.Module):###一起计算
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, divide=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_decision = decision()
        self.attn_policy = Attention_o_policy(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, divide=divide)
        # Note: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.max_pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x_v, x_i, attn_list_v, attn_list_i, template_mask_v, template_mask_i, search_feat_len, threshold, tgt_type):
        
        mask = self.attn_decision(attn_list_v, attn_list_i)
        x = torch.cat((x_v[:, :64, :], x_i[:, :64, :], x_v[:, 64:, :], x_i[:, 64:, :]), dim=1)
        
        feat = self.attn_policy(self.norm1(x), torch.cat((mask, mask),dim=1), search_feat_len*2, attn_masking=True)
      
        x = x + self.drop_path(feat)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        x_v_output = torch.cat((x[:, :64, :], x[:, (2 * 64): (320 + 64), :]), dim=1)
        x_i_output = torch.cat((x[:, 64: (64 + 64), :], x[:, (320 + 64):, :]), dim=1)
                
        if self.training:
            boolean_mask = (mask[:, :, 0] == 1) & (mask[:, :, 1] == 0)
            boolean_mask_expanded = boolean_mask.unsqueeze(-1).type_as(x_v_output)
            
            x_v_sel = x_v_output[:,64:,:]* boolean_mask_expanded
            x_i_sel = x_i_output[:,64:,:]* boolean_mask_expanded            
            
            loss = nn.MSELoss()(x_v_sel, x_i_sel) 
        
            return x_v_output, x_i_output, mask, loss
        else:
            return x_v_output, x_i_output, mask, None

