import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_

from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        split_attn = False
        len_t = 49
        if split_attn:
            attn_t = attn[..., :len_t].softmax(dim=-1)
            attn_s = attn[..., len_t:].softmax(dim=-1)
            attn = torch.cat([attn_t, attn_s], dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_st(nn.Module):
    def __init__(self, dim, mode, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # NOTE: Small scale for attention map normalization

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        
        lens_z = 64  # Number of template tokens
        lens_x = 256  # Number of search region tokens
        if self.mode == 's2t':  # Search to template
            q = x[:, :lens_z]  # B, lens_z, C
            k = x[:, lens_z:]  # B, lens_x, C
            v = x[:, lens_z:]  # B, lens_x, C
        elif self.mode == 't2s':  # Template to search
            q = x[:, lens_z:]  # B, lens_x, C
            k = x[:, :lens_z]  # B, lens_z, C
            v = x[:, :lens_z]  # B, lens_z, C
        elif self.mode=='t2t':  # Template to template
            q = x[:, :lens_z]  # B, lens_z, C
            k = x[:, lens_z:]  # B, lens_z, C
            v = x[:, lens_z:]  # B, lens_z, C
        elif self.mode=='s2s':  # Search to search
            q = x[:, :lens_x]  # B, lens_x, C
            k = x[:, lens_x:]  # B, lens_x, C
            v = x[:, lens_x:]  # B, lens_x, C
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, lens_z, lens_x; B, lens_x, lens_z

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # B, lens_z/x, C
        x = x.transpose(1, 2)  # B, C, lens_z/x
        x = x.reshape(B, -1, C)  # B, lens_z/x, C; NOTE: Rearrange channels, marginal improvement
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.mode == 's2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 't2s':
            x = torch.cat([k, x], dim=1)
        elif self.mode == 't2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 's2s':
            x = torch.cat([x, k], dim=1)

        if return_attention:
            return x, attn
        else:
            return x


class Attention_ori(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Attention_o_policy(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, attn_drop=0., proj_drop=0., divide=False, gauss=False, early=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.divide = divide
        self.gauss = gauss
        self.early = early
        if self.divide:
            if not self.early:
                self.divide_global_transform = nn.Sequential(
                    nn.Linear(dim, 384)
                )
                self.divide_local_transform = nn.Sequential(
                    nn.Linear(dim, 384)
                )
            self.divide_predict = nn.Sequential(
                nn.Linear(dim * 2, 384) if self.early else nn.Identity(),
                nn.GELU(),
                nn.Linear(384, 192),
                nn.GELU(),
                nn.Linear(192, 2),
                nn.Identity() if self.gauss else nn.LogSoftmax(dim=-1)
            )
            if self.gauss:
                self.divide_gaussian_filter = nn.Conv2d(2, 2, kernel_size=5, stride=1, padding=2)
                self.init_gaussian_filter()
                self.divide_gaussian_filter.requires_grad = False

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, N = attn.size()
        group1 = policy[:, :, 0].reshape(B, 1, N, 1) @ policy[:, :, 0].reshape(B, 1, 1, N)
        group2 = policy[:, :, 1].reshape(B, 1, N, 1) @ policy[:, :, 1].reshape(B, 1, 1, N)
        group3 = policy[:, :, 0].reshape(B, 1, N, 1) @ policy[:, :, 2].reshape(B, 1, 1, N) + \
                 policy[:, :, 1].reshape(B, 1, N, 1) @ policy[:, :, 2].reshape(B, 1, 1, N) + \
                 policy[:, :, 2].reshape(B, 1, N, 1) @ policy[:, :, 2].reshape(B, 1, 1, N) + \
                 policy[:, :, 2].reshape(B, 1, N, 1) @ policy[:, :, 0].reshape(B, 1, 1, N) + \
                 policy[:, :, 2].reshape(B, 1, N, 1) @ policy[:, :, 1].reshape(B, 1, 1, N)
        attn_policy = group1 + group2 + group3
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye

        # For stable training
        max_att, _ = torch.max(attn, dim=-1, keepdim=True)
        attn = attn - max_att
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def attn_in_group(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(1, -1, self.dim)
        return x

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
        self.divide_gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.divide_gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0).repeat(2, 2, 1, 1)
        self.divide_gaussian_filter.bias.data.zero_()

 
    def forward(self, x, decision, search_feat_len, attn_masking=True):
        B, N, C = x.shape 
               
        blank_policy = torch.zeros(B, search_feat_len, 1).to("cuda")
        template_policy = torch.zeros(B, N - search_feat_len, 3).to("cuda")
        template_policy[:, :, 0] = 1
        policy = torch.cat([template_policy, torch.cat([blank_policy, decision], dim=-1)], dim=1)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Make torchscript happy (cannot use tensor as tuple)

        if not attn_masking and not self.training:
            # Conduct three categories separately
            num_group1 = policy[:, :, 0].sum()
            num_group2 = policy[:, :, 1].sum()
            num_group3 = policy[:, :, 2].sum()
            _, E_T_ind = torch.topk(policy[:, :, 0], k=int(num_group1.item()), sorted=False)
            _, E_S_ind = torch.topk(policy[:, :, 1], k=int(num_group2.item()), sorted=False)
            _, E_A_ind = torch.topk(policy[:, :, 2], k=int(num_group3.item()), sorted=False)
            E_T_indices = E_T_ind.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, 1, self.head_dim)
            E_S_indices = E_S_ind.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, 1, self.head_dim)
            E_A_indices = E_A_ind.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, 1, self.head_dim)
            E_T_q = torch.gather(q, 2, E_T_indices)
            E_S_q = torch.gather(q, 2, E_S_indices)
            E_A_q = torch.gather(q, 2, E_A_indices)
            E_T_k = torch.gather(k, 2, torch.cat((E_T_indices, E_A_indices), dim=2))
            E_S_k = torch.gather(k, 2, torch.cat((E_S_indices, E_A_indices), dim=2))
            E_A_k = k
            E_T_v = torch.gather(v, 2, torch.cat((E_T_indices, E_A_indices), dim=2))
            E_S_v = torch.gather(v, 2, torch.cat((E_S_indices, E_A_indices), dim=2))
            E_A_v = v
            E_T_output = self.attn_in_group(E_T_q, E_T_k, E_T_v)
            E_S_output = self.attn_in_group(E_S_q, E_S_k, E_S_v)
            E_A_output = self.attn_in_group(E_A_q, E_A_k, E_A_v)

            x = torch.zeros_like(x, dtype=x.dtype, device=x.device)
            x = torch.scatter(x, 1, E_T_ind.unsqueeze(-1).repeat(1, 1, self.dim), E_T_output)
            x = torch.scatter(x, 1, E_S_ind.unsqueeze(-1).repeat(1, 1, self.dim), E_S_output)
            x = torch.scatter(x, 1, E_A_ind.unsqueeze(-1).repeat(1, 1, self.dim), E_A_output)
            x = self.proj(x)
        else:
            # Conduct three categories together
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.softmax_with_policy(attn, policy)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        # return x, decision
        return x

from lib.models.layers.SFTS import SFTS

class decision(nn.Module):
    def __init__(self):
        super(decision, self).__init__()
        self.SFTS = SFTS(ratio=0.08)  # 0.08
               
    def forward(self, x_v_a, x_i_a, img_path=None, writer=None, epoch=None, gauss=False, search_feat_len=256, threshold=0., ratio=0.):
        divide_prediction = self.SFTS(RGB_attn=x_v_a,TIR_attn=x_i_a,img_path=img_path,epoch=epoch, writer=writer)

                
        return divide_prediction
             
    

    