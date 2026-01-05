"""
HPCM Phase 1: Replace s3 CrossAttentionCell with RWKV
Proof of concept for linear attention integration
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from ..base import BB as basemodel
from src.layers import PConvRB, conv2x2_down, deconv2x2_up, DWConvRB, conv1x1, conv4x4_down, deconv4x4_up

# Import RWKV modules
from ..rwkv_modules import (
    ensure_biwkv4_loaded,
    RWKVContextCell
)


class g_a(nn.Module):
    def __init__(self):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4
        
        self.branch = nn.Sequential(
            conv4x4_down(3,96),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(96,192),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(192,384),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(384,320),
        )

    def forward(self, x):
        return self.branch(x)


class g_s(nn.Module):
    def __init__(self):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4

        self.branch = nn.Sequential(
            deconv2x2_up(320,384),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv2x2_up(384,192),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv2x2_up(192,96),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv4x4_up(96,3),
        )

    def forward(self, x):
        out = self.branch(x)
        return out


class h_a(nn.Module):
    def __init__(self):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4

        self.branch = nn.Sequential(
            PConvRB(320, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(320,256),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(256,256),
        )

    def forward(self, x):
        out = self.branch(x)
        return out


class h_s(nn.Module):
    def __init__(self):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4

        self.branch = nn.Sequential(
            deconv2x2_up(256,256),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv2x2_up(256,320*2),
            PConvRB(320*2, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
        )

    def forward(self, x):
        out = self.branch(x)
        return out


class y_spatial_prior_s1_s2(nn.Module):
    def __init__(self, M):
        super().__init__()
        
        self.branch_1 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
        )
        self.branch_2 = nn.Sequential(
            DWConvRB(M*3),
            conv1x1(3*M,2*M),
        )

    def forward(self, x, quant_step):
        return self.branch_2(self.branch_1(x)*quant_step)


class y_spatial_prior_s3(nn.Module):
    def __init__(self, M):
        super().__init__()
        
        self.branch_1 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
            DWConvRB(M*3),
        )
        self.branch_2 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
            conv1x1(3*M,2*M),
        )

    def forward(self, x, quant_step):
        return self.branch_2(self.branch_1(x)*quant_step)


class CrossAttentionCell(nn.Module):
    """Original CrossAttentionCell from HPCM (for s1 and s2)"""
    def __init__(self, input_dim, hidden_dim, window_size, kernel_size, num_heads=32):
        super(CrossAttentionCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.conv_q = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

        self.window_size = window_size

    def forward(self, x_t, h_prev):
        x_t_init = x_t
        x_t = rearrange(x_t, 'b c (w1 p1) (w2 p2)  -> (b w1 w2) c p1 p2', p1=self.window_size, p2=self.window_size)
        h_prev = rearrange(h_prev, 'b c (w1 p1) (w2 p2)  -> (b w1 w2) c p1 p2', p1=self.window_size, p2=self.window_size)
        batch_size, C, H, W = x_t.size()
        q = self.conv_q(x_t)
        k = self.conv_k(h_prev)
        v = self.conv_v(h_prev)

        q = q.view(batch_size, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(batch_size, self.num_heads, self.head_dim, H * W)
        v = v.view(batch_size, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        attn_scores = torch.matmul(q, k) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous().view(batch_size, self.hidden_dim, H, W)
        attn_output = rearrange(attn_output, '(b w1 w2) c p1 p2  -> b c (w1 p1) (w2 p2)', w1=x_t_init.shape[2]//self.window_size, w2=x_t_init.shape[3]//self.window_size)

        h_t = attn_output + self.conv_out(x_t_init)

        return h_t


class HPCM_Phase1(basemodel):
    """
    HPCM Phase 1: s3 with RWKV, s1 and s2 with original CrossAttention
    
    Modifications:
        - attn_s3: CrossAttentionCell → RWKVContextCell
        - Reduces O(N^2) complexity to O(N*T) for scale-3 (full resolution)
        - Expected: 25-35% speedup in s3 loop, +0.1~0.2dB performance gain
    """
    
    def __init__(self, M=320, N=256):
        super().__init__(N)
        
        # Load CUDA kernels once
        ensure_biwkv4_loaded()
        
        self.g_a = g_a()
        self.g_s = g_s()
        self.h_a = h_a()
        self.h_s = h_s()

        self.y_spatial_prior_adaptor_list_s1 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(1))
        self.y_spatial_prior_adaptor_list_s2 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(3))
        self.y_spatial_prior_adaptor_list_s3 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(6))
        self.y_spatial_prior_s1_s2 = y_spatial_prior_s1_s2(M)
        self.y_spatial_prior_s3 = y_spatial_prior_s3(M)

        self.adaptive_params_list = [
            torch.nn.Parameter(torch.ones((1, M*3, 1, 1), device='cuda'), requires_grad=True) for _ in range(10)
        ]

        # s1 and s2: Original CrossAttentionCell
        self.attn_s1 = CrossAttentionCell(320*2, 320*2, window_size=4, kernel_size=1)
        self.attn_s2 = CrossAttentionCell(320*2, 320*2, window_size=8, kernel_size=1)
        
        # s3: RWKV ContextCell (Phase 1 modification)
        self.attn_s3 = RWKVContextCell(320*2, hidden_rate=4)
        
        self.context_net = nn.ModuleList(conv1x1(2*M,2*M) for _ in range(2))
    
    def forward(self, x, training=None):
        if training is None:
            training=self.training 
        
        y = self.g_a(x)
        z = self.h_a(y)
        
        if training:
            z_res = z - self.means_hyper
            z_hat = self.ste_round(z_res) + self.means_hyper
            z_likelihoods = self.entropy_estimation(self.add_noise(z_res), self.scales_hyper)
        else:
            z_res_hat = torch.round(z - self.means_hyper)
            z_hat = z_res_hat + self.means_hyper
            z_likelihoods = self.entropy_estimation(z_res_hat, self.scales_hyper)   

        params = self.h_s(z_hat)
        y_res, y_q, y_hat, scales_y = self.forward_hpcm(y, params, 
                                        self.y_spatial_prior_adaptor_list_s1, self.y_spatial_prior_s1_s2, 
                                        self.y_spatial_prior_adaptor_list_s2, self.y_spatial_prior_s1_s2, 
                                        self.y_spatial_prior_adaptor_list_s3, self.y_spatial_prior_s3, 
                                        self.adaptive_params_list, self.context_net, 
                                        )

        x_hat = self.g_s(y_hat)
        
        if training:
            y_likelihoods = self.entropy_estimation(self.add_noise(y_res), scales_y)
        else:
            y_res_hat = torch.round(y_res)
            y_likelihoods = self.entropy_estimation(y_res_hat, scales_y) 
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    def forward_hpcm(self, y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              adaptive_params_list, context_net, write=False):
        """Forward pass of HPCM with Phase 1 modifications (RWKV at s3)"""
        B, C, H, W = y.size()
        dtype = common_params.dtype
        device = common_params.device

        ############### 2-step scale-1 (s1) (4× downsample) coding
        mask_list_s2 = self.get_mask_for_s2(B, C, H, W, dtype, device)
        y_s2 = self.get_s1_s2_with_mask(y, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        mask_list_rec_s2 = self.get_mask_for_rec_s2(B, C, H // 2, W // 2, dtype, device)
        y_s1 = self.get_s1_s2_with_mask(y_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)

        scales_all, means_all = common_params.chunk(2,1)
        scales_s2 = self.get_s1_s2_with_mask(scales_all, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        scales_s1 = self.get_s1_s2_with_mask(scales_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)
        means_s2 = self.get_s1_s2_with_mask(means_all, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        means_s1 = self.get_s1_s2_with_mask(means_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)
        common_params_s1 = torch.cat((scales_s1, means_s1), dim=1)
        context_next = common_params_s1

        mask_list = self.get_mask_two_parts(B, C, H // 4, W // 4, dtype, device)
        y_res_list_s1 = []
        y_q_list_s1 = []
        y_hat_list_s1 = []
        scale_list_s1 = []

        for i in range(2):
            if i == 0:
                y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y_s1, scales_s1, means_s1, mask_list[i])
                y_res_list_s1.append(y_res_0)
                y_q_list_s1.append(y_q_0)
                y_hat_list_s1.append(y_hat_0)
                scale_list_s1.append(s_hat_0)
            else:
                y_hat_so_far = torch.sum(torch.stack(y_hat_list_s1), dim=0)
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params), adaptive_params_list[i - 1])
                context_next = self.attn_s1(context, context_next)
                scales, means = context.chunk(2, 1)
                y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y_s1, scales, means, mask_list[i])
                y_res_list_s1.append(y_res_1)
                y_q_list_s1.append(y_q_1)
                y_hat_list_s1.append(y_hat_1)
                scale_list_s1.append(s_hat_1)
        
        y_res = torch.sum(torch.stack(y_res_list_s1), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s1), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s1), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s1), dim=0)

        if write:
            y_q_write_list_s1 = [self.combine_for_writing_s1(y_q_list_s1[i]) for i in range(len(y_q_list_s1))]
            scales_hat_write_list_s1 = [self.combine_for_writing_s1(scale_list_s1[i]) for i in range(len(scale_list_s1))]
        
        y_res = self.recon_for_s2_s3(y_res, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        y_q = self.recon_for_s2_s3(y_q, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        scales_hat = self.recon_for_s2_s3(scales_hat, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 4-step scale-2 (s2) (2× downsample) coding
        mask_list_s1 = self.get_mask_for_s1(B, C, H, W, dtype, device)
        scales_s2 = self.get_s2_hyper_with_mask(scales_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        means_s2 = self.get_s2_hyper_with_mask(means_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context += common_params_s2
        context_next = context_net[0](context)
        
        mask_list = self.get_mask_four_parts(B, C, H // 2, W // 2, dtype, device)[1:]
        y_res_list_s2 = [y_res]
        y_q_list_s2   = [y_q]
        y_hat_list_s2 = [y_hat]
        scale_list_s2 = [scales_hat]

        for i in range(3):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list_s2), dim=0)
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params), adaptive_params_list[i + 1])
            context_next = self.attn_s2(context, context_next)
            scales, means = context.chunk(2, 1)
            y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y_s2, scales, means, mask_list[i])
            y_res_list_s2.append(y_res_1)
            y_q_list_s2.append(y_q_1)
            y_hat_list_s2.append(y_hat_1)
            scale_list_s2.append(s_hat_1)
        
        y_res = torch.sum(torch.stack(y_res_list_s2), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s2), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s2), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s2), dim=0)

        if write:
            y_q_write_list_s2 = [self.combine_for_writing_s2(y_q_list_s2[i]) for i in range(1, len(y_q_list_s2))]
            scales_hat_write_list_s2 = [self.combine_for_writing_s2(scale_list_s2[i]) for i in range(1, len(scale_list_s2))]
       
        y_res = self.recon_for_s2_s3(y_res, mask_list_s2, B, C, H, W, dtype, device)
        y_q = self.recon_for_s2_s3(y_q, mask_list_s2, B, C, H, W, dtype, device)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_s2, B, C, H, W, dtype, device)
        scales_hat = self.recon_for_s2_s3(scales_hat, mask_list_s2, B, C, H, W, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_s2, B, C, H, W, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_s2, B, C, H, W, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 8-step scale-3 (s3) coding - WITH RWKV!
        scales_s3 = self.get_s3_hyper_with_mask(scales_all, mask_list_s2, B, C, H, W, dtype, device)
        means_s3 = self.get_s3_hyper_with_mask(means_all, mask_list_s2, B, C, H, W, dtype, device)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context += common_params_s3
        context_next = context_net[1](context)

        mask_list = self.get_mask_eight_parts(B, C, H, W, dtype, device)[2:]
        y_res_list_s3 = [y_res]
        y_q_list_s3   = [y_q]
        y_hat_list_s3 = [y_hat]
        scale_list_s3 = [scales_hat]

        for i in range(6):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list_s3), dim=0)
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params), adaptive_params_list[i + 4])
            # Use RWKV instead of CrossAttention for s3
            context_next = self.attn_s3(context, context_next)
            scales, means = context.chunk(2, 1)
            y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_list[i])
            y_res_list_s3.append(y_res_1)
            y_q_list_s3.append(y_q_1)
            y_hat_list_s3.append(y_hat_1)
            scale_list_s3.append(s_hat_1)

        y_res = torch.sum(torch.stack(y_res_list_s3), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s3), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s3), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s3), dim=0)

        if write:
            y_q_write_list_s3 = [self.combine_for_writing_s3(y_q_list_s3[i]) for i in range(1, len(y_q_list_s3))]
            scales_hat_write_list_s3 = [self.combine_for_writing_s3(scale_list_s3[i]) for i in range(1, len(scale_list_s3))]

            return y_q_write_list_s1 + y_q_write_list_s2 + y_q_write_list_s3, scales_hat_write_list_s1 + scales_hat_write_list_s2 + scales_hat_write_list_s3

        return y_res, y_q, y_hat, scales_hat
    
    # Note: compress() and decompress() methods would also need updating
    # but are omitted here for brevity. They follow the same pattern as forward_hpcm.
