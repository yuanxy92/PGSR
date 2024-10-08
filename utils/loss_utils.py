#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import math

def compute_local_normal_consistency(normal, window_size=3):
    """
    计算局部区域的法线一致性，基于余弦相似性，窗口内法线变化越小表示区域越平。
    normal: 3*H*W 的张量，表示每个像素的法线向量 (C=3,H,W)。
    window_size: 局部窗口大小。
    返回每个像素的局部法线一致性图 (H, W)。
    """
    # 将 normal 的形状从 (3, H, W) 转换为 (1, 3, H, W) 以便使用 unfold
    normal = normal.unsqueeze(0)  # (1, 3, H, W)

    # 创建一个展开操作的卷积核，用于计算局部窗口内的法线一致性
    pad = window_size // 2

    # 将 normal 展开为局部窗口
    unfolded = F.unfold(normal, kernel_size=window_size, padding=pad)  # (1, 3*window_size*window_size, H*W)
    
    # 转换形状为 (H*W, 3, window_size*window_size)
    unfolded = unfolded.view(3, window_size * window_size, -1).permute(2, 0, 1)  # (H*W, 3, window_size*window_size)
    
    # 计算每个窗口的平均法线向量
    mean_normal = unfolded.mean(dim=-1)  # (H*W, 3)
    
    # 将展开的法线重新转为与 mean_normal 对应的形状 (H*W, 3, window_size*window_size)
    unfolded = unfolded.permute(0, 2, 1)  # (H*W, window_size*window_size, 3)

    # 计算局部窗口法线向量与平均法线的余弦相似度
    cosine_similarity = F.cosine_similarity(unfolded, mean_normal.unsqueeze(1), dim=-1)  # (H*W, window_size*window_size)

    # 计算相似度的均值，值越接近 1 表示法线越相似
    consistency_map = cosine_similarity.mean(dim=-1)  # (H*W)

    # 将 consistency_map 重新 reshape 回 (H, W)
    consistency_map = consistency_map.view(normal.shape[2], normal.shape[3])  # (H, W)

    return consistency_map

def compute_local_normal_consistency_weight(normal, window_size=3):
    """
    计算局部区域的法线一致性权重，用作损失函数的权重，平坦区域的权重更大。
    normal: 3*H*W 的张量，表示每个像素的法线向量 (C=3,H,W)。
    window_size: 局部窗口大小。
    alpha: 控制权重差异的系数，越大时平坦区域权重越大。
    返回每个像素的局部法线一致性权重图 (H, W)。
    """
    # 计算法线余弦相似性
    consistency_map = compute_local_normal_consistency(normal, window_size)

    # 将 cos 范围从 [-1, 1] 变换到 [0, 1]
    weights = (1 + consistency_map) / 2  # 范围 [0, 1]

    # 对权重进行指数放大，alpha 控制权重的差异
    #weights = weights.pow(alpha)

    return weights

def l1_weight_loss(network_output, gt, weight):
    error = torch.abs((network_output - gt)) * weight
    return error.mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim2(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(0)

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img

def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask

def pearson_depth_loss(depth_src, depth_target):
    #co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))
    depth_src = depth_src[depth_target>0]
    depth_target = depth_target[depth_target>0]
    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()


    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co


def local_pearson_loss(depth_src, depth_target, box_p, p_corr):
        # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
        num_box_h = math.floor(depth_src.shape[0]/box_p)
        num_box_w = math.floor(depth_src.shape[1]/box_p)
        max_h = depth_src.shape[0] - box_p
        max_w = depth_src.shape[1] - box_p
        _loss = torch.tensor(0.0,device='cuda')
        n_corr = int(p_corr * num_box_h * num_box_w)
        x_0 = torch.randint(0, max_h, size=(n_corr,), device = 'cuda')
        y_0 = torch.randint(0, max_w, size=(n_corr,), device = 'cuda')
        x_1 = x_0 + box_p
        y_1 = y_0 + box_p
        _loss = torch.tensor(0.0,device='cuda')
        valid_num = 0
        for i in range(len(x_0)):
            if depth_target[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1).mean() == 0 or torch.count_nonzero(depth_target[x_0[i]:x_1[i],y_0[i]:y_1[i]])<(box_p*box_p)*0.6:
                continue
            _loss += pearson_depth_loss(depth_src[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1), depth_target[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1))
            valid_num += 1
        return _loss/valid_num