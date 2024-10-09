import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import sys
import copy

def weighted_guided_filter(I, p, r, eps, weight):
    """加权引导滤波器实现
    I: 引导图像
    p: 输入图像
    r: 滤波半径
    eps: 正则化参数
    weight: 权重图像，用于动态调整正则化参数
    """
    # 计算I和p的局部均值
    I_mean = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    p_mean = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    # 计算I * p 和 I * I 的均值
    I_p_mean = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    I_2_mean = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    # 计算协方差和方差
    cov_Ip = I_p_mean - I_mean * p_mean
    var_I = I_2_mean - I_mean * I_mean
    # 根据权重调整正则化参数
    # 动态调整的正则化项 eps' = weight * eps
    a = cov_Ip / (var_I + weight * eps)
    b = p_mean - a * I_mean
    # 对系数 a 和 b 进行加权平均
    a_mean = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    b_mean = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    # 计算输出图像
    q = a_mean * I + b_mean
    return q

def fuse_edge_depth(depth_guide,depth_mono, depth_bad):    
    # 计算深度图的梯度
    grad_x = cv2.Sobel(depth_guide, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_guide, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    # 设置阈值提取边缘
    _, mask = cv2.threshold(edges, np.percentile(edges, 90), 1, cv2.THRESH_BINARY)
    diff_guide = np.abs(depth_guide-depth_bad)
    mask[depth_bad>0.15] = 0
    mask[diff_guide>0.1]=0
    #mask[diff_guide<0.03]=0
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    # 替换深度值
    depth_fused = np.where(mask == 1, depth_mono, depth_bad)
    return depth_fused,mask
    


root_dir = sys.argv[1]
step = 1
fitler_flag = 1
data_folder1 = os.path.join(root_dir,'depths')
data_folder2 = os.path.join(root_dir,'mono_depths')
# 读取低清和高清深度图
num_img = len(os.listdir(data_folder2))
for i in range(0,num_img,step):
    i = f'{i}.npy'
    #print(i)
    reference = np.load(os.path.join(data_folder2, i))
    if fitler_flag>0:
        reference = reference.astype(np.float32)
        reference =cv2.medianBlur(reference, 3)
        reference =cv2.medianBlur(reference, 3)
    #cv2.medianBlur(noisy_image, 3)

    # The 1D image whose values we would like to filter
    target = np.load(os.path.join(data_folder1, i))
    #target  =cv2.resize(target,(1536,768),interpolation=cv2.INTER_CUBIC)
    mask = target == 0
    # scale = target[mask].mean() / reference[mask].mean()
    # reference = reference * scale

    low_res_depth = copy.deepcopy(target)
    high_res_depth = reference

    # high_res_depth[low_res_depth>1]=1
    # low_res_depth[low_res_depth>1]=1

    # 将图像转换为浮点数以便进行滤波
    

    # 生成权重图像（例如：基于图像梯度）
    grad_x = cv2.Sobel(high_res_depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(high_res_depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 权重图像可以根据梯度幅值生成，来动态调整平滑效果
    weight = 1 / (1 + 50*gradient_magnitude)
    

    low_res_depth = low_res_depth.astype(np.float64)
    high_res_depth = high_res_depth.astype(np.float64)

    # 引导滤波的参数
    radius = 25  # 滤波半径
    eps = 1e-8  # 正则化参数，避免除以零

    # 使用高清图引导低清深度图进行滤波，增强细节(guide, input_img, r, eps, weight)
    enhanced_depth = weighted_guided_filter(high_res_depth, low_res_depth, radius, eps,weight)
    enhanced_depth = enhanced_depth.astype(np.float32)
    if fitler_flag>0:
        enhanced_depth =cv2.medianBlur(enhanced_depth, 3)
    #enhanced_depth2 = weighted_guided_filter(high_res_depth, low_res_depth, radius, eps,np.ones_like(weight))
    enhanced_depth,_ = fuse_edge_depth(enhanced_depth,high_res_depth,low_res_depth)
    enhanced_depth = weighted_guided_filter(high_res_depth, enhanced_depth, radius, 1e-8,weight)
    if fitler_flag>0:
        enhanced_depth = enhanced_depth.astype(np.float32)
        enhanced_depth =cv2.medianBlur(enhanced_depth, 3)
    enhanced_depth[mask] = 0
    np.save(os.path.join(root_dir,'depths',i),enhanced_depth)
    # plt.figure(1)
    # plt.imshow(enhanced_depth)
    # plt.figure(2)
    # plt.imshow(enhanced_depth2)
    # plt.show()