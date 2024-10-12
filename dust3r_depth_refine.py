import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2
import copy
from sklearn.linear_model import LinearRegression
from BoostingMonocularDepth.prepare_depth import prepare_gt_depth 

## fitting the scale of depth_B to depth_A
def iterative_linear_fitting_fg(depth_A, depth_B, threshold_ratio=0.1, iterations = 15,depth_focus=0.6):
    """
    迭代线性拟合两个深度图，保留误差小的部分，并迭代拟合误差大的区域。
    :param depth_A: 深度图 A (归一化)
    :param depth_B: 深度图 B
    :param threshold_ratio: 误差大的区域占比阈值，小于该阈值时停止迭代
    :return: 线性拟合后的深度图 B 和误差
    """
    # 初始设置
    depth_A_flat = depth_A.flatten()
    depth_B_flat = depth_B.flatten()
    
    # 保留所有索引（从0到depth_A_flat的索引）
    indices = np.arange(depth_A_flat.shape[0])
    indices = np.where(depth_B_flat<depth_focus)[0]
    res = np.zeros_like(depth_A_flat)
    last = error_threshold = 0
    for i in range(iterations):
        # 线性拟合：用深度图 A 作为自变量，深度图 B 作为因变量
        model = LinearRegression().fit(depth_A_flat[indices].reshape(-1, 1), depth_B_flat[indices])
        predicted_B = model.predict(depth_A_flat.reshape(-1, 1))
        
        # 计算误差：真实值和拟合值的差异
        error = np.abs(predicted_B - depth_B_flat)
        
        # 计算误差的百分位数
        error_threshold = np.percentile(error[indices], 98)  # 误差前90%的数据
        error[~indices] = error_threshold
        #这样就无法选中那些已经被排除的点了
        good_fit_mask =(error < error_threshold)  # 选择误差小于阈值的部分
        bad_fit_mask = error > error_threshold
        indices = good_fit_mask & (depth_B_flat<depth_focus)

        # 如果误差大的区域占比小于阈值，停止迭代
        if np.sum(~good_fit_mask) / len(good_fit_mask) < threshold_ratio:
            break
        
        # 更新误差大的区域索引，继续拟合剩余的误差大的区域
        #indices = np.where(~good_fit_mask)[0]
        
    error = np.abs(predicted_B - depth_B_flat)
    return predicted_B.reshape(depth_A.shape), error.reshape(depth_A.shape)

def iterative_linear_fitting_bg(depth_A, depth_B, threshold_ratio=0.1, iterations = 15,depth_focus=0.6):
    """
    迭代线性拟合两个深度图，保留误差小的部分，并迭代拟合误差大的区域。
    
    :param depth_A: 深度图 A (归一化)
    :param depth_B: 深度图 B
    :param threshold_ratio: 误差大的区域占比阈值，小于该阈值时停止迭代
    :return: 线性拟合后的深度图 B 和误差
    """
    # 初始设置
    # depth_B[depth_B>1]=1
    # depth_A[depth_B>1]=1
    depth_A_flat = depth_A.flatten()
    depth_B_flat = depth_B.flatten()
    
    # 保留所有索引（从0到depth_A_flat的索引）
    indices = np.arange(depth_A_flat.shape[0])
    indices = np.where(depth_B_flat>depth_focus)[0]
    res = np.zeros_like(depth_A_flat)
    last = error_threshold = 0
    #print(depth_A_flat[indices].shape)
    for i in range(iterations):
        # 线性拟合：用深度图 A 作为自变量，深度图 B 作为因变量
        if depth_A_flat[indices].shape[0] == 0:
            break 
        model = LinearRegression().fit(depth_A_flat[indices].reshape(-1, 1), depth_B_flat[indices])
        predicted_B = model.predict(depth_A_flat.reshape(-1, 1))
        
        # 计算误差：真实值和拟合值的差异
        error = np.abs(predicted_B - depth_B_flat)
        
        # 计算误差的百分位数
        error_threshold = np.percentile(error[indices], 98)  # 误差前90%的数据
        error[~indices] = error_threshold
        #这样就无法选中那些已经被排除的点了
        good_fit_mask =(error < error_threshold)  # 选择误差小于阈值的部分
        bad_fit_mask = error > error_threshold
        indices = good_fit_mask & (depth_B_flat>depth_focus)

        # 如果误差大的区域占比小于阈值，停止迭代
        if np.sum(~good_fit_mask) / len(good_fit_mask) < threshold_ratio:
            break
        
        # 更新误差大的区域索引，继续拟合剩余的误差大的区域
        #indices = np.where(~good_fit_mask)[0]
        
    error = np.abs(predicted_B - depth_B_flat)
    return predicted_B.reshape(depth_A.shape), error.reshape(depth_A.shape)

def iterative_linear_fitting_fg_bg(depth_A, depth_B, threshold_ratio=0.1, iterations = 15,depth_focus=0.6):
    fitted_B1, error_map1 = iterative_linear_fitting_fg(depth_A, depth_B)
    fitted_B2, error_map2 = iterative_linear_fitting_bg(depth_A, depth_B)

    fitted_B = np.zeros_like(depth_B)
    fitted_B[depth_B>0.6] = fitted_B2[depth_B>0.6]
    fitted_B[depth_B<0.6] = fitted_B1[depth_B<0.6]

    error_map = np.zeros_like(error_map1)
    error_map[depth_B>0.6] = error_map2[depth_B>0.6]
    error_map[depth_B<0.6] = error_map1[depth_B<0.6]
    return fitted_B,error_map

def fit_depth_scale(input, reference):
    # depth_A = np.load(depth_A)
    # depth_B = np.load(depth_B)
    depth_B = input
    depth_A = reference
    if depth_A.shape[:2] != depth_B.shape[:2]:
        depth_B  =cv2.resize(depth_B,(depth_A.shape[1],depth_A.shape[0]),interpolation=cv2.INTER_CUBIC)
    fitted_B, error_map = iterative_linear_fitting_bg(depth_A, depth_B,depth_focus=0)
    return fitted_B

## refine the depth map to better fit the edges
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

def depth_filter_refine(input, reference, fitler_flag = 1, radius = 25, ksize = 3):
    if fitler_flag>0:
        reference = reference.astype(np.float32)
        reference =cv2.medianBlur(reference, ksize)
        reference =cv2.medianBlur(reference, ksize)
    # The 1D image whose values we would like to filter
    target = input
    #target  =cv2.resize(target,(1536,768),interpolation=cv2.INTER_CUBIC)
    mask = target == 0
    # scale = target[mask].mean() / reference[mask].mean()
    # reference = reference * scale
    low_res_depth = copy.deepcopy(target)
    high_res_depth = reference
    # 生成权重图像（例如：基于图像梯度）
    grad_x = cv2.Sobel(high_res_depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(high_res_depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=ksize)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # 权重图像可以根据梯度幅值生成，来动态调整平滑效果
    weight = 1 / (1 + 50*gradient_magnitude)
    low_res_depth = low_res_depth.astype(np.float64)
    high_res_depth = high_res_depth.astype(np.float64)
    # 引导滤波的参数
    # radius = 25  # 滤波半径
    eps = 1e-8  # 正则化参数，避免除以零
    # 使用高清图引导低清深度图进行滤波，增强细节(guide, input_img, r, eps, weight)
    enhanced_depth = weighted_guided_filter(high_res_depth, low_res_depth, radius, eps,weight)
    enhanced_depth = enhanced_depth.astype(np.float32)
    if fitler_flag>0:
        enhanced_depth =cv2.medianBlur(enhanced_depth, ksize)
    #enhanced_depth2 = weighted_guided_filter(high_res_depth, low_res_depth, radius, eps,np.ones_like(weight))
    enhanced_depth,_ = fuse_edge_depth(enhanced_depth,high_res_depth,low_res_depth)
    enhanced_depth = weighted_guided_filter(high_res_depth, enhanced_depth, radius, 1e-8,weight)
    if fitler_flag>0:
        enhanced_depth = enhanced_depth.astype(np.float32)
        enhanced_depth = cv2.medianBlur(enhanced_depth, ksize)
    enhanced_depth[mask] = 0
    return enhanced_depth

def get_npy_files(directory):
    # List all files in the directory
    files = os.listdir(directory)
    # Filter the list to include only .npy files
    npy_files = [file for file in files if file.endswith('.npy')]
    return npy_files

def normalized_depth8U(depth_image):
    depth_image8U = depth_image / np.max(depth_image) * 255
    return depth_image8U.astype(np.uint8)

## refine the dust3r depth map using boosting monocular depth
def refine_dust3r_depth_maps(datadir):
    # generate monocular depth map using boosting monocular depth algorithm
    input_image_folder = os.path.join(datadir, 'colmap', 'images') 
    mono_depth_orig_folder = os.path.join(datadir, 'colmap', 'mono_depths_orig') 
    if not os.path.isdir(mono_depth_orig_folder):
        prepare_gt_depth(input_folder = input_image_folder, save_folder = mono_depth_orig_folder)
    # fitting the scale
    dust3r_depth_folder = os.path.join(datadir, 'colmap', 'depths')
    basename_lists = get_npy_files(dust3r_depth_folder)
    mono_depth_folder = os.path.join(datadir, 'colmap', 'mono_depths') 
    os.makedirs(mono_depth_folder, exist_ok=True)
    for basename in basename_lists:
        # fit depth scale
        depth_dust3r = np.load(os.path.join(dust3r_depth_folder, basename))
        depth_mono = np.load(os.path.join(mono_depth_orig_folder, basename))
        depth_fitscale = fit_depth_scale(depth_dust3r, depth_mono)
        cv2.imwrite(os.path.join(mono_depth_orig_folder, f'{basename}.png'), normalized_depth8U(depth_mono))
        cv2.imwrite(os.path.join(mono_depth_orig_folder, f'{basename}_fitscale.png'), normalized_depth8U(depth_fitscale))
        # guided filter
        depth_refine = depth_filter_refine(depth_dust3r, depth_mono, fitler_flag = 1, radius = 25, ksize = 3)
        # depth_refine = depth_filter_refine(depth_dust3r, depth_fitscale, fitler_flag = 1)
        np.save(os.path.join(mono_depth_folder, basename), depth_refine)
        # save visualization results
        cv2.imwrite(os.path.join(mono_depth_folder, f'{basename}.png'), normalized_depth8U(depth_refine))
    
if __name__ == "__main__":
    # datadir = '/home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_4/020_recon'
    # refine_dust3r_depth_maps(datadir)
    datadir = sys.argv[1]
    refine_dust3r_depth_maps(datadir)

