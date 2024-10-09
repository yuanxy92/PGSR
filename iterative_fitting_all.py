import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
import os
import cv2

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

depth_dir = sys.argv[1]
mono_dir = sys.argv[2]

# 示例使用

for i in os.listdir(mono_dir):
    if '.npy' not in i:
        continue
    #A是高清
    #B是低清
    depth_A = os.path.join(mono_dir,i)  # 模拟深度图 A，范围 [0, 1]
    depth_B = os.path.join(depth_dir,i) # 模拟深度图 B，任意深度范围

    depth_A = np.load(depth_A)
    depth_B = np.load(depth_B)

    if depth_A.shape[:2] != depth_B.shape[:2]:
        depth_B  =cv2.resize(depth_B,(depth_A.shape[1],depth_A.shape[0]),interpolation=cv2.INTER_CUBIC)

    #fitted_B, error_map = iterative_linear_fitting_fg_bg(depth_A,depth_B,iterations = 25)
    fitted_B, error_map = iterative_linear_fitting_bg(depth_A, depth_B,depth_focus=0)
    # 调用迭代线性拟合函数
    # fitted_B1, error_map1 = iterative_linear_fitting_fg(depth_A, depth_B)
    # fitted_B2, error_map2 = iterative_linear_fitting_bg(depth_A, depth_B)

    # fitted_B = np.zeros_like(depth_B)
    # fitted_B[depth_B>0.6] = fitted_B2[depth_B>0.6]
    # fitted_B[depth_B<0.6] = fitted_B1[depth_B<0.6]

    # error_map = np.zeros_like(error_map1)
    # error_map[depth_B>0.6] = error_map2[depth_B>0.6]
    # error_map[depth_B<0.6] = error_map1[depth_B<0.6]


    np.save(os.path.join(mono_dir,i),fitted_B)
