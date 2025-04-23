"""
小波处理模块 - 使用小波变换进行信号分析与预处理
"""

import numpy as np
import pywt
import sympy as sp
from sympy import lambdify

def wavelet_preprocess(f_expr, x_sym, break_points, wavelet='db4', level=3):
    """
    使用小波变换预处理函数，特别是处理非光滑区域
    
    参数:
        f_expr: sympy表达式
        x_sym: sympy符号
        break_points: 断点列表
        wavelet: 小波类型
        level: 分解级别
        
    返回:
        分段点列表，每段为(起点, 终点)
    """
    segments = []
    for i in range(len(break_points) - 1):
        a, b = break_points[i], break_points[i+1]
        segments.append((a, b))
    
    return segments

def wavelet_denoise(x_values, y_values, wavelet='db4', level=3, threshold=None):
    """
    使用小波变换对函数数据进行去噪
    
    参数:
        x_values: x坐标数组
        y_values: y坐标数组
        wavelet: 小波类型
        level: 分解级别
        threshold: 阈值，默认为None（自动选择）
        
    返回:
        去噪后的y值
    """
    # 确保数据长度满足小波变换的要求
    n = len(y_values)
    
    # 找到最接近的2的幂
    p = int(np.log2(n))
    if 2**p < n:
        p += 1
    
    # 重采样到2的幂长度
    if n != 2**p:
        x_new = np.linspace(min(x_values), max(x_values), 2**p)
        y_new = np.interp(x_new, x_values, y_values)
    else:
        x_new = x_values.copy()
        y_new = y_values.copy()
    
    # 进行小波分解
    coeffs = pywt.wavedec(y_new, wavelet, level=level)
    
    # 应用阈值
    if threshold is None:
        # 使用VisuShrink方法自动选择阈值
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(y_new)))
    
    # 对细节系数应用软阈值
    coeffs_thresholded = list(coeffs)
    for i in range(1, len(coeffs)):
        coeffs_thresholded[i] = pywt.threshold(coeffs[i], threshold, 'soft')
    
    # 重构信号
    y_denoised = pywt.waverec(coeffs_thresholded, wavelet)
    
    # 截断到原始长度
    y_denoised = y_denoised[:n]
    
    # 插值回原始x值
    if n != 2**p:
        y_denoised = np.interp(x_values, x_new[:n], y_denoised)
    
    return y_denoised

def detect_singularities(f_expr, x_sym, domain, num_points=1000):
    """
    使用小波变换检测函数的奇异点
    
    参数:
        f_expr: sympy表达式
        x_sym: sympy符号
        domain: 定义域 (a, b)
        num_points: 采样点数
        
    返回:
        检测到的奇异点列表
    """
    a, b = domain
    x_vals = np.linspace(a, b, num_points)
    
    # 计算函数值
    f_lambda = lambdify(x_sym, f_expr, 'numpy')
    try:
        f_vals = f_lambda(x_vals)
    except:
        # 如果计算失败，可能是函数有奇点，采用更安全的方法
        f_vals = np.zeros(num_points)
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
            except:
                f_vals[i] = np.nan
        
        # 填充NaN值
        mask = np.isnan(f_vals)
        if np.any(mask):
            # 标记NaN位置为可能的奇点
            singular_indices = np.where(mask)[0]
            singularities = [x_vals[i] for i in singular_indices]
            
            # 补充连续区间的中点
            for i in range(len(singular_indices)-1):
                if singular_indices[i+1] - singular_indices[i] > 1:
                    midpoint = (x_vals[singular_indices[i]] + x_vals[singular_indices[i+1]]) / 2
                    singularities.append(midpoint)
                    
            return sorted(singularities)
    
    # 使用小波变换检测奇异点
    coeffs = pywt.wavedec(f_vals, 'haar', level=5)
    details = coeffs[1]  # 第一级细节系数
    
    # 寻找细节系数的局部极大值
    threshold = np.percentile(np.abs(details), 99)
    peaks = np.where(np.abs(details) > threshold)[0]
    
    # 转换索引到原始x坐标
    scale_factor = len(x_vals) / len(details)
    singular_indices = [int(p * scale_factor) for p in peaks]
    singularities = [x_vals[i] for i in singular_indices if 0 <= i < len(x_vals)]
    
    # 聚合接近的奇点
    if singularities:
        clustered = [singularities[0]]
        min_separation = (b - a) / 50  # 最小分离距离
        
        for point in singularities[1:]:
            if point - clustered[-1] > min_separation:
                clustered.append(point)
        
        return clustered
    
    return []

def wavelet_taylor_hybrid(f_expr, x_sym, breakpoints, epsilon=1e-8, max_order=10):
    """
    小波-泰勒混合逼近
    
    先使用小波变换处理函数，然后应用泰勒展开
    
    参数:
        f_expr: sympy表达式
        x_sym: 符号变量
        breakpoints: 分段点列表
        epsilon: 误差容限
        max_order: 最大阶数
        
    返回:
        分段泰勒展开表达式
    """
    from .adaptive import auto_order
    
    # 确保断点是有序的
    breakpoints = sorted(list(set(breakpoints)))
    
    # 创建分段列表
    pieces = []
    
    # 对每个分段处理
    for i in range(len(breakpoints) - 1):
        a, b = breakpoints[i], breakpoints[i+1]
        # 分段中点
        x_mid = (a + b) / 2
        
        # 生成数据点
        n_points = 100  # 采样点数
        x_vals = np.linspace(a, b, n_points)
        
        # 计算函数值
        f_func = sp.lambdify(x_sym, f_expr, 'numpy')
        try:
            y_vals = f_func(x_vals)
        except:
            # 逐点计算
            y_vals = np.zeros_like(x_vals)
            for j, x in enumerate(x_vals):
                try:
                    y_vals[j] = float(f_expr.subs(x_sym, x))
                except:
                    y_vals[j] = np.nan
        
        # 检查数据是否有异常值
        has_nan = np.any(np.isnan(y_vals))
        has_inf = np.any(np.isinf(y_vals))
        
        if has_nan or has_inf:
            # 如果有异常值，跳过小波处理，直接使用泰勒展开
            pass
        else:
            # 应用小波去噪
            y_denoised = wavelet_denoise(x_vals, y_vals)
            
            # 创建去噪后的函数逼近
            from scipy.interpolate import interp1d
            f_denoised = interp1d(x_vals, y_denoised, kind='cubic', 
                                  bounds_error=False, fill_value="extrapolate")
            
            # 创建新的sympy表达式（通过采样点）
            sample_points = 20  # 用于创建表达式的点数
            x_sample = np.linspace(a, b, sample_points)
            y_sample = f_denoised(x_sample)
            
            # 使用多项式拟合
            poly_coeffs = np.polyfit(x_sample - x_mid, y_sample, min(5, sample_points - 1))
            
            # 创建sympy表达式
            f_approx = 0
            for j, coef in enumerate(poly_coeffs[::-1]):
                f_approx += coef * (x_sym - x_mid)**j
            
            # 选择泰勒展开阶数
            order = auto_order(f_approx, x_sym, x_mid, epsilon, max_order, (a, b))
            
            # 计算泰勒展开
            expansion = 0
            for n in range(order + 1):
                try:
                    # 计算n阶导数在x_mid点的值
                    if n == 0:
                        dnf = f_approx.subs(x_sym, x_mid)
                    else:
                        dnf = sp.diff(f_approx, x_sym, n).subs(x_sym, x_mid)
                    
                    # 添加到展开式
                    from .utils import factorial
                    term = dnf * (x_sym - x_mid)**n / factorial(n)
                    expansion += term
                except:
                    # 如果计算失败，使用当前展开式
                    break
        
        # 创建条件表达式
        condition = sp.And(x_sym >= a, x_sym < b)
        # 如果是最后一段，包括右端点
        if i == len(breakpoints) - 2:
            condition = sp.And(x_sym >= a, x_sym <= b)
        
        pieces.append((expansion, condition))
    
    # 创建分段函数
    return sp.Piecewise(*pieces)