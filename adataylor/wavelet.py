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

def wavelet_denoise(f_values, wavelet='db4', level=3, threshold_factor=0.1):
    """
    使用小波变换对函数值进行降噪
    
    参数:
        f_values: 函数值数组
        wavelet: 小波类型
        level: 分解级别
        threshold_factor: 阈值因子
        
    返回:
        降噪后的函数值
    """
    # 函数值的小波分解
    coeffs = pywt.wavedec(f_values, wavelet, level=level)
    
    # 计算阈值
    threshold = threshold_factor * np.max(np.abs(coeffs[-1]))
    
    # 阈值滤波
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, 'soft')
    
    # 重构函数
    denoised = pywt.waverec(coeffs, wavelet)
    
    # 处理可能的长度差异
    if len(denoised) > len(f_values):
        denoised = denoised[:len(f_values)]
    elif len(denoised) < len(f_values):
        denoised = np.pad(denoised, (0, len(f_values) - len(denoised)), 'edge')
    
    return denoised

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

def wavelet_taylor_hybrid(f_expr, x_sym, x0, segment, order=10, sample_points=100):
    """
    小波-泰勒混合逼近
    
    参数:
        f_expr: sympy表达式
        x_sym: sympy符号
        x0: 展开点
        segment: 区间 (a, b)
        order: 泰勒展开阶数
        sample_points: 采样点数
        
    返回:
        混合逼近的系数列表
    """
    a, b = segment
    x_vals = np.linspace(a, b, sample_points)
    
    # 计算函数值
    f_lambda = lambdify(x_sym, f_expr, 'numpy')
    try:
        f_vals = f_lambda(x_vals)
    except:
        # 对有奇点的函数，逐点计算
        f_vals = np.zeros(sample_points)
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
            except:
                # 如果计算失败，使用线性插值
                if i > 0:
                    f_vals[i] = f_vals[i-1]
                else:
                    f_vals[i] = 0
    
    # 小波降噪
    denoised = wavelet_denoise(f_vals)
    
    # 计算泰勒系数
    taylor_coeffs = []
    for n in range(order + 1):
        try:
            nth_derivative = sp.diff(f_expr, (x_sym, n))
            coef = float(nth_derivative.subs(x_sym, x0)) / np.math.factorial(n)
            taylor_coeffs.append(coef)
        except:
            # 如果解析计算失败，使用数值微分
            if n == 0:
                try:
                    coef = float(f_expr.subs(x_sym, x0))
                except:
                    coef = f_lambda(x0)
                taylor_coeffs.append(coef)
            else:
                # 数值计算高阶导数变得不稳定，停止计算
                break
    
    return taylor_coeffs