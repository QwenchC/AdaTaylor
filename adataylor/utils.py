"""
辅助函数模块 - 提供泰勒展开逼近所需的通用功能
"""

import numpy as np
import sympy as sp
from itertools import combinations_with_replacement
import scipy.signal as signal

def factorial(n):
    """
    计算阶乘
    """
    if n < 0:
        raise ValueError("阶乘不能用于负数")
    if n == 0 or n == 1:
        return 1
    else:
        return np.prod(np.arange(1, n+1))

def detect_singularities(f_expr, x_sym, domain, num_points=100):
    """
    检测函数在给定区间上的奇异点
    
    参数:
        f_expr: sympy表达式
        x_sym: sympy符号
        domain: 区间 (min, max)
        num_points: 采样点数
        
    返回:
        可能的奇异点列表
    """
    x_vals = np.linspace(domain[0], domain[1], num_points)
    f_lambda = sp.lambdify(x_sym, f_expr, 'numpy')
    
    # 计算函数值
    f_vals = np.zeros(num_points)
    for i, x in enumerate(x_vals):
        try:
            f_vals[i] = float(f_expr.subs(x_sym, x))
        except:
            f_vals[i] = np.nan
    
    # 检测无穷大或NaN值
    singularity_indices = np.where(~np.isfinite(f_vals))[0]
    singularities = []
    
    # 对每个奇异点，尝试精确定位
    for idx in singularity_indices:
        x_approx = x_vals[idx]
        
        # 如果是第一个或最后一个点，直接添加
        if idx == 0 or idx == num_points - 1:
            singularities.append(x_approx)
            continue
        
        # 否则，用二分法精确定位
        x_left = x_vals[idx-1]
        x_right = x_vals[idx+1]
        
        # 简单的二分法
        for _ in range(5):  # 5次迭代应该足够
            x_mid = (x_left + x_right) / 2
            try:
                val = float(f_expr.subs(x_sym, x_mid))
                if np.isfinite(val):
                    x_left = x_mid
                else:
                    x_right = x_mid
            except:
                x_right = x_mid
        
        singularities.append((x_left + x_right) / 2)
    
    return singularities

def multiindex_generator(dim, total):
    """
    生成多指标 (用于多变量泰勒展开)
    
    参数:
        dim: 维度
        total: 总阶数
        
    返回:
        生成器，产生所有和为total的dim维多重指标
    """
    return combinations_with_replacement(range(dim), total)

def smooth_data(data, window_length=11, polyorder=3):
    """
    使用Savitzky-Golay滤波平滑数据
    
    参数:
        data: 数据数组
        window_length: 窗口长度 (必须为奇数)
        polyorder: 多项式阶数
        
    返回:
        平滑后的数据
    """
    if window_length > len(data):
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
    
    if window_length < polyorder + 2:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1
    
    return signal.savgol_filter(data, window_length, polyorder)

def function_type_analysis(f_expr, x_sym, domain, num_points=100):
    """
    分析函数类型 (光滑、分段、振荡等)
    
    参数:
        f_expr: sympy表达式
        x_sym: sympy符号
        domain: 区间 (min, max)
        num_points: 采样点数
        
    返回:
        函数类型字符串和相关信息的字典
    """
    # 检测奇异点
    singularities = detect_singularities(f_expr, x_sym, domain, num_points)
    
    if len(singularities) > 0:
        return {
            "type": "piecewise", 
            "breakpoints": singularities
        }
    
    # 检测振荡性
    x_vals = np.linspace(domain[0], domain[1], num_points)
    f_vals = np.zeros(num_points)
    
    try:
        f_lambda = sp.lambdify(x_sym, f_expr, 'numpy')
        f_vals = f_lambda(x_vals)
    except:
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
            except:
                f_vals[i] = np.nan
    
    # 计算导数变号次数
    diff = np.diff(f_vals)
    sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
    
    # 高频振荡的标准: 变号次数超过点数的一定比例
    if sign_changes > num_points * 0.2:  # 20%的点有变号
        return {
            "type": "oscillatory",
            "sign_changes": sign_changes
        }
    
    # 默认为光滑函数
    return {"type": "smooth"}

def adaptive_segmentation(f_expr, x_sym, domain, epsilon=1e-8, max_segments=10):
    """
    基于函数特性的自适应区间划分算法
    
    参数:
        f_expr: sympy表达式
        x_sym: 符号变量
        domain: 初始区间
        epsilon: 误差容限
        max_segments: 最大分段数
        
    返回:
        最优分段点列表
    """
    from scipy.signal import find_peaks
    
    # 初始化分段点列表，包含区间端点
    breakpoints = [domain[0], domain[1]]
    
    # 计算高密度样本点上的函数值
    num_points = 1000
    x_vals = np.linspace(domain[0], domain[1], num_points)
    f_vals = np.zeros(num_points)
    
    # 计算函数值
    f_func = sp.lambdify(x_sym, f_expr, 'numpy')
    try:
        f_vals = f_func(x_vals)
    except:
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
            except:
                f_vals[i] = np.nan
    
    # 1. 检测不连续点和奇异点
    discontinuities = []
    diff_vals = np.diff(f_vals)
    # 寻找显著变化点
    large_jumps = np.where(np.abs(diff_vals) > np.std(diff_vals) * 5)[0]
    for idx in large_jumps:
        discontinuities.append(x_vals[idx])
    
    # 2. 计算二阶导数以检测高曲率区域
    try:
        f_second_deriv = sp.diff(f_expr, x_sym, 2)
        f_second_deriv_func = sp.lambdify(x_sym, f_second_deriv, 'numpy')
        try:
            second_deriv_vals = f_second_deriv_func(x_vals)
        except:
            second_deriv_vals = np.zeros(num_points)
            for i, x in enumerate(x_vals):
                try:
                    second_deriv_vals[i] = float(f_second_deriv.subs(x_sym, x))
                except:
                    second_deriv_vals[i] = 0
                    
        # 找出二阶导数的局部极值点
        peaks, _ = find_peaks(np.abs(second_deriv_vals), prominence=np.mean(np.abs(second_deriv_vals)))
        high_curvature_points = [x_vals[p] for p in peaks]
        
        # 按曲率大小排序取前几个点
        high_curvature_points = sorted(high_curvature_points, 
                                     key=lambda x: abs(float(f_second_deriv.subs(x_sym, x))),
                                     reverse=True)[:max_segments-1]
    except:
        high_curvature_points = []
    
    # 3. 综合不连续点和高曲率点
    potential_breakpoints = list(discontinuities) + high_curvature_points
    
    # 按距离聚类，避免分段点过于密集
    if potential_breakpoints:
        clustered_points = [potential_breakpoints[0]]
        min_distance = (domain[1] - domain[0]) / (max_segments * 2)
        
        for point in potential_breakpoints[1:]:
            if all(abs(point - p) > min_distance for p in clustered_points):
                clustered_points.append(point)
            if len(clustered_points) >= max_segments - 1:  # 预留端点
                break
        
        # 将聚类后的点与区间端点合并，并排序
        breakpoints = sorted(list(set([domain[0]] + clustered_points + [domain[1]])))
    
    # 4. 如果没有找到足够的分段点，使用均匀分段
    if len(breakpoints) < 3 and max_segments > 2:
        segment_size = (domain[1] - domain[0]) / (max_segments - 1)
        breakpoints = [domain[0] + i * segment_size for i in range(max_segments)]
    
    # 5. 局部误差自适应优化
    optimized_breakpoints = [domain[0]]
    for i in range(len(breakpoints) - 2):
        a, b = breakpoints[i], breakpoints[i+2]
        mid = breakpoints[i+1]
        
        # 在[a,b]区间内寻找最优分割点，使得两段的总误差最小
        candidates = np.linspace(a + (b-a)*0.1, b - (b-a)*0.1, 9)
        best_error = float('inf')
        best_point = mid
        
        for c in candidates:
            # 评估以c为分割点的两段误差和
            error_left = estimate_taylor_error(f_expr, x_sym, (a+c)/2, (a, c), epsilon)
            error_right = estimate_taylor_error(f_expr, x_sym, (c+b)/2, (c, b), epsilon)
            total_error = error_left + error_right
            
            if total_error < best_error:
                best_error = total_error
                best_point = c
        
        optimized_breakpoints.append(best_point)
    
    optimized_breakpoints.append(domain[1])
    return optimized_breakpoints

def estimate_taylor_error(f_expr, x_sym, x0, domain, epsilon):
    """估计泰勒展开在给定区间的误差"""
    from .adaptive import auto_order
    
    # 使用自适应阶数
    try:
        order = auto_order(f_expr, x_sym, x0, epsilon, 15, domain)
        # 计算最大误差
        x_span = max(abs(domain[0] - x0), abs(domain[1] - x0))
        f_deriv = sp.diff(f_expr, x_sym, order+1)
        max_deriv = 0
        
        # 采样点估计最大导数值
        x_vals = np.linspace(domain[0], domain[1], 10)
        for x in x_vals:
            try:
                deriv_val = abs(float(f_deriv.subs(x_sym, x)))
                max_deriv = max(max_deriv, deriv_val)
            except:
                continue
        
        # 计算误差上界
        return max_deriv * (x_span**(order+1)) / factorial(order+1)
    except:
        return float('inf')

def detect_oscillation_regions(f_expr, x_sym, domain, threshold=0.2, window_size=20):
    """检测函数的高频振荡区域"""
    num_points = 1000
    x_vals = np.linspace(domain[0], domain[1], num_points)
    
    # 计算函数值
    f_func = sp.lambdify(x_sym, f_expr, 'numpy')
    try:
        f_vals = f_func(x_vals)
    except:
        f_vals = np.zeros(num_points)
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
            except:
                f_vals[i] = np.nan
    
    # 使用滑动窗口计算局部振荡强度
    oscillation_score = np.zeros(num_points - window_size)
    for i in range(len(oscillation_score)):
        window = f_vals[i:i+window_size]
        if not np.any(np.isnan(window)):
            # 计算窗口内变号次数
            sign_changes = np.sum(np.diff(np.signbit(np.diff(window))))
            # 计算振荡分数
            oscillation_score[i] = sign_changes / (window_size - 2)
    
    # 检测振荡区域
    oscillation_regions = []
    in_region = False
    start_idx = 0
    
    for i, score in enumerate(oscillation_score):
        if not in_region and score > threshold:
            in_region = True
            start_idx = i
        elif in_region and (score <= threshold or i == len(oscillation_score) - 1):
            in_region = False
            end_idx = i
            if end_idx - start_idx > window_size//2:  # 忽略太短的区域
                start_x = x_vals[start_idx]
                end_x = x_vals[min(end_idx + window_size, num_points - 1)]
                oscillation_regions.append((start_x, end_x))
    
    return oscillation_regions
