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