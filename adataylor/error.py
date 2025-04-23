"""
误差分析模块 - 计算并分析泰勒展开逼近的误差
"""

import numpy as np
import sympy as sp
from .utils import factorial

def compute_remainder_bound(f_expr, x_sym, x0, order, x_range):
    """
    计算泰勒展开的余项上界 (使用拉格朗日余项)
    
    参数:
        f_expr: sympy表达式
        x_sym: sympy符号
        x0: 展开点
        order: 展开阶数
        x_range: x的范围 (min, max)
        
    返回:
        余项上界
    """
    try:
        # 计算(order+1)阶导数
        f_derivative = sp.diff(f_expr, x_sym, order+1)
        
        # 在给定区间上取导数的最大绝对值
        x_vals = np.linspace(x_range[0], x_range[1], 100)
        f_deriv_func = sp.lambdify(x_sym, f_derivative, 'numpy')
        
        try:
            deriv_vals = f_deriv_func(x_vals)
            max_deriv = np.max(np.abs(deriv_vals))
        except:
            # 如果向量化计算失败，逐点计算
            max_deriv = 0
            for x in x_vals:
                try:
                    deriv_val = abs(float(f_derivative.subs(x_sym, x)))
                    max_deriv = max(max_deriv, deriv_val)
                except:
                    continue
        
        # 计算最大距离
        max_distance = max(abs(x_range[0] - x0), abs(x_range[1] - x0))
        
        # 计算拉格朗日余项上界
        remainder_bound = max_deriv * (max_distance**(order+1)) / factorial(order+1)
        
        return remainder_bound
    
    except:
        return float('inf')  # 如果计算失败，返回无穷大

def error_analysis(f_expr, taylor_expr, x_sym, x_range, num_points=1000):
    """
    分析泰勒展开的误差特性
    
    参数:
        f_expr: 原始函数表达式
        taylor_expr: 泰勒展开表达式
        x_sym: 自变量符号
        x_range: x的范围 (min, max)
        num_points: 评估点数
        
    返回:
        包含误差统计信息的字典
    """
    # 创建可调用的函数
    f_func = sp.lambdify(x_sym, f_expr, 'numpy')
    taylor_func = sp.lambdify(x_sym, taylor_expr, 'numpy')
    
    # 生成评估点
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    
    # 计算函数值
    try:
        f_vals = f_func(x_vals)
        taylor_vals = taylor_func(x_vals)
    except:
        # 逐点计算
        f_vals = np.zeros(num_points)
        taylor_vals = np.zeros(num_points)
        
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
                taylor_vals[i] = float(taylor_expr.subs(x_sym, x))
            except:
                f_vals[i] = np.nan
                taylor_vals[i] = np.nan
    
    # 计算误差
    abs_error = np.abs(f_vals - taylor_vals)
    rel_error = abs_error / (np.abs(f_vals) + 1e-15)  # 避免除零
    
    # 过滤掉NaN值
    valid_abs = abs_error[~np.isnan(abs_error)]
    valid_rel = rel_error[~np.isnan(rel_error)]
    
    if len(valid_abs) == 0:
        return {
            "max_abs_error": np.nan,
            "mean_abs_error": np.nan,
            "max_rel_error": np.nan,
            "mean_rel_error": np.nan,
            "valid_points": 0,
            "total_points": num_points
        }
    
    # 返回误差统计
    return {
        "max_abs_error": np.max(valid_abs),
        "mean_abs_error": np.mean(valid_abs),
        "max_rel_error": np.max(valid_rel),
        "mean_rel_error": np.mean(valid_rel),
        "valid_points": len(valid_abs),
        "total_points": num_points,
        "abs_error_data": abs_error,
        "rel_error_data": rel_error,
        "x_data": x_vals
    }