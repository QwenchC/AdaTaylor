"""
自适应阶数选择模块 - 为泰勒展开动态选择最优阶数
"""

import numpy as np
import sympy as sp
from .utils import factorial

def auto_order_selection(f_expr, x_sym, x0, epsilon=1e-8, max_order=15, domain=None):
    """
    自动选择满足误差要求的泰勒展开阶数
    
    参数:
        f_expr: sympy表达式
        x_sym: sympy符号
        x0: 展开点
        epsilon: 误差容限
        max_order: 最大允许阶数
        domain: 考虑的区域范围 (min, max)
        
    返回:
        选择的阶数
    """
    if domain is None:
        # 默认使用展开点附近的区域
        domain = (x0 - 5, x0 + 5)
    
    # 计算区域跨度
    x_span = max(abs(domain[0] - x0), abs(domain[1] - x0))
    
    # 存储导数和误差项
    derivatives = []
    error_terms = []
    
    # 逐阶计算导数和误差
    for n in range(max_order + 1):
        try:
            # 计算n阶导数
            if n == 0:
                dnf = f_expr.subs(x_sym, x0)
            else:
                dnf = sp.diff(f_expr, x_sym, n).subs(x_sym, x0)
            
            # 将符号结果转换为浮点数
            dnf_float = float(dnf)
            derivatives.append(dnf_float)
            
            # 计算误差上界 (基于拉格朗日余项公式)
            error = abs(dnf_float) * (x_span**(n+1)) / factorial(n+1)
            error_terms.append(error)
            
            # 检查误差是否满足要求
            if error < epsilon and n > 0:  # 至少使用一阶
                return n
        except:
            # 如果导数计算失败，使用当前阶数
            if n > 0:
                return n - 1
            else:
                return 1  # 至少使用一阶
    
    # 如果达到最大阶数仍未满足精度要求，返回最大阶数
    return max_order

# 为了保持兼容性，添加别名
auto_order = auto_order_selection

def auto_order_numerical(f_func, x0, epsilon=1e-8, max_order=15, domain=None, h=1e-4):
    """
    使用数值微分的自适应阶数选择 (适用于无解析表达式的函数)
    
    参数:
        f_func: 数值函数 (接受浮点输入并返回浮点值)
        x0: 展开点
        epsilon: 误差容限
        max_order: 最大允许阶数
        domain: 考虑的区域范围 (min, max)
        h: 数值微分步长
        
    返回:
        选择的阶数
    """
    if domain is None:
        domain = (x0 - 5, x0 + 5)
    
    x_span = max(abs(domain[0] - x0), abs(domain[1] - x0))
    
    # 数值计算各阶导数 (使用有限差分)
    derivatives = [f_func(x0)]  # 0阶导数就是函数值
    
    for n in range(1, max_order + 1):
        # 使用中心差分计算导数
        diff_n = numerical_derivative(f_func, x0, n, h)
        derivatives.append(diff_n)
        
        # 计算误差上界
        error = abs(diff_n) * (x_span**(n+1)) / factorial(n+1)
        
        # 检查误差
        if error < epsilon and n > 0:
            return n
    
    return max_order

def numerical_derivative(f, x, n, h=1e-4):
    """
    使用有限差分计算n阶数值导数
    """
    if n == 0:
        return f(x)
    
    if n == 1:
        # 一阶导数(中心差分)
        return (f(x + h) - f(x - h)) / (2 * h)
    
    if n == 2:
        # 二阶导数
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
    
    # 高阶导数 (递归计算)
    d1 = numerical_derivative(f, x + h, n - 1, h)
    d2 = numerical_derivative(f, x - h, n - 1, h)
    return (d1 - d2) / (2 * h)