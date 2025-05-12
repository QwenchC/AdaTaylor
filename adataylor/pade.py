"""
帕德逼近模块 - 实现有理函数形式的逼近
"""

import numpy as np
import sympy as sp
from .utils import factorial

def pade_approximant(f_expr, x_sym, x0, m, n):
    """
    计算函数的帕德逼近
    
    参数:
        f_expr: sympy表达式
        x_sym: 符号变量
        x0: 展开点
        m: 分子多项式阶数
        n: 分母多项式阶数
        
    返回:
        分子多项式, 分母多项式, 有理函数表达式
    """
    # 计算泰勒系数
    c = [0] * (m + n + 1)
    for k in range(m + n + 1):
        if k == 0:
            c[k] = float(f_expr.subs(x_sym, x0))
        else:
            deriv = sp.diff(f_expr, x_sym, k).subs(x_sym, x0)
            c[k] = float(deriv) / factorial(k)
    
    # 构建线性方程组
    # 设分母系数为 q = [1, q1, q2, ..., qn]
    # 设分子系数为 p = [p0, p1, p2, ..., pm]
    # 满足条件：sum(c[k-j] * q[j]) = p[k] for k=0,1,...,m+n
    
    # 先解分母系数
    if n > 0:
        # 构建矩阵方程 C * q = -d
        C = np.zeros((n, n))
        d = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                C[i, j] = c[m + i + 1 - j] if m + i + 1 - j < len(c) else 0
            d[i] = c[m + i + 1]
        
        # 求解线性方程组
        try:
            q_coeffs = np.linalg.solve(C, -d)
            q_coeffs = np.concatenate(([1.0], q_coeffs))
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，返回低阶近似
            if m > 0:
                return pade_approximant(f_expr, x_sym, x0, m-1, n-1)
            else:
                # 退化为泰勒多项式
                q_coeffs = np.array([1.0])
                m, n = m, 0
    else:
        q_coeffs = np.array([1.0])
    
    # 计算分子系数
    p_coeffs = np.zeros(m + 1)
    for k in range(m + 1):
        p_coeffs[k] = sum(c[k-j] * q_coeffs[j] for j in range(min(k+1, n+1)))
    
    # 创建sympy表达式
    P = 0
    Q = 0
    for i, p_i in enumerate(p_coeffs):
        P += p_i * (x_sym - x0)**i
    for i, q_i in enumerate(q_coeffs):
        Q += q_i * (x_sym - x0)**i
    
    R = P / Q
    return P, Q, R

def auto_pade_order(f_expr, x_sym, x0, epsilon=1e-8, max_order=10, domain=None):
    """
    自动选择满足误差要求的帕德逼近阶数
    
    参数:
        f_expr: sympy表达式
        x_sym: 符号变量
        x0: 展开点
        epsilon: 误差容限
        max_order: 最大总阶数
        domain: 考虑的区域范围
        
    返回:
        (m, n) - 分子和分母的最优阶数
    """
    if domain is None:
        domain = (x0 - 5, x0 + 5)
    
    # 采样点
    x_vals = np.linspace(domain[0], domain[1], 50)
    f_func = sp.lambdify(x_sym, f_expr, 'numpy')
    
    # 计算真实函数值
    try:
        f_vals = f_func(x_vals)
    except:
        f_vals = np.zeros(len(x_vals))
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
            except:
                f_vals[i] = np.nan
    
    best_error = float('inf')
    best_m, best_n = 1, 1
    
    # 尝试不同的m,n组合
    for total_order in range(2, max_order + 1):
        for m in range(total_order // 2, total_order + 1):
            n = total_order - m
            if n < 0: continue
            
            try:
                _, _, R = pade_approximant(f_expr, x_sym, x0, m, n)
                R_func = sp.lambdify(x_sym, R, 'numpy')
                
                # 计算近似值
                try:
                    R_vals = R_func(x_vals)
                except:
                    R_vals = np.zeros(len(x_vals))
                    for i, x in enumerate(x_vals):
                        try:
                            R_vals[i] = float(R.subs(x_sym, x))
                        except:
                            R_vals[i] = np.nan
                
                # 计算误差
                err = np.nanmean(np.abs(f_vals - R_vals))
                
                if err < best_error:
                    best_error = err
                    best_m, best_n = m, n
                
                if err < epsilon:
                    return m, n
            except:
                continue
    
    return best_m, best_n

# 然后在approximator.py中添加
def compare_with_pade(self, f_expr, domain=None, num_points=100):
    """
    将泰勒展开与帕德逼近进行比较
    
    参数:
        f_expr: 原函数表达式
        domain: 比较区域
        num_points: 评估点数量
        
    返回:
        包含比较结果的字典
    """
    if not self.fitted:
        raise ValueError("必须先调用fit方法")
    
    if domain is None:
        domain = self.domain
    
    from .pade import pade_approximant, auto_pade_order
    
    # 获取符号变量
    x_sym = list(f_expr.free_symbols)[0]
    
    # 自动选择帕德逼近阶数
    m, n = auto_pade_order(f_expr, x_sym, self.x0, self.epsilon, self.max_order, domain)
    
    # 计算帕德逼近
    P, Q, R = pade_approximant(f_expr, x_sym, self.x0, m, n)
    
    # 采样点
    x_vals = np.linspace(domain[0], domain[1], num_points)
    
    # 计算原函数值
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
    
    # 计算泰勒展开值
    taylor_vals = self.evaluate(x_vals)
    
    # 计算帕德逼近值
    R_func = sp.lambdify(x_sym, R, 'numpy')
    try:
        pade_vals = R_func(x_vals)
    except:
        pade_vals = np.zeros(num_points)
        for i, x in enumerate(x_vals):
            try:
                pade_vals[i] = float(R.subs(x_sym, x))
            except:
                pade_vals[i] = np.nan
    
    # 计算误差
    taylor_err = np.nanmean(np.abs(f_vals - taylor_vals))
    pade_err = np.nanmean(np.abs(f_vals - pade_vals))
    
    return {
        "x_values": x_vals,
        "original_values": f_vals,
        "taylor_values": taylor_vals,
        "pade_values": pade_vals,
        "taylor_error": taylor_err,
        "pade_error": pade_err,
        "taylor_order": self.order,
        "pade_orders": (m, n),
        "pade_expression": R,
        "improvement_ratio": taylor_err / pade_err if pade_err > 0 else float('inf')
    }
