"""
静态绘图模块 - 使用matplotlib创建静态图表
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def plot_taylor_comparison(f_expr, taylor_expr, x_sym, x0, domain, orders=None, figsize=(10, 6)):
    """
    绘制原始函数与不同阶数泰勒展开的比较图
    
    参数:
        f_expr: 原始函数表达式 (sympy)
        taylor_expr: 泰勒展开表达式 (如果orders指定则忽略)
        x_sym: 自变量符号
        x0: 展开点
        domain: 显示范围 (min, max)
        orders: 要比较的阶数列表 (如果提供，则计算这些阶数的泰勒展开)
        figsize: 图表大小
    """
    x_vals = np.linspace(domain[0], domain[1], 1000)
    
    # 将原始函数转换为numpy函数
    f_func = sp.lambdify(x_sym, f_expr, 'numpy')
    
    try:
        # 计算原始函数值
        f_vals = f_func(x_vals)
    except:
        # 逐点计算
        f_vals = np.zeros_like(x_vals)
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
            except:
                f_vals[i] = np.nan
    
    plt.figure(figsize=figsize)
    
    # 绘制原始函数
    plt.plot(x_vals, f_vals, 'k-', label='原始函数', linewidth=2)
    
    # 获取当前颜色循环
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    if orders is not None:
        # 绘制不同阶数的泰勒展开
        for i, order in enumerate(orders):
            # 计算泰勒展开
            taylor_n = 0
            for n in range(order + 1):
                coef = sp.diff(f_expr, x_sym, n).subs(x_sym, x0) / sp.factorial(n)
                taylor_n += coef * (x_sym - x0)**n
            
            # 计算泰勒展开值
            taylor_func = sp.lambdify(x_sym, taylor_n, 'numpy')
            try:
                taylor_vals = taylor_func(x_vals)
            except:
                taylor_vals = np.zeros_like(x_vals)
                for j, x in enumerate(x_vals):
                    try:
                        taylor_vals[j] = float(taylor_n.subs(x_sym, x))
                    except:
                        taylor_vals[j] = np.nan
            
            plt.plot(x_vals, taylor_vals, label=f'泰勒展开 (阶数={order})', 
                     color=colors[i % len(colors)])
    else:
        # 使用提供的泰勒展开表达式
        taylor_func = sp.lambdify(x_sym, taylor_expr, 'numpy')
        try:
            taylor_vals = taylor_func(x_vals)
        except:
            taylor_vals = np.zeros_like(x_vals)
            for j, x in enumerate(x_vals):
                try:
                    taylor_vals[j] = float(taylor_expr.subs(x_sym, x))
                except:
                    taylor_vals[j] = np.nan
        
        plt.plot(x_vals, taylor_vals, label='泰勒展开', color=colors[0])
    
    # 标记展开点
    f_x0 = float(f_expr.subs(x_sym, x0))
    plt.scatter([x0], [f_x0], color='red', s=50, zorder=5)
    plt.annotate(f'展开点 ({x0}, {f_x0:.2f})', 
                xy=(x0, f_x0), xytext=(x0+0.5, f_x0+0.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    plt.title('函数与泰勒展开比较')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return plt.gcf()

def plot_error_distribution(f_expr, taylor_expr, x_sym, x0, domain, figsize=(12, 5)):
    """
    绘制泰勒展开误差分布图
    
    参数:
        f_expr: 原始函数表达式
        taylor_expr: 泰勒展开表达式
        x_sym: 自变量符号
        x0: 展开点
        domain: 显示范围 (min, max)
        figsize: 图表大小
    """
    x_vals = np.linspace(domain[0], domain[1], 1000)
    
    # 将函数转换为numpy函数
    f_func = sp.lambdify(x_sym, f_expr, 'numpy')
    taylor_func = sp.lambdify(x_sym, taylor_expr, 'numpy')
    
    try:
        # 计算函数值
        f_vals = f_func(x_vals)
        taylor_vals = taylor_func(x_vals)
    except:
        # 逐点计算
        f_vals = np.zeros_like(x_vals)
        taylor_vals = np.zeros_like(x_vals)
        
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
                taylor_vals[i] = float(taylor_expr.subs(x_sym, x))
            except:
                f_vals[i] = np.nan
                taylor_vals[i] = np.nan
    
    # 计算误差
    abs_error = np.abs(f_vals - taylor_vals)
    rel_error = abs_error / (np.abs(f_vals) + 1e-15)
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 绝对误差图
    ax1.semilogy(x_vals, abs_error)
    ax1.set_title('绝对误差')
    ax1.set_xlabel('x')
    ax1.set_ylabel('|f(x) - T(x)|')
    ax1.grid(True, alpha=0.3)
    
    # 在展开点添加标记
    ax1.axvline(x=x0, color='r', linestyle='--', alpha=0.5)
    ax1.annotate('展开点', xy=(x0, ax1.get_ylim()[0]), xytext=(x0, ax1.get_ylim()[0]*1.5),
                arrowprops=dict(arrowstyle="->"))
    
    # 相对误差图
    ax2.semilogy(x_vals, rel_error)
    ax2.set_title('相对误差')
    ax2.set_xlabel('x')
    ax2.set_ylabel('|f(x) - T(x)| / |f(x)|')
    ax2.grid(True, alpha=0.3)
    
    # 在展开点添加标记
    ax2.axvline(x=x0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    return fig