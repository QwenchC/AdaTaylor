import time
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy import symbols, exp, sqrt, Piecewise
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adataylor.approximator import TaylorApproximator

def black_scholes_formula():
    """
    演示如何用泰勒展开简化Black-Scholes期权定价模型
    """
    # 创建符号变量
    S, K, r, sigma, t = symbols('S K r sigma t')
    
    # 定义辅助变量
    d1 = (sp.log(S/K) + (r + sigma**2/2)*t) / (sigma*sp.sqrt(t))
    d2 = d1 - sigma*sp.sqrt(t)
    
    # Black-Scholes公式中的累积正态分布函数
    N = lambda d: (1 + sp.erf(d/sp.sqrt(2))) / 2
    
    # 看涨期权价格公式
    call_price = S * N(d1) - K * exp(-r*t) * N(d2)
    
    # 将sigma作为逼近变量
    sigma_expr = call_price.subs({S: 100, K: 100, r: 0.05, t: 1})
    
    # 使用AdaTaylor进行逼近
    sigma_sym = symbols('sigma')
    taylor = TaylorApproximator(max_order=6)
    taylor.fit(sigma_expr, x0=0.2, domain=(0.1, 0.4))
    
    # 可视化比较
    sigma_vals = np.linspace(0.1, 0.4, 100)
    
    # 原始Black-Scholes公式
    from scipy.stats import norm
    def bs_call(S, K, r, sigma, t):
        d1 = (np.log(S/K) + (r + sigma**2/2)*t) / (sigma*np.sqrt(t))
        d2 = d1 - sigma*np.sqrt(t)
        return S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
    
    true_prices = [bs_call(100, 100, 0.05, sig, 1) for sig in sigma_vals]
    
    # 泰勒逼近
    approx_prices = taylor.evaluate(sigma_vals)
    
    # 计算误差
    abs_error = np.abs(np.array(true_prices) - approx_prices)
    rel_error = abs_error / np.array(true_prices)
    
    # 可视化比较
    plt.figure(figsize=(12, 8))
    
    # 绘制原始与近似曲线对比
    plt.subplot(2, 1, 1)
    plt.plot(sigma_vals, true_prices, 'b-', label='原始BS公式')
    plt.plot(sigma_vals, approx_prices, 'r--', label='泰勒近似')
    plt.xlabel('波动率 (σ)')
    plt.ylabel('期权价格')
    plt.title('Black-Scholes期权定价泰勒展开近似')
    plt.legend()
    plt.grid(True)
    
    # 绘制误差分析
    plt.subplot(2, 1, 2)
    plt.plot(sigma_vals, abs_error, 'g-', label='绝对误差')
    plt.plot(sigma_vals, rel_error * 100, 'm--', label='相对误差 (%)')
    plt.xlabel('波动率 (σ)')
    plt.ylabel('误差')
    plt.title('近似误差分析')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 输出性能分析
    print(f"最大绝对误差: {np.max(abs_error):.6f}")
    print(f"最大相对误差: {np.max(rel_error)*100:.6f}%")
    print(f"平均绝对误差: {np.mean(abs_error):.6f}")
    print(f"平均相对误差: {np.mean(rel_error)*100:.6f}%")
    
    # 比较计算效率
    n_trials = 10000
    
    start = time.time()
    for _ in range(n_trials):
        _ = [bs_call(100, 100, 0.05, sig, 1) for sig in sigma_vals[:10]]
    bs_time = time.time() - start
    
    start = time.time()
    for _ in range(n_trials):
        _ = taylor.evaluate(sigma_vals[:10])
    taylor_time = time.time() - start
    
    print(f"BS原始公式计算时间: {bs_time:.4f}秒")
    print(f"泰勒近似计算时间: {taylor_time:.4f}秒")
    print(f"加速比: {bs_time/taylor_time:.2f}x")