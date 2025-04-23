"""
AdaTaylor - 泰勒展开自适应逼近器

主程序入口点
"""

import os
import sys
import argparse

from dashboard.app import app
from dashboard.layouts import main_layout
from dashboard.callbacks import register_callbacks

def start_dashboard(debug=False, port=8050):
    """启动Web仪表板"""
    # 设置应用布局
    app.layout = main_layout()
    
    # 注册回调
    register_callbacks(app)
    
    # 启动服务器
    app.run_server(debug=debug, port=port)
    
def run_cli():
    """运行命令行界面"""
    import sympy as sp
    from adataylor.approximator import TaylorApproximator
    
    parser = argparse.ArgumentParser(description="AdaTaylor - 泰勒展开自适应逼近器")
    parser.add_argument("function", help="要逼近的函数表达式")
    parser.add_argument("--x0", type=float, default=0, help="展开点 (默认: 0)")
    parser.add_argument("--order", type=int, default=10, help="最大展开阶数 (默认: 10)")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="误差容限 (默认: 1e-8)")
    parser.add_argument("--domain", type=str, default="-5,5", help="逼近区域 (默认: -5,5)")
    parser.add_argument("--plot", action="store_true", help="生成图表")
    
    args = parser.parse_args()
    
    # 解析域
    domain = tuple(map(float, args.domain.split(",")))
    
    # 解析函数
    x = sp.symbols('x')
    try:
        f_expr = sp.sympify(args.function)
    except:
        print(f"错误: 无法解析函数表达式 '{args.function}'")
        return
    
    # 创建逼近器
    approximator = TaylorApproximator(max_order=args.order, epsilon=args.epsilon)
    approximator.fit(f_expr, x0=args.x0, domain=domain)
    
    # 打印结果
    print(f"\n函数 {args.function} 的泰勒展开:")
    print(f"展开点: x₀ = {args.x0}")
    print(f"选择阶数: {approximator.order}")
    print("\n展开式:")
    print(sp.pretty(approximator.get_symbolic_expansion()))
    
    # 计算误差
    import numpy as np
    x_vals = np.linspace(domain[0], domain[1], 1000)
    error_dict = approximator.compute_error(f_expr, x_vals)
    
    print("\n误差分析:")
    print(f"最大绝对误差: {error_dict['max_absolute']:.2e}")
    print(f"平均绝对误差: {error_dict['mean_absolute']:.2e}")
    print(f"最大相对误差: {error_dict['max_relative']:.2e}")
    print(f"平均相对误差: {error_dict['mean_relative']:.2e}")
    
    # 生成图表
    if args.plot:
        from visualization.plotter import plot_taylor_comparison, plot_error_distribution
        import matplotlib.pyplot as plt
        
        fig1 = plot_taylor_comparison(f_expr, None, x, args.x0, domain, 
                                     orders=[1, approximator.order // 2, approximator.order])
        fig2 = plot_error_distribution(f_expr, approximator.get_symbolic_expansion(), x, args.x0, domain)
        
        plt.show()

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        run_cli()
    else:
        print("启动Web仪表板...")
        start_dashboard(debug=True)