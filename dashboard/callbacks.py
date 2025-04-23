"""
交互回调模块 - 定义Dash应用的交互行为
"""

from dash import Input, Output, State, callback
import sympy as sp
import numpy as np

import sys
import os
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adataylor.approximator import TaylorApproximator
from visualization.interactive import plot_approximation, create_stepwise_visualization, error_analysis_visualization

def register_callbacks(app):
    """注册所有回调函数"""
    
    @app.callback(
        [
            Output("main-plot", "figure"),
            Output("stepwise-plot", "figure"),
            Output("error-plot", "figure"),
            Output("results-output", "children")
        ],
        [
            Input("compute-button", "n_clicks"),
            Input("function-dropdown", "value")
        ],
        [
            State("function-input", "value"),
            State("expansion-point", "value"),
            State("max-order", "value"),
            State("error-tolerance", "value"),
            State("domain-min", "value"),
            State("domain-max", "value")
        ],
        prevent_initial_call=True
    )
    def update_results(n_clicks, dropdown_value, function_input, x0, max_order, epsilon, domain_min, domain_max):
        """更新所有计算结果和图表"""
        from dash import html, dcc
        
        # 获取函数表达式
        x_sym = sp.symbols('x')
        
        if function_input and function_input.strip():
            function_str = function_input
        else:
            function_str = dropdown_value
        
        # 设置默认域
        if domain_min is None or domain_max is None:
            domain_min, domain_max = -5, 5
        
        # 创建逼近器并计算
        try:
            # 解析函数表达式
            f_expr = sp.sympify(function_str)
            
            approximator = TaylorApproximator(max_order=max_order, epsilon=epsilon)
            approximator.fit(f_expr, x0=x0, domain=(domain_min, domain_max))
            
            # 创建各种可视化
            main_fig = plot_approximation(approximator, f_expr, domain=(domain_min, domain_max))
            stepwise_fig = create_stepwise_visualization(f_expr, x_sym, x0, max_order=min(max_order, 10))
            error_fig = error_analysis_visualization(approximator, f_expr, domain=(domain_min, domain_max))
            
            # 获取展开式的字符串表示
            if approximator.function_type == "piecewise":
                expansion_str = "分段泰勒展开（见图表）"
                order_str = f"各段阶数: {[seg[3] for seg in approximator.segments]}"
            else:
                # 获取符号形式的展开式
                expansion = approximator.get_symbolic_expansion()
                expansion_str = sp.latex(expansion)
                order_str = f"自适应选择阶数: {approximator.order}"
            
            # 计算误差
            x_vals = np.linspace(domain_min, domain_max, 1000)
            error_dict = approximator.compute_error(f_expr, x_vals)
            
            # 创建结果显示
            results = html.Div([
                html.H5(f"函数 {function_str} 的泰勒展开:"),
                html.P(f"展开点: x₀ = {x0}"),
                html.P(order_str),
                html.Hr(),
                html.P("展开式:"),
                dcc.Markdown(f"$$f(x) \\approx {expansion_str}$$"),
                html.Hr(),
                html.P(f"最大绝对误差: {error_dict['max_absolute']:.2e}"),
                html.P(f"平均绝对误差: {error_dict['mean_absolute']:.2e}"),
                html.P(f"最大相对误差: {error_dict['max_relative']:.2e}"),
                html.P(f"平均相对误差: {error_dict['mean_relative']:.2e}"),
            ])
            
            return main_fig, stepwise_fig, error_fig, results
            
        except Exception as e:
            # 发生错误时显示错误信息
            error_message = html.Div([
                html.H5("计算出错"),
                html.P(f"错误信息: {str(e)}"),
                html.P("请检查输入函数表达式的有效性，并尝试不同的参数设置。")
            ])
            
            # 返回空图表和错误信息
            empty_fig = {
                "data": [],
                "layout": {
                    "title": "无数据",
                    "xaxis": {"title": "x"},
                    "yaxis": {"title": "y"}
                }
            }
            
            return empty_fig, empty_fig, empty_fig, error_message