"""
Dash应用模块 - 创建Web界面
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
import dash_bootstrap_components as dbc

# 导入自定义模块
from adataylor.approximator import TaylorApproximator
from visualization.interactive import plot_approximation, create_stepwise_visualization, error_analysis_visualization

# 创建Dash应用
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    # 添加以下外部脚本支持
    external_scripts=[
        # MathJax配置
        {
            "src": "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML",
        }
    ]
)

# 也可以在app.index_string中设置
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true
                }
            });
        </script>
        <script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML">
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# 设置应用标题
app.title = "AdaTaylor - 泰勒展开自适应逼近器"

# 防止回调异常停止应用
app.config.suppress_callback_exceptions = True

from dash import dcc, html, callback, Input, Output, State
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr

# 示例函数列表
example_functions = [
    {"label": "sin(x)", "value": "sin(x)"},
    {"label": "exp(x)", "value": "exp(x)"},
    {"label": "1/(1+x^2)", "value": "1/(1+x^2)"},
    {"label": "x^3 - 2*x^2 + 3*x - 1", "value": "x^3 - 2*x^2 + 3*x - 1"},
    {"label": "log(x+2)", "value": "log(x+2)"},
    {"label": "abs(x) + sin(x)", "value": "abs(x) + sin(x)"},
    {"label": "1/x", "value": "1/x"},
    {"label": "sin(1/x)", "value": "sin(1/x)"},
    {"label": "exp(-x^2)", "value": "exp(-x^2)"},
]

# 应用布局
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("AdaTaylor: 自适应泰勒展开逼近器", className="text-center my-4"),
            html.P("该工具使用泰勒展开和小波分析实现高精度函数逼近，并提供详细的可视化分析。", 
                  className="text-center text-muted"),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("函数输入"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("选择示例函数："),
                            dcc.Dropdown(
                                id="example-function-dropdown",
                                options=example_functions,
                                value="sin(x)",
                                clearable=False
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("或输入自定义函数："),
                            dbc.Input(
                                id="custom-function-input",
                                type="text",
                                placeholder="例如：x^2 * exp(-x)",
                                value=""
                            ),
                        ], width=6),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("展开点 (x₀)："),
                            dcc.Slider(
                                id="expansion-point-slider",
                                min=-5,
                                max=5,
                                step=0.1,
                                value=0,
                                marks={i: str(i) for i in range(-5, 6)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("最大阶数："),
                            dcc.Slider(
                                id="max-order-slider",
                                min=1,
                                max=15,
                                step=1,
                                value=5,
                                marks={i: str(i) for i in [1, 5, 10, 15]},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], width=6),
                    ], className="mt-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("显示范围："),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Input(
                                        id="domain-min-input",
                                        type="number",
                                        value=-5,
                                        step=0.5
                                    ),
                                ], width=6),
                                dbc.Col([
                                    dbc.Input(
                                        id="domain-max-input",
                                        type="number",
                                        value=5,
                                        step=0.5
                                    ),
                                ], width=6),
                            ]),
                        ], width=6),
                        dbc.Col([
                            html.Label("目标精度 (ε)："),
                            dbc.Input(
                                id="epsilon-input",
                                type="number",
                                value=1e-6,
                                step=1e-7,
                                min=1e-10,
                                max=1e-1
                            ),
                        ], width=6),
                    ], className="mt-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "计算泰勒展开", 
                                id="compute-button", 
                                color="primary", 
                                className="mt-3 w-100"
                            ),
                        ], width=12),
                    ]),
                ]),
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("结果分析"),
                dbc.CardBody([
                    html.Div(id="results-container", className="mt-2"),
                ]),
            ]),
        ], width=4),
        
        dbc.Col([
            dbc.Tabs([
                dbc.Tab([
                    dcc.Loading(
                        dcc.Graph(id="main-plot", style={"height": "80vh"}),
                        type="circle"
                    ),
                ], label="基本视图"),
                
                dbc.Tab([
                    dcc.Loading(
                        dcc.Graph(id="stepwise-plot", style={"height": "80vh"}),
                        type="circle"
                    ),
                ], label="逐步展开"),
                
                dbc.Tab([
                    dcc.Loading(
                        dcc.Graph(id="error-analysis-plot", style={"height": "80vh"}),
                        type="circle"
                    ),
                ], label="误差分析"),
            ]),
        ], width=8),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P(
                "AdaTaylor - 基于泰勒展开的自适应函数逼近器 | 开发于2025年",
                className="text-center text-muted"
            ),
        ], width=12),
    ], className="mt-4"),
], fluid=True)

# 回调函数
@app.callback(
    [Output("main-plot", "figure", allow_duplicate=True),
     Output("stepwise-plot", "figure", allow_duplicate=True),
     Output("error-analysis-plot", "figure", allow_duplicate=True),  # 修改为与布局匹配
     Output("results-container", "children", allow_duplicate=True)],  # 修改为与布局匹配
    [Input("compute-button", "n_clicks")],
    [State("example-function-dropdown", "value"),
     State("custom-function-input", "value"),
     State("expansion-point-slider", "value"),
     State("max-order-slider", "value"),
     State("domain-min-input", "value"),
     State("domain-max-input", "value"),
     State("epsilon-input", "value")],
     prevent_initial_call=True
)
def update_plots(n_clicks, example_function, custom_function, x0, max_order, domain_min, domain_max, epsilon):
    if n_clicks is None:
        # 初始加载时的默认图表
        x_sym = sp.symbols('x')
        f_expr = sp.sin(x_sym)
        
        # 创建一个默认的逼近器
        approximator = TaylorApproximator(max_order=5, epsilon=1e-6)
        approximator.fit(f_expr, x0=0, domain=(-5, 5))
        
        # 创建默认图表
        main_fig = plot_approximation(approximator, f_expr, domain=(-5, 5))
        stepwise_fig = create_stepwise_visualization(f_expr, x_sym, 0, max_order=5)
        error_fig = error_analysis_visualization(approximator, f_expr)
        
        # 默认结果文本
        results = html.Div([
            html.H5(f"函数 {function_str} 的泰勒展开:"),
            html.P(f"展开点: x₀ = {x0}"),
            html.P(order_str),
            html.Hr(),
            html.P("展开式:"),
            dcc.Markdown(f"$$f(x) \\approx {expansion_str}$$", mathjax=True),
            html.Hr(),
            html.P(f"最大绝对误差: {error_dict['max_absolute']:.2e}"),
            html.P(f"平均绝对误差: {error_dict['mean_absolute']:.2e}"),
            html.P(f"函数类型: {approximator.function_type}"),
        ])
        
        return main_fig, stepwise_fig, error_fig, results
    
    # 解析函数表达式
    x_sym = sp.symbols('x')
    try:
        if custom_function:
            # 使用用户自定义函数
            f_expr = parse_expr(custom_function.replace('^', '**'))
            function_str = custom_function
        else:
            # 使用示例函数
            f_expr = parse_expr(example_function.replace('^', '**'))
            function_str = example_function
    except Exception as e:
        # 函数解析错误
        error_message = html.Div([
            html.H5("函数解析错误"),
            html.P(f"错误信息: {str(e)}"),
            html.P("请检查函数语法是否正确。支持的操作符包括: +, -, *, /, ^, sin, cos, exp, log, sqrt 等。")
        ])
        # 返回空图表和错误信息
        empty_fig = go.Figure()
        return empty_fig, empty_fig, empty_fig, error_message
    
    # 确保域值有效
    if domain_min >= domain_max:
        domain_min, domain_max = -5, 5
    
    # 创建逼近器并计算
    try:
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
            dcc.Markdown(f"$$f(x) \\approx {expansion_str}$$", mathjax=True),
            html.Hr(),
            html.P(f"最大绝对误差: {error_dict['max_absolute']:.2e}"),
            html.P(f"平均绝对误差: {error_dict['mean_absolute']:.2e}"),
            html.P(f"函数类型: {approximator.function_type}"),
        ])
        
        return main_fig, stepwise_fig, error_fig, results
    
    except Exception as e:
        # 计算过程中出错
        error_message = html.Div([
            html.H5("计算错误"),
            html.P(f"错误信息: {str(e)}"),
            html.P("请尝试不同的函数或参数设置。")
        ])
        # 返回空图表和错误信息
        empty_fig = go.Figure()
        return empty_fig, empty_fig, empty_fig, error_message

# 运行服务器
if __name__ == "__main__":
    app.run_server(debug=True)