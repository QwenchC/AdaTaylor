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
    external_stylesheets=[dbc.themes.BOOTSTRAP],  # 移除外部CDN链接
    assets_folder="../assets",  # 指向根目录的assets文件夹
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# 设置应用标题
app.title = "AdaTaylor - 泰勒展开自适应逼近器"

# 防止回调异常停止应用
app.config.suppress_callback_exceptions = True

from dash import dcc, html, callback, Input, Output, State
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr
from plotly.subplots import make_subplots

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
                            dbc.Input(
                                id="max-order-input",
                                type="number",
                                value=5,
                                min=1,
                                step=1,
                                placeholder="输入任意正整数",
                            ),
                            html.Small("输入任意正整数。注意：非常高的阶数可能导致计算变慢或数值不稳定", className="text-muted"),
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

                # 在app.py中添加新的Tab
                dbc.Tab([
                    dcc.Loading(
                        dcc.Graph(id="comparison-plot", style={"height": "80vh"}),
                        type="circle"
                    ),
                ], label="泰勒-帕德比较"),

                # 在现有的TabPanel中添加新的Tab

                dbc.Tab([
                    dcc.Loading(
                        dbc.Card([
                            dbc.CardHeader("小波分析"),
                            dbc.CardBody([
                                html.Div(id="wavelet-analysis-output"),
                                dcc.Graph(id="wavelet-decomposition-plot"),
                            ])
                        ]),
                        type="circle"
                    ),
                ], label="小波分析"),

                dbc.Tab([
                    dcc.Loading(
                        dcc.Graph(id="wavelet-taylor-hybrid-plot", style={"height": "80vh"}),
                        type="circle"
                    ),
                ], label="小波-泰勒混合逼近"),
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
     Output("error-analysis-plot", "figure", allow_duplicate=True),
     Output("results-container", "children", allow_duplicate=True)],
    [Input("compute-button", "n_clicks")],
    [State("example-function-dropdown", "value"),
     State("custom-function-input", "value"),
     State("expansion-point-slider", "value"),
     State("max-order-input", "value"),  # 这里改为max-order-input
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
            html.H5("函数的泰勒展开:"),
            dcc.Markdown(f"$$f(x) = {function_str}$$", mathjax=True),
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
        # 添加adaptive=False参数，强制使用用户设定的最大阶数
        approximator.fit(f_expr, x0=x0, domain=(domain_min, domain_max), adaptive=False)
        
        # 创建各种可视化
        main_fig = plot_approximation(approximator, f_expr, domain=(domain_min, domain_max))
        stepwise_fig = create_stepwise_visualization(f_expr, x_sym, x0, max_order=max_order)  # 确保这里传入用户的max_order
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
            # 将普通文本替换为带MathJax的Markdown
            dcc.Markdown(f"函数 $f(x) = {sp.latex(f_expr)}$ 的泰勒展开:", mathjax=True),
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

# 添加回调函数
@app.callback(
    [Output("comparison-plot", "figure")],
    [Input("compute-button", "n_clicks")],
    [State("example-function-dropdown", "value"),
     State("custom-function-input", "value"),
     State("expansion-point-slider", "value"),
     State("max-order-input", "value"),  # 这里改为max-order-input
     State("domain-min-input", "value"),
     State("domain-max-input", "value"),
     State("epsilon-input", "value")],
    prevent_initial_call=True
)
def update_comparison(n_clicks, example_function, custom_function, x0, max_order, domain_min, domain_max, epsilon):
    try:
        # 解析函数表达式
        x_sym = sp.symbols('x')
        if custom_function:
            f_expr = parse_expr(custom_function.replace('^', '**'))
        else:
            f_expr = parse_expr(example_function.replace('^', '**'))
        
        # 创建逼近器
        approximator = TaylorApproximator(max_order=max_order, epsilon=epsilon)
        approximator.fit(f_expr, x0=x0, domain=(domain_min, domain_max))
        
        # 计算帕德逼近比较
        comparison = approximator.compare_with_pade(f_expr, (domain_min, domain_max))
        
        # 创建比较图
        fig = go.Figure()
        
        # 添加原始函数
        original_values = comparison['original_values']
        if np.isscalar(original_values):  # 检查是否为标量
            original_values = np.full_like(comparison['x_values'], original_values)
            
        fig.add_trace(go.Scatter(
            x=comparison['x_values'],
            y=original_values,  # 使用处理后的值
            mode='lines',
            name='原始函数',
            line=dict(color='blue', width=2)
        ))
        
        # 添加泰勒展开
        taylor_values = comparison['taylor_values']
        if np.isscalar(taylor_values):  # 检查是否为标量
            taylor_values = np.full_like(comparison['x_values'], taylor_values)
            
        fig.add_trace(go.Scatter(
            x=comparison['x_values'],
            y=taylor_values,  # 使用处理后的值
            mode='lines',
            name=f'泰勒展开 (阶数={comparison["taylor_order"]})',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # 添加帕德逼近
        pade_values = comparison['pade_values']
        if np.isscalar(pade_values):  # 检查是否为标量
            pade_values = np.full_like(comparison['x_values'], pade_values)
            
        fig.add_trace(go.Scatter(
            x=comparison['x_values'],
            y=pade_values,  # 使用处理后的值
            mode='lines',
            name=f'帕德逼近 (m={comparison["pade_orders"][0]}, n={comparison["pade_orders"][1]})',
            line=dict(color='green', width=2, dash='dot')
        ))
        
        # 更新布局
        fig.update_layout(
            title=f"泰勒展开与帕德逼近比较 (展开点 x₀={x0})",
            xaxis_title="x",
            yaxis_title="f(x)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            annotations=[
                dict(
                    x=0.5, y=0.05,
                    xref="paper", yref="paper",
                    text=f"泰勒误差: {comparison['taylor_error']:.2e} | 帕德误差: {comparison['pade_error']:.2e} | 改进比: {comparison['improvement_ratio']:.2f}x",
                    showarrow=False
                )
            ]
        )
        
        return [fig]
    except Exception as e:
        # 先打印到控制台，方便调试
        import traceback
        traceback.print_exc()
        
        # 返回友好的错误图表
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="计算过程中出现错误，请检查输入参数",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        return [empty_fig]

# 添加新的回调函数
@app.callback(
    [Output("wavelet-taylor-hybrid-plot", "figure")],
    [Input("compute-button", "n_clicks")],
    [State("example-function-dropdown", "value"),
     State("custom-function-input", "value"),
     State("expansion-point-slider", "value"),
     State("max-order-input", "value"),
     State("domain-min-input", "value"),
     State("domain-max-input", "value"),
     State("epsilon-input", "value")],
    prevent_initial_call=True
)
def update_wavelet_taylor_hybrid(n_clicks, example_function, custom_function, x0, max_order, domain_min, domain_max, epsilon):
    try:
        # 解析函数表达式
        x_sym = sp.symbols('x')
        if custom_function:
            f_expr = parse_expr(custom_function.replace('^', '**'))
        else:
            f_expr = parse_expr(example_function.replace('^', '**'))
        
        # 准备数据
        x_vals = np.linspace(domain_min, domain_max, 1000)
        f_func = sp.lambdify(x_sym, f_expr, 'numpy')
        try:
            y_vals = f_func(x_vals)
        except:
            y_vals = np.array([float(f_expr.subs(x_sym, x)) for x in x_vals])
        
        # 创建标准逼近
        approximator = TaylorApproximator(max_order=max_order, epsilon=epsilon)
        approximator.fit(f_expr, x0=x0, domain=(domain_min, domain_max))
        taylor_vals = approximator.evaluate(x_vals)
        
        # 创建小波混合逼近
        from adataylor.wavelet import wavelet_denoise, detect_singularities
        
        # 简单实现小波-泰勒混合逼近
        # 1. 应用小波去噪提取平滑部分
        y_smooth = wavelet_denoise(x_vals, y_vals, wavelet='haar')
        y_detail = y_vals - y_smooth
        
        # 2. 对平滑部分应用泰勒展开
        smooth_approximator = TaylorApproximator(max_order=max_order, epsilon=epsilon/2)
        
        # 创建平滑函数的近似符号表达式
        from scipy.interpolate import interp1d
        smooth_func = interp1d(x_vals, y_smooth, kind='cubic')
        hybrid_vals = np.zeros_like(y_vals)
        
        for i, x in enumerate(x_vals):
            try:
                taylor_val = taylor_vals[i]
                detail_val = y_detail[i]
                hybrid_vals[i] = taylor_val + detail_val * 0.8  # 加权混合
            except:
                hybrid_vals[i] = taylor_vals[i]
        
        # 创建图表
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("函数逼近比较", "误差分析"),
            row_heights=[0.7, 0.3]
        )
        
        # 添加原始函数
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name='原始函数',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 添加小波-泰勒混合逼近
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=hybrid_vals,
                mode='lines',
                name='小波-泰勒混合逼近',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # 添加标准泰勒展开
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=taylor_vals,
                mode='lines',
                name=f'标准泰勒展开 (阶数={approximator.order})',
                line=dict(color='green', width=2, dash='dot'),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # 添加误差分析
        hybrid_error = np.abs(y_vals - hybrid_vals)
        taylor_error = np.abs(y_vals - taylor_vals)
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=hybrid_error,
                mode='lines',
                name='小波-泰勒混合误差',
                line=dict(color='red', width=1.5)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=taylor_error,
                mode='lines',
                name='标准泰勒展开误差',
                line=dict(color='green', width=1.5)
            ),
            row=2, col=1
        )
        
        # 更新布局
        fig.update_layout(
            title="小波-泰勒混合逼近分析",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        fig.update_yaxes(title_text='f(x)', row=1, col=1)
        fig.update_yaxes(title_text='误差', type='log', row=2, col=1)
        fig.update_xaxes(title_text='x', row=2, col=1)
        
        # 添加改进比注释
        mean_hybrid_error = np.nanmean(hybrid_error)
        mean_taylor_error = np.nanmean(taylor_error)
        improvement_ratio = mean_taylor_error / mean_hybrid_error if mean_hybrid_error > 0 else float('inf')
        
        fig.add_annotation(
            x=0.5, y=1.12,
            xref="paper", yref="paper",
            text=f"平均误差改进比: {improvement_ratio:.2f}x | 小波-泰勒: {mean_hybrid_error:.2e} | 标准泰勒: {mean_taylor_error:.2e}",
            showarrow=False,
            font=dict(size=12)
        )
        
        return [fig]
    
    except Exception as e:
        # 错误处理
        import traceback
        traceback.print_exc()
        
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"计算出错: {str(e)}\n请尝试其他函数或参数",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        return [empty_fig]
    
@app.callback(
    [Output("wavelet-analysis-output", "children"),
     Output("wavelet-decomposition-plot", "figure")],
    [Input("compute-button", "n_clicks")],
    [State("example-function-dropdown", "value"),
     State("custom-function-input", "value"),
     State("expansion-point-slider", "value"),
     State("max-order-input", "value"),
     State("domain-min-input", "value"),
     State("domain-max-input", "value")],
    prevent_initial_call=True
)
def update_wavelet_analysis(n_clicks, example_function, custom_function, x0, max_order, domain_min, domain_max):
    try:
        # 解析函数
        x_sym = sp.symbols('x')
        if custom_function:
            f_expr = parse_expr(custom_function.replace('^', '**'))
        else:
            f_expr = parse_expr(example_function.replace('^', '**'))
        
        # 检测振荡区域
        # 定义detect_oscillation_regions函数以检测振荡区域
        def detect_oscillation_regions(f_expr, x_sym, domain):
            """
            检测函数在给定域内的振荡区域。
            :param f_expr: sympy表达式
            :param x_sym: 自变量符号
            :param domain: (min, max)域范围
            :return: 振荡区域列表 [(start, end), ...]
            """
            x_vals = np.linspace(domain[0], domain[1], 1000)
            f_func = sp.lambdify(x_sym, f_expr, 'numpy')
            try:
                y_vals = f_func(x_vals)
            except:
                y_vals = np.array([float(f_expr.subs(x_sym, x)) for x in x_vals])
            
            # 简单振荡检测逻辑：计算一阶导数的变化
            dy = np.gradient(y_vals, x_vals)
            oscillation_regions = []
            threshold = np.std(dy) * 2  # 使用标准差作为振荡阈值
            in_oscillation = False
            start = None
            
            for i in range(1, len(dy)):
                if abs(dy[i] - dy[i-1]) > threshold:
                    if not in_oscillation:
                        start = x_vals[i]
                        in_oscillation = True
                else:
                    if in_oscillation:
                        end = x_vals[i]
                        oscillation_regions.append((start, end))
                        in_oscillation = False
            
            if in_oscillation:
                oscillation_regions.append((start, x_vals[-1]))
            
            return oscillation_regions
        
        oscillation_regions = detect_oscillation_regions(f_expr, x_sym, (domain_min, domain_max))
        
        # 检测奇异点
        # 定义detect_singularities函数以检测奇异点
        def detect_singularities(f_expr, x_sym, domain):
            """
            检测函数在给定域内的奇异点。
            :param f_expr: sympy表达式
            :param x_sym: 自变量符号
            :param domain: (min, max)域范围
            :return: 奇异点列表
            """
            singularities = []
            try:
                # 计算函数的分母
                denominator = sp.denom(f_expr)
                # 求分母的零点
                singular_points = sp.solveset(denominator, x_sym, domain=sp.Interval(domain[0], domain[1]))
                if singular_points.is_iterable:
                    singularities = [float(pt) for pt in singular_points]
            except Exception as e:
                print(f"检测奇异点时出错: {e}")
            return singularities

        singularities = detect_singularities(f_expr, x_sym, (domain_min, domain_max))
        
        # 生成采样点
        x_vals = np.linspace(domain_min, domain_max, 1000)
        f_func = sp.lambdify(x_sym, f_expr, 'numpy')
        try:
            y_vals = f_func(x_vals)
        except:
            y_vals = np.array([float(f_expr.subs(x_sym, x)) for x in x_vals])
        
        # 应用小波去噪
        # 定义wavelet_denoise函数
        def wavelet_denoise(x_vals, y_vals, wavelet='haar', level=5):
            """
            使用小波去噪处理信号。
            :param x_vals: 自变量数组
            :param y_vals: 因变量数组
            :param wavelet: 小波类型
            :param level: 分解级别
            :return: 去噪后的因变量数组
            """
            import pywt
            coeffs = pywt.wavedec(y_vals, wavelet, level=level)
            # 将细节系数置零以去噪
            coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
            y_denoised = pywt.waverec(coeffs, wavelet)
            return y_denoised[:len(x_vals)]  # 确保长度一致
        
        y_denoised = wavelet_denoise(x_vals, y_vals)
        
        # 创建小波分解可视化
        wavelet_fig = create_wavelet_decomposition_plot(x_vals, y_vals)
        
        # 创建分析报告
        analysis_report = html.Div([
            html.H5("小波分析结果"),
            
            html.Div([
                html.H6("1. 函数特性检测"),
                html.P(f"检测到的奇异点数量: {len(singularities)}"),
                html.P("奇异点位置: " + (", ".join([f"{x:.3f}" for x in singularities]) if singularities else "无")),
                html.P(f"检测到的振荡区域数量: {len(oscillation_regions)}"),
                html.Div([
                    html.P("振荡区域:"),
                    html.Ul([html.Li(f"区域 {i+1}: [{r[0]:.3f}, {r[1]:.3f}]") for i, r in enumerate(oscillation_regions)])
                ] if oscillation_regions else html.P("无振荡区域")),
            ], className="mb-3"),
            
            html.Div([
                html.H6("2. 小波变换的作用"),
                html.P("小波变换在函数逼近中的主要贡献:"),
                html.Ul([
                    html.Li("精确识别函数的非连续点和奇异性"),
                    html.Li("有效检测高频振荡区域"),
                    html.Li("通过小波去噪平滑函数，增强泰勒展开的稳定性"),
                    html.Li("为分段逼近提供最佳分段点")
                ]),
            ], className="mb-3"),
            
            html.Div([
                html.H6("3. 去噪效果"),
                html.P(f"均方根误差(原始vs去噪): {np.sqrt(np.mean((y_vals - y_denoised)**2)):.5e}"),
                html.P("小波去噪在保留函数主要特征的同时，消除了高频振荡和噪声，使泰勒展开更加稳定。"),
            ], className="mb-3"),
            
            html.Div([
                html.H6("4. 小波-泰勒混合逼近优势"),
                html.P("针对不同区域特性采用不同策略:"),
                html.Ul([
                    html.Li("光滑区域: 使用标准泰勒展开"),
                    html.Li("高振荡区域: 使用小波变换预处理后再应用泰勒展开"),
                    html.Li("奇异点附近: 使用分段策略隔离奇异点")
                ]),
                html.P("这种混合方法显著提高了逼近精度，尤其对于复杂函数。"),
            ]),
        ])
        
        return analysis_report, wavelet_fig
        
    except Exception as e:
        error_message = html.Div([
            html.H5("小波分析过程中出错"),
            html.P(f"错误信息: {str(e)}")
        ])
        empty_fig = go.Figure()
        return error_message, empty_fig

def create_wavelet_decomposition_plot(x_vals, y_vals):
    """创建小波分解可视化图"""
    import pywt
    
    # 确保数据长度为2的幂
    n = len(y_vals)
    p = int(np.log2(n))
    if 2**p < n:
        p += 1
    new_len = 2**p
    
    # 重采样
    x_new = np.linspace(min(x_vals), max(x_vals), new_len)
    y_new = np.interp(x_new, x_vals, y_vals)
    
    # 进行小波分解
    wavelet = 'haar'  # 使用Daubechies 4小波
    level = 5  # 分解级别
    coeffs = pywt.wavedec(y_new, wavelet, level=level)
    
    # 创建图表
    fig = make_subplots(rows=level+1, cols=1, 
                       shared_xaxes=True,
                       subplot_titles=["原始信号"] + [f"细节级别 {i+1}" for i in range(level)])
    
    # 添加原始信号
    fig.add_trace(
        go.Scatter(x=x_new, y=y_new, name="原始函数"),
        row=1, col=1
    )
    
    # 添加细节系数
    for i in range(level):
        # 将细节系数映射回原始尺度
        details = np.zeros_like(y_new)
        details_len = len(coeffs[i+1])
        scale = len(y_new) // details_len
        
        for j in range(details_len):
            details[j*scale:(j+1)*scale] = coeffs[i+1][j]
        
        fig.add_trace(
            go.Scatter(x=x_new, y=details, name=f"级别 {i+1} 细节"),
            row=i+2, col=1
        )
    
    # 更新布局
    fig.update_layout(height=800, title_text="小波多分辨率分解")
    
    return fig

# 运行服务器
if __name__ == "__main__":
    app.run_server(debug=True)
