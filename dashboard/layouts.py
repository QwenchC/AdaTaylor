"""
UI布局模块 - 定义Dash应用的页面布局
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# 示例函数列表
example_functions = [
    {"label": "sin(x)", "value": "sin(x)"},
    {"label": "exp(x)", "value": "exp(x)"},
    {"label": "1/(1+x^2)", "value": "1/(1+x^2)"},
    {"label": "log(x)", "value": "log(x)"},
    {"label": "tan(x)", "value": "tan(x)"},
    {"label": "x*sin(1/x)", "value": "x*sin(1/x)"},
    {"label": "sqrt(abs(x))", "value": "sqrt(abs(x))"},
    {"label": "exp(-x^2)", "value": "exp(-x^2)"},
    {"label": "sin(x)/x", "value": "sin(x)/x"},
    {"label": "Piecewise((x^2, x<0), (sqrt(x), x>=0))", "value": "Piecewise((x^2, x<0), (sqrt(x), x>=0))"}
]

def create_header():
    """创建页面头部"""
    return html.Div([
        html.H1("AdaTaylor - 泰勒展开自适应逼近器", className="display-4 text-center mb-4"),
        html.Hr(),
        html.P(
            "这个工具使用自适应算法和小波分析技术，为各种函数提供高精度泰勒展开逼近。",
            className="lead text-center"
        ),
    ], className="container mt-4")

def create_footer():
    """创建页面底部"""
    return html.Footer([
        html.Hr(),
        html.P("AdaTaylor © 2023", className="text-center"),
        html.P("基于泰勒展开的函数自适应逼近工具", className="text-center"),
    ], className="container mt-4")

def main_layout():
    """创建主布局"""
    return html.Div([
        create_header(),
        
        dbc.Container([
            dbc.Row([
                # 左侧控制面板
                dbc.Col([
                    html.H3("函数输入", className="mb-3"),
                    
                    # 函数输入区域
                    dbc.Card([
                        dbc.CardBody([
                            html.Label("选择示例函数:"),
                            dcc.Dropdown(
                                id="function-dropdown",
                                options=example_functions,
                                value="sin(x)",
                                clearable=False
                            ),
                            
                            html.Div([
                                html.Label("或输入自定义函数:"),
                                dcc.Input(
                                    id="function-input",
                                    type="text",
                                    placeholder="例如: sin(x)*exp(-x^2)",
                                    className="form-control"
                                )
                            ], className="mt-3"),
                            
                            html.Div([
                                html.Label("展开点:"),
                                dcc.Input(
                                    id="expansion-point",
                                    type="number",
                                    value=0,
                                    className="form-control"
                                )
                            ], className="mt-3"),
                            
                            html.Div([
                                html.Label("最大阶数:"),
                                dcc.Slider(
                                    id="max-order",
                                    min=1,
                                    max=15,
                                    step=1,
                                    value=5,
                                    marks={i: str(i) for i in range(1, 16, 2)},
                                )
                            ], className="mt-3"),
                            
                            html.Div([
                                html.Label("误差容限:"),
                                dcc.Input(
                                    id="error-tolerance",
                                    type="number",
                                    value=1e-8,
                                    step=1e-9,
                                    className="form-control"
                                )
                            ], className="mt-3"),
                            
                            html.Div([
                                html.Label("显示区间:"),
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Input(
                                            id="domain-min",
                                            type="number",
                                            value=-5,
                                            className="form-control"
                                        )
                                    ),
                                    dbc.Col(
                                        dcc.Input(
                                            id="domain-max",
                                            type="number",
                                            value=5,
                                            className="form-control"
                                        )
                                    )
                                ])
                            ], className="mt-3"),
                            
                            dbc.Button(
                                "计算",
                                id="compute-button",
                                color="primary",
                                className="mt-3 w-100"
                            )
                        ])
                    ], className="mb-4"),
                    
                    # 结果显示区域
                    html.H3("结果", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id="results-output")
                        ])
                    ])
                ], md=4),
                
                # 右侧可视化区域
                dbc.Col([
                    html.H3("可视化", className="mb-3"),
                    
                    # 选项卡
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(id="main-plot", style={"height": "600px"})
                        ], label="函数逼近"),
                        
                        dbc.Tab([
                            dcc.Graph(id="stepwise-plot", style={"height": "600px"})
                        ], label="逐步展开"),
                        
                        dbc.Tab([
                            dcc.Graph(id="error-plot", style={"height": "600px"})
                        ], label="误差分析")
                    ])
                ], md=8)
            ])
        ], fluid=True, className="mt-4"),
        
        create_footer()
    ])