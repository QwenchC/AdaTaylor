import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp

def plot_approximation(approximator, f_expr, domain=(-5, 5), num_points=1000, 
                     error_scale='linear', title=None):
    """
    创建交互式函数逼近可视化
    
    参数:
        approximator: TaylorApproximator实例
        f_expr: 原始函数表达式
        domain: 显示范围
        num_points: 采样点数
        error_scale: 误差显示比例 ('linear'或'log')
        title: 图表标题
    
    返回:
        plotly Figure对象
    """
    x_sym = approximator.x
    x_vals = np.linspace(domain[0], domain[1], num_points)
    
    # 计算原始函数值
    f_lambda = sp.lambdify(x_sym, f_expr, 'numpy')
    try:
        f_vals = f_lambda(x_vals)
    except:
        # 对有奇点的函数，逐点计算
        f_vals = np.zeros(num_points)
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
            except:
                f_vals[i] = np.nan
    
    # 计算逼近值
    approx_vals = approximator.evaluate(x_vals)
    
    # 计算误差
    abs_error = np.abs(f_vals - approx_vals)
    # 处理潜在的无穷大或NaN值
    abs_error = np.where(np.isfinite(abs_error), abs_error, np.nan)
    
    # 创建子图：上方是函数对比，下方是误差
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.1,
                       subplot_titles=("函数与泰勒逼近", "误差分析"),
                       row_heights=[0.7, 0.3])
    
    # 添加原始函数
    fig.add_trace(
        go.Scatter(x=x_vals, y=f_vals, name="原始函数", line=dict(color='blue')),
        row=1, col=1
    )
    
    # 添加逼近函数
    if approximator.function_type == "piecewise":
        # 对分段函数，分段显示
        for i, (segment, x0, _, order) in enumerate(approximator.segments):
            mask = (x_vals >= segment[0]) & (x_vals <= segment[1])
            if np.any(mask):
                segment_x = x_vals[mask]
                segment_y = approx_vals[mask]
                fig.add_trace(
                    go.Scatter(
                        x=segment_x, 
                        y=segment_y, 
                        name=f"泰勒逼近 (分段{i+1}, 阶数:{order})", 
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
                # 添加展开点标记
                fig.add_trace(
                    go.Scatter(
                        x=[x0], 
                        y=[f_lambda(x0) if callable(f_lambda) else float(f_expr.subs(x_sym, x0))], 
                        mode='markers',
                        marker=dict(size=10, color='black'),
                        name=f"展开点 x₀={x0:.2f}"
                    ),
                    row=1, col=1
                )
    else:
        # 单一展开点
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=approx_vals, 
                name=f"泰勒逼近 (阶数:{approximator.order})", 
                line=dict(color='red')
            ),
            row=1, col=1
        )
        # 添加展开点标记
        fig.add_trace(
            go.Scatter(
                x=[approximator.expansion_point], 
                y=[f_lambda(approximator.expansion_point) if callable(f_lambda) else 
                   float(f_expr.subs(x_sym, approximator.expansion_point))], 
                mode='markers',
                marker=dict(size=10, color='black'),
                name=f"展开点 x₀={approximator.expansion_point:.2f}"
            ),
            row=1, col=1
        )
    
    # 添加误差曲线
    if error_scale == 'log':
        # 对数误差（处理零误差）
        log_error = np.log10(abs_error + 1e-15)
        fig.add_trace(
            go.Scatter(x=x_vals, y=log_error, name="对数误差", line=dict(color='green')),
            row=2, col=1
        )
        fig.update_yaxes(title_text="log₁₀(误差)", row=2, col=1)
    else:
        # 线性误差
        fig.add_trace(
            go.Scatter(x=x_vals, y=abs_error, name="绝对误差", line=dict(color='green')),
            row=2, col=1
        )
        fig.update_yaxes(title_text="绝对误差", row=2, col=1)
    
    # 更新布局
    if title is None:
        if approximator.function_type == "smooth":
            title = f"泰勒展开逼近 (阶数: {approximator.order})"
        elif approximator.function_type == "piecewise":
            title = f"分段泰勒展开逼近 ({len(approximator.segments)}段)"
        else:
            title = f"泰勒展开逼近 (振荡函数, 阶数: {approximator.order})"
    
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="f(x)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800,
        hovermode="x unified"
    )
    
    # 添加误差统计信息
    max_error = np.nanmax(abs_error)
    mean_error = np.nanmean(abs_error)
    
    error_stats = (
        f"最大误差: {max_error:.2e}<br>"
        f"平均误差: {mean_error:.2e}"
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.25,
        text=error_stats,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig

def create_stepwise_visualization(f_expr, x_sym, x0, max_order=5, domain=(-5, 5), num_points=1000):
    """
    创建逐步展开演示可视化
    
    参数:
        f_expr: sympy表达式
        x_sym: sympy符号
        x0: 展开点
        max_order: 最大展开阶数
        domain: 显示范围
        num_points: 采样点数
        
    返回:
        plotly Figure对象
    """
    x_vals = np.linspace(domain[0], domain[1], num_points)
    
    # 计算原始函数值
    f_lambda = sp.lambdify(x_sym, f_expr, 'numpy')
    f_vals = f_lambda(x_vals)
    
    # 计算各阶泰勒展开
    approximations = []
    for order in range(max_order + 1):
        taylor_sum = 0
        for n in range(order + 1):
            nth_derivative = sp.diff(f_expr, (x_sym, n))
            coef = float(nth_derivative.subs(x_sym, x0)) / np.math.factorial(n)
            taylor_sum += coef * (x_vals - x0) ** n
        approximations.append(taylor_sum)
    
    # 创建基础图表
    fig = go.Figure()
    
    # 添加原始函数
    fig.add_trace(
        go.Scatter(
            x=x_vals, 
            y=f_vals, 
            name="原始函数", 
            line=dict(color='blue', width=2)
        )
    )
    
    # 添加各阶泰勒展开
    colors = ['red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    for order, approx in enumerate(approximations):
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=approx, 
                name=f"{order}阶泰勒展开", 
                line=dict(
                    color=colors[order % len(colors)], 
                    width=1.5,
                    dash='dash' if order < max_order else None
                ),
                visible=(order == 0)  # 初始只显示0阶
            )
        )
    
    # 添加展开点标记
    fig.add_trace(
        go.Scatter(
            x=[x0], 
            y=[float(f_expr.subs(x_sym, x0))], 
            mode='markers',
            marker=dict(size=10, color='black'),
            name=f"展开点 (x₀={x0})"
        )
    )
    
    # 创建滑块
    steps = []
    for i in range(max_order + 1):
        step = dict(
            method="update",
            args=[
                {"visible": [True] + [j <= i for j in range(max_order + 1)] + [True]},
                {"title": f"泰勒展开演示 (阶数: {i})"}
            ],
            label=str(i)
        )
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "阶数: "},
        pad={"t": 50},
        steps=steps
    )]
    
    # 更新布局
    fig.update_layout(
        title="泰勒展开逐步演示 (阶数: 0)",
        xaxis_title="x",
        yaxis_title="f(x)",
        sliders=sliders,
        height=700,
        hovermode="x unified"
    )
    
    return fig

def error_analysis_visualization(approximator, f_expr, domain=(-5, 5), num_points=1000):
    """
    创建详细的误差分析可视化
    
    参数:
        approximator: TaylorApproximator实例
        f_expr: 原始函数表达式
        domain: 显示范围
        num_points: 采样点数
        
    返回:
        plotly Figure对象
    """
    x_sym = approximator.x
    x_vals = np.linspace(domain[0], domain[1], num_points)
    
    # 计算原始函数值
    f_lambda = sp.lambdify(x_sym, f_expr, 'numpy')
    try:
        f_vals = f_lambda(x_vals)
    except:
        # 对有奇点的函数，逐点计算
        f_vals = np.zeros(num_points)
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
            except:
                f_vals[i] = np.nan
    
    # 计算逼近值
    approx_vals = approximator.evaluate(x_vals)
    
    # 计算误差
    abs_error = np.abs(f_vals - approx_vals)
    rel_error = abs_error / (np.abs(f_vals) + 1e-15)
    
    # 使用对数刻度显示误差
    log_abs_error = np.log10(abs_error + 1e-15)
    log_rel_error = np.log10(rel_error + 1e-15)
    
    # 创建4个子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "函数与泰勒逼近", 
            "绝对误差 (线性刻度)", 
            "绝对误差 (对数刻度)", 
            "相对误差 (对数刻度)"
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ]
    )
    
    # 添加原始函数和逼近
    fig.add_trace(
        go.Scatter(
            x=x_vals, 
            y=f_vals, 
            name="原始函数", 
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_vals, 
            y=approx_vals, 
            name=f"泰勒逼近 (阶数:{approximator.order})", 
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # 添加展开点标记
    fig.add_trace(
        go.Scatter(
            x=[approximator.expansion_point], 
            y=[float(f_expr.subs(x_sym, approximator.expansion_point))], 
            mode='markers',
            marker=dict(size=10, color='black'),
            name=f"展开点 x₀={approximator.expansion_point:.2f}"
        ),
        row=1, col=1
    )
    
    # 添加线性绝对误差
    fig.add_trace(
        go.Scatter(
            x=x_vals, 
            y=abs_error, 
            name="绝对误差", 
            line=dict(color='green')
        ),
        row=1, col=2
    )
    
    # 添加对数绝对误差
    fig.add_trace(
        go.Scatter(
            x=x_vals, 
            y=log_abs_error, 
            name="绝对误差 (对数)", 
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    # 添加对数相对误差
    fig.add_trace(
        go.Scatter(
            x=x_vals, 
            y=log_rel_error, 
            name="相对误差 (对数)", 
            line=dict(color='orange')
        ),
        row=2, col=2
    )
    
    # 更新布局
    fig.update_layout(
        title=f"泰勒展开详细误差分析 (阶数: {approximator.order})",
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # 更新y轴标题
    fig.update_yaxes(title_text="f(x)", row=1, col=1)
    fig.update_yaxes(title_text="绝对误差", row=1, col=2)
    fig.update_yaxes(title_text="log₁₀(绝对误差)", row=2, col=1)
    fig.update_yaxes(title_text="log₁₀(相对误差)", row=2, col=2)
    
    # 更新x轴标题
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="x", row=i, col=j)
    
    # 添加误差统计信息
    max_abs_error = np.nanmax(abs_error)
    mean_abs_error = np.nanmean(abs_error)
    max_rel_error = np.nanmax(rel_error)
    mean_rel_error = np.nanmean(rel_error)
    
    error_stats = (
        f"最大绝对误差: {max_abs_error:.2e}<br>"
        f"平均绝对误差: {mean_abs_error:.2e}<br>"
        f"最大相对误差: {max_rel_error:.2e}<br>"
        f"平均相对误差: {mean_rel_error:.2e}"
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        text=error_stats,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig