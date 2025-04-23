"""
交互式可视化模块 - 创建交互式图表
"""

import numpy as np
import sympy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_approximation(approximator, f_expr, domain=(-5, 5), num_points=1000):
    """
    绘制函数与其泰勒展开的对比图
    
    参数:
        approximator: TaylorApproximator实例
        f_expr: 原始函数表达式
        domain: 显示范围 (min, max)
        num_points: 采样点数
        
    返回:
        plotly图表对象
    """
    # 生成x值
    x_vals = np.linspace(domain[0], domain[1], num_points)
    
    # 计算原始函数值
    x_sym = list(f_expr.free_symbols)[0]
    f_func = sp.lambdify(x_sym, f_expr, 'numpy')
    
    try:
        f_vals = f_func(x_vals)
    except:
        # 逐点计算
        f_vals = np.zeros_like(x_vals)
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
            except:
                f_vals[i] = np.nan
    
    # 计算泰勒展开值
    taylor_vals = approximator.evaluate(x_vals)
    
    # 计算误差
    abs_error = np.abs(f_vals - taylor_vals)
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("函数与泰勒展开比较", "绝对误差"),
        row_heights=[0.7, 0.3]
    )
    
    # 添加原始函数
    fig.add_trace(
        go.Scatter(
            x=x_vals, 
            y=f_vals, 
            mode='lines',
            name='原始函数',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # 添加泰勒展开
    if approximator.function_type == "piecewise":
        # 对于分段函数，每段用不同颜色
        for i, (a, b, x_mid, order, _) in enumerate(approximator.segments):
            # 分段的x范围
            mask = (x_vals >= a) & (x_vals <= b)
            x_segment = x_vals[mask]
            taylor_segment = taylor_vals[mask]
            
            fig.add_trace(
                go.Scatter(
                    x=x_segment, 
                    y=taylor_segment, 
                    mode='lines',
                    name=f'泰勒展开 (区段{i+1}, 阶数={order})',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
            
            # 标记展开点
            fig.add_trace(
                go.Scatter(
                    x=[x_mid], 
                    y=[approximator.evaluate(x_mid)], 
                    mode='markers',
                    name=f'展开点 {i+1}',
                    marker=dict(size=10, symbol='x')
                ),
                row=1, col=1
            )
    else:
        # 对于普通函数
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=taylor_vals, 
                mode='lines',
                name=f'泰勒展开 (阶数={approximator.order})',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # 标记展开点
        fig.add_trace(
            go.Scatter(
                x=[approximator.x0], 
                y=[approximator.evaluate(approximator.x0)], 
                mode='markers',
                name='展开点',
                marker=dict(size=10, symbol='x', color='black')
            ),
            row=1, col=1
        )
    
    # 添加误差图
    fig.add_trace(
        go.Scatter(
            x=x_vals, 
            y=abs_error, 
            mode='lines',
            name='绝对误差',
            line=dict(color='green', width=1.5)
        ),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    fig.update_yaxes(title_text='f(x)', row=1, col=1)
    fig.update_yaxes(title_text='误差', type='log', row=2, col=1)
    fig.update_xaxes(title_text='x', row=2, col=1)
    
    return fig

def create_stepwise_visualization(f_expr, x_sym, x0, max_order=10, domain=None):
    """
    创建逐步展开的可视化
    
    参数:
        f_expr: 函数表达式
        x_sym: 符号变量
        x0: 展开点
        max_order: 最大阶数
        domain: 显示范围 (min, max)
        
    返回:
        plotly图表对象
    """
    if domain is None:
        domain = (x0 - 5, x0 + 5)
    
    # 生成x值
    x_vals = np.linspace(domain[0], domain[1], 1000)
    
    # 计算原始函数值
    f_func = sp.lambdify(x_sym, f_expr, 'numpy')
    try:
        f_vals = f_func(x_vals)
    except:
        # 逐点计算
        f_vals = np.zeros_like(x_vals)
        for i, x in enumerate(x_vals):
            try:
                f_vals[i] = float(f_expr.subs(x_sym, x))
            except:
                f_vals[i] = np.nan
    
    # 创建图表
    fig = go.Figure()
    
    # 添加原始函数
    fig.add_trace(
        go.Scatter(
            x=x_vals, 
            y=f_vals, 
            mode='lines',
            name='原始函数',
            line=dict(color='blue', width=2)
        )
    )
    
    # 计算不同阶数的展开
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    
    accumulated = 0
    trace_count = 1  # 从1开始，因为已经添加了原始函数
    term_traces = []
    accum_traces = []
    
    for n in range(max_order + 1):
        # 计算n阶导数
        try:
            if n == 0:
                dnf = f_expr.subs(x_sym, x0)
            else:
                dnf = sp.diff(f_expr, x_sym, n).subs(x_sym, x0)
            
            # 计算展开项
            from adataylor.utils import factorial
            term = dnf * (x_sym - x0)**n / factorial(n)
            
            # 累加
            accumulated += term
            
            # 转换为可计算函数
            term_func = sp.lambdify(x_sym, term, 'numpy')
            accum_func = sp.lambdify(x_sym, accumulated, 'numpy')
            
            # 计算值
            try:
                term_vals = term_func(x_vals)
                accum_vals = accum_func(x_vals)
            except:
                # 逐点计算
                term_vals = np.zeros_like(x_vals)
                accum_vals = np.zeros_like(x_vals)
                for i, x in enumerate(x_vals):
                    try:
                        term_vals[i] = float(term.subs(x_sym, x))
                        accum_vals[i] = float(accumulated.subs(x_sym, x))
                    except:
                        term_vals[i] = np.nan
                        accum_vals[i] = np.nan
            
            # 添加项
            if n > 0:  # 跳过常数项
                opacity = 0.5  # 使项在图上不那么显眼
                term_trace = go.Scatter(
                    x=x_vals, 
                    y=term_vals, 
                    mode='lines',
                    name=f'项 {n}: {sp.latex(term)}',
                    line=dict(color=colors[n % len(colors)], width=1, dash='dot'),
                    opacity=opacity,
                    visible='legendonly'  # 默认不显示
                )
                fig.add_trace(term_trace)
                term_traces.append(term_trace)
                trace_count += 1
            
            # 添加累积和
            accum_trace = go.Scatter(
                x=x_vals, 
                y=accum_vals, 
                mode='lines',
                name=f'截至{n}阶',
                line=dict(color=colors[n % len(colors)], width=2),
                visible=(n == max_order)  # 只显示最高阶和0阶
            )
            fig.add_trace(accum_trace)
            accum_traces.append(accum_trace)
            trace_count += 1
            
        except:
            # 如果计算失败，跳过
            continue
    
    # 添加展开点标记
    fig.add_trace(
        go.Scatter(
            x=[x0],
            y=[float(f_expr.subs(x_sym, x0))],
            mode='markers',
            name='展开点',
            marker=dict(size=10, symbol='x', color='black')
        )
    )
    trace_count += 1
    
    # 添加滑块 - 修改这一部分
    steps = []
    for i in range(len(accum_traces)):
        # 创建可见性列表，长度与实际trace数量相同
        visibilities = [True]  # 原始函数总是可见
        
        # 展开项轨迹的可见性
        for j in range(len(term_traces)):
            visibilities.append(j < i)  # 只显示到当前阶数的项
        
        # 累积和轨迹的可见性
        for j in range(len(accum_traces)):
            visibilities.append(j == i)  # 只显示当前阶数的累积和
        
        # 展开点始终可见
        visibilities.append(True)
        
        # 确保可见性列表长度正确
        while len(visibilities) < trace_count:
            visibilities.append(False)
        
        step = dict(
            method="update",
            args=[{"visible": visibilities},
                  {"title": f"泰勒展开 (展开点 x₀={x0}, 阶数={i})"}],
            label=str(i)
        )
        steps.append(step)
    
    sliders = [dict(
        active=max_order,
        steps=steps,
        currentvalue={"prefix": "阶数: "},
        pad={"t": 50}
    )]
    
    fig.update_layout(sliders=sliders)
    
    return fig

def error_analysis_visualization(approximator, f_expr, domain=(-5, 5), num_points=1000):
    """
    创建误差分析可视化
    
    参数:
        approximator: TaylorApproximator实例
        f_expr: 原始函数表达式
        domain: 显示范围 (min, max)
        num_points: 采样点数
        
    返回:
        plotly图表对象
    """
    # 生成x值
    x_vals = np.linspace(domain[0], domain[1], num_points)
    
    # 计算误差
    error_dict = approximator.compute_error(f_expr, x_vals)
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("绝对误差", "相对误差"),
        row_heights=[0.5, 0.5]
    )
    
    # 添加绝对误差
    fig.add_trace(
        go.Scatter(
            x=x_vals, 
            y=error_dict["abs_error"], 
            mode='lines',
            name='绝对误差',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # 添加相对误差
    fig.add_trace(
        go.Scatter(
            x=x_vals, 
            y=error_dict["rel_error"] * 100, 
            mode='lines',
            name='相对误差 (%)',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    # 标记展开点
    if approximator.function_type != "piecewise":
        # 对于普通函数
        fig.add_trace(
            go.Scatter(
                x=[approximator.x0], 
                y=[0], 
                mode='markers',
                name='展开点',
                marker=dict(size=10, symbol='x', color='black')
            ),
            row=1, col=1
        )
    else:
        # 对于分段函数，标记各个分段的展开点
        for i, (_, _, x_mid, _, _) in enumerate(approximator.segments):
            fig.add_trace(
                go.Scatter(
                    x=[x_mid], 
                    y=[0], 
                    mode='markers',
                    name=f'展开点 {i+1}',
                    marker=dict(size=10, symbol='x')
                ),
                row=1, col=1
            )
    
    # 更新布局
    fig.update_layout(
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    fig.update_yaxes(title_text='绝对误差', type='log', row=1, col=1)
    fig.update_yaxes(title_text='相对误差 (%)', type='log', row=2, col=1)
    fig.update_xaxes(title_text='x', row=2, col=1)
    
    # 添加误差统计信息
    annotations = [
        dict(
            x=0.5, y=1.12,
            xref="paper", yref="paper",
            text=f"最大绝对误差: {error_dict['max_absolute']:.2e} | 平均绝对误差: {error_dict['mean_absolute']:.2e}",
            showarrow=False,
            font=dict(size=12)
        ),
        dict(
            x=0.5, y=0.55,
            xref="paper", yref="paper",
            text=f"最大相对误差: {error_dict['max_relative']*100:.2e}% | 平均相对误差: {error_dict['mean_relative']*100:.2e}%",
            showarrow=False,
            font=dict(size=12)
        )
    ]
    
    fig.update_layout(annotations=annotations)
    
    return fig