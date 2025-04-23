"""
可视化模块 - 为泰勒展开逼近提供静态和交互式可视化功能
"""

from .plotter import plot_taylor_comparison, plot_error_distribution
from .interactive import plot_approximation, create_stepwise_visualization, error_analysis_visualization

__all__ = [
    'plot_taylor_comparison', 
    'plot_error_distribution',
    'plot_approximation', 
    'create_stepwise_visualization',
    'error_analysis_visualization'
]