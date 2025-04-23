"""
AdaTaylor - 泰勒展开自适应逼近器包

这个包提供了基于泰勒展开的函数逼近工具，包括:
1. 自适应阶数选择
2. 分段函数处理
3. 小波-泰勒混合逼近
4. 误差分析与控制
"""

from .approximator import TaylorApproximator
from .adaptive import auto_order_selection
from .error import error_analysis, compute_remainder_bound
from .wavelet import wavelet_denoise, wavelet_taylor_hybrid
from .utils import detect_singularities, factorial, multiindex_generator

__version__ = "0.1.0"
__author__ = "Your Name"