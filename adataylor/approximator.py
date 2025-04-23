"""
泰勒展开逼近器模块 - 核心逼近算法实现
"""

import numpy as np
import sympy as sp
from .adaptive import auto_order
from .utils import factorial, detect_singularities, function_type_analysis
from .error import compute_remainder_bound, error_analysis
from .wavelet import wavelet_denoise, wavelet_taylor_hybrid

class TaylorApproximator:
    """
    泰勒展开自适应逼近器
    
    提供自适应阶数选择、分段函数处理和误差分析功能
    """
    
    def __init__(self, max_order=15, epsilon=1e-8):
        """
        初始化逼近器
        
        参数:
            max_order: 最大允许阶数
            epsilon: 误差容限
        """
        self.max_order = max_order
        self.epsilon = epsilon
        self.reset()
    
    def reset(self):
        """重置内部状态"""
        self.fitted = False
        self.order = None
        self.x0 = None
        self.domain = None
        self.expansion = None
        self.function_type = None
        self.segments = []
        self.symbolic_expansion = None
    
    def fit(self, f_expr, x0=0, domain=(-5, 5), adaptive=True):
        """
        拟合函数的泰勒展开
        
        参数:
            f_expr: sympy表达式或可调用函数
            x0: 展开点
            domain: 考虑的区域范围 (min, max)
            adaptive: 是否使用自适应阶数
            
        返回:
            self，以支持链式调用
        """
        self.reset()
        self.x0 = x0
        self.domain = domain
        
        # 如果输入是可调用函数而不是sympy表达式，创建一个符号化版本
        if not isinstance(f_expr, sp.Expr):
            x = sp.symbols('x')
            try:
                # 尝试将函数映射到符号表达式
                # 这通常只适用于简单函数
                # 更复杂的函数需要直接提供sympy表达式
                values = [f_expr(x0 + 0.1 * i) for i in range(-10, 11)]
                # 使用多项式插值
                f_expr = sp.interpolate(values, x)
            except:
                raise ValueError("无法将函数转换为符号表达式，请直接提供sympy表达式")
        
        # 获取函数的符号变量
        x_sym = list(f_expr.free_symbols)
        if len(x_sym) == 0:
            # 常函数
            self.function_type = "constant"
            self.order = 0
            self.expansion = f_expr
            self.symbolic_expansion = f_expr
            self.fitted = True
            return self
        elif len(x_sym) > 1:
            raise ValueError("目前只支持单变量函数")
        x_sym = x_sym[0]
        
        # 分析函数类型
        func_analysis = function_type_analysis(f_expr, x_sym, domain)
        self.function_type = func_analysis["type"]
        
        # 根据函数类型选择不同的逼近策略
        if self.function_type == "piecewise":
            # 对分段函数使用分段泰勒展开
            breakpoints = [domain[0]] + func_analysis.get("breakpoints", []) + [domain[1]]
            self._fit_piecewise(f_expr, x_sym, breakpoints, adaptive)
        elif self.function_type == "oscillatory":
            # 对高频振荡函数可能需要特殊处理
            # 这里简单地增加最小阶数
            min_order = 3
            if adaptive:
                self.order = max(min_order, auto_order(f_expr, x_sym, x0, self.epsilon, 
                                                      self.max_order, domain))
            else:
                self.order = self.max_order
            self._compute_expansion(f_expr, x_sym, x0, self.order)
        else:
            # 对光滑函数使用标准泰勒展开
            if adaptive:
                self.order = auto_order(f_expr, x_sym, x0, self.epsilon, 
                                       self.max_order, domain)
            else:
                self.order = self.max_order
            self._compute_expansion(f_expr, x_sym, x0, self.order)
        
        self.fitted = True
        return self
    
    def _fit_piecewise(self, f_expr, x_sym, breakpoints, adaptive):
        """处理分段函数"""
        self.segments = []
        
        # 确保断点是有序的
        breakpoints = sorted(list(set(breakpoints)))
        
        # 对每个分段单独进行泰勒展开
        for i in range(len(breakpoints) - 1):
            a, b = breakpoints[i], breakpoints[i+1]
            # 选择分段中点作为展开点
            x_mid = (a + b) / 2
            
            # 选择阶数
            order = self.max_order
            if adaptive:
                try:
                    order = auto_order(f_expr, x_sym, x_mid, self.epsilon, 
                                      self.max_order, (a, b))
                except:
                    # 如果自适应选择失败，使用最大阶数
                    pass
            
            # 计算展开式
            expansion = self._compute_expansion(f_expr, x_sym, x_mid, order, add_to_self=False)
            
            # 记录分段信息
            self.segments.append((a, b, x_mid, order, expansion))
        
        # 创建分段展开式
        self._create_piecewise_expansion(x_sym)
    
    def _compute_expansion(self, f_expr, x_sym, x0, order, add_to_self=True):
        """
        计算泰勒展开式
        
        参数:
            f_expr: sympy表达式
            x_sym: 符号变量
            x0: 展开点
            order: 展开阶数
            add_to_self: 是否将结果存储到self
            
        返回:
            展开式
        """
        expansion = 0
        
        # 计算各阶导数和系数
        for n in range(order + 1):
            try:
                # 计算n阶导数在x0点的值
                if n == 0:
                    dnf = f_expr.subs(x_sym, x0)
                else:
                    dnf = sp.diff(f_expr, x_sym, n).subs(x_sym, x0)
                
                # 添加到展开式
                term = dnf * (x_sym - x0)**n / factorial(n)
                expansion += term
            except:
                # 如果计算失败，使用当前展开式
                break
        
        # 检查定义域
        try:
            # 对于对数函数，检测定义域边界
            if 'log' in str(f_expr):
                # 找出对数函数的参数
                log_args = []
                for arg in sp.preorder_traversal(f_expr):
                    if isinstance(arg, sp.log):
                        log_args.append(arg.args[0])
                
                # 针对每个对数参数添加定义域检查
                for arg in log_args:
                    # 计算参数为0的x值
                    try:
                        zero_points = sp.solve(arg, x_sym)
                        for point in zero_points:
                            # 如果零点在展开区间内，调整展开域或警告
                            if self.domain[0] <= float(point) <= self.domain[1]:
                                print(f"警告: 函数在 x = {point} 处的对数参数为零")
                    except:
                        pass
        except:
            pass
        
        if add_to_self:
            self.expansion = expansion
            self.symbolic_expansion = expansion
        
        return expansion
    
    def _create_piecewise_expansion(self, x_sym):
        """为分段函数创建Piecewise表达式"""
        if not self.segments:
            return
        
        pieces = []
        for a, b, x_mid, order, expansion in self.segments:
            # 创建条件表达式
            condition = sp.And(x_sym >= a, x_sym < b)
            # 如果是最后一段，包括右端点
            if b == self.domain[1]:
                condition = sp.And(x_sym >= a, x_sym <= b)
            
            pieces.append((expansion, condition))
        
        # 创建分段函数
        self.symbolic_expansion = sp.Piecewise(*pieces)
    
    def evaluate(self, x_values):
        """
        在给定点上评估泰勒展开
        
        参数:
            x_values: 单个值或数组
            
        返回:
            泰勒展开在给定点的值
        """
        if not self.fitted:
            raise ValueError("必须先调用fit方法")
        
        # 如果是单个值
        if np.isscalar(x_values):
            return self._evaluate_single(x_values)
        
        # 对数组，向量化处理
        return np.array([self._evaluate_single(x) for x in x_values])
    
    def _evaluate_single(self, x):
        """评估单个点"""
        if self.function_type == "piecewise":
            # 找到包含x的分段
            for a, b, x_mid, order, expansion in self.segments:
                if a <= x <= b:
                    x_sym = list(expansion.free_symbols)[0]
                    return float(expansion.subs(x_sym, x))
            # 如果找不到匹配的分段，返回域中最近的点
            if x < self.domain[0]:
                a, b, x_mid, order, expansion = self.segments[0]
                x_sym = list(expansion.free_symbols)[0]
                return float(expansion.subs(x_sym, self.domain[0]))
            else:
                a, b, x_mid, order, expansion = self.segments[-1]
                x_sym = list(expansion.free_symbols)[0]
                return float(expansion.subs(x_sym, self.domain[1]))
        else:
            # 对常规函数，直接计算
            x_sym = list(self.expansion.free_symbols)[0]
            return float(self.expansion.subs(x_sym, x))
    
    def get_symbolic_expansion(self):
        """获取符号形式的展开式"""
        if not self.fitted:
            raise ValueError("必须先调用fit方法")
        return self.symbolic_expansion
    
    def compute_error(self, f_expr, x_values):
        """
        计算泰勒展开在给定点上的误差
        
        参数:
            f_expr: 原始函数表达式
            x_values: 评估点数组
            
        返回:
            包含误差统计的字典
        """
        if not self.fitted:
            raise ValueError("必须先调用fit方法")
        
        # 计算原始函数值
        x_sym = list(f_expr.free_symbols)[0]
        f_func = sp.lambdify(x_sym, f_expr, 'numpy')
        try:
            f_values = f_func(x_values)
        except:
            # 逐点计算
            f_values = np.zeros_like(x_values)
            for i, x in enumerate(x_values):
                try:
                    f_values[i] = float(f_expr.subs(x_sym, x))
                except:
                    f_values[i] = np.nan
        
        # 计算逼近值
        approx_values = self.evaluate(x_values)
        
        # 计算误差
        abs_error = np.abs(f_values - approx_values)
        rel_error = abs_error / (np.abs(f_values) + 1e-15)  # 避免除零
        
        # 过滤掉NaN值
        valid_abs = abs_error[~np.isnan(abs_error)]
        valid_rel = rel_error[~np.isnan(rel_error)]
        
        if len(valid_abs) == 0:
            return {
                "max_absolute": np.nan,
                "mean_absolute": np.nan,
                "max_relative": np.nan,
                "mean_relative": np.nan,
                "valid_points": 0,
                "total_points": len(x_values)
            }
        
        # 返回误差统计
        return {
            "max_absolute": np.max(valid_abs),
            "mean_absolute": np.mean(valid_abs),
            "max_relative": np.max(valid_rel),
            "mean_relative": np.mean(valid_rel),
            "valid_points": len(valid_abs),
            "total_points": len(x_values),
            "abs_error": abs_error,
            "rel_error": rel_error
        }