import numpy as np
import sympy as sp
from sympy import symbols, Derivative, factorial, lambdify
from .adaptive import auto_order
from .wavelet import wavelet_preprocess
from .error import compute_error

class TaylorApproximator:
    """泰勒展开自适应逼近器"""
    
    def __init__(self, max_order=15, epsilon=1e-8):
        """
        初始化泰勒逼近器
        
        参数:
            max_order (int): 最大展开阶数
            epsilon (float): 目标误差阈值
        """
        self.max_order = max_order
        self.epsilon = epsilon
        self.x = symbols('x')
        self.expansion_point = 0
        self.coefficients = []
        self.order = 0
        self.function_type = "smooth"  # 'smooth', 'piecewise', 'oscillatory'
        
    def analyze_function(self, f_expr):
        """分析输入函数类型并确定处理策略"""
        try:
            # 尝试对函数求高阶导数，检查是否在某些点不可导
            derivatives = [f_expr]
            for i in range(1, 4):  # 检查前3阶导数
                derivatives.append(sp.diff(derivatives[-1], self.x))
            
            # 检查函数是否有不连续点或奇异点
            singularities = sp.solve(1/f_expr, self.x)
            
            if singularities:
                self.function_type = "piecewise"
                return "piecewise"
                
            # 检查函数是否高度振荡(通过二阶导数符号变化频率)
            x_vals = np.linspace(-5, 5, 100)
            f_second_deriv = lambdify(self.x, derivatives[2], "numpy")
            sign_changes = np.sum(np.diff(np.signbit(f_second_deriv(x_vals))))
            
            if sign_changes > 10:  # 如果二阶导数符号变化超过10次，视为振荡函数
                self.function_type = "oscillatory"
                return "oscillatory"
                
            return "smooth"
        except Exception as e:
            print(f"函数分析出错: {e}")
            return "smooth"  # 默认为光滑函数
    
    def fit(self, f_expr, x0=0, domain=(-5, 5)):
        """
        拟合函数的泰勒展开
        
        参数:
            f_expr: sympy表达式
            x0: 展开点
            domain: 函数定义域
        """
        self.expansion_point = x0
        self.function_type = self.analyze_function(f_expr)
        
        if self.function_type == "smooth":
            # 直接泰勒展开
            self.order = auto_order(f_expr, self.x, x0, self.epsilon, self.max_order)
            self._compute_taylor_coefficients(f_expr, x0, self.order)
        
        elif self.function_type == "piecewise":
            # 对分段函数进行处理
            break_points = self._detect_breakpoints(f_expr, domain)
            segments = wavelet_preprocess(f_expr, self.x, break_points)
            # 每段分别处理并存储
            self.segments = []
            for segment in segments:
                x_mid = (segment[0] + segment[1]) / 2
                order = auto_order(f_expr, self.x, x_mid, self.epsilon, self.max_order)
                coeffs = self._compute_taylor_coefficients(f_expr, x_mid, order, return_only=True)
                self.segments.append((segment, x_mid, coeffs, order))
        
        elif self.function_type == "oscillatory":
            # 对振荡函数使用更高阶数和分段处理
            self.order = min(self.max_order, auto_order(f_expr, self.x, x0, self.epsilon/10, self.max_order))
            self._compute_taylor_coefficients(f_expr, x0, self.order)
        
        return self
    
    def _compute_taylor_coefficients(self, f_expr, x0, order, return_only=False):
        """计算泰勒系数"""
        coefficients = []
        for n in range(order + 1):
            # 计算n阶导数在x0处的值
            nth_derivative = Derivative(f_expr, (self.x, n))
            coef = nth_derivative.doit().subs(self.x, x0) / factorial(n)
            coefficients.append(coef)
        
        if return_only:
            return coefficients
        else:
            self.coefficients = coefficients
    
    def _detect_breakpoints(self, f_expr, domain):
        """检测函数的断点或奇异点"""
        # 尝试解析求解
        try:
            singularities = sp.solve(1/f_expr, self.x)
            valid_points = [point for point in singularities 
                          if point.is_real and domain[0] <= float(point) <= domain[1]]
            
            if valid_points:
                # 添加域边界
                points = [domain[0]] + sorted(float(point) for point in valid_points) + [domain[1]]
                return points
        except:
            pass
        
        # 如果解析方法失败，使用数值方法
        x_vals = np.linspace(domain[0], domain[1], 1000)
        f_numpy = lambdify(self.x, f_expr, "numpy")
        try:
            y_vals = f_numpy(x_vals)
            # 寻找值变化剧烈的点
            derivatives = np.abs(np.diff(y_vals))
            # 找出变化超过阈值的点
            threshold = np.percentile(derivatives, 99)
            potential_breaks = x_vals[1:][derivatives > threshold]
            
            if len(potential_breaks) > 0:
                # 聚类相近的点
                breaks = [potential_breaks[0]]
                for pt in potential_breaks[1:]:
                    if pt - breaks[-1] > (domain[1]-domain[0])/20:  # 最小间隔
                        breaks.append(pt)
                
                return [domain[0]] + breaks + [domain[1]]
        except:
            pass
            
        # 默认只在边界和中点展开
        return [domain[0], (domain[0]+domain[1])/2, domain[1]]
    
    def evaluate(self, x_values):
        """评估拟合的泰勒展开"""
        if self.function_type == "smooth" or self.function_type == "oscillatory":
            return self._evaluate_taylor(x_values, self.expansion_point, self.coefficients)
        
        elif self.function_type == "piecewise":
            # 对分段函数，根据x值选择对应的分段
            result = np.zeros_like(x_values, dtype=float)
            for i, x in enumerate(x_values):
                for (segment, x0, coeffs, _) in self.segments:
                    if segment[0] <= x <= segment[1]:
                        result[i] = self._evaluate_taylor_single(x, x0, coeffs)
                        break
            return result
    
    def _evaluate_taylor(self, x_values, x0, coefficients):
        """评估单点泰勒展开"""
        result = np.zeros_like(x_values, dtype=float)
        for i, x in enumerate(x_values):
            result[i] = self._evaluate_taylor_single(x, x0, coefficients)
        return result
    
    def _evaluate_taylor_single(self, x, x0, coefficients):
        """计算单个点的泰勒展开值"""
        result = 0.0
        dx = x - x0
        for n, coef in enumerate(coefficients):
            result += float(coef) * (dx ** n)
        return result
    
    def compute_error(self, f_expr, x_values):
        """计算近似误差"""
        f_numpy = lambdify(self.x, f_expr, "numpy")
        true_values = f_numpy(x_values)
        approx_values = self.evaluate(x_values)
        
        abs_error = np.abs(true_values - approx_values)
        rel_error = abs_error / (np.abs(true_values) + 1e-10)
        
        return {
            "absolute": abs_error,
            "relative": rel_error,
            "max_absolute": np.max(abs_error),
            "max_relative": np.max(rel_error),
            "mean_absolute": np.mean(abs_error),
            "mean_relative": np.mean(rel_error)
        }
    
    def get_symbolic_expansion(self):
        """获取符号形式的泰勒展开"""
        if self.function_type == "smooth" or self.function_type == "oscillatory":
            expansion = 0
            x = self.x
            x0 = self.expansion_point
            
            for n, coef in enumerate(self.coefficients):
                expansion += coef * (x - x0)**n
                
            return expansion
        else:
            # 对于分段函数，返回分段表达式
            pieces = []
            x = self.x
            
            for (segment, x0, coeffs, _) in self.segments:
                cond = sp.And(x >= segment[0], x <= segment[1])
                expr = sum(coef * (x - x0)**n for n, coef in enumerate(coeffs))
                pieces.append((expr, cond))
                
            return sp.Piecewise(*pieces)