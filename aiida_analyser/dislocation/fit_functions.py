from __future__ import annotations

import re
import numpy

from aiida_dislocation.tools import (
    A1GlidingSystem,
    A2GlidingSystem,
    B1GlidingSystem,
    B2GlidingSystem,
    C1bGlidingSystem,
    L21GlidingSystem,
)


def formula_to_latex(formula):
    latex_formula = re.sub(r'(\d+)', r'_{\1}', formula)
    return rf"${latex_formula}$"

def segmented_linear_function(x, point1, point2, point3):
    """
    分段线性背景函数：
    在 x1 <= x <= x2 时，从 y1 线性爬升到 y2
    在 x2 <= x <= x3 时，从 y2 线性爬升到 y3
    """
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    return numpy.where(
        x <= x2,
        y1 + ((y2 - y1) / (x2 - x1)) * (x - x1),
        y2 + ((y3 - y2) / (x3 - x2)) * (x - x2)
    )

def sine_power_expansion(x, cGs, period=1/2):
    """Generic sine series expansion: sum_{i=1}^N cG_i * sin(pi*x/period)**(2*i)."""
    sin_sq = numpy.sin(numpy.pi * x / period)**2
    result = numpy.zeros_like(x, dtype=float)
    for i, cG in enumerate(cGs, 1):
        result += cG * (sin_sq**i)
    return result

def cosine_power_expansion(x, cGs, period=1/2):
    """Generic cosine series expansion: sum_{i=1}^N cG_i * cos(pi*x/period)**(2*i)."""
    cos_sq = numpy.cos(numpy.pi * x / period)**2
    result = numpy.zeros_like(x, dtype=float)
    for i, cG in enumerate(cGs, 1):
        result += cG * (cos_sq**i)
    return result

def sine_expansion(x, cGs, period=1/2):
    """Generic sine series expansion: sum_{i=1}^N cG_i * sin(pi*x/period)**(2*i)."""
    result = numpy.zeros_like(x, dtype=float)
    for i, cG in enumerate(cGs, 1):
        result += cG * numpy.sin(i*numpy.pi * x / period)
    return result

def cosine_expansion(x, cGs, period=1/2, offset=0.0):
    """Generic cosine series expansion: sum_{i=1}^N cG_i * cos(pi*x/period)**(2*i)."""
    result = numpy.zeros_like(x, dtype=float)
    for i, cG in enumerate(cGs, 1):
        result += cG * (numpy.cos(i*numpy.pi * x / period) + offset)
    return result

def gamma_esf1(x, cGs, g_isf, g_esf):
    """
    模型一：分段线性背景 + 高次正弦幂次展开（周期为 1/2）。
    由于 sin_sq 在 0, 0.5, 1.0 处天生为 0，控制点由分段线性背景严格掌控。
    """
    bg = segmented_linear_function(x, (0.0, 0.0), (0.5, g_isf), (1.0, g_esf))
    val = sine_power_expansion(x, cGs, period=1/2)
    return bg + val

def gamma_esf2(x, cGs, g_isf, g_esf):
    """
    模型二：分段线性背景 + 修复漂移后的傅里叶共享系数展开。
    """
    half = len(cGs) // 2
    cGs_sin = cGs[:half]
    cGs_cos = cGs[half:]

    bg = segmented_linear_function(x, (0.0, 0.0), (0.5, g_isf), (1.0, g_esf))
    # 调用加了边界锁的傅里叶函数
    expansion = sine_expansion(x, cGs_sin, period=1/2) + cosine_expansion(x, cGs_cos, period=1/4, offset=-1)
    return bg + expansion

def gamma_isf(x, cGs, g_isf):
    """
    Calculates the value for the first region: 0 <= x <= 1
    Formula: Expansion + gamma_ISF * x
    """
    return sine_expansion(x, cGs) + g_isf * x


def gamma_usf(x, cGs):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: sine expansion with period=1
    """
    return sine_expansion(x, cGs, period=1)

def gamma_usf2(x, cGs):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: sine expansion with period=2
    """
    half = len(cGs) // 2
    cGs_sin = cGs[:half]
    cGs_cos = cGs[half:]
    return sine_expansion(x, cGs_sin, period=1) + cosine_expansion(x, cGs_cos, period=1/2, offset=-1)

def gamma_usf_symmetric(x, cGs):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: sine expansion with period=2
    """
    return sine_expansion(x, cGs, period=2)


def gamma_usf2_symmetric(x, cGs):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: sine expansion with period=2
    """
    half = len(cGs) // 2
    cGs_sin = cGs[:half]
    cGs_cos = cGs[half:]
    return sine_expansion(x, cGs_sin, period=2) + cosine_expansion(x, cGs_cos, period=1, offset=-1)


fit_function_map = {
    'A1': {
        'gliding_system': A1GlidingSystem,
        '100': {'100' : gamma_usf},
        '011': {'010' : gamma_usf},
        '111': {'110' : gamma_esf2},
    },
    'A2': {
        'gliding_system': A2GlidingSystem,
        '100': {'100' : gamma_usf},
        '011': {'100' : gamma_usf},
        '111': {'110' : gamma_esf2},
    },
    'B1': {
        'gliding_system': B1GlidingSystem,
        '100': {'100' : gamma_usf},
        '011': {'100' : gamma_usf, '010': gamma_usf},
        '111': {'110' : gamma_esf2},
    },
    'B2': {
        'gliding_system': B2GlidingSystem,
        '100': {'100' : gamma_usf},
        '011': {'100' : gamma_usf, '010': gamma_usf, '110': gamma_usf},
        '111': {'110' : gamma_isf},
    },
    'C1_b': {
        'gliding_system': C1bGlidingSystem,
        '100': {'110' : gamma_usf},
        '011': {'100' : gamma_usf2_symmetric, '010': gamma_usf2_symmetric, '210': gamma_usf2_symmetric},
        '111': {'110' : gamma_esf2},
    },
    'L2_1': {
        'gliding_system': L21GlidingSystem,
        '100': {'110': gamma_usf_symmetric},
        '011': {'100': gamma_usf_symmetric, '010': gamma_usf2_symmetric, '210': gamma_usf2_symmetric},
        '111': {'110' : gamma_esf2},
    },
}


def fit_gsfe(func, x, y, nsteps, x_plot, is_mJ=False, **kwargs):
    """
    Fit a single GSFE curve using scipy's curve_fit and extract parameters.

    Args:
        func: The fit function (e.g. gamma_isf, gamma_esf, gamma_usf, gamma_usf_symmetric).
        x: Original x values (numpy array).
        y: Original y values (numpy array).
        nsteps: Number of steps of the gliding plane.
        x_plot: High-resolution x values for plotting.
        is_mJ: If True, multiply outputs of gamma_isf by 1000.
        **kwargs: Additional parameters like order.

    Returns:
        popt: Optimized parameters from curve_fit.
        y_fit: Fitted values corresponding to x_plot.
        y_fit_orig: Fitted values corresponding to x.
        results: Dictionary containing extracted parameters.
    """
    from scipy.optimize import curve_fit
    
    results = {}

    if func == gamma_isf:
        b = y[nsteps]
        order = kwargs.get('order', 1)
        popt, _ = curve_fit(lambda x, *cGs: func(x, cGs, b), x, y, p0=[0.1] * order, maxfev=100000)
        y_fit = func(x_plot, popt, b)
        y_fit_orig = func(x, popt, b)
        x_max = numpy.arcsin(-b / numpy.pi / popt[0]) / 2 * numpy.pi if popt[0] != 0 else 0.5
        
        factor = 1000.0 if is_mJ else 1.0
        results['isf'] = b * factor
        results['usf'] = func(x_max, popt, b) * factor

    elif func == gamma_esf1:
        g_isf = y[nsteps]
        g_esf = y[2 * nsteps]
        order = kwargs.get('order', 4)
        popt, _ = curve_fit(lambda x, *cGs: func(x, cGs, g_isf, g_esf), x, y, p0=[0.1] * order, maxfev=1000000)
        y_fit = func(x_plot, popt, g_isf, g_esf)
        y_fit_orig = func(x, popt, g_isf, g_esf)
        results['usf'] = numpy.max(y_fit[:250]) * 1000.0
        results['isf'] = g_isf * 1000.0
        results['ut'] = numpy.max(y_fit[250:]) * 1000.0
        results['esf'] = g_esf * 1000.0

    elif func == gamma_esf2:
        g_isf = y[nsteps]
        g_esf = y[2 * nsteps]
        order = kwargs.get('order', 2)
        # 估算你曲线的大致能垒高度（用来做初猜的量级缩放，防止不同材料失真）
        approx_height = numpy.max(y) - g_isf
        if approx_height <= 0:
            approx_height = g_isf if g_isf > 0 else 0.1

        # 构建两组独立的初猜参数
        p0_sin = []  # 正弦组：全给正值，负责提供能垒基础高度
        p0_cos = []  # 余弦组：全给负值，负责把 0.25 和 0.75 处的刚性碰撞肩膀顶出来
        
        for i in range(1, order + 1):
            p0_sin.append((approx_height * 0.4) / i)
            p0_cos.append(-(approx_height * 0.2) / i)

        # 将两组初猜平铺合并：前段全为正弦，后段全为余弦
        p0 = p0_sin + p0_cos

        # 加上 bounds 物理保护，限制余弦项（后半段）不要过度正向震荡导致两头下塌
        lower_bounds = [-numpy.inf] * order + [-approx_height * 2.0] * order
        upper_bounds = [numpy.inf] * order + [approx_height * 0.5] * order
        bounds = (lower_bounds, upper_bounds)
        popt, _ = curve_fit(
            lambda x_data, *params: func(x_data, params, g_isf, g_esf), 
            x, y, 
            p0=p0, 
            bounds=bounds,
            maxfev=1000000
        )
        y_fit = func(x_plot, popt, g_isf, g_esf)
        y_fit_orig = func(x, popt, g_isf, g_esf)
        results['usf'] = numpy.max(y_fit[:250]) * 1000.0
        results['isf'] = g_isf * 1000.0
        results['ut'] = numpy.max(y_fit[250:]) * 1000.0
        results['esf'] = g_esf * 1000.0

    elif func == gamma_usf:
        order = kwargs.get('order', 4)
        popt, _ = curve_fit(lambda x, *cGs: func(x, cGs), x, y, p0=[0.1] * order, maxfev=100000)
        y_fit = func(x_plot, popt)
        y_fit_orig = func(x, popt)
        results['usf'] = numpy.max(y_fit) * 1000.0

    elif func == gamma_usf_symmetric:
        order = kwargs.get('order', 2)
        popt, _ = curve_fit(lambda x, *cGs: func(x, cGs), x, y, p0=[0.1] * order, maxfev=100000)
        y_fit = func(x_plot, popt)
        y_fit_orig = func(x, popt)
        results['usf'] = numpy.max(y_fit) * 1000.0

    elif func in (gamma_usf2, gamma_usf2_symmetric):
        order = kwargs.get('order', 2)
        approx_height = numpy.max(y)
        if approx_height <= 0:
            approx_height = g_isf if g_isf > 0 else 0.1

        # 构建两组独立的初猜参数
        p0_sin = []  # 正弦组：全给正值，负责提供能垒基础高度
        p0_cos = []  # 余弦组：全给负值，负责把 0.25 和 0.75 处的刚性碰撞肩膀顶出来
        
        for i in range(1, order + 1):
            p0_sin.append((approx_height * 0.4) / i)
            p0_cos.append(-(approx_height * 0.2) / i)

        # 将两组初猜平铺合并：前段全为正弦，后段全为余弦
        p0 = p0_sin + p0_cos

        # 加上 bounds 物理保护，限制余弦项（后半段）不要过度正向震荡导致两头下塌
        lower_bounds = [-numpy.inf] * order + [-approx_height * 2.0] * order
        upper_bounds = [numpy.inf] * order + [approx_height * 0.5] * order
        bounds = (lower_bounds, upper_bounds)
        popt, _ = curve_fit(lambda x, *cGs: func(x, cGs), x, y, p0=p0, bounds=bounds, maxfev=100000)
        y_fit = func(x_plot, popt)
        y_fit_orig = func(x, popt)
        results['usf'] = numpy.max(y_fit) * 1000.0
    else:
        raise ValueError(f"Unsupported fit function: {func.__name__ if hasattr(func, '__name__') else func}")

    return popt, y_fit, y_fit_orig, results
