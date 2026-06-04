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
    在 x1   <= x <= x2 时，从 y1 线性爬升到 y2
    在 x2   <= x <= x3 时，从 y2 线性爬升到 y3
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
    """Generic sine series expansion: sum_{i=1}^N cG_i * sin(pi*x)**(2*i)."""
    sin_sq = numpy.sin(numpy.pi * x / period)**2
    result = numpy.zeros_like(x, dtype=float)
    for i, cG in enumerate(cGs, 1):
        result += cG * (sin_sq**i)
    return result

def cosine_power_expansion(x, cGs, period=1/2):
    """Generic cosine series expansion: sum_{i=1}^N cG_i * cos(pi*x)**(2*i)."""
    cos_sq = numpy.cos(numpy.pi * x / period)**2
    result = numpy.zeros_like(x, dtype=float)
    for i, cG in enumerate(cGs, 1):
        result += cG * (cos_sq**i)
    return result

def sine_expansion(x, cGs, period=1/2):
    """Generic sine series expansion: sum_{i=1}^N cG_i * sin(pi*x/period * i)."""
    result = numpy.zeros_like(x, dtype=float)
    for i, cG in enumerate(cGs, 1):
        result += cG *  numpy.sin(numpy.pi * x / period * i)
    return result

def cosine_expansion(x, cGs, period=1/2):
    """Generic cosine series expansion: sum_{i=1}^N cG_i * cos(pi*x)**(2*i)."""
    result = numpy.zeros_like(x, dtype=float)
    for i, cG in enumerate(cGs, 1):
        result += cG *  numpy.cos(numpy.pi * x / period * i)
    return result
    
def fourier_expansion(x, cGs, period=1/2):
    """
    sum_{i=1}^N cG_i * sin(pi*x/period * i) + sum_{i=1}^N cG_i * cos(pi*x/period * i)
    """
    result = numpy.zeros_like(x, dtype=float)
    result += sine_expansion(x, cGs, period)
    result += cosine_expansion(x, cGs, period)
    return result

def gamma_esf1(x, cGs, g_isf, g_esf):
    """
    模型一：分段线性背景 + 高次正弦幂次展开（周期为 1/2）。
    由于 period=1/2，高次项在 0, 0.5, 1.0 处天然为 0，
    因此三大控制点处的数值完全由分段线性背景接管。
    """
    bg = segmented_linear_function(x, (0.0, 0.0), (0.5, g_isf), (1.0, g_esf))
    val = sine_power_expansion(x, cGs, period=1/2)
    return bg + val

def gamma_esf2(x, cGs, g_isf, g_esf):
    """
    模型二：分段线性背景 + 传统傅里叶展开（非平方）。
    利用 sin^2(2*pi*x) 作为全域归零调制锁，确保余弦项在 0, 0.5, 1.0 处完全归零，
    将控制权完美留给分段线性背景。
    """
    bg = segmented_linear_function(x, (0.0, 0.0), (0.5, g_isf), (1.0, g_esf))
    expansion = fourier_expansion(x, cGs, period=1/2)
    
    return bg + expansion

def gamma_isf(x, cGs, g_isf):
    """
    Calculates the value for the first region: 0 <= x <= 1
    Formula: Expansion + gamma_ISF * x
    """
    return sine_expansion(x, cGs) + g_isf * x


def gamma_esf(x, cGs, b, c):
    """
    Calculates the value for the second region: 0 < x <= 1
    Formula: Expansion + piecewise linear terms
    """
    val = sine_expansion(x, cGs)
    return numpy.where(
        x <= 1/2,
        val + b * x,
        val + 1/2*b+(x - 1/2)*c
    )

def gamma_esf(x, cGs, g_isf, g_esf):
    """
    模型一：正弦幂次级数展开。
    通过 sin^2(2*pi*x) 调制，使其在 x=0, 0.5, 1.0 处天然归零，
    完美保护 g_isf 和 g_esf 的物理边界。
    """
    # 基础物理背景
    bg = _get_esf_background(x, g_isf, g_esf)
    
    # 构建归零基底：sin^2(2*pi*x) 在 0, 0.5, 1.0 处恒为 0
    sin_sq_base = numpy.sin(2.0 * numpy.pi * x)**2
    
    # 幂次级数扩展
    expansion = numpy.zeros_like(x, dtype=float)
    for i, cG in enumerate(cGs, 1):
        expansion += cG * (sin_sq_base ** i)
        
    return bg + expansion

def gamma_usf(x, cGs):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: sine expansion with period=1
    """
    return sine_expansion(x, cGs, period=1)


def gamma_usf_symmetric(x, cGs):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: sine expansion with period=2
    """
    return sine_expansion(x, cGs, period=2)


fit_function_map = {
    'A1': {
        'gliding_system': A1GlidingSystem,
        '100': {'100' : gamma_usf},
        '011': {'010' : gamma_usf},
        '111': {'110' : gamma_esf},
    },
    'A2': {
        'gliding_system': A2GlidingSystem,
        '100': {'100' : gamma_usf},
        '011': {'100' : gamma_usf},
        '111': {'110' : gamma_esf},
    },
    'B1': {
        'gliding_system': B1GlidingSystem,
        '100': {'100' : gamma_usf},
        '011': {'100' : gamma_usf, '010': gamma_usf},
        '111': {'110' : gamma_esf},
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
        '011': {'100' : gamma_usf, '010': gamma_usf, '210': gamma_usf},
        '111': {'110' : gamma_esf},
    },
    'L2_1': {
        'gliding_system': L21GlidingSystem,
        '100': {'110': gamma_usf_symmetric},
        '011': {'100': gamma_usf_symmetric, '010': gamma_usf_symmetric, '210': gamma_usf_symmetric},
        '111': {'110' : gamma_esf},
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

    elif func in [gamma_esf1, gamma_esf2]:
        g_isf = y[nsteps]
        g_esf = y[2 * nsteps]
        order = kwargs.get('order', 4)
        popt, _ = curve_fit(lambda x, *cGs: func(x, cGs, g_isf, g_esf), x, y, p0=[0.1] * order, maxfev=1000000)
        y_fit = func(x_plot, popt, g_isf, g_esf)
        y_fit_orig = func(x, popt, g_isf, g_esf)
        results['usf'] = numpy.max(y_fit[:250]) * 1000.0
        results['isf'] = b * 1000.0
        results['ut'] = numpy.max(y_fit[250:]) * 1000.0
        results['esf'] = c * 1000.0

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

    else:
        raise ValueError(f"Unsupported fit function: {func.__name__ if hasattr(func, '__name__') else func}")

    return popt, y_fit, y_fit_orig, results
