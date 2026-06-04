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


def sine_expansion(x, cGs, period=1/2):
    """Generic sine series expansion: sum_{i=1}^N cG_i * sin(pi*x)**(2*i)."""
    sin_sq = numpy.sin(numpy.pi * x / period)**2
    result = numpy.zeros_like(x, dtype=float)
    for i, cG in enumerate(cGs, 1):
        result += cG * (sin_sq**i)
    return result


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

    elif func == gamma_esf:
        b = 2 * y[nsteps]
        c = y[2 * nsteps] * 2 - b
        order = kwargs.get('order', 4)
        popt, _ = curve_fit(lambda x, *cGs: func(x, cGs, b, c), x, y, p0=[0.1] * order, maxfev=1000000)
        y_fit = func(x_plot, popt, b, c)
        y_fit_orig = func(x, popt, b, c)
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
