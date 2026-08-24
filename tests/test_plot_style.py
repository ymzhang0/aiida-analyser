import sys
from types import ModuleType

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

aiida_epw_plot = ModuleType('aiida_epw.tools.plot')
aiida_epw_plot.plot_anisotropic_gap = lambda *args, **kwargs: None
sys.modules.setdefault('aiida_epw', ModuleType('aiida_epw'))
sys.modules.setdefault('aiida_epw.tools', ModuleType('aiida_epw.tools'))
sys.modules.setdefault('aiida_epw.tools.plot', aiida_epw_plot)

from aiida_analyser.visualization import (
    DEFAULT_FIGURE_HEIGHT,
    DEFAULT_FIGURE_WIDTH,
    DEFAULT_FONT_SIZE,
    DEFAULT_PANEL_WIDTH,
    STYLE_RESOURCE,
    figure_size,
    plot_style,
    styled_plot,
)


def test_style_public_api_exports_all_defaults():
    assert figure_size() == (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT)
    assert figure_size(columns=2) == (2 * DEFAULT_PANEL_WIDTH, DEFAULT_FIGURE_HEIGHT)
    assert DEFAULT_FONT_SIZE > 0
    assert STYLE_RESOURCE.name == 'aiida_analyser.mplstyle'


def test_plot_style_uses_shared_defaults_and_restores_rcparams():
    original_font_size = mpl.rcParams['font.size']

    with plot_style():
        assert mpl.rcParams['font.family'] == ['serif']
        assert mpl.rcParams['font.size'] == 10
        assert mpl.rcParams['axes.labelsize'] == 10
        assert mpl.rcParams['figure.figsize'] == [4.8, 3.2]

    assert mpl.rcParams['font.size'] == original_font_size


def test_plot_style_allows_local_font_overrides():
    with plot_style(font_size=11, tick_fontsize=9):
        assert mpl.rcParams['font.size'] == 11
        assert mpl.rcParams['axes.labelsize'] == 11
        assert mpl.rcParams['xtick.labelsize'] == 9


def test_figure_size_scales_multi_panel_figures():
    assert figure_size() == (4.8, 3.2)
    assert figure_size(columns=3, rows=2) == pytest.approx((9.6, 6.4))


def test_styled_plot_applies_style_without_leaking_it():
    original_font_size = mpl.rcParams['font.size']

    @styled_plot
    def make_plot():
        fig, axis = plt.subplots()
        axis.set_xlabel('x')
        return fig, axis

    fig, axis = make_plot()
    assert axis.xaxis.label.get_fontsize() == 10
    assert tuple(fig.get_size_inches()) == pytest.approx((4.8, 3.2))
    assert mpl.rcParams['font.size'] == original_font_size
    plt.close(fig)
