"""Shared Matplotlib style and figure dimensions.

The defaults target figures that can be inserted into a paper or a slide at
their native size while keeping plot text close to ordinary 10-point text.
"""

from contextlib import contextmanager
from functools import wraps
from importlib.resources import as_file, files

__all__ = [
    'DEFAULT_DOS_FIGURE_HEIGHT',
    'DEFAULT_DOS_PANEL_WIDTH',
    'DEFAULT_FIGURE_HEIGHT',
    'DEFAULT_FIGURE_WIDTH',
    'DEFAULT_FONT_SIZE',
    'DEFAULT_PANEL_WIDTH',
    'STYLE_RESOURCE',
    'dos_figure_size',
    'figure_size',
    'plot_style',
    'styled_plot',
]


DEFAULT_FONT_SIZE = 10
DEFAULT_FIGURE_WIDTH = 4.8
DEFAULT_FIGURE_HEIGHT = 3.2
DEFAULT_PANEL_WIDTH = 3.2
DEFAULT_DOS_PANEL_WIDTH = 3.2
DEFAULT_DOS_FIGURE_HEIGHT = 4.8
STYLE_RESOURCE = files('aiida_analyser.visualization').joinpath('aiida_analyser.mplstyle')


def figure_size(columns=1, rows=1):
    """Return a consistent figure size in inches for a subplot grid."""
    if columns < 1 or rows < 1:
        raise ValueError('columns and rows must both be positive integers.')
    width = DEFAULT_FIGURE_WIDTH if columns == 1 else DEFAULT_PANEL_WIDTH * columns
    return (width, DEFAULT_FIGURE_HEIGHT * rows)


def dos_figure_size(columns=1, rows=1):
    """Return a portrait-oriented figure size for DOS subplot grids."""
    if columns < 1 or rows < 1:
        raise ValueError('columns and rows must both be positive integers.')
    return (DEFAULT_DOS_PANEL_WIDTH * columns, DEFAULT_DOS_FIGURE_HEIGHT * rows)


@contextmanager
def plot_style(
    *, font_size=None, title_fontsize=None, label_fontsize=None,
    tick_fontsize=None, legend_fontsize=None,
):
    """Apply the packaged plot style temporarily, with optional overrides."""
    import matplotlib as mpl

    font_size = DEFAULT_FONT_SIZE if font_size is None else font_size
    overrides = {
        'font.size': font_size,
        'axes.titlesize': title_fontsize if title_fontsize is not None else font_size,
        'axes.labelsize': label_fontsize if label_fontsize is not None else font_size,
        'xtick.labelsize': tick_fontsize if tick_fontsize is not None else font_size,
        'ytick.labelsize': tick_fontsize if tick_fontsize is not None else font_size,
        'legend.fontsize': legend_fontsize if legend_fontsize is not None else font_size,
    }
    with as_file(STYLE_RESOURCE) as style_path:
        with mpl.rc_context(fname=style_path):
            with mpl.rc_context(overrides):
                yield


def styled_plot(function):
    """Run a plotting function inside :func:`plot_style`."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with plot_style(
            font_size=kwargs.get('font_size'),
            title_fontsize=kwargs.get('title_fontsize'),
            label_fontsize=kwargs.get('label_fontsize'),
            tick_fontsize=kwargs.get('tick_fontsize', kwargs.get('ticklabel_fontsize')),
            legend_fontsize=kwargs.get('legend_fontsize'),
        ):
            return function(*args, **kwargs)
    return wrapped
