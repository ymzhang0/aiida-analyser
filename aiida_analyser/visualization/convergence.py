"""Shared axis configuration for convergence plots."""

import numpy


# MathText-compatible rendering of ``$\\Delta_{\\mathbf{k}}$~\\si{\\per\\angstrom}``.
KPOINT_DISTANCE_LABEL = r'$\Delta_{\mathbf{k}}\,\mathrm{\AA}^{-1}$'


def configure_kpoint_distance_axis(axis, *, xlim=None, xticks=None, xlabel=None, cubic_scale=True):
    """Configure a k-point-distance axis with cubic spacing and descending values.

    A cube-root forward transform and cube inverse define the function scale.
    The displayed direction always runs from coarse (large distance) to dense
    (small distance).
    """
    if cubic_scale:
        axis.set_xscale(
            'function',
            functions=(numpy.cbrt, lambda value: numpy.asarray(value) ** 3),
        )

    if xticks is not None:
        ticks = sorted((float(value) for value in xticks), reverse=True)
        axis.set_xticks(ticks)
        axis.set_xticklabels([f'{value:g}' for value in ticks])

    limits = axis.get_xlim() if xlim is None else xlim
    axis.set_xlim(max(limits), min(limits))
    axis.set_xlabel(KPOINT_DISTANCE_LABEL if xlabel is None else xlabel)
    return axis
