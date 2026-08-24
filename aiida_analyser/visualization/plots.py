# encoding: utf-8
"""Plotting utilities shared across analyser domains."""
from aiida import orm
from typing import Tuple
from matplotlib import pyplot as plt
from .style import DEFAULT_FONT_SIZE, figure_size, styled_plot
import numpy
from io import StringIO
from scipy.optimize import curve_fit
from rich import print as rprint
import re
import os
import pandas as pd

@styled_plot
def plot_eldos(
    dos_xydata,
    fermi_energy_coarse=0.0,
    axis = None,
    **kwargs,
    ):
    color = kwargs.pop('color', 'r')
    linestyle = kwargs.pop('linestyle', '-')
    label = kwargs.pop('label', r"phdos")

    ticklabel_fontsize = kwargs.pop('ticklabel_fontsize', DEFAULT_FONT_SIZE)
    label_fontsize = kwargs.pop('label_fontsize', DEFAULT_FONT_SIZE)

    E        = dos_xydata.get_array('Energy') - fermi_energy_coarse
    dos = dos_xydata.get_array('EDOS')

    if axis is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
    else:
        ax = axis

    ax.plot(
        dos,
        E,
        color=color,
        linestyle=linestyle,
        label=label)

    ax.set_xticks(
        [0, round(numpy.max(dos) * 1.05, 1)],
        [0, round(numpy.max(dos) * 1.05, 1)],
        fontsize=ticklabel_fontsize,
        )
    ax.set_yticks([], [])

    ax.set_xlim(0, round(numpy.max(dos) * 1.05, 1))
    ax.set_ylim(-2, 2)  
    ax.set_ylabel(r"Energy (eV)", fontsize=label_fontsize)


    if axis is None:
        return plt

@styled_plot
def plot_phdos(
    phdos_xydata,
    axis = None,
    **kwargs,
    ):
    color = kwargs.pop('color', 'r')
    linestyle = kwargs.pop('linestyle', '-')
    label = kwargs.pop('label', r"phdos")

    ticklabel_fontsize = kwargs.pop('ticklabel_fontsize', DEFAULT_FONT_SIZE)
    label_fontsize = kwargs.pop('label_fontsize', DEFAULT_FONT_SIZE)

    w        = phdos_xydata.get_array('Frequency')
    dos = phdos_xydata.get_array('PHDOS')
    idos = pd.Series(dos).cumsum()

    idos /= numpy.max(idos)

    if axis is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
    else:
        ax = axis

    ax.plot(
        dos,
        w,
        color=color,
        linestyle=linestyle,
        label=label)

    ax.plot(
        idos,
        w,
        color=color,
        linestyle='--',
        label=label)

    ax.set_xticks(
        [0, round(numpy.max(dos) * 1.05, 1)],
        [0, round(numpy.max(dos) * 1.05, 1)],
        fontsize=ticklabel_fontsize,
        )
    ax.set_yticks(
        [0, round(numpy.max(w) * 1.05, 1)],
        [0, round(numpy.max(w) * 1.05, 1)],
        fontsize=ticklabel_fontsize,
        )
    ax.set_xlim(0, round(numpy.max(dos) * 1.05, 1))
    ax.set_ylim(0, round(numpy.max(w) * 1.05, 1))
    ax.set_ylabel(r"$\omega$ [meV]", fontsize=label_fontsize)


    if axis is None:
        return plt

@styled_plot
def plot_a2f(
    a2f_arraydata,
    output_parameters,
    axis = None,
    show_data = False,
    integrated_a2f = True,
    **kwargs,
    ):
    w        = a2f_arraydata.get_array('frequency')
    spectral = a2f_arraydata.get_array('a2f')
    if spectral.ndim == 1:
        a2f_curve = spectral
        integrated_curve = None
    else:
        a2f_curve = spectral[:, min(9, spectral.shape[1] - 1)]
        integrated_curve = spectral[:, min(19, spectral.shape[1] - 1)] if spectral.shape[1] > 1 else None

    if axis is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', figure_size()))
    else:
        ax = axis


    ax.plot(
        a2f_curve,
        w,
        color=kwargs.get('color', 'r'),
        linestyle=kwargs.get('linestyle', '-'),
        label=kwargs.get('label1', r"$\alpha^2F$"))
    if integrated_a2f and integrated_curve is not None:
        ax.plot(
            integrated_curve,
            w,
            color=kwargs.get('color', 'k'),
            linestyle=kwargs.get('linestyle', '--'),
            label=kwargs.get('label2', r"$\lambda$")
            )

    # ax.set_xticks(
    #     [0, round(numpy.max(spectral[:, [9, 19]]) * 1.05, 1)],
    #     [0, round(numpy.max(spectral[:, [9, 19]]) * 1.05, 1)],
    #     fontsize=kwargs.get('ticklabel_fontsize', DEFAULT_FONT_SIZE),
    #     )
    # ax.set_yticks(
    #     [0, round(numpy.max(w) * 1.05, 1)],
    #     [0, round(numpy.max(w) * 1.05, 1)],
    #     fontsize=kwargs.get('ticklabel_fontsize', DEFAULT_FONT_SIZE),
    #     )
    max_curve = numpy.max(a2f_curve)
    if integrated_curve is not None:
        max_curve = max(max_curve, numpy.max(integrated_curve))
    ax.set_xlim(0, round(max_curve * 1.05, 1))
    ax.set_ylim(0, round(numpy.max(w) * 1.0, 1))
    # ax.set_ylabel(r"$\alpha^2F$")
    ax.set_ylabel(r"$\omega$ [meV]", fontsize=kwargs.get('label_fontsize', DEFAULT_FONT_SIZE))
    # ax.legend(fontsize=kwargs.get('legend_fontsize', DEFAULT_FONT_SIZE))

    if show_data and output_parameters:
        lambda_ = output_parameters.get('lambda')
        wlog = output_parameters.get('w_log')
        allen_dynes_tc = output_parameters.get('Allen_Dynes_Tc')

        title = f'$\\lambda$ = {lambda_:.2f}\n$\\omega_{{log}}$ = {wlog:.2f} \n$T_c^{{AD}}$ = {allen_dynes_tc:.2f} K'

        # ax.legend(
        #     title,
        #     loc='upper right',
        #     fontsize=kwargs.get('legend_fontsize', DEFAULT_FONT_SIZE),
        #     framealpha=0.5,
        # )

        props = dict(boxstyle='round', facecolor='#526AB1', alpha=0.3)
        ax.text(
            0.05, 0.95, title,
            transform=ax.transAxes,
            fontsize=kwargs.get('legend_fontsize', DEFAULT_FONT_SIZE),
            verticalalignment='top',
            bbox=props)

    if axis is None:
        return plt

@styled_plot
def plot_aniso(epw_calc, axis=None, ignore_temps=0, add_fit=False):

    temps = []
    average_deltas = []
    temp_clusters = []

    if axis is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=figure_size(), sharex=True)
        fig.patch.set_facecolor('white')
    else:
        ax = axis

    for filename in epw_calc.outputs.retrieved.list_object_names():
        if filename.startswith("aiida.imag_aniso_gap0_"):
            text = epw_calc.outputs.retrieved.base.repository.get_object_content(filename)
        else:
            continue

        parse = numpy.loadtxt(StringIO(text))
        try:
            temp = parse[:, 0]
            delta_nk = parse[:, 1]
        except IndexError:
            continue

        temps.append(min(temp))
        average_deltas.append(numpy.average(delta_nk, weights=temp))

        ax.plot(temp, delta_nk, color="blue")
        ax.axvline(x=min(temp), color="blue", linestyle="-")

    p_opt = None

    if add_fit:
        try:
            # Perform the curve fitting
            if ignore_temps > 0:
                p_opt = curve_fit(
                    fitting_function,
                    temps[:-ignore_temps],
                    average_deltas[:-ignore_temps],
                    p0=[1, average_deltas[0], max(temps)],
                    bounds=([0, 1, 0], [numpy.inf, numpy.inf, numpy.inf]),
                )[0]
            else:
                p_opt = curve_fit(
                    fitting_function,
                    temps,
                    average_deltas,
                    p0=[1, average_deltas[0], max(temps)],
                    bounds=([0, 1, 0], [numpy.inf, numpy.inf, numpy.inf]),
                )[0]
            zero_average, exponent, Tc = p_opt
        except (TypeError, RuntimeError, IndexError, ValueError):
            pass

        ax.plot(temps, average_deltas, "ro")

    title = ''

    if p_opt is not None:
        plt.plot(numpy.arange(min(temps), Tc, 0.1), fitting_function(numpy.arange(min(temps), Tc, 0.1), *p_opt))
        title += f" - {zero_average:.1f} - {exponent:.1f} - {Tc:.1f}"

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Delta_nk (meV)")

    plt.title(title)

    return plt

@styled_plot
def plot_bands(
    bands: orm.BandsData,
    axis=None,
    reference_energy=0,
    seekpath_params=None,
    ylabel='Energy (eV)',
    **kwargs,
    ):
    """Plot a band structure from a ``BandsData`` node."""

    color = kwargs.pop('color', 'black')
    ticklabel_fontsize = kwargs.pop('ticklabel_fontsize', DEFAULT_FONT_SIZE)
    label_fontsize = kwargs.pop('label_fontsize', DEFAULT_FONT_SIZE)
    ylim = kwargs.pop('ylim', [-2, 2])

    if axis is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=figure_size(), sharex=True)
        fig.patch.set_facecolor('white')
    else:
        ax = axis

    if seekpath_params is not None:
        xticks, xtick_labels = create_xticks(seekpath_params)
    else:
        try:
            xticks, xtick_labels = create_xticks_bands(bands)
        except IndexError:
            xticks = []
            xtick_labels = []

    xlim = kwargs.pop('xlim', [xticks[0], xticks[-1]])
    ax.set_xlim(xlim)

    label = kwargs.pop('label', None)

    bands_array = bands.get_bands()
    nkpt, nbnd = bands_array.shape

    for i, band in enumerate(bands_array.transpose()):
        ax.plot(numpy.arange(nkpt) / nkpt * xlim[-1], band - reference_energy, color=color, **kwargs)
    
    for tick in xticks:
        ax.axvline(tick/ nkpt * xlim[-1], color='k')

    if len(xticks) > 0:
        ax.set_xticks([xtick/ nkpt * xlim[-1] for xtick in xticks], xtick_labels, fontsize=ticklabel_fontsize)
    
    ax.plot([], [], label=label)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_ylim(ylim)
    ax.set_yticks(ylim + [0])
    ax.set_yticklabels(ylim + ["$E_F$"], fontsize=ticklabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if axis is None:
        return plt


def create_xticks(seekpath_params):
    """Create xticks and xtick_labels for a band structure plot from the Seek-path parameters."""

    def transform_gamma(label):
        if label == 'GAMMA':
            return r'$\Gamma$'
        return label

    path = seekpath_params['path']
    xtick_labels = [transform_gamma(path[0][0]), ]

    for segment_number, segment_labels in enumerate(path):
        try:
            if segment_labels[1] == path[segment_number + 1][0]:
                xtick_labels.append(transform_gamma(segment_labels[1]))
            else:
                xtick_labels.append(f'{transform_gamma(segment_labels[1])}|{path[segment_number + 1][0]}')
        except IndexError:
            xtick_labels.append(transform_gamma(segment_labels[1]))

    explicit_segments = seekpath_params['explicit_segments']
    xticks = [explicit_segments[0][0]]

    for explicit_segment in explicit_segments:
        xticks.append(explicit_segment[1])

    return xticks, xtick_labels

def create_xticks_bands(bands: orm.BandsData) -> Tuple[list, list]:
    """Create xticks and xtick_labels for a band structure plot.

    Takes a BandsData object and returns a tuple of xticks and xtick_labels. The script takes care
    of two things:

    1. If the last label of a segment is not the same as the first label of the next segment, a
       vertical line is added between the two symmetry point labels.
    2. In case the label is "GAMMA", it is replaced with the greek capital letter gamma, as is the
       convention.

    """
    def transform_gamma(label):
        if label == 'GAMMA':
            return r'$\Gamma$'
        return label

    labels = bands.base.attributes.get('labels')
    label_numbers = bands.base.attributes.get('label_numbers')

    xticks = [label_numbers[0], ]
    xtick_labels = [transform_gamma(labels[0]), ]

    for label, label_number in zip(labels[1:], label_numbers[1:]):

        if label_number - xticks[-1] == 1:
            xtick_labels.append(f'{xtick_labels.pop()}|{transform_gamma(label)}')
        else:
            xtick_labels.append(transform_gamma(label))
            xticks.append(label_number)

    return xticks, xtick_labels


@styled_plot
def plot_bands_comparison(bands_qe, bands_w90, fermi_qe, fermi_w90, axis=None):

    xticks, xtick_labels = create_xticks_bands(bands_w90)

    if axis is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=figure_size(), sharex=True)
        fig.patch.set_facecolor('white')
    else:
        ax = axis

    ax.plot([], '-r')
    ax.plot([], 'b.')
    ax.legend(['Quantum ESPRESSO', 'W90'])

    try:
        bands_qe = bands_qe.get_bands()
        bands_qe -= fermi_qe

        for band_qe in bands_qe.transpose():
            ax.plot(band_qe, '-r')
    except KeyError:
        pass

    try:
        bands_w90 = bands_w90.get_bands()
        bands_w90 -= fermi_w90

        for band_w90 in bands_w90.transpose():
            ax.plot(band_w90, 'b.', markersize=2)
    except KeyError:
        pass

    for tick in xticks:
        ax.axvline(tick)

    ax.set_xticks(xticks, xtick_labels)
    ax.axhline(0, color='k', linestyle='--')

    ax.set_ylabel('Energy (eV)')
    ax.set_yticks([-10, -5, 0, 5, 10, 15], [-10, -5, '$E_F$', 5, 10, 15])
    ax.set_ylim([-10, 10])

    if axis is None:
        return plt

def create_xticklabels(plain_path):
    GREEK_LETTERS_TO_LATEX = {
        'GAMMA': r'\Gamma',
        'SIGMA': r'\Sigma',
        'DELTA': r'\Delta',
        'LAMBDA': r'\Lambda',
        'PI': r'\Pi',
        'THETA': r'\Theta',
        'PHI': r'\Phi',
        'OMEGA': r'\Omega',
        'ALPHA': r'\Alpha',
        'BETA': r'\Beta',
        'XI': r'\Xi',
        'MU': r'\Mu',
        'ZETA': r'\Zeta',
        'NU': r'\Nu',
        'KAPPA': r'\Kappa',
    }

    def greek_letter_to_latex(letter):
        match = re.fullmatch(r'([A-Z]+)(?:_(\d+))?', letter)

        base, sub = match.groups()

        if base in GREEK_LETTERS_TO_LATEX:
            latex_base = GREEK_LETTERS_TO_LATEX[base]
        else:
            latex_base = rf'\mathrm{{{base}}}'
        if sub:
            return rf'${latex_base}_{{{sub}}}$'
        else:
            return rf'${latex_base}$'

    xticklabels = []

    for points in plain_path:
        if isinstance(points, tuple):
            p0 = greek_letter_to_latex(points[0])
            p1 = greek_letter_to_latex(points[1])
            xticklabels.append(f'{p0}|{p1}')
        else:
            xticklabels.append(greek_letter_to_latex(points))

    return xticklabels

@styled_plot
def plot_epw_interpolated_bands(
    epw_workchain,
    axes=None,
    elabel = 'Energy (eV)',
    plabel = 'Frequency (meV)',
    **kwargs,
    ):
    """Plot the interpolated bands from an ``EpwWorkChain`` node."""

    if axes is None:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=figure_size(rows=2), sharex=True)
    elif axes.shape != (2,):
        raise ValueError("axes must be a 2D array")

    fermi_energy_coarse = epw_workchain.outputs.output_parameters.get('fermi_energy_coarse')
    # parameters = epw_workchain.outputs.seekpath_parameters.get_dict()
    parameters = epw_workchain.inputs.kfpoints.creator.outputs.parameters.get_dict()
    path = parameters['path']
    explicit_segments = parameters['explicit_segments']
    explicit_kpoints_linearcoord = parameters['explicit_kpoints_linearcoord']

    plain_path = [path[0][0]]

    for [path_start, path_end], [dec_seg_start, dec_seg_end] in zip(path[:-1], path[1:]):
        if path_end == dec_seg_start:
            plain_path.append(dec_seg_start)
        else:
            plain_path.append((path_end, dec_seg_start))

    plain_path.append(path[-1][1])

    kpoints_indicies = [0]

    for [seg_start, seg_end] in explicit_segments[1:]:
        kpoints_indicies.append(seg_start)

    kpoints_indicies.append(explicit_segments[-1][1]-1)

    xticks = [explicit_kpoints_linearcoord[k] for k in kpoints_indicies]
    xticklabels = create_xticklabels(plain_path)

    # el_bands = epw_workchain.outputs.bands.el_band_structure.get_bands()
    # ph_bands = epw_workchain.outputs.bands.ph_band_structure.get_bands()
    el_bands = epw_workchain.outputs.el_band_structure.get_bands()
    ph_bands = epw_workchain.outputs.ph_band_structure.get_bands()

    max_freq = numpy.max(ph_bands)
    min_freq = numpy.min(ph_bands)

    axes[0].set_ylim([-2, 2])
    axes[0].set_yticks([-2, 0, 2])
    axes[0].set_yticklabels([-2, '$E_F$', 2], fontsize=kwargs.get('ticklabel_fontsize', DEFAULT_FONT_SIZE))
    axes[0].set_ylabel(elabel, fontsize=kwargs.get('label_fontsize', DEFAULT_FONT_SIZE))
    axes[0].set_xlim([explicit_kpoints_linearcoord[0], explicit_kpoints_linearcoord[-1]])

    axes[1].set_ylim([min_freq, max_freq*1.05])
    axes[1].set_yticks([numpy.floor(min_freq), numpy.ceil(max_freq*1.05)])
    axes[1].set_yticklabels(
        [numpy.floor(min_freq), numpy.ceil(max_freq*1.05)],
        fontsize=kwargs.get('ticklabel_fontsize', DEFAULT_FONT_SIZE),
    )
    axes[1].set_ylabel(plabel, fontsize=kwargs.get('label_fontsize', DEFAULT_FONT_SIZE))
    axes[1].set_xlim([explicit_kpoints_linearcoord[0], explicit_kpoints_linearcoord[-1]])

    for tick in xticks:
        axes[0].axvline(tick, color='k', linewidth=1, linestyle='--')
        axes[1].axvline(tick, color='k', linewidth=1, linestyle='--')

    axes[0].axhline(0, color='k', linewidth=1, linestyle='--')
    axes[1].axhline(0, color='k', linewidth=1, linestyle='--')

    axes[0].set_xticks([])
    axes[0].set_xticklabels([])

    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xticklabels, fontsize=kwargs.get('ticklabel_fontsize', DEFAULT_FONT_SIZE))

    axes[0].plot([], [], color=kwargs.get('color_el', 'r'), label = kwargs.get('label', ''))
    axes[1].plot([], [], color=kwargs.get('color_ph', 'b'), label = kwargs.get('label', ''))


    for el_band in el_bands.T:
        axes[0].plot(
            explicit_kpoints_linearcoord,
            el_band-fermi_energy_coarse,
            linestyle=kwargs.get('linestyle', '--'),
            color=kwargs.get('color_el', 'r'),
            )
    for ph_band in ph_bands.T:
        axes[1].plot(
            explicit_kpoints_linearcoord,
            ph_band,
            linestyle=kwargs.get('linestyle', '-'),
            color=kwargs.get('color_ph', 'k'),
            )

    axes[0].legend()
    return axes

def check_wannier_optimize(w90_optimize_workchain, filename=None):

    bands_qe = w90_optimize_workchain.inputs.optimize_reference_bands
    bands_w90 = w90_optimize_workchain.outputs.band_structure
    fermi_qe = bands_qe.creator.outputs.output_parameters.get_dict()['fermi_energy']
    fermi_w90 = w90_optimize_workchain.outputs.nscf.output_parameters.get_dict()['fermi_energy']

    plt = plot_bands_comparison(bands_qe, bands_w90, fermi_qe, fermi_w90)

    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close()


def check_wannier_bands(w90_bands_workchain, bands_workchain_qe, filename=None):

    bands_w90 = w90_bands_workchain.outputs.band_structure
    bands_qe = bands_workchain_qe.outputs.band_structure
    fermi_qe = bands_workchain_qe.outputs.band_parameters.get_dict()['fermi_energy']
    fermi_w90 = w90_bands_workchain.outputs.nscf.output_parameters.get_dict()['fermi_energy']

    plt = plot_bands_comparison(bands_qe, bands_w90, fermi_qe, fermi_w90)

    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close()


def find_clusters(temps, delta_nk, threshold):
    """Find the clusters of temperatures where the gap is above a certain threshold"""
    # Find the minimum temperature
    min_temp = min(temps)

    # Find the clusters where temp is above the threshold
    clusters = []
    cluster = ([], [])

    for t, d in zip(temps, delta_nk):

        if t > min_temp + threshold:
            cluster[0].append(t)
            cluster[1].append(d)
        else:
            if len(cluster[0]) > 0:
                clusters.append(cluster)
                cluster = ([], [])

    # Add the last cluster if it exists
    if len(cluster) > 0:
        clusters.append(cluster)

    return clusters


def fitting_function(T, p, delta_zero, Tc):
    import numpy
    T = numpy.atleast_1d(T)
    gap = numpy.zeros_like(T, dtype=float)
    mask = T < Tc
    gap[mask] = delta_zero * (1.0 - (T[mask] / Tc) ** p) ** 0.5
    return gap if len(gap) > 1 else gap[0]


# Import gap-plotting and analysis functions from aiida-epw
from aiida_epw.tools.plot import plot_anisotropic_gap


def _iter_iso_gap_data(gap_functions):
    """Yield temperature and gap data from typed and legacy AiiDA nodes."""
    if hasattr(gap_functions, 'get_itergap_functions'):
        yield from gap_functions.get_itergap_functions()
        return

    for array_name, array in gap_functions.get_iterarrays():
        yield float(array_name.replace('_', '.')), array

@styled_plot
def plot_iso_gap_function(
    iso_gap_function,
    axis=None,
    fit=False,
    p0=None,
    **kwargs,
):
    """
    Adapter wrapper for gap_iso_imag_temp to support external axes (axis)
    and optional tempmax matching the original aiida-analyser signature.
    """
    import numpy as numpy
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from scipy.optimize import curve_fit
    from aiida_epw.tools.calculators import bcs_gap_function

    imag_delta = []
    imag_temp = []

    for temperature, data in _iter_iso_gap_data(iso_gap_function):
        if hasattr(data, 'get'):
            gap = data['deltaw'][0] * 1000
        else:
            gap = numpy.asarray(data)[0, -1] * 1000
        if numpy.isnan(gap):
            continue
        imag_delta.append(gap)  # Convert to meV
        imag_temp.append(temperature)

    if not imag_temp:
        return axis

    tempmax = kwargs.get('tempmax', numpy.max(imag_temp))
    font = kwargs.get('font', kwargs.get('label_fontsize', DEFAULT_FONT_SIZE))

    if axis is None:
        fig = plt.figure(figsize=kwargs.get('figsize', figure_size()))
        ax1 = fig.add_subplot(1, 1, 1)
    else:
        ax1 = axis

    ax1.set_title("Superconducting Gap vs. Temperature", fontsize=font)
    ax1.set_xlabel("Temeperature (K)", fontsize=font)
    ax1.set_xlim(0, tempmax*1.1)
    ax1.set_ylabel(r"$\Delta_0$ (meV)", fontsize=font)
    ax1.tick_params(axis="y", labelsize=font)
    ax1.tick_params(axis="x", labelsize=font)
    ax1.plot(
        imag_temp,
        imag_delta,
        linestyle="-",
        marker="o",
        c="k",
        label="Im. axis",
    )
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    if fit:
        if p0 is None:
            p0 = [imag_temp[-1], 3.3, imag_delta[0]]
        try:
            popt, pcov = curve_fit(bcs_gap_function, imag_temp, imag_delta, p0=p0, maxfev=100000)
            Tc, p, Delta_0 = popt
            T = numpy.linspace(0, Tc, 100)
            ax1.plot(
                T,
                bcs_gap_function(T, Tc, p, Delta_0),
                linestyle="--",
                c="r",
                label="Fit",
            )
            ax1.set_xlim(0, Tc*1.1)
        except Exception as e:
            print(f"BCS fit failed: {e}")

    destpath = kwargs.get('destpath', None)
    prefix = kwargs.get('prefix', 'aiida')
    if destpath and axis is None:
        import os
        plt.savefig(os.path.join(destpath, f"{prefix}_iso_gap_imag_vs_Temp.pdf"))
        
    return ax1


@styled_plot
def plot_aniso_gap_function(
    aniso_gap_functions_arraydata,
    axis=None,
    **kwargs,
):
    """
    Adapter wrapper for plot_anisotropic_gap to support ArrayData node input
    and map the 'axis' parameter to 'ax'.
    """
    from aiida import orm
    import numpy as numpy

    if isinstance(aniso_gap_functions_arraydata, orm.ArrayData):
        aniso_gap_functions_dict = {}
        for arrayname in aniso_gap_functions_arraydata.get_arraynames():
            array = aniso_gap_functions_arraydata.get_array(arrayname)
            if array.size > 0:
                temp = float(numpy.min(array[:, 0]))
                aniso_gap_functions_dict[temp] = array
    else:
        aniso_gap_functions_dict = aniso_gap_functions_arraydata

    return plot_anisotropic_gap(
        aniso_gap_functions_dict=aniso_gap_functions_dict,
        ax=axis,
        **kwargs,
    )
