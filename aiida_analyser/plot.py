# encoding: utf-8
"""Temporary home for various plotting tools."""
from aiida import orm
from typing import Tuple
from matplotlib import pyplot as plt
import numpy
from io import StringIO
from scipy.optimize import curve_fit
from rich import print as rprint
import re
import os


def plot_eldos(
    a2f_workchain,
    axis = None,
    **kwargs,
    ):

    fermi_energy_coarse = a2f_workchain.outputs.output_parameters.get('fermi_energy_coarse')

    E        = a2f_workchain.outputs.a2f.dos.get_array('Energy') - fermi_energy_coarse
    dos = a2f_workchain.outputs.a2f.dos.get_array('EDOS')

    if axis is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
    else:
        ax = axis

    ax.plot(
        dos,
        E,
        color=kwargs.get('color', 'r'),
        linestyle=kwargs.get('linestyle', '-'),
        label=r"phdos")

    ax.set_xticks(
        [0, round(numpy.max(dos) * 1.05, 1)],
        [0, round(numpy.max(dos) * 1.05, 1)],
        fontsize=kwargs.get('ticklabel_fontsize', 16),
        )
    ax.set_yticks([], [])

    ax.set_xlim(0, round(numpy.max(dos) * 1.05, 1))
    ax.set_ylim(-2, 2)
    ax.set_ylabel(r"Energy (eV)", fontsize=kwargs.get('label_fontsize', 16))


    if axis is None:
        return plt

def plot_phdos(
    phdos,
    axis = None,
    **kwargs,
    ):
    w        = phdos.get_array('Frequency')
    dos = phdos.get_array('PHDOS')

    if axis is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
    else:
        ax = axis

    ax.plot(
        dos,
        w,
        color=kwargs.get('color', 'r'),
        linestyle=kwargs.get('linestyle', '-'),
        label=r"phdos")

    ax.set_xticks(
        [0, round(numpy.max(dos) * 1.05, 1)],
        [0, round(numpy.max(dos) * 1.05, 1)],
        fontsize=kwargs.get('ticklabel_fontsize', 16),
        )
    ax.set_yticks(
        [0, round(numpy.max(w) * 1.05, 1)],
        [0, round(numpy.max(w) * 1.05, 1)],
        fontsize=kwargs.get('ticklabel_fontsize', 16),
        )
    ax.set_xlim(0, round(numpy.max(dos) * 1.05, 1))
    ax.set_ylim(0, round(numpy.max(w) * 1.05, 1))
    ax.set_ylabel(r"$\omega$ [meV]", fontsize=kwargs.get('label_fontsize', 16))


    if axis is None:
        return plt

def plot_a2f(
    a2f_workchain,
    axis = None,
    show_data = False,
    plot_dos = True,
    **kwargs,
    ):
    a2f = a2f_workchain.outputs.a2f.a2f
    output_parameters = a2f_workchain.outputs.output_parameters
    w        = a2f.get_array('frequency')
    spectral = a2f.get_array('a2f')

    if axis is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(4, 8))
    else:
        ax = axis

    if plot_dos:
        dos = a2f_workchain.outputs.a2f.phdos
        plot_phdos(
            dos,
            axis=ax,
            color='black',
        )

    ax.plot(
        spectral[:, 9],
        w,
        color=kwargs.get('color', 'r'),
        linestyle=kwargs.get('linestyle', '-'),
        label=r"$\alpha^2F$")
    ax.plot(
        spectral[:, 19],
        w,
        color=kwargs.get('color', 'k'),
        linestyle=kwargs.get('linestyle', '--'),
        label=r"$\lambda$")

    ax.set_xticks(
        [0, round(numpy.max(spectral[:, [9, 19]]) * 1.05, 1)],
        [0, round(numpy.max(spectral[:, [9, 19]]) * 1.05, 1)],
        fontsize=kwargs.get('ticklabel_fontsize', 16),
        )
    ax.set_yticks(
        [0, round(numpy.max(w) * 1.05, 1)],
        [0, round(numpy.max(w) * 1.05, 1)],
        fontsize=kwargs.get('ticklabel_fontsize', 16),
        )
    ax.set_xlim(0, round(numpy.max(spectral[:, [9, 19]]) * 1.05, 1))
    ax.set_ylim(0, round(numpy.max(w) * 1.05, 1))
    # ax.set_ylabel(r"$\alpha^2F$")
    ax.set_ylabel(r"$\omega$ [meV]", fontsize=kwargs.get('label_fontsize', 16))
    ax.legend(fontsize=kwargs.get('legend_fontsize', 16))

    if show_data and output_parameters:
        lambda_ = output_parameters.get('lambda')
        wlog = output_parameters.get('w_log')
        allen_dynes_tc = output_parameters.get('Allen_Dynes_Tc')

        title = f'$\lambda$ = {lambda_:.2f}\n$\omega_{{log}}$ = {wlog:.2f} \n$T_c^{{AD}}$ = {allen_dynes_tc:.2f} K'

        # ax.legend(
        #     title,
        #     loc='upper right',
        #     fontsize=kwargs.get('legend_fontsize', 10),
        #     framealpha=0.5,
        # )

        props = dict(boxstyle='round', facecolor='#526AB1', alpha=0.3)
        ax.text(
            0.05, 0.95, title,
            transform=ax.transAxes,
            fontsize=kwargs.get('legend_fontsize', 10),
            verticalalignment='top',
            bbox=props)

    if axis is None:
        return plt

def plot_aniso(epw_calc, axis=None, ignore_temps=0, add_fit=False):

    temps = []
    average_deltas = []
    temp_clusters = []

    if axis is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
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

def plot_bands(bands: orm.BandsData, axis=None, reference_energy=0, seekpath_params=None, **kwargs):
    """Plot a band structure from a ``BandsData`` node."""

    color = kwargs.pop('color', 'blue')

    if axis is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
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

    if len(xticks) > 0:
        ax.set_xticks(xticks, xtick_labels)

    for tick in xticks:
        ax.axvline(tick, color='k')

    for band in bands.get_bands().transpose():
        ax.plot(band - reference_energy, color=color, **kwargs)

    ax.axhline(0, color='k', linestyle='--')

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

    labels = bands.attributes['labels']
    label_numbers = bands.attributes['label_numbers']

    xticks = [label_numbers[0], ]
    xtick_labels = [transform_gamma(labels[0]), ]

    for label, label_number in zip(labels[1:], label_numbers[1:]):

        if label_number - xticks[-1] == 1:
            xtick_labels.append(f'{xtick_labels.pop()}|{transform_gamma(label)}')
        else:
            xtick_labels.append(transform_gamma(label))
            xticks.append(label_number)

    return xticks, xtick_labels


def plot_bands_comparison(bands_qe, bands_w90, fermi_qe, fermi_w90, axis=None):

    xticks, xtick_labels = create_xticks_bands(bands_w90)

    if axis is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
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

def plot_epw_interpolated_bands(
    epw_workchain,
    axes=None,
    **kwargs,
    ):
    """Plot the interpolated bands from an ``EpwWorkChain`` node."""

    if axes is None:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    elif axes.shape != (2,):
        raise ValueError("axes must be a 2D array")

    fermi_energy_coarse = epw_workchain.outputs.output_parameters.get('fermi_energy_coarse')
    parameters = epw_workchain.outputs.seekpath_parameters.get_dict()

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

    el_bands = epw_workchain.outputs.bands.el_band_structure.get_bands()
    ph_bands = epw_workchain.outputs.bands.ph_band_structure.get_bands()

    max_freq = numpy.max(ph_bands)
    min_freq = numpy.min(ph_bands)

    axes[0].set_ylim([-2, 2])
    axes[0].set_yticks([-2, 0, 2])
    axes[0].set_yticklabels([-2, '$E_F$', 2], fontsize=kwargs.get('ticklabel_fontsize', 16))
    axes[0].set_ylabel('Energy (eV)', fontsize=kwargs.get('label_fontsize', 16))
    axes[0].set_xlim([explicit_kpoints_linearcoord[0], explicit_kpoints_linearcoord[-1]])

    axes[1].set_ylim([min_freq, max_freq*1.05])
    axes[1].set_yticks([numpy.floor(min_freq), numpy.ceil(max_freq*1.05)])
    axes[1].set_yticklabels([numpy.floor(min_freq), numpy.ceil(max_freq*1.05)], fontsize=kwargs.get('ticklabel_fontsize', 16))
    axes[1].set_ylabel('Frequency (meV)', fontsize=kwargs.get('label_fontsize', 16))
    axes[1].set_xlim([explicit_kpoints_linearcoord[0], explicit_kpoints_linearcoord[-1]])

    for tick in xticks:
        axes[0].axvline(tick, color='k', linewidth=1, linestyle='--')
        axes[1].axvline(tick, color='k', linewidth=1, linestyle='--')

    axes[0].axhline(0, color='k', linewidth=1, linestyle='--')
    axes[1].axhline(0, color='k', linewidth=1, linestyle='--')

    axes[0].set_xticks([])
    axes[0].set_xticklabels([])

    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xticklabels, fontsize=kwargs.get('ticklabel_fontsize', 16))

    axes[1].plot([], [])


    for el_band, ph_band in zip(el_bands.T, ph_bands.T):
        axes[0].plot(
            explicit_kpoints_linearcoord,
            el_band-fermi_energy_coarse,
            linestyle=kwargs.get('linestyle', '--'),
            color=kwargs.get('color', 'r'),
            )
        axes[1].plot(
            explicit_kpoints_linearcoord,
            ph_band,
            linestyle=kwargs.get('linestyle', '-'),
            color=kwargs.get('color', 'k'),
            )
    return axes

def check_wannier_optimize(w90_optimize_workchain, filename=None):

    bands_qe = w90_optimize_workchain.inumpyuts.optimize_reference_bands
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


def fitting_function(T, p, delta_zero, Tc):
    return delta_zero * (1 - (T / Tc) ** p) ** 0.5


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
    return delta_zero * (1 - (T / Tc) ** p) ** 0.5


def find_average(data, T):

    data = data[data[:,1] >= 0]

    T = float(T)

    midpoint = numpy.average(data[:, 1], weights=data[:, 0]-T)
    return midpoint


def plot_gap_function(
    aniso_workchain,
    axis,
    remove_temps = [0, 0],
    suggested_p = None,
    exclude_points = None,
    suppression = 1.0,
    **kwargs,
    ):

    if axis is None:
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
        fig.patch.set_facecolor('white')
    else:
        ax = axis

    temps = []
    average_deltas = []
    rhos = []


    aniso_gap_functions = aniso_workchain.outputs.aniso.aniso_gap_functions

    for filename in aniso_gap_functions.get_arraynames():
        parse = aniso_gap_functions.get_array(filename)
        temp = numpy.min(parse[:,0])
        try:
            rho = parse[:,0]
            delta_nk = parse[:,1]

            if len(rho) <=2:
                continue
            mid = find_average(parse, temp)

            temps.append(temp)
            average_deltas.append(mid)
            rhos.append(parse)

        except IndexError:
            continue


    _, sorted_rhos = zip(*sorted(zip(temps, rhos)))

    sorted_pairs = sorted(zip(temps, average_deltas), key=lambda x: x[0])
    temps_sorted, average_deltas_sorted = zip(*sorted_pairs)

    if remove_temps != [0, 0]:
        rprint(f'[italic green]Removed temps[/italic green]: {remove_temps}')
        temps_sorted = temps_sorted[remove_temps[0]:ntemps-remove_temps[1]]
        average_deltas_sorted = average_deltas_sorted[remove_temps[0]:ntemps-remove_temps[1]]
        sorted_rhos = sorted_rhos[remove_temps[0]:ntemps-remove_temps[1]]

    if exclude_points:
        rprint(f'[italic green]Excluded points[/italic green]: {exclude_points}')
        temps_sorted = [temps_sorted[i] for i in range(ntemps) if i not in exclude_points]
        average_deltas_sorted = [average_deltas_sorted[i] for i in range(ntemps) if i not in exclude_points]
        sorted_rhos = [sorted_rhos[i] for i in range(ntemps) if i not in exclude_points]


    ntemps = len(temps_sorted)
    if ntemps <=2:
        rprint('[bold red]less than 2 temperature, can\'t be fit[/bold red]')
        return {
            'well_fit': False,
            'remove_temps': remove_temps,
            'exclude_points': exclude_points,
            'temps': temps,
            'average_deltas': average_deltas,
            'rhos': rhos,
        }
    well_fit = False

    if ntemps == 3:
        if suggested_p is not None:
            p = suggested_p
        else:
            p = [1, average_deltas[0], max(temps)]
        rprint(f'[italic green]Using initial guess[/italic green]: {p}')
        p_opt, p_cov = curve_fit(fitting_function, temps_sorted, average_deltas_sorted, p0=p, maxfev=1000000)
        diff = fitting_function(temps_sorted, *p_opt) -  average_deltas_sorted
        well_fit = True

    else:
        if suggested_p is not None:
            p = suggested_p
        else:
            p = [2, average_deltas[0], temps_sorted[-1]]
        rprint(f'[bold green]Using initial guess[/bold green]: {p}')
        try:
            p_opt, p_cov = curve_fit(fitting_function, temps_sorted, average_deltas_sorted, p0=p, maxfev=1000000)
            diff = fitting_function(temps_sorted, *p_opt) -  average_deltas_sorted
            if numpy.linalg.norm(p_cov) < 1:
                well_fit = True
                rprint(f'[bold green]Fit successfully[/bold green]')
            else:
                rprint('[bold red]Curve fit failed with p_cov[/bold red]')
        except:
            rprint('[bold red]Curve fit failed with exception[/bold red]')


    if well_fit:
        final_temps = temps_sorted
        final_rhos  = sorted_rhos

        width_and_incre = []
        for idr, rho in enumerate(final_rhos[:-1]):
            width_of_rho  = numpy.max(rho[:,0]) - final_temps[idr]
            incre_of_temp = final_temps[idr+1] - final_temps[idr]
            width_and_incre.append([width_of_rho/incre_of_temp])

        max_ratio = numpy.max(width_and_incre)
        if max_ratio > 1:
            suppression = 0.9/max_ratio
        else:
            suppression = 1

        max_rho = 0

        for T, rho in zip(final_temps, final_rhos):
            axis.axvline(numpy.min(rho[:,0]), color='k', linestyle='-', linewidth=1.5)
            axis.plot(T + suppression*(rho[:,0] - T), rho[:,1], color='black', linewidth=1.5)
            max_rho = numpy.max([numpy.max(rho[:,1]), max_rho])

        axis.scatter(final_temps, average_deltas_sorted, c = 'r', s=70)
        axis.set_xlabel('$T$ [K]', fontsize=kwargs.get('label_fontsize', 16))
        axis.set_ylabel('$\Delta_{nk}$ [meV]', fontsize=kwargs.get('label_fontsize', 16))
        axis.set_ylim([0, max_rho])

        x_fit = numpy.linspace(numpy.min(temps)*0.7, p_opt[-1]-0.0001, 1000)
        y_fit = fitting_function(x_fit, *p_opt)

        axis.plot(x_fit, y_fit, color='red', linestyle='--', linewidth=1.5,)

        axis.set_xlim([numpy.min(temps)*0.8, numpy.round(p_opt[-1]*1.2, decimals=0)])
        axis.set_xticks(
            [numpy.round(numpy.min(temps)*0.8, decimals=0), numpy.round(p_opt[-1]*1.2, decimals=0)],
            [numpy.round(numpy.min(temps)*0.8, decimals=0), numpy.round(p_opt[-1]*1.2, decimals=0)],
            fontsize=kwargs.get('ticklabel_fontsize', 14),
        )

        axis.set_yticks(
            [0.0, numpy.round(p_opt[1]*1.3, decimals=0)],
            [0.0, numpy.round(p_opt[1]*1.3, decimals=0)],
            fontsize=kwargs.get('ticklabel_fontsize', 14),
        )


        rprint(f'[bold green]Aniso par:[/bold green] {p_opt}')
        return {
            'remove_temps': remove_temps,
            'exclude_points': exclude_points,
            'temps': temps,
            'average_deltas': average_deltas,
            'rhos': rhos,
            'Tc': p_opt[2],
            'delta0': p_opt[1],
            'expo': p_opt[0],
            'well_fit': True,
            }

    else:
        return {
            'remove_temps': remove_temps,
            'exclude_points': exclude_points,
            'temps': temps,
            'average_deltas': average_deltas,
            'rhos': rhos,
            'well_fit': False,
            }

