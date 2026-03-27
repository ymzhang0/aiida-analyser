from __future__ import annotations

import re
import typing as ty
from collections import defaultdict
from copy import deepcopy

import numpy
from scipy.optimize import curve_fit

from aiida import orm

from ..base import BaseWorkChainAnalyser
from .basegroup import BaseGroupData
import logging
from ..quantumespresso.pw_base import PwBaseWorkChainAnalyser
from ..quantumespresso.pw_relax import PwRelaxWorkChainAnalyser

from aiida_dislocation.tools import (
    A1GlidingSystem,
    A2GlidingSystem,
    B1GlidingSystem,
    B2GlidingSystem,
    C1bGlidingSystem,
    L21GlidingSystem,
    get_strukturbericht,
    calculate_surface_area
)


def formula_to_latex(formula):
    latex_formula = re.sub(r'(\d+)', r'_{\1}', formula)
    return rf"${latex_formula}$"


def sine_expansion(x, cGs):
    """Generic sine series expansion: sum_{i=1}^N cG_i * sin(pi*x)**(2*i)."""
    sin_sq = numpy.sin(numpy.pi * x)**2
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
        x <= 1,
        val + b * x,
        val + c * x + (b - c)
    )


def gamma_usf(x, cGs):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: Expansion
    """
    return sine_expansion(x, cGs)


def gamma_usf2(x, e_usf1, e_usf2):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: e_usf1 * sin^2(pi*x) + e_usf2 * sin^2(2*pi*x)
    """
    return (e_usf1 * numpy.sin(numpy.pi * x)**2 +
            e_usf2 * numpy.sin(2 * numpy.pi * x)**2)


fit_function_map = {
    'A1': {
        'gliding_system': A1GlidingSystem,
        '100': gamma_usf,
        '011': gamma_usf,
        '111': gamma_esf,
    },
    'A2': {
        'gliding_system': A2GlidingSystem,
        '100': gamma_usf,
        '011': gamma_usf,
        '111': gamma_esf,
    },
    'B1': {
        'gliding_system': B1GlidingSystem,
        '100': gamma_usf,
        '011': gamma_usf2,
        '111': gamma_esf,
    },
    'B2': {
        'gliding_system': B2GlidingSystem,
        '100': gamma_usf,
        '011': gamma_usf2,
        '111': gamma_isf,
    },
    'C1_b': {
        'gliding_system': C1bGlidingSystem,
        '100': gamma_usf,
        '011': gamma_usf2,
        '111': gamma_esf,
    },
    'L2_1': {
        'gliding_system': L21GlidingSystem,
        '100': gamma_usf,
        '011': gamma_usf2,
        '111': gamma_esf,
    },
}


class GSFEWorkChainAnalyserLatest(BaseWorkChainAnalyser):
    """Analyser for the current `GSFEWorkChain` output contract."""

    @staticmethod
    def _shift_series_to_first_value(values: list[float | None]) -> list[float | None]:
        """Shift a numeric series so the first point becomes the zero reference."""
        if not values or values[0] is None:
            return values

        reference = values[0]
        return [
            None if value is None else value - reference
            for value in values
        ]

    @property
    def relax(self) -> orm.WorkChainNode:
        if 'relax' not in self.process_tree:
            raise AttributeError('relax is not found')
        return self.process_tree.relax.node

    @property
    def scf(self) -> orm.WorkChainNode:
        if 'scf' not in self.process_tree:
            raise AttributeError('scf is not found')
        return self.process_tree.scf.node

    @property
    def strukturbericht(self) -> str:
        if 'scf' in self.process_tree:
            structure = self.scf.inputs.pw.structure
        else:
            structure = self.node.inputs.structure
        return get_strukturbericht(structure.get_ase())

    @property
    def results_node(self) -> orm.Dict:
        for label in ('results', 'gsfe_results'):
            if label in self.node.outputs:
                return self.node.outputs[label]
        raise AttributeError('GSFE results output is not found')

    @property
    def surface_area(self) -> float | None:
        return self.results_node.get_dict().get('surface_area_angstrom2')

    @property
    def conventional_energy(self) -> float | None:
        return self.results_node.get_dict().get('conventional_energy_ev')

    @property
    def gliding_plane(self) -> str:
        """Return the gliding plane label."""
        if 'gliding_plane' in self.node.inputs:
            return self.node.inputs.gliding_plane.value
        if 'faulted_structure_data' in self.node.inputs:
            return self.node.inputs.faulted_structure_data.gliding_plane
        raise AttributeError("GSFE gliding plane not found in inputs")

    @property
    def gliding_system(self):
        """Return the gliding system object."""
        return fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(self.gliding_plane)

    def get_state(self) -> tuple[str, str, int]:
        """Get the state of the workchain."""
        subprocesses: tuple[tuple[str, type[BaseWorkChainAnalyser]], ...] = (
            ('relax', PwRelaxWorkChainAnalyser),
            ('scf', PwBaseWorkChainAnalyser),
        )
        return self._get_state_from_subprocesses(subprocesses)

    def get_results(self) -> dict[str, dict[str, dict[str, ty.Any]]]:
        """Return the nested GSFE result payload keyed by direction and step."""
        payload = self.results_node.get_dict()
        results = payload.get('results', payload)

        normalized: dict[str, dict[str, dict[str, ty.Any]]] = {}
        for direction, entries in results.items():
            normalized[direction] = {
                str(step): dict(value)
                for step, value in sorted(entries.items(), key=lambda item: int(item[0]))
            }
        return normalized

    def get_direction_results(self, direction_name: str) -> dict[str, dict[str, ty.Any]]:
        """Return all step results for a specific slip direction."""
        results = self.get_results()
        if direction_name not in results:
            raise KeyError(f'direction `{direction_name}` is not present in GSFE results')
        return results[direction_name]

    def get_sfe_energies(self) -> dict[str, dict[int, float | None]]:
        """Return only the SFE values grouped by direction and step."""
        return {
            direction: {
                int(step): entry.get('sfe')
                for step, entry in entries.items()
            }
            for direction, entries in self.get_results().items()
        }

    def get_total_energies(self) -> dict[str, dict[int, float | None]]:
        """Return only the total energies grouped by direction and step."""
        return {
            direction: {
                int(step): entry.get('energy')
                for step, entry in entries.items()
            }
            for direction, entries in self.get_results().items()
        }

    def get_plot_data(
        self,
        x_axis: str = 'step',
        zero_reference: bool = False,
    ) -> dict[str, dict[str, list[float | None]]]:
        """Return simple plot-ready x/y arrays for each GSFE direction."""
        plot_data: dict[str, dict[str, list[float | None]]] = {}

        for direction, entries in self.get_results().items():
            xs: list[float] = []
            energies: list[float | None] = []
            sfes: list[float | None] = []

            for step, entry in entries.items():
                if x_axis == 'displacement':
                    xs.append(float(numpy.linalg.norm(entry.get('total_cell_shift', [0.0, 0.0, 0.0]))))
                else:
                    xs.append(float(step))

                energies.append(entry.get('energy'))
                sfes.append(entry.get('sfe'))

            if zero_reference:
                energies = self._shift_series_to_first_value(energies)
                sfes = self._shift_series_to_first_value(sfes)

            plot_data[direction] = {
                'x': xs,
                'energy': energies,
                'sfe': sfes,
            }

        return plot_data

    def serialize_faults(self) -> dict[str, list[list[float]]]:
        """Serialize the faults for compatibility, normalized by nsteps."""
        plot_data = self.get_plot_data()  # uses default x_axis='step'
        nsteps = self.gliding_system.general.nsteps
        return {
            direction: [[val / nsteps for val in data['x']]]
            for direction, data in plot_data.items()
        }

    def fit_curve(self, plot=False, axis=None, **kwargs):
        """Fit the curve."""
        from matplotlib.legend_handler import HandlerTuple

        def get_gradient_shades(hex_color, num=5):
            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", [hex_color, "#ffffff"])
            return [mcolors.to_hex(cmap(i)) for i in numpy.linspace(0, 0.8, num)]

        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x']
        xs_dict = self.serialize_faults()
        sfe_energies = self.get_sfe_energies()
        # Convert sfe_energies to the format expected by the port: direction -> [[value, ...]]
        energies = {
            direction: [[val for _, val in sorted(steps.items())]]
            for direction, steps in sfe_energies.items()
        }

        results = {}

        gliding_plane = self.gliding_plane
        gliding_system = self.gliding_system
        func = fit_function_map[self.strukturbericht][gliding_plane]
        nsteps = gliding_system.general.nsteps

        sorted_keys = sorted(energies, key=lambda k: max(energies[k][0]), reverse=True)
        colors = get_gradient_shades(kwargs.get('color', 'black'), num=len(sorted_keys))

        import warnings
        from scipy.optimize import OptimizeWarning

        for slipping_direction, color in zip(sorted_keys, colors):
            results[slipping_direction] = {}
            print(f"Fitting slip system ({gliding_plane})<{slipping_direction}> using function: {func.__name__}")
            for x, y in zip(xs_dict[slipping_direction], energies[slipping_direction]):
                x = numpy.array(x, dtype=float)
                y = numpy.array(y, dtype=float)

                if kwargs.get('zero_reference', True):
                    y = y - y[0]

                x_plot = numpy.linspace(0, x[-1], 500)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    if func == gamma_isf:
                        b = y[nsteps]
                        order = kwargs.get('order', 1)
                        popt, _ = curve_fit(lambda x, *cGs: func(x, cGs, b), x, y, p0=[0.1] * order, maxfev=100000)
                        y_fit = func(x_plot, popt, b)
                        x_max = numpy.arcsin(-b / numpy.pi / popt[0]) / 2 * numpy.pi if popt[0] != 0 else 0.5
                        results[slipping_direction]['isf'] = b
                        results[slipping_direction]['usf'] = func(x_max, popt, b)

                    if func == gamma_esf:
                        b = y[nsteps]
                        c = y[2 * nsteps] - y[nsteps]
                        order = kwargs.get('order', 4)
                        popt, _ = curve_fit(lambda x, *cGs: func(x, cGs, b, c), x, y, p0=[0.1] * order, maxfev=100000)
                        y_fit = func(x_plot, popt, b, c)
                        results[slipping_direction]['usf'] = numpy.max(y_fit[:250])
                        results[slipping_direction]['isf'] = b
                        results[slipping_direction]['ut'] = numpy.max(y_fit[250:])
                        results[slipping_direction]['esf'] = b + c

                    if func == gamma_usf:
                        order = kwargs.get('order', 4)
                        popt, _ = curve_fit(lambda x, *cGs: func(x, cGs), x, y, p0=[0.1] * order, maxfev=100000)
                        y_fit = func(x_plot, popt)
                        results[slipping_direction]['usf'] = numpy.sum(popt)

                    if func == gamma_usf2:
                        (e_usf1, e_usf2), pcov = curve_fit(func, x, y, maxfev=100000)
                        y_fit = func(x_plot, e_usf1, e_usf2)
                        results[slipping_direction]['usf'] = numpy.max(y_fit)
                        results[slipping_direction]['s'] = e_usf2

                if plot:
                    if axis is None:
                        import matplotlib.pyplot as plt
                        fig, axis = plt.subplots(figsize=(10, 6))
                    axis.scatter(
                        x, y,
                        color=color,
                        s=50,
                        zorder=5,
                        marker=markers.pop(0))

                    axis.plot(
                        x_plot,
                        y_fit,
                        linestyle=kwargs.get('linestyle', '--'),
                        color=color,
                        lw=kwargs.get('lw', 1.0),
                        label=kwargs.get('label', '') + f' <{slipping_direction}>')
                    axis.grid(True, alpha=0.3)

        return results

    def plot(
        self,
        ax=None,
        value: str = 'sfe',
        x_axis: str = 'step',
        zero_reference: bool = False,
        **kwargs,
    ):
        """Plot the selected GSFE quantity for each direction."""
        if value not in {'sfe', 'energy'}:
            raise ValueError(f'unsupported value `{value}`, expected `sfe` or `energy`')

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        plot_data = self.get_plot_data(x_axis=x_axis, zero_reference=zero_reference)

        for direction, data in plot_data.items():
            ax.plot(
                data['x'],
                data[value],
                label=kwargs.get('label_prefix', '') + direction,
                marker=kwargs.get('marker', 'o'),
                linestyle=kwargs.get('linestyle', '-'),
                color=kwargs.get('color', None),
                alpha=kwargs.get('alpha', 1.0),
            )

        return ax


class GSFEGroupDataLatest(BaseGroupData):

    def __init__(self, groups=None):
        super().__init__(groups)
        # Data structure: StructureType -> Material -> Plane -> Process -> Layers -> K_Dist -> Node
        self._data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: defaultdict(
                                lambda: None
                            )
                        )
                    )
                )
            )
        )
        self.get_data()

    @property
    def groups(self):
        return self._groups

    @property
    def data(self):
        return self._data

    def get_data(self):
        for grpname in self._groups:
            group = orm.load_group(grpname)
            for node in group.nodes:
                try:
                    process_label = node.process_label
                    structure = node.inputs.structure
                    structuretype = get_strukturbericht(structure.get_ase())
                    formula = structure.get_formula()

                    if 'n_repeats' in node.inputs:
                        n_repeats = node.inputs.n_repeats.value
                    elif 'faulted_structure_data' in node.inputs:
                        n_repeats = node.inputs.faulted_structure_data.n_unit_cells
                    else:
                        raise AttributeError(f"Node<{node.pk}>: Neither 'n_repeats' nor 'faulted_structure_data' found in inputs")

                    if 'gliding_plane' in node.inputs:
                        gliding_plane = node.inputs.gliding_plane.value
                    elif 'faulted_structure_data' in node.inputs:
                        gliding_plane = node.inputs.faulted_structure_data.gliding_plane
                    else:
                        raise AttributeError(f"Node<{node.pk}>: Neither 'gliding_plane' nor 'faulted_structure_data' found in inputs")

                    kpoints_distance = node.inputs.kpoints_distance.value

                    # Structure: StructureType -> Formula -> Plane -> Process -> Layers -> K_Dist -> Node
                    if process_label in ['GSFEWorkChain']:
                        self._data[structuretype][formula][gliding_plane][process_label][n_repeats][kpoints_distance] = node

                except Exception as e:
                    logging.warning(f'Node<{node.pk}> processing failed: {e}')
                    continue

    def _flatten_data(self):
        flattened_list = []

        # Iterate over the nested dictionary:
        # StructureType -> Formula -> Plane -> Process -> Layers -> K_Dist -> Node
        for struct_type, formulas in self._data.items():
            for formula, planes in formulas.items():
                for plane, processes in planes.items():
                    for process_label, layers_dict in processes.items():
                        for layers, k_dists in layers_dict.items():
                            for k_dist, node in k_dists.items():
                                flattened_list.append({
                                    'Structure': struct_type,
                                    'Material': formula,
                                    'Plane': plane,
                                    'Process': process_label,
                                    'Layers': layers,
                                    'K_Dist': k_dist,
                                    'Status': self.get_status_string(node) + f' {node.pk}' if node else 'N/A',
                                })
        return flattened_list

    def fit(self, destpath=None, axs=None, **kwargs):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        base_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        # Use a list of structures to maintain order
        structures = sorted([s for s in self._data.keys() if s is not None], key=lambda x: str(x))
        all_planes = set()
        for struct in structures:
            for planes_dict in self._data[struct].values():
                all_planes.update(planes_dict.keys())
        planes = sorted(list(all_planes), key=lambda x: str(x))

        if not structures or not planes:
            return {}

        n_rows = len(structures)
        n_cols = len(planes)

        if axs is None:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

        results = {}

        for i, struct in enumerate(structures):
            results[struct] = {}
            for j, plane in enumerate(planes):
                results[struct][plane] = {}
                _base_colors = deepcopy(base_colors)
                ax = axs[i, j]

                mat_dict = self._data[struct]

                for formula, planes_dict in mat_dict.items():
                    if plane in planes_dict:
                        process_dict = planes_dict[plane]
                        color = _base_colors.pop(0) if _base_colors else None
                        if 'GSFEWorkChain' in process_dict:
                            for layers, k_dist_dict in process_dict['GSFEWorkChain'].items():
                                for k_dist, node in k_dist_dict.items():
                                    if node and node.is_finished_ok:
                                        print(node.pk, formula, plane)
                                        analyser = GSFEWorkChainAnalyserLatest(node)
                                        results[struct][plane][formula] = analyser.fit_curve(
                                            plot=True,
                                            axis=ax,
                                            label=formula_to_latex(formula),
                                            color=color,
                                            linestyle='-',
                                            lw=kwargs.get('lw', 1.5),
                                            **kwargs
                                        )

                ax.set_title(f"${struct}$ ({plane})", fontsize=kwargs.get('title_fontsize', 16))

        for ax in axs[:, 0]:
            ax.set_ylabel(r'$\gamma [J/m^2]$', fontsize=kwargs.get('ylabel_fontsize', 16))
        for ax in axs[-1, :]:
            ax.set_xlabel(r'$\vec{b}$', fontsize=kwargs.get('xlabel_fontsize', 16))

        if destpath and axs is None:
            plt.tight_layout()
            plt.savefig(destpath)
        return results

    def plot_kpoints_convergence(self, structure_type, formula, gliding_plane, n_repeats=None, ax=None, **kwargs):
        """Plot GSFE curves for different k-points on a single axis."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        struct_data = self._data.get(structure_type, {})
        formula_data = struct_data.get(formula, {})
        plane_data = formula_data.get(gliding_plane, {})
        process_data = plane_data.get('GSFEWorkChain', {})

        if not process_data:
            print(f"No GSFEWorkChain data found for {formula} {gliding_plane}")
            return ax

        if n_repeats is None:
            n_repeats = sorted(process_data.keys())[0]

        k_dist_dict = process_data.get(n_repeats, {})
        if not k_dist_dict:
            print(f"No data found for n_repeats={n_repeats}")
            return ax

        sorted_k_dists = sorted(k_dist_dict.keys(), reverse=True)
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=0, vmax=max(1, len(sorted_k_dists) - 1))

        for i, k_dist in enumerate(sorted_k_dists):
            node = k_dist_dict[k_dist]
            if node and node.is_finished_ok:
                analyser = GSFEWorkChainAnalyserLatest(node)
                color = cmap(norm(i))

                if kwargs.get('fit', True):
                    analyser.fit_curve(
                        plot=True,
                        axis=ax,
                        label=f"k-dist: {k_dist}",
                        color=color,
                        **kwargs
                    )
                else:
                    analyser.plot(
                        ax=ax,
                        label_prefix=f"{k_dist} ",
                        color=color,
                        zero_reference=kwargs.get('zero_reference', True),
                        **kwargs
                    )

        ax.set_title(f"K-points Convergence for {formula_to_latex(formula)} ({gliding_plane})")
        ax.set_ylabel(r'$\gamma [J/m^2]$')
        ax.set_xlabel(r'Displacement')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax
