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
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
from scipy.optimize import curve_fit, OptimizeWarning
from ..quantumespresso.pw_base import PwBaseWorkChainAnalyser
from ..quantumespresso.pw_relax import PwRelaxWorkChainAnalyser
from ..constants import eVA22Jm2
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

from pathlib import Path

def formula_to_latex(formula):
    latex_formula = re.sub(r'(\d+)', r'_{\1}', formula)
    return rf"${latex_formula}$"


def sine_expansion(x, cGs):
    """Generic sine series expansion: sum_{i=1}^N cG_i * sin(pi*x)**(2*i)."""
    sin_sq = numpy.sin(2*numpy.pi * x)**2
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

def gamma_usf(x, e_usf1):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: Expansion
    """
    return e_usf1 * numpy.sin(numpy.pi * x)**2

def gamma_usf_symmetric(x, e_usf1):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: e_usf1 * sin^2(pi*x)
    """
    return e_usf1 * numpy.sin(numpy.pi * x / 2)**2


def gamma_usf2(x, e_usf1, e_usf2):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: e_usf1 * sin^2(pi*x) + e_usf2 * sin^2(2*pi*x)
    """
    return (
        e_usf1 * numpy.sin(numpy.pi * x)**2 +
        e_usf2 * numpy.sin(2 * numpy.pi * x)**2
        )


def gamma_usf2_symmetric(x, e_usf1, e_usf2, e_usf3):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: e_usf1 * sin^2(pi*x) + e_usf2 * sin^2(2*pi*x)
    """
    return (
        e_usf1 * numpy.sin(numpy.pi/2 * x)**2 + 
        e_usf2 * numpy.sin(numpy.pi * x)**2 +
        e_usf3 * numpy.sin(2 * numpy.pi * x)**2
        )


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
        '011': {'100' : gamma_usf2, '010': gamma_usf2},
        '111': {'110' : gamma_esf},
    },
    'B2': {
        'gliding_system': B2GlidingSystem,
        '100': {'100' : gamma_usf},
        '011': {'100' : gamma_usf2, '010': gamma_usf2, '110': gamma_usf2},
        '111': {'110' : gamma_isf},
    },
    'C1_b': {
        'gliding_system': C1bGlidingSystem,
        '100': {'110' : gamma_usf},
        '011': {'100' : gamma_usf2, '010': gamma_usf2, '210': gamma_usf2},
        '111': {'110' : gamma_esf},
    },
    'L2_1': {
        'gliding_system': L21GlidingSystem,
        '100': {'110': gamma_usf_symmetric},
        '011': {'100': gamma_usf_symmetric, '010': gamma_usf2_symmetric, '210': gamma_usf2_symmetric},
        '111': {'110' : gamma_esf},
    },
}


class GSFERelaxWorkChainAnalyser(BaseWorkChainAnalyser):
    """Analyser for the current `GSFERelaxWorkChain` output contract."""

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct QE child to its own analyser."""
        def _resolve(child_name, child):
            process_label = child.node.process_label

            if process_label == 'PwRelaxWorkChain' and (
                child_name == 'relax' or child_name.startswith('structure_') or child_name.startswith('sfe_')
            ):
                return PwRelaxWorkChainAnalyser
            if process_label == 'PwBaseWorkChain' and (
                child_name == 'scf'
            ):
                return PwBaseWorkChainAnalyser
            return None

        return self._copy_tree_for_direct_children(destpath, _resolve)

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct QE child to its analyser."""
        def _resolve(child_name, child):
            process_label = child.node.process_label

            if process_label == 'PwRelaxWorkChain' and (
                child_name == 'relax' or child_name.startswith('structure_') or child_name.startswith('sfe_')
            ):
                return PwRelaxWorkChainAnalyser
            if process_label == 'PwBaseWorkChain' and (
                child_name == 'scf'
            ):
                return PwBaseWorkChainAnalyser
            return None

        return self._get_calcjob_paths_for_direct_children(_resolve)

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
    def relax(self):
        return self._get_node_from_tree('relax')

    @property
    def scf(self):
        return self._get_node_from_tree('scf')

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
        raise AttributeError(f'Node<{self.node.pk}>: GSFE results output is not found')

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
        raise AttributeError(f"Node<{self.node.pk}>: GSFE gliding plane not found in inputs")

    @property
    def gliding_system(self):
        """Return the gliding system object."""
        return fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(self.gliding_plane)

    def get_state(self) -> tuple[str, str, int]:
        """Get the state of the workchain."""
        subprocesses = []

        for label in self._get_child_labels(labels=('relax',), process_label='PwRelaxWorkChain'):
            subprocesses.append((label, PwRelaxWorkChainAnalyser))

        for label in self._get_child_labels(labels=('scf',), process_label='PwBaseWorkChain'):
            subprocesses.append((label, PwBaseWorkChainAnalyser))

        for label in self._get_child_labels(
            prefixes=('structure_', 'sfe_'),
            process_label='PwBaseWorkChain',
        ):
            subprocesses.append((label, PwBaseWorkChainAnalyser))

        return self._get_state_from_subprocesses(subprocesses)

    def get_results(self) -> dict[str, dict[str, dict[str, ty.Any]]]:
        """Return the nested GSFE result payload keyed by direction and step."""
        payload = self.results_node.get_dict()
        results = payload.get('results', payload)

        normalized: dict[str, dict[str, dict[str, ty.Any]]] = {}
        for direction, entries in results.items():
            normalized[direction] = {
                str(step): dict(entry)
                for step, entry in sorted(entries.items(), key=lambda item: int(item[0]))
            }
        return normalized

    def get_direction_results(self, direction_name: str) -> dict[str, dict[str, ty.Any]]:
        """Return all step results for a specific slip direction."""
        results = self.get_results()
        if direction_name not in results:
            raise KeyError(f'Node<{self.node.pk}>: direction `{direction_name}` is not present in GSFE results')
        return results[direction_name]


    # @property
    # def pristine_energy(self) -> float | None:
    #     """Return the pristine energy from the conventional structure."""
    #     if 'conventional_structure' in self.node.inputs:
    #         conventional_node = self.node.inputs.conventional_structure
    #         return self._get_safe_energy(conventional_node)
    #     return None
    @property
    def pristine_energy(self):
        """Get the pristine energy."""
        if 'structure_01' in self.process_tree:
            return self._get_safe_energy(self.process_tree.structure_01.node)
        
        # Try to find the first sfe_ child (e.g., sfe_111_000)
        sfe_labels = sorted([l for l in self.process_tree.children.keys() if l.startswith('sfe_')])
        if sfe_labels:
            return self._get_safe_energy(self.process_tree[sfe_labels[0]].node)
            
        raise AttributeError(f'Node<{self.node.pk}>: Pristine energy (structure_01 or sfe_*) not found in process tree')
    
    @property
    def conv_thr(self):
        """Return the convergence threshold of the calculation."""
        return self.node.inputs.sfe.base_relax.pw.parameters.get('ELECTRONS', {}).get('conv_thr', 1e-6)
    
    @property
    def conv_error(self):
        """Return the convergence error of the calculation."""
        return (self.conv_thr / self.surface_area) * eVA22Jm2
    
    def get_sfe_energies(self) -> dict[str, dict[int, float | None]]:
        """Return only the SFE values grouped by direction and step."""
        sfe_energies = {}
        pristine_energy = self.pristine_energy
        if pristine_energy is None:
            logging.warning(f"Node<{self.node.pk}>: Pristine energy not found, cannot calculate SFE.")
            return {}

        for direction, entries in self.get_results().items():
            sfe_energies[direction] = {}
            for step, entry in entries.items():
                total_energy_faulted_geometry = entry.get('energy')
                if total_energy_faulted_geometry is None:
                    logging.warning(f"Node<{self.node.pk}>: Energy not found for direction {direction}, step {step}.")
                    sfe_energies[direction][int(step)] = None
                    continue

                # Assuming 'conventional_multiplier' and 'faulted_multiplier' are available if needed
                # For now, using the direct SFE calculation from the workchain output if available,
                # otherwise calculating based on energy difference.
                sfe = entry.get('sfe')
                if sfe is None:
                    # If SFE is not directly in results, calculate it
                    # This calculation might need more context (e.g., surface area, number of layers)
                    # For now, a simple energy difference is used as a placeholder if 'sfe' is missing.
                    # The original code had `energy_difference = total_energy_faulted_geometry - self.pristine_energy`
                    # which implies SFE is just the energy difference.
                    sfe = total_energy_faulted_geometry - pristine_energy
                sfe_energies[direction][int(step)] = sfe
        return sfe_energies

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
        directions: list[str] | None = None,
    ) -> dict[str, dict[str, list[float | None]]]:
        """Return simple plot-ready x/y arrays for each GSFE direction."""
        plot_data: dict[str, dict[str, list[float | None]]] = {}

        for direction, entries in self.get_results().items():
            if directions is not None and direction not in directions:
                continue

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

    def serialize_faults(self) -> dict[str, list[float]]:
        """Serialize the faults for compatibility, normalized by nsteps."""
        serialized_faults = {}
        plot_data = self.get_plot_data()  # uses default x_axis='step'
        for direction, data in plot_data.items():
            xs = data['x']
            if None in xs:
                logging.warning(f"Node<{self.node.pk}>: Direction `{direction}` has `None` in `x` values, skipping")
                continue
            nsteps = len(xs) - 1
            if nsteps == 0:
                logging.warning(f"Node<{self.node.pk}>: Direction `{direction}` has 1 or fewer `x` values, skipping")
                continue
            serialized_faults[direction] = [val / nsteps for val in xs]
        return serialized_faults

    def fit_curve(self, fit_functions: dict[str, Callable | None] | None = None, plot: bool = False, axis=None, directions: list[str]|None = None, **kwargs):
        """Fit the curve."""
        from matplotlib.legend_handler import HandlerTuple

        def get_gradient_shades(hex_color, num=5):
            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", [hex_color, "#ffffff"])
            return [mcolors.to_hex(cmap(i)) for i in numpy.linspace(0, 0.8, num)]

        xs_dict = self.serialize_faults()
        sfe_energies = self.get_sfe_energies()
        # Filter energies first
        if directions is not None:
            filtered_energies = {d: v for d, v in sfe_energies.items() if d in directions}
        else:
            filtered_energies = sfe_energies

        energies = {
            direction: [val for _, val in sorted(steps.items())]
            for direction, steps in filtered_energies.items()
        }

        results = {}

        gliding_plane = self.gliding_plane
        gliding_system = self.gliding_system
        nsteps = gliding_system.general.nsteps
        if fit_functions is None:
            fit_functions = deepcopy(fit_function_map[self.strukturbericht][gliding_plane])


        sorted_keys = sorted(energies, key=lambda k: max(energies[k]), reverse=True)
        num_to_plot = len(sorted_keys)
        colors = itertools.cycle(get_gradient_shades(kwargs.get('color', 'black'), num=max(1, num_to_plot)))
        markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x'])
        for slipping_direction in sorted_keys:
            color = next(colors)
            results[slipping_direction] = {}
            func = fit_functions.get(slipping_direction)
            if func is None:
                logging.warning(f"Node<{self.node.pk}>: No fit function found for direction {slipping_direction}")
                continue
            logging.info(f"Node<{self.node.pk}>: Fitting slip system ({gliding_plane})<{slipping_direction}> using function: {func.__name__}")
            
            if slipping_direction not in xs_dict:
                logging.warning(f"Node<{self.node.pk}>: No x-axis data found for direction {slipping_direction}")
                continue

            x = numpy.array(xs_dict[slipping_direction], dtype=float)
            y = numpy.array(energies[slipping_direction], dtype=float)

            if kwargs.get('zero_reference', True):
                y = y - y[0]

            x_plot = numpy.linspace(0, x[-1], 500)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                try:
                    if func == gamma_isf:
                        b = y[nsteps]
                        order = kwargs.get('order', 1)
                        popt, _ = curve_fit(
                            lambda x, *cGs: func(x, cGs, b), 
                            x, y, 
                            p0=[0.1] * order, 
                            maxfev=100000
                            )
                        y_fit = func(x_plot, popt, b)
                        y_fit_orig = func(x, popt, b)
                        x_max = numpy.arcsin(-b / numpy.pi / popt[0]) / 2 * numpy.pi if popt[0] != 0 else 0.5
                        results[slipping_direction]['isf'] = b
                        results[slipping_direction]['usf'] = func(x_max, popt, b)

                    elif func == gamma_esf:
                        b = 2*y[nsteps]
                        c = y[2 * nsteps]*2 - b
                        order = kwargs.get('order', 4)
                        popt, _ = curve_fit(lambda x, *cGs: func(x, cGs, b, c), x, y, p0=[0.1] * order, maxfev=1000000)
                        y_fit = func(x_plot, popt, b, c)
                        y_fit_orig = func(x, popt, b, c)
                        results[slipping_direction]['usf'] = numpy.max(y_fit[:250])
                        results[slipping_direction]['isf'] = b
                        results[slipping_direction]['ut'] = numpy.max(y_fit[250:])
                        results[slipping_direction]['esf'] = c

                    elif func == gamma_usf:
                        popt, _ = curve_fit(
                            lambda x, e_usf1: func(x, e_usf1), 
                            x, y, 
                            p0=[0.1], 
                            maxfev=100000
                            )
                        y_fit = func(x_plot, popt[0])
                        y_fit_orig = func(x, popt[0])
                        results[slipping_direction]['usf'] = popt[0]

                    elif func == gamma_usf_symmetric:
                        popt, _ = curve_fit(
                            lambda x, e_usf1: func(x, e_usf1), 
                            x, y, 
                            p0=[0.1], 
                            maxfev=100000
                            )
                        y_fit = func(x_plot, popt[0])
                        y_fit_orig = func(x, popt[0])
                        results[slipping_direction]['usf'] = popt[0]

                    elif func == gamma_usf2:
                        (e_usf1, e_usf2), pcov = curve_fit(func, x, y, p0=[0.1, 0.1], maxfev=100000)
                        y_fit = func(x_plot, e_usf1, e_usf2)
                        y_fit_orig = func(x, e_usf1, e_usf2)
                        results[slipping_direction]['usf'] = numpy.max(y_fit)
                        results[slipping_direction]['s'] = e_usf2

                    elif func == gamma_usf2_symmetric:
                        e_usf1 = y[-1]
                        (e_usf2, e_usf3), pcov = curve_fit(
                            lambda x, e_usf2, e_usf3: func(x, e_usf1, e_usf2, e_usf3), 
                            x, y, 
                            p0=[0.1, 0.1], 
                            maxfev=100000
                            )
                        y_fit = func(x_plot, e_usf1, e_usf2, e_usf3)
                        y_fit_orig = func(x, e_usf1, e_usf2, e_usf3)
                        results[slipping_direction]['usf'] = numpy.max(y_fit)
                        results[slipping_direction]['s'] = e_usf2

                    # Calculate fit quality metrics
                    residuals = y - y_fit_orig
                    ss_res = numpy.sum(residuals**2)
                    ss_tot = numpy.sum((y - numpy.mean(y))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
                    rmse = numpy.sqrt(numpy.mean(residuals**2))

                    # Log extracted parameters and fit quality
                    params_str = ", ".join([f"{k}={v:.4f}" for k, v in results[slipping_direction].items() if isinstance(v, (int, float, numpy.floating))])
                    logging.info(
                        f"Node<{self.node.pk}>: Fitting slip system ({gliding_plane})<{slipping_direction}> completed. "
                        f"Extracted parameters: {params_str} | Fit quality: R²={r_squared:.6f}, RMSE={rmse:.6f} J/m²"
                    )
                    
                except Exception as e:
                    logging.warning(f"Node<{self.node.pk}>: Fitting failed for {slipping_direction}: {e}")
                    continue

            if plot:
                if axis is None:
                    import matplotlib.pyplot as plt
                    fig, axis = plt.subplots(figsize=(10, 6))
                axis.scatter(
                    x, y,
                    color=color,
                    s=50,
                    zorder=5,
                    marker=next(markers))

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
        directions: list[str] | None = None,
        **kwargs,
    ):
        """Plot the selected GSFE quantity for each direction."""
        if value not in {'sfe', 'energy'}:
            raise ValueError(f'unsupported value `{value}`, expected `sfe` or `energy`')

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        plot_data = self.get_plot_data(x_axis=x_axis, zero_reference=zero_reference, directions=directions)

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


class GSFERelaxGroupData(BaseGroupData):

    def __init__(self, groups=None):
        super().__init__(groups)
        # Data structure: StructureType -> Material -> Plane -> Process -> Layers -> K_Dist -> Conv_thr -> Node
        self._data = defaultdict(
            lambda: defaultdict(
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

                    logging.info(f"Processing node<{node.pk}> for {formula}")
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
                    conv_thr = node.inputs.sfe.base_relax.pw.parameters.get('ELECTRONS', None).get('conv_thr', 1e-6)

                    # Structure: StructureType -> Formula -> Plane -> Process -> Layers -> K_Dist -> Conv_thr -> Node
                    if process_label in ['GSFERelaxWorkChain']:
                        self._data[structuretype][formula][gliding_plane][process_label][n_repeats][kpoints_distance][conv_thr] = node

                except Exception as e:
                    logging.warning(f'Node<{node.pk}> processing failed: {e}')
                    continue

    def _flatten_data(self):
        flattened_list = []

        # Iterate over the nested dictionary:
        # StructureType -> Formula -> Plane -> Process -> Layers -> K_Dist -> Conv_thr -> Node
        for struct_type, formulas in self._data.items():
            for formula, planes in formulas.items():
                for plane, processes in planes.items():
                    for process_label, layers_dict in processes.items():
                        for layers, k_dists in layers_dict.items():
                            for k_dist, conv_thr_dict in k_dists.items():
                                for conv_thr, node in conv_thr_dict.items():
                                    if node.is_finished_ok:
                                        analyser = GSFERelaxWorkChainAnalyser(node)
                                        flattened_list.append({
                                            'Structure': struct_type,
                                            'Material': formula,
                                            'Plane': plane,
                                            'Process': process_label,
                                            'Layers': layers,
                                            'K_Dist': k_dist,
                                            'Conv_thr': rf'{conv_thr:.1e} Ry (+- {analyser.conv_error*1000:.1e} mJ/m^2)',
                                            'Status': self.get_status_string(node) + f' {node.pk}' if node else 'N/A',
                                        })
                                    else:
                                        flattened_list.append({
                                            'Structure': struct_type,
                                            'Material': formula,
                                            'Plane': plane,
                                            'Process': process_label,
                                            'Layers': layers,
                                            'K_Dist': k_dist,
                                            'Conv_thr': 'N/A',
                                            'Status': self.get_status_string(node) + f' {node.pk}' if node else 'N/A',
                                        })
        return flattened_list

    def fit(self, destpath=None, axs=None, **kwargs):
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
                
                base_colors = itertools.cycle([
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
                ])
                markers = itertools.cycle(['o', 's', 'v', '^', '<', '>', '8', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])

                ax = axs[i, j]

                mat_dict = self._data[struct]

                for formula, planes_dict in mat_dict.items():
                    if plane in planes_dict:
                        process_dict = planes_dict[plane]
                        color = next(base_colors)
                        marker = next(markers)
                        if 'GSFERelaxWorkChain' in process_dict:
                            for layers, k_dist_dict in process_dict['GSFERelaxWorkChain'].items():
                                for k_dist, conv_thr_dict in k_dist_dict.items():
                                    for conv_thr, node in conv_thr_dict.items():
                                        if node and node.is_finished_ok:
                                            # print(node.pk, formula, plane)
                                            logging.info(f"Fitting node<{node.pk}> for {formula} {plane}")
                                            analyser = GSFERelaxWorkChainAnalyser(node)
                                            results[struct][plane][formula] = analyser.fit_curve(
                                                axis=ax,
                                                plot=True,
                                                label=formula_to_latex(formula),
                                                color=color,
                                                linestyle='-',
                                                lw=kwargs.get('lw', 1.5),
                                                **kwargs
                                            )

                ax.set_title(f"${struct}$ ({plane})", fontsize=kwargs.get('title_fontsize', 16))

        for ax in axs[:, 0]:
            ax.set_ylabel(r'$\gamma^{GSFE}$ (J/m$^2$)', fontsize=kwargs.get('ylabel_fontsize', 16))
        for ax in axs[-1, :]:
            ax.set_xlabel(r'$\vec{b}$', fontsize=kwargs.get('xlabel_fontsize', 16))

        if destpath and axs is None:
            plt.tight_layout()
            plt.savefig(destpath)
        return results

    def plot_kpoints_convergence(self, structure_type, formula, gliding_plane, n_repeats=None, ax=None, kpoints_distances=None, directions=None, **kwargs):
        """Plot GSFE curves for different k-points on a single axis."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        struct_data = self._data.get(structure_type, {})
        formula_data = struct_data.get(formula, {})
        plane_data = formula_data.get(gliding_plane, {})
        process_data = plane_data.get('GSFERelaxWorkChain', {})

        if not process_data:
            print(f"No GSFERelaxWorkChain data found for {formula} {gliding_plane}")
            return ax

        if n_repeats is None:
            n_repeats = sorted(process_data.keys())[0]

        k_dist_dict = process_data.get(n_repeats, {})
        if not k_dist_dict:
            print(f"No data found for n_repeats={n_repeats}")
            return ax

        if kpoints_distances is not None:
            filtered_k_dists = {k: v for k, v in k_dist_dict.items() if k in kpoints_distances}
        else:
            filtered_k_dists = k_dist_dict

        sorted_k_dists = sorted(filtered_k_dists.keys(), reverse=True)
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=0, vmax=max(1, len(sorted_k_dists) - 1))

        kpoints_convergence_results = {}

        for i, k_dist in enumerate(sorted_k_dists):
            node = filtered_k_dists[k_dist]
            if node and node.is_finished_ok:
                analyser = GSFERelaxWorkChainAnalyser(node)
                color = cmap(norm(i))

                if kwargs.get('fit', True):
                    results = analyser.fit_curve(
                        plot=True,
                        axis=ax,
                        label=f"{k_dist}",
                        color=color,
                        directions=directions,
                        **kwargs
                    )
                    kpoints_convergence_results[k_dist] = results
                else:
                    analyser.plot(
                        ax=ax,
                        label_prefix=f"{k_dist} ",
                        color=color,
                        zero_reference=kwargs.get('zero_reference', True),
                        directions=directions,
                        **kwargs
                    )

        ax.set_title(f"K-points Convergence for {formula_to_latex(formula)} ({gliding_plane})")
        ax.set_ylabel(r'$\gamma^{GSFE}$ (J/m$^2$)')
        ax.set_xlabel(r'Displacement')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return (ax, kpoints_convergence_results)


    def plot_supercell_convergence(self, structure_type, formula, gliding_plane, kpoints_distances, n_repeats=None, ax=None, directions=None, **kwargs):
        """Plot GSFE curves for different k-points on a single axis."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        struct_data = self._data.get(structure_type, {})
        formula_data = struct_data.get(formula, {})
        plane_data = formula_data.get(gliding_plane, {})
        process_data = plane_data.get('GSFERelaxWorkChain', {})

        if not process_data:
            print(f"No GSFERelaxWorkChain data found for {formula} {gliding_plane}")
            return ax

        if n_repeats is not None:
            filtered_n_repeats_dict = {n: v for n, v in process_data.items() if n in n_repeats}
        else:
            filtered_n_repeats_dict = process_data

        sorted_n_repeats_dict = sorted(filtered_n_repeats_dict.keys(), reverse=True)
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=0, vmax=max(1, len(sorted_n_repeats_dict) - 1))


        for i, n_repeats_dist in enumerate(sorted_n_repeats_dict):
            kpoints_distances_dict = filtered_n_repeats_dict[n_repeats_dist]
            if kpoints_distances is None:
                k_dist = sorted(kpoints_distances_dict.keys())[0]
                node = kpoints_distances_dict[k_dist]
            else:
                node = kpoints_distances_dict[kpoints_distances]
            if node and node.is_finished_ok:
                analyser = GSFERelaxWorkChainAnalyser(node)
                color = cmap(norm(i))

                if kwargs.get('fit', True):
                    analyser.fit_curve(
                        plot=True,
                        axis=ax,
                        label=f"{n_repeats_dist}",
                        color=color,
                        directions=directions,
                        **kwargs
                    )
                else:
                    analyser.plot(
                        ax=ax,
                        label_prefix=f"{k_dist} ",
                        color=color,
                        zero_reference=kwargs.get('zero_reference', True),
                        directions=directions,
                        **kwargs
                    )

        ax.set_title(f"Supercell Convergence for {formula_to_latex(formula)} ({gliding_plane})")
        ax.set_ylabel(r'$\gamma^{GSFE}$ (J/m$^2$)')
        ax.set_xlabel(r'Displacement')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def dump(self, dest:Path|str, struct_type_list:list = None, formula_list:list = None, planes_list:list = None, process_label_list:list = None, layers_list:list = None, k_dist_list:list = None,):
        if type(dest) == str:
            dest = Path(dest)
        if not dest.exists():
            dest.mkdir(parents=True)

        for struct_type, formulas in self._data.items():
            if struct_type_list and struct_type not in struct_type_list:
                continue
            for formula, planes in formulas.items():
                if formula_list and formula not in formula_list:
                    continue
                for plane, processes in planes.items():
                    if planes_list and plane not in planes_list:
                        continue
                    for process_label, layers_dict in processes.items():
                        if process_label_list and process_label not in process_label_list:
                            continue
                        for layers, k_dists in layers_dict.items():
                            for k_dist, conv_thr_dict in k_dists.items():
                                for conv_thr, node in conv_thr_dict.items():
                                    if node:
                                        analyser = GSFERelaxWorkChainAnalyser(node)
                                        analyser.copy_tree(
                                            dest / struct_type / formula / plane / process_label / f"{layers}" / f"{k_dist}" / f"{conv_thr}" / f"{node.pk}"
                                            )
