from __future__ import annotations

import re
import typing as ty
from collections import defaultdict
from copy import deepcopy

import numpy
from scipy.optimize import curve_fit

from aiida import orm

from ..core.base import BaseWorkChainAnalyser
from ..core.groupdata import BaseGroupData, render_process_node_details
import logging
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
from scipy.optimize import curve_fit, OptimizeWarning
from ..quantumespresso.pw_base import PwBaseAnalyser
from ..quantumespresso.pw_relax import PwRelaxAnalyser

from aiida_dislocation.tools import (
    get_strukturbericht,
    calculate_surface_area
)

from .fit_functions import (
    formula_to_latex,

    fit_function_map,
    fit_gsfe,
)
import re
from copy import deepcopy
import itertools

from pathlib import Path

class GSFEAnalyserLatest(BaseWorkChainAnalyser):
    """Analyser for the current `GSFEWorkChain` output contract."""

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct QE child to its own analyser."""
        def _resolve(child_name, child):
            process_label = child.node.process_label

            if process_label == 'PwRelaxWorkChain':
                return PwRelaxAnalyser
            if process_label == 'PwBaseWorkChain' and (
                child_name == 'scf' or child_name.startswith('structure_') or child_name.startswith('sfe_')
            ):
                return PwBaseAnalyser
            return None

        return self._copy_tree_for_direct_children(destpath, _resolve)

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct QE child to its analyser."""
        def _resolve(child_name, child):
            process_label = child.node.process_label

            if process_label == 'PwRelaxWorkChain':
                return PwRelaxAnalyser
            if process_label == 'PwBaseWorkChain' and (
                child_name == 'scf' or child_name.startswith('structure_') or child_name.startswith('sfe_')
            ):
                return PwBaseAnalyser
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
    def surface_energy(self):
        return self._get_node_from_tree('surface_energy')
    
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
    def conv_thr(self) -> float | None:
        """Return the convergence threshold of the calculation."""
        try:
            return float(self.node.inputs.sfe.pw.parameters.get('ELECTRONS', {}).get('conv_thr'))
        except Exception:
            return None

    @property
    def conv_error(self) -> float | None:
        """Return the convergence error of the Stacking Fault Stretches calculation."""
        conv_thr = self.conv_thr
        surface_area = self.surface_area
        if conv_thr is None or surface_area is None:
            return None
        return (conv_thr / surface_area) * self._eVA22Jm2

    @property
    def scf_energy(self) -> float | None:
        return self._get_safe_energy(self.scf)

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
            raise KeyError(f'direction `{direction_name}` is not present in GSFE results')
        return results[direction_name]

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
    
    def get_sfe_energies(self, zero_reference: bool = False) -> dict[str, dict[int, float | None]]:
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
            if zero_reference:
                energy_ref = sfe_energies[direction][0]
                for step in sfe_energies[direction]:
                    sfe_energies[direction][step] -= energy_ref
                
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

    def fit_curve(self, fit_functions: dict[str, Callable | None] | None = None, plot=False, axis=None, directions: list[str]|None = None, **kwargs):
        """Fit the curve."""
        from matplotlib.legend_handler import HandlerTuple

        def get_gradient_shades(hex_color, num=5):
            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", [hex_color, "#ffffff"])
            return [mcolors.to_hex(cmap(i)) for i in numpy.linspace(0, 0.8, num)]

        xs_dict = self.serialize_faults()
        sfe_energies = self.get_sfe_energies(zero_reference=kwargs.get('zero_reference', True))
        
        # Filter energies first
        if directions is not None:
            filtered_energies = {d: v for d, v in sfe_energies.items() if d in directions}
        else:
            filtered_energies = sfe_energies

        # Convert sfe_energies to the format expected by the port: direction -> [value, ...]
        energies = {
            direction: [val for _, val in sorted(steps.items())]
            for direction, steps in filtered_energies.items()
        }

        results = {}

        try:
            formula = self.node.inputs.structure.get_formula()
        except Exception:
            formula = "Unknown"

        gliding_plane = self.gliding_plane
        gliding_system = self.gliding_system
        nsteps = gliding_system.general.nsteps
        if fit_functions is None:
            fit_functions = deepcopy(fit_function_map[self.strukturbericht][gliding_plane])

        sorted_keys = sorted(energies, key=lambda k: max(energies[k]))
        num_to_plot = len(sorted_keys)
        colors = itertools.cycle(get_gradient_shades(kwargs.get('color', 'black'), num=max(1, num_to_plot)))
        markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x'])
        
        logging.info(f"Node<{self.node.pk}>: Starting GSFE curve fitting for {formula} on plane {gliding_plane}")
        for slipping_direction in sorted_keys:
            color = next(colors)
            results[slipping_direction] = {}
            func = fit_functions.get(slipping_direction)
            if func is None:
                logging.warning(f"Node<{self.node.pk}>: No fit function found for direction {slipping_direction}")
                continue
            logging.info(f"Node<{self.node.pk}>: Fitting slip system <{slipping_direction}> using function: {func.__name__}")
            
            if slipping_direction not in xs_dict:
                logging.warning(f"Node<{self.node.pk}>: No x-axis data found for direction {slipping_direction}")
                continue

            x = numpy.array(xs_dict[slipping_direction], dtype=float)
            y = numpy.array(energies[slipping_direction], dtype=float)

            x_plot = numpy.linspace(0, x[-1], 500)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                try:
                    popt, y_fit, y_fit_orig, results[slipping_direction] = fit_gsfe(
                        func, x, y, nsteps, x_plot, is_mJ=False, **kwargs
                    )
                    
                    # Calculate fit quality metrics
                    residuals = y - y_fit_orig
                    ss_res = numpy.sum(residuals**2)
                    ss_tot = numpy.sum((y - numpy.mean(y))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
                    rmse = numpy.sqrt(numpy.mean(residuals**2))

                    # Log extracted parameters and fit quality
                    params_str = ", ".join([f"{k}={v:.4f} mJ/m²" for k, v in results[slipping_direction].items() if isinstance(v, (int, float, numpy.floating))])
                    logging.info(
                        f"Node<{self.node.pk}>: Fitting slip system <{slipping_direction}> completed. "
                        f"Extracted parameters: {params_str} | Fit quality: R²={r_squared:.6f}, RMSE={rmse:.6f} J/m²"
                    )
                except Exception as e:
                    logging.warning(f"Node<{self.node.pk}>: Fitting failed for <{slipping_direction}>: {e}")
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
                    label=kwargs.get('label', f'{kwargs.get("label_prefix", "")}{slipping_direction}')
                )
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
    
    def get_surface_energy(self):
        """Get the surface energy of the workchain."""
        from ase.formula import Formula
        from aiida_dislocation.tools import calculate_surface_area
        from aiida_dislocation.workflows.gsfe import GSFEWorkChain

        logging.warning(
            "In the latest version of GSFEWorkChain, the `surface_energy` subprocess is removed and isolated into an independent workchain. "
            "This function is kept for backward compatibility but for new workchains, please use the independent `SurfaceEnergyWorkChain`."
        )

        if any(
            namespace not in self.process_tree for namespace in (GSFEWorkChain._SCF_NAMESPACE, GSFEWorkChain._SURFACE_ENERGY_NAMESPACE)
        ):
            logging.warning(f'Node<{self.node.pk}>: {GSFEWorkChain._SCF_NAMESPACE} or {GSFEWorkChain._SURFACE_ENERGY_NAMESPACE} is not found')
            return None
        
        conventional_formula = Formula(self.scf.inputs.pw.structure.get_ase().get_chemical_formula())
        _, conventional_multiplier = conventional_formula.reduce()
        
        surface_formula = Formula(self.surface_energy.inputs.pw.structure.get_ase().get_chemical_formula())
        _, surface_multiplier = surface_formula.reduce()
        
        surface_area = calculate_surface_area(self.scf.inputs.pw.structure.get_ase())
        
        total_energy_cleavaged_geometry = self._get_safe_energy(self.surface_energy)
        if total_energy_cleavaged_geometry is None:
            logging.warning(f"Node<{self.surface_energy.pk}>: energy not found in output_parameters")
            return None
        energy_difference = total_energy_cleavaged_geometry - self.scf_energy * surface_multiplier / conventional_multiplier
        surface_energy = energy_difference / (2*surface_area) * self._eVA22Jm2
        
        return surface_energy

class GSFEGroupDataLatest(BaseGroupData):

    def __init__(self, groups=None):
        super().__init__(groups)
        # Data structure: StructureType -> Formula -> Plane -> Process -> Layers -> K_Dist -> Conv_thr -> Node
        self._data = defaultdict(
            lambda: defaultdict(
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
        )
        self.get_data()

    @property
    def groups(self):
        return self._groups

    @property
    def data(self):
        return self._data

    def get_data(self):
        nodes_loaded = 0
        groups_loaded = 0
        for grpname in self._groups:
            try:
                group = orm.load_group(grpname)
                groups_loaded += 1
            except Exception as e:
                logging.error(f"Failed to load group {grpname}: {e}")
                continue
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
                    conv_thr = GSFEAnalyserLatest(node).conv_thr
                    if conv_thr is None:
                        conv_thr = 1e-6
                    # Structure: StructureType -> Formula -> Plane -> Process -> Layers -> K_Dist -> Conv_thr -> Node
                    if process_label in ['GSFEWorkChain']:
                        self._data[structuretype][formula][gliding_plane][process_label][n_repeats][kpoints_distance][conv_thr] = node
                        nodes_loaded += 1
                        logging.debug(f"Added Node<{node.pk}>: Structure={structuretype}, Formula={formula}, Plane={gliding_plane}, K-point={kpoints_distance}")

                except Exception as e:
                    logging.warning(f'Node<{node.pk}> processing failed: {e}')
                    continue
        
        logging.info(f"Finished loading GSFE data: imported {nodes_loaded} total nodes from {groups_loaded} groups.")
    def get_surface_energies(self):
        results = {}
        structures = sorted(self._data.keys(), key=lambda x: str(x))
        all_planes = set()
        for struct in structures:
            for planes_dict in self._data[struct].values(): # val is dict of planes
                all_planes.update(planes_dict.keys())
        planes = sorted(list(all_planes), key=lambda x: str(x))
        
        for struct in structures:
            for plane in planes:
                
                # Check if we have data for this cell
                mat_dict = self._data[struct]
                
                found_any = False
                for formula, planes_dict in mat_dict.items():
                    if plane in planes_dict:
                        # Found data for this (Struct, Plane) for this Formula
                        process_dict = planes_dict[plane]
                        if 'GSFEWorkChain' in process_dict:
                            layers_dict = process_dict['GSFEWorkChain']
                            for layers, k_dist_dict in layers_dict.items():
                                for k_dist, conv_thr_dict in k_dist_dict.items():
                                    for conv_thr, node in conv_thr_dict.items():
                                        if node and node.is_finished_ok:
                                            try:
                                                analyser = GSFEAnalyserLatest(node)
                                                # fit_curve plots on axis if provided
                                                # Label with formula, maybe layers/kdist if distinct?
                                                # For now just formula as typically we compare materials
                                                label = f'{formula}'
                                                res = analyser.get_surface_energy()
                                                
                                                # Store results
                                                if struct not in results: results[struct] = {}
                                                if plane not in results[struct]: results[struct][plane] = {}
                                                results[struct][plane][formula] = {
                                                    'surface_energy': res
                                                    }
                                                found_any = True
                                            except Exception as e:
                                                print(f"Failed to retrieve surface energy from {node.pk}: {e}")
        return results

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
                                    if node and node.is_finished_ok:
                                        analyser = GSFEAnalyserLatest(node)
                                        conv_error = analyser.conv_error
                                        if conv_error is not None:
                                            conv_thr_str = rf'{conv_thr:.1e} Ry (+- {conv_error*1000:.1e} mJ/m^2)'
                                        else:
                                            conv_thr_str = rf'{conv_thr:.1e} Ry'
                                    else:
                                        conv_thr_str = 'N/A'
                                    flattened_list.append({
                                        'StructureType': struct_type,
                                        'Formula': formula,
                                        'Plane': plane,
                                        'Process': process_label,
                                        'Layers': layers,
                                        'K_Dist': k_dist,
                                        'Conv_thr': conv_thr_str,
                                        'Status': self.get_status_string(node) + f' {node.pk}' if node else 'N/A',
                                    })
        return flattened_list

    def _flatten_data_flat(self):
        flattened_list = []
        # StructureType -> Formula -> Plane -> Process -> Layers -> K_Dist -> Conv_thr -> Node
        for struct_type, formulas in self._data.items():
            for formula, planes in formulas.items():
                for plane, processes in planes.items():
                    for process_label, layers_dict in processes.items():
                        for layers, k_dists in layers_dict.items():
                            for k_dist, conv_thr_dict in k_dists.items():
                                for conv_thr, node in conv_thr_dict.items():
                                    if node:
                                        flattened_list.append({
                                            'PK': node.pk,
                                            'Structure': struct_type,
                                            'Material': formula,
                                            'Plane': plane,
                                            'Process': process_label,
                                            'Layers': layers,
                                            'K_Dist': k_dist,
                                            'Status': self.get_status_string(node),
                                            'node': node,
                                        })
        return flattened_list

    def show_interactive(self):
        """
        Displays an interactive Jupyter table of GSFEWorkChain nodes.
        Clicking on a row triggers a Python callback to highlight that row
        and display the node's full nested parameters in a collapsible HTML viewer.
        """
        import ipywidgets as widgets
        from IPython.display import display
        import pandas as pd
        
        flat_data = self._flatten_data_flat()
        if not flat_data:
            print("No data available to display.")
            return

        df = pd.DataFrame(flat_data)
        df.index = df['PK'].map(int)

        node_map = {int(item['PK']): item['node'] for item in flat_data if item['node'] is not None}
        
        details_output = widgets.Output()
        
        def render_node_details(node):
            return render_process_node_details(node)

        # Table headers styled with flexbox to match row layout
        headers = widgets.HTML(f"""
        <div style="display: flex; align-items: center; background-color: #2c3e50; color: #ffffff; font-weight: bold; padding: 8px; border-radius: 4px 4px 0 0; width: 100%; box-sizing: border-box;">
            <div style="width: 80px; text-align: center; flex-shrink: 0;">Select</div>
            <div style="width: 80px; flex-shrink: 0; padding-left: 8px;">PK</div>
            <div style="width: 90px; flex-shrink: 0;">Structure</div>
            <div style="width: 120px; flex-shrink: 0;">Material</div>
            <div style="width: 80px; flex-shrink: 0;">Plane</div>
            <div style="width: 130px; flex-shrink: 0;">Process</div>
            <div style="width: 70px; flex-shrink: 0;">Layers</div>
            <div style="width: 80px; flex-shrink: 0;">K_Dist</div>
            <div style="width: 100px; flex-grow: 1;">Status</div>
        </div>
        """, layout=widgets.Layout(width='100%'))
        
        row_fields = {} # pk -> (btn, html_widget, row_box)
        
        def select_row(selected_pk):
            selected_pk = int(selected_pk)
            for pk, (btn, html_widget, row_box) in row_fields.items():
                row_data = df.loc[pk]
                struct_val = str(row_data['Structure'])
                material_val = str(row_data['Material'])
                plane_val = str(row_data['Plane'])
                process_val = str(row_data['Process'])
                layers_val = str(row_data['Layers'])
                k_dist_val = str(row_data['K_Dist'])
                status_val = str(row_data['Status'])
                
                if pk == selected_pk:
                    btn.button_style = 'success'
                    btn.icon = 'check-circle'
                    bg_color = "#e8f4fd"
                    border_style = "border-left: 4px solid #3498db; padding-left: 4px;"
                else:
                    btn.button_style = ''
                    btn.icon = 'circle-o'
                    bg_color = "transparent"
                    border_style = "padding-left: 8px;" # match selected padding to prevent alignment shifts
                
                # HTML content of columns styled to avoid overflow scrollbars
                html_widget.value = f"""
                <div style="display: flex; align-items: center; background-color: {bg_color}; {border_style} width: 100%; height: 28px; box-sizing: border-box; overflow: hidden;">
                    <div style="width: 80px; flex-shrink: 0; font-family: monospace;">{pk}</div>
                    <div style="width: 90px; flex-shrink: 0;">{struct_val}</div>
                    <div style="width: 120px; flex-shrink: 0; font-weight: bold;">{material_val}</div>
                    <div style="width: 80px; flex-shrink: 0;">{plane_val}</div>
                    <div style="width: 130px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{process_val}</div>
                    <div style="width: 70px; flex-shrink: 0; font-family: monospace;">{layers_val}</div>
                    <div style="width: 80px; flex-shrink: 0; font-family: monospace;">{k_dist_val}</div>
                    <div style="width: 100px; flex-grow: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{status_val}</div>
                </div>
                """
                
            node = node_map.get(selected_pk)
            with details_output:
                details_output.clear_output()
                if node:
                    display(widgets.HTML(render_node_details(node)))

        rows = []
        for pk, row_data in df.iterrows():
            pk = int(pk)
            
            # Select button
            btn = widgets.Button(
                description="",
                icon="circle-o",
                tooltip=f"Select Node {pk}",
                layout=widgets.Layout(width='40px', height='26px')
            )
            
            # Button click handler
            def on_button_click(b, target_pk=pk):
                select_row(target_pk)
            btn.on_click(on_button_click)
            
            # HTML container for row columns
            html_widget = widgets.HTML(
                layout=widgets.Layout(width='100%', overflow='hidden')
            )
            
            # Row box with HBox
            row_box = widgets.HBox(
                [widgets.Box([btn], layout=widgets.Layout(width='80px', justify_content='center', flex_shrink=0)), html_widget],
                layout=widgets.Layout(width='100%', overflow='hidden', border_bottom='1px solid #ecf0f1', padding='4px 0')
            )
            
            row_fields[pk] = (btn, html_widget, row_box)
            rows.append(row_box)
            
        table_body = widgets.VBox(rows, layout=widgets.Layout(max_height='400px', overflow_y='auto', border='1px solid #ecf0f1', border_top='none', border_radius='0 0 4px 4px'))
        
        table_container = widgets.VBox([headers, table_body], layout=widgets.Layout(width='62%'))
        
        details_container = widgets.VBox([
            details_output
        ], layout=widgets.Layout(width='36%', margin='0 0 0 2%'))
        
        main_layout = widgets.HBox([table_container, details_container], layout=widgets.Layout(width='100%'))
        
        if not df.empty:
            select_row(df.index[0])
            
        display(main_layout)

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
                        if 'GSFEWorkChain' in process_dict:
                            for layers, k_dist_dict in process_dict['GSFEWorkChain'].items():
                                for k_dist, conv_thr_dict in k_dist_dict.items():
                                    for conv_thr, node in conv_thr_dict.items():
                                        if node and node.is_finished_ok:
                                            # print(node.pk, formula, plane)
                                            analyser = GSFEAnalyserLatest(node)
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
            ax.set_xlabel(r'$\mathbf{b}$', fontsize=kwargs.get('xlabel_fontsize', 16))

        if destpath and axs is None:
            plt.tight_layout()
            plt.savefig(destpath)
            
        logging.info(f"GSFE Fit Summary: Successfully processed and fitted data for {len(results)} structure types across {len(planes)} planes.")
        return results

    def plot_kpoints_convergence(self, structure_type, formula, gliding_plane, n_repeats=None, ax=None, kpoints_distances=None, directions=None, **kwargs):
        """Plot GSFE curves for different k-points on a single axis."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        legend_ncol = kwargs.pop('legend_ncol', 1)
        logging.info(f"Plotting k-points convergence for {formula} ({gliding_plane})...")

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        struct_data = self._data.get(structure_type, {})
        formula_data = struct_data.get(formula, {})
        plane_data = formula_data.get(gliding_plane, {})
        process_data = plane_data.get('GSFEWorkChain', {})

        if not process_data:
            logging.warning(f"No GSFEWorkChain data found for {formula} {gliding_plane}")
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
            for conv, node in filtered_k_dists[k_dist].items():
                if node and node.is_finished_ok:
                    analyser = GSFEAnalyserLatest(node)
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
                            label=f"{k_dist} ",
                            color=color,
                            zero_reference=kwargs.get('zero_reference', True),
                            directions=directions,
                            **kwargs
                        )

        ax.set_title(f"K-points Convergence for {formula_to_latex(formula)} ({gliding_plane})")
        ax.set_ylabel(r'$\gamma^{GSFE}$ (J/m$^2$)')
        ax.set_xlabel(r'$\mathbf{b}$')
        ax.legend(ncol = legend_ncol)
        ax.grid(True, alpha=0.3)
        
        logging.info(f"Successfully plotted k-points convergence with {len(kpoints_convergence_results)} different k-point distances.")

        return (ax, kpoints_convergence_results)


    def plot_supercell_convergence(self, structure_type, formula, gliding_plane, kpoints_distance, n_repeats=None, ax=None, directions=None, **kwargs):
        """Plot GSFE curves for different k-points on a single axis."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        legend_ncol = kwargs.pop('legend_ncol', 1)
        logging.info(f"Plotting supercell convergence for {formula} ({gliding_plane})...")

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        struct_data = self._data.get(structure_type, {})
        formula_data = struct_data.get(formula, {})
        plane_data = formula_data.get(gliding_plane, {})
        process_data = plane_data.get('GSFEWorkChain', {})

        if not process_data:
            logging.warning(f"No GSFEWorkChain data found for {formula} {gliding_plane}")
            return ax

        if n_repeats is not None:
            filtered_n_repeats_dict = {n: v for n, v in process_data.items() if n in n_repeats}
        else:
            filtered_n_repeats_dict = process_data

        sorted_n_repeats_dict = sorted(filtered_n_repeats_dict.keys(), reverse=True)
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=0, vmax=max(1, len(sorted_n_repeats_dict) - 1))

        supercell_convergence_results = {}
        for i, n_repeats_dist in enumerate(sorted_n_repeats_dict):
            kpoints_distances_dict = filtered_n_repeats_dict[n_repeats_dist]
            if kpoints_distance is None:
                k_dist = sorted(kpoints_distances_dict.keys())[0]
            else:
                k_dist = kpoints_distances_dict[kpoints_distance]
            for conv, node in k_dist.items():
                if node and node.is_finished_ok:
                    analyser = GSFEAnalyserLatest(node)
                    color = cmap(norm(i))

                    if kwargs.get('fit', True):
                        results = analyser.fit_curve(
                            plot=True,
                            axis=ax,
                            label=f"{n_repeats_dist}",
                            color=color,
                            directions=directions,
                            **kwargs
                        )
                        supercell_convergence_results[n_repeats_dist] = results
                    else:
                        analyser.plot(
                            ax=ax,
                            label=f"{k_dist} ",
                            color=color,
                            zero_reference=kwargs.get('zero_reference', True),
                            directions=directions,
                            **kwargs
                        )

        ax.set_title(f"Supercell Convergence for {formula_to_latex(formula)} ({gliding_plane})")
        ax.set_ylabel(r'$\gamma^{GSFE}$ (J/m$^2$)')
        ax.set_xlabel(r'$\mathbf{b}$')
        ax.legend(ncol = legend_ncol)
        ax.grid(True, alpha=0.3)

        return (ax, supercell_convergence_results)

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
                                        analyser = GSFEAnalyserLatest(node)
                                        analyser.copy_tree(
                                            dest / struct_type / formula / plane / process_label / f"{layers}" / f"{k_dist}" / f"{conv_thr}" / f"{node.pk}"
                                            )
