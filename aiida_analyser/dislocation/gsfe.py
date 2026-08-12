from collections import defaultdict, deque
from typing import Callable
from aiida import orm
from ..quantumespresso.pw_base import PwBaseAnalyser
from ..quantumespresso.pw_relax import PwRelaxAnalyser
from ..core.base import BaseWorkChainAnalyser
from ..core.groupdata import BaseGroupData, render_process_node_details
import logging
from scipy.optimize import curve_fit
import numpy
from aiida_dislocation.tools.structure_utils import (
    get_strukturbericht
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

class GSFEAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the GsfeWorkChain.
    """

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
    
    @property
    def strukturbericht(self):
        return get_strukturbericht(self.scf.inputs.pw.structure.get_ase())

    @property
    def relax(self):
        return self._get_node_from_tree('relax')

    @property
    def surface_area(self) -> float:
        """Return the surface area of the calculation."""
        from aiida_dislocation.tools import calculate_surface_area
        return calculate_surface_area(self.scf.inputs.pw.structure.get_ase())

    @property
    def conv_thr(self) -> float | None:
        """Return the convergence threshold of the calculation."""
        try:
            return float(self.node.inputs.sfe.pw.parameters.get('ELECTRONS', {}).get('conv_thr'))
        except Exception:
            return None

    @property
    def conv_error(self) -> float | None:
        """Return the convergence error of the Stacking Stacking Stacking Fault Stretches calculation."""
        conv_thr = self.conv_thr
        surface_area = self.surface_area
        if conv_thr is None or surface_area is None:
            return None
        return (conv_thr / surface_area) * self._eVA22Jm2

    @property
    def scf(self):
        return self._get_node_from_tree('scf')

    @property
    def surface_energy(self):
        return self._get_node_from_tree('surface_energy')
    
    def get_state(self):
        """Get the state of the workchain."""
        subprocesses = []

        for label in self._get_child_labels(labels=('relax',), process_label='PwRelaxWorkChain'):
            subprocesses.append((label, PwRelaxAnalyser))

        for label in self._get_child_labels(labels=('scf',), process_label='PwBaseWorkChain'):
            subprocesses.append((label, PwBaseAnalyser))

        for label in self._get_child_labels(
            prefixes=('structure_', 'sfe_'),
            process_label='PwBaseWorkChain',
        ):
            subprocesses.append((label, PwBaseAnalyser))

        return self._get_state_from_subprocesses(subprocesses)

    @property
    def scf_energy(self):
        """Get the energy of the scf calculation."""

        return self._get_safe_energy(self.scf)

    @property
    def pristine_energy(self):
        """Get the pristine energy."""
        if 'structure_01' in self.process_tree:
            return self._get_safe_energy(self.process_tree.structure_01.node)
        
        # Try to find the first sfe_ child (e.g., sfe_111_000)
        sfe_labels = sorted([l for l in self.process_tree.children.keys() if l.startswith('sfe_')])
        if sfe_labels:
            return self._get_safe_energy(self.process_tree[sfe_labels[0]].node)
            
        raise AttributeError('Pristine energy (structure_01 or sfe_*) not found in process tree')
    
    def get_sfe_energies(self, zero_reference: bool = False):
        """Get the energies of the workchain."""
        from ase.formula import Formula
        from aiida_dislocation.tools import calculate_surface_area
        if 'gliding_plane' in self.node.inputs:
            gliding_plane = self.node.inputs.gliding_plane.value
        elif 'faulted_structure_data' in self.node.inputs:
            gliding_plane = self.node.inputs.faulted_structure_data.gliding_plane
        else:
            raise AttributeError("Neither 'gliding_plane' nor 'faulted_structure_data' found in inputs")
        gliding_system = fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(gliding_plane)

        surface_area = calculate_surface_area(self.scf.inputs.pw.structure.get_ase())
        
        # Find all SFE subprocesses
        sfe_children = []
        for call_link_label, child in self.process_tree.children.items():
            if call_link_label.startswith('structure_') or call_link_label.startswith('sfe_'):
                sfe_children.append((call_link_label, child))
        
        # Sort them by their label (natural sort to avoid sfe_10 < sfe_2)
        sfe_children.sort(key=lambda x: [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', x[0])])

        _energies = deque()
        for call_link_label, child in sfe_children:
            total_energy_faulted_geometry = self._get_safe_energy(child.node)
            if total_energy_faulted_geometry is None:
                logging.warning(f"Node<{child.node.pk}>: energy not found in output_parameters")
                continue
            # energy_difference = total_energy_faulted_geometry - total_energy_conventional_geometry / conventional_multiplier * faulted_multiplier
            energy_difference = total_energy_faulted_geometry - self.pristine_energy
            faulted_stacking_fault_energy = energy_difference / surface_area * self._eVA22Jm2
            _energies.append(faulted_stacking_fault_energy)
        
        nsteps = gliding_system.general.nsteps
        energies = {}
        for slipping_direction, sections in gliding_system.general.burger_vectors.items():
            # Join all sections/segments into a single continuous path
            energy_path = []
            energy_path.append(_energies.popleft())
            
            for section in sections:
                if isinstance(section[0], int):
                    segments = [section]
                else:
                    segments = section
                
                for _ in segments:
                    for _ in range(nsteps):
                        energy_path.append(_energies.popleft())
            
            if zero_reference:
                energy_ref = energy_path[0]
                energy_path = [None if val is None else val - energy_ref for val in energy_path]
            energies[slipping_direction] = energy_path
    
        return deepcopy(energies)

    def get_surface_energy(self):
        """Get the surface energy."""
        return self.surface_energy_value

    @property
    def surface_energy_value(self):
        """Get the surface energy of the workchain."""
        from ase.formula import Formula
        from aiida_dislocation.tools import calculate_surface_area
        from aiida_dislocation.workflows.gsfe import GSFEWorkChain

        if any(
            namespace not in self.process_tree for namespace in (GSFEWorkChain._SCF_NAMESPACE, GSFEWorkChain._SURFACE_ENERGY_NAMESPACE)
        ):
            raise AttributeError(f'{GSFEWorkChain._SCF_NAMESPACE} or {GSFEWorkChain._SURFACE_ENERGY_NAMESPACE} is not found')
        
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

    def serialize_faults(self):
        """Serialize the faults."""
        
        xs = {}

        if 'gliding_plane' in self.node.inputs:
            gliding_plane = self.node.inputs.gliding_plane.value
        elif 'faulted_structure_data' in self.node.inputs:
            gliding_plane = self.node.inputs.faulted_structure_data.gliding_plane
        else:
            raise AttributeError(f"Node<{self.node.pk}>: Neither 'gliding_plane' nor 'faulted_structure_data' found in inputs")
        gliding_system = fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(gliding_plane)

        nsteps = gliding_system.general.nsteps
        
        for slipping_direction, sections in gliding_system.general.burger_vectors.items():
            # Collect all segments for this slipping direction
            all_segments = []
            for section in sections:
                if isinstance(section[0], int):
                    all_segments.append(section)
                else:
                    all_segments.extend(section)
            
            # Create a single continuous x-axis for all segments
            x = numpy.linspace(0, 1, nsteps+1)
            for i in range(len(all_segments)-1):
                x = numpy.concatenate((x, numpy.linspace(i+1, i+2, nsteps+1)[1:]))
            
            # Renormalize to 1.0 to match the updated fit_curve segment boundaries (e.g. x <= 1/2 in gamma_esf)
            if all_segments:
                x = x / len(all_segments)
            xs[slipping_direction] = x
        
        return xs

    def fit_curve(self, fit_functions: dict[str, Callable | None] | None = None, plot: bool = False, axis=None, directions: list[str]|None = None, **kwargs):
        """Fit the curve."""
        from matplotlib.legend_handler import HandlerTuple
        import warnings
        from scipy.optimize import OptimizeWarning

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

        energies = filtered_energies
        results = {}

        try:
            formula = self.node.inputs.structure.get_formula()
        except Exception:
            formula = "Unknown"

        if 'gliding_plane' in self.node.inputs:
            gliding_plane = self.node.inputs.gliding_plane.value
        elif 'faulted_structure_data' in self.node.inputs:
            gliding_plane = self.node.inputs.faulted_structure_data.gliding_plane
        else:
            raise AttributeError(f"Node<{self.node.pk}>: Neither 'gliding_plane' nor 'faulted_structure_data' found in inputs")
        gliding_system = fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(gliding_plane)
        nsteps = gliding_system.general.nsteps
        
        if fit_functions is None:
            fit_functions = deepcopy(fit_function_map[self.strukturbericht][gliding_plane])

        sorted_keys = sorted(energies, key=lambda k: max(energies[k]), reverse=True)
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

    def get_plot_data(
        self,
        x_axis: str = 'step',
        zero_reference: bool = False,
        directions: list[str] | None = None,
    ):
        """Return simple plot-ready x/y arrays for each GSFE direction."""
        plot_data = {}
        xs_dict = self.serialize_faults()
        sfe_energies = self.get_sfe_energies(zero_reference=zero_reference)
        
        for direction, sfes in sfe_energies.items():
            if directions is not None and direction not in directions:
                continue
            
            if x_axis == 'displacement':
                xs = xs_dict[direction]
            else:
                xs = list(range(len(sfes)))
                
            plot_data[direction] = {
                'x': xs,
                'sfe': sfes,
            }
        return plot_data

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
        if value not in {'sfe'}:
            raise ValueError(f'unsupported value `{value}`, expected `sfe`')

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

class GSFEGroup(BaseGroupData):

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
                    conv_thr = GSFEAnalyser(node).conv_thr
                    if conv_thr is None:
                        conv_thr = 1e-6
                                        
                    # Structure: StructureType -> Formula -> Plane -> Process -> Layers -> K_Dist -> Conv_thr -> Node
                    if process_label in ['GSFEWorkChain']:
                        self._data[structuretype][formula][gliding_plane][process_label][n_repeats][kpoints_distance][conv_thr] = node

                except Exception as e:
                    logging.warning(f'Node<{node.pk}> processing failed: {e}')
                    continue

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
                                                analyser = GSFEAnalyser(node)
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
                                                print(f"Failed to fit/plot {node.pk}: {e}")
                                                            
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
                                        analyser = GSFEAnalyser(node)
                                        conv_error = analyser.conv_error
                                        if conv_error is not None:
                                            conv_thr_str = rf'{conv_thr:.1e} Ry (+- {conv_error*1000:.1e} mJ/m^2)'
                                        else:
                                            conv_thr_str = rf'{conv_thr:.1e} Ry'
                                    else:
                                        conv_thr_str = 'N/A'
                                    flattened_list.append({
                                        'Structure': struct_type,
                                        'Material': formula,
                                        'Plane': plane,
                                        'Process': process_label,
                                        'Layers': layers,
                                        'K_Dist': k_dist,
                                        'Conv_thr': conv_thr_str,
                                        'Status': self.get_status_string(node) + f' {node.pk}' if node else 'N/A',
                                    })
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

    def fit(self, destpath = None, axs = None, **kwargs):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        self._data.pop(None, None)
        structures = sorted(self._data.keys(), key=lambda x: str(x))
        all_planes = set()
        for struct in structures:
            for planes_dict in self._data[struct].values(): # val is dict of planes
                all_planes.update(planes_dict.keys())
        planes = sorted(list(all_planes), key=lambda x: str(x))
        
        if not structures or not planes:
             return {}
             
        n_rows = len(structures)
        n_cols = len(planes)
        
        if axs is None:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
        
        results = {}

        for i, struct in enumerate(structures):
            results[struct] = {}
            for j, plane in enumerate(planes):
                results[struct][plane] = {}
                ax = axs[i, j]
                
                mat_dict = self._data[struct]

                all_formulas = set()
                for formula in mat_dict.keys():
                    all_formulas.add(formula)

                sorted_formulas = sorted(list(all_formulas))

                base_colors = itertools.cycle([
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
                ])
                markers = itertools.cycle(['o', 's', 'v', '^', '<', '>', '8', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])

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
                                            logging.info(f"Fitting node<{node.pk}> for {formula} {plane}")
                                            analyser = GSFEAnalyser(node)
                                            
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
                
                ax.legend(
                    loc='upper right',
                    fontsize=kwargs.get('legend_fontsize', 16), 
                    frameon=False,      
                    facecolor='white',  
                    edgecolor='white',
                )

        for ax in axs[:, 0]:
            ax.set_ylabel(r'$\gamma [J/m^2]$', fontsize=kwargs.get('ylabel_fontsize', 16))

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
                    analyser = GSFEAnalyser(node)
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
                    analyser = GSFEAnalyser(node)
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
                            label_prefix=f"{k_dist} ",
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
                                        analyser = GSFEAnalyser(node)
                                        analyser.copy_tree(
                                            dest / struct_type / formula / plane / process_label / f"{layers}" / f"{k_dist}" / f"{conv_thr}" / f"{node.pk}"
                                            )
