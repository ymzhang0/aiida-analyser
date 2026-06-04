from collections import defaultdict, deque
from typing import Callable
from aiida import orm
from aiida_analyser.dislocation.gsfe_latest import GSFEWorkChainAnalyserLatest
from ..quantumespresso.pw_base import PwBaseWorkChainAnalyser
from ..quantumespresso.pw_relax import PwRelaxWorkChainAnalyser
from ..base import BaseWorkChainAnalyser
from .basegroup import BaseGroupData
import logging
from scipy.optimize import curve_fit
import numpy
from aiida_dislocation.tools import (
    A1GlidingSystem,
    A2GlidingSystem,
    B1GlidingSystem,
    B2GlidingSystem,
    C1bGlidingSystem,
    L21GlidingSystem,
    get_strukturbericht
    )
from aiida_dislocation.tools.structure_utils import (
    get_strukturbericht
    )

import re
from copy import deepcopy
import itertools

from pathlib import Path

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

class GSFEWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the GsfeWorkChain.
    """

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct QE child to its own analyser."""
        def _resolve(child_name, child):
            process_label = child.node.process_label

            if process_label == 'PwRelaxWorkChain':
                return PwRelaxWorkChainAnalyser
            if process_label == 'PwBaseWorkChain' and (
                child_name == 'scf' or child_name.startswith('structure_') or child_name.startswith('sfe_')
            ):
                return PwBaseWorkChainAnalyser
            return None

        return self._copy_tree_for_direct_children(destpath, _resolve)

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct QE child to its analyser."""
        def _resolve(child_name, child):
            process_label = child.node.process_label

            if process_label == 'PwRelaxWorkChain':
                return PwRelaxWorkChainAnalyser
            if process_label == 'PwBaseWorkChain' and (
                child_name == 'scf' or child_name.startswith('structure_') or child_name.startswith('sfe_')
            ):
                return PwBaseWorkChainAnalyser
            return None

        return self._get_calcjob_paths_for_direct_children(_resolve)
    
    @property
    def strukturbericht(self):
        return get_strukturbericht(self.scf.inputs.pw.structure.get_ase())

    @property
    def relax(self):
        return self._get_node_from_tree('relax')

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
            subprocesses.append((label, PwRelaxWorkChainAnalyser))

        for label in self._get_child_labels(labels=('scf',), process_label='PwBaseWorkChain'):
            subprocesses.append((label, PwBaseWorkChainAnalyser))

        for label in self._get_child_labels(
            prefixes=('structure_', 'sfe_'),
            process_label='PwBaseWorkChain',
        ):
            subprocesses.append((label, PwBaseWorkChainAnalyser))

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
                    if func == gamma_isf:
                        b = y[nsteps]
                        order = kwargs.get('order', 1)
                        popt, _ = curve_fit(lambda x, *cGs: func(x, cGs, b), x, y, p0=[0.1] * order, maxfev=100000)
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
                        results[slipping_direction]['usf'] = numpy.max(y_fit[:250])*1000
                        results[slipping_direction]['isf'] = b*1000
                        results[slipping_direction]['ut'] = numpy.max(y_fit[250:])*1000
                        results[slipping_direction]['esf'] = c*1000

                    elif func == gamma_usf:
                        order = kwargs.get('order', 4)
                        popt, _ = curve_fit(
                            lambda x, *cGs: func(x, cGs), 
                            x, y, 
                            p0=[0.1] * order, 
                            maxfev=100000
                            )
                        y_fit = func(x_plot, popt)
                        y_fit_orig = func(x, popt)
                        results[slipping_direction]['usf'] = numpy.max(y_fit)*1000

                    elif func == gamma_usf_symmetric:
                        order = kwargs.get('order', 2)
                        popt, _ = curve_fit(
                            lambda x, *cGs: func(x, cGs), 
                            x, y, 
                            p0=[0.1] * order,
                            maxfev=100000
                            )
                        y_fit = func(x_plot, popt)
                        y_fit_orig = func(x, popt)
                        results[slipping_direction]['usf'] = numpy.max(y_fit)*1000

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

class GSFEGroupData(BaseGroupData):

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
                    conv_thr = node.inputs.sfe.pw.parameters.get('ELECTRONS', None).get('conv_thr', 1e-6)
                                        
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
                                                analyser = GSFEWorkChainAnalyser(node)
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
                                    flattened_list.append({
                                        'Structure': struct_type,
                                        'Material': formula,
                                        'Plane': plane,
                                        'Process': process_label,
                                        'Layers': layers,
                                        'K_Dist': k_dist,
                                        'Conv_thr': conv_thr,
                                        'Status': self.get_status_string(node) + f' {node.pk}' if node else 'N/A',
                                    })
        return flattened_list

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
                                            analyser = GSFEWorkChainAnalyser(node)
                                            
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
                                        analyser = GSFEWorkChainAnalyser(node)
                                        analyser.copy_tree(
                                            dest / struct_type / formula / plane / process_label / f"{layers}" / f"{k_dist}" / f"{conv_thr}" / f"{node.pk}"
                                            )
