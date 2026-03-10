from collections import defaultdict
from aiida import orm
from .pw_relax import PwRelaxWorkChainAnalyser
from .base import BaseWorkChainAnalyser
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

def formula_to_latex(formula):
    latex_formula = re.sub(r'(\d+)', r'_{\1}', formula)
    return rf"${latex_formula}$"

def gamma_isf(x, cG, g_isf):
    """
    Calculates the value for the first region: 0 <= x <= 1
    Formula: cG * sin^2(pi*x) + gamma_ISF * x
    Note: g_esf is included as an argument for consistency but not used here.
    """
    return cG * numpy.sin(numpy.pi * x)**2 + g_isf * x

def gamma_esf(x, cG1, cG2, cG3, cG4, b, c):
    """
    Calculates the value for the second region: 0 < x <= 1
    Formula: cG * sin^2(pi*x) + gamma_ISF * (2 - c) + gamma_ESF * (x - c)
    """
    return numpy.where(
        x <= 1, 
        cG1 * numpy.sin(numpy.pi * x)**2 + 
        cG2 * numpy.sin(numpy.pi * x)**4 + 
        cG3 * numpy.sin(numpy.pi * x)**6 + 
        cG4 * numpy.sin(numpy.pi * x)**8 + 
        b * x, 
        cG1 * numpy.sin(numpy.pi * x)**2 + 
        cG2 * numpy.sin(numpy.pi * x)**4 + 
        cG3 * numpy.sin(numpy.pi * x)**6 + 
        cG4 * numpy.sin(numpy.pi * x)**8 + 
        c * x + (b - c)
    )

def gamma_usf(x, a, b, c, d):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: gamma_USF * sin^2(pi*x)
    """
    return  a * numpy.sin(numpy.pi * x)**2 + \
            b * numpy.sin(numpy.pi * x)**4 + \
            c * numpy.sin(numpy.pi * x)**6 + \
            d * numpy.sin(numpy.pi * x)**8

def gamma_usf2(x, e_usf1, e_usf2):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: gamma_USF * sin^2(pi*x)
    """
    return  e_usf1 * numpy.sin(numpy.pi * x)**2 + \
            e_usf2 * numpy.sin(2*numpy.pi * x)**2

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

class GSFEWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the GsfeWorkChain.
    """
    _RY2eV    = 13.605693122990
    _RYA22Jm2 = 4.3597447222071E-18/2 * 1E+20
    _eVA22Jm2 = 1.602176634E-19 * 1E+20
    
    @property
    def strukturbericht(self):
        return get_strukturbericht(self.scf.inputs.pw.structure.get_ase())

    @property
    def relax(self):
        if 'relax' not in self.process_tree:
            raise AttributeError('relax is not found')
        else:
            return self.process_tree.relax.node

    @property
    def scf(self):
        if 'scf' not in self.process_tree:
            raise AttributeError('scf is not found')
        else:
            return self.process_tree.scf.node
    
    @property
    def surface_energy(self):
        if 'surface_energy' not in self.process_tree:
            raise AttributeError('surface_energy is not found')
        else:
            return self.process_tree.surface_energy.node
    
    def get_state(self):
        """Get the state of the workchain."""

        if self.node.is_finished_ok:
            return 'ROOT', 'finished_ok', 0
        
        # If all subprocesses are finished but main node is not, use tree traversal
        # to find the actual error in the process tree
        return self._get_state_from_tree()

    @property
    def scf_energy(self):
        """Get the energy of the scf calculation."""

        return self.scf.outputs.output_parameters.get('energy')

    @property
    def pristine_energy(self):
        """Get the pristine energy."""

        return self.process_tree.structure_01.node.outputs.output_parameters.get('energy')
    
    def get_sfe_energies(self):
        """Get the energies of the workchain."""
        from ase.formula import Formula
        from aiida_dislocation.tools import calculate_surface_area
        gliding_plane = self.node.inputs.gliding_plane.value
        gliding_system = fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(gliding_plane)

        nsteps = gliding_system.general.nsteps

        _energies =[]
        energies = {}
        # total_energy_conventional_geometry = self.scf_energy

        # faulted_formula = Formula(self.process_tree.structure_01.node.inputs.pw.structure.get_ase().get_chemical_formula())
        # _, faulted_multiplier = faulted_formula.reduce()
        # conventional_formula = Formula(self.scf.inputs.pw.structure.get_ase().get_chemical_formula())
        # _, conventional_multiplier = conventional_formula.reduce()
        surface_area = calculate_surface_area(self.scf.inputs.pw.structure.get_ase())
        for call_link_label, child in self.process_tree.children.items():
            if call_link_label.startswith('structure_'):
                total_energy_faulted_geometry = child.node.outputs.output_parameters.get('energy')
                # energy_difference = total_energy_faulted_geometry - total_energy_conventional_geometry / conventional_multiplier * faulted_multiplier
                energy_difference = total_energy_faulted_geometry - self.pristine_energy
                faulted_stacking_fault_energy = energy_difference / surface_area * self._eVA22Jm2
                _energies.append(faulted_stacking_fault_energy)

        for slipping_direction, sections in gliding_system.general.burger_vectors.items():
            energies[slipping_direction] = []
            for section in sections:
                energy_section = []
                energy_section.append(_energies.pop(0))
                for _ in section:
                    for _ in range(nsteps):
                        energy_section.append(_energies.pop(0))
                energies[slipping_direction].append(energy_section)
    
        return deepcopy(energies)

    def get_surface_energy(self):
        """Get the surface energy."""
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
        
        total_energy_cleavaged_geometry = self.surface_energy.outputs.output_parameters.get('energy')
        energy_difference = 2*(total_energy_cleavaged_geometry - self.scf_energy * surface_multiplier / conventional_multiplier)
        surface_energy = energy_difference / surface_area * self._eVA22Jm2
        
        return surface_energy

    def serialize_faults(self):
        """Serialize the faults."""
        
        xs = {}

        gliding_plane = self.node.inputs.gliding_plane.value
        gliding_system = fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(gliding_plane)

        nsteps = gliding_system.general.nsteps
        
        for slipping_direction, sections in gliding_system.general.burger_vectors.items():
            xs[slipping_direction] = []
            for section in sections:
                x = numpy.linspace(0, 1, nsteps+1)
                for i in range(len(section)-1):
                    x = numpy.concatenate((x, numpy.linspace(i+1, i+2, nsteps+1)[1:]))
                xs[slipping_direction].append(x)
        
        return xs

    def fit_curve(self, plot=False, axis=None, **kwargs):
        """Fit the curve."""
        from matplotlib.legend_handler import HandlerTuple
        def get_gradient_shades(hex_color, num=5):
            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", [hex_color, "#ffffff"])
            return [mcolors.to_hex(cmap(i)) for i in numpy.linspace(0, 0.8, num)]
        
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x']
        # color = kwargs.get('color', 'black')
        xs = self.serialize_faults()
        energies = self.get_sfe_energies()
        results = {}

        gliding_plane = self.node.inputs.gliding_plane.value
        gliding_system = fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(gliding_plane)
        
        func = fit_function_map[self.strukturbericht][gliding_plane]
        nsteps = gliding_system.general.nsteps

        sorted_keys = sorted(energies, key=lambda k: max(energies[k]), reverse=True)
        colors = get_gradient_shades(kwargs.get('color', 'black'), num = len(sorted_keys))
        for slipping_direction, color in zip(sorted_keys, colors):
            results[slipping_direction] = {}
            for x, y in zip(xs[slipping_direction], energies[slipping_direction]):
                x_plot = numpy.linspace(0, x[-1], 500)

                # y = numpy.array(energies[slipping_direction])

                if func == gamma_isf:
                    b = y[nsteps]
                    cG, pcov = curve_fit(
                        lambda x, cG: func(x, cG, b)
                        , x, y, maxfev=100000)

                    y_fit = func(x_plot, cG, b)

                    x_max = numpy.arcsin(-b / numpy.pi / cG) / 2 * numpy.pi
                    results[slipping_direction]['isf'] = b
                    results[slipping_direction]['usf'] = func(x_max, cG, b)[0]

                if func == gamma_esf:
                    b = y[nsteps]
                    c = y[2*nsteps]-y[nsteps]
                    (cG1, cG2, cG3, cG4), pcov = curve_fit(
                        lambda x, cG1, cG2, cG3, cG4: func(x, cG1, cG2, cG3, cG4, b, c)
                        , x, y, maxfev=100000)

                    y_fit = func(x_plot, cG1, cG2, cG3, cG4, b, c)

                    x_max = numpy.arcsin(-b / numpy.pi / cG1) / 2 * numpy.pi
                    results[slipping_direction]['usf'] = numpy.max(y_fit[:250])
                    results[slipping_direction]['isf'] = b
                    results[slipping_direction]['ut'] = numpy.max(y_fit[250:])
                    # results[slipping_direction]['usf'] = func(x_max, cG1, cG2, cG3, cG4, b, c)
                    results[slipping_direction]['esf'] = b+c


                if func == gamma_usf:
                    (a, b, c, d), pcov = curve_fit(
                        func, x, y, maxfev=100000)

                    y_fit = func(x_plot, a, b, c, d)

                    results[slipping_direction]['usf'] = a + b + c + d

                if func == gamma_usf2:
                    (e_usf1, e_usf2), pcov = curve_fit(
                        func, x, y, maxfev=100000)

                    y_fit = func(x_plot, e_usf1, e_usf2)

                    results[slipping_direction]['usf'] = numpy.max(y_fit)
                    results[slipping_direction]['s'] = e_usf2
                                        
                if plot:
                    if not axis:
                        import matplotlib.pyplot as plt
                        fig, axis = plt.subplots(figsize=(10, 6))
                    scatter = axis.scatter(
                        x, y, 
                        color=color, 
                        s=50, 
                        zorder=5, 
                        marker=markers.pop(0))

                    line, = axis.plot(
                        x_plot, 
                        y_fit, 
                        linestyle=kwargs.get('linestyle', '--'), 
                        color=color,
                        lw=kwargs.get('lw', 1.0),
                        label = kwargs.get('label', '') + f' <{slipping_direction}>')
                    axis.grid(True, alpha=0.3)

        return results

class GSFEGroupData:

    def __init__(self, groups = []):
        self._groups = groups
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
                    n_repeats = node.inputs.n_repeats.value
                    gliding_plane = node.inputs.gliding_plane.value
                    kpoints_distance = node.inputs.kpoints_distance.value
                                        
                    # Structure: StructureType -> Formula -> Plane -> Process -> Layers -> K_Dist -> Node
                    if process_label in ['GSFEWorkChain']:
                        self._data[structuretype][formula][gliding_plane][process_label][n_repeats][kpoints_distance] = node

                except Exception as e:
                    # Provide more context in error message
                    raise ValueError(f'Node<{node.pk}> processing failed: {e}')

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
                                for k_dist, node in k_dist_dict.items():
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

    def get_table(self):
        import pandas as pd
        import numpy as np

        def get_status_string(node):
            if node is None:
                return 'N/A'

            if not node.is_terminated:
                return '⏳'
            if node.is_finished_ok:
                return '✅'
            elif node.is_failed:
                return f'❌ ({node.exit_status})'
            elif node.is_excepted:
                return '⚠️ Excepted'
            elif node.is_killed:
                return '💀 Killed'
            else:
                return f'🏃 {node.process_state.value}'

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
                                    'Status': get_status_string(node) + f' {node.pk}' if node else 'N/A',
                                })

        if not flattened_list:
            return pd.DataFrame()

        df = pd.DataFrame(flattened_list)
        
        # Pivot table to show status for each Material with shared parameters
        # Index: Structure, Plane, Layers, K_Dist
        # Columns: Material
        
        pivot_df = df.pivot_table(
            values='Status',
            index=['Structure', 'Material', 'Layers', 'K_Dist'],
            columns='Plane',
            aggfunc='first' 
        )

        pivot_df = pivot_df.fillna('')

        # Sort columns (Materials) alphabetically
        pivot_df = pivot_df.sort_index(axis=1)
        return pivot_df

    def fit(self, destpath = None, axs = None, **kwargs):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        base_colors = [
            '#1f77b4', 
            '#ff7f0e', 
            '#2ca02c',
            '#d62728', # 红
            '#9467bd', # 紫
            '#8c564b', # 棕
            '#e377c2', # 粉
            '#7f7f7f', # 灰
            '#bcbd22', # 黄绿
            '#17becf'  # 青
        ]
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x']

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

        if n_rows > len(base_colors):
            raise ValueError(f"Number of structures ({n_rows}) is greater than number of base colors ({len(base_colors)})")
        if n_cols > len(markers):
            raise ValueError(f"Number of planes ({n_cols}) is greater than number of markers ({len(markers)})")
        
        results = {}

        for i, struct in enumerate(structures):
            results[struct] = {}
            for j, plane in enumerate(planes):
                results[struct][plane] = {}
                _base_colors = deepcopy(base_colors)
                ax = axs[i, j]
                
                mat_dict = self._data[struct]

                all_formulas = set()
                for formula in mat_dict.keys():
                    all_formulas.add(formula)

                sorted_formulas = sorted(list(all_formulas))

                cmap = plt.get_cmap('tab10') 
                formula_colors = {
                    formula: mcolors.to_hex(cmap(i % 10)) 
                    for i, formula in enumerate(sorted_formulas)
                }
                for formula, planes_dict in mat_dict.items():

                    if plane in planes_dict:
                        process_dict = planes_dict[plane]
                        color = _base_colors.pop(0)
                        if 'GSFEWorkChain' in process_dict:
                            for layers, k_dist_dict in process_dict['GSFEWorkChain'].items():
                                for k_dist, node in k_dist_dict.items():
                                    if node and node.is_finished_ok:
                                        print(node.pk, formula, plane)
                                        analyser = GSFEWorkChainAnalyser(node)
                                        
                                        # alpha_val = max(0.3, 1.0 - rank * 0.2) 
                                        
                                        # current_marker = markers[rank % len(markers)]

                                        results[struct][plane][formula] = analyser.fit_curve(
                                            plot=True, 
                                            axis=ax, 
                                            label=formula_to_latex(formula),
                                            color=color,
                                            # alpha=alpha_val,
                                            # marker=current_marker,
                                            markevery=5,
                                            linestyle='-',
                                            lw=kwargs.get('lw', 1.5),
                                            **kwargs
                                        )
                                    
                ax.set_title(f"${struct}$ ({plane})", fontsize=kwargs.get('title_fontsize', 16))
                
                ax.legend(
                    loc='upper right',
                    fontsize=10, 
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