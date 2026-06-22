from collections import defaultdict
from aiida import orm
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

class SurfaceWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the SurfaceWorkChain.
    """

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct QE child to its own analyser."""
        def _resolve(child_name, child):
            process_label = child.node.process_label

            if process_label == 'PwRelaxWorkChain':
                return PwRelaxWorkChainAnalyser
            if process_label == 'PwBaseWorkChain' and (
                child_name == 'scf' or child_name.startswith('slab_') or child_name.startswith('spacing_')
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
                child_name == 'scf' or child_name.startswith('slab_') or child_name.startswith('spacing_')
            ):
                return PwBaseWorkChainAnalyser
            return None

        return self._get_calcjob_paths_for_direct_children(_resolve)

    @staticmethod
    def _parse_spacing_key(key) -> float:
        if isinstance(key, (float, int)):
            return float(key)
        key = str(key)
        for prefix in ('slab_', 'vacuum_spacing_'):
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                break
        return float(key.replace('_', '.'))
    
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
            if 'scf' in self.node.inputs and 'pw' in self.node.inputs.scf and 'parameters' in self.node.inputs.scf.pw:
                return float(self.node.inputs.scf.pw.parameters.get('ELECTRONS', {}).get('conv_thr'))
            elif 'pw' in self.node.inputs and 'parameters' in self.node.inputs.pw:
                return float(self.node.inputs.pw.parameters.get('ELECTRONS', {}).get('conv_thr'))
            # Fallback to process tree
            scf_node = self.scf
            if scf_node and 'pw' in scf_node.inputs and 'parameters' in scf_node.inputs.pw:
                return float(scf_node.inputs.pw.parameters.get('ELECTRONS', {}).get('conv_thr'))
        except Exception:
            pass
        return None

    @property
    def conv_error(self) -> float | None:
        """Return the convergence error of the Surface energy calculation."""
        conv_thr = self.conv_thr
        surface_area = self.surface_area
        if conv_thr is None or surface_area is None:
            return None
        return (conv_thr / surface_area) * self._eVA22Jm2

    @property
    def scf(self):
        return self._get_node_from_tree('scf')
    
    @property
    def surface_energies(self):
        for label in ('results', 'surface_results'):
            if label in self.node.outputs:
                return self.node.outputs[label]
        raise AttributeError('surface results output is not found')
    
    def get_state(self):
        """Get the state of the workchain."""
        subprocesses = []

        for label in self._get_child_labels(labels=('relax',), process_label='PwRelaxWorkChain'):
            subprocesses.append((label, PwRelaxWorkChainAnalyser))

        for label in self._get_child_labels(labels=('scf',), process_label='PwBaseWorkChain'):
            subprocesses.append((label, PwBaseWorkChainAnalyser))

        for label in self._get_child_labels(
            prefixes=('slab_', 'spacing_'),
            process_label='PwBaseWorkChain',
        ):
            subprocesses.append((label, PwBaseWorkChainAnalyser))

        return self._get_state_from_subprocesses(subprocesses)

    @property
    def scf_energy(self):
        """Get the energy of the scf calculation."""

        return self.scf.outputs.output_parameters.get('energy')


    def get_surface_energies(self):
        """Get the energies of the workchain."""
        if 'results' in self.node.outputs or 'surface_results' in self.node.outputs:
            aggregated_results = self.surface_energies.get_dict()
            if 'results' in aggregated_results:
                aggregated_results = aggregated_results['results']
            return {
                float(result.get('vacuum_spacing', self._parse_spacing_key(spacing))): result['surface_energy_j_m2']
                for spacing, result in sorted(
                    aggregated_results.items(),
                    key=lambda item: float(item[1].get('vacuum_spacing', self._parse_spacing_key(item[0]))),
                )
            }

        from ase.formula import Formula
        from aiida_dislocation.tools import calculate_surface_area

        if 'cleavaged_structure_data' in self.node.inputs:
            spacings = sorted(self.node.inputs.cleavaged_structure_data.vacuum_spacings, reverse=True)
        else:
            spacings = sorted(self.node.inputs.vacuum_spacings.get_list(), reverse=True)

        conventional_formula = Formula(self.scf.inputs.pw.structure.get_ase().get_chemical_formula())
        _, conventional_multiplier = conventional_formula.reduce()

        child_labels = [
            label for label in self.process_tree.children
            if label.startswith('spacing_') or label.startswith('slab_idx_') or label.startswith('slab_')
        ]
        if not child_labels:
            raise ValueError('No surface-energy child calculations were found in the process tree.')

        first_child_label = sorted(child_labels)[0]
        surface_formula = Formula(
            self.process_tree[first_child_label].node.inputs.pw.structure.get_ase().get_chemical_formula()
        )
        _, surface_multiplier = surface_formula.reduce()

        energies = {}
        # total_energy_conventional_geometry = self.scf_energy

        # faulted_formula = Formula(self.process_tree.structure_01.node.inputs.pw.structure.get_ase().get_chemical_formula())
        # _, faulted_multiplier = faulted_formula.reduce()
        # conventional_formula = Formula(self.scf.inputs.pw.structure.get_ase().get_chemical_formula())
        # _, conventional_multiplier = conventional_formula.reduce()
        surface_area = calculate_surface_area(self.scf.inputs.pw.structure.get_ase())
        
        for call_link_label, child in self.process_tree.children.items():
            if (
                call_link_label.startswith('spacing_')
                or call_link_label.startswith('slab_idx_')
                or call_link_label.startswith('slab_')
            ):
                total_energy_cleavaged_geometry = self._get_safe_energy(child.node)
                if total_energy_cleavaged_geometry is None:
                    logging.warning(f"Node<{child.node.pk}>: energy not found in output_parameters")
                    continue
                energy_difference = (total_energy_cleavaged_geometry - self.scf_energy * surface_multiplier / conventional_multiplier)
                cleavaged_surface_energy = energy_difference / (2*surface_area) * self._eVA22Jm2
                # energies.append(cleavaged_surface_energy)
                energies[spacings.pop()] = cleavaged_surface_energy

        if spacings != []:
            raise ValueError('Not all spacings are processed, left:', spacings)
        return energies

    def plot(self, ax=None, **kwargs):
        """Plot the surface energies."""
        import numpy
        energies = self.get_surface_energies()
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        array_2d = numpy.array(list(energies.items()))
        ax.plot(
            array_2d[:, 0], array_2d[:, 1], 
            label = kwargs.get('label', ''),
            marker = kwargs.get('marker', 'o'),
            linestyle = kwargs.get('linestyle', '-'),
            color = kwargs.get('color', 'black'),
            zorder = kwargs.get('zorder', 5),
            alpha = kwargs.get('alpha', 0.3)
        )
        return ax

class SurfaceEnergyData(BaseGroupData):

    def __init__(self, groups=None):
        super().__init__(groups)
        # Data structure: Material -> Degauss -> K_Dist -> Q_Dist -> node (actually StructureType -> Formula -> Plane -> Layers -> K_Dist -> Conv_thr -> NodeData)
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
                if not node.is_finished_ok:
                    continue
                try:
                    # Structure: Material -> Degauss -> K_Dist -> Q_Dist -> node
                    if node.process_label == 'SurfaceEnergyWorkChain':
                        formula = node.inputs.structure.get_formula()
                        kpoints_distance = node.inputs.kpoints_distance.value
                        if 'n_repeats' in node.inputs:
                            n_repeats = node.inputs.n_repeats.value
                        elif 'cleavaged_structure_data' in node.inputs:
                            n_repeats = node.inputs.cleavaged_structure_data.n_unit_cells
                        else:
                            raise AttributeError(f"Node<{node.pk}>: Neither 'n_repeats' nor 'cleavaged_structure_data' found in inputs")

                        if 'gliding_plane' in node.inputs:
                            gliding_plane = node.inputs.gliding_plane.value
                        elif 'cleavaged_structure_data' in node.inputs:
                            gliding_plane = node.inputs.cleavaged_structure_data.gliding_plane
                        else:
                            raise AttributeError(f"Node<{node.pk}>: Neither 'gliding_plane' nor 'cleavaged_structure_data' found in inputs")
                        a = SurfaceWorkChainAnalyser(node)
                        energies_dict = a.get_surface_energies()
                        energies_dict['pk'] = node.pk
                        conv_thr = a.conv_thr
                        if conv_thr is None:
                            conv_thr = 1e-6
                        self._data[a.strukturbericht][formula][gliding_plane][n_repeats][kpoints_distance][conv_thr] = energies_dict
                except Exception as e:
                    logging.warning(f'Node<{node.pk}> processing failed: {e}')
                    continue

    def _flatten_data(self):
        flattened_list = []

        # Iterate over the nested dictionary:
        # StructureType -> Formula -> Plane -> Layers -> K_Dist -> Conv_thr -> NodeData
        for struct_type, formulas in self._data.items():
            for formula, planes in formulas.items():
                for plane, processes in planes.items():
                    for layers, k_dists in processes.items():
                        for k_dist, conv_thr_dict in k_dists.items():
                            for conv_thr, node_data in conv_thr_dict.items():
                                if node_data:
                                    pk = node_data.get('pk', 'N/A')
                                    if pk != 'N/A':
                                        try:
                                            node = orm.load_node(pk)
                                            analyser = SurfaceWorkChainAnalyser(node)
                                            conv_error = analyser.conv_error
                                            if conv_error is not None:
                                                conv_thr_str = rf'{conv_thr:.1e} Ry (+- {conv_error*1000:.1e} mJ/m^2)'
                                            else:
                                                conv_thr_str = rf'{conv_thr:.1e} Ry'
                                        except Exception:
                                            conv_thr_str = rf'{conv_thr:.1e} Ry'
                                    else:
                                        conv_thr_str = 'N/A'
                                    flattened_list.append({
                                        'Structure': struct_type,
                                        'Material': formula,
                                        'Plane': plane,
                                        'Layers': layers,
                                        'K_Dist': k_dist,
                                        'Conv_thr': conv_thr_str,
                                        'Status': f'✅ {pk}',
                                    })  
        return flattened_list

    def plot(self, structure_type, formula, gliding_plane, n_repeats=None, ax=None, kpoints_distance=None, destpath=None, **kwargs):
        """Plot the surface energies (vacuum ratio convergence) for all structures and planes."""
        import matplotlib.pyplot as plt
        import numpy
        import re
        
        def formula_to_latex(formula):
            latex_formula = re.sub(r'(\d+)', r'_{\1}', formula)
            return rf"${latex_formula}$"


        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        struct_data = self._data.get(structure_type, {})
        formula_data = struct_data.get(formula, {})
        plane_data = formula_data.get(gliding_plane, {})

        if not plane_data:
            print(f"No SurfaceEnergyWorkChain data found for {formula} {gliding_plane}")
            return ax

        if n_repeats is None:
            n_repeats = sorted(plane_data.keys(), reverse=True)[0]

        k_dist_dict = plane_data.get(n_repeats, {})
        if not k_dist_dict:
            print(f"No data found for n_repeats={n_repeats}")
            return ax


        node_data = None
        conv_thr_dict = k_dist_dict.get(kpoints_distance, {})
        if conv_thr_dict:
            conv_thr = sorted(conv_thr_dict.keys())[0]
            node_data = conv_thr_dict[conv_thr]
        if node_data:
            energies = {k: v for k, v in node_data.items() if k != 'pk'}
            if energies:
                array_2d = numpy.array(sorted(energies.items(), key=lambda item: item[0]))
                ax.plot(
                    array_2d[:, 0], array_2d[:, 1],
                    label=kwargs.pop('label', f'{formula_to_latex(formula)} {gliding_plane}'),
                    **kwargs
                )
                ax.scatter(
                    array_2d[:, 0], array_2d[:, 1],
                    label="",
                    **kwargs
                )


            ax.set_title(f"{formula_to_latex(formula)} ({gliding_plane})")
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()

        ax.set_ylabel(r'$\gamma^{surface}$ (J/m$^2$)')
        ax.set_xlabel('Vacuum ratio')

        if destpath and ax is None:
            plt.tight_layout()
            plt.savefig(destpath)
        return ax

    def plot_kpoints_convergence(self, structure_type, formula, gliding_plane, spacing, n_repeats=None, ax=None, kpoints_distances=None, **kwargs):
        """Plot surface energies for different k-points on a single axis."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import re
        import numpy
        
        def formula_to_latex(formula):
            latex_formula = re.sub(r'(\d+)', r'_{\1}', formula)
            return rf"${latex_formula}$"

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        struct_data = self._data.get(structure_type, {})
        formula_data = struct_data.get(formula, {})
        plane_data = formula_data.get(gliding_plane, {})

        if not plane_data:
            print(f"No SurfaceEnergyWorkChain data found for {formula} {gliding_plane}")
            return ax

        if n_repeats is None:
            n_repeats = sorted(plane_data.keys(), reverse=True)[0]

        k_dist_dict = plane_data.get(n_repeats, {})
        if not k_dist_dict:
            print(f"No data found for n_repeats={n_repeats}")
            return ax

        if kpoints_distances is not None:
            filtered_k_dists = {k: v for k, v in k_dist_dict.items() if k in kpoints_distances}
        else:
            filtered_k_dists = k_dist_dict

        sorted_k_dists = sorted(filtered_k_dists.keys(), reverse=True)
        surface_energies = []

        for i, k_dist in enumerate(sorted_k_dists):
            conv_thr_dict = filtered_k_dists[k_dist]
            if conv_thr_dict:
                conv_thr = sorted(conv_thr_dict.keys())[0]
                node_data = conv_thr_dict[conv_thr]
                if node_data and spacing in node_data:
                    surface_energies.append([k_dist, node_data[spacing]])
        
        # def forward(x):
        #     x = numpy.array(x, dtype=float)
        #     with numpy.errstate(divide='ignore', invalid='ignore'):
        #         return numpy.where(x == 0, numpy.inf, 1.0 / (x ** 3))

        # def inverse(x):
        #     x = numpy.array(x, dtype=float)
        #     with numpy.errstate(divide='ignore', invalid='ignore'):
        #         return numpy.where(x == 0, numpy.inf, numpy.sign(x) * (numpy.abs(x) ** (-1.0/3.0)))

        # ax.set_xscale('function', functions=(forward, inverse))

        surface_energies = numpy.array(surface_energies)
        if surface_energies.size > 0:
            if kwargs.pop('use_mJ', False):
                surface_energies[:, 1] *= 1000
            ax.plot(surface_energies[:, 0], surface_energies[:, 1], marker=kwargs.get('marker', 'o'), label=kwargs.get('label', f'{formula}'))
            
        ax.set_title(f"K-points Convergence for {formula_to_latex(formula)} ({gliding_plane})")
        ax.set_ylabel(r'$\gamma^{surface}$ (J/m$^2$)')
        ax.set_xlabel(r'K-points distance (1/Å) [scaled as $1/d^3$]')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return (ax, surface_energies)

    def plot_supercell_convergence(self, structure_type, formula, gliding_plane, spacing, kpoints_distance=None, n_repeats=None, ax=None, **kwargs):
        """Plot surface energies for different supercell sizes (n_repeats) on a single axis."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import re
        import numpy
        
        def formula_to_latex(formula):
            latex_formula = re.sub(r'(\d+)', r'_{\1}', formula)
            return rf"${latex_formula}$"

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        struct_data = self._data.get(structure_type, {})
        formula_data = struct_data.get(formula, {})
        plane_data = formula_data.get(gliding_plane, {})

        if not plane_data:
            print(f"No SurfaceEnergyWorkChain data found for {formula} {gliding_plane}")
            return ax

        if n_repeats is not None:
            filtered_n_repeats_dict = {n: v for n, v in plane_data.items() if n in n_repeats}
        else:
            filtered_n_repeats_dict = plane_data

        sorted_n_repeats_dict = sorted(filtered_n_repeats_dict.keys(), reverse=True)
        surface_energies = []

        for i, n_rep in enumerate(sorted_n_repeats_dict):
            kpoints_distances_dict = filtered_n_repeats_dict[n_rep]
            if kpoints_distance is None:
                if not kpoints_distances_dict:
                    continue
                k_dist = sorted(kpoints_distances_dict.keys())[0]
                conv_thr_dict = kpoints_distances_dict[k_dist]
            else:
                conv_thr_dict = kpoints_distances_dict.get(kpoints_distance)
            
            if conv_thr_dict:
                conv_thr = sorted(conv_thr_dict.keys())[0]
                node_data = conv_thr_dict[conv_thr]
                if node_data and spacing in node_data:
                    surface_energies.append([n_rep, node_data[spacing]])

        if surface_energies:
            surface_energies = numpy.array(surface_energies)

            if kwargs.pop('use_mJ', False):
                surface_energies[:, 1] *= 1000
            # Sort by n_rep
            idx = numpy.argsort(surface_energies[:, 0])
            surface_energies = surface_energies[idx]
            ax.plot(surface_energies[:, 0], surface_energies[:, 1], marker=kwargs.get('marker', 'o'), label=kwargs.get('label', f'{formula}'))

        ax.set_title(f"Supercell Convergence for {formula_to_latex(formula)} ({gliding_plane})")
        ax.set_ylabel(r'$\gamma^{surface}$ (J/m$^2$)')
        ax.set_xlabel('Number of conventional cells')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return (ax, surface_energies)

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
                                        analyser = SurfaceWorkChainAnalyser(node)
                                        analyser.copy_tree(
                                            dest / struct_type / formula / plane / process_label / f"{layers}" / f"{k_dist}" / f"{conv_thr}" / f"{node.pk}"
                                            )
