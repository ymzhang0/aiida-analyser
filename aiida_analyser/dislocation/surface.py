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
        # Data structure: Material -> Degauss -> K_Dist -> Q_Dist -> node
        self._data = defaultdict(
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
                        self._data[a.strukturbericht][formula][gliding_plane][n_repeats][kpoints_distance] = node
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
                    for layers, k_dists in processes.items():
                        for k_dist, node in k_dists.items():
                            flattened_list.append({
                                'Structure': struct_type,
                                'Material': formula,
                                'Plane': plane,
                                'Layers': layers,
                                'K_Dist': k_dist,
                                'Status': self.get_status_string(node) + f' {node.pk}' if node else 'N/A',
                            })
        return flattened_list

    def plot(self, axes=None, kpoints_distance=None, n_repeats=None, dest=None):
        """Plot the surface energies."""
        import numpy
        if axes is None:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('tab10')
            n_strukturbericht = len(self._data.keys())
            fig, axes = plt.subplots(n_strukturbericht, 1, figsize=(5, 5 * n_strukturbericht))
        for ax, (strukturbericht, data) in zip(numpy.atleast_1d(axes), self._data.items()):
            n_formula = len(data.keys())
            colors = [cmap(i) for i in range(n_formula)]
            for (formula, node), color in zip(data.items(), colors):
                a = SurfaceWorkChainAnalyser(node)
                a.plot(ax=ax, label=formula, color = color)
            ax.legend()
            ax.set_title(f'${strukturbericht}$')
            ax.set_ylabel('Surface Energy (J/m$^2$)')
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Relative vacuum')
        if dest is not None and axes is not None:
            plt.savefig(dest)
        return
