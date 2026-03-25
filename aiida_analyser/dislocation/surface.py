from collections import defaultdict
from aiida import orm
from ..quantumespresso.pw_relax import PwRelaxWorkChainAnalyser
from ..base import BaseWorkChainAnalyser
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
    _RY2eV    = 13.605693122990
    _RYA22Jm2 = 4.3597447222071E-18/2 * 1E+20
    _eVA22Jm2 = 1.602176634E-19 * 1E+20

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
    def surface_energies(self):
        for label in ('results', 'surface_results'):
            if label in self.node.outputs:
                return self.node.outputs[label]
        raise AttributeError('surface results output is not found')
    
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
                total_energy_cleavaged_geometry = child.node.outputs.output_parameters.get('energy')
                energy_difference = (total_energy_cleavaged_geometry - self.scf_energy * surface_multiplier / conventional_multiplier)
                cleavaged_surface_energy = energy_difference / (surface_area) * self._eVA22Jm2
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

class SurfaceEnergyData:

    def __init__(self, groups = []):
        self._groups = groups
        # Data structure: Material -> Degauss -> K_Dist -> Q_Dist -> node
        self._data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: None
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
                    formula = node.inputs.structure.get_formula()
                    
                    # Structure: Material -> Degauss -> K_Dist -> Q_Dist -> node
                    if node.process_label == 'SurfaceEnergyWorkChain':
                        a = SurfaceWorkChainAnalyser(node)
                        self._data[a.strukturbericht][formula] = node
                except Exception as e:
                    # Provide more context in error message
                    raise ValueError(f'Node<{node.pk}> processing failed: {e}')

    def plot(self, axes=None, dest=None):
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
