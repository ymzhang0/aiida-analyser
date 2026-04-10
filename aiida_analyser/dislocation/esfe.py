from ..quantumespresso.pw_relax import PwRelaxWorkChainAnalyser
from ..quantumespresso.pw_base import PwBaseWorkChainAnalyser
from .sfebase import SFEBaseWorkChainAnalyser

class ESFEWorkChainAnalyser(SFEBaseWorkChainAnalyser):
    """
    Analyser for the ESFEWorkChain.
    """

    @property
    def esfe(self):
        return self._get_node_from_tree('esfe')

    def copy_tree(self, destpath):
        """Copy the tree using the layered SFEBase child delegation."""
        return super().copy_tree(destpath)

    def get_calcjob_paths(self):
        """Get calcjob remote paths using the layered SFEBase child delegation."""
        return super().get_calcjob_paths()

    def get_state(self):
        """Get the state of the workchain."""
        return self._get_state_from_subprocesses([
            ('relax', PwRelaxWorkChainAnalyser),
            ('scf', PwBaseWorkChainAnalyser),
            ('esfe', PwRelaxWorkChainAnalyser),
            ('surface_energy', PwBaseWorkChainAnalyser),
        ])

    def calculate_esfe(self):
        """Calculate the ESFE."""
        from ase.formula import Formula
        from aiida_dislocation.tools import calculate_surface_area
        if 'esfe' not in self.process_tree or 'scf' not in self.process_tree:
            raise AttributeError('esfe or scf is not found')
        
        if not self.esfe.is_finished_ok or not self.scf.is_finished_ok:
            raise AttributeError('esfe or scf is not finished ok')
            
        extrinsic_formula = Formula(self.esfe.inputs.structure.get_ase().get_chemical_formula())
        _, extrinsic_multiplier = extrinsic_formula.reduce()
        conventional_formula = Formula(self.scf.inputs.pw.structure.get_ase().get_chemical_formula())
        _, conventional_multiplier = conventional_formula.reduce()
        surface_area = calculate_surface_area(self.scf.inputs.pw.structure.get_ase())
        total_energy_esf_geometry = self._get_safe_energy(self.esfe)
        total_energy_conventional_geometry = self._get_safe_energy(self.scf)
        
        if total_energy_esf_geometry is None or total_energy_conventional_geometry is None:
            raise ValueError('Energy not found in output_parameters for esfe or scf')

        energy_difference = total_energy_esf_geometry - total_energy_conventional_geometry / conventional_multiplier * extrinsic_multiplier
        extrinsic_stacking_fault_energy = energy_difference / surface_area * self._eVA22Jm2
        return extrinsic_stacking_fault_energy
