from ..quantumespresso.pw_relax import PwRelaxAnalyser
from ..quantumespresso.pw_base import PwBaseAnalyser
from .sfebase import SFEBaseAnalyser

class ISFEAnalyser(SFEBaseAnalyser):
    """
    Analyser for the ISFEWorkChain.
    """

    @property
    def isfe(self):
        return self._get_node_from_tree('isfe')

    def copy_tree(self, destpath):
        """Copy the tree using the layered SFEBase child delegation."""
        return super().copy_tree(destpath)

    def get_calcjob_paths(self):
        """Get calcjob remote paths using the layered SFEBase child delegation."""
        return super().get_calcjob_paths()

    def get_state(self):
        """Get the state of the workchain."""
        return self._get_state_from_subprocesses([
            ('relax', PwRelaxAnalyser),
            ('scf', PwBaseAnalyser),
            ('isfe', PwRelaxAnalyser),
            ('surface_energy', PwBaseAnalyser),
        ])

    def calculate_isfe(self):
        """Calculate the ISFE."""
        from ase.formula import Formula
        from aiida_dislocation.tools import calculate_surface_area
        if 'isfe' not in self.process_tree or 'scf' not in self.process_tree:
            raise AttributeError('isfe or scf is not found')
        
        if not self.isfe.is_finished_ok or not self.scf.is_finished_ok:
            raise AttributeError('isfe or scf is not finished ok')
            
        intrinsic_formula = Formula(self.isfe.inputs.structure.get_ase().get_chemical_formula())
        _, intrinsic_multiplier = intrinsic_formula.reduce()
        conventional_formula = Formula(self.scf.inputs.pw.structure.get_ase().get_chemical_formula())
        _, conventional_multiplier = conventional_formula.reduce()
        surface_area = calculate_surface_area(self.scf.inputs.pw.structure.get_ase())
        total_energy_isf_geometry = self._get_safe_energy(self.isfe)
        total_energy_conventional_geometry = self._get_safe_energy(self.scf)
        
        if total_energy_isf_geometry is None or total_energy_conventional_geometry is None:
            raise ValueError('Energy not found in output_parameters for isfe or scf')

        energy_difference = total_energy_isf_geometry - total_energy_conventional_geometry / conventional_multiplier * intrinsic_multiplier
        intrinsic_stacking_fault_energy = energy_difference / surface_area * self._eVA22Jm2
        return intrinsic_stacking_fault_energy
