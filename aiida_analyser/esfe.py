from .pw_relax import PwRelaxWorkChainAnalyser
from .pw_base import PwBaseWorkChainAnalyser
from .sfebase import SFEBaseWorkChainAnalyser

class ESFEWorkChainAnalyser(SFEBaseWorkChainAnalyser):
    """
    Analyser for the ESFEWorkChain.
    """

    @property
    def esfe(self):
        if 'esfe' not in self.process_tree:
            raise AttributeError('esfe is not found')
        else:
            return self.process_tree.esfe.node

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
        surface_area = calculate_surface_area(self.scf.inputs.pw.structure.get_ase().cell)
        total_energy_esf_geometry = self.esfe.outputs.output_parameters.get('energy')
        total_energy_conventional_geometry = self.scf.outputs.output_parameters.get('energy')
        energy_difference = total_energy_esf_geometry - total_energy_conventional_geometry / conventional_multiplier * extrinsic_multiplier
        extrinsic_stacking_fault_energy = energy_difference / surface_area * self._eVA22Jm2
        return extrinsic_stacking_fault_energy
