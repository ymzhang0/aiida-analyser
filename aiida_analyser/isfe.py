from .pw_relax import PwRelaxWorkChainAnalyser
from .pw_base import PwBaseWorkChainAnalyser
from .sfebase import SFEBaseWorkChainAnalyser

class ISFEWorkChainAnalyser(SFEBaseWorkChainAnalyser):
    """
    Analyser for the ISFEWorkChain.
    """

    _RY2eV    = 13.605693122990
    _RYA22Jm2 = 4.3597447222071E-18/2 * 1E+20
    _eVA22Jm2 = 1.602176634E-19 * 1E+20

    @property
    def isfe(self):
        if 'isfe' not in self.process_tree:
            raise AttributeError('isfe is not found')
        else:
            return self.process_tree.isfe.node

    def get_state(self):
        """Get the state of the workchain."""

        # Check subprocesses in order
        for subprocess_name, subprocess_analyser in [
            ('relax', PwRelaxWorkChainAnalyser), 
            ('scf', PwBaseWorkChainAnalyser), 
            ('isfe', PwRelaxWorkChainAnalyser), 
            ('surface_energy', PwBaseWorkChainAnalyser)
            ]:
            if subprocess_name in self.process_tree:
                if not self.process_tree[subprocess_name].node.is_finished_ok:
                    analyser = subprocess_analyser(self.process_tree[subprocess_name].node)
                    path, process_state, exit_code = analyser.get_state()
                    return f'{subprocess_name}/{path}' if path != 'ROOT' else subprocess_name, process_state, exit_code
        
        if self.node.is_finished_ok:
            return 'ROOT', 'finished_ok', 0
        
        # If all subprocesses are finished but main node is not, use tree traversal
        # to find the actual error in the process tree
        return self._get_state_from_tree()

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
        surface_area = calculate_surface_area(self.scf.inputs.pw.structure.get_ase().cell)
        total_energy_isf_geometry = self.isfe.outputs.output_parameters.get('energy')
        total_energy_conventional_geometry = self.scf.outputs.output_parameters.get('energy')
        energy_difference = total_energy_isf_geometry - total_energy_conventional_geometry / conventional_multiplier * intrinsic_multiplier
        intrinsic_stacking_fault_energy = energy_difference / surface_area * self._eVA22Jm2
        return intrinsic_stacking_fault_energy
