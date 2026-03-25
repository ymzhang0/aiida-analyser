from ..quantumespresso.pw_relax import PwRelaxWorkChainAnalyser
from ..quantumespresso.pw_base import PwBaseWorkChainAnalyser
from .sfebase import SFEBaseWorkChainAnalyser
from .layer_relax import LayerRelaxWorkChainAnalyser

class USFEWorkChainAnalyser(SFEBaseWorkChainAnalyser):
    """
    Analyser for the USFEWorkChain.
    """

    @property
    def layer_relax(self):
        if 'layer_relax' not in self.process_tree:
            raise AttributeError('layer_relax is not found')
        else:
            return self.process_tree.layer_relax.node

    def get_state(self):
        """Get the state of the workchain."""
        return self._get_state_from_subprocesses([
            ('relax', PwRelaxWorkChainAnalyser),
            ('scf', PwBaseWorkChainAnalyser),
            ('layer_relax', LayerRelaxWorkChainAnalyser),
            ('surface_energy', PwBaseWorkChainAnalyser),
        ])
