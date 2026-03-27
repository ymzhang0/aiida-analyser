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
        return self._get_node_from_tree('layer_relax')

    def get_state(self):
        """Get the state of the workchain."""
        return self._get_state_from_subprocesses([
            ('relax', PwRelaxWorkChainAnalyser),
            ('scf', PwBaseWorkChainAnalyser),
            ('layer_relax', LayerRelaxWorkChainAnalyser),
            ('surface_energy', PwBaseWorkChainAnalyser),
        ])
