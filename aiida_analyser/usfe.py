from .pw_relax import PwRelaxWorkChainAnalyser
from .pw_base import PwBaseWorkChainAnalyser
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

        # Check subprocesses in order
        for subprocess_name, subprocess_analyser in [
            ('relax', PwRelaxWorkChainAnalyser), 
            ('scf', PwBaseWorkChainAnalyser), 
            ('layer_relax', LayerRelaxWorkChainAnalyser), 
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
