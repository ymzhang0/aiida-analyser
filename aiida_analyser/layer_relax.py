from .pw_relax import PwRelaxWorkChainAnalyser
from .base import BaseWorkChainAnalyser

class LayerRelaxWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the LayerRelaxWorkChain.
    """

    def get_state(self):
        """Get the state of the workchain."""

        if self.node.is_finished_ok:
            return 'ROOT', 'finished_ok', 0
        
        # If all subprocesses are finished but main node is not, use tree traversal
        # to find the actual error in the process tree
        return self._get_state_from_tree()

    def get_energies(self):
        """Get the energies of the workchain."""

        energies = {}
        for spacing, child in zip(self.node.inputs.layer_spacings, self.process_tree.children.values()):
            energies[spacing] = child.node.outputs.output_parameters.get('energy')

        return energies