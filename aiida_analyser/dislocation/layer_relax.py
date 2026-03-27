from ..quantumespresso.pw_relax import PwRelaxWorkChainAnalyser
from ..base import BaseWorkChainAnalyser

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
        # Assuming children are labeled relax_1, relax_2, ... based on index
        for i, spacing in enumerate(self.node.inputs.layer_spacings, 1):
            label = f'relax_{i}'
            if label in self.process_tree:
                child = self.process_tree[label]
                energies[spacing] = child.node.outputs.output_parameters.get('energy')
            else:
                # Fallback to check other possible common label patterns if relax_i is not found
                # but following the plan to fetch by explicit link label/index.
                # If the label is different, we might need a more robust way to find it.
                continue

        return energies