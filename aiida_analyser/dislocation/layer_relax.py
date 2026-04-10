from ..quantumespresso.pw_relax import PwRelaxWorkChainAnalyser
from ..base import BaseWorkChainAnalyser

class LayerRelaxWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the LayerRelaxWorkChain.
    """

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct PwRelaxWorkChain child."""
        return self._copy_tree_for_direct_children(
            destpath,
            lambda _, child: PwRelaxWorkChainAnalyser if child.node.process_label == 'PwRelaxWorkChain' else None,
        )

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct PwRelaxWorkChain child."""
        return self._get_calcjob_paths_for_direct_children(
            lambda _, child: PwRelaxWorkChainAnalyser if child.node.process_label == 'PwRelaxWorkChain' else None,
        )

    def get_state(self):
        """Get the state of the workchain."""
        subprocesses = [
            (label, PwRelaxWorkChainAnalyser)
            for label in self._get_child_labels(
                prefixes=('relax_',),
                process_label='PwRelaxWorkChain',
            )
        ]
        return self._get_state_from_subprocesses(subprocesses)

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
