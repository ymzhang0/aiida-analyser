from aiida import orm
from ..base import BaseWorkChainAnalyser
from .matdyn_calculation import MatdynCalculationAnalyser

class MatdynBaseWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the MatdynBaseWorkChain.
    """

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct MatdynCalculation child."""
        return self._copy_tree_for_direct_children(
            destpath,
            lambda _, child: MatdynCalculationAnalyser if child.node.process_label == 'MatdynCalculation' else None,
        )

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct MatdynCalculation child."""
        return self._get_calcjob_paths_for_direct_children(
            lambda _, child: MatdynCalculationAnalyser if child.node.process_label == 'MatdynCalculation' else None,
        )

    def get_source(self):
        """Get the source of the workchain."""
        if all(key in self.node.base.extras for key in ['source_db', 'source_id']):
            return (self.node.base.extras.get('source_db'), self.node.base.extras.get('source_id'))
        else:
            raise ValueError('Source is not set')

    def get_state(self):
        """Get the state of the workchain."""
        # Start with the base implementation

        return self._get_state_from_tree()

    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message, success = super().clean_workchain(dry_run=dry_run)

        return message
