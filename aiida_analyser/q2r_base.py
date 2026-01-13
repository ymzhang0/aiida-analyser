from aiida import orm
from .base import BaseWorkChainAnalyser

class Q2rBaseWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the Q2rBaseWorkChain.
    """

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

