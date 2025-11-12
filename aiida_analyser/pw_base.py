from aiida import orm
from .base import BaseWorkChainAnalyser

class PwBaseWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the PwBaseWorkChain.
    """

    def get_source(self):
        """Get the source of the workchain."""
        if all(key in self.node.base.extras for key in ['source_db', 'source_id']):
            return (self.node.base.extras.get('source_db'), self.node.base.extras.get('source_id'))
        elif all(key in self.node.inputs.structure.base.extras for key in ['source_db', 'source_id']):
            return (self.node.inputs.structure.base.extras.get('source_db'), self.node.inputs.structure.base.extras.get('source_id'))
        else:
            raise ValueError('Source is not set')

    def get_state(self):
        """Get the state of the workchain."""
        # Start with the base implementation
        try:
            path, exit_status, message = self._get_state_from_tree()
        except (AttributeError, ValueError) as e:
            print(f'PwBaseWorkChain<{self.node.pk}> has unknown exit status: {e}')
            return 'ROOT', -1, 'Unknown status'

        # Handle specific error codes for PW calculations if needed
        # For now, just return the base state
        # In the future, we can add specific error handling for PW calculation errors

        return path, exit_status, message

    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message, success = super().clean_workchain(dry_run=dry_run)

        return message

