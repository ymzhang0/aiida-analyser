from ..core.base import BaseRestartWorkChainAnalyser


class PwBaseAnalyser(BaseRestartWorkChainAnalyser):
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


    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message, success = super().clean_workchain(dry_run=dry_run)

        return message
