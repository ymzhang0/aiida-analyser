from aiida_analyser.core.base import BaseRestartWorkChainAnalyser

from .convergence import EpwDegaussKQGroup


class EpwBaseAnalyser(BaseRestartWorkChainAnalyser):
    """Analyser for the EpwBaseWorkChain."""

    def get_source(self):
        if all(key in self.node.base.extras for key in ('source_db', 'source_id')):
            return (
                self.node.base.extras.get('source_db'),
                self.node.base.extras.get('source_id'),
            )
        if all(key in self.node.inputs.structure.base.extras for key in ('source_db', 'source_id')):
            return (
                self.node.inputs.structure.base.extras.get('source_db'),
                self.node.inputs.structure.base.extras.get('source_id'),
            )
        raise ValueError('Source is not set')


    def clean_workchain(self, dry_run=True):
        return super().clean_workchain(dry_run=dry_run)


class EpwBaseGroup(EpwDegaussKQGroup):
    """Convergence-grid view of ``EpwBaseWorkChain`` nodes."""

    analyser_class = EpwBaseAnalyser
    process_label = 'EpwBaseWorkChain'
