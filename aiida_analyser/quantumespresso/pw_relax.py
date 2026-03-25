from aiida import orm
from ..base import BaseWorkChainAnalyser
from .pw_base import PwBaseWorkChainAnalyser

class PwRelaxWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the PwRelaxWorkChain.
    """

    def get_source(self):
        """Get the source of the workchain."""
        source = super().get_source()
        if source is None:
            try:
                source_db, source_id = self.node.inputs.structure.base.extras.get_many(('source_db', 'source_id'))
                source = f"{source_db}-{source_id}"
            except Exception:
                print('Source is not set')
                return None
        return source

    def get_state(self):
        """Get the state of the workchain."""
        return self._get_state_from_subprocesses([
            ('base_init_relax', PwBaseWorkChainAnalyser),
            ('base_relax', PwBaseWorkChainAnalyser),
        ])
