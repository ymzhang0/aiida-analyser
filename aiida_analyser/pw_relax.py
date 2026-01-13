from aiida import orm
from .base import BaseWorkChainAnalyser
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

        # Check subprocesses in order
        for subprocess_name, subprocess_analyser in [
            ('base_init_relax', PwBaseWorkChainAnalyser), 
            ('base_relax', PwBaseWorkChainAnalyser), 
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