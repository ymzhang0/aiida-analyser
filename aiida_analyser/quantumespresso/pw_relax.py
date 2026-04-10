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
                self._log_source_missing()
                return None
        return source

    def get_state(self):
        """Get the state of the workchain."""
        subprocesses = []

        for label in ('init_relax', 'base_init_relax'):
            if label in self.process_tree:
                subprocesses.append((label, PwBaseWorkChainAnalyser))

        iteration_labels = sorted(
            (
                child_name for child_name, child_tree in self.process_tree.children.items()
                if child_tree.node.process_label == 'PwBaseWorkChain' and child_name.startswith('iteration_')
            ),
            key=lambda label: int(label.split('_')[1]),
        )
        subprocesses.extend((label, PwBaseWorkChainAnalyser) for label in iteration_labels)

        trailing_pw_bases = [
            child_name for child_name, child_tree in self.process_tree.children.items()
            if child_tree.node.process_label == 'PwBaseWorkChain'
            and child_name not in {name for name, _ in subprocesses}
        ]
        subprocesses.extend((label, PwBaseWorkChainAnalyser) for label in trailing_pw_bases)

        return self._get_state_from_subprocesses(subprocesses)

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct PwBaseWorkChain child."""
        return self._copy_tree_for_direct_children(
            destpath,
            lambda _, child: PwBaseWorkChainAnalyser if child.node.process_label == 'PwBaseWorkChain' else None,
        )

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct PwBaseWorkChain child."""
        return self._get_calcjob_paths_for_direct_children(
            lambda _, child: PwBaseWorkChainAnalyser if child.node.process_label == 'PwBaseWorkChain' else None,
        )
