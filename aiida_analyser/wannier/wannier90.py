from ..base import BaseWorkChainAnalyser
from ..quantumespresso.pw_base import PwBaseWorkChainAnalyser
from ..quantumespresso.projwfc_base import ProjwfcBaseWorkChainAnalyser
from ..quantumespresso.pw2wannier90_base import Pw2Wannier90BaseWorkChainAnalyser
from .wannier90_base import Wannier90BaseWorkChainAnalyser

class Wannier90WorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the Wannier90WorkChain.
    This is a composite workchain analyser that handles a workchain containing
    multiple sub-workchains: scf, nscf, projwfc, wannier90_pp, pw2wannier90, wannier90.
    
    For individual base workchains, use:
    - PwBaseWorkChainAnalyser for scf, nscf
    - ProjwfcBaseWorkChainAnalyser for projwfc
    - Pw2Wannier90BaseWorkChainAnalyser for pw2wannier90
    - Wannier90BaseWorkChainAnalyser for wannier90_pp, wannier90
    """

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct child to its own analyser."""
        def _resolve(_, child):
            process_label = child.node.process_label

            if process_label == 'PwBaseWorkChain':
                return PwBaseWorkChainAnalyser
            if process_label == 'ProjwfcBaseWorkChain':
                return ProjwfcBaseWorkChainAnalyser
            if process_label == 'Pw2Wannier90BaseWorkChain':
                return Pw2Wannier90BaseWorkChainAnalyser
            if process_label == 'Wannier90BaseWorkChain':
                return Wannier90BaseWorkChainAnalyser
            return None

        return self._copy_tree_for_direct_children(destpath, _resolve)

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct child to its analyser."""
        def _resolve(_, child):
            process_label = child.node.process_label

            if process_label == 'PwBaseWorkChain':
                return PwBaseWorkChainAnalyser
            if process_label == 'ProjwfcBaseWorkChain':
                return ProjwfcBaseWorkChainAnalyser
            if process_label == 'Pw2Wannier90BaseWorkChain':
                return Pw2Wannier90BaseWorkChainAnalyser
            if process_label == 'Wannier90BaseWorkChain':
                return Wannier90BaseWorkChainAnalyser
            return None

        return self._get_calcjob_paths_for_direct_children(_resolve)

    def _get_wannier90_pp_labels(self):
        return self._get_child_labels(
            labels=('wannier90_pp',),
            prefixes=('wannier90_pp_',),
            process_label='Wannier90BaseWorkChain',
        )

    def _get_pw2wannier90_labels(self):
        return self._get_child_labels(
            labels=('pw2wannier90',),
            prefixes=('pw2wannier90_',),
            process_label='Pw2Wannier90BaseWorkChain',
        )

    def _get_wannier90_run_labels(self):
        return [
            label for label in self._get_child_labels(
                labels=('wannier90',),
                prefixes=('wannier90_',),
                process_label='Wannier90BaseWorkChain',
            )
            if label not in set(self._get_wannier90_pp_labels())
        ]

    @property
    def scf(self):
        try:
            return self.process_tree.scf
        except AttributeError:
            return None

    @property
    def nscf(self):
        try:
            return self.process_tree.nscf
        except AttributeError:
            return None

    @property
    def projwfc(self):
        try:
            return self.process_tree.projwfc
        except AttributeError:
            return None

    @property
    def wannier90_pp(self):
        try:
            return self.process_tree.wannier90_pp
        except AttributeError:
            return None

    @property
    def pw2wannier90(self):
        try:
            return self.process_tree.pw2wannier90
        except AttributeError:
            return None

    @property
    def wannier90(self):
        try:
            return self.process_tree.wannier90
        except AttributeError:
            return None

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
        subprocesses = []
        required_subprocesses = []

        for label in self._get_child_labels(labels=('scf',), process_label='PwBaseWorkChain'):
            subprocesses.append((label, PwBaseWorkChainAnalyser))
            required_subprocesses.append(label)

        for label in self._get_child_labels(labels=('nscf',), process_label='PwBaseWorkChain'):
            subprocesses.append((label, PwBaseWorkChainAnalyser))
            required_subprocesses.append(label)

        for label in self._get_child_labels(labels=('projwfc',), process_label='ProjwfcBaseWorkChain'):
            subprocesses.append((label, ProjwfcBaseWorkChainAnalyser))

        wannier90_pp_labels = self._get_wannier90_pp_labels()
        for label in wannier90_pp_labels:
            subprocesses.append((label, Wannier90BaseWorkChainAnalyser))
            required_subprocesses.append(label)

        pw2wannier90_labels = self._get_pw2wannier90_labels()
        for label in pw2wannier90_labels:
            subprocesses.append((label, Pw2Wannier90BaseWorkChainAnalyser))
            required_subprocesses.append(label)

        wannier90_labels = self._get_wannier90_run_labels()
        for label in wannier90_labels:
            subprocesses.append((label, Wannier90BaseWorkChainAnalyser))
            required_subprocesses.append(label)

        return self._get_state_from_subprocesses(
            subprocesses,
            required_subprocesses=tuple(required_subprocesses),
        )

    def print_state(self):
        """Print the state of the workchain."""
        result = self.get_state()
        if not result:
            print(f"Can't check the state of Wannier90WorkChain<{self.node.pk}>.")
            return
        path, process_state, exit_code = result
        print(f"Wannier90WorkChain<{self.node.pk}> is {process_state} at {path} with exit code {exit_code}.")
    
    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message, success = super().clean_workchain(dry_run=dry_run)

        return message
