from .base import BaseWorkChainAnalyser
from .pw_base import PwBaseWorkChainAnalyser
from .projwfc_base import ProjwfcBaseWorkChainAnalyser
from .pw2wannier90_base import Pw2Wannier90BaseWorkChainAnalyser
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
        if self.node.is_finished_ok:
            return 'ROOT', 0, 'finished OK'
        
        process_tree = self.process_tree
        
        # Define subprocesses in execution order
        # Required subprocesses: scf, nscf, wannier90_pp, pw2wannier90, wannier90
        # Optional subprocesses: projwfc
        subprocesses = [
            ('scf', True),  # (name, required)
            ('nscf', True),
            ('projwfc', False),  # optional
            ('wannier90_pp', True),
            ('pw2wannier90', True),
            ('wannier90', True),
        ]
        
        # Check each subprocess in order
        for subprocess_name, required in subprocesses:
            if subprocess_name in process_tree:
                if not process_tree[subprocess_name].node.is_finished_ok:
                    subprocess_node = process_tree[subprocess_name].node
                    
                    # Determine the appropriate analyser based on subprocess type
                    # scf, nscf are PW base workchains
                    # projwfc is Projwfc base workchain
                    # pw2wannier90 is Pw2Wannier90 base workchain
                    # wannier90_pp, wannier90 are Wannier90 base workchains
                    if subprocess_name in ['scf', 'nscf']:
                        analyser = PwBaseWorkChainAnalyser(subprocess_node)
                        path, exit_code, message = analyser.get_state()
                    elif subprocess_name == 'projwfc':
                        analyser = ProjwfcBaseWorkChainAnalyser(subprocess_node)
                        path, exit_code, message = analyser.get_state()
                    elif subprocess_name == 'pw2wannier90':
                        analyser = Pw2Wannier90BaseWorkChainAnalyser(subprocess_node)
                        path, exit_code, message = analyser.get_state()
                    elif subprocess_name in ['wannier90_pp', 'wannier90']:
                        analyser = Wannier90BaseWorkChainAnalyser(subprocess_node)
                        path, exit_code, message = analyser.get_state()
                    else:
                        # Fall back to tree traversal for unknown subprocess types
                        # This should not happen if subprocesses list is correct
                        temp_analyser = BaseWorkChainAnalyser(subprocess_node)
                        path, exit_code, message = temp_analyser._get_state_from_tree()
                    
                    # Return the state with subprocess name prefix
                    return subprocess_name if path == 'ROOT' else f'{subprocess_name}/{path}', exit_code, message
            elif required:
                # Required subprocess is missing, fall back to tree traversal
                return self._get_state_from_tree()
        
        # If all subprocesses are finished but main node is not, use tree traversal
        # to find the actual error in the process tree
        return self._get_state_from_tree()

    def print_state(self):
        """Print the state of the workchain."""
        result = self.get_state()
        if not result:
            print(f"Can't check the state of Wannier90WorkChain<{self.node.pk}>.")
            return
        path, process_state = result
        print(f"Wannier90WorkChain<{self.node.pk}> is {process_state} at {path}.")
    
    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message, success = super().clean_workchain(dry_run=dry_run)

        return message