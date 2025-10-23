from re import S
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from enum import Enum
from collections import OrderedDict
from .base import BaseWorkChainAnalyser

class Wannier90WorkChainState(Enum):
    """
    Analyser for the Wannier90WorkChain.
    """
    FINISHED_OK = 0
    WAITING = 1
    RUNNING = 2
    EXCEPTED = 3
    KILLED = 4
    SCF_FAILED = 4004
    NSCF_FAILED = 4005
    PW2WAN_FAILED = 4006
    WANNIER_FAILED = 4007
    UNKNOWN = 999

class Wannier90WorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the Wannier90WorkChain.
    """

    def __init__(self, workchain: orm.WorkChainNode):
        self.node = workchain

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
        process_tree = self.process_tree

        # if self.node.process_state.name in ['CREATED', 'RUNNING', 'EXCEPTED', 'KILLED']:
        #     return 'ROOT', -2, self.node.process_state.value
        if self.node.is_finished_ok:
            return 'ROOT', 0, 'finished OK'
        elif not process_tree.scf.node.is_finished_ok:
            scf_analyser = BaseWorkChainAnalyser(process_tree.scf.node)
            path, exit_code, message = scf_analyser.get_state()
            return 'scf', exit_code, message
        elif not process_tree.nscf.node.is_finished_ok:
            nscf_analyser = BaseWorkChainAnalyser(process_tree.nscf.node)
            path, exit_code, message = nscf_analyser.get_state()
            return 'nscf', exit_code, message
        elif 'projwfc' in process_tree and not process_tree.projwfc.node.is_finished_ok:
            projwfc_analyser = BaseWorkChainAnalyser(process_tree.projwfc.node)
            path, exit_code, message = projwfc_analyser.get_state()
            return 'projwfc', exit_code, message
        elif not process_tree.wannier90_pp.node.is_finished_ok:
            wannier90_pp_analyser = BaseWorkChainAnalyser(process_tree.wannier90_pp.node)
            path, exit_code, message = wannier90_pp_analyser.get_state()
            return 'wannier90_pp', exit_code, message
        elif not process_tree.pw2wannier90.node.is_finished_ok:
            pw2wannier90_analyser = BaseWorkChainAnalyser(process_tree.pw2wannier90.node)
            path, exit_code, message = pw2wannier90_analyser.get_state()
            return 'pw2wannier90', exit_code, message
        elif not process_tree.wannier90.node.is_finished_ok:
            wannier90_analyser = BaseWorkChainAnalyser(process_tree.wannier90.node)
            path, exit_code, message = wannier90_analyser.get_state()
            return 'wannier90', exit_code, message
        else:
            # return super().get_state()
            raise ValueError(f'Unknown exit status: {self.node.exit_status}')
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

        message = super().clean_workchain([
            Wannier90WorkChainState.WAITING,
            Wannier90WorkChainState.RUNNING,
            ],
            dry_run=dry_run
            )

        return message