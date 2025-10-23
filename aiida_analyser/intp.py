from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from enum import Enum
from collections import OrderedDict
from .b2w import EpwB2WWorkChainAnalyser
from .base import EpwBaseWorkChainAnalyser
class EpwIntpWorkChainState(Enum):
    FINISHED_OK = 0
    WAITING = 1
    RUNNING = 2
    EXCEPTED = 3
    KILLED = 4
    FAILED = 5
    B2W_FAILED = 6
    B2W_EXCEPTED = 7
    EPW_BASE_FAILED = 8
    EPW_BASE_EXCEPTED = 9
    UNKNOWN = 999

class EpwIntpWorkChainAnalyser:
    """
    Analyser for the EpwIntpWorkChain.
    """
    _all_descendants = OrderedDict([
        ('b2w', None),
        ('intp', None),
    ])

    def __init__(self, workchain: orm.WorkChainNode):
        self.node = workchain
        self.state = EpwIntpWorkChainState.UNKNOWN
        self.descendants = {}
        for link_label, _ in self._all_descendants.items():
            descendants = workchain.base.links.get_outgoing(link_label_filter=link_label).all_nodes()
            if descendants != []:
                self.descendants[link_label] = descendants

    @staticmethod
    def base_check(
        workchain: orm.WorkChainNode,
        excepted_state: EpwIntpWorkChainState,
        failed_state: EpwIntpWorkChainState,
        namespace: str,
        ) -> tuple[EpwIntpWorkChainState, str]:
        if workchain.process_state == ProcessState.WAITING:
            state = EpwIntpWorkChainState.WAITING
            message = f'is waiting at {namespace}'
        elif workchain.process_state == ProcessState.RUNNING:
            state = EpwIntpWorkChainState.RUNNING
            message = f'is running at {namespace}'
        elif workchain.process_state == ProcessState.EXCEPTED:
            state = excepted_state
            message = f'has excepted at {namespace}'
        elif workchain.process_state == ProcessState.FINISHED:
            if not workchain.is_finished_ok:
                state = failed_state
                message = f'has failed at {namespace}'
            else:
                state = EpwIntpWorkChainState.UNKNOWN
                message = f'{namespace} is finished'
        else:
            state = EpwIntpWorkChainState.UNKNOWN
            message = f'unknown state at {namespace}'

        return state, message

    def check_b2w(self):
        """Check the state of the b2w workchain."""
        return self.base_check(
            self.descendants['b2w'][0],
            EpwIntpWorkChainState.B2W_EXCEPTED,
            EpwIntpWorkChainState.B2W_FAILED,
            'b2w'
        )

    def check_intp(self):
        """Check the state of the intp workchain."""
        return self.base_check(
            self.node,
            EpwIntpWorkChainState.EPW_BASE_EXCEPTED,
            EpwIntpWorkChainState.EPW_BASE_FAILED,
            'intp'
        )

    def check_process_state(self):
        """Check the state of the workchain."""
        if self.node.is_excepted:
            state = EpwIntpWorkChainState.EXCEPTED
            message = 'has excepted at intp'
        elif self.node.process_state == ProcessState.WAITING:
            state = EpwIntpWorkChainState.WAITING
            message = 'is waiting at intp'
        elif self.node.process_state == ProcessState.RUNNING:
            state = EpwIntpWorkChainState.RUNNING
            message = 'is running at intp'
        elif self.node.exit_status == 0:
            state = EpwIntpWorkChainState.FINISHED_OK
            message = 'has finished successfully at intp'
        elif self.node.exit_status == 400:
            state, message = self.check_b2w()
        elif self.node.exit_status == 401:
            state, message = self.check_intp()

        return state, message