from re import S
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from enum import Enum
from collections import OrderedDict
from .base import BaseWorkChainAnalyser

class EpwBaseWorkChainState(Enum):
    """
    Analyser for the EpwBaseWorkChain.
    """
    FINISHED_OK = 0
    WAITING = 1
    RUNNING = 2
    EXCEPTED = 3
    KILLED = 4
    S_MATRIX_NOT_POSITIVE_DEFINITE = 4010
    NODE_FAILURE = 4011
    UNSTABLE = 4012
    UNKNOWN = 999

class EpwBaseWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the EpwBaseWorkChain.
    """
    _all_descendants = OrderedDict([
        ('epw',  None),
    ])

    def __init__(self, workchain: orm.WorkChainNode):
        self.node = workchain
        self.state = EpwBaseWorkChainState.UNKNOWN
        self.descendants = {}
        for link_label, _ in self._all_descendants.items():
            descendants = workchain.base.links.get_outgoing(link_label_filter=link_label).all_nodes()
            if descendants != []:
                self.descendants[link_label] = descendants

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

        message = super().clean_workchain([
            EpwBaseWorkChainState.WAITING,
            EpwBaseWorkChainState.RUNNING,
            ],
            dry_run=dry_run
            )

        return message