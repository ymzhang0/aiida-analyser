from re import S
import re
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from enum import Enum
from collections import OrderedDict
from .ph import check_stability_matdyn_base
from .base import BaseWorkChainAnalyser
from .wannier90 import Wannier90WorkChainAnalyser
from .ph_base import PhBaseWorkChainAnalyser
from .epw_base import EpwBaseWorkChainAnalyser

class EpwB2WWorkChainState(Enum):
    """
    Analyser for the B2WWorkChain.
    """
    FINISHED_OK = 0
    WAITING = 1
    RUNNING = 2
    EXCEPTED = 3
    KILLED = 4
    W90_INTP_SCF_FAILED = 4004
    W90_INTP_NSCF_FAILED = 4005
    W90_INTP_PW2WAN_FAILED = 4006
    W90_INTP_WANNIER_FAILED = 4007
    W90_INTP_FINISHED_OK = 4008
    PH_BASE_FAILED = 4009
    PH_BASE_S_MATRIX_NOT_POSITIVE_DEFINITE = 4010
    PH_BASE_NODE_FAILURE = 4011
    PH_BASE_UNSTABLE = 4012
    PH_BASE_FINISHED_OK = 4013
    Q2R_BASE_FAILED = 4014
    Q2R_BASE_FINISHED_OK = 4015
    MATDYN_BASE_FAILED = 4016
    MATDYN_BASE_UNSTABLE = 4017
    MATDYN_BASE_FINISHED_OK = 4018
    EPW_BASE_FAILED = 4019
    EPW_BASE_FINISHED_OK = 4020
    W90_INTP_EXCEPTED = 904
    PH_BASE_EXCEPTED = 905
    Q2R_BASE_EXCEPTED = 906
    MATDYN_BASE_EXCEPTED = 907
    EPW_BASE_EXCEPTED = 908
    UNKNOWN = 999

class EpwB2WWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the B2WWorkChain.
    """
    _all_descendants = OrderedDict([
        ('w90_intp', None),
        ('ph_base',  None),
        ('q2r_base', None),
        ('matdyn_base', None),
        ('epw_base', None),
    ])

    def __init__(self, workchain: orm.WorkChainNode):
        self.node = workchain
        self.state = EpwB2WWorkChainState.UNKNOWN
        self.descendants = {}
        for link_label, _ in self._all_descendants.items():
            descendants = workchain.base.links.get_outgoing(link_label_filter=link_label).all_nodes()
            if descendants != []:
                self.descendants[link_label] = descendants

    @property
    def w90_intp(self):
        if self.descendants['w90_intp'] == []:
            raise ValueError('w90_intp is not found')
        else:
            return self.descendants['w90_intp']

    @property
    def ph_base(self):
        if self.descendants['ph_base'] == []:
            raise ValueError('ph_base is not found')
        else:
            return self.descendants['ph_base']

    @property
    def q2r_base(self):
        if self.descendants['q2r_base'] == []:
            raise ValueError('q2r_base is not found')
        else:
            return self.descendants['q2r_base']

    @property
    def matdyn_base(self):
        if self.descendants['matdyn_base'] == []:
            raise ValueError('matdyn_base is not found')
        else:
            return self.descendants['matdyn_base']

    @property
    def epw_base(self):
        if self.descendants['epw_base'] == []:
            raise ValueError('epw_base is not found')
        else:
            return self.descendants['epw_base']

    def get_source(self):
        """Get the source of the workchain."""
        return super().get_source()

    def get_iterations(self, link_label: str):
        """Get the iterations of the workchain."""

        iterations = []
        for (node, link_type, link_label) in self.descendants[link_label][-1].base.links.get_outgoing().all():
            if link_label.startswith('iteration'):
                iterations.append(node)
        return iterations

    def get_state(self):
        """Get the state of the workchain."""
        process_tree = self.process_tree

        # if self.node.process_state.name in ['CREATED', 'RUNNING', 'EXCEPTED', 'KILLED']:
        #     return 'ROOT', -2, self.node.process_state.value
        if self.node.is_finished_ok:
            return 'ROOT', 0, 'finished OK'
        elif not process_tree.w90_bands.node.is_finished_ok:
            w90_bands_analyser = Wannier90WorkChainAnalyser(process_tree.w90_bands.node)
            path, exit_code, message = w90_bands_analyser.get_state()
            return 'w90_bands', exit_code, message
        elif not process_tree.ph_base.node.is_finished_ok:
            ph_base_analyser = PhBaseWorkChainAnalyser(process_tree.ph_base.node)
            path, exit_code, message = ph_base_analyser.get_state()
            return 'ph_base', exit_code, message
        elif 'epw' in process_tree and not process_tree.epw.node.is_finished_ok:
            epw_analyser = EpwBaseWorkChainAnalyser(process_tree.epw.node)
            path, exit_code, message = epw_analyser.get_state()
            return 'epw', exit_code, message
        elif 'epw_base' in process_tree and not process_tree.epw_base.node.is_finished_ok:
            epw_base_analyser = EpwBaseWorkChainAnalyser(process_tree.epw_base.node)
            path, exit_code, message = epw_base_analyser.get_state()
            return 'epw_base', exit_code, message
        else:
            # return super().get_state()
            raise ValueError(f'Unknown exit status: {self.node.exit_status}')

    def check_stability_matdyn_base(self):
        """Get the qpoints and frequencies of the matdyn_base workchain."""
        state, _ = self.check_matdyn_base()
        if state == EpwB2WWorkChainState.MATDYN_BASE_FINISHED_OK:
            return check_stability_matdyn_base(self.matdyn_base[0])
        else:
            raise ValueError('matdyn_base is not finished')

    def clean_workchain(self, exempted_states=[], dry_run=True):
        """Clean the workchain."""
        message = super().clean_workchain(
            exempted_states,
            dry_run=dry_run
        )
        return message