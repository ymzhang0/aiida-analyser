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


class EpwPrepWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the EpwPrepWorkChain.
    """
    def __init__(self, workchain: orm.WorkChainNode):
        self.node = workchain

    @property
    def w90_intp(self):
        if 'w90_intp' not in self.process_tree:
            raise AttributeError('w90_intp is not found')
        else:
            return self.process_tree.w90_intp.node

    @property
    def ph_base(self):
        if 'ph_base' not in self.process_tree:
            raise AttributeError('ph_base is not found')
        else:
            return self.process_tree.ph_base.node
    @property
    def ph_base_analyser(self):
        if self.ph_base is None:
            raise AttributeError('ph_base is not found')
        else:
            return PhBaseWorkChainAnalyser(self.process_tree.ph_base.node)
    @property
    def q2r_base(self):
        if 'q2r_base' not in self.process_tree:
            raise ValueError('q2r_base is not found')
        else:
            return self.process_tree.q2r_base.node

    @property
    def matdyn_base(self):
        if 'matdyn_base' not in self.process_tree:
            raise ValueError('matdyn_base is not found')
        else:
            return self.process_tree.matdyn_base.node

    @property
    def epw_base(self):
        if self.process_tree.epw_base.node is None:
            raise ValueError('epw_base is not found')
        else:
            return self.process_tree.epw_base.node

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

        if self.node.is_finished_ok:
            return 'ROOT', 0, 'finished OK'
        
        for subprocess_name, subprocess_analyser in [
            ('w90_bands', Wannier90WorkChainAnalyser), 
            ('ph_base', PhBaseWorkChainAnalyser), 
            ('epw_base', EpwBaseWorkChainAnalyser), 
            ('epw_bands', EpwBaseWorkChainAnalyser)
            ]:
            if subprocess_name in self.process_tree:
                if not self.process_tree[subprocess_name].node.is_finished_ok:
                    analyser = subprocess_analyser(self.process_tree[subprocess_name].node)
                    path, exit_code, message = analyser.get_state()
                    return path, exit_code, message
                else:
                    continue

        raise ValueError(f'Unknown exit status: {self.node.exit_status}')

    def check_stability_matdyn_base(self):
        """Get the qpoints and frequencies of the matdyn_base workchain."""
        state, _ = self.check_matdyn_base()
        if state == EpwPrepWorkChainState.MATDYN_BASE_FINISHED_OK:
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