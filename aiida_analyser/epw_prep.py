from aiida import orm
from .ph import check_stability_matdyn_base
from .base import BaseWorkChainAnalyser
from .wannier90 import Wannier90WorkChainAnalyser
from .ph_base import PhBaseWorkChainAnalyser
from .epw_base import EpwBaseWorkChainAnalyser


class EpwPrepWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the EpwPrepWorkChain.
    """

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
    def epw_base(self):
        if self.process_tree.epw_base.node is None:
            raise ValueError('epw_base is not found')
        else:
            return self.process_tree.epw_base.node

    @property
    def epw_bands(self):
        if 'epw_bands' not in self.process_tree:
            raise ValueError('epw_bands is not found')
        if self.process_tree.epw_bands.node is None:
            raise ValueError('epw_bands is not found')
        return self.process_tree.epw_bands.node

    def get_source(self):
        """Get the source of the workchain."""
        return super().get_source()


    def get_state(self):
        """Get the state of the workchain."""

        if self.node.is_finished_ok:
            return 'ROOT', 0, 'finished OK'
        
        # Check subprocesses in order
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
                    return f'{subprocess_name}/{path}' if path != 'ROOT' else subprocess_name, exit_code, message
        
        # If all subprocesses are finished but main node is not, use tree traversal
        # to find the actual error in the process tree
        return self._get_state_from_tree()

    def check_stability_matdyn_base(self):
        """Get the qpoints and frequencies of the matdyn_base workchain."""
        # TODO: This method needs to be implemented properly
        # It should check if matdyn_base workchain exists and is finished
        # For now, raise NotImplementedError
        raise NotImplementedError('check_stability_matdyn_base method is not yet implemented')

    def clean_workchain(self, exempted_states=[], dry_run=True):
        """Clean the workchain."""
        path, status, _ = self.get_state()
        message = f'Process<{self.node.pk}> is now {status} at {path}. Please check if you really want to clean this workchain.\n'
        if status in exempted_states:
            print(message)
            return message, False

        message, success = super().clean_workchain(dry_run=dry_run)
        return message, True