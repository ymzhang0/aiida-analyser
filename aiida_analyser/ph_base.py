from re import S
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from enum import Enum
from collections import OrderedDict

from .base import BaseWorkChainAnalyser

class PhBaseWorkChainState(Enum):
    """
    Analyser for the PhWorkChain.
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

class PhBaseWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the PhBaseWorkChain.
    """


    def get_qpoints_and_frequencies(self):

        nqpoints = self.node.outputs.output_parameters.get('number_of_qpoints')
        frequencies = [
            self.node.outputs.output_parameters.get(f'dynamical_matrix_{iq}').get('frequencies') # unit: cm^{-1} (THz)
            for iq in range(1, 1+nqpoints)
        ]
        q_points = [
            self.node.outputs.output_parameters.get(f'dynamical_matrix_{iq}').get('q_point')
            for iq in range(1, 1+nqpoints)
        ]
        return q_points, frequencies

    def is_stable(
        self, 
        tolerance: float = -5.0 # cm^{-1}
        ) -> tuple[bool, str]:
        """Check if the workchain is stable."""
        is_stable = True
        message = ''
        negative_freqs = {}

        qpoints, frequencies = self.get_qpoints_and_frequencies()

        neg_freq0 = [f for f in frequencies[0] if f < 0]
        if len(neg_freq0) > 3:
            is_stable = False
            negative_freqs[1] = neg_freq0

        for iq, freq in enumerate(frequencies[1:]):
            neg_freq = [f for f in freq if f < tolerance]
            if len(neg_freq) > 0:
                is_stable = False
                negative_freqs[iq+2] = neg_freq

        if is_stable:
            message = 'Phonon is stable from `ph_base`.'
        else:
            message = f'Phonon is unstable from `ph_base`.\n'
            for iq, freqs in negative_freqs.items():
                q_points_str = ', '.join(map(str, qpoints[iq-1]))
                negative_freqs_str = ', '.join(map(str, freqs))
                message += f'{iq}th qpoint [{q_points_str}] has negative frequencies: {negative_freqs_str} cm^{-1}\n'
        print(message)
        return is_stable

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

        path, exit_status, message = super().get_state()
        
        if exit_status == 312:
            last_node = self.process_tree.find_last_node()
            if last_node.process_label != 'PhCalculation':
                raise ValueError('The last node is not a PhCalculation.')
            aiida_out = last_node.node.outputs.retrieved.get_object_content('aiida.out')[-2000:]
            print(aiida_out)

        if exit_status == 0:
            is_stable, message = self.is_stable()
            if not is_stable:
                message += f'\n    Phonon is unstable from `ph_base`.'
            return path, 'UNSTABLE', message

    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message = super().clean_workchain([
            PhBaseWorkChainState.WAITING,
            PhBaseWorkChainState.RUNNING,
            ],
            dry_run=dry_run
            )

        return message