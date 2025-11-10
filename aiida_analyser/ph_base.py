from re import S
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from enum import Enum
from collections import OrderedDict

from .base import BaseWorkChainAnalyser

class PhBaseWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the PhBaseWorkChain.
    """

    def merge_output_parameters(self):
        """Merge the output parameters of the workchain."""

        output_parameters = {}

        for child in self.process_tree.children.values():
            if child.node.process_label == 'PhCalculation':
                if child.node.is_finished and 'output_parameters' in child.node.outputs:
                    output_parameters.update(child.node.outputs.output_parameters.get_dict())
                else:
                    continue
            else:
                continue

        return output_parameters

    @staticmethod
    def get_qpoints_and_frequencies(output_parameters):

        nqpoints = output_parameters.get('number_of_qpoints')
        iqs = []
        q_points = []
        frequencies = []

        for key, value in output_parameters.items():
            if key.startswith('dynamical_matrix_'):
                iqs.append(int(key.split('_')[2]))
                frequencies.append(value.get('frequencies'))
                q_points.append(value.get('q_point'))

        q_points = sorted(q_points, key=lambda x: iqs[q_points.index(x)])
        frequencies = sorted(frequencies, key=lambda x: iqs[frequencies.index(x)])

        return nqpoints, q_points, frequencies
        
    @staticmethod
    def _is_stable(
        qpoints, 
        frequencies,
        message = '',
        tolerance: float = -5.0 # cm^{-1}
        ) -> tuple[bool, str]:
        """Check if the workchain is stable."""
        is_stable = True
        negative_freqs = {}

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
            message += 'Phonon is stable from `ph_base`.'
        else:
            message += f'Phonon is unstable from `ph_base`.\n'
            for iq, freqs in negative_freqs.items():
                q_points_str = ', '.join(map(str, qpoints[iq-1]))
                negative_freqs_str = ', '.join(map(str, freqs))
                message += f'{iq}th qpoint ({q_points_str}) has negative frequencies: {negative_freqs_str} cm^{-1}\n'
        return is_stable, message

    @property
    def is_stable(self):
        """Check if the workchain is stable."""

        if self.node.is_finished_ok:
            header = f"PhBaseWorkChain<{self.node.pk}> finished OK:\n"
            output_parameters = self.node.outputs.output_parameters.get_dict()
        else:
            output_parameters = self.merge_output_parameters()
            header = f"PhBaseWorkChain<{self.node.pk}> exited with status {self.node.exit_status}:\n"

        nqs, q_points, frequencies = self.get_qpoints_and_frequencies(output_parameters)
        if not len(q_points):
            print('No q-points found')
            return True

        is_stable, message = self._is_stable(
            q_points,
            frequencies,
            message = header + f"From the calculated {len(q_points)} q-points out of {nqs} we find:\n"
            )
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
            exit_status = None
            last_node = self.process_tree.find_last_node().node
            if last_node.process_label != 'PhCalculation':
                raise ValueError(f'The last node is {last_node.process_label}<{last_node.pk}>, not a PhCalculation.')
            aiida_out = last_node.outputs.retrieved.get_object_content('aiida.out')
            stderr = last_node.get_scheduler_stderr()
            for error_flag, error_message in [
                ('ERROR_FIND_MODE_SYM', 'Error in routine find_mode_sym (1)'),
                ('ERROR_SET_IRR_SYM_NEW', 'Error in routine set_irr_sym_new (922)'),
                ('ERROR_WRONG_REPRESENTATION', 'Error in routine set_irr_sym_new (822)'),
                ('ERROR_CDIAGHG', 'Error in routine cdiaghg (4)'),
                ('ERROR_S_MATRIX_NOT_POSITIVE_DEFINITE', 'Error in routine cdiaghg (126)'),
                ('ERROR_PHQ_SETUP', 'Error in routine phq_setup (1)'),
                ('ERROR_Q_POINTS', 'Error in routine q_points (1)'),
                ('ERROR_DAVCIO', 'Error in routine davcio (99)'),
                ('ERROR_CHECK_ALL_CONVT', 'Error in routine check_all_convt (1)'),
                ('ERROR_READ_WFC', 'Error in routine read_wfc (29)'),
            ]:
                if error_message in aiida_out:
                    exit_status = error_flag
                    break

            if not exit_status:
                if 'TIME LIMIT' in stderr:
                    exit_status = 'SCHEDULER_TIME_LIMIT'
                elif 'process killed' in stderr:
                    exit_status = 'KILLED_BY_SCHEDULER'
            else:   
                exit_status = -1

        if exit_status == 0:
            is_stable, message = self.is_stable()
            if not is_stable:
                message += f'\n    Phonon is unstable from `ph_base`.'
                exit_status = 'UNSTABLE'

        return path, exit_status, message

    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message = super().clean_workchain([
            ],
            dry_run=dry_run
            )

        return message