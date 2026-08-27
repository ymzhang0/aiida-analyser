from itertools import chain
import logging
from loguru import logger
from ..core.groupdata import DegaussKGroup

from ..core.base import BaseWorkChainAnalyser

class PhBaseAnalyser(BaseWorkChainAnalyser):
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
        qpoint_data = []

        for key, value in output_parameters.items():
            if key.startswith('dynamical_matrix_'):
                qpoint_data.append((
                    int(key.split('_')[2]),
                    value.get('q_point'),
                    value.get('frequencies'),
                ))

        qpoint_data.sort(key=lambda entry: entry[0])
        q_points = [q_point for _, q_point, _ in qpoint_data]
        frequencies = [frequency for _, _, frequency in qpoint_data]

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
            min_freq = min(chain.from_iterable(negative_freqs.values()))
            message += f'Phonon is unstable from `ph_base`.\n'
            # for iq, freqs in negative_freqs.items():
                # q_points_str = ', '.join(map(str, qpoints[iq-1]))
                # negative_freqs_str = ', '.join(map(str, freqs))
                # message += f'{iq}th qpoint ({q_points_str}) has negative frequencies: {negative_freqs_str} cm^{-1}\n'
            message += f'{len(negative_freqs)} qpoints have negative frequency. minimum frequency: {min_freq} cm^{-1}'
        return is_stable, message

    def is_stable(self, mute_print=False):
        """Check if the workchain is stable."""

        if self.node.is_finished_ok:
            header = f"PhBaseWorkChain<{self.node.pk}> finished OK:\n"
            output_parameters = self.node.outputs.output_parameters.get_dict()
        else:
            output_parameters = self.merge_output_parameters()
            header = f"PhBaseWorkChain<{self.node.pk}> exited with status {self.node.exit_status}:\n"

        nqs, q_points, frequencies = self.get_qpoints_and_frequencies(output_parameters)
        if not len(q_points):
            logging.getLogger('aiida_analyser').warning(f'{self.node_ref} no q-points found.')
            return (False, 0, nqs)
        else:
            is_stable, message = self._is_stable(
                q_points,
                frequencies,
                message = header + f"From the calculated {len(q_points)} q-points out of {nqs} we find:\n"
                )
        if not mute_print:
            package_logger = logging.getLogger('aiida_analyser')
            if is_stable:
                package_logger.info(message)
            else:
                package_logger.warning(message)
        return (is_stable, len(q_points), nqs)

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
        path, process_state, exit_code = self._get_state_from_tree()

        if self.node.is_finished_ok:
            try:
                is_stable, _, _ = self.is_stable(mute_print=True)
            except Exception:
                return path, process_state, exit_code

            if not is_stable:
                return path, 'UNSTABLE', exit_code

        if process_state == 'finished' and getattr(exit_code, 'status', exit_code) == 312:
            frontiers = self.get_failure_frontiers()
            if not frontiers:
                return path, process_state, exit_code

            _, failure_tree = frontiers[0]
            failure_node = failure_tree.node
            if failure_node.process_label != 'PhCalculation':
                return path, process_state, exit_code

            try:
                aiida_out = failure_node.outputs.retrieved.get_object_content('aiida.out')
            except (AttributeError, KeyError):
                aiida_out = ''

            try:
                stderr = failure_node.get_scheduler_stderr() or ''
            except (AttributeError, KeyError):
                stderr = ''

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
                    return path, error_flag, exit_code

            if 'TIME LIMIT' in stderr:
                return path, 'SCHEDULER_TIME_LIMIT', exit_code

            if 'process killed' in stderr.lower():
                return path, 'KILLED_BY_SCHEDULER', exit_code

        return path, process_state, exit_code

    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message, success = super().clean_workchain(dry_run=dry_run)

        return message


class PhBaseGroup(DegaussKGroup):
    analyser_class = PhBaseAnalyser
    process_label = 'PhBaseWorkChain'

    def plot_convergence(self):
        import re
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for degauss, degauss_dict in self._data.items():
            sorted_degauss_dict = dict(sorted(degauss_dict.items()))
            ax.scatter(list(sorted_degauss_dict.keys()), list(sorted_degauss_dict.values()))
            ax.plot(list(sorted_degauss_dict.keys()), list(sorted_degauss_dict.values()), label=degauss)
        ax.set_xlabel('Number of Kpoints')
        ax.set_ylabel('Frequency [cm$^{-1}$] @ [-0.5, 0, 0]')
        ax.legend()
        ax.set_xscale('log')
        fig.tight_layout()
