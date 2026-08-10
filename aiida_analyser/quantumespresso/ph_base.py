from collections import defaultdict
from itertools import chain
import logging
import warnings
from loguru import logger
from ..groupdata import BaseGroupData, render_process_node_details

from aiida import orm

from ..base import BaseWorkChainAnalyser
from .ph_calculation import PhCalculationAnalyser
from pathlib import Path

class PhBaseWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the PhBaseWorkChain.
    """

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct PhCalculation child."""
        return self._copy_tree_for_direct_children(
            destpath,
            lambda _, child: PhCalculationAnalyser if child.node.process_label == 'PhCalculation' else None,
        )

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct PhCalculation child."""
        return self._get_calcjob_paths_for_direct_children(
            lambda _, child: PhCalculationAnalyser if child.node.process_label == 'PhCalculation' else None,
        )

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
            last_node = self.process_tree.find_last_node().node
            if last_node.process_label != 'PhCalculation':
                return path, process_state, self.node.exit_message

            try:
                aiida_out = last_node.outputs.retrieved.get_object_content('aiida.out')
            except (AttributeError, KeyError):
                aiida_out = ''

            try:
                stderr = last_node.get_scheduler_stderr() or ''
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
                    return path, error_flag, self.node.exit_message

            if 'TIME LIMIT' in stderr:
                return path, 'SCHEDULER_TIME_LIMIT', self.node.exit_message

            if 'process killed' in stderr.lower():
                return path, 'KILLED_BY_SCHEDULER', self.node.exit_message

        return path, process_state, exit_code

    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message, success = super().clean_workchain(dry_run=dry_run)

        return message


class PhData(BaseGroupData):
    analyser_class = PhBaseWorkChainAnalyser
    dataframe_columns = ('Material', 'degauss', 'kpoints_distance', 'status')
    dump_process_labels = 'PhBaseWorkChain'

    def __init__(self, groups=None):
        super().__init__(groups)
        self._nested_data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: None
                )
            )
        )
        self.get_data()
        self._data = self._flatten_data()
    
    def check_protocol(self, node):
        if node.process_label not in ['PhBaseWorkChain']:
            raise ValueError(f'Node<{node.pk}> is not a PhBaseWorkChain')
        extras = node.base.extras.all
        for key in ['formula', 'source_db', 'source_id', 'kpoints_distance', 'degauss', ]:
            if key not in extras:
                logger.debug(f'Extra {key} is not found in node<{node.pk}>', stacklevel=2)

    def get_data(self):
        for node in self.iter_group_nodes('PhBaseWorkChain'):
            try:
                extras = node.base.extras.all
                self.check_protocol(node)
                formula = self.get_node_formula(node, default=extras.get('formula', 'N/A'))
                degauss = extras.get('degauss', 'unknown')
                kpoints_distance = extras.get('kpoints_distance', 'unknown')
                self._nested_data[formula][degauss][kpoints_distance] = node
            except Exception as e:
                logging.error(f'Node<{node.pk}> processing failed: {e}')

    def _flatten_data(self):
        flattened_list = []
        for formula, degauss_dict in self._nested_data.items():
            for degauss, k_dist_dict in degauss_dict.items():
                for kpoints_distance, node in k_dist_dict.items():

                        # Emojified Status
                        status_emoji = self.get_status_string(node)

                        flattened_list.append({
                            'PK': node.pk,
                            'Material': formula,
                            'degauss': degauss,
                            'kpoints_distance': kpoints_distance,
                            'status': status_emoji,
                            'node': node,
                        })

        return flattened_list

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
