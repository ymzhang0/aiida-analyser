from aiida import orm
import logging
from ..base import BaseWorkChainAnalyser
from .pw_base import PwBaseAnalyser
from ..groupdata import BaseGroupData, render_process_node_details
from pathlib import Path

from collections import defaultdict
from loguru import logger

class PwRelaxAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the PwRelaxWorkChain.
    """

    def get_source(self):
        """Get the source of the workchain."""
        source = super().get_source()
        if source is None:
            try:
                source_db, source_id = self.node.inputs.structure.base.extras.get_many(('source_db', 'source_id'))
                source = f"{source_db}-{source_id}"
            except Exception:
                self._log_source_missing()
                return None
        return source

    def get_state(self):
        """Get the state of the workchain."""
        subprocesses = []

        for label in ('init_relax', 'base_init_relax'):
            if label in self.process_tree:
                subprocesses.append((label, PwBaseAnalyser))

        iteration_labels = sorted(
            (
                child_name for child_name, child_tree in self.process_tree.children.items()
                if child_tree.node.process_label == 'PwBaseWorkChain' and child_name.startswith('iteration_')
            ),
            key=lambda label: int(label.split('_')[1]),
        )
        subprocesses.extend((label, PwBaseAnalyser) for label in iteration_labels)

        trailing_pw_bases = [
            child_name for child_name, child_tree in self.process_tree.children.items()
            if child_tree.node.process_label == 'PwBaseWorkChain'
            and child_name not in {name for name, _ in subprocesses}
        ]
        subprocesses.extend((label, PwBaseAnalyser) for label in trailing_pw_bases)

        return self._get_state_from_subprocesses(subprocesses)

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct PwBaseWorkChain child."""
        return self._copy_tree_for_direct_children(
            destpath,
            lambda _, child: PwBaseAnalyser if child.node.process_label == 'PwBaseWorkChain' else None,
        )

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct PwBaseWorkChain child."""
        return self._get_calcjob_paths_for_direct_children(
            lambda _, child: PwBaseAnalyser if child.node.process_label == 'PwBaseWorkChain' else None,
        )


def _safe_get_extras(node):
    extras = node.base.extras.all
    
    # Degauss
    degauss = extras.get('degauss', 'unknown')
    # K-point distance (can be stored as 'kpoints_distance_scf' or 'kpoints_distance')

    kpoints_distance = extras.get('kpoints_distance', None)
    return degauss, kpoints_distance


class PwRelaxGroup(BaseGroupData):

    analyser_class = PwRelaxAnalyser
    dataframe_columns = ('Material', 'degauss', 'kpoints_distance', 'status')

    def __init__(self, groups=None):
        super().__init__(groups)
        # Data structure: Material -> Degauss -> K_Dist -> [Node, ...]
        self._nested_data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    list
                )
            )
        )
        self.get_data()
        self._data = self._flatten_data()

    @staticmethod
    def check_protocol(node):
        extras = node.base.extras.all
        if node.process_label in ['PwRelaxWorkChain']:
            for key in ['formula', 'source_db', 'source_id', 'kpoints_distance', 'degauss']:
                if key not in extras:
                    logger.debug(f'Extra {key} is not found in node<{node.pk}>', stacklevel=2)
                
        return True

    def get_data(self):
        for node in self.iter_group_nodes('PwRelaxWorkChain'):
            formula = self.get_node_formula(node)
            try:
                self.check_protocol(node)
                degauss, kpoints_distance = _safe_get_extras(node)
                self._nested_data[formula][degauss][kpoints_distance].append(node)
            except Exception as e:
                logging.error(f'Node<{node.pk}> processing failed: {e}')

    def _flatten_data(self):

        flattened_list = []
        for formula, degauss_data in self._nested_data.items():
            for degauss, kpoints_data in degauss_data.items():
                for kpoints_distance, nodes in kpoints_data.items():
                    for node in nodes:
                        flattened_list.append({
                            'PK': node.pk,
                            'Material': formula,
                            'degauss': degauss,
                            'kpoints_distance': kpoints_distance,
                            'status': self.get_status_string(node),
                            'node': node,
                        })

        return flattened_list
