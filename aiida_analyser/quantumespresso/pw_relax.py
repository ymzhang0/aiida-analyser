from aiida import orm
import logging
from ..base import BaseWorkChainAnalyser
from .pw_base import PwBaseWorkChainAnalyser
from ..groupdata import BaseGroupData, render_process_node_details
from pathlib import Path

from collections import defaultdict
from loguru import logger

class PwRelaxWorkChainAnalyser(BaseWorkChainAnalyser):
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
                subprocesses.append((label, PwBaseWorkChainAnalyser))

        iteration_labels = sorted(
            (
                child_name for child_name, child_tree in self.process_tree.children.items()
                if child_tree.node.process_label == 'PwBaseWorkChain' and child_name.startswith('iteration_')
            ),
            key=lambda label: int(label.split('_')[1]),
        )
        subprocesses.extend((label, PwBaseWorkChainAnalyser) for label in iteration_labels)

        trailing_pw_bases = [
            child_name for child_name, child_tree in self.process_tree.children.items()
            if child_tree.node.process_label == 'PwBaseWorkChain'
            and child_name not in {name for name, _ in subprocesses}
        ]
        subprocesses.extend((label, PwBaseWorkChainAnalyser) for label in trailing_pw_bases)

        return self._get_state_from_subprocesses(subprocesses)

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct PwBaseWorkChain child."""
        return self._copy_tree_for_direct_children(
            destpath,
            lambda _, child: PwBaseWorkChainAnalyser if child.node.process_label == 'PwBaseWorkChain' else None,
        )

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct PwBaseWorkChain child."""
        return self._get_calcjob_paths_for_direct_children(
            lambda _, child: PwBaseWorkChainAnalyser if child.node.process_label == 'PwBaseWorkChain' else None,
        )


def _safe_get_extras(node):
    extras = node.base.extras.all
    
    # Degauss
    degauss = extras.get('degauss', 'unknown')
    # K-point distance (can be stored as 'kpoints_distance_scf' or 'kpoints_distance')

    kpoints_distance = extras.get('kpoints_distance', None)
    return degauss, kpoints_distance


class PwRelaxWorkChainData(BaseGroupData):

    analyser_class = PwRelaxWorkChainAnalyser
    dataframe_columns = ('Material', 'degauss', 'kpoints_distance', 'status')

    def __init__(self, groups=None):
        super().__init__(groups)
        # Data structure: Material -> Degauss -> K_Dist -> Node
        self._nested_data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: None
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

    def dump(self, dest:Path|str,):
        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={'label': {'in': self._groups}}, tag='group')
        qb.append(orm.ProcessNode, with_group='group', filters={'attributes.process_label': 'PwRelaxWorkChain'})

        if type(dest) == str:
            dest = Path(dest)
        if not dest.exists():
            dest.mkdir(parents=True)

        for [node] in qb.all():
            try:
                analyser = PwRelaxWorkChainAnalyser(node)
                analyser.copy_tree(dest / str(node.pk))
            except Exception as e:
                logging.warning(f"Failed to dump node {node.pk}: {e}")
