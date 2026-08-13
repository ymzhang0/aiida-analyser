from aiida import orm
import logging
from ..core.base import BaseWorkChainAnalyser
from .pw_base import PwBaseAnalyser
from ..core.groupdata import BaseGroupData, render_process_node_details
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

    def select_nodes_by_extras(self, **criteria):
        """Return workchain nodes whose extras match every supplied value.

        The search is limited to the ``PwRelaxWorkChain`` nodes loaded from
        this group's configured AiiDA groups.  With no criteria, all loaded
        nodes are returned.
        """
        selected_nodes = []
        for row in self._data:
            node = row.get('node')
            if node is None:
                continue
            try:
                extras = node.base.extras.all
            except Exception as exception:
                logger.warning(f'Could not read extras from node<{node.pk}>: {exception}')
                continue
            if all(extras.get(key) == value for key, value in criteria.items()):
                selected_nodes.append(node)
        return selected_nodes

    def select_node_by_extras(self, **criteria):
        """Return exactly one matching workchain node.

        Raises:
            ValueError: If the criteria match zero or more than one node.
        """
        selected_nodes = self.select_nodes_by_extras(**criteria)
        if len(selected_nodes) != 1:
            raise ValueError(
                f'Expected one PwRelaxWorkChain node for extras {criteria!r}; '
                f'found {len(selected_nodes)}.'
            )
        return selected_nodes[0]

    def plot_structure_convergence(self, quantity='celldm1', formula=None, ax=None,
                                   degauss_values=None, kpoints_distances=None,
                                   marker='o', legend=True, **plot_kwargs):
        """Plot a relaxed cell parameter against k-point distance and degauss.

        Each curve represents one degauss. ``quantity`` accepts celldm1--6
        (or a, b, c, alpha, beta, gamma), mapped to output_structure cell
        lengths and angles. Returns ``(ax, {degauss: {distance: value}})``.
        """
        import matplotlib.pyplot as plt

        quantities = {
            'celldm1': ('cell_lengths', 0, r'$a$ ($\AA$)'),
            'celldm2': ('cell_lengths', 1, r'$b$ ($\AA$)'),
            'celldm3': ('cell_lengths', 2, r'$c$ ($\AA$)'),
            'celldm4': ('cell_angles', 0, r'$\alpha$ (deg)'),
            'celldm5': ('cell_angles', 1, r'$\beta$ (deg)'),
            'celldm6': ('cell_angles', 2, r'$\gamma$ (deg)'),
        }
        aliases = {'a': 'celldm1', 'b': 'celldm2', 'c': 'celldm3',
                   'alpha': 'celldm4', 'beta': 'celldm5', 'gamma': 'celldm6'}
        if isinstance(quantity, int):
            quantity = f'celldm{quantity}'
        quantity_key = aliases.get(str(quantity).lower(), str(quantity).lower())
        if quantity_key not in quantities:
            raise ValueError("quantity must be one of 'celldm1' through 'celldm6' "
                             "(or 'a', 'b', 'c', 'alpha', 'beta', 'gamma').")
        attribute, index, ylabel = quantities[quantity_key]

        formulas = list(self._nested_data)
        if formula is None:
            if len(formulas) != 1:
                raise ValueError(f'formula is required; available materials: {formulas}')
            formula = formulas[0]
        formula_data = self._nested_data.get(formula, {})
        if not formula_data:
            raise ValueError(f'No PwRelaxWorkChain data found for formula {formula!r}.')

        allowed_degauss = set(degauss_values) if degauss_values is not None else None
        allowed_kpoints = set(kpoints_distances) if kpoints_distances is not None else None
        values = {}
        for degauss, kpoints_data in formula_data.items():
            if allowed_degauss is not None and degauss not in allowed_degauss:
                continue
            points = {}
            for kpoints_distance, candidates in kpoints_data.items():
                if allowed_kpoints is not None and kpoints_distance not in allowed_kpoints:
                    continue
                try:
                    distance = float(kpoints_distance)
                except (TypeError, ValueError):
                    logger.warning(f'Skipping non-numeric kpoints_distance: {kpoints_distance!r}')
                    continue
                candidates = [node for node in candidates if getattr(node, 'is_finished_ok', False)]
                if not candidates:
                    continue
                node = max(candidates, key=lambda item: getattr(item, 'pk', -1))
                try:
                    points[distance] = float(getattr(node.outputs.output_structure, attribute)[index])
                except (AttributeError, IndexError, TypeError, ValueError) as exception:
                    logger.warning(f'Could not read {quantity_key} from node<{node.pk}>: {exception}')
            if points:
                values[degauss] = dict(sorted(points.items()))

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        for degauss in sorted(values, key=lambda value: str(value)):
            points = values[degauss]
            ax.plot(list(points), list(points.values()), marker=marker,
                    label=rf'$\sigma$ = {degauss} Ry', **plot_kwargs)
        ax.set_xlabel(r'k-points distance ($\AA^{-1}$)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{formula}: {quantity_key} convergence')
        ax.grid(True, alpha=0.3)
        if legend and values:
            ax.legend(title='degauss')
        return ax, values
