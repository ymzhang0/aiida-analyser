from aiida import orm
from ..base import BaseWorkChainAnalyser
from .pw_base import PwBaseWorkChainAnalyser
from ..groupdata import BaseGroupData, render_process_node_details
from pathlib import Path

from collections import defaultdict

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
        for grpname in self._groups:
            group = orm.load_group(grpname)
            for node in group.nodes:
                if node.process_label not in ['PwRelaxWorkChain']:
                    continue
                if 'structure' in node.inputs:
                    try:
                        formula = node.inputs.structure.get_formula()
                    except Exception:
                        logger.error(f'Error getting formula for node<{node.pk}>')
                        formula = 'N/A'
                if formula == 'N/A' and 'formula' in node.base.extras:
                    formula = node.base.extras.get('formula')
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
                        # Material
                        if 'structure' in node.inputs._get_keys():
                            try:
                                formula = node.inputs.structure.get_formula()
                            except Exception:
                                logging.error(f'Error getting formula for node<{node.pk}>')
                        if formula == 'N/A' and 'formula' in node.base.extras:
                            formula = node.base.extras.get('formula')

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

    def _get_dataframe(self, property_filter=None):
        """Build one row per EpwPrep work chain, indexed by PK."""
        import pandas as pd

        if not self._data:
            return pd.DataFrame(columns=['Material', 'degauss', 'kpoints_distance', 'Process', 'Status'])

        data_list = self._data
        if property_filter:
            filtered_list = []
            for item in data_list:
                try:
                    analyser = EpwPrepWorkChainAnalyser(item['node'])
                    if callable(property_filter):
                        res = property_filter(analyser)
                    elif isinstance(property_filter, str):
                        prop = property_filter.strip()
                        if prop.startswith('not '):
                            res = not bool(getattr(analyser, prop[4:].strip(), False))
                        elif prop.startswith('!'):
                            res = not bool(getattr(analyser, prop[1:].strip(), False))
                        elif prop.startswith('~'):
                            res = not bool(getattr(analyser, prop[1:].strip(), False))
                        else:
                            res = bool(getattr(analyser, prop, False))
                    else:
                        res = False
                    
                    if res:
                        filtered_list.append(item)
                except Exception as e:
                    logging.warning(f"Error filtering node {item['PK']}: {e}")
                    pass
            data_list = filtered_list
            
        if not data_list:
            return pd.DataFrame(columns=['PK', 'Material', 'degauss', 'kpoints_distance', 'Status']).set_index('PK')

        dataframe = pd.DataFrame(data_list).drop(columns=['node'])
        return dataframe.set_index('PK').sort_index()

    @staticmethod
    def _filter_by_formula(dataframe, formula_contains=None, formula_match='any'):
        """Filter rows by case-insensitive substrings in the material formula."""
        if formula_contains is None or formula_contains == '':
            return dataframe

        terms = (
            [formula_contains]
            if isinstance(formula_contains, str)
            else list(formula_contains)
        )
        terms = [str(term).strip().lower() for term in terms if str(term).strip()]
        if not terms:
            return dataframe
        if formula_match not in {'any', 'all'}:
            raise ValueError("formula_match must be either 'any' or 'all'")

        formulas = dataframe['Material'].astype(str).str.lower()
        masks = [formulas.str.contains(term, regex=False) for term in terms]
        mask = masks[0]
        for next_mask in masks[1:]:
            mask = mask | next_mask if formula_match == 'any' else mask & next_mask
        return dataframe.loc[mask]

    def get_table(
        self,
        display_mode='dataframe',
        *,
        max_height=600,
        page_size=25,
        formula_contains=None,
        formula_match='any',
        property_filter=None,
    ):
        """Return or display the EpwPrep table.

        :param display_mode: One of ``dataframe`` (return the normal DataFrame),
            ``all`` (display every row), ``scroll`` (display a scrollable table),
            or ``interactive`` (searchable, paginated node browser).
        :param max_height: Maximum table height in pixels for scrollable modes.
        :param page_size: Initial number of rows per page in interactive mode.
        :param formula_contains: A substring or iterable of substrings that must
            occur in the material formula, matched case-insensitively.
        :param formula_match: Use ``any`` to match at least one substring or
            ``all`` to require every substring.
        :param property_filter: A string of a boolean property in EpwPrepWorkChainAnalyser
            to filter the workchains (e.g. 'is_stable').
        """
        dataframe = self._filter_by_formula(
            self._get_dataframe(property_filter=property_filter),
            formula_contains=formula_contains,
            formula_match=formula_match,
        )
        mode = 'dataframe' if display_mode is None else str(display_mode).lower()

        if mode in {'dataframe', 'default'}:
            return dataframe

        if mode == 'all':
            import pandas as pd
            from IPython.display import display

            with pd.option_context('display.max_rows', None):
                display(dataframe)
            return None

        if mode == 'scroll':
            from IPython.display import HTML, display

            display(HTML(
                f'<div style="max-height:{int(max_height)}px; overflow:auto;">'
                f'{dataframe.to_html()}</div>'
            ))
            return None

        if mode == 'interactive':
            return self.show_interactive(
                max_height=max_height,
                page_size=page_size,
                formula_contains=formula_contains,
                formula_match=formula_match,
            )

        raise ValueError(
            "display_mode must be one of 'dataframe', 'all', 'scroll', or 'interactive'"
        )

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
