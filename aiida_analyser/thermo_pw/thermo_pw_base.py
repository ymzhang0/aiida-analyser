from aiida import orm
import numpy
from ..base import BaseWorkChainAnalyser
from ..groupdata import BaseGroupData, render_process_node_details
from pathlib import Path
from .thermo_pw_calculation import ThermoPwCalculationAnalyser

class ThermoPwBaseAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the ThermoPwBaseWorkChain.
    """
    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct calcjob to its analyser."""
        return self._copy_tree_for_direct_children(
            destpath,
            lambda _, child: ThermoPwCalculationAnalyser if child.node.process_label == 'Thermo_pwCalculation' else None,
        )

    def get_source(self):
        """Get the source of the workchain."""
        source = super().get_source()
        if source is None:
            try:
                source_db, source_id = self.node.inputs.thermo_pw.structure.base.extras.get_many(('source_db', 'source_id'))
                source = f"{source_db}-{source_id}"
            except Exception:
                print('Source is not set')
                return None
        return source

    def get_state(self):
        """Get the state of the workchain."""
        return self._get_state_from_tree()

    def print_state(self):
        """Print the state of the workchain."""
        result = self.get_state()
        if not result:
            print(f"Can't check the state of ThermoPwBaseWorkChain<{self.node.pk}>.")
            return
        path, process_state, exit_code = result
        normalized_exit_code = getattr(exit_code, 'status', exit_code)
        print(
            f"ThermoPwBaseWorkChain<{self.node.pk}> is {process_state} at {path} "
            f"(exit code: {normalized_exit_code})."
        )
    
    def get_moduli(self, modulus_type: str):
        """Get the moduli of the workchain."""
        if not self.node.is_finished_ok:
            return None
        moduli = {
            average: self.node.outputs.output_parameters.get('moduli').get(average).get(modulus_type) 
            for average in ['voigt', 'reuss', 'vrh']
            }
        return moduli

    @property
    def code(self):
        """Get the code of the workchain."""
        return self.node.inputs.thermo_pw.code

    @property
    def elastic_constants(self):
        """Get the elastic constants of the workchain."""
        if not self.node.is_finished_ok:
            return None
        return self.node.outputs.elastic_constants.get_array('elastic_constants')

    @property
    def is_stable(self):
        elastic_constants = self.elastic_constants
        if elastic_constants is None:
            return False
        return numpy.all(elastic_constants > 0)

    @property
    def bulk_modulus(self):
        """Get the moduli of the workchain."""
        return self.get_moduli('bulk_modulus_B')

    @property
    def young_modulus(self):
        """Get the Young modulus of the workchain."""
        return self.get_moduli('young_modulus_E')

    @property
    def shear_modulus(self):
        """Get the Shear modulus of the workchain."""
        return self.get_moduli('shear_modulus_G')

    @property
    def poisson_ratio(self):
        """Get the Poisson ratio of the workchain."""
        return self.get_moduli('poisson_ratio_n')

    @property
    def pugh_ratio(self):
        """Get the Pugh ratio of the workchain."""
        return self.get_moduli('pugh_ratio_r')

    @property
    def modified_pettifor_ratio(self):
        """Get the Pettifor ratio of the workchain."""
        bulk_modulus = self.bulk_modulus
        # Note that both the elastic constants and the bulk modulus are in kbar.
        elastic_constants = self.elastic_constants
        if bulk_modulus is None or elastic_constants is None:
            return None
        return {
            average: (elastic_constants[0][1] - elastic_constants[3][3]) / bulk_modulus[average] for average in ['voigt', 'reuss', 'vrh']
        }

    @property
    def pettifor_ratio(self):
        """Get the modified Pettifor ratio of the workchain."""
        young_modulus = self.young_modulus
        elastic_constants = self.elastic_constants
        if young_modulus is None or elastic_constants is None:
            return None
        return {
            average: (elastic_constants[0][1] - elastic_constants[3][3]) / young_modulus[average] for average in ['voigt', 'reuss', 'vrh']
        }

    def clean_workchain(self, dry_run: bool = True):
        """Clean the workchain."""
        message, success = super().clean_workchain(dry_run=dry_run)
        return message

    def get_fitting_coefficients(self):
        """Get the fitting coefficients of the workchain."""
        if not self.node.is_finished_ok:
            return None
        return self.node.outputs.output_parameters.get('elastic_constants_fitting')

    def plot_elastic_fitting(self, axis=None):
        """Plot the elastic fitting of the workchain."""
        if not self.node.is_finished_ok:
            return None
        fitting_coefficients = self.get_fitting_coefficients()
        if not axis:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        else:
            ax = axis

        for x, x_info in fitting_coefficients.items():
            for y, y_info in x_info.items():
                strains = numpy.array(y_info.get('strains'))
                stresses = numpy.array(y_info.get('stresses'))
                coefficients = numpy.array(y_info.get('coefficients'))
                ax.scatter(strains, stresses, color='blue')
                # ax.plot(strains, stresses, color='blue', label=f'{x}-{y}')
                polynomial = numpy.poly1d(coefficients[::-1])
                ax.plot(strains, 147100*polynomial(strains), color='red', label=f'{x}-{y} fitting')
        ax.legend(loc='best')  
        return ax

    def get_RMS_error(self):
        """Get the RMS error of the workchain."""
        if not self.node.is_finished_ok:
            return None

        RMS_errors = {}
        fitting_coefficients = self.get_fitting_coefficients()
        for x, x_info in fitting_coefficients.items():
            RMS_errors[x] = {}
            for y, y_info in x_info.items():
                strains = numpy.array(y_info.get('strains'))
                stresses = numpy.array(y_info.get('stresses'))
                coefficients = numpy.array(y_info.get('coefficients'))
                polynomial = numpy.poly1d(coefficients[::-1])
                errors = stresses - 147100*polynomial(strains)
                RMS_error = numpy.sqrt(numpy.mean(errors**2))
                # print(f'RMS error for {x}-{y} is {RMS_error}')
                RMS_errors[x][y] = RMS_error
        return RMS_errors


class ThermoPwGroupData(BaseGroupData):
    """Tabular view of ThermoPW work chains stored in AiiDA groups."""

    _PROCESS_LABELS = {'Thermo_pwBaseWorkChain', 'ThermoPwBaseWorkChain'}

    def __init__(self, groups=None):
        super().__init__(groups)
        self._data = self._flatten_data()

    @staticmethod
    def _get_structure(node):
        """Return the input structure for supported ThermoPW namespaces."""
        try:
            return node.inputs.thermo_pw.structure
        except (AttributeError, KeyError):
            try:
                return node.inputs.structure
            except (AttributeError, KeyError):
                return None

    @classmethod
    def _get_material(cls, node):
        structure = cls._get_structure(node)
        if structure is not None:
            try:
                return structure.get_formula()
            except (AttributeError, ValueError):
                pass
        return node.base.extras.get('formula', 'N/A')

    @classmethod
    def _get_source(cls, node):
        structure = cls._get_structure(node)
        extras = structure.base.extras if structure is not None else None
        if extras is not None:
            source_db = extras.get('source_db', None)
            source_id = extras.get('source_id', None)
            if source_db is not None and source_id is not None:
                return f'{source_db}-{source_id}'

        source_db = node.base.extras.get('source_db', None)
        source_id = node.base.extras.get('source_id', None)
        if source_db is not None and source_id is not None:
            return f'{source_db}-{source_id}'
        return 'N/A'

    def _flatten_data(self):
        flattened_list = []
        for group_label in self._groups:
            group = orm.load_group(group_label)
            for node in group.nodes:
                process_label = getattr(node, 'process_label', '')
                if process_label not in self._PROCESS_LABELS:
                    continue
                flattened_list.append({
                    'PK': node.pk,
                    'Material': self._get_material(node),
                    'Source': self._get_source(node),
                    'Process': process_label,
                    'Status': self.get_status_string(node),
                    'node': node,
                })
        return flattened_list

    def _get_dataframe(self, property_filter=None):
        """Build one row per ThermoPW work chain, indexed by PK."""
        import pandas as pd

        if not self._data:
            return pd.DataFrame(columns=['Material', 'Source', 'Process', 'Status'])

        data_list = self._data
        if property_filter:
            filtered_list = []
            for item in data_list:
                try:
                    analyser = ThermoPwBaseAnalyser(item['node'])
                    if getattr(analyser, property_filter, False):
                        filtered_list.append(item)
                except Exception:
                    pass
            data_list = filtered_list

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
        """Return or display the ThermoPW table.

        :param display_mode: One of ``dataframe`` (return the normal DataFrame),
            ``all`` (display every row), ``scroll`` (display a scrollable table),
            or ``interactive`` (searchable, paginated node browser).
        :param max_height: Maximum table height in pixels for scrollable modes.
        :param page_size: Initial number of rows per page in interactive mode.
        :param formula_contains: A substring or iterable of substrings that must
            occur in the material formula, matched case-insensitively.
        :param formula_match: Use ``any`` to match at least one substring or
            ``all`` to require every substring.
        :param property_filter: A string of a boolean property in ThermoPwBaseAnalyser
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

    def show_interactive(
        self,
        *,
        max_height=600,
        page_size=25,
        formula_contains=None,
        formula_match='any',
        property_filter=None,
    ):
        """Display a searchable, paginated table with selectable node details."""
        try:
            import ipywidgets as widgets
            from IPython.display import HTML, display
        except ImportError as exception:
            raise ImportError(
                'Interactive display requires IPython and ipywidgets.'
            ) from exception

        dataframe = self._filter_by_formula(
            self._get_dataframe(property_filter=property_filter),
            formula_contains=formula_contains,
            formula_match=formula_match,
        )
        if dataframe.empty:
            display(HTML('<i>No ThermoPW data available.</i>'))
            return None

        valid_page_sizes = sorted({10, 25, 50, 100, int(page_size)})
        search = widgets.Text(
            placeholder='Filter by PK, material, source, process, or status',
            description='Search:',
            layout=widgets.Layout(width='100%'),
        )
        formula_search = widgets.Text(
            placeholder='Filter material formulas, e.g. Fe',
            description='Formula:',
            layout=widgets.Layout(width='100%'),
        )
        rows_per_page = widgets.Dropdown(
            options=valid_page_sizes,
            value=int(page_size),
            description='Rows:',
        )
        previous_button = widgets.Button(description='Previous', icon='arrow-left')
        next_button = widgets.Button(description='Next', icon='arrow-right')
        page_label = widgets.HTML()
        node_selector = widgets.Dropdown(description='Node:', options=[])
        table_output = widgets.Output()
        details_output = widgets.Output()
        state = {'page': 0, 'filtered': dataframe}
        node_map = {
            int(item['PK']): item['node']
            for item in self._data
            if item.get('node') is not None
        }

        def filter_dataframe():
            filtered = dataframe
            formula_query = formula_search.value.strip()
            if formula_query:
                filtered = self._filter_by_formula(
                    filtered,
                    formula_contains=formula_query,
                )
            query = search.value.strip().lower()
            if not query:
                return filtered
            searchable = filtered.reset_index().astype(str)
            mask = searchable.apply(
                lambda column: column.str.lower().str.contains(query, regex=False)
            ).any(axis=1)
            return filtered.iloc[mask.to_numpy()]

        def render_table(*_):
            filtered = filter_dataframe()
            state['filtered'] = filtered
            size = rows_per_page.value
            page_count = max(1, (len(filtered) + size - 1) // size)
            state['page'] = min(state['page'], page_count - 1)
            start = state['page'] * size
            page = filtered.iloc[start:start + size]

            previous_button.disabled = state['page'] == 0
            next_button.disabled = state['page'] >= page_count - 1
            page_label.value = (
                f'<b>Page {state["page"] + 1} / {page_count}</b> '
                f'({len(filtered)} rows)'
            )
            node_selector.options = [
                (f'{pk}: {row["Material"]} — {row["Source"]}', int(pk))
                for pk, row in page.iterrows()
            ]

            with table_output:
                table_output.clear_output(wait=True)
                display(HTML(
                    f'<div style="max-height:{int(max_height)}px; overflow:auto;">'
                    f'{page.to_html()}</div>'
                ))

        def render_details(change):
            if change.get('name') != 'value' or change.get('new') is None:
                return
            node = node_map.get(int(change['new']))
            with details_output:
                details_output.clear_output(wait=True)
                if node is not None:
                    display(HTML(render_process_node_details(node)))

        def previous_page(_):
            state['page'] = max(0, state['page'] - 1)
            render_table()

        def next_page(_):
            state['page'] += 1
            render_table()

        def reset_page(*_):
            state['page'] = 0
            render_table()

        search.observe(reset_page, names='value')
        formula_search.observe(reset_page, names='value')
        rows_per_page.observe(reset_page, names='value')
        node_selector.observe(render_details, names='value')
        previous_button.on_click(previous_page)
        next_button.on_click(next_page)
        render_table()

        widget = widgets.VBox([
            search,
            formula_search,
            widgets.HBox([rows_per_page, previous_button, next_button, page_label]),
            table_output,
            node_selector,
            details_output,
        ])
        display(widget)
        return widget

    def dump(self, dest:Path|str,):
        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={'label': {'in': self._groups}}, tag='group')
        qb.append(orm.ProcessNode, with_group='group', filters={'attributes.process_label': 'Thermo_pwBaseWorkChain'})

        if type(dest) == str:
            dest = Path(dest)
        if not dest.exists():
            dest.mkdir(parents=True)

        for [node] in qb.all():
            try:
                analyser = ThermoPwBaseAnalyser(node)
                analyser.copy_tree(dest / str(node.pk))
            except Exception as e:
                logging.warning(f"Failed to dump node {node.pk}: {e}")
