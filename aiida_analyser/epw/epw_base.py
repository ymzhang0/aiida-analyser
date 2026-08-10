from aiida import orm
from ..base import BaseWorkChainAnalyser
from .epw_calculation import EpwCalculationAnalyser
from ..groupdata import BaseGroupData, render_process_node_details
from pathlib import Path
import logging

class EpwBaseWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the EpwBaseWorkChain.
    """

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct EpwCalculation child."""
        return self._copy_tree_for_direct_children(
            destpath,
            lambda _, child: EpwCalculationAnalyser if child.node.process_label == 'EpwCalculation' else None,
        )

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct EpwCalculation child."""
        return self._get_calcjob_paths_for_direct_children(
            lambda _, child: EpwCalculationAnalyser if child.node.process_label == 'EpwCalculation' else None,
        )

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
        return self._get_state_from_tree()

    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message, success = super().clean_workchain(dry_run=dry_run)

        return message


class EpwData(BaseGroupData):
    """
    Data processor for EPW process groups.
    """

    analyser_class = EpwBaseWorkChainAnalyser
    dump_process_labels = 'EpwBaseWorkChain'

    def __init__(self, groups=None):
        super().__init__(groups)
        self._data = self._flatten_data()

    def _flatten_data(self):
        from aiida import orm
        flattened_list = []

        if not self._groups:
            return flattened_list

        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={'label': {'in': self._groups}}, tag='group')
        qb.append(orm.ProcessNode, with_group='group', filters={'attributes.process_label': 'EpwBaseWorkChain'})

        for r in qb.all():
            node = r[0]
            # Material
            formula = 'N/A'
            if 'structure' in node.inputs:
                try:
                    formula = node.inputs.structure.get_formula()
                except Exception:
                    pass
            if formula == 'N/A' and 'formula' in node.base.extras:
                formula = node.base.extras.get('formula')

            # Emojified Status
            status_emoji = self.get_status_string(node)

            flattened_list.append({
                'PK': node.pk,
                'Material': formula,
                'status': status_emoji,
                'node': node,
            })

        return flattened_list

    def get_table(self):
        import pandas as pd
        flattened_list = self._flatten_data()
        if not flattened_list:
            return pd.DataFrame()
        df = pd.DataFrame(flattened_list)
        if 'node' in df.columns:
            df = df.drop(columns=['node'])
        return df.set_index('PK') if 'PK' in df.columns else df

    def show(self):
        """Display the table as a rendered Markdown table in Jupyter Notebooks."""
        import pandas as pd
        from IPython.display import display, Markdown
        df = self.get_table()
        display(Markdown(df.to_markdown()))

    def show_interactive(self):
        """
        Displays an interactive Jupyter table of EpwBaseWorkChain nodes.
        Clicking on a row triggers a Python callback to highlight that row
        and display the node's full nested parameters in a collapsible HTML viewer.
        """
        import ipywidgets as widgets
        from IPython.display import display
        import pandas as pd
        
        df = self.get_table()
        if df.empty:
            print("No data available to display.")
            return

        # Ensure index elements are python ints
        df.index = df.index.map(int)

        node_map = {int(item['PK']): item['node'] for item in self._data if item['node'] is not None}
        
        details_output = widgets.Output()
        
        def render_node_details(node):
            return render_process_node_details(node)

        # Table headers styled with flexbox to match row layout
        headers = widgets.HTML(f"""
        <div style="display: flex; align-items: center; background-color: #2c3e50; color: #ffffff; font-weight: bold; padding: 8px; border-radius: 4px 4px 0 0; width: 100%; box-sizing: border-box;">
            <div style="width: 80px; text-align: center; flex-shrink: 0;">Select</div>
            <div style="width: 80px; flex-shrink: 0; padding-left: 8px;">PK</div>
            <div style="width: 150px; flex-shrink: 0;">Material</div>
            <div style="width: 100px; flex-grow: 1;">Status</div>
        </div>
        """, layout=widgets.Layout(width='100%'))
        
        row_fields = {} # pk -> (btn, html_widget, row_box)
        
        def select_row(selected_pk):
            selected_pk = int(selected_pk)
            for pk, (btn, html_widget, row_box) in row_fields.items():
                row_data = df.loc[pk]
                material_val = str(row_data['Material'])
                status_val = str(row_data['status'])
                
                if pk == selected_pk:
                    btn.button_style = 'success'
                    btn.icon = 'check-circle'
                    bg_color = "#e8f4fd"
                    border_style = "border-left: 4px solid #3498db; padding-left: 4px;"
                else:
                    btn.button_style = ''
                    btn.icon = 'circle-o'
                    bg_color = "transparent"
                    border_style = "padding-left: 8px;" # match selected padding to prevent alignment shifts
                
                # HTML content of columns styled to avoid overflow scrollbars
                html_widget.value = f"""
                <div style="display: flex; align-items: center; background-color: {bg_color}; {border_style} width: 100%; height: 28px; box-sizing: border-box; overflow: hidden;">
                    <div style="width: 80px; flex-shrink: 0; font-family: monospace;">{pk}</div>
                    <div style="width: 150px; flex-shrink: 0; font-weight: bold;">{material_val}</div>
                    <div style="width: 100px; flex-grow: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{status_val}</div>
                </div>
                """
                
            node = node_map.get(selected_pk)
            with details_output:
                details_output.clear_output()
                if node:
                    display(widgets.HTML(render_node_details(node)))

        rows = []
        for pk, row_data in df.iterrows():
            pk = int(pk)
            
            # Select button
            btn = widgets.Button(
                description="",
                icon="circle-o",
                tooltip=f"Select Node {pk}",
                layout=widgets.Layout(width='40px', height='26px')
            )
            
            # Button click handler
            def on_button_click(b, target_pk=pk):
                select_row(target_pk)
            btn.on_click(on_button_click)
            
            # HTML container for row columns
            html_widget = widgets.HTML(
                layout=widgets.Layout(width='100%', overflow='hidden')
            )
            
            # Row box with HBox
            row_box = widgets.HBox(
                [widgets.Box([btn], layout=widgets.Layout(width='80px', justify_content='center', flex_shrink=0)), html_widget],
                layout=widgets.Layout(width='100%', overflow='hidden', border_bottom='1px solid #ecf0f1', padding='4px 0')
            )
            
            row_fields[pk] = (btn, html_widget, row_box)
            rows.append(row_box)
            
        table_body = widgets.VBox(rows, layout=widgets.Layout(max_height='400px', overflow_y='auto', border='1px solid #ecf0f1', border_top='none', border_radius='0 0 4px 4px'))
        
        table_container = widgets.VBox([headers, table_body], layout=widgets.Layout(width='55%'))
        
        details_container = widgets.VBox([
            details_output
        ], layout=widgets.Layout(width='43%', margin='0 0 0 2%'))
        
        main_layout = widgets.HBox([table_container, details_container], layout=widgets.Layout(width='100%'))
        
        if not df.empty:
            select_row(df.index[0])
            
        display(main_layout)
