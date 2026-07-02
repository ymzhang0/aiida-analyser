from aiida import orm
from ..base import BaseWorkChainAnalyser
from .epw_calculation import EpwCalculationAnalyser
from ..groupdata import BaseGroupData
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

            # Coarse Grid
            coarse_k = None
            if 'kpoints' in node.inputs:
                try:
                    coarse_k = "x".join(map(str, node.inputs.kpoints.get_kpoints_mesh()[0]))
                except Exception:
                    try:
                        coarse_k = str(len(node.inputs.kpoints.get_kpoints()))
                    except Exception:
                        pass
            coarse_q = None
            if 'qpoints' in node.inputs:
                try:
                    coarse_q = "x".join(map(str, node.inputs.qpoints.get_kpoints_mesh()[0]))
                except Exception:
                    try:
                        coarse_q = str(len(node.inputs.qpoints.get_kpoints()))
                    except Exception:
                        pass

            # Fine Grid
            fine_k = None
            if 'kfpoints' in node.inputs:
                try:
                    fine_k = "x".join(map(str, node.inputs.kfpoints.get_kpoints_mesh()[0]))
                except Exception:
                    try:
                        fine_k = str(len(node.inputs.kfpoints.get_kpoints()))
                    except Exception:
                        pass
            elif 'kfpoints_factor' in node.inputs:
                fine_k = rf"${node.inputs.kfpoints_factor.value} \times \Delta_\mathbf{{q}}$"

            fine_q = None
            if 'qfpoints' in node.inputs:
                try:
                    fine_q = "x".join(map(str, node.inputs.qfpoints.get_kpoints_mesh()[0]))
                except Exception:
                    try:
                        fine_q = str(len(node.inputs.qfpoints.get_kpoints()))
                    except Exception:
                        pass
            elif 'qfpoints_distance' in node.inputs:
                fine_q = rf"$\Delta_\mathbf{{q}}$={node.inputs.qfpoints_distance.value}"


            # Other inputs
            restart_type = node.inputs.restart_type.value if 'restart_type' in node.inputs else '-'
            calculation_type = node.inputs.calculation_type.value if 'calculation_type' in node.inputs else '-'
            momentum_dependence = node.inputs.momentum_dependence.value if 'momentum_dependence' in node.inputs else '-'
            full_bandwidth = node.inputs.full_bandwidth.value if 'full_bandwidth' in node.inputs else '-'
            real_axis = node.inputs.real_axis.value if 'real_axis' in node.inputs else '-'
            analytical_continuation = node.inputs.analytical_continuation.value if 'analytical_continuation' in node.inputs else '-'

            try:
                remote_folder = node.outputs.remote_folder.get_remote_path()
            except Exception:
                remote_folder = 'N/A'
            # status string
            if not node.is_terminated:
                status_str = node.process_state.value
            else:
                status_str = f"{node.process_state.value} ({node.exit_status})"

            flattened_list.append({
                'PK': node.pk,
                'Material': formula,
                'Calculation type': calculation_type,
                'Restart type': restart_type,
                'Coarse k': coarse_k or '?',
                'Coarse q': coarse_q or '?',
                'Fine k': fine_k or '?',
                'Fine q': fine_q or '?',
                'Momentum dependence': momentum_dependence,
                'Fermi restriction': full_bandwidth,
                'Real axis': real_axis,
                'Analytical continuation': analytical_continuation,
                'status': status_emoji,
                'Remote path': remote_folder,
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
            import html
            
            def dict_to_html_details(data, name="Parameters"):
                if isinstance(data, dict):
                    if not data:
                        return "<i style='color: #7f8c8d;'>(empty)</i>"
                    html_str = f"<details style='margin: 4px 0 4px 12px; border: 1px solid #e0e0e0; border-radius: 4px;'>"
                    html_str += f"<summary style='font-weight: 600; cursor: pointer; padding: 6px 10px; background-color: #f8f9fa; border-bottom: 1px solid #e0e0e0;'>{html.escape(name)} ({len(data)} items)</summary>"
                    html_str += "<div style='padding: 8px 12px; background-color: #ffffff;'>"
                    for k, v in sorted(data.items()):
                        html_str += f"<div style='margin-bottom: 6px;'><b>{html.escape(k)}:</b> {dict_to_html_details(v, k)}</div>"
                    html_str += "</div></details>"
                    return html_str
                elif isinstance(data, list):
                    if not data:
                        return "<i style='color: #7f8c8d;'>(empty)</i>"
                    html_str = f"<details style='margin: 4px 0 4px 12px; border: 1px solid #e0e0e0; border-radius: 4px;'>"
                    html_str += f"<summary style='font-weight: 600; cursor: pointer; padding: 6px 10px; background-color: #f8f9fa; border-bottom: 1px solid #e0e0e0;'>{html.escape(name)} [{len(data)} items]</summary>"
                    html_str += "<div style='padding: 8px 12px; background-color: #ffffff;'>"
                    for idx, item in enumerate(data):
                        html_str += f"<div style='margin-bottom: 6px;'><b>[{idx}]:</b> {dict_to_html_details(item, f'[{idx}]')}</div>"
                    html_str += "</div></details>"
                    return html_str
                else:
                    if hasattr(data, 'pk'):
                        return f"<span style='background: #eef2f7; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em;'>Node &lt;{data.pk}&gt; ({data.process_label if hasattr(data, 'process_label') else data.__class__.__name__})</span>"
                    return f"<span style='font-family: monospace; color: #2c3e50;'>{html.escape(str(data))}</span>"

            def get_process_html(p_node):
                # Inputs: incoming links that are not call links
                inputs = {}
                for link in p_node.base.links.get_incoming():
                    if not link.link_type.value.startswith('call_'):
                        val = link.node
                        if hasattr(val, 'get_dict'):
                            try: inputs[link.link_label] = val.get_dict()
                            except Exception: inputs[link.link_label] = val
                        elif hasattr(val, 'value'):
                            inputs[link.link_label] = val.value
                        else:
                            inputs[link.link_label] = val
                
                # Outputs: outgoing links that are return links
                outputs = {}
                for link in p_node.base.links.get_outgoing():
                    if link.link_type.value == 'return':
                        val = link.node
                        if hasattr(val, 'get_dict'):
                            try: outputs[link.link_label] = val.get_dict()
                            except Exception: outputs[link.link_label] = val
                        elif hasattr(val, 'value'):
                            outputs[link.link_label] = val.value
                        else:
                            outputs[link.link_label] = val

                # Sub-processes: outgoing links that start with call_
                sub_processes = []
                for link in p_node.base.links.get_outgoing().all():
                    if link.link_type.value.startswith('call_'):
                        sub_processes.append((link.link_label, link.node))

                process_name = p_node.process_label if hasattr(p_node, 'process_label') else p_node.__class__.__name__
                exit_status_str = f" (Exit: {p_node.exit_status})" if p_node.exit_status is not None else ""
                state_emoji = "⏳" if not p_node.is_terminated else ("✅" if p_node.is_finished_ok else "❌")

                html_str = f"<details style='margin: 8px 0; border: 1px solid #dcdde1; border-radius: 6px; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.03);'>"
                html_str += f"<summary style='font-weight: bold; cursor: pointer; padding: 10px 14px; background-color: #f5f6fa; border-bottom: 1px solid #dcdde1;'>"
                html_str += f"<span style='margin-right: 8px;'>{state_emoji}</span>{process_name} &lt;{p_node.pk}&gt;{exit_status_str}"
                html_str += f"</summary>"
                html_str += "<div style='padding: 12px;'>"
                
                # Show metadata
                html_str += f"<div style='margin-bottom: 8px; font-size: 0.9em; color: #7f8c8d;'>"
                html_str += f"<b>Type:</b> {p_node.process_type or 'Unknown'}<br>"
                html_str += f"<b>State:</b> {p_node.process_state.value}<br>"
                html_str += f"</div>"

                # Show inputs
                html_str += dict_to_html_details(inputs, "Inputs")
                html_str += "<div style='height: 6px;'></div>"
                
                # Show outputs
                html_str += dict_to_html_details(outputs, "Outputs")
                
                # Show called sub-workflows/sub-calculations recursively
                if sub_processes:
                    html_str += "<div style='height: 10px;'></div>"
                    sub_html = "<div style='margin-left: 12px; border-left: 2px dashed #b2bec3; padding-left: 12px;'>"
                    sub_html += "<div style='font-size: 0.9em; font-weight: bold; color: #636e72; margin-bottom: 4px;'>Called Processes:</div>"
                    for label, sub_node in sorted(sub_processes, key=lambda x: x[1].pk):
                        sub_html += f"<div style='margin-bottom: 8px;'><i>Call link: {html.escape(label)}</i>"
                        sub_html += get_process_html(sub_node)
                        sub_html += "</div>"
                    sub_html += "</div>"
                    html_str += sub_html

                html_str += "</div></details>"
                return html_str

            return get_process_html(node)

        # Table headers styled with flexbox to match row layout
        headers = widgets.HTML(f"""
        <div style="display: flex; align-items: center; background-color: #2c3e50; color: #ffffff; font-weight: bold; padding: 8px; border-radius: 4px 4px 0 0; width: 100%; box-sizing: border-box;">
            <div style="width: 80px; text-align: center; flex-shrink: 0;">Select</div>
            <div style="width: 80px; flex-shrink: 0; padding-left: 8px;">PK</div>
            <div style="width: 120px; flex-shrink: 0;">Material</div>
            <div style="width: 130px; flex-shrink: 0;">Calc Type</div>
            <div style="width: 90px; flex-shrink: 0;">Fine k</div>
            <div style="width: 90px; flex-shrink: 0;">Fine q</div>
            <div style="width: 100px; flex-grow: 1;">Status</div>
        </div>
        """, layout=widgets.Layout(width='100%'))
        
        row_fields = {} # pk -> (btn, html_widget, row_box)
        
        def select_row(selected_pk):
            selected_pk = int(selected_pk)
            for pk, (btn, html_widget, row_box) in row_fields.items():
                row_data = df.loc[pk]
                material_val = str(row_data['Material'])
                calc_type_val = str(row_data['Calculation type'])
                fine_k_val = str(row_data['Fine k'])
                fine_q_val = str(row_data['Fine q'])
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
                    <div style="width: 120px; flex-shrink: 0; font-weight: bold;">{material_val}</div>
                    <div style="width: 130px; flex-shrink: 0;">{calc_type_val}</div>
                    <div style="width: 90px; flex-shrink: 0; font-family: monospace;">{fine_k_val}</div>
                    <div style="width: 90px; flex-shrink: 0; font-family: monospace;">{fine_q_val}</div>
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

    def dump(self, dest:Path|str,):
        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={'label': {'in': self._groups}}, tag='group')
        qb.append(orm.ProcessNode, with_group='group', filters={'attributes.process_label': 'EpwBaseWorkChain'})

        if type(dest) == str:
            dest = Path(dest)
        if not dest.exists():
            dest.mkdir(parents=True)

        for [node] in qb.all():
            try:
                analyser = EpwBaseWorkChainAnalyser(node)
                analyser.copy_tree(dest / str(node.pk))
            except Exception as e:
                logging.warning(f"Failed to dump node {node.pk}: {e}")
