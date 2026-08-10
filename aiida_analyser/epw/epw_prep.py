from aiida_analyser.plot import plot_bands, plot_epw_interpolated_bands
import numpy
from collections import defaultdict
import logging
import warnings
from aiida import orm
from ..quantumespresso.ph import check_stability_matdyn_base
from ..base import BaseWorkChainAnalyser
from ..wannier.wannier90 import Wannier90Analyser
from ..quantumespresso.ph_base import PhBaseAnalyser
from ..quantumespresso.pw_base import PwBaseAnalyser
from .epw_base import EpwBaseAnalyser
from ..groupdata import BaseGroupData, render_process_node_details
from pathlib import Path
from loguru import logger

def _get_a2f_arraydata(workchain: orm.WorkChainNode):
    """Return whichever A2F output is available on an EPW workchain."""
    if 'a2f_data' in workchain.outputs:
        return workchain.outputs.a2f_data
    if 'a2f' in workchain.outputs:
        return workchain.outputs.a2f
    return None

def _safe_get_extras(node):
    extras = node.base.extras.all
    
    # Degauss
    degauss = extras.get('degauss', 'unknown')
    
    # K-point distance (can be stored as 'kpoints_distance_scf' or 'kpoints_distance')
    kpoints_distance = extras.get('kpoints_distance_scf', None)
    if kpoints_distance is None:
        kpoints_distance = extras.get('kpoints_distance', 'unknown')
        
    # Q-point distance
    qpoints_distance = extras.get('qpoints_distance', 'unknown')
    
    return degauss, kpoints_distance, qpoints_distance

class EpwPrepAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the EpwPrepWorkChain.
    """

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct child to its own analyser."""
        def _resolve(_, child):
            process_label = child.node.process_label

            if process_label == 'PwBaseWorkChain':
                return PwBaseAnalyser
            if process_label == 'PhBaseWorkChain':
                return PhBaseAnalyser
            if process_label == 'EpwBaseWorkChain':
                return EpwBaseAnalyser
            if process_label in {'Wannier90BandsWorkChain', 'Wannier90OptimizeWorkChain'}:
                return Wannier90Analyser
            return None

        return self._copy_tree_for_direct_children(destpath, _resolve)

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct child to its analyser."""
        def _resolve(_, child):
            process_label = child.node.process_label

            if process_label == 'PwBaseWorkChain':
                return PwBaseAnalyser
            if process_label == 'PhBaseWorkChain':
                return PhBaseAnalyser
            if process_label == 'EpwBaseWorkChain':
                return EpwBaseAnalyser
            if process_label in {'Wannier90BandsWorkChain', 'Wannier90OptimizeWorkChain'}:
                return Wannier90Analyser
            return None

        return self._get_calcjob_paths_for_direct_children(_resolve)

    @property
    def w90_intp(self):
        labels = self._get_child_labels(labels=('w90_bands', 'w90_intp'))
        if not labels:
            raise AttributeError('w90_bands is not found')
        return self.process_tree[labels[0]].node

    @property
    def ph_base(self):
        if 'ph_base' not in self.process_tree:
            raise AttributeError('ph_base is not found')
        else:
            return self.process_tree.ph_base.node
    @property
    def ph_base_analyser(self):
        if self.ph_base is None:
            raise AttributeError('ph_base is not found')
        else:
            return PhBaseAnalyser(self.process_tree.ph_base.node)

    @property
    def epw_base(self):
        if self.process_tree.epw_base.node is None:
            raise ValueError('epw_base is not found')
        else:
            return self.process_tree.epw_base.node

    @property
    def epw_bands(self):
        if 'epw_bands' not in self.process_tree:
            raise ValueError('epw_bands is not found')
        if self.process_tree.epw_bands.node is None:
            raise ValueError('epw_bands is not found')
        return self.process_tree.epw_bands.node

    def get_source(self):
        """Get the source of the workchain."""
        return super().get_source()


    def get_state(self):
        """Get the state of the workchain."""
        return self._get_state_from_subprocesses([
            ('w90_bands', Wannier90Analyser),
            ('ph_base', PhBaseAnalyser),
            ('epw_base', EpwBaseAnalyser),
            ('epw_bands', EpwBaseAnalyser),
        ])

    def check_stability_matdyn_base(self):
        """Get the qpoints and frequencies of the matdyn_base workchain."""
        if 'matdyn_base' not in self.process_tree:
            raise AttributeError('matdyn_base is not found')

        matdyn_base = self.process_tree.matdyn_base.node
        if not matdyn_base.is_finished_ok:
            raise ValueError('matdyn_base is not finished ok')

        return check_stability_matdyn_base(matdyn_base)

    def clean_workchain(self, exempted_states=None, dry_run=True):
        """Clean the workchain."""
        exempted_states = [] if exempted_states is None else exempted_states
        path, process_state, exit_code = self.get_state()
        message = f'Process<{self.node.pk}> is now {process_state} at {path} with exit code {exit_code}. Please check if you really want to clean this workchain.\n'
        if process_state in exempted_states:
            print(message)
            return message, False

        message, success = super().clean_workchain(dry_run=dry_run)
        return message, True

class EpwPrepConvergenceData:

    def __init__(self, groups=None):
        self._groups = [] if groups is None else groups
        # Data structure: Material -> Degauss -> K_Dist -> {'PwRelaxWorkChain': node, 'q_dist': {Q_Dist -> {'EpwPrepWorkChain': node, 'supercon': node}}}
        self._data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: {
                        'PwRelaxWorkChain': None,
                        'q_dist': defaultdict(
                            lambda: {
                                'EpwPrepWorkChain': None, 
                            }
                        )
                    }
                )
            )
        )
        self.get_data()

    @property
    def groups(self):
        return self._groups

    @property
    def data(self):
        return self._data

    @staticmethod
    def check_protocol(node):
        extras = node.base.extras.all
        if node.process_label in ['PwRelaxWorkChain']:
            for key in ['formula', 'source_db', 'source_id', 'kpoints_distance', 'degauss', ]:
                if key not in extras:
                    warnings.warn(f'Extra {key} is not found in node<{node.pk}>', stacklevel=2)
        else:
            for key in ['formula', 'source_db', 'source_id', 'kpoints_distance', 'degauss', 'qpoints_distance']:
                if key not in extras:
                    warnings.warn(f'Extra {key} is not found in node<{node.pk}>', stacklevel=2)
                
        return True

    def get_data(self):
        for grpname in self._groups:
            group = orm.load_group(grpname)
            for node in group.nodes:
                try:
                    self.check_protocol(node)
                    mat_key, degauss, kpoints_distance, qpoints_distance = _safe_get_extras(node)
                    
                    # Structure: Material -> Degauss -> K_Dist -> ...
                    if node.process_label in ['PwRelaxWorkChain']:
                        self._data[mat_key][degauss][kpoints_distance]['PwRelaxWorkChain'] = node
                    elif node.process_label in ['EpwPrepWorkChain']:
                        self._data[mat_key][degauss][kpoints_distance]['q_dist'][qpoints_distance]['EpwPrepWorkChain'] = node
                except Exception as e:
                    # Provide more context in error message
                    raise ValueError(f'Node<{node.pk}> processing failed: {e}')

    def get_table(self):
        import pandas as pd
        import numpy as np

        def get_status_string(node):
            if node is None:
                return 'N/A'

            if not node.is_terminated:
                return '⏳'
            if node.is_finished_ok:
                return '✅'
            elif node.is_failed:
                return f'❌ ({node.exit_status})'
            elif node.is_excepted:
                return '⚠️ Excepted'
            elif node.is_killed:
                return '💀 Killed'
            else:
                return f'🏃 {node.process_state.value}'

        flattened_list = []

        # Loop variables matching new dictionary structure:
        # Material -> Degauss -> K_Dist -> {'relax': ..., 'q_dist': ...}
        for material, degauss_dict in self._data.items():
            for degauss, k_dist_dict in degauss_dict.items():
                for k_dist, content in k_dist_dict.items():
                    
                    relax_node = content['PwRelaxWorkChain']
                    q_dist_data = content['q_dist']
                    relax_pk = relax_node.pk if relax_node else None
                    
                    # Base relax row
                    flattened_list.append({
                        'Material': material,
                        'Degauss': degauss,
                        'K_Density': k_dist,
                        'Q_Density': '-', 
                        'Type': 'PwRelaxWorkChain',
                        'Status': get_status_string(relax_node) + f" ({relax_pk})",
                    })
                    
                    if q_dist_data:
                        for q_dist, types_dict in q_dist_data.items():
                            epw_node = types_dict.get('EpwPrepWorkChain')
                            if epw_node:
                                flattened_list.append({
                                    'Material': material,
                                    'Degauss': degauss,
                                    'K_Density': k_dist,
                                    'Q_Density': q_dist,
                                    'Type': 'EpwPrepWorkChain',
                                    'Status': get_status_string(epw_node) + f" ({epw_node.pk})",
                                })
                            

        if not flattened_list:
            return pd.DataFrame()

        df = pd.DataFrame(flattened_list)
        
        pivot_df = df.pivot(
            index=['Degauss', 'K_Density', 'Q_Density', 'Type'],
            columns='Material',
            values='Status'
        )

        pivot_df = pivot_df.fillna('')

        # Sort columns (Materials) alphabetically
        pivot_df = pivot_df.sort_index(axis=1)

        return pivot_df

    def get_epw_bands_nodes(self):
        """
        Get epw bands node.
        """
        # Structure: Material -> Degauss -> K_Dist -> Q_Dist -> AllenDynesTc
        nodes = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: None
                    )
                )
            )
        )
        for material, degauss_dict in self._data.items():
            for degauss, k_dist_dict in degauss_dict.items():
                for k_dist, q_dist_dict in k_dist_dict.items():
                    q_dist_data = q_dist_dict['q_dist']
                    for q_dist, types_dict in q_dist_data.items():
                        node = types_dict.get('EpwPrepWorkChain')
                        if node:
                            analyser = EpwPrepAnalyser(node)
                            try:
                                nodes[material][degauss][k_dist][q_dist] = analyser.epw_bands
                            except Exception as e:
                                print(f'Node<{node.pk}> processing failed: {e}')
        
        # Convert nested defaultdict to regular dict for cleaner output
        def default_to_regular(d):
            if isinstance(d, defaultdict):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d

        return default_to_regular(nodes)

    def plot_elbands(self):
        import matplotlib.pyplot as plt

        materials = sorted(self._data.keys())
        num_materials = len(materials)
        if num_materials == 0:
            print("No data to plot.")
            return

        fig, axs = plt.subplots(2, num_materials, figsize=(24, 4*num_materials), squeeze=False)
        # axs = axs.flatten()

        for i, material in enumerate(materials):
            degauss_dict = self._data[material]
            
            cmap = plt.get_cmap('Blues')
            blues = cmap(numpy.linspace(0.3, 1.0, 5)).tolist()
            cmap = plt.get_cmap('Reds')
            reds = cmap(numpy.linspace(0.3, 1.0, 5)).tolist()
            
            for degauss, k_dist_dict in degauss_dict.items():
                for k_dist, content in k_dist_dict.items():
                    # Check q_dist for EpwPrep nodes
                    q_dist_dict = content['q_dist']
                    
                    for q_dist, types in q_dist_dict.items():
                        node = types.get('EpwPrepWorkChain')
                        if node and node.is_finished_ok:
                             # Use Analyser to get bands
                            try:
                                analyser = EpwPrepAnalyser(node)

                                plot_epw_interpolated_bands(
                                    analyser.process_tree.epw_bands.node,
                                    axes = axs[:, i], 
                                    label = f'{degauss} {k_dist} {q_dist}',
                                    color_el = blues.pop() if blues else 'blue',
                                    color_ph = reds.pop() if reds else 'red',
                                    )
                            except Exception as e:
                                print(f"Failed to plot {material} {degauss} {k_dist} {q_dist}: {e}")
                                continue # Skip if failing

            axs[0, i].set_title(material)

        # Cleanup unused axes
        for j in range(i + 1, num_materials):
            axs[:, j].axis('off')

        fig.tight_layout()


    def plot_a2f(
        self,
        axis = None,
        show_data = False,
        **kwargs,
        ):
        from matplotlib import pyplot as plt
        import numpy

        # Identify all unique parameters to set up grid
        materials = sorted(self._data.keys())
        all_k_densities = set()
        for mat in materials:
            for d in self._data[mat].values():
                 all_k_densities.update(d.keys())
                 
        k_densities = sorted(list(all_k_densities), reverse=True) # Sort descending

        num_materials = len(materials)
        num_k_dens = len(k_densities)
        
        if num_materials == 0 or num_k_dens == 0:
            print("No data to plot.")
            return

        fig, axs = plt.subplots(
            num_k_dens, num_materials, 
            figsize=(5*num_materials, 4*num_k_dens),
            squeeze=False 
        )

        cmap_name = kwargs.get('cmap', 'viridis')
        cmap = plt.get_cmap(cmap_name)
        
        for j, material in enumerate(materials):
            degauss_dict = self._data[material]
            
            for i, k_dist in enumerate(k_densities):
                ax = axs[i, j]
                
                # Check if this K_Dist exists for this material (in any degauss group)
                # Structure: Mat -> Degauss -> K_Dist
                
                found_data = False
                sorted_degausses = sorted(degauss_dict.keys())
                colors = cmap(numpy.linspace(0, 1, len(sorted_degausses)))
                
                for d_idx, degauss in enumerate(sorted_degausses):
                    k_dist_dict = degauss_dict[degauss]
                    
                    if k_dist in k_dist_dict:
                        content = k_dist_dict[k_dist]
                        q_dist_dict = content['q_dist']
                        
                        for q_dist, types in q_dist_dict.items():
                            node = types.get('EpwPrepWorkChain')
                            
                            if node is None or not node.is_finished_ok:
                                continue
                            
                            try:
                                analyser = EpwPrepAnalyser(node)
                                a2f_data = _get_a2f_arraydata(analyser.epw_bands)
                                if a2f_data is None:
                                    continue
                                    
                                w = a2f_data.get_array('frequency')
                                spectral = a2f_data.get_array('a2f') 

                                label = f"D={degauss}"
                                if len(q_dist_dict) > 1:
                                    label += f", Q={q_dist}"

                                if kwargs.get('do_a2f', True):
                                    y_val = spectral
                                    if len(spectral.shape) > 1:
                                        if spectral.shape[1] > 9:
                                            y_val = spectral[:, 9]
                                        else:
                                            y_val = spectral[:, 0]
                                    
                                    ax.plot(
                                        y_val,
                                        w,
                                        color=colors[d_idx],
                                        label=label
                                    )
                                found_data = True
                                
                            except Exception as e:
                                print(f"Error extracting/plotting for node {node.pk}: {e}")
                                continue

                if not found_data:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                else:
                    # Formatting
                    ax.set_title(f"{material}\nK={k_dist}")
                    if i == num_k_dens - 1:
                        ax.set_xlabel(r"$\alpha^2F$")
                    if j == 0:
                        ax.set_ylabel(r"$\omega$")
                    ax.legend(fontsize='x-small')

        plt.tight_layout()
        return fig, axs

    def dump(self, dest:Path, k_dist_list:list = None, degauss_list:list = None, q_dist_list:list = None):
        for material, degauss_dict in self._data.items():
            if degauss_list:
                degauss_dict = {k: v for k, v in degauss_dict.items() if k in degauss_list}
            for degauss, k_dist_dict in degauss_dict.items():
                if k_dist_list:
                    k_dist_dict = {k: v for k, v in k_dist_dict.items() if k in k_dist_list}
                for k_dist, content in k_dist_dict.items():
                    q_dist_data = content['q_dist']
                    if q_dist_list:
                        q_dist_data = {k: v for k, v in q_dist_data.items() if k in q_dist_list}
                    for q_dist, epw_dict in q_dist_data.items():
                        epw_node = epw_dict.get('EpwPrepWorkChain', None)
                        if epw_node:
                            analyser = EpwPrepAnalyser(epw_node)
                            analyser.copy_tree(
                                dest / material.split("-")[-1] / f"{degauss}" / f"{k_dist}" / f"{q_dist}" / f"{epw_node.pk}"
                            )

class EpwPrepGroup(BaseGroupData):

    analyser_class = EpwPrepAnalyser
    dataframe_columns = ('Material', 'degauss', 'kpoints_distance_scf', 'qpoints_distance', 'status')

    def __init__(self, groups=None):
        super().__init__(groups)
        # Data structure: Material -> Degauss -> K_Dist -> Q_Dist -> Node
        self._nested_data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: None
                    )
                )
            )
        )
        self._flat_nodes = []
        self.get_data()
        self._data = self._flatten_data()

    @staticmethod
    def check_protocol(node):
        extras = node.base.extras.all
        if node.process_label in ['EpwPrepWorkChain']:
            for key in ['formula', 'source_db', 'source_id', 'kpoints_distance_scf', 'degauss', 'qpoints_distance']:
                if key not in extras:
                    logger.debug(f'Extra {key} is not found in node<{node.pk}>', stacklevel=2)
                
        return True

    def get_data(self):
        for node in self.iter_group_nodes('EpwPrepWorkChain'):
            formula = self.get_node_formula(node)
            try:
                self.check_protocol(node)
                degauss, kpoints_distance, qpoints_distance = _safe_get_extras(node)
                self._nested_data[formula][degauss][kpoints_distance][qpoints_distance] = node
                self._flat_nodes.append((formula, degauss, kpoints_distance, qpoints_distance, node))
            except Exception as e:
                logging.error(f'Node<{node.pk}> processing failed: {e}')

    def _flatten_data(self):
        flattened_list = []
        for formula, degauss, kpoints_distance_scf, qpoints_distance, node in self._flat_nodes:
            flattened_list.append({
                'PK': node.pk,
                'Material': formula,
                'degauss': degauss,
                'kpoints_distance_scf': kpoints_distance_scf,
                'qpoints_distance': qpoints_distance,
                'status': self.get_status_string(node),
                'node': node,
            })

        return flattened_list

    def show_interactive(self):
        """
        Displays an interactive Jupyter table of EpwPrepWorkChain nodes.
        Clicking on a row triggers a Python callback to highlight that row
        and display the node's full nested parameters in a collapsible HTML viewer.
        """
        import ipywidgets as widgets
        from IPython.display import display
        import pandas as pd
        
        flat_data = self._data
        if not flat_data:
            print("No data available to display.")
            return

        df = pd.DataFrame(flat_data)
        df.index = df['PK'].map(int)

        node_map = {int(item['PK']): item['node'] for item in flat_data if item['node'] is not None}
        
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
        
        table_container = widgets.VBox([headers, table_body], layout=widgets.Layout(width='62%'))
        
        details_container = widgets.VBox([
            details_output
        ], layout=widgets.Layout(width='36%', margin='0 0 0 2%'))
        
        main_layout = widgets.HBox([table_container, details_container], layout=widgets.Layout(width='100%'))
        
        if not df.empty:
            select_row(df.index[0])
            
        display(main_layout)

    def get_epw_bands_nodes(self):
        """
        Get epw bands node.
        """
        # Structure: Material -> Degauss -> K_Dist -> Q_Dist -> AllenDynesTc
        nodes = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: None
                    )
                )
            )
        )
        for material, degauss_dict in self._nested_data.items():
            for degauss, k_dist_dict in degauss_dict.items():
                for k_dist, q_dist_dict in k_dist_dict.items():
                    for q_dist, node in q_dist_dict.items():
                        if node:
                            analyser = EpwPrepAnalyser(node)
                            try:
                                nodes[material][degauss][k_dist][q_dist] = analyser.epw_bands
                            except Exception as e:
                                print(f'Node<{node.pk}> processing failed: {e}')
        
        # Convert nested defaultdict to regular dict for cleaner output
        def default_to_regular(d):
            if isinstance(d, defaultdict):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d

        return default_to_regular(nodes)

    def plot_elbands(self):
        import matplotlib.pyplot as plt

        materials = sorted(self._nested_data.keys())
        num_materials = len(materials)
        if num_materials == 0:
            print("No data to plot.")
            return

        fig, axs = plt.subplots(2, num_materials, figsize=(24, 4*num_materials), squeeze=False)
        # axs = axs.flatten()

        for i, material in enumerate(materials):
            degauss_dict = self._nested_data[material]
            
            cmap = plt.get_cmap('Blues')
            blues = cmap(numpy.linspace(0.3, 1.0, 5)).tolist()
            cmap = plt.get_cmap('Reds')
            reds = cmap(numpy.linspace(0.3, 1.0, 5)).tolist()
            
            for degauss, k_dist_dict in degauss_dict.items():
                for k_dist, q_dist_dict in k_dist_dict.items():
                    for q_dist, node in q_dist_dict.items():
                        if node and node.is_finished_ok:
                            try:
                                analyser = EpwPrepAnalyser(node)

                                plot_epw_interpolated_bands(
                                    analyser.process_tree.epw_bands.node,
                                    axes = axs[:, i], 
                                    label = f'{degauss} {k_dist} {q_dist}',
                                    color_el = blues.pop() if blues else 'blue',
                                    color_ph = reds.pop() if reds else 'red',
                                    )
                            except Exception as e:
                                print(f"Failed to plot {material} {degauss} {k_dist} {q_dist}: {e}")
                                continue

            axs[0, i].set_title(material)

        for j in range(i + 1, num_materials):
            axs[:, j].axis('off')

        fig.tight_layout()

    def plot_a2f(
        self,
        axis = None,
        show_data = False,
        **kwargs,
    ):
        from matplotlib import pyplot as plt
        import numpy

        # Identify all unique parameters to set up grid
        materials = sorted(self._nested_data.keys())
        all_k_densities = set()
        for mat in materials:
            for d in self._nested_data[mat].values():
                 all_k_densities.update(d.keys())
                 
        k_densities = sorted(list(all_k_densities), reverse=True) # Sort descending

        num_materials = len(materials)
        num_k_dens = len(k_densities)
        
        if num_materials == 0 or num_k_dens == 0:
            print("No data to plot.")
            return

        fig, axs = plt.subplots(
            num_k_dens, num_materials, 
            figsize=(5*num_materials, 4*num_k_dens),
            squeeze=False 
        )

        cmap_name = kwargs.get('cmap', 'viridis')
        cmap = plt.get_cmap(cmap_name)
        
        for j, material in enumerate(materials):
            degauss_dict = self._nested_data[material]
            
            for i, k_dist in enumerate(k_densities):
                ax = axs[i, j]
                
                # Check if this K_Dist exists for this material (in any degauss group)
                # Structure: Mat -> Degauss -> K_Dist
                
                found_data = False
                sorted_degausses = sorted(degauss_dict.keys())
                colors = cmap(numpy.linspace(0, 1, len(sorted_degausses)))
                
                for d_idx, degauss in enumerate(sorted_degausses):
                    k_dist_dict = degauss_dict[degauss]
                    
                    if k_dist in k_dist_dict:
                        q_dist_dict = k_dist_dict[k_dist]
                        
                        for q_dist, node in q_dist_dict.items():
                            
                            if node is None or not node.is_finished_ok:
                                continue
                            
                            try:
                                analyser = EpwPrepAnalyser(node)
                                a2f_data = _get_a2f_arraydata(analyser.epw_bands)
                                if a2f_data is None:
                                    continue
                                    
                                w = a2f_data.get_array('frequency')
                                spectral = a2f_data.get_array('a2f') 

                                label = f"D={degauss}"
                                if len(q_dist_dict) > 1:
                                    label += f", Q={q_dist}"

                                if kwargs.get('do_a2f', True):
                                    y_val = spectral
                                    if len(spectral.shape) > 1:
                                        if spectral.shape[1] > 9:
                                            y_val = spectral[:, 9]
                                        else:
                                            y_val = spectral[:, 0]
                                    
                                    ax.plot(
                                        y_val,
                                        w,
                                        color=colors[d_idx],
                                        label=label
                                    )
                                found_data = True
                                
                            except Exception as e:
                                print(f"Error extracting/plotting for node {node.pk}: {e}")
                                continue

                if not found_data:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                else:
                    # Formatting
                    ax.set_title(f"{material}\nK={k_dist}")
                    if i == num_k_dens - 1:
                        ax.set_xlabel(r"$\alpha^2F$")
                    if j == 0:
                        ax.set_ylabel(r"$\omega$")
                    ax.legend(fontsize='x-small')

        plt.tight_layout()
        return fig, axs
