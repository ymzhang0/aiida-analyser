from aiida import orm
from ..core.base import BaseWorkChainAnalyser
from .epw_calculation import EpwAnalyser
from ..core.groupdata import BaseGroupData, render_process_node_details
from pathlib import Path
import logging
from collections import defaultdict
from aiida_analyser.visualization.plots import plot_bands, plot_epw_interpolated_bands
from aiida_analyser.visualization._axes import axis_limits as _axis_limits
from aiida_analyser.visualization._axes import plot_axes as _plot_axes
import numpy


logger = logging.getLogger(__name__)

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


def _matching_key(mapping, requested):
    """Return a key from ``mapping`` that matches a user-supplied value.

    AiiDA extras can be persisted as either strings or floats.  Matching only
    by dictionary lookup makes e.g. ``0.3`` fail to select ``'0.3'`` (or a
    float carrying a tiny serialisation error), even though the table displays
    the same value.
    """
    for key in mapping:
        if key == requested:
            return key
        try:
            if numpy.isclose(float(key), float(requested), rtol=0, atol=1e-10):
                return key
        except (TypeError, ValueError):
            continue
    return None


def _output_by_label(outputs, label):
    """Read an AiiDA output namespace without assuming its concrete type."""
    try:
        return getattr(outputs, label)
    except (AttributeError, KeyError):
        pass
    try:
        return outputs[label]
    except (KeyError, TypeError):
        return None


def _phonon_bands_output(node):
    """Find a phonon-band output on an EPW workchain or one of its children."""
    seen = set()
    pending = [node]
    while pending:
        candidate = pending.pop(0)
        identifier = getattr(candidate, 'uuid', None) or getattr(candidate, 'pk', None) or id(candidate)
        if identifier in seen:
            continue
        seen.add(identifier)

        outputs = getattr(candidate, 'outputs', None)
        if outputs is not None:
            bands = _output_by_label(outputs, 'ph_band_structure')
            if bands is None:
                output_bands = _output_by_label(outputs, 'bands')
                if output_bands is not None:
                    bands = _output_by_label(output_bands, 'ph_band_structure')
            if bands is not None:
                return bands

        try:
            pending.extend(candidate.called)
        except (AttributeError, TypeError):
            continue
    return None


class EpwBaseAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the EpwBaseWorkChain.
    """
    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct EpwCalculation child."""
        return self._copy_tree_for_direct_children(
            destpath,
            lambda _, child: EpwAnalyser if child.node.process_label == 'EpwCalculation' else None,
        )

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct EpwCalculation child."""
        return self._get_calcjob_paths_for_direct_children(
            lambda _, child: EpwAnalyser if child.node.process_label == 'EpwCalculation' else None,
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


class EpwBaseGroup(BaseGroupData):
    """
    Data processor for EPW process groups.
    """

    analyser_class = EpwBaseAnalyser
    dataframe_columns = (
        'Material',
        'degauss',
        'kpoints_distance_scf',
        'qpoints_distance',
        'status',
        'structure_PK',
        'structure_incoming',
        'node'
    )
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
        if node.process_label in ['EpwBaseWorkChain']:
            for key in ['formula', 'source_db', 'source_id', 'kpoints_distance_scf', 'degauss', 'qpoints_distance']:
                if key not in extras:
                    logger.debug(f'Extra {key} is not found in node<{node.pk}>', stacklevel=2)
                
        return True

    @staticmethod
    def _get_structure_provenance(node):
        """Return the input structure PK and its incoming provenance links.

        A structure can have more than one incoming link.  Each source is
        rendered on a separate line so the value remains readable in regular,
        paged, and HTML dataframe views.
        """
        try:
            structure = node.inputs.structure
        except Exception:
            return 'N/A', 'N/A'

        structure_pk = getattr(structure, 'pk', 'N/A')
        try:
            incoming_links = structure.base.links.get_incoming().all()
        except Exception:
            return structure_pk, 'N/A'

        sources = []
        for link in incoming_links:
            source_node = getattr(link, 'node', None)
            if source_node is None:
                continue
            source_type = (
                getattr(source_node, 'process_label', None)
                or getattr(source_node, 'node_type', None)
                or source_node.__class__.__name__
            )
            source = f'{source_type}<{getattr(source_node, "pk", "N/A")}>'
            link_label = getattr(link, 'link_label', None)
            if link_label:
                source += f' [{link_label}]'
            if source not in sources:
                sources.append(source)

        return structure_pk, '\n'.join(sources) if sources else 'N/A'


    def get_data(self):
        for node in self.iter_group_nodes('EpwBaseWorkChain'):
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
            structure_pk, structure_incoming = self._get_structure_provenance(node)
            flattened_list.append({
                'PK': node.pk,
                'Material': formula,
                'degauss': degauss,
                'kpoints_distance_scf': kpoints_distance_scf,
                'qpoints_distance': qpoints_distance,
                'status': self.get_status_string(node),
                'structure_PK': structure_pk,
                'structure_incoming': structure_incoming,
                'node': node,
            })

        return flattened_list

    def show_interactive(self):
        """
        Displays an interactive Jupyter table of EpwBaseWorkChain nodes.
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
                            nodes[material][degauss][k_dist][q_dist] = node
        
        # Convert nested defaultdict to regular dict for cleaner output
        def default_to_regular(d):
            if isinstance(d, defaultdict):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d

        return default_to_regular(nodes)

    def plot_phonon_bands_vs_degauss(
        self,
        kpoints_distance=0.15,
        qpoints_distance=0.5,
        *,
        materials=None,
        degauss_values=None,
        exclude_degauss=None,
        cmap='OrRd',
        figsize=None,
        axes=None,
        ylim=(-2, 24),
        yticks=(-2, 24),
        legend=True,
        **kwargs,
    ):
        """Plot EPW phonon bands against degauss for each material.

        The k- and q-point distances select an ``EpwPrepWorkChain`` and its
        ``epw_bands`` child. Set ``kpoints_distance=None`` to select the
        smallest available k-point distance for each degauss value.
        """
        import matplotlib.pyplot as plt

        all_nodes = self._nested_data
        if materials is None:
            selected_materials = list(all_nodes)
        elif isinstance(materials, str):
            selected_materials = [materials]
        else:
            selected_materials = list(materials)
        selected_materials = [material for material in selected_materials if material in all_nodes]
        if not selected_materials:
            raise ValueError('No EPW band nodes match the requested materials.')

        def as_list(value):
            if value is None:
                return []
            if isinstance(value, (str, bytes)):
                return [value]
            try:
                return list(value)
            except TypeError:
                return [value]

        def sort_key(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return str(value)

        allowed_degauss = set(as_list(degauss_values)) if degauss_values is not None else None
        excluded_degauss = set(as_list(exclude_degauss))
        if figsize is None:
            figsize = (max(2.5 * len(selected_materials), 2.5), 2.1)
        y_limits = _axis_limits(ylim, len(selected_materials))

        font_size = kwargs.get('font_size', 9)
        rc_params = {
            'font.size': font_size,
            'axes.titlesize': kwargs.get('title_fontsize', font_size),
            'axes.labelsize': kwargs.get('label_fontsize', font_size),
            'xtick.labelsize': kwargs.get('tick_fontsize', font_size),
            'ytick.labelsize': kwargs.get('tick_fontsize', font_size),
            'legend.fontsize': kwargs.get('legend_fontsize', font_size),
            'font.family': 'serif',
            'font.serif': ['STIXGeneral'],
        }

        with plt.rc_context(rc_params):
            fig, axes = _plot_axes(axes, len(selected_materials), plt=plt, figsize=figsize)

            for material_index, material in enumerate(selected_materials):
                axis = axes[material_index]
                material_data = all_nodes[material]
                degauss_keys = [
                    degauss for degauss in material_data
                    if (allowed_degauss is None or degauss in allowed_degauss)
                    and degauss not in excluded_degauss
                ]
                degauss_keys = sorted(degauss_keys, key=sort_key, reverse=True)
                colours = plt.get_cmap(cmap)(numpy.linspace(0.2, 0.8, len(degauss_keys)))

                for colour, degauss in zip(colours, degauss_keys):
                    kpoint_data = material_data[degauss]
                    if not kpoint_data:
                        continue
                    selected_kpoint_distance = (
                        min(kpoint_data, key=sort_key)
                        if kpoints_distance is None
                        else _matching_key(kpoint_data, kpoints_distance)
                    )
                    if selected_kpoint_distance is None:
                        continue
                    qpoint_data = kpoint_data[selected_kpoint_distance]
                    selected_qpoint_distance = _matching_key(qpoint_data, qpoints_distance)
                    if selected_qpoint_distance is None:
                        continue
                    node = qpoint_data[selected_qpoint_distance]
                    if node is None:
                        continue
                    bands_data = _phonon_bands_output(node)
                    if bands_data is None:
                        logger.warning(
                            'No ph_band_structure output found for EpwBaseWorkChain<%s> or its children.',
                            getattr(node, 'pk', 'unknown'),
                        )
                        continue

                    plot_bands(
                        bands_data,
                        axis=axis,
                        color=colour,
                        ticklabel_fontsize=kwargs.get('tick_fontsize', font_size),
                        label_fontsize=kwargs.get('label_fontsize', font_size),
                    )
                    try:
                        sigma = f'{float(degauss) * 1000:g}'
                    except (TypeError, ValueError):
                        sigma = str(degauss)
                    axis.plot([], [], label=rf'$\sigma$={sigma} mRy', color=colour)

                axis.text(
                    0.05,
                    0.9,
                    material.split('-')[-1],
                    transform=axis.transAxes,
                    bbox={'facecolor': 'white', 'edgecolor': 'none'},
                )
                axis.set_ylabel('')
                axis.set_yticks([])
                axis.set_yticklabels([])
                if y_limits[material_index] is not None:
                    axis.set_ylim(y_limits[material_index])

            if yticks is not None:
                axes[0].set_yticks(yticks)
                axes[0].set_yticklabels([str(tick) for tick in yticks])
            axes[0].set_ylabel('Frequency (meV)')
            if legend:
                axes[0].legend(
                    loc='upper center',
                    facecolor='white',
                    bbox_to_anchor=(1.35, 1.05, 0.6, 0.2),
                    borderaxespad=0,
                    ncol=kwargs.get('legend_ncol', 4),
                    framealpha=1.0,
                    frameon=True,
                )

        if not any(axis.lines for axis in axes):
            available = [
                (material, degauss, kpoint, qpoint)
                for material, degauss_data in all_nodes.items()
                for degauss, kpoint_data in degauss_data.items()
                for kpoint, qpoint_data in kpoint_data.items()
                for qpoint in qpoint_data
            ]
            raise ValueError(
                'No phonon bands could be plotted for '
                f'kpoints_distance={kpoints_distance!r}, qpoints_distance={qpoints_distance!r}. '
                f'Available (material, degauss, kpoints_distance, qpoints_distance): {available!r}'
            )

        return fig, axes
