"""Shared convergence-grid support for EPW workchain groups."""

from collections import defaultdict
import logging

import numpy

from aiida_analyser.core.groupdata import DegaussKQGroup, render_process_node_details
from aiida_analyser.visualization._axes import axis_limits as _axis_limits
from aiida_analyser.visualization._axes import plot_axes as _plot_axes
from aiida_analyser.visualization.plots import plot_bands
from aiida_analyser.visualization.style import DEFAULT_FONT_SIZE, figure_size, plot_style


logger = logging.getLogger(__name__)


def _safe_get_extras(node):
    extras = node.base.extras.all
    kpoints_distance = extras.get('kpoints_distance_scf')
    if kpoints_distance is None:
        kpoints_distance = extras.get('kpoints_distance', 'unknown')
    return (
        extras.get('degauss', 'unknown'),
        kpoints_distance,
        extras.get('qpoints_distance', 'unknown'),
    )


def _matching_key(mapping, requested):
    """Find a parameter-grid key using exact or tolerant numeric matching."""
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
    try:
        return getattr(outputs, label)
    except (AttributeError, KeyError):
        pass
    try:
        return outputs[label]
    except (KeyError, TypeError):
        return None


def phonon_bands_output(node):
    """Find ``ph_band_structure`` on a node or a registered workflow child."""
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
                nested_outputs = _output_by_label(outputs, 'bands')
                if nested_outputs is not None:
                    bands = _output_by_label(nested_outputs, 'ph_band_structure')
            if bands is not None:
                return bands

        try:
            pending.extend(candidate.called)
        except (AttributeError, TypeError):
            continue
    return None


class EpwDegaussKQGroup(DegaussKQGroup):
    """Base class for EPW convergence scans indexed by degauss, k, and q.

    Subclasses only declare their process label and, if necessary, override
    :meth:`_band_node_for_workchain` to reach the child that owns band outputs.
    """

    process_label = None
    required_extras = (
        'formula', 'source_db', 'source_id', 'kpoints_distance_scf',
        'degauss', 'qpoints_distance',
    )
    kpoint_extra_keys = ('kpoints_distance_scf', 'kpoints_distance')
    dataframe_columns = (
        'Material', 'degauss', 'kpoints_distance_scf', 'qpoints_distance',
        'status', 'structure_PK', 'structure_incoming', 'node',
    )

    @staticmethod
    def _get_structure_provenance(node):
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
            if getattr(link, 'link_label', None):
                source += f' [{link.link_label}]'
            if source not in sources:
                sources.append(source)
        return structure_pk, '\n'.join(sources) if sources else 'N/A'

    def _row_from_parameters(self, formula, degauss, kpoints_distance, qpoints_distance, node):
        row = super()._row_from_parameters(
            formula, degauss, kpoints_distance, qpoints_distance, node,
        )
        structure_pk, structure_incoming = self._get_structure_provenance(node)
        row['kpoints_distance_scf'] = row.pop('kpoints_distance')
        row['structure_PK'] = structure_pk
        row['structure_incoming'] = structure_incoming
        return row

    def show_interactive(self):
        """Display a shared, compact EPW node selector and detail view."""
        import ipywidgets as widgets
        import pandas as pd
        from IPython.display import display

        if not self._data:
            print('No data available to display.')
            return
        dataframe = pd.DataFrame(self._data).set_index('PK')
        selector = widgets.Select(
            options=[(f'{pk}: {row.Material} ({row.status})', pk) for pk, row in dataframe.iterrows()],
            description='Node:',
            layout=widgets.Layout(width='42%', height='400px'),
        )
        details = widgets.HTML(layout=widgets.Layout(width='56%', height='400px', overflow='auto'))

        def render(change=None):
            pk = selector.value
            node = dataframe.loc[pk, 'node']
            details.value = render_process_node_details(node) if node is not None else ''

        selector.observe(render, names='value')
        render()
        display(widgets.HBox([selector, details]))

    def _band_node_for_workchain(self, node):
        return node

    def get_epw_bands_nodes(self):
        nodes = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for material, degauss_data in self._nested_data.items():
            for degauss, kpoint_data in degauss_data.items():
                for kpoint, qpoint_data in kpoint_data.items():
                    for qpoint, workchain in qpoint_data.items():
                        if workchain is None:
                            continue
                        try:
                            nodes[material][degauss][kpoint][qpoint] = self._band_node_for_workchain(workchain)
                        except Exception as exception:
                            logger.warning(
                                'Could not resolve EPW bands from node<%s>: %s',
                                getattr(workchain, 'pk', 'N/A'), exception,
                            )
        return {
            material: {
                degauss: {kpoint: dict(qpoint_data) for kpoint, qpoint_data in kpoint_data.items()}
                for degauss, kpoint_data in degauss_data.items()
            }
            for material, degauss_data in nodes.items()
        }

    def plot_phonon_bands_vs_degauss(
        self, kpoints_distance=0.15, qpoints_distance=0.5, *, materials=None,
        degauss_values=None, exclude_degauss=None, cmap='OrRd', figsize=None,
        axes=None, ylim=(-2, 24), yticks=(-2, 24), legend=True, **kwargs,
    ):
        """Plot EPW phonon bands against degauss for each material."""
        import matplotlib.pyplot as plt

        all_nodes = self.get_epw_bands_nodes()
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
            figsize = figure_size(columns=len(selected_materials))
        y_limits = _axis_limits(ylim, len(selected_materials))
        font_size = kwargs.get('font_size', DEFAULT_FONT_SIZE)
        plotted = 0
        with plot_style(
            font_size=font_size,
            title_fontsize=kwargs.get('title_fontsize'),
            label_fontsize=kwargs.get('label_fontsize'),
            tick_fontsize=kwargs.get('tick_fontsize'),
            legend_fontsize=kwargs.get('legend_fontsize'),
        ):
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
                    selected_kpoint = min(kpoint_data, key=sort_key) if kpoints_distance is None else _matching_key(kpoint_data, kpoints_distance)
                    if selected_kpoint is None:
                        continue
                    qpoint_data = kpoint_data[selected_kpoint]
                    selected_qpoint = _matching_key(qpoint_data, qpoints_distance)
                    if selected_qpoint is None:
                        continue
                    node = qpoint_data[selected_qpoint]
                    bands_data = phonon_bands_output(node)
                    if bands_data is None:
                        logger.warning('No ph_band_structure output found for node<%s>.', getattr(node, 'pk', 'N/A'))
                        continue
                    plot_bands(
                        bands_data, axis=axis, color=colour,
                        ticklabel_fontsize=kwargs.get('tick_fontsize', font_size),
                        label_fontsize=kwargs.get('label_fontsize', font_size),
                    )
                    try:
                        sigma = f'{float(degauss) * 1000:g}'
                    except (TypeError, ValueError):
                        sigma = str(degauss)
                    axis.plot([], [], label=rf'$\sigma$={sigma} mRy', color=colour)
                    plotted += 1
                axis.text(0.05, 0.9, material.split('-')[-1], transform=axis.transAxes,
                          bbox={'facecolor': 'white', 'edgecolor': 'none'})
                axis.set_ylabel('')
                axis.set_yticks([])
                axis.set_yticklabels([])
                if y_limits[material_index] is not None:
                    axis.set_ylim(y_limits[material_index])
            if yticks is not None:
                axes[0].set_yticks(yticks)
                axes[0].set_yticklabels([str(tick) for tick in yticks])
            axes[0].set_ylabel('Frequency (meV)')
            if legend and plotted:
                axes[0].legend(loc='upper center', facecolor='white', bbox_to_anchor=(1.35, 1.05, 0.6, 0.2),
                               borderaxespad=0, ncol=kwargs.get('legend_ncol', 4), framealpha=1.0, frameon=True)

        if not plotted:
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

    def plot_phonon_bands_vs_kpoints(
        self, degauss=0.02, qpoints_distance=0.5, *, materials=None,
        kpoints_values=None, exclude_kpoints=None, cmap='OrRd', figsize=None,
        axes=None, ylim=(-2, 24), yticks=(-2, 24), legend=True, **kwargs,
    ):
        """Plot EPW phonon bands against k-point distance for each material.

        The degauss and q-point distances select a convergence slice, while
        every available k-point distance in that slice is overlaid.  Use
        ``kpoints_values`` and ``exclude_kpoints`` to select a subset.
        """
        import matplotlib.pyplot as plt

        all_nodes = self.get_epw_bands_nodes()
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

        def matches_any(value, requested_values):
            return any(_matching_key({value: None}, requested) is not None for requested in requested_values)

        allowed_kpoints = as_list(kpoints_values) if kpoints_values is not None else None
        excluded_kpoints = as_list(exclude_kpoints)
        if figsize is None:
            figsize = figure_size(columns=len(selected_materials))
        y_limits = _axis_limits(ylim, len(selected_materials))
        font_size = kwargs.get('font_size', DEFAULT_FONT_SIZE)
        plotted = 0
        with plot_style(
            font_size=font_size,
            title_fontsize=kwargs.get('title_fontsize'),
            label_fontsize=kwargs.get('label_fontsize'),
            tick_fontsize=kwargs.get('tick_fontsize'),
            legend_fontsize=kwargs.get('legend_fontsize'),
        ):
            fig, axes = _plot_axes(axes, len(selected_materials), plt=plt, figsize=figsize)
            for material_index, material in enumerate(selected_materials):
                axis = axes[material_index]
                material_data = all_nodes[material]
                selected_degauss = _matching_key(material_data, degauss)
                if selected_degauss is None:
                    continue
                kpoint_data = material_data[selected_degauss]
                kpoint_keys = [
                    kpoint for kpoint in kpoint_data
                    if (allowed_kpoints is None or matches_any(kpoint, allowed_kpoints))
                    and not matches_any(kpoint, excluded_kpoints)
                ]
                kpoint_keys = sorted(kpoint_keys, key=sort_key, reverse=True)
                colours = plt.get_cmap(cmap)(numpy.linspace(0.2, 0.8, len(kpoint_keys)))
                for colour, kpoint in zip(colours, kpoint_keys):
                    qpoint_data = kpoint_data[kpoint]
                    selected_qpoint = _matching_key(qpoint_data, qpoints_distance)
                    if selected_qpoint is None:
                        continue
                    node = qpoint_data[selected_qpoint]
                    bands_data = phonon_bands_output(node)
                    if bands_data is None:
                        logger.warning('No ph_band_structure output found for node<%s>.', getattr(node, 'pk', 'N/A'))
                        continue
                    plot_bands(
                        bands_data, axis=axis, color=colour,
                        ticklabel_fontsize=kwargs.get('tick_fontsize', font_size),
                        label_fontsize=kwargs.get('label_fontsize', font_size),
                    )
                    try:
                        distance = f'{float(kpoint):g}'
                    except (TypeError, ValueError):
                        distance = str(kpoint)
                    axis.plot([], [], label=rf'$d_k$={distance} $\AA^{{-1}}$', color=colour)
                    plotted += 1
                axis.text(0.05, 0.9, material.split('-')[-1], transform=axis.transAxes,
                          bbox={'facecolor': 'white', 'edgecolor': 'none'})
                axis.set_ylabel('')
                axis.set_yticks([])
                axis.set_yticklabels([])
                if y_limits[material_index] is not None:
                    axis.set_ylim(y_limits[material_index])
            if yticks is not None:
                axes[0].set_yticks(yticks)
                axes[0].set_yticklabels([str(tick) for tick in yticks])
            axes[0].set_ylabel('Frequency (meV)')
            if legend and plotted:
                axes[0].legend(loc='upper center', facecolor='white', bbox_to_anchor=(1.35, 1.05, 0.6, 0.2),
                               borderaxespad=0, ncol=kwargs.get('legend_ncol', 4), framealpha=1.0, frameon=True)

        if not plotted:
            available = [
                (material, degauss_value, kpoint, qpoint)
                for material, degauss_data in all_nodes.items()
                for degauss_value, kpoint_data in degauss_data.items()
                for kpoint, qpoint_data in kpoint_data.items()
                for qpoint in qpoint_data
            ]
            raise ValueError(
                'No phonon bands could be plotted for '
                f'degauss={degauss!r}, qpoints_distance={qpoints_distance!r}. '
                f'Available (material, degauss, kpoints_distance, qpoints_distance): {available!r}'
            )
        return fig, axes

    def plot_phonon_bands_vs_qpoints(
        self, degauss=0.02, kpoints_distance=0.15, *, materials=None,
        qpoints_values=None, exclude_qpoints=None, cmap='OrRd', figsize=None,
        axes=None, ylim=(-2, 24), yticks=(-2, 24), legend=True, **kwargs,
    ):
        """Plot EPW phonon bands against q-point distance for each material.

        The degauss and k-point distances select a convergence slice, while
        every available q-point distance in that slice is overlaid. Use
        qpoints_values and exclude_qpoints to select a subset.
        """
        import matplotlib.pyplot as plt

        all_nodes = self.get_epw_bands_nodes()
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

        def matches_any(value, requested_values):
            return any(_matching_key({value: None}, requested) is not None for requested in requested_values)

        allowed_qpoints = as_list(qpoints_values) if qpoints_values is not None else None
        excluded_qpoints = as_list(exclude_qpoints)
        if figsize is None:
            figsize = figure_size(columns=len(selected_materials))
        y_limits = _axis_limits(ylim, len(selected_materials))
        font_size = kwargs.get('font_size', DEFAULT_FONT_SIZE)
        plotted = 0
        with plot_style(
            font_size=font_size,
            title_fontsize=kwargs.get('title_fontsize'),
            label_fontsize=kwargs.get('label_fontsize'),
            tick_fontsize=kwargs.get('tick_fontsize'),
            legend_fontsize=kwargs.get('legend_fontsize'),
        ):
            fig, axes = _plot_axes(axes, len(selected_materials), plt=plt, figsize=figsize)
            for material_index, material in enumerate(selected_materials):
                axis = axes[material_index]
                material_data = all_nodes[material]
                selected_degauss = _matching_key(material_data, degauss)
                if selected_degauss is None:
                    continue
                kpoint_data = material_data[selected_degauss]
                selected_kpoint = _matching_key(kpoint_data, kpoints_distance)
                if selected_kpoint is None:
                    continue
                qpoint_data = kpoint_data[selected_kpoint]
                qpoint_keys = [
                    qpoint for qpoint in qpoint_data
                    if (allowed_qpoints is None or matches_any(qpoint, allowed_qpoints))
                    and not matches_any(qpoint, excluded_qpoints)
                ]
                qpoint_keys = sorted(qpoint_keys, key=sort_key, reverse=True)
                colours = plt.get_cmap(cmap)(numpy.linspace(0.2, 0.8, len(qpoint_keys)))
                for colour, qpoint in zip(colours, qpoint_keys):
                    node = qpoint_data[qpoint]
                    bands_data = phonon_bands_output(node)
                    if bands_data is None:
                        logger.warning('No ph_band_structure output found for node<%s>.', getattr(node, 'pk', 'N/A'))
                        continue
                    plot_bands(
                        bands_data, axis=axis, color=colour,
                        ticklabel_fontsize=kwargs.get('tick_fontsize', font_size),
                        label_fontsize=kwargs.get('label_fontsize', font_size),
                    )
                    try:
                        distance = f'{float(qpoint):g}'
                    except (TypeError, ValueError):
                        distance = str(qpoint)
                    axis.plot([], [], label=rf'$d_q$={distance} $\AA^{{-1}}$', color=colour)
                    plotted += 1
                axis.text(0.05, 0.9, material.split('-')[-1], transform=axis.transAxes,
                          bbox={'facecolor': 'white', 'edgecolor': 'none'})
                axis.set_ylabel('')
                axis.set_yticks([])
                axis.set_yticklabels([])
                if y_limits[material_index] is not None:
                    axis.set_ylim(y_limits[material_index])
            if yticks is not None:
                axes[0].set_yticks(yticks)
                axes[0].set_yticklabels([str(tick) for tick in yticks])
            axes[0].set_ylabel('Frequency (meV)')
            if legend and plotted:
                axes[0].legend(loc='upper center', facecolor='white', bbox_to_anchor=(1.35, 1.05, 0.6, 0.2),
                               borderaxespad=0, ncol=kwargs.get('legend_ncol', 4), framealpha=1.0, frameon=True)

        if not plotted:
            available = [
                (material, degauss_value, kpoint, qpoint)
                for material, degauss_data in all_nodes.items()
                for degauss_value, kpoint_data in degauss_data.items()
                for kpoint, qpoint_data in kpoint_data.items()
                for qpoint in qpoint_data
            ]
            raise ValueError(
                'No phonon bands could be plotted for '
                f'degauss={degauss!r}, kpoints_distance={kpoints_distance!r}. '
                f'Available (material, degauss, kpoints_distance, qpoints_distance): {available!r}'
            )
        return fig, axes
