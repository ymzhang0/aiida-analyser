from ..core.base import BaseWorkChainAnalyser
from ..core.groupdata import BaseGroupData
from .pw_base import PwBaseAnalyser
from collections import defaultdict
import logging
from ..visualization.plots import plot_bands

logger = logging.getLogger(__name__)

class PwBandsAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the PwBandsWorkChain.
    """

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
        return self._get_state_from_tree()

    def plot_bands(
        self,
        axis=None,
        seekpath_params=None,
        ylabel='Energy (eV)',
    **kwargs,
    ):
        """
        Plot the band structure.
        """
        bands = self.node.outputs.band_structure
        fermi_energy = self.node.outputs.scf_parameters.get('fermi_energy')
        plot_bands(
            bands,
            axis=axis,
            reference_energy=fermi_energy,
            seekpath_params=seekpath_params,
            ylabel=ylabel,
            **kwargs,
        )

    def show_mpl(self, y_min_lim=-2, y_max_lim=2):
        """Show the bands in matplotlib."""
        bands = self.node.outputs.band_structure
        fermi_energy = self.node.outputs.scf_parameters.get('fermi_energy')
        bands.show_mpl(y_origin = fermi_energy, y_min_lim=y_min_lim, y_max_lim=y_max_lim)

    def export(self, path, y_min_lim=-2, y_max_lim=2, overwrite=True):
        """Export the bands in matplotlib."""
        bands = self.node.outputs.band_structure
        fermi_energy = self.node.outputs.scf_parameters.get('fermi_energy')
        bands.export(
            path, 
            fileformat='mpl_pdf', 
            y_origin = fermi_energy, 
            y_min_lim=y_min_lim, 
            y_max_lim=y_max_lim,
            plot_zero_axis=True,
            overwrite=overwrite
        )


class PwBandsGroup(BaseGroupData):

    analyser_class = PwBandsAnalyser
    dataframe_columns = ('Material', 'degauss', 'kpoints_distance', 'with_soc', 'status')
    def __init__(self, groups=None):
        super().__init__(groups)
        # Data structure: Material -> Degauss -> K_Dist -> Node
        self._nested_data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    list
                )
            )
        )
        self.get_data()
        self._data = self._flatten_data()

    def get_data(self):
        for node in self.iter_group_nodes('PwBandsWorkChain'):
            try:
                extras = node.base.extras.all
                formula = self.get_node_formula(node)
                degauss = extras.get('degauss', 'unknown')
                kpoints_distance = extras.get(
                    'kpoints_distance_scf', extras.get('kpoints_distance', 'unknown')
                )
                with_soc = extras.get('with_soc', 'unknown')

                logging.info(f"Processing node<{node.pk}> for {formula}")
                self._nested_data[formula][degauss][kpoints_distance].append((node, with_soc))
            except Exception as exception:
                logging.warning(f'Node<{node.pk}> processing failed: {exception}')

    def _flatten_data(self):
        flattened_list = []

        # Iterate over the nested dictionary:
        # Formula -> Degauss -> K_Dist -> Process -> Node
        for formula, degausses in self._nested_data.items():
            for degauss, k_dists in degausses.items():
                for k_dist, nodes in k_dists.items():
                    for node, with_soc in nodes:
                        flattened_list.append({
                            'PK': node.pk,
                            'Material': formula,
                            'degauss': degauss,
                            'kpoints_distance': k_dist,
                            'with_soc': with_soc,
                            'status': self.get_status_string(node),
                            'node': node,
                        })
        return flattened_list

    @staticmethod
    def _selection(values):
        """Normalise a scalar or iterable plot filter to a set."""
        if values is None:
            return None
        if isinstance(values, (str, bytes)) or not hasattr(values, '__iter__'):
            return {values}
        return set(values)

    @staticmethod
    def _soc_label(with_soc):
        """Return a concise display label for the SOC setting."""
        if with_soc is True or with_soc == 'with SOC':
            return 'with SOC'
        if with_soc is False or with_soc == 'without SOC':
            return 'without SOC'
        return 'SOC unknown'

    def _iter_band_comparisons(self, *, formula=None, degausses=None,
                               kpoints_distances=None, with_soc=None):
        """Yield one latest successful band node per parameter combination."""
        formulas = self._selection(formula)
        allowed_degauss = self._selection(degausses)
        allowed_kpoints = self._selection(kpoints_distances)
        allowed_soc = self._selection(with_soc)

        for material in sorted(self._nested_data, key=str):
            if formulas is not None and material not in formulas:
                continue
            for degauss in sorted(self._nested_data[material], key=str):
                if allowed_degauss is not None and degauss not in allowed_degauss:
                    continue
                for kpoints_distance in sorted(self._nested_data[material][degauss], key=str):
                    if allowed_kpoints is not None and kpoints_distance not in allowed_kpoints:
                        continue
                    nodes_by_soc = {}
                    for node, soc_setting in self._nested_data[material][degauss][kpoints_distance]:
                        if not getattr(node, 'is_finished_ok', False):
                            continue
                        if allowed_soc is not None and soc_setting not in allowed_soc:
                            continue
                        previous = nodes_by_soc.get(soc_setting)
                        if previous is None or getattr(node, 'pk', -1) > getattr(previous, 'pk', -1):
                            nodes_by_soc[soc_setting] = node
                    for soc_setting, node in sorted(nodes_by_soc.items(), key=lambda item: str(item[0])):
                        yield material, degauss, kpoints_distance, soc_setting, node

    def plot_bands(self, axs=None, formula=None, kpoints_distances=None,
                   degausses=None, with_soc=None, destpath=None, **kwargs):
        """Compare finished bands for selected degauss and k-point distances.

        Every material is drawn on a separate axis.  Each line set corresponds
        to one ``(degauss, kpoints_distance, with_soc)`` combination; if the
        group contains reruns with identical settings, only the highest-PK
        finished node is shown.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        legend_fontsize = kwargs.pop('legend_fontsize', 12)
        title_fontsize = kwargs.pop('title_fontsize', 16)
        legend = kwargs.pop('legend', True)
        linewidth = kwargs.pop('lw', 1.5)
        colour_cycle = kwargs.pop('colours', plt.rcParams['axes.prop_cycle'].by_key()['color'])
        if not colour_cycle:
            raise ValueError('colours must contain at least one matplotlib colour.')
        comparisons = list(self._iter_band_comparisons(
            formula=formula,
            degausses=degausses,
            kpoints_distances=kpoints_distances,
            with_soc=with_soc,
        ))
        structures = sorted({material for material, *_ in comparisons}, key=str)

        if not structures:
            selected = f' for formula {formula!r}' if formula is not None else ''
            raise ValueError(
                f'No finished PwBandsWorkChain nodes match the requested comparison{selected}.'
            )

        created_axes = axs is None
        if axs is None:
            _, axs = plt.subplots(1, len(structures), figsize=(6 * len(structures), 5), squeeze=False)
        flat_axes = list(np.asarray(axs, dtype=object).flat)
        if len(flat_axes) < len(structures):
            raise ValueError(f'Expected at least {len(structures)} axes, received {len(flat_axes)}.')

        comparisons_by_material = {material: [] for material in structures}
        for comparison in comparisons:
            comparisons_by_material[comparison[0]].append(comparison)

        for axis, material in zip(flat_axes, structures):
            material_comparisons = comparisons_by_material[material]
            for colour_index, (_, degauss, kpoints_distance, soc_setting, node) in enumerate(material_comparisons):
                soc_label = self._soc_label(soc_setting)
                logger.info(
                    'Plotting node<%s> for %s: degauss=%s, kpoints_distance=%s, %s',
                    node.pk, material, degauss, kpoints_distance, soc_label,
                )
                PwBandsAnalyser(node).plot_bands(
                    axis=axis,
                    label=rf'$\sigma$={degauss} Ry, $|k|$={kpoints_distance} $\AA^{{-1}}$, {soc_label}',
                    color=colour_cycle[colour_index % len(colour_cycle)],
                    linestyle='-',
                    lw=linewidth,
                    **kwargs,
                )
            axis.set_title(f'${material}$', fontsize=title_fontsize)
            axis.grid(axis='y', alpha=0.2)
            if legend:
                axis.legend(loc='upper left', fontsize=legend_fontsize)

        for axis in flat_axes[1:len(structures)]:
            axis.set_ylabel('')

        if destpath and created_axes:
            plt.tight_layout()
            plt.savefig(destpath)
        return axs
