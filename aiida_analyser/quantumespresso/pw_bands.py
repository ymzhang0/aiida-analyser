from ..core.base import BaseWorkChainAnalyser
from ..core.groupdata import BaseGroupData
from .pw_base import PwBaseAnalyser
from collections import defaultdict
import logging
from ..visualization.plots import plot_bands
import itertools
from pathlib import Path

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

    def plot_bands(self, axs=None, formula=None, kpoints_distances=None, degausses=None, with_soc=None, destpath=None, **kwargs):
        """Plot bands for different k-points on a single axis."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        legend_fontsize = kwargs.pop('legend_fontsize', 12)
        title_fontsize = kwargs.pop('title_fontsize', 16)
        structures = sorted([s for s in self._nested_data if s is not None], key=lambda x: str(x))

        if not structures:
            return None

        n_cols = len(structures)

        created_axes = axs is None
        if axs is None:
            fig, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)

        for i, struct in enumerate(structures):
                
            base_colors = itertools.cycle([
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ])
            markers = itertools.cycle(['o', 's', 'v', '^', '<', '>', '8', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])

            ax = axs[0, i]

            mat_dict = self._nested_data[struct]

            for degauss, k_dist_dict in mat_dict.items():
                for k_dist, node_list in k_dist_dict.items():
                    for node, with_soc in node_list:
                        if node and node.is_finished_ok:
                            color = next(base_colors)
                            soc_label = 'with SOC' if with_soc is True else 'without SOC' if with_soc is False else 'SOC unknown'
                            logging.info(f"Fitting node<{node.pk}> for {formula} {degauss} {k_dist} {soc_label}")
                            analyser = PwBandsAnalyser(node)
                            analyser.plot_bands(
                                axis=ax,
                                label=rf'$\sigma = {degauss}$ Ry, |k| = {k_dist} Å$^{{-1}}$, {soc_label}',
                                color=color,
                                # marker=marker,
                                linestyle='-',
                                lw=kwargs.pop('lw', 1.5),
                                **kwargs
                        )
            ax.set_title(f"${struct}$", fontsize=title_fontsize)
            ax.legend(loc='upper left', fontsize=legend_fontsize)
            
        for ax in axs[0, 1:]:
            ax.set_ylabel('')


        if destpath and created_axes:
            plt.tight_layout()
            plt.savefig(destpath)
        return axs

    def dump(self, destpath: Path):
        """Dump the bands to a folder."""
        for struct, mat_dict in self._nested_data.items():
            for degauss, k_dist_dict in mat_dict.items():
                for k_dist, node_list in k_dist_dict.items():
                    for node, with_soc in node_list:
                        if node and node.is_finished_ok:
                            logging.info(f"Copying node<{node.pk}> for {struct} {degauss} {k_dist} {with_soc}")
                            analyser = PwBandsAnalyser(node)
                            analyser.copy_tree(destpath / struct / str(degauss) / str(k_dist) / str(with_soc).replace(' ', '_'))
