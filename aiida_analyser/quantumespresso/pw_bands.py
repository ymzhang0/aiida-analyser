from aiida import orm
from ..base import BaseWorkChainAnalyser
from .basegroup import BaseGroupData
from collections import defaultdict
import logging
from ..plot import plot_bands
import itertools
from pathlib import Path

logger = logging.getLogger(__name__)

class PwBandsWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the PwBandsWorkChain.
    """

    def get_source(self):
        """Get the source of the workchain."""
        source = super().get_source()
        if source is None:
            try:
                source_db, source_id = self.node.inputs.structure.base.extras.get_many(('source_db', 'source_id'))
                source = f"{source_db}-{source_id}"
            except Exception:
                print('Source is not set')
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


class PwBandsGroupData(BaseGroupData):

    def __init__(self, groups=None):
        super().__init__(groups)
        # Data structure: Material -> Degauss -> K_Dist -> Node
        self._data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: None
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

    def get_data(self):
        for grpname in self._groups:
            group = orm.load_group(grpname)
            for node in group.nodes:
                try:
                    process_label = node.process_label
                    extras = node.base.extras.all
                    formula = extras.get('formula')
                    degauss = extras.get('degauss')
                    kpoints_distance = extras.get('kpoints_distance_scf')
                    try:
                        with_soc = "with SOC" if extras.get('with_soc') else "without SOC"
                    except KeyError:
                        with_soc = 'SOC unknown'

                    logging.info(f"Processing node<{node.pk}> for {formula}")


                    # Structure: StructureType -> Formula -> Plane -> Process -> Layers -> K_Dist -> Node
                    if process_label in ['PwBandsWorkChain']:
                        if self._data.get(formula, {}).get(degauss, {}).get(kpoints_distance) is None:
                            self._data[formula][degauss][kpoints_distance] = [(node, with_soc)]
                        else:
                            self._data[formula][degauss][kpoints_distance].append((node, with_soc))

                except Exception as e:
                    logging.warning(f'Node<{node.pk}> processing failed: {e}')
                    continue

    def _flatten_data(self):
        flattened_list = []

        # Iterate over the nested dictionary:
        # Formula -> Degauss -> K_Dist -> Process -> Node
        for formula, degausses in self._data.items():
            for degauss, k_dists in degausses.items():
                for k_dist, nodes in k_dists.items():
                    for node, with_soc in nodes:
                        flattened_list.append({
                            'Material': formula,
                            'Degauss': degauss,
                            'K_Dist': k_dist,
                            'With SOC': with_soc,
                            'Status': self.get_status_string(node) + f' {node.pk}' if node else 'N/A',
                        })
        return flattened_list

    def plot_bands(self, axs=None, formula=None, kpoints_distances=None, degausses=None, with_soc=None, destpath=None, **kwargs):
        """Plot bands for different k-points on a single axis."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        legend_fontsize = kwargs.pop('legend_fontsize', 12)
        title_fontsize = kwargs.pop('title_fontsize', 16)
        structures = sorted([s for s in self.data.keys() if s is not None], key=lambda x: str(x))

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

            mat_dict = self.data[struct]

            for degauss, k_dist_dict in mat_dict.items():
                for k_dist, node_list in k_dist_dict.items():
                    for node, with_soc in node_list:
                        if node and node.is_finished_ok:
                            color = next(base_colors)
                            logging.info(f"Fitting node<{node.pk}> for {formula} {degauss} {k_dist} {with_soc}")
                            analyser = PwBandsWorkChainAnalyser(node)
                            analyser.plot_bands(
                                axis=ax,
                                label=rf'$\sigma = {degauss}$ Ry, |k| = {k_dist} Å$^{{-1}}$, {with_soc}',
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
        for struct, mat_dict in self.data.items():
            for degauss, k_dist_dict in mat_dict.items():
                for k_dist, node_list in k_dist_dict.items():
                    for node, with_soc in node_list:
                        if node and node.is_finished_ok:
                            logging.info(f"Copying node<{node.pk}> for {struct} {degauss} {k_dist} {with_soc}")
                            analyser = PwBandsWorkChainAnalyser(node)
                            analyser.copy_tree(destpath / struct / str(degauss) / str(k_dist) / str(with_soc).replace(' ', '_'))
