from aiida import orm
from pathlib import Path
from ..core.base import BaseWorkChainAnalyser
from ..core.groupdata import BaseGroupData
from .dos_calculation import DosAnalyser
from .pw_base import PwBaseAnalyser
from .projwfc_calculation import ProjwfcAnalyser
from collections import defaultdict
import logging
import itertools

logger = logging.getLogger(__name__)

class PdosAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the PdosWorkChain.
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

        if 'scf' in self.process_tree:
            subprocesses.append(('scf', PwBaseAnalyser))

        subprocesses.extend([
            ('nscf', PwBaseAnalyser),
            ('dos', DosAnalyser),
            ('projwfc', ProjwfcAnalyser),
        ])

        return self._get_state_from_subprocesses(
            subprocesses,
            required_subprocesses=('nscf', 'dos', 'projwfc'),
        )

    def plot_pdos(self,
        axis = None,
        **kwargs,
    ):
        """Plot the pdos."""
        import numpy
        color = kwargs.pop('color', 'r')
        linestyle = kwargs.pop('linestyle', '-')
        label = kwargs.pop('label', r"phdos")

        ticklabel_fontsize = kwargs.pop('ticklabel_fontsize', 16)
        label_fontsize = kwargs.pop('label_fontsize', 16)
        scf = self.node.base.links.get_outgoing(link_label_filter='scf').first().node
        fermi_energy = scf.outputs.output_parameters.get('fermi_energy')
        dos_xydata = self.node.outputs.dos.output_dos
        E        = dos_xydata.get_array('x_array') - fermi_energy
        dos = dos_xydata.get_array('y_array_1')

        if axis is None:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots()
        else:
            ax = axis

        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.plot(
            dos,
            E,
            color=color,
            linestyle=linestyle,
            label=label)

        ax.set_xticks(
            [0, round(numpy.max(dos) * 1.05, 1)],
            [0, round(numpy.max(dos) * 1.05, 1)],
            fontsize=ticklabel_fontsize,
            )
        ax.set_yticks([], [])

        _, old_x_max = ax.get_xlim()
        ax.set_xlim(0, max(old_x_max, round(numpy.max(dos) * 1.05, 1)))
        # ax.set_xlim(0, round(numpy.max(dos) * 1.05, 1))
        ax.set_ylim(-2, 2)  
        ax.set_yticks([-2, 0, 2])
        ax.set_yticklabels([-2, 0, 2], fontsize=ticklabel_fontsize)
        ax.set_ylabel(r"Energy (eV)", fontsize=label_fontsize)


        if axis is None:
            return plt


class PdosGroup(BaseGroupData):
    analyser_class = PdosAnalyser
    process_label = 'PdosWorkChain'    
    kpoint_extra_keys = ('kpoints_distance_scf', 'kpoints_distance')
    dataframe_columns = ('Material', 'degauss', 'kpoints_distance', 'with_soc', 'status')

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
                    try:
                        with_hubbard_u = "with Hubbard U" if extras.get('with_hubbard_u') else "without Hubbard U"
                    except KeyError:
                        with_hubbard_u = 'Hubbard U unknown'
                    logging.info(f"Processing node<{node.pk}> for {formula}")


                    # Structure: StructureType -> Formula -> Plane -> Process -> Layers -> K_Dist -> Node
                    if process_label in ['PdosWorkChain']:
                        if self._data.get(formula, {}).get(degauss, {}).get(kpoints_distance) is None:
                            self._data[formula][degauss][kpoints_distance] = [(node, with_soc, with_hubbard_u)]
                        else:
                            self._data[formula][degauss][kpoints_distance].append((node, with_soc, with_hubbard_u))

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
                    for node, with_soc, with_hubbard_u in nodes:
                        flattened_list.append({
                            'Material': formula,
                            'Degauss': degauss,
                            'K_Dist': k_dist,
                            'With SOC': with_soc,
                            'With Hubbard U': with_hubbard_u,
                            'Status': self.get_status_string(node) + f' {node.pk}' if node else 'N/A',
                        })
        return flattened_list


    def plot_pdos(self, axs=None, formula=None, kpoints_distances=None, degausses=None, with_soc=None, destpath=None, **kwargs):
        """Plot GSFE curves for different k-points on a single axis."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        legend_fontsize = kwargs.pop('legend_fontsize', 12)
        title_fontsize = kwargs.pop('title_fontsize', 16)
        legend_bbox_to_anchor = kwargs.pop('legend_bbox_to_anchor', (1.0, 1.0, 0.6, 0.2))
        structures = sorted([s for s in self.data.keys() if s is not None], key=lambda x: str(x))

        if not structures:
            return None

        n_cols = len(structures)

        created_axes = axs is None
        if axs is None:
            fig, axs = plt.subplots(1, n_cols, figsize=(2 * n_cols, 4), squeeze=False)

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
                    for node, with_soc, with_hubbard_u in node_list:
                        if node and node.is_finished_ok:
                            color = next(base_colors)
                            logging.info(f"Fitting node<{node.pk}> for {formula} {degauss} {k_dist} {with_soc} {with_hubbard_u}")
                            analyser = PdosAnalyser(node)
                            analyser.plot_pdos(
                                axis=ax,
                                label=rf'$\sigma = {degauss}$ Ry, |k| = {k_dist} Å$^{{-1}}$, {with_soc}, {with_hubbard_u}',
                                color=color,
                                # marker=marker,
                                linestyle='-',
                                lw=kwargs.pop('lw', 1.5),
                                **kwargs
                        )
            ax.set_title(f"${struct}$", fontsize=title_fontsize)
        
        # axs[0, 0].legend(loc='upper left', fontsize=legend_fontsize)
        axs[0, 0].legend(loc='upper right', 
                facecolor='white', 
                fontsize=legend_fontsize,
                bbox_to_anchor=legend_bbox_to_anchor, # (x, y, width, height)
                # mode="expand",                 
                borderaxespad=0, 
                ncol=1,
                framealpha=1.0, 
                frameon=True)

        for ax in axs[0, 1:]:
            ax.set_ylabel('')


        if destpath and created_axes:
            plt.tight_layout()
            plt.savefig(destpath)
        return axs    

    def dump(self, destpath: Path):
        """Dump the pdos to a folder."""
        if isinstance(destpath, str):
            destpath = Path(destpath)
        destpath.mkdir(parents=True, exist_ok=True)
        for struct, mat_dict in self.data.items():
            for degauss, k_dist_dict in mat_dict.items():
                for k_dist, node_list in k_dist_dict.items():
                    for node, with_soc, with_hubbard_u in node_list:
                        if node and node.is_finished_ok:
                            logging.info(f"Copying node<{node.pk}> for {struct} {degauss} {k_dist} {with_soc.replace(' ', '-')} {with_hubbard_u.replace(' ', '-')}")
                            analyser = PdosAnalyser(node)
                            analyser.copy_tree(destpath / struct / str(degauss) / str(k_dist) / with_soc.replace(' ', '-') / with_hubbard_u.replace(' ', '-'))
