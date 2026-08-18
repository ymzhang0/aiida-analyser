import logging

from ..core.base import BaseWorkChainAnalyser
from ..core.groupdata import DegaussKGroup
from .dos_calculation import DosAnalyser
from .projwfc_calculation import ProjwfcAnalyser
from .pw_base import PwBaseAnalyser

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


class PdosGroup(DegaussKGroup):
    """Collection of PDOS work chains across degauss and k-point scans."""

    analyser_class = PdosAnalyser
    process_label = 'PdosWorkChain'
    keep_duplicate_nodes = True
    kpoint_extra_keys = ('kpoints_distance_scf', 'kpoints_distance')
    dataframe_columns = (
        'Material', 'degauss', 'kpoints_distance', 'with_soc', 'with_hubbard_u', 'status',
    )

    @staticmethod
    def _setting_value(value):
        """Normalise the labels stored by older PdosGroup instances."""
        if value in ('with SOC', 'with Hubbard U'):
            return True
        if value in ('without SOC', 'without Hubbard U', 'SOC unknown', 'Hubbard U unknown', 'unknown'):
            return False
        return value

    @classmethod
    def _node_settings(cls, candidate):
        """Read SOC and Hubbard-U from a node or the legacy tuple format."""
        if isinstance(candidate, tuple):
            node, with_soc, with_hubbard_u = candidate
            return node, cls._setting_value(with_soc), cls._setting_value(with_hubbard_u)
        try:
            extras = candidate.base.extras.all
            return candidate, extras.get('with_soc', False), extras.get('with_hubbard_u', False)
        except (AttributeError, KeyError):
            return candidate, False, False

    def _flatten_data(self):
        flattened_list = []
        for formula, degausses in self._nested_data.items():
            for degauss, k_dists in degausses.items():
                for k_dist, nodes in k_dists.items():
                    for candidate in nodes:
                        node, with_soc, with_hubbard_u = self._node_settings(candidate)
                        flattened_list.append({
                            'PK': node.pk,
                            'Material': formula,
                            'degauss': degauss,
                            'kpoints_distance': k_dist,
                            'with_soc': with_soc,
                            'with_hubbard_u': with_hubbard_u,
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
    def _setting_label(setting, name):
        """Return a concise display label for SOC or Hubbard-U."""
        if setting is True:
            return f'with {name}'
        if setting is False:
            return f'without {name}'
        return f'{name} unknown'

    def _iter_pdos_comparisons(self, *, formula=None, degausses=None,
                               kpoints_distances=None, with_soc=None, with_hubbard_u=None):
        """Yield the latest successful PDOS node for each parameter combination."""
        formulas = self._selection(formula)
        allowed_degauss = self._selection(degausses)
        allowed_kpoints = self._selection(kpoints_distances)
        allowed_soc = self._selection(with_soc)
        allowed_hubbard_u = self._selection(with_hubbard_u)

        for material in sorted(self._nested_data, key=str):
            if formulas is not None and material not in formulas:
                continue
            for degauss in sorted(self._nested_data[material], key=str):
                if allowed_degauss is not None and degauss not in allowed_degauss:
                    continue
                for kpoints_distance in sorted(self._nested_data[material][degauss], key=str):
                    if allowed_kpoints is not None and kpoints_distance not in allowed_kpoints:
                        continue
                    nodes_by_settings = {}
                    for candidate in self._nested_data[material][degauss][kpoints_distance]:
                        node, soc_setting, hubbard_u_setting = self._node_settings(candidate)
                        if not getattr(node, 'is_finished_ok', False):
                            continue
                        if allowed_soc is not None and soc_setting not in allowed_soc:
                            continue
                        if allowed_hubbard_u is not None and hubbard_u_setting not in allowed_hubbard_u:
                            continue
                        settings = (soc_setting, hubbard_u_setting)
                        previous = nodes_by_settings.get(settings)
                        if previous is None or getattr(node, 'pk', -1) > getattr(previous, 'pk', -1):
                            nodes_by_settings[settings] = node
                    for (soc_setting, hubbard_u_setting), node in sorted(
                        nodes_by_settings.items(), key=lambda item: str(item[0])
                    ):
                        yield material, degauss, kpoints_distance, soc_setting, hubbard_u_setting, node

    def plot_pdos(self, axs=None, formula=None, kpoints_distances=None,
                  degausses=None, with_soc=None, with_hubbard_u=None, destpath=None, **kwargs):
        """Compare finished PDOS results for the selected convergence settings."""
        import matplotlib.pyplot as plt
        import numpy as np

        legend_fontsize = kwargs.pop('legend_fontsize', 12)
        title_fontsize = kwargs.pop('title_fontsize', 16)
        legend = kwargs.pop('legend', True)
        linewidth = kwargs.pop('lw', 1.5)
        colour_cycle = kwargs.pop('colours', plt.rcParams['axes.prop_cycle'].by_key()['color'])
        if not colour_cycle:
            raise ValueError('colours must contain at least one matplotlib colour.')
        comparisons = list(self._iter_pdos_comparisons(
            formula=formula,
            degausses=degausses,
            kpoints_distances=kpoints_distances,
            with_soc=with_soc,
            with_hubbard_u=with_hubbard_u,
        ))
        structures = sorted({material for material, *_ in comparisons}, key=str)
        if not structures:
            selected = f' for formula {formula!r}' if formula is not None else ''
            raise ValueError(
                f'No finished PdosWorkChain nodes match the requested comparison{selected}.'
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
            for colour_index, (_, degauss, kpoints_distance, soc_setting, hubbard_u_setting, node) in enumerate(
                comparisons_by_material[material]
            ):
                soc_label = self._setting_label(soc_setting, 'SOC')
                hubbard_u_label = self._setting_label(hubbard_u_setting, 'Hubbard U')
                logger.info(
                    'Plotting node<%s> for %s: degauss=%s, kpoints_distance=%s, %s, %s',
                    node.pk, material, degauss, kpoints_distance, soc_label, hubbard_u_label,
                )
                PdosAnalyser(node).plot_pdos(
                    axis=axis,
                    label=(
                        rf'$\sigma$={degauss} Ry, $|k|$={kpoints_distance} '
                        rf'$\AA^{{-1}}$, {soc_label}, {hubbard_u_label}'
                    ),
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
