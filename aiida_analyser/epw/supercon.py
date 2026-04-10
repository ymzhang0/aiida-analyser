from pathlib import Path

from aiida import orm
import numpy
from collections import defaultdict
import warnings
from ..base import BaseWorkChainAnalyser
from .epw_base import EpwBaseWorkChainAnalyser
from ..calculators import _calculate_iso_tc, check_convergence
from ..quantumespresso.ph import check_stability_epw_bands
from ..plot import (
    plot_a2f,
    plot_eldos,
    plot_aniso_gap_function,
    plot_phdos,
    plot_iso_gap_function
)


def _iter_calcjob_trees(process_tree):
    """Yield all calcjob leaves in a ``ProcessTree`` subtree."""
    if process_tree is None:
        return
    if not process_tree.children and isinstance(process_tree.node, orm.CalcJobNode):
        yield process_tree
        return
    for child in process_tree.children.values():
        yield from _iter_calcjob_trees(child)


def _get_a2f_arraydata(workchain: orm.WorkChainNode):
    """Return whichever A2F output is available on an EPW workchain."""
    if 'a2f_data' in workchain.outputs:
        return workchain.outputs.a2f_data
    if 'a2f' in workchain.outputs:
        return workchain.outputs.a2f
    return None


class SuperConWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the EpwSuperConWorkChain.
    """

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct EpwBaseWorkChain child."""
        return self._copy_tree_for_direct_children(
            destpath,
            lambda _, child: EpwBaseWorkChainAnalyser if child.node.process_label == 'EpwBaseWorkChain' else None,
        )


    @property
    def structure(self):
        if self.node.inputs.structure is None:
            raise ValueError('structure is not found')
        else:
            return self.node.inputs.structure

    @property
    def a2f(self):
        return getattr(self.process_tree, 'a2f', None)

    @property
    def conv(self):
        conv = {}
        for node_name, node in self.process_tree.children.items():
            if node_name.startswith('conv_'):
                conv[node_name] = node
        return conv

    @property
    def iso(self):
        return getattr(self.process_tree, 'epw_final_iso', None)

    @property
    def aniso(self):
        return getattr(self.process_tree, 'epw_final_aniso', None)

    @staticmethod
    def _qfpoints_distance(node: orm.WorkChainNode):
        try:
            return node.inputs.qfpoints_distance.value
        except AttributeError:
            return None

    def _finished_conv_items(self):
        return [
            (name, tree)
            for name, tree in sorted(self.conv.items())
            if tree.node.is_finished_ok
        ]

    def _latest_a2f_workchain(self):
        if self.a2f and self.a2f.node.is_finished_ok:
            return self.a2f.node
        finished_conv_items = self._finished_conv_items()
        if finished_conv_items:
            return finished_conv_items[-1][1].node
        return None

    @property
    def outputs_parameters(self):
        from ase.spacegroup import get_spacegroup
        outputs_parameters = {}

        structure = self.structure

        outputs_parameters['Formula'] = structure.get_formula()
        sg = get_spacegroup(structure.get_ase(), symprec=1e-6)
        outputs_parameters['Space group'] = f"[{sg.no}] {sg.symbol}"

        a2f_workchain = self._latest_a2f_workchain()
        if a2f_workchain is not None:
            a2f_output_parameters = a2f_workchain.outputs.output_parameters
            outputs_parameters['w log'] = a2f_output_parameters.get('w_log')
            outputs_parameters['lambda'] = a2f_output_parameters.get('lambda')
            outputs_parameters['Allen_Dynes_Tc'] = a2f_output_parameters.get('Allen_Dynes_Tc')
        if self.iso:
            outputs_parameters['w log'] = self.iso.node.outputs.output_parameters.get('w_log')
            outputs_parameters['lambda'] = self.iso.node.outputs.output_parameters.get('lambda')
            outputs_parameters['Allen_Dynes_Tc'] = self.iso.node.outputs.output_parameters.get('Allen_Dynes_Tc')
        return outputs_parameters

    def get_source(self):
        """Get the source of the workchain."""
        source = super().get_source()
        if not source:
            try:
                source_db, source_id = self.node.inputs.structure.base.extras.get_many(('source_db', 'source_id'))
                source = f"{source_db}-{source_id}"
            except Exception:
                print('Source is not set')
                return None
        return source


    def get_state(self):
        """Get the state of the workchain."""
        subprocesses = [
            *( (name, EpwBaseWorkChainAnalyser) for name, _ in sorted(self.conv.items()) ),
            ('epw_final_iso', EpwBaseWorkChainAnalyser),
            ('epw_final_aniso', EpwBaseWorkChainAnalyser),
        ]
        return self._get_state_from_subprocesses(subprocesses)

    @property
    def a2f_results(self):
        """Get the results of the a2f workchain."""
        if self.a2f and self.a2f.node.is_finished_ok:
            return {'a2f': self.a2f.node.outputs.output_parameters}

        conv_results = {}
        for name, tree in self._finished_conv_items():
            qfpoints_distance = self._qfpoints_distance(tree.node)
            conv_results[qfpoints_distance if qfpoints_distance is not None else name] = tree.node.outputs.output_parameters
        return conv_results

    @property
    def converged_allen_dynes_Tc(self, threshold=0.1):
        """Get the results of the a2f workchain."""
        if self.conv == {}:
            print('No a2f_conv workchain found')
            return None
        else:
            Tcs = [a2f_result.get('Allen_Dynes_Tc') for a2f_result in list(self.a2f_results.values())]
            _, converged_allen_dynes_Tc = check_convergence(
                Tcs,
                threshold
            )
            return converged_allen_dynes_Tc
        
    # TODO: This function is only used temporarily before the error handler of EpwSuperconWorkChain
    #       is completed.
    @property
    def iso_results(self):
        """Get the results of the iso workchain."""
        from aiida_epw.parsers.epw import EpwParser
        results = {}
        if not self.iso:
            return results
        for calcjob_tree in _iter_calcjob_trees(self.iso):
            retrieved = getattr(calcjob_tree.node.outputs, 'retrieved', None)
            if retrieved is None or 'aiida.out' not in retrieved.list_object_names():
                continue
            parsed_stdout, _ = EpwParser.parse_stdout(retrieved.get_object_content('aiida.out'), None)
            results[calcjob_tree.name] = parsed_stdout
        
        return results

    @property
    def iso_max_eigenvalues(self):
        """Get the max eigenvalues of the iso workchain."""
        max_eigenvalues = []
        for iteration, parsed_stdout in self.iso_results.items():
            max_eigenvalues.append(parsed_stdout['max_eigenvalue'].get_array('max_eigenvalue'))
        return numpy.concatenate(max_eigenvalues, axis=1)

    # TODO: This function can't treat the case where minimal eigenvalue is larger than 1.0.
    @property
    def iso_tc(self):
        """Get the tc of the iso workchain."""
        try:
            return _calculate_iso_tc(self.iso_max_eigenvalues, allow_extrapolation=True)
        except (AttributeError, KeyError, ValueError):
            return None

    def get_aniso_remote_path(self):
        """Get the remote directory of the aniso workchain."""
        if not self.aniso:
            raise ValueError('No anisotropic EPW workchain found.')
        paths = self._get_calcjob_paths(self.aniso)
        if not paths:
            raise ValueError('No calcjob remote paths found under `epw_final_aniso`.')
        return list(paths.values())[-1]

    @property
    def processes_dict(self):
        """Get the processes dictionary."""
        return self.get_calcjob_paths()

    @property
    def retrieved(self):
        """Get the retrieved dictionary."""
        return self.get_retrieved(self.node)

    @property
    def source(self):
        """Get the source of the workchain."""
        return self.get_source()

    def set_source(self):
        """Set the source of the workchain."""
        if all(key in self.node.base.extras for key in ['source_db', 'source_id']):
            warnings.warn('Source is already set', stacklevel=2)
            return

        source = self.split_source(self.get_source())
        if source is None:
            raise ValueError('Source is not set')
        source_db, source_id = source
        self.node.base.extras.set_many({
            'source_db': source_db,
            'source_id': source_id
        })

    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message, success = super().clean_workchain(dry_run=dry_run)

        return message

    def check_convergence_allen_dynes_tc(
        self,
        convergence_threshold: float
        ) -> tuple[bool, str]:
        """Check if the convergence is reached."""
        finished_conv_items = self._finished_conv_items()
        tcs = [
            tree.node.outputs.output_parameters.get('Allen_Dynes_Tc')
            for _, tree in finished_conv_items
        ]
        tcs = [tc for tc in tcs if tc is not None]

        if len(tcs) < 2:
            return (False, 'Not enough data to check convergence.')

        is_converged, converged_value = check_convergence(tcs, convergence_threshold)
        return (
            is_converged,
            f'Checking convergence from {tcs[-2]} to {tcs[-1]} -> {converged_value}',
        )

    def check_stability_epw_bands(
        self,
        min_freq: float # meV ~ 8.1 cm-1
        ) -> tuple[bool, str]:
        """Check if the epw.x interpolated phonon band structure is stable."""
        a2f_workchain = self._latest_a2f_workchain()
        if a2f_workchain is None:
            raise ValueError('No A2F/EPW workchain found.')
        return check_stability_epw_bands(a2f_workchain, tolerance=min_freq)

    def dump_inputs(self, destpath: Path):
        self.copy_tree(destpath)

    def show_pw_bands(self):
        """Show the qe bands."""
        a2f_workchain = self._latest_a2f_workchain()
        if a2f_workchain is None or 'band_structure' not in a2f_workchain.outputs:
            raise ValueError('No PW bands available on the selected workchain.')
        bands = a2f_workchain.outputs.band_structure
        bands.show_mpl()

    def show_eldos(
        self,
        axis = None,
        **kwargs,
        ):
        a2f_workchain = self._latest_a2f_workchain()
        if a2f_workchain is None:
            raise ValueError('No a2f workchain found.')
        plot_eldos(
            dos_xydata = a2f_workchain.outputs.dos,
            fermi_energy_coarse = a2f_workchain.outputs.output_parameters.get('fermi_energy_coarse'),
            axis = axis,
            **kwargs,
        )
    def show_phdos(
        self,
        axis = None,
        **kwargs,
        ):
        a2f_workchain = self._latest_a2f_workchain()
        if a2f_workchain is None:
            raise ValueError('No a2f workchain found.')
        plot_phdos(
            phdos_xydata = a2f_workchain.outputs.phdos,
            axis = axis,
            **kwargs,
        )

    def show_a2f(self, axis=None, **kwargs):
        a2f_workchain = self._latest_a2f_workchain()
        if a2f_workchain is None:
            raise ValueError('No a2f workchain found.')
        plot_a2f(
            a2f_arraydata = _get_a2f_arraydata(a2f_workchain),
            output_parameters = a2f_workchain.outputs.output_parameters,
            axis = axis,
            **kwargs,
        )

    def show_all_a2f(self, axis=None, **kwargs):
        if self.conv == {}:
            raise ValueError('No a2f workchain found.')
        
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        integrated_a2f = kwargs.pop('integrated_a2f', False)
        for key, value in sorted(self.conv.items()):
            a2f_workchain = value.node
            if not a2f_workchain.is_finished_ok:
                continue
            fine_grid = None
            try:
                fine_grid = value.iteration_01.node.inputs.qfpoints.get_kpoints_mesh()[0]
            except AttributeError:
                pass
            label_suffix = "x".join(map(str, fine_grid)) if fine_grid is not None else str(self._qfpoints_distance(a2f_workchain) or key)
            plot_a2f(
                a2f_arraydata = _get_a2f_arraydata(a2f_workchain),
                output_parameters = a2f_workchain.outputs.output_parameters,
                axis = axis,
                integrated_a2f = integrated_a2f,
                label1 = kwargs.get('label', '') + label_suffix,
                label2 = kwargs.get('label', '') + label_suffix,
                color = colors.pop(),
                **kwargs,
            )

    def show_iso_gap_function(self, axis=None, **kwargs):
        if self.iso:
            iso_workchain = self.iso.node
        else:
            raise ValueError('No iso workchain found.')
        plot_iso_gap_function(
            iso_gap_function = iso_workchain.outputs.iso_gap_functions,
            axis = axis,
            **kwargs,
        )
    def show_aniso_gap_function(self, axis=None, **kwargs):
        if self.aniso:
            aniso_workchain = self.aniso.node   
        else:
            raise ValueError('No aniso workchain found.')
        plot_aniso_gap_function(
            aniso_gap_functions_arraydata = aniso_workchain.outputs.aniso_gap_functions,
            axis = axis,
            **kwargs,
        )
    def show_all_plots(
        self,
        ax_table,
        ax_eldos,
        ax_phdos,
        ax_a2f,
        ax_iso_gap_function,
        ax_aniso_gap_function,
        ):
        kwargs = {
            'label_fontsize': 18,
            'ticklabel_fontsize': 18,
            'legend_fontsize': 12,
        }


        if ax_eldos:
            self.show_eldos(
                axis = ax_eldos,
                **kwargs,
                )
            ax_eldos.set_ylabel("")
            ax_eldos.set_yticks([], [])
        if ax_phdos:
            self.show_phdos(
                axis = ax_phdos,
                **kwargs,
                )
            ax_phdos.set_ylabel("")
            ax_phdos.set_yticks([], [])
        if ax_a2f:
            self.show_a2f(
                axis = ax_a2f,
                show_data = False,
                **kwargs,
                )
            ax_a2f.set_ylabel("")
            ax_a2f.set_yticks([], [])

        if ax_iso_gap_function:
            self.show_iso_gap_function(
                axis = ax_iso_gap_function,
                **kwargs,
                )
            ax_iso_gap_function.set_ylabel("")
            ax_iso_gap_function.set_yticks([], [])
            
        if ax_aniso_gap_function:
            self.show_aniso_gap_function(
                axis = ax_aniso_gap_function,
                **kwargs,
                )


        ax_table.axis('off')
        data = list(self.outputs_parameters.items())

        the_table = ax_table.table(
            cellText=data,
            loc='center',
            cellLoc='left',
            )

        for _, cell in the_table.get_celld().items():
            cell.set_edgecolor('none')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(kwargs['legend_fontsize'])
        the_table.scale(1, 1.2)


class SuperConData:

    def __init__(self, groups=None):
        self._groups = [] if groups is None else groups
        # Data structure: Material -> Degauss -> K_Dist -> Q_Dist -> node
        self._data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: None
                    )
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
        if node.process_label == 'SuperConWorkChain':
            for key in ['formula', 'source_db', 'source_id', 'degauss', 'qpoints_distance']:
                if key not in extras:
                    warnings.warn(f'Extra {key} is not found in node<{node.pk}>', stacklevel=2)
            
            if not any([key in extras for key in ['kpoints_distance_scf', 'kpoints_distance']]):
                warnings.warn(
                    f'Extra kpoints_distance_scf or kpoints_distance is not found in node<{node.pk}>',
                    stacklevel=2,
                )
        return True

    def get_data(self):
        for grpname in self._groups:
            group = orm.load_group(grpname)
            for node in group.nodes:
                try:
                    extras = node.base.extras.all
                    self.check_protocol(node)
                    
                    mat_key = f"{extras['source_db']}-{extras['source_id']}-{extras['formula']}"
                    
                    # Structure: Material -> Degauss -> K_Dist -> Q_Dist -> node
                    if node.process_label == 'SuperConWorkChain':
                        if 'kpoints_distance_scf' in extras:
                            self._data[mat_key][extras['degauss']][extras['kpoints_distance_scf']][extras['qpoints_distance']] = node
                        else:
                            self._data[mat_key][extras['degauss']][extras['kpoints_distance']][extras['qpoints_distance']] = node
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
                for k_dist, q_dist_dict in k_dist_dict.items():
                    for q_dist, supercon_node in q_dist_dict.items():
                        if supercon_node:
                            flattened_list.append({
                                'Material': material,
                                'Degauss': degauss,
                                'K_Density': k_dist,
                                'Q_Density': q_dist,
                                'Status': get_status_string(supercon_node) + f" ({supercon_node.pk})",
                            })

        if not flattened_list:
            return pd.DataFrame()

        df = pd.DataFrame(flattened_list)
        
        pivot_df = df.pivot(
            index=['Degauss', 'K_Density', 'Q_Density'],
            columns='Material',
            values='Status'
        )

        pivot_df = pivot_df.fillna('')

        # Sort columns (Materials) alphabetically
        pivot_df = pivot_df.sort_index(axis=1)

        return pivot_df

    def get_allen_dynes_tc(self):
        """
        Get Allen-Dynes superconducting critical temperatures.
        """
        # Structure: Material -> Degauss -> K_Dist -> Q_Dist -> AllenDynesTc
        allen_dynes_tcs = defaultdict(
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
                    
                    if q_dist_dict:
                        for q_dist, supercon_node in q_dist_dict.items():
                            if supercon_node:
                                analyser = SuperConWorkChainAnalyser(supercon_node)
                                results = analyser.a2f_results
                                if results:
                                    allen_dynes_tcs[material][degauss][k_dist][q_dist] = results
        
        # Convert nested defaultdict to regular dict for cleaner output
        def default_to_regular(d):
            if isinstance(d, defaultdict):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d

        return default_to_regular(allen_dynes_tcs)

    def get_a2f_nodes(self):
        """
        Get a2f node.
        """
        # Structure: Material -> Degauss -> K_Dist -> Q_Dist -> AllenDynesTc
        nodes = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: None
                        )
                    )
                )
            )
        )
        for material, degauss_dict in self._data.items():
            for degauss, k_dist_dict in degauss_dict.items():
                for k_dist, q_dist_dict in k_dist_dict.items():
                    for q_dist, supercon_node in q_dist_dict.items():
                        if supercon_node:
                            analyser = SuperConWorkChainAnalyser(supercon_node)
                            for link_label, value in analyser.conv.items():
                                qf_dist = value.node.inputs.qfpoints_distance.value
                                nodes[material][degauss][k_dist][q_dist][qf_dist] = value.node
        
        # Convert nested defaultdict to regular dict for cleaner output
        def default_to_regular(d):
            if isinstance(d, defaultdict):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d

        return default_to_regular(nodes)

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
                        q_dist_dict = k_dist_dict[k_dist]
                        
                        for q_dist, node in q_dist_dict.items():
                            
                            if node is None or not node.is_finished_ok:
                                continue
                            
                            try:
                                analyser = SuperConWorkChainAnalyser(node)
                                workchains = [(None, analyser._latest_a2f_workchain())]
                                if analyser.conv:
                                    workchains = [
                                        (analyser._qfpoints_distance(tree.node), tree.node)
                                        for _, tree in analyser._finished_conv_items()
                                    ]

                                for qf_dist, workchain in workchains:
                                    if workchain is None:
                                        continue
                                    a2f_data = _get_a2f_arraydata(workchain)
                                    if a2f_data is None:
                                        continue

                                    w = a2f_data.get_array('frequency')
                                    spectral = a2f_data.get_array('a2f')
                                    label = f"D={degauss}, Q={q_dist}"
                                    if qf_dist is not None:
                                        label += f", Qf={qf_dist}"

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
                                            label=label,
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
                for k_dist, q_dist_data in k_dist_dict.items():
                    if q_dist_list:
                        q_dist_data = {k: v for k, v in q_dist_data.items() if k in q_dist_list}
                    for q_dist, epw_node in q_dist_data.items():
                        if epw_node:
                            analyser = SuperConWorkChainAnalyser(epw_node)
                            analyser.copy_tree(
                                dest / material.split("-")[-1] / f"{degauss}" / f"{k_dist}" / f"{q_dist}" / f"{epw_node.pk}"
                            )
