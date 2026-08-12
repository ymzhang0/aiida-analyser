from pathlib import Path

from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
import numpy
from collections import defaultdict
from ..core.workchains import clean_workdir
from ..core.base import BaseWorkChainAnalyser
from ..quantumespresso.pw_relax import PwRelaxAnalyser
from ..quantumespresso.pw_bands import PwBandsAnalyser
from ..epw.epw_prep import EpwPrepAnalyser
from ..epw.epw_base import EpwBaseAnalyser
from ..epw.calculators import _calculate_iso_tc, check_convergence
from ..visualization.plots import (
    plot_a2f,
    plot_eldos,
    plot_aniso_gap_function,
    plot_phdos,
    plot_iso_gap_function
)

class EpwSuperConAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the EpwSuperConWorkChain.
    """


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
    def a2f_conv(self):
        a2f_conv = []
        for node in self.node.called:
            if node.process_label == 'EpwA2fWorkChain':
                a2f_conv.append(node)
        return a2f_conv

    @property
    def iso(self):
        return getattr(self.process_tree, 'epw_final_iso', None)

    @property
    def aniso(self):
        return getattr(self.process_tree, 'epw_final_aniso', None)

    @property
    def outputs_parameters(self):
        from ase.spacegroup import get_spacegroup
        outputs_parameters = {}

        structure = self.structure

        outputs_parameters['Formula'] = structure.get_formula()
        sg = get_spacegroup(structure.get_ase(), symprec=1e-6)
        outputs_parameters['Space group'] = f"[{sg.no}] {sg.symbol}"

        if self.a2f:
            a2f_output_parameters = self.a2f.node.outputs.output_parameters
            outputs_parameters['w log'] = a2f_output_parameters.get('w_log')
            outputs_parameters['lambda'] = a2f_output_parameters.get('lambda')
            outputs_parameters['Allen_Dynes_Tc'] = a2f_output_parameters.get('Allen_Dynes_Tc')
        elif self.conv != {}:
            a2f_conv_output_parameters = self.conv[list(self.conv.keys())[-1]].node.outputs.output_parameters
            outputs_parameters['w log'] = a2f_conv_output_parameters.get('w_log')
            outputs_parameters['lambda'] = a2f_conv_output_parameters.get('lambda')
            outputs_parameters['Allen_Dynes_Tc'] = a2f_conv_output_parameters.get('Allen_Dynes_Tc')
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

        # Check subprocesses in order
        for subprocess_name, subprocess_analyser in [
            ('pw_relax', PwRelaxAnalyser),
            ('pw_bands', PwBandsAnalyser),
            ('b2w', EpwPrepAnalyser),
            ('bands', EpwBaseAnalyser),
            ('a2f', EpwBaseAnalyser),
            ('a2f_conv', EpwBaseAnalyser),
            ('iso', EpwBaseAnalyser),
            ('aniso', EpwBaseAnalyser),
            ]:
            if subprocess_name in self.process_tree:
                if not self.process_tree[subprocess_name].node.is_finished_ok:
                    analyser = subprocess_analyser(self.process_tree[subprocess_name].node)
                    path, process_state, exit_code = analyser.get_state()
                    return f'{subprocess_name}/{path}' if path != 'ROOT' else subprocess_name, process_state, exit_code

        if self.node.is_finished_ok:
            return 'ROOT', 'finished_ok', 0
        
        # If all subprocesses are finished but main node is not, use tree traversal
        # to find the actual error in the process tree
        return self._get_state_from_tree()

    @property
    def a2f_results(self):
        """Get the results of the a2f workchain."""
        if 'a2f' in self.process_tree:
            return self.process_tree.a2f.node.node.outputs.output_parameters
        else:
            conv_results = {}
            for node in self.a2f_conv:
                if node.is_finished_ok:
                    qfpoints_distance = node.inputs.a2f.qfpoints_distance.value
                    conv_results[qfpoints_distance] = node.outputs.output_parameters
            return conv_results

    @property
    def convergence_results(self):
        """Get the results of the a2f workchain."""
        results = {}
        if self.conv == {}:
            print('No a2f_conv workchain found')
            return None
        else:
            for nodes in self.conv.values():
                if nodes.node.is_finished_ok:
                    qfpoints_distance = nodes.node.inputs.qfpoints_distance.value
                    results[qfpoints_distance] = nodes.node.outputs.output_parameters
            return results

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
        for iteration, folderdata in self.retrieved['iso']['iso'].items():
            parsed_stdout, _ = EpwParser.parse_stdout(folderdata.get_object_content('aiida.out'), None)
            results[iteration] = parsed_stdout
        
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
        return self.processes_dict['aniso']['aniso']

    @property
    def processes_dict(self):
        """Get the processes dictionary."""
        return EpwSuperConAnalyser.get_processes_dict(self.node)

    @property
    def retrieved(self):
        """Get the retrieved dictionary."""
        return EpwSuperConAnalyser.get_retrieved(self.node)

    @property
    def source(self):
        """Get the source of the workchain."""
        try:
            source_db, source_id = self.get_source()
            return f'{source_db}-{source_id}'
        except (ValueError, KeyError):
            return None

    def set_source(self):
        """Set the source of the workchain."""
        if all(key in self.node.base.extras for key in ['source_db', 'source_id']):
            raise Warning('Source is already set')
        else:
            source_db, source_id = self.get_source()
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

        a2f_conv_workchains = self.a2f_conv

        try:
            prev_allen_dynes = a2f_conv_workchains[-2].outputs.output_parameters['Allen_Dynes_Tc']
            new_allen_dynes = a2f_conv_workchains[-1].outputs.output_parameters['Allen_Dynes_Tc']
            is_converged = (
                abs(prev_allen_dynes - new_allen_dynes) / new_allen_dynes
                < convergence_threshold
            )
            return (
                is_converged,
                f'Checking convergence: old {prev_allen_dynes}; new {new_allen_dynes} -> Converged = {is_converged}')
        except (AttributeError, IndexError, KeyError):
            return (False, 'Not enough data to check convergence.')

    def check_stability_epw_bands(
        self,
        min_freq: float # meV ~ 8.1 cm-1
        ) -> tuple[bool, str]:
        """Check if the epw.x interpolated phonon band structure is stable."""
        if self.epw_bands is None:
            raise ValueError('No epw bands found.')
        ph_bands = self.epw_bands[-1].outputs.ph_band_structure.get_bands()
        min_freq = numpy.min(ph_bands)
        max_freq = numpy.max(ph_bands)

        if min_freq < min_freq:
            return (False, max_freq)
        else:
            return (True, max_freq)

    def dump_inputs(self, destpath: Path):
        super()._dump_inputs(
            self.processes_dict,
            destpath=destpath,
            repository_files=['aiida.in', 'aiida.win'],
            retrieved_files=['aiida.out', 'aiida.fc', 'phonon_frequencies.dat', 'phonon_displacements.dat'],
        )

    def show_pw_bands(self):
        """Show the qe bands."""
        bands = self.pw_bands[0].outputs.band_structure
        bands.show_mpl()

    def show_eldos(
        self,
        axis = None,
        **kwargs,
        ):
        if self.a2f:
            a2f_workchain = self.a2f.node
        elif self.conv != {}:
            a2f_workchain = self.conv[list(self.conv.keys())[-1]].node
        else:
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

        if self.a2f:
            a2f_workchain = self.a2f.node
        elif self.conv != {}:
            a2f_workchain = self.conv[list(self.conv.keys())[-1]].node
        else:
            raise ValueError('No a2f workchain found.')
        plot_phdos(
            phdos_xydata = a2f_workchain.outputs.phdos,
            axis = axis,
            **kwargs,
        )

    def show_a2f(self, axis=None, **kwargs):
        if self.a2f:
            a2f_workchain = self.a2f.node
        elif self.conv != {}:
            a2f_workchain = self.conv[list(self.conv.keys())[-1]].node
        else:
            raise ValueError('No a2f workchain found.')
        plot_a2f(
            a2f_arraydata = a2f_workchain.outputs.a2f,
            output_parameters = a2f_workchain.outputs.output_parameters,
            axis = axis,
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


class EpwSuperConGroup:

    def __init__(self, groups = []):
        self._groups = groups
        # Data structure: Material -> Degauss -> K_Dist -> {'PwRelaxWorkChain': node, 'q_dist': {Q_Dist -> {'SuperConWorkChain': node}}}
        self._data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: {
                        'q_dist': defaultdict(
                            lambda: {
                                'EpwSuperConWorkChain': None
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
        if node.process_label == 'EpwSuperConWorkChain':
            for key in ['formula', 'source_db', 'source_id', 'kpoints_distance', 'degauss', 'qpoints_distance']:
                if key not in extras:
                    raise Warning(f'Extra {key} is not found in node<{node.pk}>')
        else:
            raise Warning(f'Unknown process label: {node.process_label}')
                
        return True

    def get_data(self):
        for grpname in self._groups:
            group = orm.load_group(grpname)
            for node in group.nodes:
                try:
                    extras = node.base.extras.all
                    self.check_protocol(node)
                    
                    mat_key = f"{extras['source_db']}-{extras['source_id']}-{extras['formula']}"
                    
                    # Structure: Material -> Degauss -> K_Dist -> ...
                    if node.process_label == 'EpwSuperConWorkChain':
                         self._data[mat_key][extras['degauss']][extras['kpoints_distance']]['q_dist'][extras['qpoints_distance']]['EpwSuperConWorkChain'] = node
                        
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
                    
                    q_dist_data = content['q_dist']
                    
                    if q_dist_data:
                        for q_dist, types_dict in q_dist_data.items():
                            supercon_node = types_dict.get('EpwSuperConWorkChain')
                            
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
                for k_dist, content in k_dist_dict.items():
                    
                    q_dist_data = content['q_dist']
                    
                    if q_dist_data:
                        for q_dist, types_dict in q_dist_data.items():
                            supercon_node = types_dict.get('EpwSuperConWorkChain')
                            
                            if supercon_node:
                                analyser = EpwSuperConAnalyser(supercon_node)
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
                for k_dist, content in k_dist_dict.items():
                    
                    q_dist_data = content['q_dist']
                    
                    if q_dist_data:
                        for q_dist, types_dict in q_dist_data.items():
                            supercon_node = types_dict.get('EpwSuperConWorkChain')
                            
                            if supercon_node:
                                analyser = EpwSuperConAnalyser(supercon_node)
                                for node in analyser.a2f_conv:
                                    qfpoints_distance = node.inputs.a2f.qfpoints_distance.value
                                    nodes[material][degauss][k_dist][q_dist][qfpoints_distance] = node
        
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
                        content = k_dist_dict[k_dist]
                        q_dist_dict = content['q_dist']
                        
                        for q_dist, types in q_dist_dict.items():
                            node = types.get('SuperConWorkChain')
                            
                            if node is None or not node.is_finished_ok:
                                continue
                            
                            try:
                                analyser = EpwSuperConAnalyser(node)
                                a2f_node = analyser.a2f 
                                if a2f_node and a2f_node.node:
                                    
                                    epw_node = a2f_node.node
                                    
                                    if 'a2f_data' in epw_node.outputs:
                                        a2f_data = epw_node.outputs.a2f_data
                                    elif 'a2f' in epw_node.outputs:
                                        a2f_data = epw_node.outputs.a2f
                                    else:
                                        continue
                                        
                                    w = a2f_data.get_array('frequency')
                                    spectral = a2f_data.get_array('a2f') 

                                    label = f"D={degauss}"
                                    if len(q_dist_dict) > 1:
                                        label += f", Q={q_dist}"

                                    if kwargs.get('do_a2f', True):
                                        y_val = spectral
                                        # Handle multi-column a2f if necessary
                                        if len(spectral.shape) > 1:
                                             # similar logic to epw_prep
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
