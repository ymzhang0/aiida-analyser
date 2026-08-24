import html
from pathlib import Path

from aiida import orm
import numpy
from collections import defaultdict
import warnings
from ..core.groupdata import DegaussKQGroup, render_process_node_details
from ..core.base import BaseWorkChainAnalyser
from .epw_base import EpwBaseAnalyser
from .calculators import _calculate_iso_tc, check_convergence
from ..quantumespresso.ph import check_stability_epw_bands
from ..visualization.plots import (
    plot_a2f,
    plot_eldos,
    plot_aniso_gap_function,
    plot_phdos,
    plot_iso_gap_function
)
from ..visualization._axes import axis_limits as _axis_limits
from ..visualization._axes import plot_axes as _plot_axes
from ..visualization.convergence import configure_kpoint_distance_axis
from ..visualization.style import DEFAULT_FONT_SIZE, figure_size, plot_style, styled_plot


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
    outputs = getattr(workchain, 'outputs', None)
    if outputs is None:
        return None
    for attr in ('a2f_data', 'a2f'):
        if hasattr(outputs, attr):
            return getattr(outputs, attr)
        try:
            if attr in outputs:
                return outputs[attr]
        except (TypeError, KeyError):
            pass
    return None


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


class SuperConAnalyser(BaseWorkChainAnalyser):
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
            *( (name, EpwBaseAnalyser) for name, _ in sorted(self.conv.items()) ),
            ('epw_final_iso', EpwBaseAnalyser),
            ('epw_final_aniso', EpwBaseAnalyser),
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
        paths = EpwBaseAnalyser(self.aniso.node).get_calcjob_paths()
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

    @staticmethod
    def _set_plot_defaults(**kwargs):
        import matplotlib as mpl
        fontsize = kwargs.get('font_size', kwargs.get('label_fontsize', DEFAULT_FONT_SIZE))
        mpl.rcParams.update({
            'font.size': fontsize,
            'axes.labelsize': kwargs.get('label_fontsize', fontsize),
            'xtick.labelsize': kwargs.get('ticklabel_fontsize', fontsize),
            'ytick.labelsize': kwargs.get('ticklabel_fontsize', fontsize),
            'legend.fontsize': kwargs.get('legend_fontsize', fontsize),
        })

    @styled_plot
    def show_pw_bands(self):
        """Show the qe bands."""
        self._set_plot_defaults()
        a2f_workchain = self._latest_a2f_workchain()
        if a2f_workchain is None or 'band_structure' not in a2f_workchain.outputs:
            raise ValueError('No PW bands available on the selected workchain.')
        bands = a2f_workchain.outputs.band_structure
        bands.show_mpl()

    @styled_plot
    def show_eldos(
        self,
        axis = None,
        **kwargs,
        ):
        self._set_plot_defaults(**kwargs)
        a2f_workchain = self._latest_a2f_workchain()
        if a2f_workchain is None:
            raise ValueError('No a2f workchain found.')
        plot_eldos(
            dos_xydata = a2f_workchain.outputs.dos,
            fermi_energy_coarse = a2f_workchain.outputs.output_parameters.get('fermi_energy_coarse'),
            axis = axis,
            **kwargs,
        )
    @styled_plot
    def show_phdos(
        self,
        axis = None,
        **kwargs,
        ):
        self._set_plot_defaults(**kwargs)
        a2f_workchain = self._latest_a2f_workchain()
        if a2f_workchain is None:
            raise ValueError('No a2f workchain found.')
        plot_phdos(
            phdos_xydata = a2f_workchain.outputs.phdos,
            axis = axis,
            **kwargs,
        )

    @styled_plot
    def show_a2f(self, axis=None, **kwargs):
        self._set_plot_defaults(**kwargs)
        a2f_workchain = self._latest_a2f_workchain()
        if a2f_workchain is None:
            raise ValueError('No a2f workchain found.')
        plot_a2f(
            a2f_arraydata = _get_a2f_arraydata(a2f_workchain),
            output_parameters = a2f_workchain.outputs.output_parameters,
            axis = axis,
            **kwargs,
        )

    @styled_plot
    def show_all_a2f(self, axis=None, **kwargs):
        self._set_plot_defaults(**kwargs)
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

    @styled_plot
    def show_iso_gap_function(self, axis=None, **kwargs):
        self._set_plot_defaults(**kwargs)
        if self.iso:
            iso_workchain = self.iso.node
        else:
            raise ValueError('No iso workchain found.')
        plot_iso_gap_function(
            iso_gap_function = iso_workchain.outputs.iso_gap_functions,
            axis = axis,
            **kwargs,
        )
    @styled_plot
    def show_aniso_gap_function(self, axis=None, **kwargs):
        self._set_plot_defaults(**kwargs)
        if self.aniso:
            aniso_workchain = self.aniso.node   
        else:
            raise ValueError('No aniso workchain found.')
        plot_aniso_gap_function(
            aniso_gap_functions_arraydata = aniso_workchain.outputs.aniso_gap_functions,
            axis = axis,
            **kwargs,
        )
    @styled_plot
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
            'label_fontsize': DEFAULT_FONT_SIZE,
            'ticklabel_fontsize': DEFAULT_FONT_SIZE,
            'legend_fontsize': DEFAULT_FONT_SIZE,
        }
        self._set_plot_defaults(**kwargs)


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


    def inspect_aiida_node_details(pk_value):
        """
        Retrieves and displays deep system-level details of a specific AiiDA node,
        including its repository path on the disk and its immediate descendants.

        Args:
            pk_value (int): The primary key of the AiiDA node to inspect.
        """
        from aiida import orm
        import json

        try:
            node = orm.load_node(pk_value)
        except Exception as e:
            print(f"Error loading node {pk_value}: {e}")
            return

        print(f"--- Deep Inspection for Node PK: {node.pk} ({node.process_label}) ---")
        print(f"Status: {node.process_state.value if node.is_terminated else 'Running'}")
        
        # 1. Hardware & File System Location
        repo_folder = node.repository._repository_folder
        print(f"\n[Storage Location]")
        print(f"Raw Directory Path: {repo_folder.abspath}")
        
        # List files available in the repository
        repo_files = node.repository.list_object_names()
        print(f"Files in Repository: {', '.join(repo_files)}")

        # 2. Descendant Nodes (Children calculations or outputs)
        print(f"\n[Descendant Nodes]")
        outgoing = node.get_outgoing().all()
        if not outgoing:
            print("No outgoing child nodes found.")
        else:
            for link in outgoing:
                child = link.node
                print(f" -> {link.link_label} | PK: {child.pk} | Type: {child.__class__.__name__}")

        # 3. Execution Information (if it's a calculation job)
        if isinstance(node, orm.CalcJobNode):
            print(f"\n[Cluster Execution Details]")
            print(f"Computer: {node.computer.label if node.computer else 'Local'}")
            print(f"Job ID (Scheduler): {node.get_job_id()}")
            
class SuperConGroup(DegaussKQGroup):

    analyser_class = SuperConAnalyser
    process_label = 'SuperConWorkChain'
    required_extras = ('formula', 'source_db', 'source_id', 'degauss', 'qpoints_distance')
    kpoint_extra_keys = ('kpoints_distance_scf', 'kpoints_distance')
    dataframe_columns = (
        'Material',
        'degauss',
        'kpoints_distance',
        'qpoints_distance',
        'parent_epw',
        'status',
    )

    @staticmethod
    def _extract_degauss_k_q(node):
        extras = node.base.extras.all
        degauss = extras.get('degauss', '-')
        if 'kpoints_distance_scf' in extras:
            k_dist = extras.get('kpoints_distance_scf', '-')
        else:
            k_dist = extras.get('kpoints_distance', '-')
        q_dist = extras.get('qpoints_distance', '-')
        return degauss, k_dist, q_dist

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

    def _material_label(self, node, extras=None):
        """Return the historical source-qualified material label when available."""
        try:
            extras = node.base.extras.all if extras is None else extras
            return f"{extras['source_db']}-{extras['source_id']}-{extras['formula']}"
        except (AttributeError, KeyError, TypeError):
            return self.get_node_formula(node)

    @staticmethod
    def _parent_epw_pk(node):
        """Return the creator PK of ``parent_folder_epw`` when it is present."""
        try:
            parent_epw_node = node.inputs.parent_folder_epw.creator
            return parent_epw_node.pk if parent_epw_node is not None else None
        except (AttributeError, KeyError):
            return None

    def _row_from_parameters(self, formula, degauss, kpoints_distance, qpoints_distance, node):
        row = super()._row_from_parameters(
            formula, degauss, kpoints_distance, qpoints_distance, node,
        )
        row['parent_epw'] = self._parent_epw_pk(node)
        return row

    def get_parallel_plot(self, target_metric='tc'):
        """
        Generates an interactive Plotly Parallel Coordinates plot for the stored nodes.
        Filters out unfinished or failed calculations.
        """
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        import warnings

        flattened_list = []
        
        materials = sorted(list(set(item['Material'] for item in self._data)))
        if not materials:
            return None
            
        material_to_id = {mat: i for i, mat in enumerate(materials)}

        for item in self._data:
            supercon_node = item['node']
            if supercon_node and supercon_node.is_finished_ok:
                try:
                    result_dict = supercon_node.outputs.output_parameters.get_dict()
                    metric_value = result_dict.get(target_metric, 0.0)
                except Exception as e:
                    warnings.warn(f"Failed to extract metric from node<{supercon_node.pk}>: {e}", stacklevel=2)
                    continue
                
                degauss, k_dist, q_dist = self._extract_degauss_k_q(supercon_node)
                
                flattened_list.append({
                    'Material': item['Material'],
                    'Material_ID': material_to_id[item['Material']],
                    'Degauss': float(degauss) if degauss != '-' else 0.0,
                    'K_Density': float(k_dist) if k_dist != '-' else 0.0,
                    'Q_Density': float(q_dist) if q_dist != '-' else 0.0,
                    'Result_Value': float(metric_value),
                    'PK': supercon_node.pk
                })

        if not flattened_list:
            warnings.warn("No successfully finished nodes with output metrics found.", stacklevel=2)
            return None

        df = pd.DataFrame(flattened_list)

        fig = px.parallel_coordinates(
            df,
            dimensions=['Material_ID', 'Degauss', 'K_Density', 'Q_Density', 'Result_Value'],
            color='Result_Value',
            color_continuous_scale=px.colors.diverging.Tealrose,
            title="Superconducting Parameters Convergence Space"
        )

        fig.data[0].dimensions[0].update(
            tickvals=list(material_to_id.values()),
            ticktext=list(material_to_id.keys()),
            label='Material'
        )
        
        fig.data[0].dimensions[1].update(label='Degauss (Ry)')
        fig.data[0].dimensions[2].update(label='K-Point Dist')
        fig.data[0].dimensions[3].update(label='Q-Point Dist')
        fig.data[0].dimensions[4].update(label=f'Target Metric ({target_metric})')

        return fig

    def get_hiplot_experiment(self, target_metric='tc'):
        """
        Generates a HiPlot interactive experiment for the stored AiiDA nodes.
        """
        import hiplot as hip
        import warnings

        flattened_list = []

        for item in self._data:
            supercon_node = item['node']
            if supercon_node and supercon_node.is_finished_ok:
                try:
                    result_dict = supercon_node.outputs.output_parameters.get_dict()
                    metric_value = result_dict.get(target_metric, 0.0)
                except Exception as e:
                    warnings.warn(f"Extraction failed for node<{supercon_node.pk}>: {e}", stacklevel=2)
                    continue
                
                degauss, k_dist, q_dist = self._extract_degauss_k_q(supercon_node)
                
                flattened_list.append({
                    'Material': item['Material'],
                    'Degauss': float(degauss) if degauss != '-' else 0.0,
                    'K_Density': float(k_dist) if k_dist != '-' else 0.0,
                    'Q_Density': float(q_dist) if q_dist != '-' else 0.0,
                    'Result_Value': float(metric_value),
                    'Node_PK': supercon_node.pk,
                    'UUID': supercon_node.uuid.split('-')[0]
                })

        if not flattened_list:
            warnings.warn("No valid data available for HiPlot.", stacklevel=2)
            return None

        experiment = hip.Experiment.from_iterable(flattened_list)
        return experiment

    def get_allen_dynes_tc(self):
        """
        Get Allen-Dynes superconducting critical temperatures.
        """
        allen_dynes_tcs = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: None
                    )
                )
            )
        )
        for item in self._data:
            supercon_node = item['node']
            if supercon_node:
                analyser = SuperConAnalyser(supercon_node)
                results = analyser.a2f_results
                if results:
                    degauss, k_dist, q_dist = self._extract_degauss_k_q(supercon_node)
                    allen_dynes_tcs[item['Material']][degauss][k_dist][q_dist] = results
        
        def default_to_regular(d):
            if isinstance(d, defaultdict):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d

        return default_to_regular(allen_dynes_tcs)

    def plot_allen_dynes_tc_vs_kpoints(
        self,
        qpoints_distance=0.5,
        qfpoints_distance=0.07,
        *,
        materials=None,
        degauss_values=None,
        exclude_degauss=None,
        colors=None,
        figsize=None,
        axes=None,
        xlim=(0.55, 0.04),
        xticks=(0.5, 0.3, 0.2, 0.1, 0.05),
        ylim=None,
        cubic_kpoints_scale=True,
        highlight_points=None,
        legend=True,
        xlabel=None,
        ylabel=None,
        **kwargs,
    ):
        """Compatibility wrapper for k-point Allen-Dynes Tc convergence plots."""
        return self.plot_allen_dynes_tc_convergence(
            sweep='kpoints',
            qpoints_distance=qpoints_distance,
            qfpoints_distance=qfpoints_distance,
            materials=materials,
            degauss_values=degauss_values,
            exclude_degauss=exclude_degauss,
            colors=colors,
            figsize=figsize,
            axes=axes,
            xlim=xlim,
            xticks=xticks,
            ylim=ylim,
            cubic_scale=cubic_kpoints_scale,
            highlight_points=highlight_points,
            legend=legend,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )

    def plot_allen_dynes_tc_convergence(
        self,
        sweep='kpoints',
        *,
        degauss=0.02,
        kpoints_distance=0.15,
        qpoints_distance=0.5,
        qfpoints_distance=0.07,
        materials=None,
        degauss_values=None,
        exclude_degauss=None,
        colors=None,
        figsize=None,
        axes=None,
        xlim=None,
        xticks=None,
        ylim=None,
        cubic_scale=True,
        highlight_points=None,
        legend=None,
        xlabel=None,
        ylabel=None,
        **kwargs,
    ):
        """Plot Allen-Dynes Tc convergence by k-point or q-point distance.

        ``sweep='kpoints'`` fixes q/qf distances and draws one curve for each
        degauss value.  ``sweep='qpoints'`` fixes degauss/k/qf distances and
        draws one q-point convergence curve per material.

        The specialised :meth:`plot_allen_dynes_tc_vs_kpoints` method remains
        available; this method provides one consistent interface for both
        convergence directions.

        ``ylim`` may be one two-value range for every subplot or a range per
        selected material/subplot.

        Pass ``axes`` to use an existing Matplotlib layout; it must contain one
        axis per selected material.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import FixedLocator

        sweep = str(sweep).lower()
        if sweep not in {'kpoints', 'qpoints'}:
            raise ValueError("sweep must be either 'kpoints' or 'qpoints'.")

        all_tcs = self.get_allen_dynes_tc()
        if materials is None:
            selected_materials = list(all_tcs)
        elif isinstance(materials, str):
            selected_materials = [materials]
        else:
            selected_materials = list(materials)
        selected_materials = [material for material in selected_materials if material in all_tcs]
        if not selected_materials:
            raise ValueError('No Allen-Dynes Tc data matches the requested materials.')

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

        def read_tc(parameters):
            if hasattr(parameters, 'get_dict'):
                parameters = parameters.get_dict()
            return parameters.get('Allen_Dynes_Tc') if hasattr(parameters, 'get') else None

        allowed_degauss = set(as_list(degauss_values)) if degauss_values is not None else None
        excluded_degauss = set(as_list(exclude_degauss))
        if (
            isinstance(highlight_points, tuple)
            and len(highlight_points) == 2
            and not isinstance(highlight_points[0], (list, tuple))
        ):
            highlight_points = [highlight_points]
        highlight_points = {tuple(point) for point in as_list(highlight_points)}
        palette = colors if colors is not None else ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8']
        if not palette:
            raise ValueError('colors must contain at least one colour.')

        if xticks is None:
            xticks = (0.5, 0.3, 0.2, 0.1, 0.05) if sweep == 'kpoints' else (1.0, 0.7, 0.5, 0.3)
        if xlim is None and sweep == 'kpoints':
            xlim = (0.55, 0.04)
        if figsize is None:
            figsize = figure_size(columns=len(selected_materials))
        if legend is None:
            legend = sweep == 'kpoints'
        y_limits = _axis_limits(ylim, len(selected_materials))

        font_size = kwargs.get('font_size', DEFAULT_FONT_SIZE)
        with plot_style(
            font_size=font_size,
            title_fontsize=kwargs.get('title_fontsize'),
            label_fontsize=kwargs.get('label_fontsize'),
            tick_fontsize=kwargs.get('tick_fontsize'),
            legend_fontsize=kwargs.get('legend_fontsize'),
        ):
            fig, axes = _plot_axes(
                axes, len(selected_materials), plt=plt, figsize=figsize
            )

            for material_index, material in enumerate(selected_materials):
                axis = axes[material_index]
                material_data = all_tcs[material]

                if sweep == 'kpoints':
                    degauss_keys = [
                        value for value in material_data
                        if (allowed_degauss is None or value in allowed_degauss)
                        and value not in excluded_degauss
                    ]
                    for degauss_index, degauss_value in enumerate(
                        sorted(degauss_keys, key=sort_key, reverse=True)
                    ):
                        points = []
                        for kpoint_value, qpoint_data in material_data[degauss_value].items():
                            tc_value = read_tc(
                                qpoint_data.get(qpoints_distance, {}).get(qfpoints_distance)
                            )
                            if tc_value is not None:
                                points.append((kpoint_value, tc_value))
                        if not points:
                            continue
                        points.sort(key=lambda point: sort_key(point[0]), reverse=True)
                        x_values, y_values = zip(*points)
                        color = palette[degauss_index % len(palette)]
                        try:
                            sigma = f'{float(degauss_value) * 1000:g}'
                        except (TypeError, ValueError):
                            sigma = str(degauss_value)
                        axis.scatter(x_values, y_values, marker='o', s=20, c=color)
                        axis.plot(x_values, y_values, label=rf'$\sigma$={sigma} mRy', c=color)
                        for kpoint_value, tc_value in points:
                            if (degauss_value, kpoint_value) in highlight_points:
                                axis.scatter(kpoint_value, tc_value, marker='*', s=60, c='black', zorder=20)
                    axis.text(
                        0.05, 0.9, material.split('-')[-1], transform=axis.transAxes,
                        bbox={'facecolor': 'white', 'edgecolor': 'none'},
                    )
                    kpoint_xlabel = xlabel if xlabel is not None else kwargs.get('xlabel')
                else:
                    points = []
                    qpoint_data = material_data.get(degauss, {}).get(kpoints_distance, {})
                    for qpoint_value, qfpoint_data in qpoint_data.items():
                        tc_value = read_tc(qfpoint_data.get(qfpoints_distance))
                        if tc_value is not None:
                            points.append((qpoint_value, tc_value))
                    points.sort(key=lambda point: sort_key(point[0]), reverse=True)
                    if points:
                        x_values, y_values = zip(*points)
                        axis.plot(x_values, y_values, marker='o', c='black')
                    axis.text(
                        0.7, 0.9, material.split('-')[-1], transform=axis.transAxes,
                        bbox={'facecolor': 'white', 'edgecolor': 'none'},
                    )
                    axis.set_xlabel(
                        xlabel if xlabel is not None else kwargs.get('xlabel', r'$\Delta_{\mathbf{q}}$ [$\AA^{-1}$]')
                    )

                axis.grid(True, linestyle='--', alpha=0.6)
                if sweep == 'kpoints':
                    configure_kpoint_distance_axis(
                        axis, xlim=xlim, xticks=xticks, xlabel=kpoint_xlabel,
                        cubic_scale=cubic_scale,
                    )
                else:
                    if cubic_scale:
                        axis.set_xscale('function', functions=(np.cbrt, lambda value: value**3))
                    if xlim is not None:
                        axis.set_xlim(xlim)
                    else:
                        axis.invert_xaxis()
                    if xticks is not None:
                        axis.set_xticks(xticks)
                        axis.xaxis.set_major_locator(FixedLocator(xticks))
                        axis.set_xticklabels([str(tick) for tick in xticks])
                if y_limits[material_index] is not None:
                    axis.set_ylim(y_limits[material_index])

            axes[0].set_ylabel(
                ylabel if ylabel is not None else kwargs.get('ylabel', r'$T_c$ (K)')
            )
            if legend:
                axes[0].legend(
                    loc='upper center',
                    facecolor='white',
                    bbox_to_anchor=(1.35, 1.0, 0.6, 0.2),
                    borderaxespad=0,
                    ncol=kwargs.get('legend_ncol', len(palette)),
                    framealpha=1.0,
                    frameon=True,
                )

        return fig, axes


    def plot_a2f_vs_degauss(
        self,
        kpoints_distance=0.15,
        qpoints_distance=0.5,
        qfpoints_distance=0.07,
        *,
        materials=None,
        degauss_values=None,
        exclude_degauss=None,
        cmap='OrRd',
        figsize=None,
        axes=None,
        xlim=(0, 1),
        material_xlims=None,
        ylim=(0, 30),
        legend=True,
        xlabel=None,
        ylabel=None,
        **kwargs,
    ):
        """Plot α²F spectra at fixed k-, q-, and fine-q-point distances.

        Each material receives one subplot and its spectra are overlaid by
        degauss value.  Set ``kpoints_distance=None`` to select the smallest
        available k-point distance for each degauss value, matching the old
        notebook script.

        Parameters
        ----------
        material_xlims
            Optional mapping from material label to a material-specific
            x-axis limit, useful when one spectrum extends beyond ``xlim``.
        ylim
            A two-value range for every subplot, or one range per selected
            material/subplot.
        axes
            Optional Matplotlib Axes (or iterable of axes), one per selected
            material.
        xlabel
            Custom label for the x-axis (defaults to r'$\\alpha^2 F$').
        ylabel
            Custom label for the y-axis (defaults to r'$\\omega$ [meV]').

        Returns
        -------
        tuple
            The Matplotlib ``(figure, axes)`` pair.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        all_nodes = self.get_a2f_nodes()
        if materials is None:
            selected_materials = list(all_nodes)
        elif isinstance(materials, str):
            selected_materials = [materials]
        else:
            selected_materials = list(materials)
        selected_materials = [material for material in selected_materials if material in all_nodes]
        if not selected_materials:
            raise ValueError('No A2F nodes match the requested materials.')

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

        allowed_degauss = as_list(degauss_values) if degauss_values is not None else None
        excluded_degauss = as_list(exclude_degauss)
        material_xlims = material_xlims or {}
        if figsize is None:
            figsize = figure_size(columns=len(selected_materials))
        y_limits = _axis_limits(ylim, len(selected_materials))

        font_size = kwargs.get('font_size', DEFAULT_FONT_SIZE)
        with plot_style(
            font_size=font_size,
            title_fontsize=kwargs.get('title_fontsize'),
            label_fontsize=kwargs.get('label_fontsize'),
            tick_fontsize=kwargs.get('tick_fontsize'),
            legend_fontsize=kwargs.get('legend_fontsize'),
        ):
            fig, axes = _plot_axes(
                axes, len(selected_materials), plt=plt, figsize=figsize
            )

            for material_index, material in enumerate(selected_materials):
                axis = axes[material_index]
                material_data = all_nodes[material]
                degauss_keys = [
                    degauss for degauss in material_data
                    if (allowed_degauss is None or matches_any(degauss, allowed_degauss))
                    and not matches_any(degauss, excluded_degauss)
                ]
                degauss_keys = sorted(degauss_keys, key=sort_key, reverse=True)
                colors = plt.get_cmap(cmap)(np.linspace(0.2, 0.8, len(degauss_keys)))

                for color, degauss in zip(colors, degauss_keys):
                    kpoint_data = material_data[degauss]
                    if not kpoint_data:
                        continue
                    selected_kpoint_distance = (
                        min(kpoint_data, key=sort_key)
                        if kpoints_distance is None else _matching_key(kpoint_data, kpoints_distance)
                    )
                    if selected_kpoint_distance is None or selected_kpoint_distance not in kpoint_data:
                        continue
                    qpoint_data = kpoint_data[selected_kpoint_distance]
                    selected_qpoint_distance = (
                        min(qpoint_data, key=sort_key)
                        if qpoints_distance is None else _matching_key(qpoint_data, qpoints_distance)
                    )
                    if selected_qpoint_distance is None or selected_qpoint_distance not in qpoint_data:
                        continue
                    qfpoint_data = qpoint_data[selected_qpoint_distance]
                    selected_qfpoint_distance = (
                        min(qfpoint_data, key=sort_key)
                        if qfpoints_distance is None else _matching_key(qfpoint_data, qfpoints_distance)
                    )
                    if selected_qfpoint_distance is None or selected_qfpoint_distance not in qfpoint_data:
                        continue
                    node = qfpoint_data.get(selected_qfpoint_distance)
                    if node is None:
                        continue

                    a2f_data = _get_a2f_arraydata(node)
                    try:
                        output_parameters = node.outputs.output_parameters
                    except AttributeError:
                        output_parameters = None
                    if a2f_data is None or output_parameters is None:
                        continue

                    plot_a2f(
                        a2f_arraydata=a2f_data,
                        output_parameters=output_parameters,
                        axis=axis,
                        label1=None,
                        label2=None,
                        color=color,
                    )
                    try:
                        sigma = f'{float(degauss) * 1000:g}'
                    except (TypeError, ValueError):
                        sigma = str(degauss)
                    axis.plot([], [], label=rf'$\sigma$={sigma} mRy', color=color)

                axis.text(
                    0.07,
                    0.95,
                    material.split('-')[-1],
                    transform=axis.transAxes,
                    bbox={'facecolor': 'white', 'edgecolor': 'none'},
                )
                axis.set_xlabel(
                    xlabel if xlabel is not None else kwargs.get('xlabel', r'$\alpha^2 F$')
                )
                axis.set_ylabel('')
                axis.set_yticks([])
                axis.set_yticklabels([])
                axis.set_xlim(material_xlims.get(material, xlim))
                if y_limits[material_index] is not None:
                    axis.set_ylim(y_limits[material_index])
                axis.grid(True, linestyle='--', alpha=0.6)

            axes[0].set_ylabel(
                ylabel if ylabel is not None else kwargs.get('ylabel', r'$\omega$ [meV]')
            )
            if legend:
                axes[0].legend(
                    loc='upper center',
                    facecolor='white',
                    bbox_to_anchor=(1.35, 0.9, 0.6, 0.2),
                    borderaxespad=0,
                    ncol=kwargs.get('legend_ncol', 4),
                    framealpha=1.0,
                    frameon=True,
                )

        return fig, axes


    def plot_a2f_vs_kpoints(
        self,
        degauss=0.02,
        qpoints_distance=0.5,
        qfpoints_distance=0.07,
        *,
        materials=None,
        kpoints_values=None,
        exclude_kpoints=None,
        cmap='OrRd',
        figsize=None,
        axes=None,
        xlim=(0, 1),
        material_xlims=None,
        ylim=(0, 30),
        legend=True,
        xlabel=None,
        ylabel=None,
        **kwargs,
    ):
        """Plot α²F spectra at fixed degauss, q-, and fine-q-point distances across k-point distances.

        Each material receives one subplot and its spectra are overlaid by
        k-point distance.  Set ``degauss=None`` to select the smallest
        available smearing value for each material.  Use ``kpoints_values``
        and ``exclude_kpoints`` to select a subset of k-point distances.

        Parameters
        ----------
        degauss
            Smearing value. Set to ``None`` to select the smallest available
            degauss value for each material.
        qpoints_distance
            Fixed q-point distance.
        qfpoints_distance
            Fixed fine q-point distance.
        materials
            Optional material name or list of material names to include.
        kpoints_values
            Optional whitelist of k-point distances to plot.
        exclude_kpoints
            Optional blacklist of k-point distances to ignore.
        cmap
            Matplotlib colormap name used to generate colors across k-point distances.
        figsize
            Matplotlib figure size tuple.
        axes
            Optional Matplotlib Axes (or iterable of axes), one per selected material.
        xlim
            Default x-axis limits for the α²F spectra.
        material_xlims
            Optional mapping from material label to a material-specific x-axis limit.
        ylim
            A two-value range for every subplot, or one range per selected material/subplot.
        legend
            Whether to display a legend.
        xlabel
            Custom label for the x-axis (defaults to r'$\\alpha^2 F$').
        ylabel
            Custom label for the y-axis (defaults to r'$\\omega$ [meV]').

        Returns
        -------
        tuple
            The Matplotlib ``(figure, axes)`` pair.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        all_nodes = self.get_a2f_nodes()
        if materials is None:
            selected_materials = list(all_nodes)
        elif isinstance(materials, str):
            selected_materials = [materials]
        else:
            selected_materials = list(materials)
        selected_materials = [material for material in selected_materials if material in all_nodes]
        if not selected_materials:
            raise ValueError('No A2F nodes match the requested materials.')

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
        material_xlims = material_xlims or {}
        if figsize is None:
            figsize = figure_size(columns=len(selected_materials))
        y_limits = _axis_limits(ylim, len(selected_materials))

        font_size = kwargs.get('font_size', DEFAULT_FONT_SIZE)
        with plot_style(
            font_size=font_size,
            title_fontsize=kwargs.get('title_fontsize'),
            label_fontsize=kwargs.get('label_fontsize'),
            tick_fontsize=kwargs.get('tick_fontsize'),
            legend_fontsize=kwargs.get('legend_fontsize'),
        ):
            fig, axes = _plot_axes(
                axes, len(selected_materials), plt=plt, figsize=figsize
            )

            for material_index, material in enumerate(selected_materials):
                axis = axes[material_index]
                material_data = all_nodes[material]
                selected_degauss = (
                    min(material_data, key=sort_key)
                    if degauss is None else _matching_key(material_data, degauss)
                )
                if selected_degauss is None or selected_degauss not in material_data:
                    continue
                kpoint_data = material_data[selected_degauss]
                kpoint_keys = [
                    kpoint for kpoint in kpoint_data
                    if (allowed_kpoints is None or matches_any(kpoint, allowed_kpoints))
                    and not matches_any(kpoint, excluded_kpoints)
                ]
                kpoint_keys = sorted(kpoint_keys, key=sort_key, reverse=True)
                colors = plt.get_cmap(cmap)(np.linspace(0.2, 0.8, len(kpoint_keys)))

                for color, kpoint in zip(colors, kpoint_keys):
                    qpoint_data = kpoint_data[kpoint]
                    selected_qpoint = (
                        min(qpoint_data, key=sort_key)
                        if qpoints_distance is None else _matching_key(qpoint_data, qpoints_distance)
                    )
                    if selected_qpoint is None or selected_qpoint not in qpoint_data:
                        continue
                    qfpoint_data = qpoint_data[selected_qpoint]
                    selected_qfpoint = (
                        min(qfpoint_data, key=sort_key)
                        if qfpoints_distance is None else _matching_key(qfpoint_data, qfpoints_distance)
                    )
                    if selected_qfpoint is None or selected_qfpoint not in qfpoint_data:
                        continue
                    node = qfpoint_data.get(selected_qfpoint)
                    if node is None:
                        continue

                    a2f_data = _get_a2f_arraydata(node)
                    try:
                        output_parameters = node.outputs.output_parameters
                    except AttributeError:
                        output_parameters = None
                    if a2f_data is None or output_parameters is None:
                        continue

                    plot_a2f(
                        a2f_arraydata=a2f_data,
                        output_parameters=output_parameters,
                        axis=axis,
                        label1=None,
                        label2=None,
                        color=color,
                    )
                    try:
                        distance = f'{float(kpoint):g}'
                    except (TypeError, ValueError):
                        distance = str(kpoint)
                    axis.plot([], [], label=rf'$d_k$={distance} $\AA^{{-1}}$', color=color)

                axis.text(
                    0.07,
                    0.95,
                    material.split('-')[-1],
                    transform=axis.transAxes,
                    bbox={'facecolor': 'white', 'edgecolor': 'none'},
                )
                axis.set_xlabel(
                    xlabel if xlabel is not None else kwargs.get('xlabel', r'$\alpha^2 F$')
                )
                axis.set_ylabel('')
                axis.set_yticks([])
                axis.set_yticklabels([])
                axis.set_xlim(material_xlims.get(material, xlim))
                if y_limits[material_index] is not None:
                    axis.set_ylim(y_limits[material_index])
                axis.grid(True, linestyle='--', alpha=0.6)

            axes[0].set_ylabel(
                ylabel if ylabel is not None else kwargs.get('ylabel', r'$\omega$ [meV]')
            )
            if legend:
                axes[0].legend(
                    loc='upper center',
                    facecolor='white',
                    bbox_to_anchor=(1.35, 0.9, 0.6, 0.2),
                    borderaxespad=0,
                    ncol=kwargs.get('legend_ncol', 4),
                    framealpha=1.0,
                    frameon=True,
                )

        return fig, axes

    def get_a2f_nodes(self):
        """
        Get a2f node.
        """
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
        for item in self._data:
            supercon_node = item['node']
            if supercon_node:
                analyser = SuperConAnalyser(supercon_node)
                degauss, k_dist, q_dist = self._extract_degauss_k_q(supercon_node)
                for link_label, value in analyser.conv.items():
                    qf_dist = value.node.inputs.qfpoints_distance.value
                    nodes[item['Material']][degauss][k_dist][q_dist][qf_dist] = value.node
        
        def default_to_regular(d):
            if isinstance(d, defaultdict):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d

        return default_to_regular(nodes)

    @styled_plot
    def plot_a2f(
        self,
        axis = None,
        show_data = False,
        **kwargs,
        ):
        from matplotlib import pyplot as plt
        import numpy

        # Identify all unique parameters to set up grid
        materials = sorted(list(set(item['Material'] for item in self._data)))
        
        all_k_densities = set()
        for item in self._data:
            node = item['node']
            if node:
                _, k_dist, _ = self._extract_degauss_k_q(node)
                all_k_densities.add(k_dist)
        k_densities = sorted(list(all_k_densities), reverse=True)

        num_materials = len(materials)
        num_k_dens = len(k_densities)
        
        if num_materials == 0 or num_k_dens == 0:
            print("No data to plot.")
            return

        fig, axs = plt.subplots(
            num_k_dens, num_materials, 
            figsize=figure_size(columns=num_materials, rows=num_k_dens),
            squeeze=False 
        )

        cmap_name = kwargs.get('cmap', 'viridis')
        cmap = plt.get_cmap(cmap_name)
        
        for j, material in enumerate(materials):
            material_items = [item for item in self._data if item['Material'] == material]
            
            all_degausses = set()
            for item in material_items:
                if item['node']:
                    degauss, _, _ = self._extract_degauss_k_q(item['node'])
                    all_degausses.add(degauss)
            sorted_degausses = sorted(list(all_degausses))
            colors = cmap(numpy.linspace(0, 1, len(sorted_degausses)))
            
            for i, k_dist in enumerate(k_densities):
                ax = axs[i, j]
                found_data = False
                
                for d_idx, degauss in enumerate(sorted_degausses):
                    matching_items = []
                    for item in material_items:
                        if item['node']:
                            d, k, q = self._extract_degauss_k_q(item['node'])
                            if d == degauss and k == k_dist:
                                matching_items.append((item, q))
                                
                    for item, q_dist in matching_items:
                        node = item['node']
                        if node is None or not node.is_finished_ok:
                            continue
                        
                        try:
                            analyser = SuperConAnalyser(node)
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
                    ax.set_title(f"{material}\nK={k_dist}")
                    if i == num_k_dens - 1:
                        ax.set_xlabel(r"$\alpha^2F$")
                    if j == 0:
                        ax.set_ylabel(r"$\omega$")
                    ax.legend(fontsize='x-small')

        plt.tight_layout()
        return fig, axs

    def show_interactive(self):
        """
        Displays an interactive Jupyter table of SuperConWorkChain nodes.
        Clicking on a row triggers a Python callback to highlight that row
        and display the node's full nested parameters in a collapsible HTML viewer.
        """
        import ipywidgets as widgets
        from IPython.display import display
        import pandas as pd
        
        df = self.get_table()
        if df.empty:
            print("No data available to display.")
            return

        # Ensure index elements are python ints
        df.index = df.index.map(int)

        node_map = {int(item['PK']): item['node'] for item in self._data if item['node'] is not None}
        
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
                status_val = str(row_data['status'])
                material_val = str(row_data['Material'])
                
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
        
        table_container = widgets.VBox([headers, table_body], layout=widgets.Layout(width='55%'))
        
        details_container = widgets.VBox([
            details_output
        ], layout=widgets.Layout(width='43%', margin='0 0 0 2%'))
        
        main_layout = widgets.HBox([table_container, details_container], layout=widgets.Layout(width='100%'))
        
        if not df.empty:
            select_row(df.index[0])
            
        display(main_layout)
