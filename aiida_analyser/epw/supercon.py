import html
from pathlib import Path

from aiida import orm
import numpy
from collections import defaultdict
import warnings
from ..groupdata import BaseGroupData, render_process_node_details
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

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct EpwBaseWorkChain child."""
        return self._get_calcjob_paths_for_direct_children(
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
        paths = EpwBaseWorkChainAnalyser(self.aniso.node).get_calcjob_paths()
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
        import matplotlib.pyplot as plt
        fontsize = kwargs.get('font_size', kwargs.get('label_fontsize', 18))
        plt.rcParams.update({'font.size': fontsize})
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['STIXGeneral']
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams['mathtext.default'] = 'regular'

    def show_pw_bands(self):
        """Show the qe bands."""
        self._set_plot_defaults()
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
            
class SuperConData(BaseGroupData):

    def __init__(self, groups=None):
        super().__init__(groups)
        self.get_data()

    def get_data(self):
        self._data = self._flatten_data()

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

    def _flatten_data(self):
        from aiida import orm
        flattened_list = []

        if not self._groups:
            return flattened_list

        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={'label': {'in': self._groups}}, tag='group')
        qb.append(orm.ProcessNode, with_group='group', filters={'attributes.process_label': 'SuperConWorkChain'})

        for r in qb.all():
            node = r[0]
            self.check_protocol(node)
            extras = node.base.extras.all
            try:
                formula = f"{extras['source_db']}-{extras['source_id']}-{extras['formula']}"
            except (KeyError, ValueError) as e:
                try:
                    formula = node.inputs.structure.get_formula()
                except Exception:
                    formula = 'N/A'
            try:
                self.check_protocol(node)
                degauss, kpoints_distance, qpoints_distance = _safe_get_extras(node)
            except Exception as e:
                logging.error(f'Node<{node.pk}> processing failed: {e}')
            # parent_epw pointing to parent_folder_epw's creator
            parent_epw = None
            if 'parent_folder_epw' in node.inputs:
                try:
                    parent_epw_node = node.inputs.parent_folder_epw.creator
                    if parent_epw_node:
                        parent_epw = parent_epw_node.pk
                except Exception:
                    pass

            status_emoji = self.get_status_string(node)

            flattened_list.append({
                'PK': node.pk,
                'Material': formula,
                'status': status_emoji,
                'degauss': degauss,
                'kpoints_distance': kpoints_distance,
                'qpoints_distance': qpoints_distance,
                'parent_epw': parent_epw,
                'node': node,
            })

        return flattened_list

    def _get_dataframe(self, property_filter=None):
        """Build one row per EpwPrep work chain, indexed by PK."""
        import pandas as pd

        if not self._data:
            return pd.DataFrame(columns=['Material', 'degauss', 'kpoints_distance', 'qpoints_distance', 'Process', 'Status'])

        data_list = self._data
        if property_filter:
            filtered_list = []
            for item in data_list:
                try:
                    analyser = EpwPrepWorkChainAnalyser(item['node'])
                    if callable(property_filter):
                        res = property_filter(analyser)
                    elif isinstance(property_filter, str):
                        prop = property_filter.strip()
                        if prop.startswith('not '):
                            res = not bool(getattr(analyser, prop[4:].strip(), False))
                        elif prop.startswith('!'):
                            res = not bool(getattr(analyser, prop[1:].strip(), False))
                        elif prop.startswith('~'):
                            res = not bool(getattr(analyser, prop[1:].strip(), False))
                        else:
                            res = bool(getattr(analyser, prop, False))
                    else:
                        res = False
                    
                    if res:
                        filtered_list.append(item)
                except Exception as e:
                    logging.warning(f"Error filtering node {item['PK']}: {e}")
                    pass
            data_list = filtered_list
            
        if not data_list:
            return pd.DataFrame(columns=['PK', 'Material', 'degauss', 'kpoints_distance', 'qpoints_distance', 'Status']).set_index('PK')

        dataframe = pd.DataFrame(data_list).drop(columns=['node'])
        return dataframe.set_index('PK').sort_index()

    @staticmethod
    def _filter_by_formula(dataframe, formula_contains=None, formula_match='any'):
        """Filter rows by case-insensitive substrings in the material formula."""
        if formula_contains is None or formula_contains == '':
            return dataframe

        terms = (
            [formula_contains]
            if isinstance(formula_contains, str)
            else list(formula_contains)
        )
        terms = [str(term).strip().lower() for term in terms if str(term).strip()]
        if not terms:
            return dataframe
        if formula_match not in {'any', 'all'}:
            raise ValueError("formula_match must be either 'any' or 'all'")

        formulas = dataframe['Material'].astype(str).str.lower()
        masks = [formulas.str.contains(term, regex=False) for term in terms]
        mask = masks[0]
        for next_mask in masks[1:]:
            mask = mask | next_mask if formula_match == 'any' else mask & next_mask
        return dataframe.loc[mask]

    def get_table(
        self,
        display_mode='dataframe',
        *,
        max_height=600,
        page_size=25,
        formula_contains=None,
        formula_match='any',
        property_filter=None,
    ):
        """Return or display the EpwPrep table.

        :param display_mode: One of ``dataframe`` (return the normal DataFrame),
            ``all`` (display every row), ``scroll`` (display a scrollable table),
            or ``interactive`` (searchable, paginated node browser).
        :param max_height: Maximum table height in pixels for scrollable modes.
        :param page_size: Initial number of rows per page in interactive mode.
        :param formula_contains: A substring or iterable of substrings that must
            occur in the material formula, matched case-insensitively.
        :param formula_match: Use ``any`` to match at least one substring or
            ``all`` to require every substring.
        :param property_filter: A string of a boolean property in EpwPrepWorkChainAnalyser
            to filter the workchains (e.g. 'is_stable').
        """
        dataframe = self._filter_by_formula(
            self._get_dataframe(property_filter=property_filter),
            formula_contains=formula_contains,
            formula_match=formula_match,
        )
        mode = 'dataframe' if display_mode is None else str(display_mode).lower()

        if mode in {'dataframe', 'default'}:
            return dataframe

        if mode == 'all':
            import pandas as pd
            from IPython.display import display

            with pd.option_context('display.max_rows', None):
                display(dataframe)
            return None

        if mode == 'scroll':
            from IPython.display import HTML, display

            display(HTML(
                f'<div style="max-height:{int(max_height)}px; overflow:auto;">'
                f'{dataframe.to_html()}</div>'
            ))
            return None

        if mode == 'interactive':
            return self.show_interactive(
                max_height=max_height,
                page_size=page_size,
                formula_contains=formula_contains,
                formula_match=formula_match,
            )

        raise ValueError(
            "display_mode must be one of 'dataframe', 'all', 'scroll', or 'interactive'"
        )

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
                analyser = SuperConWorkChainAnalyser(supercon_node)
                results = analyser.a2f_results
                if results:
                    degauss, k_dist, q_dist = self._extract_degauss_k_q(supercon_node)
                    allen_dynes_tcs[item['Material']][degauss][k_dist][q_dist] = results
        
        def default_to_regular(d):
            if isinstance(d, defaultdict):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d

        return default_to_regular(allen_dynes_tcs)

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
                analyser = SuperConWorkChainAnalyser(supercon_node)
                degauss, k_dist, q_dist = self._extract_degauss_k_q(supercon_node)
                for link_label, value in analyser.conv.items():
                    qf_dist = value.node.inputs.qfpoints_distance.value
                    nodes[item['Material']][degauss][k_dist][q_dist][qf_dist] = value.node
        
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
            figsize=(5*num_materials, 4*num_k_dens),
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
                    ax.set_title(f"{material}\nK={k_dist}")
                    if i == num_k_dens - 1:
                        ax.set_xlabel(r"$\alpha^2F$")
                    if j == 0:
                        ax.set_ylabel(r"$\omega$")
                    ax.legend(fontsize='x-small')

        plt.tight_layout()
        return fig, axs

    def dump(self, dest:Path|str,):
        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={'label': {'in': self._groups}}, tag='group')
        qb.append(orm.ProcessNode, with_group='group', filters={'attributes.process_label': 'SuperConWorkChain'})

        if type(dest) == str:
            dest = Path(dest)
        if not dest.exists():
            dest.mkdir(parents=True)

        for [node] in qb.all():
            try:
                analyser = SuperConWorkChainAnalyser(node)
                analyser.copy_tree(dest / str(node.pk))
            except Exception as e:
                logging.warning(f"Failed to dump node {node.pk}: {e}")

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
