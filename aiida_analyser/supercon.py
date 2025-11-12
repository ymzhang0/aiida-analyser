from pathlib import Path

from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
import numpy
from .workchains import clean_workdir
from .base import BaseWorkChainAnalyser
from .epw_prep import EpwPrepWorkChainAnalyser
from .calculators import _calculate_iso_tc, check_convergence
from .plot import (
    plot_a2f,
    plot_eldos,
    plot_aniso_gap_function,
    plot_phdos,
    plot_iso_gap_function
)

class EpwSuperConWorkChainAnalyser(BaseWorkChainAnalyser):
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
        return self._get_state_from_tree()

    @property
    def a2f_results(self):
        """Get the results of the a2f workchain."""
        if self.process_tree.a2f.node:
            return self.process_tree.a2f.node.node.outputs.output_parameters
        else:
            conv_results = {}
            for _, node in self.conv.items():
                qfpoints_distance = node.node.inputs.qfpoints_distance.value
                conv_results[qfpoints_distance] = node.node.outputs.output_parameters
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
        from aiida_epw_workflows.parsers.epw import EpwParser
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
        return EpwSuperConWorkChainAnalyser.get_processes_dict(self.node)

    @property
    def retrieved(self):
        """Get the retrieved dictionary."""
        return EpwSuperConWorkChainAnalyser.get_retrieved(self.node)

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