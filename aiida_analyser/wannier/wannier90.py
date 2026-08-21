from ..core.analyser_registry import resolve_analyser
from ..core.base import BaseWorkChainAnalyser
from ..quantumespresso.projwfc_base import ProjwfcBaseAnalyser
from ..quantumespresso.pw_base import PwBaseAnalyser
from ..quantumespresso.pw2wannier90_base import Pw2Wannier90BaseAnalyser
from ..visualization.plots import plot_bands
from .wannier90_base import Wannier90BaseAnalyser


class Wannier90Analyser(BaseWorkChainAnalyser):
    """
    Analyser for the Wannier90WorkChain.
    This is a composite workchain analyser that handles a workchain containing
    multiple sub-workchains: scf, nscf, projwfc, wannier90_pp, pw2wannier90, wannier90.
    
    For individual base workchains, use:
    - PwBaseAnalyser for scf, nscf
    - ProjwfcBaseAnalyser for projwfc
    - Pw2Wannier90BaseAnalyser for pw2wannier90
    - Wannier90BaseAnalyser for wannier90_pp, wannier90
    """

    def _get_wannier90_pp_labels(self):
        return self._get_child_labels(
            labels=('wannier90_pp',),
            prefixes=('wannier90_pp_',),
        )

    def _get_pw2wannier90_labels(self):
        return self._get_child_labels(
            labels=('pw2wannier90',),
            prefixes=('pw2wannier90_',),
        )

    def _get_wannier90_run_labels(self):
        return [
            label for label in self._get_child_labels(
                labels=('wannier90',),
                prefixes=('wannier90_',),
            )
            if label not in set(self._get_wannier90_pp_labels())
        ]

    @property
    def seekpath_structure_analysis(self):
        try:
            return self.process_tree.seekpath_structure_analysis
        except AttributeError:
            return None

    @property
    def scf(self):
        try:
            return self.process_tree.scf
        except AttributeError:
            return None

    @property
    def nscf(self):
        try:
            return self.process_tree.nscf
        except AttributeError:
            return None

    @property
    def projwfc(self):
        try:
            return self.process_tree.projwfc
        except AttributeError:
            return None

    @property
    def wannier90_pp(self):
        try:
            return self.process_tree.wannier90_pp
        except AttributeError:
            return None

    @property
    def pw2wannier90(self):
        try:
            return self.process_tree.pw2wannier90
        except AttributeError:
            return None

    @property
    def wannier90(self):
        try:
            return self.process_tree.wannier90
        except AttributeError:
            return None

    def get_source(self):
        """Get the source of the workchain."""
        if all(key in self.node.base.extras for key in ['source_db', 'source_id']):
            return (self.node.base.extras.get('source_db'), self.node.base.extras.get('source_id'))
        elif all(key in self.node.inputs.structure.base.extras for key in ['source_db', 'source_id']):
            return (self.node.inputs.structure.base.extras.get('source_db'), self.node.inputs.structure.base.extras.get('source_id'))
        else:
            raise ValueError('Source is not set')

    def get_state(self):
        """Get the state of the workchain."""
        subprocesses = []
        required_subprocesses = []

        for label in self._get_child_labels(labels=('scf',)):
            analyser_cls = resolve_analyser(self.process_tree[label].node) or PwBaseAnalyser
            subprocesses.append((label, analyser_cls))
            required_subprocesses.append(label)

        for label in self._get_child_labels(labels=('nscf',)):
            analyser_cls = resolve_analyser(self.process_tree[label].node) or PwBaseAnalyser
            subprocesses.append((label, analyser_cls))
            required_subprocesses.append(label)

        for label in self._get_child_labels(labels=('projwfc',)):
            analyser_cls = resolve_analyser(self.process_tree[label].node) or ProjwfcBaseAnalyser
            subprocesses.append((label, analyser_cls))

        wannier90_pp_labels = self._get_wannier90_pp_labels()
        for label in wannier90_pp_labels:
            analyser_cls = resolve_analyser(self.process_tree[label].node) or Wannier90BaseAnalyser
            subprocesses.append((label, analyser_cls))
            required_subprocesses.append(label)

        pw2wannier90_labels = self._get_pw2wannier90_labels()
        for label in pw2wannier90_labels:
            analyser_cls = resolve_analyser(self.process_tree[label].node) or Pw2Wannier90BaseAnalyser
            subprocesses.append((label, analyser_cls))
            required_subprocesses.append(label)

        wannier90_labels = self._get_wannier90_run_labels()
        for label in wannier90_labels:
            analyser_cls = resolve_analyser(self.process_tree[label].node) or Wannier90BaseAnalyser
            subprocesses.append((label, analyser_cls))
            required_subprocesses.append(label)

        return self._get_state_from_subprocesses(
            subprocesses,
            required_subprocesses=tuple(required_subprocesses),
        )

    def plot_bands(
        self,
        axis=None,
        ylabel='Energy (eV)',
        **kwargs,
    ):
        """
        Plot the band structure.
        """
        bands = self.node.outputs.band_structure
        seekpath_params = self.node.outputs.seekpath_parameters
        fermi_energy = self.node.outputs.scf.output_parameters.get('fermi_energy')
        plot_bands(
            bands,
            axis=axis,
            reference_energy=fermi_energy,
            seekpath_params=seekpath_params,
            ylabel=ylabel,
            **kwargs,
        )

    def print_state(self):
        """Print the state of the workchain."""
        result = self.get_state()
        if not result:
            print(f"Can't check the state of Wannier90WorkChain<{self.node.pk}>.")
            return
        path, process_state, exit_code = result
        print(f"Wannier90WorkChain<{self.node.pk}> is {process_state} at {path} with exit code {exit_code}.")
    
    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message, success = super().clean_workchain(dry_run=dry_run)

        return message
