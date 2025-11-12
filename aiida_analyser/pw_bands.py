from aiida import orm
from .base import BaseWorkChainAnalyser

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