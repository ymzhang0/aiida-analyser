from ..quantumespresso.pw_relax import PwRelaxAnalyser
from ..quantumespresso.pw_base import PwBaseAnalyser
from .sfebase import SFEBaseAnalyser
from .layer_relax import LayerRelaxAnalyser

class USFEAnalyser(SFEBaseAnalyser):
    """
    Analyser for the USFEWorkChain.
    """

    @property
    def layer_relax(self):
        return self._get_node_from_tree('layer_relax')

    def copy_tree(self, destpath):
        """Copy the tree using the layered SFEBase child delegation."""
        return super().copy_tree(destpath)

    def get_calcjob_paths(self):
        """Get calcjob remote paths using the layered SFEBase child delegation."""
        return super().get_calcjob_paths()


