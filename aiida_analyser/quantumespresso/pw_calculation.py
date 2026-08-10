from pathlib import Path

from ..base import BaseCalculationAnalyser


class PwAnalyser(BaseCalculationAnalyser):
    """Analyser for the PwCalculation calcjob."""

    def copy_tree(self, destpath: Path) -> Path:
        """Copy the calcjob files and export the pseudos used by the calculation."""
        super().copy_tree(destpath)

        pseudo_dir = destpath / 'pseudo'
        pseudo_dir.mkdir(parents=True, exist_ok=True)

        for pseudo in self.node.inputs.pseudos.values():
            with (pseudo_dir / pseudo.filename).open('w') as handle:
                handle.write(pseudo.get_content())

        return destpath
