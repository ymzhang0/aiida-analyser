from pathlib import Path

from ..core.base import AnalysisExitCode, BaseCalculationAnalyser, BaseParser


class PwAnalyser(BaseCalculationAnalyser):
    """Analyser for the PwCalculation calcjob."""

    failure_parsers = (
        BaseParser(
            (
                (
                    ('convergence not achieved',),
                    AnalysisExitCode(
                        7501,
                        'PW_CONVERGENCE_NOT_REACHED',
                        'pw.x did not reach self-consistency',
                    ),
                ),
                (
                    ('error in routine davcio',),
                    AnalysisExitCode(7502, 'PW_ERROR_DAVCIO', 'pw.x failed in davcio'),
                ),
                (
                    ('error in routine read_wfc',),
                    AnalysisExitCode(
                        7503,
                        'PW_ERROR_READ_WFC',
                        'pw.x could not read wavefunctions',
                    ),
                ),
            )
        ),
    )

    def copy_tree(self, destpath: Path) -> Path:
        """Copy the calcjob files and export the pseudos used by the calculation."""
        super().copy_tree(destpath)

        pseudo_dir = destpath / 'pseudo'
        pseudo_dir.mkdir(parents=True, exist_ok=True)

        for pseudo in self.node.inputs.pseudos.values():
            with (pseudo_dir / pseudo.filename).open('w') as handle:
                handle.write(pseudo.get_content())

        return destpath
