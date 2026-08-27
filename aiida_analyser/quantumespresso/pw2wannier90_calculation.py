from ..core.base import AnalysisExitCode, BaseCalculationAnalyser, BaseParser


class Pw2Wannier90Analyser(BaseCalculationAnalyser):
    """Analyser for the Pw2wannier90Calculation calcjob."""

    failure_parsers = (
        BaseParser(
            (
                (
                    ('error in routine read_wfc',),
                    AnalysisExitCode(
                        7401,
                        'PW2W90_ERROR_READ_WFC',
                        'pw2wannier90.x could not read wavefunctions',
                    ),
                ),
                (
                    ('error in routine davcio',),
                    AnalysisExitCode(
                        7402,
                        'PW2W90_ERROR_DAVCIO',
                        'pw2wannier90.x failed in davcio',
                    ),
                ),
            )
        ),
    )
