from ..core.base import AnalysisExitCode, BaseCalculationAnalyser, BaseParser


class EpwAnalyser(BaseCalculationAnalyser):
    """Analyser for the EpwCalculation calcjob."""

    failure_parsers = (
        BaseParser(
            (
                (
                    ('error in routine davcio',),
                    AnalysisExitCode(7201, 'EPW_ERROR_DAVCIO', 'EPW failed in davcio'),
                ),
                (
                    ('error in routine read_wfc',),
                    AnalysisExitCode(
                        7202,
                        'EPW_ERROR_READ_WFC',
                        'EPW could not read wavefunctions',
                    ),
                ),
                (
                    ('maximum cpu time exceeded',),
                    AnalysisExitCode(
                        7203,
                        'EPW_MAXIMUM_CPU_TIME_EXCEEDED',
                        'EPW reached its configured CPU-time limit',
                    ),
                ),
            )
        ),
    )
