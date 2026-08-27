from ..core.base import AnalysisExitCode, BaseCalculationAnalyser, BaseParser


class Wannier90CalculationAnalyser(BaseCalculationAnalyser):
    """Analyser for the Wannier90Calculation calcjob."""

    failure_parsers = (
        BaseParser(
            (
                (
                    ('disentanglement', 'converg'),
                    AnalysisExitCode(
                        7101,
                        'W90_DISENTANGLEMENT_NOT_CONVERGED',
                        'Wannier90 disentanglement did not converge',
                    ),
                ),
                (
                    ('param_get_projections',),
                    AnalysisExitCode(
                        7102,
                        'W90_PROJECTIONS_ERROR',
                        'Wannier90 could not construct projections',
                    ),
                ),
                (
                    ('has too many projections to be used without selecting a subset',),
                    AnalysisExitCode(
                        7102,
                        'W90_PROJECTIONS_ERROR',
                        'Wannier90 requires a selected projection subset',
                    ),
                ),
            )
        ),
    )
