from ..core.base import AnalysisExitCode, BaseCalculationAnalyser, BaseParser


PH_OUTPUT_RULES = (
    (
        ('mpich error',),
        AnalysisExitCode(3, 'MPICH_ERROR', 'MPI job aborted'),
    ),
    (
        ('no convergence has been achieved',),
        AnalysisExitCode(
            3000,
            'ERROR_CONVERGENCE_NOT_REACHED',
            'No convergence has been achieved',
        ),
    ),
    (
        ('error in routine davcio',),
        AnalysisExitCode(3001, 'ERROR_DAVCIO', 'Error in routine davcio'),
    ),
    (
        ('wrong representation',),
        AnalysisExitCode(3002, 'ERROR_WRONG_REPRESENTATION', 'Wrong representation'),
    ),
    (
        ('q-mesh breaks symmetry',),
        AnalysisExitCode(
            3003,
            'ERROR_QMESH_BREAKS_SYMMETRY',
            'q-mesh breaks symmetry',
        ),
    ),
    (
        ('fft grid incompatible with symmetry',),
        AnalysisExitCode(
            3004,
            'ERROR_FFT_GRID_INCOMPATIBLE',
            'FFT grid incompatible with symmetry',
        ),
    ),
    (
        ('only some processors converged',),
        AnalysisExitCode(
            3005,
            'ERROR_PARTIAL_PROCESSORS_CONVERGED',
            'Only some processors converged',
        ),
    ),
    (
        ('problems computing cholesky',),
        AnalysisExitCode(
            3006,
            'ERROR_PROBLEMS_COMPUTING_CHOLESKY',
            'Problems computing Cholesky decomposition',
        ),
    ),
    (
        ('unknown mode symmetry',),
        AnalysisExitCode(
            3007,
            'ERROR_UNKNOWN_MODE_SYMMETRY',
            'Unknown mode symmetry',
        ),
    ),
    (
        ('maximum cpu time exceeded',),
        AnalysisExitCode(
            3008,
            'ERROR_MAXIMUM_CPU_TIME_EXCEEDED',
            'Maximum CPU time exceeded',
        ),
    ),
    (
        ('s matrix not positive definite',),
        AnalysisExitCode(
            3009,
            'ERROR_S_MATRIX_NOT_POSITIVE_DEFINITE',
            'S matrix is not positive definite',
        ),
    ),
    (
        ('eigenvectors failed to converge',),
        AnalysisExitCode(
            3010,
            'ERROR_EIGENVECTORS_NOT_CONVERGE',
            'Eigenvectors failed to converge',
        ),
    ),
    (
        ('error in routine dirop',),
        AnalysisExitCode(3011, 'ERROR_DIOPN', 'Error in routine dirop'),
    ),
    (
        ('error in routine read_wfc',),
        AnalysisExitCode(3012, 'ERROR_READ_WFC', 'Error in routine read_wfc'),
    ),
    (
        ('error in routine find_mode_sym',),
        AnalysisExitCode(
            7301,
            'PH_ERROR_FIND_MODE_SYM',
            'ph.x failed in find_mode_sym',
        ),
    ),
    (
        ('error in routine cdiaghg',),
        AnalysisExitCode(7302, 'PH_ERROR_CDIAGHG', 'ph.x failed in cdiaghg'),
    ),
)


class PhAnalyser(BaseCalculationAnalyser):
    """Analyser for the PhCalculation calcjob."""

    failure_parsers = (BaseParser(PH_OUTPUT_RULES),)
