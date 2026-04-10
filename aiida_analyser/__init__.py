"""AiiDA analysis helpers for AiiDA work chains."""

from importlib import import_module

__version__ = "0.1.0"

_EXTRA_MODULES = {
    'structure': {'ase'},
    'thermo': {'matplotlib', 'numpy'},
    'dislocation': {'aiida_dislocation', 'ase', 'matplotlib', 'numpy', 'pandas', 'scipy'},
    'epw': {'aiida_epw', 'ase', 'matplotlib', 'numpy', 'pandas', 'scipy'},
    'hubbard': {'aiida_hubbard', 'ase', 'matplotlib', 'numpy', 'pandas', 'scipy'},
}

_EXPORTS = {
    'NestedDict': ('dict', 'NestedDict', None),
    'dislocation': ('dislocation', None, 'dislocation'),
    'quantumespresso': ('quantumespresso', None, None),
    'wannier': ('wannier', None, None),
    'epw': ('epw', None, 'epw'),
    'Printer': ('printer', 'Printer', None),
    'ProcessTree': ('base', 'ProcessTree', None),
    'BaseWorkChainAnalyser': ('base', 'BaseWorkChainAnalyser', None),
    'PwBaseWorkChainAnalyser': ('quantumespresso.pw_base', 'PwBaseWorkChainAnalyser', None),
    'PwCalculationAnalyser': ('quantumespresso.pw_calculation', 'PwCalculationAnalyser', None),
    'PwRelaxWorkChainAnalyser': ('quantumespresso.pw_relax', 'PwRelaxWorkChainAnalyser', None),
    'PwBandsWorkChainAnalyser': ('quantumespresso.pw_bands', 'PwBandsWorkChainAnalyser', None),
    'PwBandsGroupData': ('quantumespresso.pw_bands', 'PwBandsGroupData', None),
    'ProjwfcBaseWorkChainAnalyser': ('quantumespresso.projwfc_base', 'ProjwfcBaseWorkChainAnalyser', None),
    'PdosWorkChainAnalyser': ('quantumespresso.pdos', 'PdosWorkChainAnalyser', None),
    'PdosGroupData': ('quantumespresso.pdos', 'PdosGroupData', None),
    'Pw2Wannier90BaseWorkChainAnalyser': ('quantumespresso.pw2wannier90_base', 'Pw2Wannier90BaseWorkChainAnalyser', None),
    'Wannier90WorkChainAnalyser': ('wannier.wannier90', 'Wannier90WorkChainAnalyser', None),
    'Wannier90BaseWorkChainAnalyser': ('wannier.wannier90_base', 'Wannier90BaseWorkChainAnalyser', None),
    'PhBaseWorkChainAnalyser': ('quantumespresso.ph_base', 'PhBaseWorkChainAnalyser', None),
    'Q2rBaseWorkChainAnalyser': ('quantumespresso.q2r_base', 'Q2rBaseWorkChainAnalyser', None),
    'MatdynBaseWorkChainAnalyser': ('quantumespresso.matdyn_base', 'MatdynBaseWorkChainAnalyser', None),
    'LayerRelaxWorkChainAnalyser': ('dislocation.layer_relax', 'LayerRelaxWorkChainAnalyser', None),
    'SFEBaseWorkChainAnalyser': ('dislocation.sfebase', 'SFEBaseWorkChainAnalyser', None),
    'ISFEWorkChainAnalyser': ('dislocation.isfe', 'ISFEWorkChainAnalyser', None),
    'ESFEWorkChainAnalyser': ('dislocation.esfe', 'ESFEWorkChainAnalyser', None),
    'USFEWorkChainAnalyser': ('dislocation.usfe', 'USFEWorkChainAnalyser', None),
    'GSFEWorkChainAnalyser': ('dislocation.gsfe', 'GSFEWorkChainAnalyser', 'dislocation'),
    'GSFEWorkChainAnalyserLatest': ('dislocation.gsfe_latest', 'GSFEWorkChainAnalyserLatest', 'dislocation'),
    'GSFEGroupDataLatest': ('dislocation.gsfe_latest', 'GSFEGroupDataLatest', 'dislocation'),
    'GSFEGroupData': ('dislocation.gsfe', 'GSFEGroupData', 'dislocation'),
    'SurfaceWorkChainAnalyser': ('dislocation.surface', 'SurfaceWorkChainAnalyser', 'dislocation'),
    'SurfaceEnergyData': ('dislocation.surface', 'SurfaceEnergyData', 'dislocation'),
    'ScHubbardWorkChainAnalyser': ('hubbard.sc_hubbard', 'ScHubbardWorkChainAnalyser', 'hubbard'),
    'ScHubbardGroup': ('hubbard.sc_hubbard', 'ScHubbardGroup', 'hubbard'),
    'EpwBaseWorkChainAnalyser': ('epw.epw_base', 'EpwBaseWorkChainAnalyser', None),
    'EpwPrepWorkChainAnalyser': ('epw.epw_prep', 'EpwPrepWorkChainAnalyser', 'epw'),
    'EpwPrepConvergenceData': ('epw.epw_prep', 'EpwPrepConvergenceData', 'epw'),
    'EpwPrepData': ('epw.epw_prep', 'EpwPrepData', 'epw'),
    'SuperConWorkChainAnalyser': ('epw.supercon', 'SuperConWorkChainAnalyser', 'epw'),
    'SuperConData': ('epw.supercon', 'SuperConData', 'epw'),
    'ThermoPwBaseAnalyser': ('thermo_pw', 'ThermoPwBaseAnalyser', 'thermo'),
    'create_structure': ('structure', 'create_structure', 'structure'),
    'deprecated': ('deprecated', None, 'epw'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    """Load exported objects lazily so optional extras stay optional."""
    try:
        module_name, attribute_name, extra = _EXPORTS[name]
    except KeyError as exception:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exception

    try:
        module = import_module(f'.{module_name}', __name__)
    except ModuleNotFoundError as exception:
        missing_module = exception.name.split('.')[0]
        if extra is None or missing_module not in _EXTRA_MODULES.get(extra, set()):
            raise

        raise ModuleNotFoundError(
            f'`aiida_analyser.{name}` requires optional dependencies from the `{extra}` extra. '
            f'Install with `pip install aiida-analyser[{extra}]`.'
        ) from exception

    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
