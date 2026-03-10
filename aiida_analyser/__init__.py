"""AiiDA analysis helpers for AiiDA work chains."""

from importlib import import_module

__version__ = "0.1.0"

_EXTRA_MODULES = {
    'structure': {'ase'},
    'thermo': {'matplotlib', 'numpy'},
    'dislocation': {'aiida_dislocation', 'ase', 'matplotlib', 'numpy', 'pandas', 'scipy'},
    'epw': {'aiida_epw', 'ase', 'matplotlib', 'numpy', 'pandas', 'scipy'},
}

_EXPORTS = {
    'NestedDict': ('dict', 'NestedDict', None),
    'Printer': ('printer', 'Printer', None),
    'ProcessTree': ('base', 'ProcessTree', None),
    'BaseWorkChainAnalyser': ('base', 'BaseWorkChainAnalyser', None),
    'PwBaseWorkChainAnalyser': ('pw_base', 'PwBaseWorkChainAnalyser', None),
    'PwRelaxWorkChainAnalyser': ('pw_relax', 'PwRelaxWorkChainAnalyser', None),
    'PwBandsWorkChainAnalyser': ('pw_bands', 'PwBandsWorkChainAnalyser', None),
    'ProjwfcBaseWorkChainAnalyser': ('projwfc_base', 'ProjwfcBaseWorkChainAnalyser', None),
    'Pw2Wannier90BaseWorkChainAnalyser': ('pw2wannier90_base', 'Pw2Wannier90BaseWorkChainAnalyser', None),
    'Wannier90WorkChainAnalyser': ('wannier90', 'Wannier90WorkChainAnalyser', None),
    'Wannier90BaseWorkChainAnalyser': ('wannier90_base', 'Wannier90BaseWorkChainAnalyser', None),
    'PhBaseWorkChainAnalyser': ('ph_base', 'PhBaseWorkChainAnalyser', None),
    'Q2rBaseWorkChainAnalyser': ('q2r_base', 'Q2rBaseWorkChainAnalyser', None),
    'MatdynBaseWorkChainAnalyser': ('matdyn_base', 'MatdynBaseWorkChainAnalyser', None),
    'LayerRelaxWorkChainAnalyser': ('layer_relax', 'LayerRelaxWorkChainAnalyser', None),
    'SFEBaseWorkChainAnalyser': ('sfebase', 'SFEBaseWorkChainAnalyser', None),
    'ISFEWorkChainAnalyser': ('isfe', 'ISFEWorkChainAnalyser', None),
    'ESFEWorkChainAnalyser': ('esfe', 'ESFEWorkChainAnalyser', None),
    'USFEWorkChainAnalyser': ('usfe', 'USFEWorkChainAnalyser', None),
    'GSFEWorkChainAnalyser': ('gsfe', 'GSFEWorkChainAnalyser', 'dislocation'),
    'GSFEGroupData': ('gsfe', 'GSFEGroupData', 'dislocation'),
    'SurfaceWorkChainAnalyser': ('surface', 'SurfaceWorkChainAnalyser', 'dislocation'),
    'SurfaceEnergyData': ('surface', 'SurfaceEnergyData', 'dislocation'),
    'EpwBaseWorkChainAnalyser': ('epw_base', 'EpwBaseWorkChainAnalyser', None),
    'EpwPrepWorkChainAnalyser': ('epw_prep', 'EpwPrepWorkChainAnalyser', 'epw'),
    'EpwPrepConvergenceData': ('epw_prep', 'EpwPrepConvergenceData', 'epw'),
    'EpwPrepData': ('epw_prep', 'EpwPrepData', 'epw'),
    'SuperConWorkChainAnalyser': ('supercon', 'SuperConWorkChainAnalyser', 'epw'),
    'SuperConData': ('supercon', 'SuperConData', 'epw'),
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
