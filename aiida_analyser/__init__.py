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
    'move_stashed_folder': ('data', 'move_stashed_folder', None),
    'dislocation': ('dislocation', None, 'dislocation'),
    'quantumespresso': ('quantumespresso', None, None),
    'wannier': ('wannier', None, None),
    'epw': ('epw', None, 'epw'),
    'Printer': ('printer', 'Printer', None),
    'ProcessTree': ('base', 'ProcessTree', None),
    'BaseWorkChainAnalyser': ('base', 'BaseWorkChainAnalyser', None),
    'PwBaseWorkChainAnalyser': ('quantumespresso.pw_base', 'PwBaseWorkChainAnalyser', 'quantumespresso'),
    'PwCalculationAnalyser': ('quantumespresso.pw_calculation', 'PwCalculationAnalyser', 'quantumespresso'),
    'DosCalculationAnalyser': ('quantumespresso.dos_calculation', 'DosCalculationAnalyser', 'quantumespresso'),
    'PhCalculationAnalyser': ('quantumespresso.ph_calculation', 'PhCalculationAnalyser', 'quantumespresso'),
    'MatdynCalculationAnalyser': ('quantumespresso.matdyn_calculation', 'MatdynCalculationAnalyser', 'quantumespresso'),
    'ProjwfcCalculationAnalyser': ('quantumespresso.projwfc_calculation', 'ProjwfcCalculationAnalyser', 'quantumespresso'),
    'Pw2Wannier90CalculationAnalyser': ('quantumespresso.pw2wannier90_calculation', 'Pw2Wannier90CalculationAnalyser', 'quantumespresso'),
    'Q2rCalculationAnalyser': ('quantumespresso.q2r_calculation', 'Q2rCalculationAnalyser', 'quantumespresso'),
    'PwRelaxWorkChainAnalyser': ('quantumespresso.pw_relax', 'PwRelaxWorkChainAnalyser', 'quantumespresso'),
    'PwRelaxWorkChainData': ('quantumespresso.pw_relax', 'PwRelaxWorkChainData', 'quantumespresso'),
    'PwBandsWorkChainAnalyser': ('quantumespresso.pw_bands', 'PwBandsWorkChainAnalyser', 'quantumespresso'),
    'PwBandsGroupData': ('quantumespresso.pw_bands', 'PwBandsGroupData', 'quantumespresso'),
    'ProjwfcBaseWorkChainAnalyser': ('quantumespresso.projwfc_base', 'ProjwfcBaseWorkChainAnalyser', 'quantumespresso'),
    'PdosWorkChainAnalyser': ('quantumespresso.pdos', 'PdosWorkChainAnalyser', 'quantumespresso'),
    'PdosGroupData': ('quantumespresso.pdos', 'PdosGroupData', 'quantumespresso'),
    'Pw2Wannier90BaseWorkChainAnalyser': ('quantumespresso.pw2wannier90_base', 'Pw2Wannier90BaseWorkChainAnalyser', 'quantumespresso'),
    'Wannier90WorkChainAnalyser': ('wannier.wannier90', 'Wannier90WorkChainAnalyser', 'wannier'),
    'Wannier90BaseWorkChainAnalyser': ('wannier.wannier90_base', 'Wannier90BaseWorkChainAnalyser', 'wannier'),
    'PhBaseWorkChainAnalyser': ('quantumespresso.ph_base', 'PhBaseWorkChainAnalyser', 'quantumespresso'),
    'PhData': ('quantumespresso.ph_base', 'PhData', 'quantumespresso'),
    'Q2rBaseWorkChainAnalyser': ('quantumespresso.q2r_base', 'Q2rBaseWorkChainAnalyser', 'quantumespresso'),
    'MatdynBaseWorkChainAnalyser': ('quantumespresso.matdyn_base', 'MatdynBaseWorkChainAnalyser', 'quantumespresso'),
    'LayerRelaxWorkChainAnalyser': ('dislocation.layer_relax', 'LayerRelaxWorkChainAnalyser', 'dislocation'),
    'SFEBaseWorkChainAnalyser': ('dislocation.sfebase', 'SFEBaseWorkChainAnalyser', 'dislocation'),
    'ISFEWorkChainAnalyser': ('dislocation.isfe', 'ISFEWorkChainAnalyser', 'dislocation'),
    'ESFEWorkChainAnalyser': ('dislocation.esfe', 'ESFEWorkChainAnalyser', 'dislocation'),
    'USFEWorkChainAnalyser': ('dislocation.usfe', 'USFEWorkChainAnalyser', 'dislocation'),
    'GSFEWorkChainAnalyser': ('dislocation.gsfe', 'GSFEWorkChainAnalyser', 'dislocation'),
    'GSFEWorkChainAnalyserLatest': ('dislocation.gsfe_latest', 'GSFEWorkChainAnalyserLatest', 'dislocation'),
    'GSFEGroupDataLatest': ('dislocation.gsfe_latest', 'GSFEGroupDataLatest', 'dislocation'),
    'GSFEGroupData': ('dislocation.gsfe', 'GSFEGroupData', 'dislocation'),
    'GSFERelaxWorkChainAnalyser': ('dislocation.gsfe_relax', 'GSFERelaxWorkChainAnalyser', 'dislocation'),
    'GSFERelaxGroupData': ('dislocation.gsfe_relax', 'GSFERelaxGroupData', 'dislocation'),
    'SurfaceWorkChainAnalyser': ('dislocation.surface', 'SurfaceWorkChainAnalyser', 'dislocation'),
    'SurfaceEnergyData': ('dislocation.surface', 'SurfaceEnergyData', 'dislocation'),
    'ScHubbardWorkChainAnalyser': ('hubbard.sc_hubbard', 'ScHubbardWorkChainAnalyser', 'hubbard'),
    'ScHubbardGroup': ('hubbard.sc_hubbard', 'ScHubbardGroup', 'hubbard'),
    'EpwCalculationAnalyser': ('epw.epw_calculation', 'EpwCalculationAnalyser', 'epw'),
    'EpwBaseWorkChainAnalyser': ('epw.epw_base', 'EpwBaseWorkChainAnalyser', 'epw'),
    'EpwData': ('epw.epw_base', 'EpwData', 'epw'),
    'EpwPrepWorkChainAnalyser': ('epw.epw_prep', 'EpwPrepWorkChainAnalyser', 'epw'),
    'EpwPrepConvergenceData': ('epw.epw_prep', 'EpwPrepConvergenceData', 'epw'),
    'EpwPrepData': ('epw.epw_prep', 'EpwPrepData', 'epw'),
    'SuperConWorkChainAnalyser': ('epw.supercon', 'SuperConWorkChainAnalyser', 'epw'),
    'SuperConData': ('epw.supercon', 'SuperConData', 'epw'),
    'ThermoPwCalculationAnalyser': ('thermo_pw.thermo_pw_calculation', 'ThermoPwCalculationAnalyser', 'thermo'),
    'ThermoPwBaseAnalyser': ('thermo_pw.thermo_pw_base', 'ThermoPwBaseAnalyser', 'thermo'),
    'ThermoPwGroupData': ('thermo_pw.thermo_pw_base', 'ThermoPwGroupData', 'thermo'),
    'create_structure': ('structure', 'create_structure', 'structure'),
    'deprecated': ('deprecated', None, 'epw'),
    'read_labelinfo': ('wannier.io', 'read_labelinfo', None),
    'load_bandsdata': ('wannier.io', 'load_bandsdata', None),
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
