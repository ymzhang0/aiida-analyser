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
    'archive_context': ('core.archive', 'archive_context', None),
    'NestedDict': ('core.dict', 'NestedDict', None),
    'move_stashed_folder': ('data', 'move_stashed_folder', None),
    'dislocation': ('dislocation', None, 'dislocation'),
    'quantumespresso': ('quantumespresso', None, None),
    'wannier': ('wannier', None, None),
    'epw': ('epw', None, 'epw'),
    'Printer': ('core.printer', 'Printer', None),
    'print_tree': ('core.printer', 'print_tree', None),
    'display_tree': ('core.printer', 'display_tree', None),
    'in_notebook': ('core.printer', 'in_notebook', None),
    'ProcessTree': ('core.base', 'ProcessTree', None),
    'ProcessReport': ('core.base', 'ProcessReport', None),
    'ProcessReportNode': ('core.base', 'ProcessReportNode', None),
    'FailureReport': ('core.base', 'FailureReport', None),
    'FailureReportNode': ('core.base', 'FailureReportNode', None),
    'create_archive_profile': ('core.archive', 'create_archive_profile', None),
    'count_groups': ('core.groups', 'count_groups', None),
    'count_nodes': ('core.groups', 'count_nodes', None),
    'get_and_count_types': ('core.groups', 'get_and_count_types', None),
    'BaseWorkChainAnalyser': ('core.base', 'BaseWorkChainAnalyser', None),
    'BaseRestartWorkChainAnalyser': ('core.base', 'BaseRestartWorkChainAnalyser', None),
    'BaseRestartAnalyser': ('core.base', 'BaseRestartAnalyser', None),
    'CompareOptions': ('core.compare', 'CompareOptions', None),
    'DiffEntry': ('core.compare', 'DiffEntry', None),
    'NodeDiff': ('core.compare', 'NodeDiff', None),
    'NodeReference': ('core.compare', 'NodeReference', None),
    'compare': ('core.compare', 'compare', None),
    'compare_nodes': ('core.compare', 'compare_nodes', None),
    'PwBaseAnalyser': ('quantumespresso.pw_base', 'PwBaseAnalyser', 'quantumespresso'),
    'PwAnalyser': ('quantumespresso.pw_calculation', 'PwAnalyser', 'quantumespresso'),
    'DosAnalyser': ('quantumespresso.dos_calculation', 'DosAnalyser', 'quantumespresso'),
    'PhAnalyser': ('quantumespresso.ph_calculation', 'PhAnalyser', 'quantumespresso'),
    'MatdynAnalyser': ('quantumespresso.matdyn_calculation', 'MatdynAnalyser', 'quantumespresso'),
    'ProjwfcAnalyser': ('quantumespresso.projwfc_calculation', 'ProjwfcAnalyser', 'quantumespresso'),
    'Pw2Wannier90Analyser': ('quantumespresso.pw2wannier90_calculation', 'Pw2Wannier90Analyser', 'quantumespresso'),
    'Q2rAnalyser': ('quantumespresso.q2r_calculation', 'Q2rAnalyser', 'quantumespresso'),
    'PwRelaxAnalyser': ('quantumespresso.pw_relax', 'PwRelaxAnalyser', 'quantumespresso'),
    'PwRelaxGroup': ('quantumespresso.pw_relax', 'PwRelaxGroup', 'quantumespresso'),
    'PwBandsAnalyser': ('quantumespresso.pw_bands', 'PwBandsAnalyser', 'quantumespresso'),
    'PwBandsGroup': ('quantumespresso.pw_bands', 'PwBandsGroup', 'quantumespresso'),
    'ProjwfcBaseAnalyser': ('quantumespresso.projwfc_base', 'ProjwfcBaseAnalyser', 'quantumespresso'),
    'PdosAnalyser': ('quantumespresso.pdos', 'PdosAnalyser', 'quantumespresso'),
    'PdosGroup': ('quantumespresso.pdos', 'PdosGroup', 'quantumespresso'),
    'Pw2Wannier90BaseAnalyser': ('quantumespresso.pw2wannier90_base', 'Pw2Wannier90BaseAnalyser', 'quantumespresso'),
    'Wannier90Analyser': ('wannier.wannier90', 'Wannier90Analyser', 'wannier'),
    'Wannier90BaseAnalyser': ('wannier.wannier90_base', 'Wannier90BaseAnalyser', 'wannier'),
    'Wannier90CalculationAnalyser': ('wannier.wannier90_calculation', 'Wannier90CalculationAnalyser', 'wannier'),
    'PhBaseAnalyser': ('quantumespresso.ph_base', 'PhBaseAnalyser', 'quantumespresso'),
    'PhBaseGroup': ('quantumespresso.ph_base', 'PhBaseGroup', 'quantumespresso'),
    'Q2rBaseAnalyser': ('quantumespresso.q2r_base', 'Q2rBaseAnalyser', 'quantumespresso'),
    'MatdynBaseAnalyser': ('quantumespresso.matdyn_base', 'MatdynBaseAnalyser', 'quantumespresso'),
    'LayerRelaxAnalyser': ('dislocation.layer_relax', 'LayerRelaxAnalyser', 'dislocation'),
    'SFEBaseAnalyser': ('dislocation.sfebase', 'SFEBaseAnalyser', 'dislocation'),
    'ISFEAnalyser': ('dislocation.isfe', 'ISFEAnalyser', 'dislocation'),
    'ESFEAnalyser': ('dislocation.esfe', 'ESFEAnalyser', 'dislocation'),
    'USFEAnalyser': ('dislocation.usfe', 'USFEAnalyser', 'dislocation'),
    'GSFEAnalyser': ('dislocation.gsfe', 'GSFEAnalyser', 'dislocation'),
    'GSFEAnalyserLatest': ('dislocation.gsfe_latest', 'GSFEAnalyserLatest', 'dislocation'),
    'GSFEGroupDataLatest': ('dislocation.gsfe_latest', 'GSFEGroupDataLatest', 'dislocation'),
    'GSFEGroup': ('dislocation.gsfe', 'GSFEGroup', 'dislocation'),
    'GSFERelaxAnalyser': ('dislocation.gsfe_relax', 'GSFERelaxAnalyser', 'dislocation'),
    'GSFERelaxGroup': ('dislocation.gsfe_relax', 'GSFERelaxGroup', 'dislocation'),
    'SurfaceEnergyAnalyser': ('dislocation.surface', 'SurfaceEnergyAnalyser', 'dislocation'),
    'SurfaceEnergyGroup': ('dislocation.surface', 'SurfaceEnergyGroup', 'dislocation'),
    'ScHubbardAnalyser': ('hubbard.sc_hubbard', 'ScHubbardAnalyser', 'hubbard'),
    'ScHubbardGroup': ('hubbard.sc_hubbard', 'ScHubbardGroup', 'hubbard'),
    'EpwAnalyser': ('epw.epw_calculation', 'EpwAnalyser', 'epw'),
    'EpwBaseAnalyser': ('epw.epw_base', 'EpwBaseAnalyser', 'epw'),
    'EpwBaseGroup': ('epw.epw_base', 'EpwBaseGroup', 'epw'),
    'EpwPrepAnalyser': ('epw.epw_prep', 'EpwPrepAnalyser', 'epw'),
    'EpwPrepConvergenceData': ('epw.epw_prep', 'EpwPrepConvergenceData', 'epw'),
    'EpwPrepGroup': ('epw.epw_prep', 'EpwPrepGroup', 'epw'),
    'SuperConAnalyser': ('epw.supercon', 'SuperConAnalyser', 'epw'),
    'SuperConGroup': ('epw.supercon', 'SuperConGroup', 'epw'),
    'Thermo_pwAnalyser': ('thermo_pw.thermo_pw_calculation', 'Thermo_pwAnalyser', 'thermo'),
    'Thermo_pwBaseAnalyser': ('thermo_pw.thermo_pw_base', 'Thermo_pwBaseAnalyser', 'thermo'),
    'Thermo_pwBaseGroup': ('thermo_pw.thermo_pw_base', 'Thermo_pwBaseGroup', 'thermo'),
    'create_structure': ('materials.structures', 'create_structure', 'structure'),
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
