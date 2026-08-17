"""Resolve specialised analysers from persisted AiiDA process metadata.

``process_label`` is a readable class name and has changed in historical
workflows.  ``process_type`` is the preferred, machine-readable identifier;
labels remain as a compatibility fallback for old databases and local
workflows that do not expose a stable entry point.
"""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class AnalyserSpec:
    """One lazily imported analyser registration."""

    target: str
    process_types: tuple[str, ...] = ()
    process_labels: tuple[str, ...] = ()

    def load(self):
        module_name, class_name = self.target.rsplit(':', 1)
        return getattr(import_module(module_name), class_name)


class AnalyserRegistry:
    """Map persisted process metadata to analyser classes.

    Registrations intentionally use import strings: importing ``core.base``
    must not eagerly import every optional workflow package.
    """

    def __init__(self):
        self._by_process_type = {}
        self._by_process_label = {}

    def register(self, spec):
        for process_type in spec.process_types:
            self._by_process_type[process_type] = spec
        for process_label in spec.process_labels:
            self._by_process_label[process_label] = spec

    def resolve(self, node):
        """Return the registered analyser class for *node*, or ``None``.

        Unregistered nodes deliberately return ``None``.  Callers must skip
        them rather than recursively extracting an unknown workflow tree.
        """
        process_type = getattr(node, 'process_type', None)
        process_label = getattr(node, 'process_label', None)
        spec = self._by_process_type.get(process_type)
        if spec is None:
            spec = self._by_process_label.get(process_label)
        return None if spec is None else spec.load()


registry = AnalyserRegistry()


def _register(target, *, process_types=(), process_labels=()):
    registry.register(AnalyserSpec(
        target=target,
        process_types=tuple(process_types),
        process_labels=tuple(process_labels),
    ))


def register_analyser(target, *, process_types=(), process_labels=()):
    """Register an analyser import path for project-specific workflow types.

    ``target`` uses ``"package.module:ClassName"`` so registration does not
    eagerly import optional workflow dependencies.
    """
    _register(target, process_types=process_types, process_labels=process_labels)


# Quantum ESPRESSO workflow entry points are stable across supported AiiDA QE
# releases.  The process-label fallback keeps databases produced by older
# analyser releases usable.
_register(
    'aiida_analyser.quantumespresso.pw_calculation:PwAnalyser',
    process_types=('aiida.calculations:quantumespresso.pw',),
    process_labels=('PwCalculation',),
)
_register(
    'aiida_analyser.quantumespresso.pw_base:PwBaseAnalyser',
    process_types=('aiida.workflows:quantumespresso.pw.base',),
    process_labels=('PwBaseWorkChain',),
)
_register(
    'aiida_analyser.quantumespresso.pw_relax:PwRelaxAnalyser',
    process_labels=('PwRelaxWorkChain',),
)
_register(
    'aiida_analyser.quantumespresso.pw_bands:PwBandsAnalyser',
    process_labels=('PwBandsWorkChain',),
)
_register(
    'aiida_analyser.quantumespresso.ph_calculation:PhAnalyser',
    process_types=('aiida.calculations:quantumespresso.ph',),
    process_labels=('PhCalculation',),
)
_register(
    'aiida_analyser.quantumespresso.ph_base:PhBaseAnalyser',
    process_types=('aiida.workflows:quantumespresso.ph.base',),
    process_labels=('PhBaseWorkChain',),
)
_register(
    'aiida_analyser.quantumespresso.pw2wannier90_calculation:Pw2Wannier90Analyser',
    process_types=('aiida.calculations:quantumespresso.pw2wannier90',),
    process_labels=('Pw2wannier90Calculation',),
)
_register(
    'aiida_analyser.quantumespresso.pw2wannier90_base:Pw2Wannier90BaseAnalyser',
    process_labels=('Pw2Wannier90BaseWorkChain',),
)
_register(
    'aiida_analyser.quantumespresso.projwfc_calculation:ProjwfcAnalyser',
    process_types=('aiida.calculations:quantumespresso.projwfc',),
    process_labels=('ProjwfcCalculation',),
)
_register(
    'aiida_analyser.quantumespresso.projwfc_base:ProjwfcBaseAnalyser',
    process_labels=('ProjwfcBaseWorkChain',),
)
_register(
    'aiida_analyser.quantumespresso.matdyn_calculation:MatdynAnalyser',
    process_types=('aiida.calculations:quantumespresso.matdyn',),
    process_labels=('MatdynCalculation',),
)
_register(
    'aiida_analyser.quantumespresso.matdyn_base:MatdynBaseAnalyser',
    process_labels=('MatdynBaseWorkChain',),
)
_register(
    'aiida_analyser.quantumespresso.q2r_calculation:Q2rAnalyser',
    process_types=('aiida.calculations:quantumespresso.q2r',),
    process_labels=('Q2rCalculation',),
)
_register(
    'aiida_analyser.quantumespresso.q2r_base:Q2rBaseAnalyser',
    process_labels=('Q2rBaseWorkChain',),
)

# Project and optional-plugin workflows currently expose stable labels only.
_register('aiida_analyser.epw.epw_calculation:EpwAnalyser', process_labels=('EpwCalculation',))
_register('aiida_analyser.epw.epw_base:EpwBaseAnalyser', process_labels=('EpwBaseWorkChain',))
_register('aiida_analyser.epw.epw_prep:EpwPrepAnalyser', process_labels=('EpwPrepWorkChain',))
_register('aiida_analyser.epw.supercon:SuperConAnalyser', process_labels=('SuperConWorkChain',))
_register('aiida_analyser.wannier.wannier90:Wannier90Analyser', process_labels=(
    'Wannier90BandsWorkChain', 'Wannier90OptimizeWorkChain',
))
_register('aiida_analyser.wannier.wannier90_base:Wannier90BaseAnalyser', process_labels=(
    'Wannier90BaseWorkChain',
))
_register('aiida_analyser.thermo_pw.thermo_pw_calculation:Thermo_pwAnalyser', process_labels=(
    'Thermo_pwCalculation',
))
_register('aiida_analyser.thermo_pw.thermo_pw_base:Thermo_pwBaseAnalyser', process_labels=(
    'Thermo_pwBaseWorkChain',
))


def resolve_analyser(node):
    """Resolve the specialised analyser class registered for an AiiDA node."""
    return registry.resolve(node)
