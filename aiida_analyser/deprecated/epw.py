"""Analysis helpers for the historical ``EpwWorkChain`` workflow.

The archived screening workflow predates :class:`EpwPrepWorkChain`.  Its
top-level children are ``w90_bands``, ``ph_base``, and a direct
``EpwCalculation`` called ``epw``.  This module keeps support for that
layout separate from the current EPW workflow API.
"""

from __future__ import annotations

from typing import Any

from aiida_analyser.core.base import BaseWorkChainAnalyser
from aiida_analyser.epw.convergence import EpwDegaussKQGroup
from aiida_analyser.epw.epw_calculation import EpwAnalyser as EpwCalculationAnalyser
from aiida_analyser.quantumespresso.ph_base import PhBaseAnalyser
from aiida_analyser.wannier.wannier90 import Wannier90Analyser


class EpwAnalyser(BaseWorkChainAnalyser):
    """Analyse a historical ``EpwWorkChain``.

    Use :meth:`get_failure_report` for the complete nested failure tree.  The
    inherited :meth:`get_state` compatibility API returns its first failed
    branch, while this implementation retains the historical direct-child
    path names.
    """

    @property
    def w90_bands(self):
        """The ``Wannier90BandsWorkChain`` child."""
        return self._get_node_from_tree('w90_bands')

    @property
    def ph_base(self):
        """The ``PhBaseWorkChain`` child."""
        return self._get_node_from_tree('ph_base')

    @property
    def epw(self):
        """The direct ``EpwCalculation`` child."""
        return self._get_node_from_tree('epw')



class _FailureAnalysisGroupMixin:
    """Provide a dataframe view over failure reports for EPW workflow groups."""

    def analyse_failures(self, include_outputs: bool = False, include_unfinished: bool = False):
        """Return one exception-tree summary row for every selected workchain.

        ``failure_report`` retains the complete hierarchy.  The scalar columns
        describe the primary (first, call-order) terminal failure, which makes
        the returned dataframe convenient for filtering and aggregation.
        """
        import pandas as pd

        columns = [
            'PK',
            'Material',
            'degauss',
            'kpoints_distance',
            'qpoints_distance',
            'workchain_state',
            'workchain_exit_status',
            'failure_path',
            'failure_process_label',
            'failure_pk',
            'raw_exit_status',
            'raw_exit_message',
            'analysis_exit_status',
            'analysis_exit_label',
            'analysis_exit_message',
            'analysis_evidence',
            'frontier_count',
            'outputs',
            'failure_report',
            'analysis_error',
        ]
        rows = []

        for material, degauss, kpoints_distance, qpoints_distance, node in self._flat_nodes:
            is_terminal_failure = (
                not getattr(node, 'is_finished_ok', False)
                and (
                    getattr(node, 'is_finished', False)
                    or getattr(node, 'is_failed', False)
                    or getattr(node, 'is_excepted', False)
                    or getattr(node, 'is_killed', False)
                )
            )
            if not include_unfinished and not is_terminal_failure:
                continue

            exit_code = getattr(node, 'exit_code', None)
            exit_status = getattr(exit_code, 'status', exit_code)
            if not isinstance(exit_status, int):
                exit_status = getattr(node, 'exit_status', None)
            row = {
                'PK': getattr(node, 'pk', None),
                'Material': material,
                'degauss': degauss,
                'kpoints_distance': kpoints_distance,
                'qpoints_distance': qpoints_distance,
                'workchain_state': getattr(getattr(node, 'process_state', None), 'value', None),
                'workchain_exit_status': exit_status,
                'failure_path': None,
                'failure_process_label': None,
                'failure_pk': None,
                'raw_exit_status': None,
                'raw_exit_message': None,
                'analysis_exit_status': None,
                'analysis_exit_label': None,
                'analysis_exit_message': None,
                'analysis_evidence': None,
                'frontier_count': 0,
                'outputs': None,
                'failure_report': None,
                'analysis_error': None,
            }

            try:
                report = self.analyser_class(node).get_failure_report(
                    include_outputs=include_outputs
                )
                primary = report.primary
                row['failure_report'] = report
                row['frontier_count'] = len(report.frontiers)
                if primary is not None:
                    diagnostic = primary.analysis_exit_code
                    row.update({
                        'failure_path': primary.path,
                        'failure_process_label': primary.process_label,
                        'failure_pk': primary.pk,
                        'raw_exit_status': primary.raw_exit_status,
                        'raw_exit_message': primary.raw_exit_message,
                        'analysis_exit_status': (
                            diagnostic.status if diagnostic is not None else None
                        ),
                        'analysis_exit_label': (
                            diagnostic.label if diagnostic is not None else None
                        ),
                        'analysis_exit_message': (
                            diagnostic.message if diagnostic is not None else None
                        ),
                        'analysis_evidence': (
                            diagnostic.evidence if diagnostic is not None else None
                        ),
                        'outputs': primary.outputs if include_outputs else None,
                    })
            except Exception as exception:
                row['analysis_error'] = str(exception)

            rows.append(row)

        return pd.DataFrame(rows, columns=columns)

    def get_failure_table(self, include_outputs: bool = False, include_unfinished: bool = False):
        """Alias for :meth:`analyse_failures`."""
        return self.analyse_failures(
            include_outputs=include_outputs,
            include_unfinished=include_unfinished,
        )


class EpwGroup(_FailureAnalysisGroupMixin, EpwDegaussKQGroup):
    """Historical ``EpwWorkChain`` nodes arranged on the EPW convergence grid.

    The archive stores grid values in workflow inputs rather than node extras,
    so this class reads both locations.  Newer nodes that have the current
    extras continue to work without a migration.
    """

    analyser_class = EpwAnalyser
    process_label = 'EpwWorkChain'

    @classmethod
    def check_protocol(cls, node):
        """Validate only the process type; this workflow predates extras."""
        if node.process_label != cls.process_label:
            raise ValueError(f'Node<{node.pk}> is not a {cls.process_label}.')

    @classmethod
    def _convergence_extras(cls, node) -> tuple[dict[str, Any], Any, Any, Any]:
        extras = dict(node.base.extras.all)
        _, degauss, kpoints_distance, qpoints_distance = super()._convergence_extras(node)

        try:
            kpoints_distance = node.inputs.kpoints_distance_scf.value
        except (AttributeError, KeyError):
            pass
        try:
            qpoints_distance = node.inputs.qpoints_distance.value
        except (AttributeError, KeyError):
            pass
        try:
            parameters = node.inputs.w90_bands.scf.pw.parameters.get_dict()
            degauss = parameters['SYSTEM']['degauss']
        except (AttributeError, KeyError):
            pass

        return extras, degauss, kpoints_distance, qpoints_distance

    def _material_label(self, node, extras):
        """Prefer the historical ``formula_hill`` extra when available."""
        return extras.get('formula_hill') or super()._material_label(node, extras)

    def _band_node_for_workchain(self, node):
        """The archived workflow's direct EPW calculation owns EPW outputs."""
        return EpwAnalyser(node).epw
