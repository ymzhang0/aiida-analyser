"""Shared analyser abstractions and utilities."""

from .archive import archive_context, create_archive_profile
from .base import (
    BaseCalculationAnalyser,
    BaseRestartAnalyser,
    BaseRestartWorkChainAnalyser,
    BaseWorkChainAnalyser,
    ProcessTree,
)
from .analyser_registry import UnregisteredProcessError, register_analyser, resolve_analyser
from .compare import CompareOptions, DiffEntry, NodeDiff, NodeReference, compare, compare_nodes
from .dict import NestedDict
from .groupdata import BaseGroupData, DegaussKGroup, DegaussKQGroup, render_process_node_details
from .printer import Printer, display_tree, in_notebook, print_tree, render_collapsible_tree
from .groups import count_groups, count_nodes, get_and_count_types

__all__ = [
    'BaseCalculationAnalyser',
    'UnregisteredProcessError',
    'archive_context',
    'count_groups',
    'count_nodes',
    'get_and_count_types',
    'BaseGroupData',
    'DegaussKGroup',
    'DegaussKQGroup',
    'BaseRestartAnalyser',
    'BaseRestartWorkChainAnalyser',
    'BaseWorkChainAnalyser',
    'CompareOptions',
    'DiffEntry',
    'NestedDict',
    'NodeDiff',
    'NodeReference',
    'ProcessTree',
    'Printer',
    'print_tree',
    'display_tree',
    'in_notebook',
    'render_collapsible_tree',
    'render_process_node_details',
    'compare',
    'compare_nodes',
    'create_archive_profile',
    'register_analyser',
    'resolve_analyser',
]
