from types import SimpleNamespace

import pytest

from aiida_analyser.quantumespresso.pw_relax import PwRelaxGroup


def make_node(pk, **extras):
    return SimpleNamespace(
        pk=pk,
        base=SimpleNamespace(extras=SimpleNamespace(all=extras)),
    )


def test_pw_relax_group_selects_nodes_by_extras():
    silicon = make_node(1, formula='Si', degauss=0.01, kpoints_distance=0.15)
    germanium = make_node(2, formula='Ge', degauss=0.01, kpoints_distance=0.15)
    group = PwRelaxGroup.__new__(PwRelaxGroup)
    group._data = [{'node': silicon}, {'node': germanium}]

    selected = group.select_nodes_by_extras(formula='Si', degauss=0.01)

    assert selected == [silicon]
    assert group.select_node_by_extras(formula='Ge') is germanium


def test_pw_relax_group_requires_one_node_for_singular_selection():
    first = make_node(1, formula='Si')
    second = make_node(2, formula='Si')
    group = PwRelaxGroup.__new__(PwRelaxGroup)
    group._data = [{'node': first}, {'node': second}]

    with pytest.raises(ValueError, match='found 2'):
        group.select_node_by_extras(formula='Si')
