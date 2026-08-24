from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from aiida_analyser.quantumespresso.pw_bands import PwBandsGroup
from aiida_analyser.quantumespresso.pw_relax import PwRelaxGroup
from aiida_analyser.visualization.convergence import KPOINT_DISTANCE_LABEL


def make_node(pk, **extras):
    return SimpleNamespace(
        pk=pk,
        base=SimpleNamespace(extras=SimpleNamespace(all=extras)),
        is_terminated=True,
        is_finished_ok=True,
        is_failed=False,
        is_excepted=False,
        is_killed=False,
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


def test_pw_bands_group_uses_the_standard_flat_table_schema():
    node = make_node(11)
    group = PwBandsGroup.__new__(PwBandsGroup)
    group._nested_data = {
        'Si': {
            0.01: {
                0.15: [(node, True)],
            },
        },
    }
    group._data = group._flatten_data()

    assert group.available_columns == (
        'Material', 'degauss', 'kpoints_distance', 'with_soc', 'status', 'node'
    )
    assert group._data == [{
        'PK': 11,
        'Material': 'Si',
        'degauss': 0.01,
        'kpoints_distance': 0.15,
        'with_soc': True,
        'status': '✅',
        'node': node,
    }]

    table = group.get_table(
        index=['degauss', 'kpoints_distance'],
        columns='Material',
        values=('node', 'status'),
    )

    assert table.loc[(0.01, 0.15), 'Si'] == '(11) ✅'


def test_pw_bands_group_selects_latest_finished_parameter_combinations():
    first = make_node(1)
    latest = make_node(2)
    failed = make_node(3)
    failed.is_finished_ok = False
    group = PwBandsGroup.__new__(PwBandsGroup)
    group._nested_data = {
        'Si': {
            0.01: {0.15: [(first, False), (latest, False), (failed, True)]},
            0.02: {0.15: [(make_node(4), True)]},
        },
        'Ge': {0.01: {0.15: [(make_node(5), False)]}},
    }

    comparisons = list(group._iter_band_comparisons(
        formula='Si', degausses=[0.01], kpoints_distances=[0.15], with_soc=False,
    ))

    assert comparisons == [('Si', 0.01, 0.15, False, latest)]


def test_pw_bands_group_treats_missing_soc_extra_as_false(monkeypatch):
    node = make_node(11, formula='Si', degauss=0.02, kpoints_distance_scf=0.15)
    node.process_label = 'PwBandsWorkChain'
    monkeypatch.setattr('aiida.orm.load_group', lambda _: SimpleNamespace(nodes=[node]))

    group = PwBandsGroup(['bands'])

    assert group._data[0]['with_soc'] is False


def test_pw_bands_group_accepts_legacy_unknown_soc_as_non_soc():
    node = make_node(11)
    group = PwBandsGroup.__new__(PwBandsGroup)
    group._nested_data = {'Si': {0.02: {0.15: [(node, 'unknown')]}}}

    comparisons = list(group._iter_band_comparisons(with_soc=False))

    assert comparisons == [('Si', 0.02, 0.15, False, node)]


def test_pw_bands_group_dump_uses_the_base_progress_implementation(tmp_path):
    node = make_node(11)
    group = PwBandsGroup.__new__(PwBandsGroup)
    group._data = [{'node': node}]
    captured = {}

    def capture_dump(destination, nodes, analyser_class=None, path_builder=None, progress=True):
        captured['destination'] = destination
        captured['nodes'] = list(nodes)
        captured['progress'] = progress

    group._dump_nodes = capture_dump
    group.dump(tmp_path, progress=False)

    assert captured == {
        'destination': tmp_path,
        'nodes': [node],
        'progress': False,
    }


def test_plot_structure_convergence_uses_descending_cubic_kpoint_axis():
    def relaxed_node(pk, distance, cell_length):
        node = make_node(pk, formula='Si', degauss=0.01, kpoints_distance=distance)
        node.outputs = SimpleNamespace(
            output_structure=SimpleNamespace(
                cell_lengths=(cell_length, 5.5, 5.5),
                cell_angles=(90.0, 90.0, 90.0),
            ),
        )
        return node

    coarse = relaxed_node(1, 0.3, 5.6)
    medium = relaxed_node(2, 0.2, 5.55)
    dense = relaxed_node(3, 0.1, 5.5)
    group = PwRelaxGroup.__new__(PwRelaxGroup)
    group._nested_data = {
        'Si': {
            0.01: {
                0.3: [coarse],
                0.2: [medium],
                0.1: [dense],
            },
        },
    }

    axis, values = group.plot_structure_convergence()

    assert axis.get_xscale() == 'function'
    assert axis.get_xlim()[0] > axis.get_xlim()[1]
    assert axis.get_xlabel() == KPOINT_DISTANCE_LABEL
