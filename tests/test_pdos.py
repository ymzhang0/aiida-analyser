from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aiida_analyser.quantumespresso.pdos import PdosAnalyser, PdosGroup


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


def test_pdos_group_uses_the_standard_flat_table_schema():
    node = make_node(11)
    group = PdosGroup.__new__(PdosGroup)
    group._nested_data = {
        'Si': {
            0.01: {
                0.15: [node],
            },
        },
    }
    group._data = group._flatten_data()

    assert group.available_columns == (
        'Material', 'degauss', 'kpoints_distance', 'with_soc', 'with_hubbard_u', 'status', 'node',
    )
    assert group._data == [{
        'PK': 11,
        'Material': 'Si',
        'degauss': 0.01,
        'kpoints_distance': 0.15,
        'with_soc': False,
        'with_hubbard_u': False,
        'status': '✅',
        'node': node,
    }]

    table = group.get_table(
        index=['degauss', 'kpoints_distance'],
        columns='Material',
        values=('node', 'status'),
    )

    assert table.loc[(0.01, 0.15), 'Si'] == '(11) ✅'


def test_pdos_group_reads_convergence_and_setting_extras(monkeypatch):
    node = make_node(
        11,
        formula='Si',
        degauss=0.02,
        kpoints_distance_scf=0.15,
        with_soc=True,
        with_hubbard_u=True,
    )
    node.process_label = 'PdosWorkChain'
    monkeypatch.setattr('aiida.orm.load_group', lambda _: SimpleNamespace(nodes=[node]))

    group = PdosGroup(['pdos'])

    assert group._data[0]['Material'] == 'Si'
    assert group._data[0]['with_soc'] is True
    assert group._data[0]['with_hubbard_u'] is True


def test_pdos_group_selects_latest_finished_parameter_combinations():
    first = make_node(1)
    latest = make_node(2)
    failed = make_node(3)
    failed.is_finished_ok = False
    group = PdosGroup.__new__(PdosGroup)
    group._nested_data = {
        'Si': {
            0.01: {0.15: [(first, False, False), (latest, False, False), (failed, True, True)]},
            0.02: {0.15: [(make_node(4), True, False)]},
        },
        'Ge': {0.01: {0.15: [(make_node(5), False, True)]}},
    }

    comparisons = list(group._iter_pdos_comparisons(
        formula='Si', degausses=[0.01], kpoints_distances=[0.15],
        with_soc=False, with_hubbard_u=False,
    ))

    assert comparisons == [('Si', 0.01, 0.15, False, False, latest)]


def test_pdos_group_dump_uses_the_base_progress_implementation(tmp_path):
    node = make_node(11)
    group = PdosGroup.__new__(PdosGroup)
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


@pytest.mark.parametrize('figsize', [None, (5.5, 2.5)])
def test_pdos_analyser_uses_dos_default_and_accepts_custom_figsize(figsize):
    output_dos = SimpleNamespace(
        get_array=lambda name: {
            'x_array': np.array([0.0, 1.0]),
            'y_array_1': np.array([0.0, 2.0]),
        }[name],
    )
    scf = SimpleNamespace(outputs=SimpleNamespace(output_parameters={'fermi_energy': 0.0}))
    outgoing = SimpleNamespace(first=lambda: SimpleNamespace(node=scf))
    node = SimpleNamespace(
        base=SimpleNamespace(links=SimpleNamespace(get_outgoing=lambda **kwargs: outgoing)),
        outputs=SimpleNamespace(dos=SimpleNamespace(output_dos=output_dos)),
    )

    plt.close('all')
    PdosAnalyser(node).plot_pdos(figsize=figsize)

    expected = (3.2, 4.8) if figsize is None else figsize
    assert tuple(plt.gcf().get_size_inches()) == pytest.approx(expected)
    plt.close(plt.gcf())


@pytest.mark.parametrize('figsize', [None, (5.5, 2.5)])
def test_pdos_group_uses_dos_default_and_accepts_custom_figsize(monkeypatch, figsize):
    node = make_node(11)
    group = PdosGroup.__new__(PdosGroup)
    group._nested_data = {'Si': {0.01: {0.15: [node]}}}
    monkeypatch.setattr(PdosAnalyser, 'plot_pdos', lambda self, axis=None, **kwargs: None)

    axs = group.plot_pdos(figsize=figsize, legend=False)

    expected = (3.2, 4.8) if figsize is None else figsize
    assert tuple(axs.flat[0].figure.get_size_inches()) == pytest.approx(expected)
    plt.close(axs.flat[0].figure)
