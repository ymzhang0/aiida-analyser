from types import SimpleNamespace

import matplotlib.pyplot as plt

from aiida_analyser.epw.epw_prep import EpwPrepGroup
from aiida_analyser.visualization.convergence import KPOINT_DISTANCE_LABEL


def test_epw_prep_group_table_includes_structure_provenance():
    source_one = SimpleNamespace(pk=101, process_label='PwRelaxWorkChain')
    source_two = SimpleNamespace(pk=102, node_type='data.core.structure.StructureData.')
    incoming_links = [
        SimpleNamespace(node=source_one, link_label='output_structure'),
        SimpleNamespace(node=source_two, link_label='result'),
    ]
    structure = SimpleNamespace(
        pk=88,
        base=SimpleNamespace(
            links=SimpleNamespace(
                get_incoming=lambda: SimpleNamespace(all=lambda: incoming_links),
            )
        ),
    )
    node = SimpleNamespace(
        pk=10,
        inputs=SimpleNamespace(structure=structure),
        is_terminated=True,
        is_finished_ok=True,
    )
    group = EpwPrepGroup.__new__(EpwPrepGroup)
    group._flat_nodes = [('Si', 0.01, 0.2, 0.4, node)]
    group._data = group._flatten_data()

    table = group.get_table()

    assert list(table.columns) == list(EpwPrepGroup.dataframe_columns)
    assert table.loc[10, 'structure_PK'] == 88
    assert table.loc[10, 'structure_incoming'].splitlines() == [
        'PwRelaxWorkChain<101> [output_structure]',
        'data.core.structure.StructureData.<102> [result]',
    ]


def test_epw_prep_group_marks_missing_structure_provenance_as_na():
    structure = SimpleNamespace(
        pk=88,
        base=SimpleNamespace(
            links=SimpleNamespace(get_incoming=lambda: (_ for _ in ()).throw(RuntimeError()))
        ),
    )
    node = SimpleNamespace(inputs=SimpleNamespace(structure=structure))

    assert EpwPrepGroup._get_structure_provenance(node) == (88, 'N/A')

def test_check_structure_uses_shared_kpoint_distance_axis():
    def make_structure(pk, cell_length):
        return SimpleNamespace(
            pk=pk,
            uuid=f'structure-{pk}',
            cell_lengths=(cell_length, 5.5, 5.5),
            cell_angles=(90.0, 90.0, 90.0),
            get_formula=lambda: 'Si',
        )

    coarse_structure = make_structure(1, 5.6)
    dense_structure = make_structure(2, 5.5)
    coarse = SimpleNamespace(
        pk=10,
        inputs=SimpleNamespace(structure=coarse_structure),
    )
    dense = SimpleNamespace(
        pk=11,
        inputs=SimpleNamespace(structure=dense_structure),
    )
    group = EpwPrepGroup.__new__(EpwPrepGroup)
    group._flat_nodes = [
        ('Si', 0.01, 0.3, 0.5, coarse),
        ('Si', 0.01, 0.1, 0.5, dense),
    ]

    axis, values = group.check_structure(quantity='a', formula='Si')

    assert axis.get_xscale() == 'function'
    assert axis.get_xlim()[0] > axis.get_xlim()[1]
    assert axis.get_xlabel() == KPOINT_DISTANCE_LABEL
    assert list(axis.lines[0].get_xdata()) == [0.3, 0.1]
    assert list(values[(0.01, 0.5)]) == [0.3, 0.1]
    plt.close(axis.figure)
