from types import SimpleNamespace

from aiida_analyser.core.groupdata import BaseGroupData


class DummyAnalyser:
    def __init__(self, node):
        self.is_stable = node.is_stable


class DummyGroupData(BaseGroupData):
    analyser_class = DummyAnalyser
    dataframe_columns = ('Material', 'status')

    def __init__(self):
        super().__init__()
        self._data = [
            {'PK': 2, 'Material': 'Fe2O3', 'status': '✅', 'node': SimpleNamespace(pk=2, is_stable=True)},
            {'PK': 1, 'Material': 'Al2O3', 'status': '✅', 'node': SimpleNamespace(pk=1, is_stable=False)},
        ]


class DumpAnalyser:
    copied_paths = []

    def __init__(self, node):
        self.node = node

    def copy_tree(self, destination):
        self.copied_paths.append(destination)


class DumpGroupData(BaseGroupData):
    analyser_class = DumpAnalyser

    def __init__(self):
        super().__init__()
        self._data = [
            {
                'PK': 12,
                'node': [
                    SimpleNamespace(pk=12, process_label='Wanted'),
                    SimpleNamespace(pk=13, process_label='Wanted'),
                ],
            },
        ]


class PivotGroupData(BaseGroupData):
    def __init__(self):
        super().__init__()
        self._data = [
            {
                'PK': 11,
                'Degauss': 0.005,
                'K_Density': 0.10,
                'Q_Density': 0.5,
                'Type': 'EpwPrepWorkChain',
                'Material': 'mpds-S1612209-RuSbTi',
                'Status': '✅ (40579)',
            },
            {
                'PK': 12,
                'Degauss': 0.005,
                'K_Density': 0.10,
                'Q_Density': 0.5,
                'Type': 'EpwPrepWorkChain',
                'Material': 'mpds-S1612210-RuSbZr',
                'Status': '✅ (40637)',
            },
            {
                'PK': 13,
                'Degauss': 0.005,
                'K_Density': 0.10,
                'Q_Density': '-',
                'Type': 'PwRelaxWorkChain',
                'Material': 'mpds-S1612209-RuSbTi',
                'Status': '✅ (36196)',
            },
        ]


def test_tabular_group_data_filters_formula_and_properties():
    table = DummyGroupData().get_table(formula_contains='fe', property_filter='is_stable')

    assert list(table.index) == [2]
    assert table.loc[2, 'Material'] == 'Fe2O3'


def test_available_columns_includes_node_data():
    assert DummyGroupData().available_columns == ('Material', 'status', 'node')


def test_get_table_can_select_nodes_with_or_without_status():
    nodes_only = DummyGroupData().get_table(values='node')
    nodes_and_status = DummyGroupData().get_table(values=['status', 'node'])

    assert list(nodes_only.columns) == ['node']
    assert nodes_only.loc[1, 'node'] == '1'
    assert list(nodes_and_status.columns) == ['status', 'node']
    assert nodes_and_status.loc[2, 'node'] == '2'


def test_get_table_can_use_nodes_as_pivot_values():
    table = DummyGroupData().get_table(index='status', columns='Material', values='node')

    assert table.loc['✅', 'Fe2O3'] == '2'
    assert table.loc['✅', 'Al2O3'] == '1'


def test_get_table_keeps_status_and_node_pairs_on_separate_lines():
    data = DummyGroupData()
    data._data.append(
        {'PK': 3, 'Material': 'Fe2O3', 'status': '✅', 'node': SimpleNamespace(pk=3, is_stable=True)}
    )

    table = data.get_table(index='status', columns='Material', values=('status', 'node'))

    assert table.loc['✅', 'Fe2O3'].splitlines() == ['✅ (2)', '✅ (3)']
    assert '✅ (2)<br>✅ (3)' in table._repr_html_()


def test_iter_group_nodes_filters_process_labels(monkeypatch):
    nodes = [
        SimpleNamespace(process_label='Wanted'),
        SimpleNamespace(process_label='Ignored'),
    ]
    monkeypatch.setattr('aiida.orm.load_group', lambda _: SimpleNamespace(nodes=nodes))

    data = DummyGroupData()
    data._groups = ['test/group']

    assert list(data.iter_group_nodes('Wanted')) == [nodes[0]]


def test_dump_uses_flattened_nodes(tmp_path):
    DumpAnalyser.copied_paths.clear()
    data = DumpGroupData()
    data.dump(tmp_path)

    assert DumpAnalyser.copied_paths == [tmp_path / '12', tmp_path / '13']


def test_get_table_can_create_a_hierarchical_index():
    table = PivotGroupData().get_table(index=['Degauss', 'K_Density', 'Type'])

    assert table.index.names == ['Degauss', 'K_Density', 'Type']
    assert table.loc[(0.005, 0.10, 'EpwPrepWorkChain'), 'Material'].tolist() == [
        'mpds-S1612209-RuSbTi',
        'mpds-S1612210-RuSbZr',
    ]


def test_get_table_can_pivot_an_arbitrary_key_into_columns():
    table = PivotGroupData().get_table(
        index=['Degauss', 'K_Density', 'Q_Density', 'Type'],
        columns='Material',
    )

    assert table.index.names == ['Degauss', 'K_Density', 'Q_Density', 'Type']
    assert table.loc[(0.005, 0.10, 0.5, 'EpwPrepWorkChain'), 'mpds-S1612209-RuSbTi'] == '✅ (40579)'
    assert table.loc[(0.005, 0.10, 0.5, 'EpwPrepWorkChain'), 'mpds-S1612210-RuSbZr'] == '✅ (40637)'
    assert table.loc[(0.005, 0.10, '-', 'PwRelaxWorkChain'), 'mpds-S1612210-RuSbZr'] == ''
