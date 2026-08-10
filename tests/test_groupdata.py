from types import SimpleNamespace

from aiida_analyser.groupdata import BaseGroupData


class DummyAnalyser:
    def __init__(self, node):
        self.is_stable = node.is_stable


class DummyGroupData(BaseGroupData):
    analyser_class = DummyAnalyser
    dataframe_columns = ('Material', 'status')

    def __init__(self):
        super().__init__()
        self._data = [
            {'PK': 2, 'Material': 'Fe2O3', 'status': '✅', 'node': SimpleNamespace(is_stable=True)},
            {'PK': 1, 'Material': 'Al2O3', 'status': '✅', 'node': SimpleNamespace(is_stable=False)},
        ]


class DumpAnalyser:
    copied_paths = []

    def __init__(self, node):
        self.node = node

    def copy_tree(self, destination):
        self.copied_paths.append(destination)


class DumpGroupData(BaseGroupData):
    analyser_class = DumpAnalyser
    dump_process_labels = 'Wanted'


def test_tabular_group_data_filters_formula_and_properties():
    table = DummyGroupData().get_table(formula_contains='fe', property_filter='is_stable')

    assert list(table.index) == [2]
    assert table.loc[2, 'Material'] == 'Fe2O3'


def test_iter_group_nodes_filters_process_labels(monkeypatch):
    nodes = [
        SimpleNamespace(process_label='Wanted'),
        SimpleNamespace(process_label='Ignored'),
    ]
    monkeypatch.setattr('aiida.orm.load_group', lambda _: SimpleNamespace(nodes=nodes))

    data = DummyGroupData()
    data._groups = ['test/group']

    assert list(data.iter_group_nodes('Wanted')) == [nodes[0]]


def test_dump_uses_declared_analyser_and_process_label(monkeypatch, tmp_path):
    DumpAnalyser.copied_paths.clear()
    wanted = SimpleNamespace(pk=12, process_label='Wanted')
    monkeypatch.setattr(
        'aiida.orm.load_group',
        lambda _: SimpleNamespace(nodes=[wanted, SimpleNamespace(pk=13, process_label='Ignored')]),
    )

    data = DumpGroupData(['test/group'])
    data.dump(tmp_path)

    assert DumpAnalyser.copied_paths == [tmp_path / '12']
