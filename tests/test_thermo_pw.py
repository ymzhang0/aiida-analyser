from types import SimpleNamespace

import pandas as pd

from aiida_analyser.thermo_pw import ThermoPwGroupData


class Extras(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def make_node(pk, process_label='Thermo_pwBaseWorkChain', *, formula='Si', extras=None, finished_ok=True):
    structure = SimpleNamespace(
        get_formula=lambda: formula,
        base=SimpleNamespace(extras=Extras(extras or {})),
    )
    return SimpleNamespace(
        pk=pk,
        process_label=process_label,
        inputs=SimpleNamespace(thermo_pw=SimpleNamespace(structure=structure)),
        base=SimpleNamespace(extras=Extras()),
        is_terminated=True,
        is_finished_ok=finished_ok,
        is_failed=not finished_ok,
        is_excepted=False,
        is_killed=False,
        exit_status=401 if not finished_ok else 0,
    )


def test_get_table(monkeypatch):
    nodes = [
        make_node(3, extras={'source_db': 'mp', 'source_id': '149'}),
        make_node(1, formula='Al', finished_ok=False),
        make_node(2, process_label='UnrelatedWorkChain'),
    ]
    monkeypatch.setattr(
        'aiida_analyser.thermo_pw.thermo_pw.orm.load_group',
        lambda label: SimpleNamespace(nodes=nodes),
    )

    table = ThermoPwGroupData(['thermo/group']).get_table()

    assert list(table.index) == [1, 3]
    assert list(table.columns) == ['Material', 'Source', 'Process', 'Status']
    assert table.loc[3, 'Material'] == 'Si'
    assert table.loc[3, 'Source'] == 'mp-149'
    assert table.loc[3, 'Status'] == '✅'
    assert table.loc[1, 'Status'] == '❌ (401)'


def test_get_table_is_empty_without_groups():
    table = ThermoPwGroupData().get_table()

    assert isinstance(table, pd.DataFrame)
    assert table.empty
    assert list(table.columns) == ['Material', 'Source', 'Process', 'Status']


def test_get_table_rejects_unknown_display_mode():
    try:
        ThermoPwGroupData().get_table(display_mode='unknown')
    except ValueError as exception:
        assert 'display_mode' in str(exception)
    else:
        raise AssertionError('Expected an invalid display mode to raise ValueError')
