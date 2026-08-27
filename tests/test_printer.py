import sys
from html import unescape
from types import ModuleType

import pytest

from aiida_analyser.core.printer import (
    Printer,
    display_tree,
    in_notebook,
    print_tree,
    render_collapsible_tree,
)


class Dict:
    __module__ = 'aiida.orm.nodes.data.dict'

    def __init__(self, data):
        self.data = data

    def get_dict(self):
        return self.data


class _OrmScalar:
    def __init__(self, value):
        self.value = value


class Int(_OrmScalar):
    __module__ = 'aiida.orm.nodes.data.int'


class Float(_OrmScalar):
    __module__ = 'aiida.orm.nodes.data.float'


class Bool(_OrmScalar):
    __module__ = 'aiida.orm.nodes.data.bool'


class Computer:
    def __init__(self, label):
        self.label = label


class InstalledCode:
    __module__ = 'aiida.orm.nodes.data.code.installed'

    def __init__(self, label, computer_label, pk):
        self.label = label
        self.computer = Computer(computer_label)
        self.pk = pk

    def __repr__(self):
        return '<InstalledCode: full representation containing uuid: unwanted>'


def test_print_renders_text_tree_outside_notebook(monkeypatch):
    output = []
    printer = Printer({'parent': {'value': 1}, 'items': ['a', 'b']})
    monkeypatch.setattr(printer, '_print', output.append)
    monkeypatch.setattr(printer, '_in_notebook', lambda: False)

    printer.print()

    assert output == [
        '├── parent',
        '│   └── value',
        '│       └── 1',
        '└── items',
        '    ├── [0]',
        "    │   └── 'a'",
        '    └── [1]',
        "        └── 'b'",
    ]


def test_print_uses_rich_display_inside_notebook(monkeypatch):
    displayed = []
    display_module = ModuleType('IPython.display')
    display_module.display = displayed.append
    ipython_module = ModuleType('IPython')
    ipython_module.__path__ = []
    ipython_module.display = display_module
    monkeypatch.setitem(sys.modules, 'IPython', ipython_module)
    monkeypatch.setitem(sys.modules, 'IPython.display', display_module)

    printer = Printer({'value': 1})
    monkeypatch.setattr(printer, '_in_notebook', lambda: True)

    printer.print()

    assert displayed == [printer]


def test_orm_scalar_values_are_converted_to_python_values():
    printer = Printer(
        {
            'kpoints_distance_scf': Float(0.1),
            'kpoints_factor_nscf': Int(1),
            'clean_workdir': Bool(True),
        }
    )

    assert printer.data == {
        'kpoints_distance_scf': 0.1,
        'kpoints_factor_nscf': 1,
        'clean_workdir': True,
    }
    rendered = unescape(printer._repr_html_())
    assert '<Float:' not in rendered
    assert '<Int:' not in rendered
    assert '<Bool:' not in rendered


def test_installed_code_only_displays_remote_label_computer_and_pk(monkeypatch):
    output = []
    code = InstalledCode('qe-760-epw', 'lumi_async', 102109)
    printer = Printer({'code': code})
    monkeypatch.setattr(printer, '_print', output.append)
    monkeypatch.setattr(printer, '_in_notebook', lambda: False)

    printer.print()

    expected = "Remote code 'qe-760-epw' on lumi_async, pk: 102109"
    assert output == ['└── code', f'    └── {expected}']
    rendered = unescape(printer._repr_html_())
    assert expected in rendered
    assert 'InstalledCode' not in rendered
    assert 'uuid' not in rendered


def test_orm_dict_is_recursively_converted_to_plain_dict():
    printer = Printer({'parameters': Dict({'CONTROL': {'calculation': 'scf'}})})

    assert printer.data == {'parameters': {'CONTROL': {'calculation': 'scf'}}}
    assert type(printer.data['parameters']) is dict


def test_root_orm_dict_is_accepted_and_converted():
    printer = Printer(Dict({'CONTROL': {'calculation': 'scf'}}))

    assert printer.data == {'CONTROL': {'calculation': 'scf'}}
    assert type(printer.data) is dict


def test_html_representation_is_collapsible_complete_and_escaped():
    printer = Printer({'parameters': Dict({'CONTROL': {'label': '<scf>'}})})

    rendered = printer._repr_html_()

    assert rendered.count('<details') == 2
    assert rendered.count('<details open>') == 1
    assert 'dict · 1 item' in unescape(rendered)
    assert '&lt;scf&gt;' in rendered
    assert "'<scf>'" in unescape(rendered)


def test_empty_html_representation_preserves_container_type():
    assert Printer({})._repr_html_().endswith('>{}</pre>')
    assert Printer([])._repr_html_().endswith('>[]</pre>')


def test_rejects_scalar_input():
    with pytest.raises(TypeError, match='Input data must be a map or iterable'):
        Printer('not a collection')


def test_in_notebook_detection(monkeypatch):
    monkeypatch.setitem(sys.modules, 'IPython', None)
    assert in_notebook() is False

    mock_shell = ModuleType('MockShell')
    mock_shell.kernel = object()
    mock_ipython = ModuleType('IPython')
    mock_ipython.get_ipython = lambda: mock_shell
    monkeypatch.setitem(sys.modules, 'IPython', mock_ipython)
    assert in_notebook() is True


def test_render_collapsible_tree_wraps_content_and_style():
    html = render_collapsible_tree('<li>item</li>', root_class='my-tree', key_class='my-key')
    assert '<div class="my-tree">' in html
    assert '.my-tree ul {' in html
    assert '<ul><li>item</li></ul>' in html


def test_print_tree_function_with_dict(monkeypatch):
    output = []
    monkeypatch.setattr(Printer, '_in_notebook', staticmethod(lambda: False))
    monkeypatch.setattr(Printer, '_print', staticmethod(output.append))

    print_tree({'a': {'b': 1}})

    assert output == [
        '└── a',
        '    └── b',
        '        └── 1',
    ]


def test_print_tree_function_in_notebook(monkeypatch):
    displayed = []
    display_module = ModuleType('IPython.display')
    display_module.display = displayed.append
    ipython_module = ModuleType('IPython')
    ipython_module.__path__ = []
    ipython_module.display = display_module
    monkeypatch.setitem(sys.modules, 'IPython', ipython_module)
    monkeypatch.setitem(sys.modules, 'IPython.display', display_module)
    monkeypatch.setattr(Printer, '_in_notebook', staticmethod(lambda: True))

    print_tree({'key': 'val'})

    assert len(displayed) == 1
    assert isinstance(displayed[0], Printer)
    assert displayed[0].data == {'key': 'val'}


def test_print_tree_delegates_to_object_with_print_tree():
    calls = []

    class MockTree:
        def print_tree(self, **kwargs):
            calls.append(kwargs)

    tree = MockTree()
    print_tree(tree, prefix='  ', is_last=False)

    assert calls == [{'prefix': '  ', 'is_last': False}]


def test_display_tree_alias():
    assert display_tree is print_tree

