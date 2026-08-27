import sys
from html import unescape
from types import ModuleType, SimpleNamespace

from aiida_analyser.core.base import ProcessTree
from aiida_analyser.core.printer import print_tree


def _tree(name='ROOT', process_label='PhBaseWorkChain', pk=1, state='finished', children=None):
    tree = ProcessTree.__new__(ProcessTree)
    tree.name = name
    tree.node = SimpleNamespace(
        process_label=process_label,
        pk=pk,
        is_finished_ok=state == 'finished_ok',
        is_terminated=state in {'finished_ok', 'failed'},
        process_state=SimpleNamespace(value=state),
        exit_status=0 if state == 'finished_ok' else 312 if state == 'failed' else None,
    )
    tree.children = children or {}
    return tree


def test_process_tree_html_is_collapsible_and_includes_process_metadata():
    child = _tree('iteration_01', 'PhCalculation', 2, 'finished_ok')
    root = _tree(children={'iteration_01': child})

    rendered = unescape(root._repr_html_())

    assert rendered.count('<details') == 1
    assert '<details open>' in rendered
    assert 'ROOT' in rendered
    assert 'iteration_01' in rendered
    assert 'PhBaseWorkChain · PK 1 · finished' in rendered
    assert 'PhCalculation · PK 2 · finished_ok' in rendered


def test_process_tree_uses_html_display_in_notebook(monkeypatch):
    displayed = []
    display_module = ModuleType('IPython.display')
    display_module.display = displayed.append
    ipython_module = ModuleType('IPython')
    ipython_module.__path__ = []
    ipython_module.display = display_module
    monkeypatch.setitem(sys.modules, 'IPython', ipython_module)
    monkeypatch.setitem(sys.modules, 'IPython.display', display_module)

    root = _tree()
    monkeypatch.setattr(root, '_in_notebook', lambda: True)

    root.print_tree()

    assert displayed == [root]


def test_process_tree_to_dict():
    child = _tree('iteration_01', 'PhCalculation', 2, 'finished_ok')
    root = _tree(children={'iteration_01': child})

    data = root.to_dict()

    assert data == {
        'name': 'ROOT',
        'pk': 1,
        'process_label': 'PhBaseWorkChain',
        'state': 'finished',
        'exit_status': None,
        'children': {
            'iteration_01': {
                'name': 'iteration_01',
                'pk': 2,
                'process_label': 'PhCalculation',
                'state': 'finished_ok',
                'exit_status': 0,
                'children': {},
            }
        },
    }


def test_print_tree_function_with_process_tree(monkeypatch):
    displayed = []
    display_module = ModuleType('IPython.display')
    display_module.display = displayed.append
    ipython_module = ModuleType('IPython')
    ipython_module.__path__ = []
    ipython_module.display = display_module
    monkeypatch.setitem(sys.modules, 'IPython', ipython_module)
    monkeypatch.setitem(sys.modules, 'IPython.display', display_module)

    root = _tree()
    monkeypatch.setattr(root, '_in_notebook', lambda: True)

    print_tree(root)

    assert displayed == [root]

