from types import SimpleNamespace

from aiida_analyser.core.base import BaseWorkChainAnalyser


def _analyser_with_child(child_node):
    root = SimpleNamespace(pk=1, process_label='RootWorkChain')
    analyser = BaseWorkChainAnalyser(root)
    analyser.__dict__['process_tree'] = SimpleNamespace(
        children={'child': SimpleNamespace(node=child_node)},
    )
    return analyser


def test_copy_tree_skips_unregistered_child(tmp_path):
    analyser = _analyser_with_child(SimpleNamespace(
        pk=2,
        process_label='UnknownWorkChain',
        process_type='example:unknown',
    ))

    destination = analyser.copy_tree(tmp_path / 'tree')

    assert destination.exists()
    assert list(destination.iterdir()) == []


def test_calcjob_paths_skip_unregistered_child():
    analyser = _analyser_with_child(SimpleNamespace(
        pk=2,
        process_label='UnknownWorkChain',
        process_type='example:unknown',
    ))

    assert analyser.get_calcjob_paths() == {}
