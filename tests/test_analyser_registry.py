from types import SimpleNamespace

from aiida_analyser.core.analyser_registry import AnalyserRegistry, AnalyserSpec


def test_registry_prefers_process_type_over_process_label():
    registry = AnalyserRegistry()
    type_spec = AnalyserSpec('collections:Counter', process_types=('entry.point:type',))
    label_spec = AnalyserSpec('collections:defaultdict', process_labels=('ProcessLabel',))
    registry.register(type_spec)
    registry.register(label_spec)

    node = SimpleNamespace(process_type='entry.point:type', process_label='ProcessLabel')

    assert registry.resolve(node).__name__ == 'Counter'


def test_registry_does_not_resolve_an_unregistered_node():
    registry = AnalyserRegistry()

    assert registry.resolve(SimpleNamespace(process_type='unknown', process_label='Unknown')) is None
