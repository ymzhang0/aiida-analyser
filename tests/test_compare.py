from types import SimpleNamespace

from aiida.common.links import LinkType

from aiida_analyser.core.compare import DiffEntry, NodeDiff, NodeReference, compare_nodes


class MappingStore:
    def __init__(self, values=None):
        self.all = values or {}


class Repository:
    def __init__(self, files=None):
        self.files = files or {}

    def hash(self):
        return repr(sorted(self.files.items()))

    def glob(self):
        return self.files

    def get_object(self, path):
        return SimpleNamespace(
            key=self.files[str(path)],
            is_file=lambda: True,
        )


class LinkManager:
    def __init__(self, triples):
        self.triples = triples

    def all(self):
        return list(self.triples)


class Links:
    def __init__(self):
        self.incoming = []
        self.outgoing = []

    @staticmethod
    def _matching(triples, link_type):
        allowed = set(link_type)
        return [triple for triple in triples if triple.link_type in allowed]

    def get_incoming(self, *, link_type):
        return LinkManager(self._matching(self.incoming, link_type))

    def get_outgoing(self, *, link_type):
        return LinkManager(self._matching(self.outgoing, link_type))


class Node:
    _next_pk = 1

    def __init__(self, node_type, attributes=None, extras=None, files=None):
        self.pk = Node._next_pk
        Node._next_pk += 1
        self.uuid = f'uuid-{self.pk}'
        self.node_type = node_type
        self.process_type = 'test:process' if node_type.startswith('process.') else None
        self.process_label = 'TestProcess' if node_type.startswith('process.') else None
        self.label = ''
        self.description = ''
        self.computer = None
        self.user = SimpleNamespace(email='test@example.com')
        self.base = SimpleNamespace(
            attributes=MappingStore(attributes),
            extras=MappingStore(extras),
            repository=Repository(files),
            links=Links(),
        )


def connect(source, target, link_type, label):
    triple = SimpleNamespace(node=source, link_type=link_type, link_label=label)
    if link_type in (LinkType.INPUT_CALC, LinkType.INPUT_WORK):
        target.base.links.incoming.append(triple)
    else:
        target.base.links.outgoing.append(triple)


def process(attributes=None):
    return Node('process.workflow.workchain.WorkChainNode.', attributes)


def calculation(attributes=None):
    return Node('process.calculation.calcjob.CalcJobNode.', attributes)


def data(attributes=None):
    return Node('data.core.dict.Dict.', attributes)


def test_compare_nodes_follows_ports_calls_and_outputs():
    left = process({'root_setting': 1})
    right = process({'root_setting': 2})
    left_input = data({'value': 10})
    right_input = data({'value': 11})
    left_calc = calculation({'options': {'max_wallclock_seconds': 100}})
    right_calc = calculation({'options': {'max_wallclock_seconds': 200}})
    left_output = data({'energy': -1.0})
    right_output = data({'energy': -0.9})

    connect(left_input, left, LinkType.INPUT_WORK, 'settings__parameters')
    connect(right_input, right, LinkType.INPUT_WORK, 'settings__parameters')
    connect(left_calc, left, LinkType.CALL_CALC, 'iteration_01')
    connect(right_calc, right, LinkType.CALL_CALC, 'iteration_01')
    connect(left_output, left_calc, LinkType.CREATE, 'output_parameters')
    connect(right_output, right_calc, LinkType.CREATE, 'output_parameters')

    result = compare_nodes(left, right)
    paths = {entry.path for entry in result.entries}

    assert '$.attributes.root_setting' in paths
    assert '$.inputs.settings.parameters.attributes.value' in paths
    assert '$.called.iteration_01.attributes.options.max_wallclock_seconds' in paths
    assert '$.called.iteration_01.outputs.output_parameters.attributes.energy' in paths
    assert result.compared_node_pairs == 4


def test_compare_nodes_max_depth_still_compares_root_ports():
    left = process()
    right = process()
    left_input = data({'value': 1})
    right_input = data({'value': 2})
    left_calc = calculation({'different': 1})
    right_calc = calculation({'different': 2})

    connect(left_input, left, LinkType.INPUT_WORK, 'parameters')
    connect(right_input, right, LinkType.INPUT_WORK, 'parameters')
    connect(left_calc, left, LinkType.CALL_CALC, 'iteration_01')
    connect(right_calc, right, LinkType.CALL_CALC, 'iteration_01')

    result = compare_nodes(left, right, max_depth=0)
    paths = {entry.path for entry in result.entries}

    assert '$.inputs.parameters.attributes.value' in paths
    assert not any(path.startswith('$.called') for path in paths)


def test_compare_nodes_ignores_identity_unless_requested():
    left = data({'value': 1})
    right = data({'value': 1})

    assert compare_nodes(left, right).equal

    result = compare_nodes(left, right, include_identity=True)
    paths = {entry.path for entry in result.entries}
    assert '$.metadata.pk' in paths
    assert '$.metadata.uuid' in paths


def test_compare_nodes_reports_missing_port():
    left = process()
    right = process()
    connect(data({'value': 1}), right, LinkType.INPUT_WORK, 'new_input')

    result = compare_nodes(left, right)

    assert result.entries[-1].path == '$.inputs.new_input'
    assert result.entries[-1].kind == 'missing_left'


def test_node_diff_groups_and_hides_volatile_details():
    reference = NodeReference(1, 'uuid-1', 'process.workflow.workchain.WorkChainNode.')
    result = NodeDiff(
        reference,
        reference,
        [
            DiffEntry('$.inputs.parameters.attributes.ecutwfc', 'changed', 40, 50),
            DiffEntry('$.called.scf.outputs.parameters.attributes.energy', 'changed', -1.0, -0.9),
            DiffEntry(
                '$.called.scf.outputs.retrieved.repository.files["aiida.out"]',
                'changed',
                'hash-output-a',
                'hash-output-b',
            ),
            DiffEntry('$.called.CALL.repository.files.source_file', 'changed', 'hash-a', 'hash-b'),
            DiffEntry('$.called.scf.repository.files["aiida.out"]', 'changed', 'hash-c', 'hash-d'),
            DiffEntry('$.called.scf.attributes.job_id', 'changed', '123', '456'),
        ],
        compared_node_pairs=5,
    )

    grouped = result.grouped_entries()
    assert len(grouped['inputs']) == 1
    assert len(grouped['outputs']) == 1
    assert len(grouped['output_files']) == 1
    assert len(grouped['process']) == 1
    assert len(grouped['files']) == 1
    assert len(grouped['runtime']) == 1

    rendered = result.format(max_entries=None)
    assert 'Inputs' in rendered
    assert 'ecutwfc' in rendered
    assert 'source_file' in rendered
    assert 'aiida.out' not in rendered
    assert 'job_id' not in rendered
    assert 'use show_files=True' in rendered
    assert 'use show_runtime=True' in rendered

    expanded = result.format(max_entries=None, show_files=True, show_runtime=True)
    assert 'aiida.out' in expanded
    assert 'job_id' in expanded
    assert 'scf' in expanded

    flat = result.format_flat(max_entries=None)
    assert '$.called.scf.attributes.job_id' in flat
