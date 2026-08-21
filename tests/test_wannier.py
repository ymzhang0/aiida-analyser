from pathlib import Path
from types import SimpleNamespace

from aiida_analyser.core.analyser_registry import resolve_analyser
from aiida_analyser.quantumespresso.pw2wannier90_base import Pw2Wannier90BaseAnalyser
from aiida_analyser.quantumespresso.pw2wannier90_calculation import Pw2Wannier90Analyser
from aiida_analyser.wannier.wannier90 import Wannier90Analyser
from aiida_analyser.wannier.wannier90_base import Wannier90BaseAnalyser
from aiida_analyser.wannier.wannier90_calculation import Wannier90CalculationAnalyser


class MockRepository:
    def __init__(self, filenames):
        self.filenames = filenames

    def copy_tree(self, destpath: Path):
        for name in self.filenames:
            target = destpath / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"content of {name}")


class MockOutputs:
    def __init__(self, retrieved_files=None, remote_path=None):
        self._retrieved_files = retrieved_files or []
        self._remote_path = remote_path

    @property
    def retrieved(self):
        return MockRepository(self._retrieved_files)

    @property
    def remote_folder(self):
        if self._remote_path is None:
            raise KeyError("No remote folder")
        return SimpleNamespace(get_remote_path=lambda: self._remote_path)


def _make_calc_node(pk, label, process_type, call_link_label='iteration_01', filenames=None, retrieved=None, remote_path=None):
    return SimpleNamespace(
        pk=pk,
        ctime=1.0,
        node_type='process.calculation.calcjob.CalcJobNode.',
        process_label=label,
        process_type=process_type,
        is_finished_ok=True,
        is_finished=True,
        base=SimpleNamespace(
            repository=MockRepository(filenames or ['aiida.in']),
            extras=SimpleNamespace(all={}),
            attributes=SimpleNamespace(all={
                'metadata_inputs': {'metadata': {'call_link_label': call_link_label}},
                'process_label': label,
            }),
        ),
        outputs=MockOutputs(retrieved or ['aiida.out'], remote_path=remote_path),
        inputs=SimpleNamespace(pseudos={}),
    )


def _make_workchain_node(pk, label, process_type, call_link_label=None, called=None):
    return SimpleNamespace(
        pk=pk,
        ctime=1.0,
        node_type='process.workflow.workchain.WorkChainNode.',
        process_label=label,
        process_type=process_type,
        is_finished_ok=True,
        is_finished=True,
        called=called or [],
        base=SimpleNamespace(
            repository=MockRepository([]),
            extras=SimpleNamespace(all={}),
            attributes=SimpleNamespace(all={
                'metadata_inputs': {'metadata': {'call_link_label': call_link_label}} if call_link_label else {},
                'process_label': label,
            }),
        ),
        outputs=SimpleNamespace(),
        inputs=SimpleNamespace(),
    )


def test_wannier_registry_resolutions():
    w90_calc = SimpleNamespace(
        process_label='Wannier90Calculation',
        process_type='aiida.calculations:wannier90.wannier90',
    )
    assert resolve_analyser(w90_calc) is Wannier90CalculationAnalyser

    w90_base = SimpleNamespace(
        process_label='Wannier90BaseWorkChain',
        process_type='aiida.workflows:wannier90_workflows.base',
    )
    assert resolve_analyser(w90_base) is Wannier90BaseAnalyser

    w90_wc = SimpleNamespace(
        process_label='Wannier90WorkChain',
        process_type='aiida.workflows:wannier90_workflows.wannier90',
    )
    assert resolve_analyser(w90_wc) is Wannier90Analyser

    w90_bands = SimpleNamespace(
        process_label='Wannier90BandsWorkChain',
        process_type='aiida.workflows:wannier90_workflows.bands',
    )
    assert resolve_analyser(w90_bands) is Wannier90Analyser

    pw2w90_calc = SimpleNamespace(
        process_label='Pw2wannier90Calculation',
        process_type='aiida.calculations:quantumespresso.pw2wannier90',
    )
    assert resolve_analyser(pw2w90_calc) is Pw2Wannier90Analyser

    pw2w90_base = SimpleNamespace(
        process_label='Pw2Wannier90BaseWorkChain',
        process_type='aiida.workflows:quantumespresso.pw2wannier90.base',
    )
    assert resolve_analyser(pw2w90_base) is Pw2Wannier90BaseAnalyser

    pw2w90_base_lower = SimpleNamespace(
        process_label='Pw2wannier90BaseWorkChain',
        process_type=None,
    )
    assert resolve_analyser(pw2w90_base_lower) is Pw2Wannier90BaseAnalyser


def test_wannier90_analyser_copy_tree_with_direct_pw2wannier90(tmp_path):
    scf_calc = _make_calc_node(10, 'PwCalculation', 'aiida.calculations:quantumespresso.pw', 'iteration_01', ['aiida.in'], ['aiida.out'])
    nscf_calc = _make_calc_node(20, 'PwCalculation', 'aiida.calculations:quantumespresso.pw', 'iteration_01', ['aiida.in'], ['aiida.out'])
    w90_pp_calc = _make_calc_node(30, 'Wannier90Calculation', 'aiida.calculations:wannier90.wannier90', 'iteration_01', ['aiida.win'], ['aiida.wout', 'aiida.nnkp'])
    pw2w90_calc = _make_calc_node(40, 'Pw2wannier90Calculation', 'aiida.calculations:quantumespresso.pw2wannier90', 'pw2wannier90', ['aiida.in'], ['aiida.out', 'aiida.mmn', 'aiida.amn'])
    w90_calc = _make_calc_node(50, 'Wannier90Calculation', 'aiida.calculations:wannier90.wannier90', 'iteration_01', ['aiida.win', 'aiida.mmn', 'aiida.amn'], ['aiida.wout', 'aiida_band.dat'])

    scf_wc = _make_workchain_node(100, 'PwBaseWorkChain', 'aiida.workflows:quantumespresso.pw.base', 'scf', [scf_calc])
    nscf_wc = _make_workchain_node(200, 'PwBaseWorkChain', 'aiida.workflows:quantumespresso.pw.base', 'nscf', [nscf_calc])
    w90_pp_wc = _make_workchain_node(300, 'Wannier90BaseWorkChain', 'aiida.workflows:wannier90_workflows.base', 'wannier90_pp', [w90_pp_calc])
    w90_wc = _make_workchain_node(500, 'Wannier90BaseWorkChain', 'aiida.workflows:wannier90_workflows.base', 'wannier90', [w90_calc])
    seekpath_node = SimpleNamespace(
        pk=600,
        ctime=0.5,
        process_label='seekpath_structure_analysis',
        process_type=None,
        base=SimpleNamespace(attributes=SimpleNamespace(all={'metadata_inputs': {'metadata': {'call_link_label': 'seekpath_structure_analysis'}}})),
    )

    root_node = _make_workchain_node(
        1,
        'Wannier90WorkChain',
        'aiida.workflows:wannier90_workflows.wannier90',
        called=[seekpath_node, scf_wc, nscf_wc, w90_pp_wc, pw2w90_calc, w90_wc],
    )
    analyser = Wannier90Analyser(root_node)

    dest = analyser.copy_tree(tmp_path / 'export')

    assert dest.exists()
    assert (dest / 'scf' / 'iteration_01' / 'aiida.in').exists()
    assert (dest / 'scf' / 'iteration_01' / 'aiida.out').exists()
    assert (dest / 'nscf' / 'iteration_01' / 'aiida.in').exists()
    assert (dest / 'nscf' / 'iteration_01' / 'aiida.out').exists()
    assert (dest / 'wannier90_pp' / 'iteration_01' / 'aiida.win').exists()
    assert (dest / 'wannier90_pp' / 'iteration_01' / 'aiida.nnkp').exists()
    assert (dest / 'pw2wannier90' / 'aiida.in').exists()
    assert (dest / 'pw2wannier90' / 'aiida.mmn').exists()
    assert (dest / 'wannier90' / 'iteration_01' / 'aiida.win').exists()
    assert (dest / 'wannier90' / 'iteration_01' / 'aiida_band.dat').exists()
    assert not (dest / 'seekpath_structure_analysis').exists()


def test_wannier90_analyser_copy_tree_with_pw2wannier90_base_workchain(tmp_path):
    pw2w90_calc = _make_calc_node(40, 'Pw2wannier90Calculation', 'aiida.calculations:quantumespresso.pw2wannier90', 'iteration_01', ['aiida.in'], ['aiida.out', 'aiida.mmn'])
    pw2w90_wc = _make_workchain_node(400, 'Pw2Wannier90BaseWorkChain', 'aiida.workflows:quantumespresso.pw2wannier90.base', 'pw2wannier90', [pw2w90_calc])

    root_node = _make_workchain_node(
        1,
        'Wannier90WorkChain',
        'aiida.workflows:wannier90_workflows.wannier90',
        called=[pw2w90_wc],
    )
    analyser = Wannier90Analyser(root_node)

    dest = analyser.copy_tree(tmp_path / 'export')

    assert dest.exists()
    assert (dest / 'pw2wannier90' / 'iteration_01' / 'aiida.in').exists()
    assert (dest / 'pw2wannier90' / 'iteration_01' / 'aiida.mmn').exists()


def test_wannier90_analyser_get_calcjob_paths():
    scf_calc = _make_calc_node(10, 'PwCalculation', 'aiida.calculations:quantumespresso.pw', 'iteration_01', remote_path='/remote/scf')
    w90_pp_calc = _make_calc_node(30, 'Wannier90Calculation', 'aiida.calculations:wannier90.wannier90', 'iteration_01', remote_path='/remote/w90_pp')
    pw2w90_calc = _make_calc_node(40, 'Pw2wannier90Calculation', 'aiida.calculations:quantumespresso.pw2wannier90', 'pw2wannier90', remote_path='/remote/pw2w90')

    scf_wc = _make_workchain_node(100, 'PwBaseWorkChain', 'aiida.workflows:quantumespresso.pw.base', 'scf', [scf_calc])
    w90_pp_wc = _make_workchain_node(300, 'Wannier90BaseWorkChain', 'aiida.workflows:wannier90_workflows.base', 'wannier90_pp', [w90_pp_calc])

    root_node = _make_workchain_node(
        1,
        'Wannier90WorkChain',
        'aiida.workflows:wannier90_workflows.wannier90',
        called=[scf_wc, w90_pp_wc, pw2w90_calc],
    )
    analyser = Wannier90Analyser(root_node)
    paths = analyser.get_calcjob_paths()

    assert paths == {
        'scf/iteration_01': '/remote/scf',
        'wannier90_pp/iteration_01': '/remote/w90_pp',
        'pw2wannier90': '/remote/pw2w90',
    }


def test_wannier90_analyser_get_state():
    scf_calc = _make_calc_node(10, 'PwCalculation', 'aiida.calculations:quantumespresso.pw', 'iteration_01')
    nscf_calc = _make_calc_node(20, 'PwCalculation', 'aiida.calculations:quantumespresso.pw', 'iteration_01')
    w90_pp_calc = _make_calc_node(30, 'Wannier90Calculation', 'aiida.calculations:wannier90.wannier90', 'iteration_01')
    pw2w90_calc = _make_calc_node(40, 'Pw2wannier90Calculation', 'aiida.calculations:quantumespresso.pw2wannier90', 'pw2wannier90')
    w90_calc = _make_calc_node(50, 'Wannier90Calculation', 'aiida.calculations:wannier90.wannier90', 'iteration_01')

    scf_wc = _make_workchain_node(100, 'PwBaseWorkChain', 'aiida.workflows:quantumespresso.pw.base', 'scf', [scf_calc])
    nscf_wc = _make_workchain_node(200, 'PwBaseWorkChain', 'aiida.workflows:quantumespresso.pw.base', 'nscf', [nscf_calc])
    w90_pp_wc = _make_workchain_node(300, 'Wannier90BaseWorkChain', 'aiida.workflows:wannier90_workflows.base', 'wannier90_pp', [w90_pp_calc])
    w90_wc = _make_workchain_node(500, 'Wannier90BaseWorkChain', 'aiida.workflows:wannier90_workflows.base', 'wannier90', [w90_calc])

    root_node = _make_workchain_node(
        1,
        'Wannier90WorkChain',
        'aiida.workflows:wannier90_workflows.wannier90',
        called=[scf_wc, nscf_wc, w90_pp_wc, pw2w90_calc, w90_wc],
    )
    analyser = Wannier90Analyser(root_node)

    assert analyser.get_state() == ('ROOT', 'finished_ok', 0)
