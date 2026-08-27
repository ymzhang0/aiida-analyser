from types import SimpleNamespace

from aiida_analyser.epw.epw_prep import EpwPrepAnalyser, EpwPrepGroup
from aiida_analyser.quantumespresso.ph_base import PhBaseAnalyser
from aiida_analyser.quantumespresso.ph_calculation import PhAnalyser
from aiida_analyser.wannier.wannier90_calculation import Wannier90CalculationAnalyser


def _node(
    label,
    call_label,
    *,
    failed=False,
    called=(),
    exit_code=311,
    ctime=1,
    process_state='finished',
    is_killed=False,
):
    attributes = {'metadata_inputs': {'metadata': {'call_link_label': call_label}}}
    return SimpleNamespace(
        ctime=ctime,
        pk=abs(hash((label, call_label, ctime))) % 10000 + 1,
        process_label=label,
        process_type=None,
        is_finished_ok=not failed and not is_killed,
        is_finished=not is_killed,
        is_failed=failed,
        is_excepted=False,
        is_killed=is_killed,
        process_state=SimpleNamespace(value='killed' if is_killed else process_state),
        exit_code=exit_code if (failed and not is_killed) else (0 if not is_killed else None),
        called=list(called),
        base=SimpleNamespace(attributes=SimpleNamespace(all=attributes)),
    )


def _failed_calc(label='PwCalculation'):
    return _node(label, 'iteration_01', failed=True)


def _analyser_for_failed_stage(stage):
    leaf = _failed_calc('PwCalculation' if stage == 'w90_bands' else 'PhCalculation')
    if stage == 'w90_bands':
        scf = _node(
            'PwBaseWorkChain', 'scf', failed=True,
            called=[leaf, _node('CleanupCalculation', 'cleanup')],
        )
        child = _node('Wannier90WorkChain', 'w90_bands', failed=True, called=[scf])
        expected = 'w90_bands/scf/iteration_01'
    elif stage == 'ph_base':
        child = _node('PhBaseWorkChain', 'ph_base', failed=True, called=[leaf])
        expected = 'ph_base/iteration_01'
    else:
        child = _node('EpwBaseWorkChain', 'epw_base', failed=True, called=[_failed_calc('EpwCalculation')])
        expected = 'epw_base/iteration_01'

    root = _node('EpwPrepWorkChain', 'ROOT', failed=True, called=[child])
    return EpwPrepAnalyser(root), expected


def test_epwprep_resolves_leaf_failures_through_all_supported_stages():
    for stage in ('w90_bands', 'ph_base', 'epw_base'):
        analyser, expected_path = _analyser_for_failed_stage(stage)
        report = analyser.get_report()
        assert report.state == 'finished'
        assert report.exit_code == 311
        assert report.path == f'ROOT/{expected_path}'
        assert [(path, tree.name) for path, tree in analyser.get_failure_frontiers()] == [
            (f'ROOT/{expected_path}', 'iteration_01')
        ]


def test_epwprep_reports_its_own_failed_validation_when_children_succeed():
    successful_child = _node('EpwBaseWorkChain', 'epw_base')
    root = _node('EpwPrepWorkChain', 'ROOT', failed=True, called=[successful_child], exit_code=405)

    analyser = EpwPrepAnalyser(root)
    report = analyser.get_report()
    assert report.state == 'finished'
    assert report.exit_code == 405
    assert report.path == 'ROOT'


def test_epwprep_propagates_calculation_specific_incomplete_stdout_state():
    leaf = _node('Wannier90Calculation', 'iteration_01', failed=True, exit_code=404)
    leaf.exit_message = 'The stdout output file was incomplete probably because the calculation got interrupted.'
    retrieved_filenames = []
    leaf.process_class = SimpleNamespace(_DEFAULT_OUTPUT_FILE='aiida.wout')
    leaf.outputs = SimpleNamespace(
        retrieved=SimpleNamespace(
            get_object_content=lambda filename: (
                retrieved_filenames.append(filename) or 'Disentanglement did not converge'
            )
        )
    )
    leaf.get_scheduler_stdout = lambda: ''
    leaf.get_scheduler_stderr = lambda: ''
    wannier_base = _node('Wannier90BaseWorkChain', 'wannier90', failed=True, called=[leaf])
    wannier = _node('Wannier90WorkChain', 'w90_bands', failed=True, called=[wannier_base])
    epwprep = _node('EpwPrepWorkChain', 'ROOT', failed=True, called=[wannier])

    analyser = EpwPrepAnalyser(epwprep)
    report = analyser.get_report()
    assert report.path == 'ROOT/w90_bands/wannier90/iteration_01'
    assert report.state == 'W90_DISENTANGLEMENT_NOT_CONVERGED'
    assert report.exit_code == 404

    report = analyser.get_report(include_outputs=True)
    assert [node.path for node in report.primary_chain] == [
        'ROOT',
        'ROOT/w90_bands',
        'ROOT/w90_bands/wannier90',
        'ROOT/w90_bands/wannier90/iteration_01',
    ]
    assert report.primary.raw_exit_status == 404
    assert report.primary.analysis_exit_code.status == 7101
    assert report.primary.analysis_exit_code.label == 'W90_DISENTANGLEMENT_NOT_CONVERGED'
    assert report.primary.analysis_exit_code.evidence == ('disentanglement', 'converg')
    assert report.primary.outputs['output_filename'] == 'aiida.wout'
    assert report.primary.outputs['calculation_output'] == 'Disentanglement did not converge'
    assert set(retrieved_filenames) == {'aiida.wout'}


def test_wannier90_parser_reads_scheduler_output_and_default_wout_file():
    leaf = _node('Wannier90Calculation', 'iteration_01', failed=True, exit_code=404)
    leaf.exit_message = 'The stdout output file was incomplete probably because the calculation got interrupted.'
    leaf.process_class = SimpleNamespace(_DEFAULT_OUTPUT_FILE='aiida.wout')
    leaf.outputs = SimpleNamespace(retrieved=SimpleNamespace(get_object_content=lambda _name: ''))
    leaf.get_scheduler_stdout = lambda: (
        'aiida.amn has too many projections to be used without selecting a subset'
    )
    leaf.get_scheduler_stderr = lambda: 'MPICH ERROR: application called MPI_Abort'

    analyser = Wannier90CalculationAnalyser(leaf)

    assert analyser.get_analysis_exit_code().label == 'W90_PROJECTIONS_ERROR'
    outputs = analyser.get_failure_outputs()
    assert outputs['output_filename'] == 'aiida.wout'
    assert 'too many projections' in outputs['scheduler_stdout']
    assert 'MPI_Abort' in outputs['scheduler_stderr']


def test_ph_parser_contains_migrated_legacy_output_rules():
    leaf = _node('PhCalculation', 'iteration_01', failed=True, exit_code=404)
    leaf.exit_message = 'The stdout output file was incomplete probably because the calculation got interrupted.'
    leaf.outputs = SimpleNamespace(
        retrieved=SimpleNamespace(
            get_object_content=lambda _name: 'FFT grid incompatible with symmetry'
        )
    )
    leaf.get_scheduler_stdout = lambda: ''
    leaf.get_scheduler_stderr = lambda: ''

    analysis_exit_code = PhAnalyser(leaf).get_analysis_exit_code()
    assert (analysis_exit_code.status, analysis_exit_code.label) == (3004, 'ERROR_FFT_GRID_INCOMPATIBLE')


def test_epwprep_group_analyses_all_failed_workchains():
    leaf = _node('Wannier90Calculation', 'iteration_01', failed=True, exit_code=404)
    leaf.exit_message = 'The stdout output file was incomplete probably because the calculation got interrupted.'
    leaf.process_class = SimpleNamespace(_DEFAULT_OUTPUT_FILE='aiida.wout')
    leaf.outputs = SimpleNamespace(
        retrieved=SimpleNamespace(get_object_content=lambda _name: 'Disentanglement did not converge')
    )
    leaf.get_scheduler_stdout = lambda: ''
    leaf.get_scheduler_stderr = lambda: ''
    wannier_base = _node('Wannier90BaseWorkChain', 'wannier90', failed=True, called=[leaf])
    wannier = _node('Wannier90WorkChain', 'w90_bands', failed=True, called=[wannier_base])
    epwprep = _node('EpwPrepWorkChain', 'ROOT', failed=True, called=[wannier])

    group = EpwPrepGroup.__new__(EpwPrepGroup)
    group._flat_nodes = [('MgB2', 0.01, 0.2, 0.1, epwprep)]

    failures = group.analyse_failures(include_outputs=True)

    assert len(failures) == 1
    row = failures.iloc[0]
    assert row['PK'] == epwprep.pk
    assert row['failure_path'] == 'ROOT/w90_bands/wannier90/iteration_01'
    assert row['analysis_exit_label'] == 'W90_DISENTANGLEMENT_NOT_CONVERGED'
    assert row['outputs']['output_filename'] == 'aiida.wout'
    assert row['frontier_count'] == 1


def test_base_workchain_multi_iteration_selects_last_subprocess_failure():
    """Verify that earlier handled iterations in a BaseWorkChain are not selected as failure frontiers."""
    iter1 = _node('PhCalculation', 'iteration_01', failed=True, exit_code=400, ctime=1)
    iter1.exit_message = 'The calculation stopped prematurely because it ran out of walltime.'

    iter2 = _node('PhCalculation', 'iteration_02', failed=True, exit_code=400, ctime=2)
    iter2.exit_message = 'The calculation stopped prematurely because it ran out of walltime.'

    iter3 = _node('PhCalculation', 'iteration_03', failed=True, exit_code=312, ctime=3)
    iter3.exit_message = 'The stdout output file was incomplete probably because the calculation got interrupted.'
    iter3.outputs = SimpleNamespace(
        retrieved=SimpleNamespace(get_object_content=lambda _name: 'mpich error: job aborted')
    )
    iter3.get_scheduler_stdout = lambda: ''
    iter3.get_scheduler_stderr = lambda: ''

    ph_base = _node(
        'PhBaseWorkChain', 'ph_base', failed=True, exit_code=300,
        called=[iter1, iter2, iter3], ctime=10,
    )
    epwprep = _node('EpwPrepWorkChain', 'ROOT', failed=True, exit_code=403, called=[ph_base], ctime=20)

    analyser = EpwPrepAnalyser(epwprep)
    frontiers = analyser.get_failure_frontiers()
    assert len(frontiers) == 1
    path, tree = frontiers[0]
    assert path == 'ROOT/ph_base/iteration_03'
    assert tree.name == 'iteration_03'

    report = analyser.get_failure_report(include_outputs=True)
    assert report.primary is not None
    assert report.primary.path == 'ROOT/ph_base/iteration_03'
    assert report.primary.raw_exit_status == 312
    assert report.primary.analysis_exit_code.status == 3
    assert report.primary.analysis_exit_code.label == 'MPICH_ERROR'

    # Check handled iterations are tracked
    ph_base_report = report.root.children[0]
    assert len(ph_base_report.handled_children) == 2
    assert ph_base_report.handled_children[0].path == 'ROOT/ph_base/iteration_01'
    assert ph_base_report.handled_children[0].raw_exit_status == 400
    assert ph_base_report.handled_children[1].path == 'ROOT/ph_base/iteration_02'
    assert ph_base_report.handled_children[1].raw_exit_status == 400

    # Check format output shows handled
    formatted = report.format()
    assert '[handled] ROOT/ph_base/iteration_01' in formatted
    assert '[handled] ROOT/ph_base/iteration_02' in formatted
    assert 'ROOT/ph_base/iteration_03' in formatted

    # Check group failure table
    group = EpwPrepGroup.__new__(EpwPrepGroup)
    group._flat_nodes = [('GaAs', 0.02, 0.15, 0.5, epwprep)]
    failures = group.analyse_failures()
    assert len(failures) == 1
    row = failures.iloc[0]
    assert row['failure_path'] == 'ROOT/ph_base/iteration_03'
    assert row['failure_process_label'] == 'PhCalculation'
    assert row['raw_exit_status'] == 312
    assert row['analysis_exit_status'] == 3
    assert row['analysis_exit_label'] == 'MPICH_ERROR'
    assert row['frontier_count'] == 1


def test_base_workchain_killed_selects_last_subprocess():
    """Verify that a killed process in the final iteration is identified as primary."""
    iter1 = _node('PhCalculation', 'iteration_01', failed=True, exit_code=400, ctime=1)
    iter2 = _node('PhCalculation', 'iteration_02', ctime=2, is_killed=True)

    ph_base = _node('PhBaseWorkChain', 'ph_base', called=[iter1, iter2], ctime=10, is_killed=True)
    epwprep = _node('EpwPrepWorkChain', 'ROOT', called=[ph_base], ctime=20, is_killed=True)

    analyser = EpwPrepAnalyser(epwprep)
    report = analyser.get_failure_report()
    assert len(report.frontiers) == 1
    assert report.primary.path == 'ROOT/ph_base/iteration_02'
    assert report.primary.process_state == 'killed'
    assert report.primary.raw_exit_status is None


def test_base_workchain_fails_when_last_subprocess_ok():
    """Verify that if the last calculation succeeded but the BaseWorkChain failed, the workchain is the frontier."""
    iter1 = _node('PhCalculation', 'iteration_01', failed=True, exit_code=400, ctime=1)
    iter2 = _node('PhCalculation', 'iteration_02', failed=False, exit_code=0, ctime=2)

    ph_base = _node(
        'PhBaseWorkChain', 'ph_base', failed=True, exit_code=401,
        called=[iter1, iter2], ctime=10,
    )
    ph_base.exit_message = 'The work chain failed to merge the q-points data.'
    epwprep = _node('EpwPrepWorkChain', 'ROOT', failed=True, exit_code=403, called=[ph_base], ctime=20)

    analyser = EpwPrepAnalyser(epwprep)
    frontiers = analyser.get_failure_frontiers()
    assert len(frontiers) == 1
    path, tree = frontiers[0]
    assert path == 'ROOT/ph_base'
    assert tree.node.process_label == 'PhBaseWorkChain'

    report = analyser.get_failure_report()
    assert report.primary.path == 'ROOT/ph_base'
    assert report.primary.process_label == 'PhBaseWorkChain'
    assert report.primary.raw_exit_status == 401


def test_ph_base_analyser_direct_multi_iteration():
    """Verify PhBaseAnalyser directly resolves the terminal iteration."""
    iter1 = _node('PhCalculation', 'iteration_01', failed=True, exit_code=400, ctime=1)
    iter2 = _node('PhCalculation', 'iteration_02', failed=True, exit_code=312, ctime=2)
    iter2.exit_message = 'The stdout output file was incomplete probably because the calculation got interrupted.'
    iter2.outputs = SimpleNamespace(
        retrieved=SimpleNamespace(get_object_content=lambda _name: 'mpich error: abort')
    )
    iter2.get_scheduler_stdout = lambda: ''
    iter2.get_scheduler_stderr = lambda: ''

    ph_base = _node(
        'PhBaseWorkChain', 'ROOT', failed=True, exit_code=300,
        called=[iter1, iter2], ctime=10,
    )

    analyser = PhBaseAnalyser(ph_base)
    report = analyser.get_report()
    assert report.path == 'ROOT/iteration_02'
    assert report.state == 'MPICH_ERROR'
    assert report.exit_code == 312


def test_unregistered_node_raises_error():
    """Verify that resolving an unmanaged node directly raises UnregisteredProcessError."""
    import pytest
    from aiida_analyser.core.analyser_registry import UnregisteredProcessError, resolve_analyser
    from aiida_analyser.core.base import ProcessTree

    unregistered_node = _node('UnknownProcessCustomWorkChain', 'custom', failed=True)
    with pytest.raises(UnregisteredProcessError, match='No analyser registered'):
        resolve_analyser(unregistered_node)

    # ProcessTree should not have find_failure_frontiers
    tree = ProcessTree(unregistered_node)
    assert not hasattr(tree, 'find_failure_frontiers')


def test_workchain_with_unregistered_failed_child_raises_error():
    """Verify that a composite workchain encountering an unregistered failed child raises UnregisteredProcessError."""
    import pytest
    from aiida_analyser.core.analyser_registry import UnregisteredProcessError

    unregistered_child = _node('CustomUnknownCalc', 'calc_step', failed=True)
    wc = _node('EpwPrepWorkChain', 'ROOT', failed=True, called=[unregistered_child])

    analyser = EpwPrepAnalyser(wc)
    with pytest.raises(UnregisteredProcessError, match='CustomUnknownCalc'):
        analyser.get_failure_frontiers()


def test_process_report_lazy_properties_and_node_access():
    """Verify that ProcessReport and ProcessReportNode provide direct access to stderr, stdout, output, and node."""
    leaf = _node('PhCalculation', 'iteration_01', failed=True, exit_code=312)
    leaf.outputs = SimpleNamespace(
        retrieved=SimpleNamespace(get_object_content=lambda _name: 'JOB LOG CONTENT')
    )
    leaf.get_scheduler_stdout = lambda: 'SCHEDULER STDOUT LOG'
    leaf.get_scheduler_stderr = lambda: 'SCHEDULER STDERR LOG'

    ph_base = _node('PhBaseWorkChain', 'ph_base', failed=True, called=[leaf])
    epwprep = _node('EpwPrepWorkChain', 'ROOT', failed=True, called=[ph_base])

    analyser = EpwPrepAnalyser(epwprep)
    report = analyser.get_report()

    assert report.primary is not None
    assert report.primary.node is leaf
    assert report.primary.stderr == 'SCHEDULER STDERR LOG'
    assert report.primary.stdout == 'SCHEDULER STDOUT LOG'
    assert report.primary.output == 'JOB LOG CONTENT'

    # Direct top-level accessors
    assert report.stderr == 'SCHEDULER STDERR LOG'
    assert report.stdout == 'SCHEDULER STDOUT LOG'
    assert report.output == 'JOB LOG CONTENT'


def test_killed_workchain_sets_active_path_without_leaf_diagnosis():
    """Verify that a killed workchain points to the active interrupted step without diagnosing leaf calculations."""
    leaf_done = _node('PhCalculation', 'iteration_01', failed=True, exit_code=400)
    leaf_running = _node('PhCalculation', 'iteration_02', failed=False)
    leaf_running.is_finished_ok = False
    leaf_running.process_state = SimpleNamespace(value='running')

    ph_base = _node(
        'PhBaseWorkChain', 'ph_base', failed=False,
        called=[leaf_done, leaf_running],
    )
    ph_base.is_finished_ok = False
    ph_base.process_state = SimpleNamespace(value='running')

    epwprep = _node('EpwPrepWorkChain', 'ROOT', failed=False, called=[ph_base])
    epwprep.is_finished_ok = False
    epwprep.process_state = SimpleNamespace(value='killed')

    analyser = EpwPrepAnalyser(epwprep)
    report = analyser.get_report()

    assert report.state == 'killed'
    assert report.path == 'ROOT/ph_base/iteration_02'
    assert report.primary is not None
    # No calculation-level diagnosis attached since workchain was killed
    assert report.primary.analysis_exit_code is None


