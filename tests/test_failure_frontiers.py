from types import SimpleNamespace

from aiida_analyser.epw.epw_prep import EpwPrepAnalyser, EpwPrepGroup
from aiida_analyser.wannier.wannier90_calculation import Wannier90CalculationAnalyser
from aiida_analyser.quantumespresso.ph_calculation import PhAnalyser


def _node(label, call_label, *, failed=False, called=(), exit_code=311):
    attributes = {'metadata_inputs': {'metadata': {'call_link_label': call_label}}}
    return SimpleNamespace(
        ctime=1,
        pk=hash((label, call_label)) % 10000,
        process_label=label,
        process_type=None,
        is_finished_ok=not failed,
        is_finished=True,
        is_failed=failed,
        is_excepted=False,
        is_killed=False,
        process_state=SimpleNamespace(value='finished'),
        exit_code=exit_code if failed else 0,
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
        parent_path, leaf_label = expected_path.rsplit('/', 1)
        assert analyser.get_state() == (f'{parent_path}/ROOT/{leaf_label}', 'finished', 311)
        assert [(path, tree.name) for path, tree in analyser.get_failure_frontiers()] == [
            (f'ROOT/{expected_path}', 'iteration_01')
        ]


def test_epwprep_reports_its_own_failed_validation_when_children_succeed():
    successful_child = _node('EpwBaseWorkChain', 'epw_base')
    root = _node('EpwPrepWorkChain', 'ROOT', failed=True, called=[successful_child], exit_code=405)

    analyser = EpwPrepAnalyser(root)

    assert analyser.get_state() == ('ROOT', 'finished', 405)


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

    assert analyser.get_state() == (
        'w90_bands/wannier90/ROOT/iteration_01',
        'W90_DISENTANGLEMENT_NOT_CONVERGED',
        404,
    )

    report = analyser.get_failure_report(include_outputs=True)
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
