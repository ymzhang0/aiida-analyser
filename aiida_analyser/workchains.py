from aiida import orm


from collections import deque
from io import StringIO
import re

from aiida import orm
from aiida.common.links import LinkType

THZ_TO_MEV = 4.14

def find_iterations(
    wc: orm.WorkChainNode
):
    iterations = []
    for label in wc.base.links.get_outgoing().all_link_labels():
        if label.startswith('iteration'):
            iterations.append(label)
    return iterations

def parse_scon_raw_out(
    wc: orm.WorkChainNode
):
    if wc.process_label == 'SuperConWorkChain':
        if wc.is_killed:
            return (999, 'Killed by user')

        if wc.is_excepted:
            return (999, 'Excepted')



def parse_raw_out(
    wc: orm.WorkChainNode
    ):
    __ERRORS = [
        (1   , 'KILLED',                               'Job is killed'),
        (2   , 'EXCEPTED',                             'Job is excepted'),
        (3   , 'MPICH_ERROR',                          'MPICH error'),
        (404 , 'FAILURE_IN_WANNIER90WORKCHAIN',        'Wannier90WorkChain failed'),
        (405 , 'FAILURE_IN_EPWWORKCHAIN',              'EpwWorkChain failed'),
        (3000, 'ERROR_CONVERGENCE_NOT_REACHED',        'No convergence has been achieved'),
        (3001, 'ERROR_DAVCIO',                         'Error in routine davcio'),
        (3002, 'ERROR_WRONG_REPRESENTATION',           'wrong representation'),
        (3003, 'ERROR_QMESH_BREAKS_SYMMETRY',          'q-mesh breaks symmetry'),
        (3004, 'ERROR_FFT_GRID_INCOMPATIBLE',          'FFT grid incompatible with symmetry'),
        (3005, 'ERROR_PARTIAL_PROCESSORS_CONVERGED',   'Only some processors converged'),
        (3006, 'ERROR_PROBLEMS_COMPUTING_CHOLESKY',    'problems computing cholesky'),
        (3007, 'ERROR_UNKNOWN_MODE_SYMMETRY',          'unknown mode symmetry'),
        (3008, 'ERROR_MAXIMUM_CPU_TIME_EXCEEDED',      'Maximum CPU time exceeded'),
        (3009, 'ERROR_S_MATRIX_NOT_POSITIVE_DEFINITE', 'S matrix not positive definite'),
        (3010, 'ERROR_EIGENVECTORS_NOT_CONVERGE',      'eigenvectors failed to converge'),
        (3011, 'ERROR_DIOPN',                          'Error in routine dirop'),
        (3012, 'ERROR_STDOUT_NOT_FOUND',               'Can\'t find standard output'),
        (3012, 'ERROR_READ_WFC',                       'Error in routine read_wfc'),
    ]

    if wc.is_killed:
        return (1, 'KILLED')

    if wc.is_excepted:
        return (2, 'EXCEPTED')

    if wc.process_label == 'PwBandsWorkChain':
        return (wc.exit_status, wc.exit_message)
    if wc.process_label == 'Wannier90OptimizeWorkChain':
        return (wc.exit_status, wc.exit_message)
    if wc.process_label == 'EpwWorkChain':
        if  wc.exit_status == 404:
            # print(404, wc.pk)
            return (404, 'FAILURE_IN_WANNIER90WORKCHAIN')
        if wc.exit_status == 405:
            # print(405, wc.pk)
            return (405, 'FAILURE_IN_EPWWORKCHAIN')
        if wc.exit_status == 0:
            return (0, 'SUCCESS')
        wc_ph = wc.base.links.get_outgoing(link_label_filter='ph_base').first().node

    elif wc.process_label == 'PhBaseWorkChain':
        wc_ph = wc
    else:
        raise ValueError('Only EpwWorkChain amd PhBaseWorkChain are accepted.')

    iterations = find_iterations(wc_ph)

    max_iteration = max(iterations, key=lambda x: int(x.split('_')[1]))

    final_calcjob = wc_ph.base.links.get_outgoing(link_label_filter=max_iteration).first().node

    if final_calcjob.is_killed:
        return (1, 'KILLED')

    if final_calcjob.is_excepted:
        return (2, 'EXCEPTED')

    if 'aiida.out' not in final_calcjob.outputs.retrieved.list_object_names():

        return (3012, 'ERROR_STDOUT_NOT_FOUND')


    aiida_out = final_calcjob.outputs.retrieved.get_object_content('aiida.out')
    # stderr = final_calcjob.outputs.retrieved.get_object_content('_scheduler-stderr.txt')

    stderr = final_calcjob.get_scheduler_stderr()
    tails = ''.join(deque(StringIO(aiida_out), maxlen=200))

    short_tails = ''.join(deque(StringIO(aiida_out), maxlen=20))


    for error_code, error_flag, error_message in __ERRORS:
        if error_message in tails:
            return (error_code, error_flag)

    if 'JOB DONE' in tails:
        return (0, 'Either job finished successfully or insufficient buffer of aiida.out used')
    else:
        if 'error' in stderr:
            # print(3, wc.pk)
            return (3, 'MPICH_ERROR')
        else:
            return(999, f'{wc.pk} has no JOB DONE in aiida.out')

def get_qpoints_and_frequencies(
    wc: orm.WorkChainNode
    ):

    if wc.process_label == 'EpwWorkChain':
        wc_ph = wc.base.links.get_outgoing(link_label_filter='ph_base').first().node
    elif wc.process_label == 'PhBaseWorkChain':
        wc_ph = wc
    else:
        wc_ph = wc.base.links.get_outgoing(link_label_filter='ph_base').first().node
        if wc_ph is not None:
            raise Warning(
                'This workchain is unknown, but it called a PhBaseWorkChain'
                )
        else:
            raise ValueError('Invalid input workchain')

    if wc_ph.exit_status != 0:
        raise ValueError('The PhBaseWorkChain failed')


    iterations = find_iterations(wc_ph)

    max_iteration = max(iterations, key=lambda x: int(x.split('_')[1]))

    final_calcjob = wc_ph.base.links.get_outgoing(link_label_filter=max_iteration).first().node

    output_parameters = final_calcjob.outputs.output_parameters.get_dict()

    q_list = []
    freq_list = []

    pattern_q = re.compile(
        r'''q\s*=\s*
        \(\s*
        ([+-]?\d+\.\d+)
        \s+([+-]?\d+\.\d+)
        \s+([+-]?\d+\.\d+)
        \s*\)
        ''', re.VERBOSE)

    pattern_freq = re.compile(
        r'''freq\s*\( \s*\d+ \s*\)
        \s*=\s*
        ([+-]?\d+\.\d+)
        \s*\[THz\]\s*=\s*
        ([+-]?\d+\.\d+)
        \s*\[cm-1\]
        ''', re.VERBOSE)

    dyn0 = final_calcjob.outputs.retrieved.get_object_content('DYN_MAT/dynamical-matrix-0')
    lines = dyn0.strip().splitlines()
    nirrqpts = int(lines[1].strip())

    for iq in range(1, 1+nirrqpts):
        dyn_file = final_calcjob.outputs.retrieved.get_object_content(f'DYN_MAT/dynamical-matrix-{iq}')
        # lines = dyn_file.strip().splitlines()
        lines = dyn_file
        # for line in lines:
        q_match = pattern_q.search(lines)
        freq_match = pattern_freq.findall(lines)
        if q_match:
            qx, qy, qz = q_match.groups()
            q_list.append([float(qx), float(qy), float(qz)])
        if freq_match:
            # thz, _ = freq_match.groups()
            freq_list.append([float(thz) for thz, _ in freq_match])

    return q_list, freq_list

def check_instability(
    wc: orm.WorkChainNode,
    tolerance = -0.01
    ):
    stability = True
    info = {}

    qs, freqs = get_qpoints_and_frequencies(wc)
    q0 = qs[0]
    neg_freq0 = [f * THZ_TO_MEV for f in freqs[0] if f < 0]
    if len(neg_freq0) > 3:
        stability = False
        info[" ".join(map(str, q0))] = neg_freq0
    for q, freq in zip(qs[1:], freqs[1:]):
    # for q, freq in zip(qs, freqs):
        neg_freq = [f * THZ_TO_MEV for f in freq if f < tolerance]
        if len(neg_freq) > 0:
            stability = False
            info[" ".join(map(str, q))] = neg_freq
    return stability, info

def plot_phonon_dispersion(
    band_int_calcjob,
    prefix
    ):
    import matplotlib.pyplot as plt

    import numpy as np

    bands = band_int_calcjob.outputs.ph_band_structure.get_array('bands')

    plt.plot(np.linspace(0, 1, bands.shape[0]), bands)

    # plt.savefig(f'/home/ucl/modl/yimzhang/aiida_projects/supercon/data/{group_label}/{prefix}.pdf', format='pdf')

def is_phonon_cleaned(
    wc: orm.WorkChainNode
    ) -> bool:
    if wc.process_label == 'PhBaseWorkChain':
        return wc.outputs.remote_folder.is_cleaned
    if wc.process_label == 'EpwWorkChain':
        ph_base_wc = wc.base.links.get_outgoing(link_label_filter='ph_base').first().node
        return ph_base_wc.outputs.remote_folder.is_cleaned
    else:
        raise ValueError('Invalid input workchain')

def get_subprocess_from_epw_wc(
    epw_wc: orm.WorkChainNode,
    link_label_filter: str
    ) -> orm.WorkChainNode:

    if epw_wc.process_label in ['EpwWorkChain', 'ElectronPhononWorkChain']:
        subprocess = epw_wc.base.links.get_outgoing(link_label_filter=link_label_filter).first().node
        return subprocess
    else:
        raise ValueError('Invalid input workchain')

def clean_workdir(node, dry_run=False):
    """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""

    cleaned_calcs = []

    for called_descendant in node.called_descendants:
        if isinstance(called_descendant, orm.CalcJobNode):
            try:
                if not dry_run:
                    if 'remote_folder' in node.outputs:
                        called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                cleaned_calcs.append(called_descendant.pk)
            except (IOError, OSError, KeyError):
                pass

    return cleaned_calcs

def get_descendants(
    node: orm.WorkChainNode,
    link_type: LinkType = LinkType.CALL_WORK
    ) -> dict:
    """Get the descendant nodes of the parent workchain."""

    descendants = {}
    try:
        for node, link_type, link_label in node.base.links.get_outgoing(link_type=link_type).all():
            if link_label not in descendants:
                descendants[link_label] = []
            descendants[link_label].append(node)
        return descendants
    except AttributeError:
        return None