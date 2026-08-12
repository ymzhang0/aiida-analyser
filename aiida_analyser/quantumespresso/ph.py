from aiida import orm

from ..core.constants import THZ_TO_CM


def _get_outgoing_node(node: orm.WorkChainNode, link_label: str):
    """Return the first outgoing node matching a link label, if present."""
    result = node.base.links.get_outgoing(link_label_filter=link_label).first()
    return None if result is None else result.node


def get_phonon_wc_from_epw_wc(epw_wc: orm.WorkChainNode) -> orm.WorkChainNode:
    if epw_wc.process_label != 'EpwWorkChain':
        raise ValueError('Invalid input workchain')

    ph_base_wc = _get_outgoing_node(epw_wc, 'ph_base')
    if ph_base_wc is not None:
        return ph_base_wc

    epw_calcjob = _get_outgoing_node(epw_wc, 'epw')
    if epw_calcjob is None:
        raise ValueError(f'Failed to get phonon workchain from EpwWorkChain {epw_wc.pk}: missing `epw` subprocess')

    try:
        return epw_calcjob.inputs.parent_folder_ph.creator.caller
    except AttributeError as exception:
        raise ValueError(
            f'Failed to get phonon workchain from EpwWorkChain {epw_wc.pk}: broken `parent_folder_ph` provenance'
        ) from exception


def check_stability_matdyn_base(
    workchain: orm.WorkChainNode,
    tolerance: float = -5.0,
) -> tuple[bool, str]:
    """Check if the matdyn.x interpolated phonon band structure is stable."""
    import numpy

    bands = workchain.outputs.output_phonon_bands.get_bands() * THZ_TO_CM
    min_freq = numpy.min(bands)
    if min_freq < tolerance:
        return (
            False,
            f'The phonon from `matdyn_base` is unstable.\nWith the minimum frequency {min_freq:.2f} cm^-1.',
        )

    return (
        True,
        f'The phonon from `matdyn_base` is stable, with the minimum frequency {min_freq:.2f} cm^-1.',
    )


def check_stability_epw_bands(
    workchain: orm.WorkChainNode,
    tolerance: float = -5.0,
) -> tuple[bool, str, float]:
    """Check if the epw.x interpolated phonon band structure is stable."""
    import numpy

    ph_bands = workchain.outputs.bands.ph_band_structure.get_bands()
    min_freq = numpy.min(ph_bands)
    max_freq = numpy.max(ph_bands)

    if min_freq < tolerance:
        return (
            False,
            f'The phonon from `epw_bands` is unstable, with the minimum frequency {min_freq:.2f} cm^-1.',
            max_freq,
        )

    return (
        True,
        f'The phonon from `epw_bands` is stable, with the minimum frequency {min_freq:.2f} cm^-1.',
        max_freq,
    )
