
from math import e
from aiida import orm
import re
import tempfile
from aiida.common.exceptions import NotExistentAttributeError
from .constants import THZ_TO_MEV


def get_phonon_wc_from_epw_wc(
    epw_wc: orm.WorkChainNode
    ) -> orm.WorkChainNode:

    if epw_wc.process_label == 'EpwWorkChain':
        try:
            ph_base_wc = epw_wc.base.links.get_outgoing(link_label_filter='ph_base').first().node
            return ph_base_wc
        except Exception as e:
            try:
                epw_calcjob = epw_wc.base.links.get_outgoing(link_label_filter='epw').first().node
                ph_base_wc = epw_calcjob.inputs.parent_folder_ph.creator.caller
                return ph_base_wc
            except Exception as e:
                raise ValueError(f"Failed to get phonon workchain from EpwWorkChain {epw_wc.pk}: {e}")
    else:
        raise ValueError('Invalid input workchain')

def check_stability_matdyn_base(
    workchain: orm.WorkChainNode,
    tolerance: float = -5.0 # cm^{-1}
    ) -> tuple[bool, str]:
    from ..data.constants import THZ_TO_CM
    import numpy
    """Check if the matdyn.x interpolated phonon band structure is stable."""

    bands = workchain.outputs.output_phonon_bands.get_bands() * THZ_TO_CM # unit: THz
    min_freq = numpy.min(bands)
    if min_freq < tolerance:
        return (
            False,
            f'The phonon from `matdyn_base` is unstable.\nWith the minimum frequency {min_freq:.2f} cm^{-1}.')
    else:
        return (True, f'The phonon from `matdyn_base` is stable, with the minimum frequency {min_freq:.2f} cm^{-1}.')

def check_stability_epw_bands(
    workchain: orm.WorkChainNode,
    tolerance: float = -5.0 # meV ~ 8.1 cm-1
    ) -> tuple[bool, str, float]:
    """Check if the epw.x interpolated phonon band structure is stable."""
    import numpy
    ph_bands = workchain.outputs.bands.ph_band_structure.get_bands()
    min_freq = numpy.min(ph_bands)
    max_freq = numpy.max(ph_bands)

    if min_freq < tolerance:
        return (False, f'The phonon from `epw_bands` is unstable, with the minimum frequency {min_freq:.2f} cm^{-1}.', max_freq)
    else:
        return (True, f'The phonon from `epw_bands` is stable, with the minimum frequency {min_freq:.2f} cm^{-1}.', max_freq)
