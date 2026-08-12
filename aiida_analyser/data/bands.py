"""Utilities for creating :class:`aiida.orm.BandsData` from QE XML output."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from aiida import orm
from qe_tools import CONSTANTS


def load_bands_from_qe_xml(path: str | Path) -> orm.BandsData:
    """Read a Quantum ESPRESSO QEXSD XML file into an unstored ``BandsData`` node.

    The eigenvalues in QE XML are expressed in Hartree and the k-points in
    ``2 pi / alat``.  The returned node uses eV and Cartesian reciprocal-space
    coordinates in inverse Angstrom, respectively.  Collinear spin-polarized
    calculations are returned with a leading spin dimension.

    Parameters
    ----------
    path
        Path to a QE ``data-file-schema.xml`` (or equivalent QEXSD) file.

    Returns
    -------
    aiida.orm.BandsData
        An unstored node containing k-points, eigenvalues, occupations, and
        the output cell needed to interpret Cartesian k-points.

    Raises
    ------
    ValueError
        If the XML does not contain a complete QE band structure.
    """
    root = ElementTree.parse(Path(path)).getroot()
    output = _child(root, 'output')
    atomic_structure = _child(output, 'atomic_structure')
    band_structure = _child(output, 'band_structure')
    ks_energies = _children(band_structure, 'ks_energies')

    if not ks_energies:
        raise ValueError('The XML file does not contain any `ks_energies` entries.')

    alat_bohr = _float_attribute(atomic_structure, 'alat')
    alat_angstrom = alat_bohr * CONSTANTS.bohr_to_ang
    cell = np.array([_floats(_child(_child(atomic_structure, 'cell'), name).text) for name in ('a1', 'a2', 'a3')])
    cell *= CONSTANTS.bohr_to_ang

    kpoints = []
    weights = []
    eigenvalues = []
    occupations = []

    for state in ks_energies:
        kpoint = _child(state, 'k_point')
        kpoints.append(np.array(_floats(kpoint.text)) * 2 * np.pi / alat_angstrom)
        weights.append(_float_attribute(kpoint, 'weight'))
        eigenvalues.append(_floats(_child(state, 'eigenvalues').text))
        occupations.append(_floats(_child(state, 'occupations').text))

    eigenvalues_array = np.asarray(eigenvalues, dtype=float) * CONSTANTS.hartree_to_ev
    occupations_array = np.asarray(occupations, dtype=float)

    if _boolean(_child(band_structure, 'lsda').text):
        eigenvalues_array, occupations_array = _split_spin_channels(
            band_structure, eigenvalues_array, occupations_array
        )
    else:
        # QE reports occupation per spin for non-spin-polarized calculations.
        occupations_array *= 2.0

    kpoints_data = orm.KpointsData()
    kpoints_data.set_cell(cell)
    kpoints_data.set_kpoints(kpoints, cartesian=True, weights=weights)

    bands_data = orm.BandsData()
    bands_data.set_kpointsdata(kpoints_data)
    bands_data.set_bands(eigenvalues_array, units='eV', occupations=occupations_array)
    return bands_data


def _split_spin_channels(
    band_structure: ElementTree.Element,
    eigenvalues: np.ndarray,
    occupations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split QE's concatenated collinear-spin bands into the AiiDA convention."""
    bands_up = _integer(_child(band_structure, 'nbnd_up').text)
    bands_down = _integer(_child(band_structure, 'nbnd_dw').text)

    if bands_up != bands_down:
        raise ValueError(
            'The XML contains unequal numbers of spin-up and spin-down bands, which BandsData cannot represent.'
        )
    if eigenvalues.shape[1] != bands_up + bands_down:
        raise ValueError('The eigenvalue count does not match `nbnd_up + nbnd_dw`.')

    return (
        np.array([eigenvalues[:, :bands_up], eigenvalues[:, bands_up:]]),
        np.array([occupations[:, :bands_up], occupations[:, bands_up:]]),
    )


def _child(element: ElementTree.Element, name: str) -> ElementTree.Element:
    """Return a direct child regardless of the XML namespace."""
    for child in element:
        if _local_name(child.tag) == name:
            return child
    raise ValueError(f'Missing required `{name}` element in the QE XML file.')


def _children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    """Return all direct children with ``name``, regardless of XML namespace."""
    return [child for child in element if _local_name(child.tag) == name]


def _local_name(tag: str) -> str:
    """Strip an XML namespace from a tag name."""
    return tag.rsplit('}', maxsplit=1)[-1]


def _floats(text: str | None) -> list[float]:
    """Parse whitespace-delimited floating-point values from XML text."""
    if text is None:
        raise ValueError('Expected numerical XML content but found an empty element.')
    return [float(value) for value in text.split()]


def _float_attribute(element: ElementTree.Element, name: str) -> float:
    """Read a required floating-point XML attribute."""
    try:
        return float(element.attrib[name])
    except KeyError as exception:
        raise ValueError(f'Missing required `{name}` attribute in the QE XML file.') from exception


def _integer(text: str | None) -> int:
    """Parse a required integer XML value."""
    values = _floats(text)
    if len(values) != 1:
        raise ValueError('Expected a single integer value in the QE XML file.')
    return int(values[0])


def _boolean(text: str | None) -> bool:
    """Parse a QE XML boolean value."""
    if text is None:
        raise ValueError('Expected a boolean XML value but found an empty element.')
    if text.strip().lower() == 'true':
        return True
    if text.strip().lower() == 'false':
        return False
    raise ValueError(f'Invalid QE XML boolean value: {text!r}.')
