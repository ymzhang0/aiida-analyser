"""Unit conversion helpers used by plotting functions."""

RY_TO_EV = 13.605693122994

_DEGAUSS_FACTORS = {
    'Ry': 1.0,
    'mRy': 1_000.0,
    'eV': RY_TO_EV,
    'meV': RY_TO_EV * 1_000.0,
}


def format_degauss(value, unit='Ry'):
    """Format a degauss value stored in Ry in the requested energy unit."""
    try:
        factor = _DEGAUSS_FACTORS[unit]
    except KeyError as exception:
        allowed = ', '.join(_DEGAUSS_FACTORS)
        raise ValueError(f'unit_degauss must be one of: {allowed}.') from exception

    try:
        converted = float(value) * factor
    except (TypeError, ValueError):
        return str(value), unit
    return f'{converted:g}', unit
