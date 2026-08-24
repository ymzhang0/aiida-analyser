import pytest

from aiida_analyser.visualization.units import format_degauss


@pytest.mark.parametrize(
    ('unit', 'expected'),
    [
        ('Ry', '0.02'),
        ('mRy', '20'),
        ('eV', '0.272114'),
        ('meV', '272.114'),
    ],
)
def test_format_degauss_converts_from_ry(unit, expected):
    assert format_degauss(0.02, unit) == (expected, unit)


def test_format_degauss_defaults_to_ry():
    assert format_degauss(0.02) == ('0.02', 'Ry')


def test_format_degauss_rejects_unknown_unit():
    with pytest.raises(
        ValueError, match='unit_degauss must be one of: Ry, mRy, eV, meV'
    ):
        format_degauss(0.02, 'joule')
