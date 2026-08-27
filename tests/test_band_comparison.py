import sys
from types import ModuleType, SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

aiida_epw_plot = ModuleType('aiida_epw.tools.plot')
aiida_epw_plot.plot_anisotropic_gap = lambda *args, **kwargs: None
sys.modules.setdefault('aiida_epw', ModuleType('aiida_epw'))
sys.modules.setdefault('aiida_epw.tools', ModuleType('aiida_epw.tools'))
sys.modules.setdefault('aiida_epw.tools.plot', aiida_epw_plot)

from aiida_analyser.visualization import plot_pw_w90_comparison


class FakeAttributes:
    def __init__(self, values):
        self.values = values

    def get(self, key):
        return self.values.get(key)


class FakeBands:
    def __init__(self, bands, labels, label_numbers):
        self._bands = np.asarray(bands)
        self.base = SimpleNamespace(attributes=FakeAttributes({
            'labels': labels,
            'label_numbers': label_numbers,
        }))

    def get_bands(self):
        return self._bands.copy()


class Parameters(dict):
    pass


def analyser(bands, fermi, *, w90=False):
    outputs = SimpleNamespace(band_structure=bands)
    if w90:
        outputs.scf = SimpleNamespace(output_parameters=Parameters(fermi_energy=fermi))
    else:
        outputs.scf_parameters = Parameters(fermi_energy=fermi)
    return SimpleNamespace(node=SimpleNamespace(outputs=outputs))


def test_plot_pw_w90_comparison_aligns_different_segment_sampling():
    pw_bands = FakeBands(np.zeros((8, 2)), ['GAMMA', 'X', 'W', 'K'], [0, 2, 3, 7])
    w90_bands = FakeBands(np.ones((10, 2)), ['GAMMA', 'X', 'W', 'K'], [0, 4, 5, 9])
    fig, axis = plt.subplots()

    result = plot_pw_w90_comparison(
        analyser(pw_bands, 0.0),
        analyser(w90_bands, 1.0, w90=True),
        axis=axis,
        annotation='settings',
    )

    assert result is axis
    assert axis.get_xlim() == pytest.approx((0, 1))
    assert axis.get_ylim() == pytest.approx((-0.8, 0.8))
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [r'$\Gamma$', 'X|W', 'K']
    assert axis.texts[0].get_text() == 'settings'
    assert axis.lines[0].get_xdata()[2] == pytest.approx(0.5)
    assert axis.lines[2].get_xdata()[4] == pytest.approx(0.5)
    plt.close(fig)


def test_plot_pw_w90_comparison_rejects_different_paths():
    pw_bands = FakeBands(np.zeros((3, 1)), ['GAMMA', 'X'], [0, 2])
    w90_bands = FakeBands(np.zeros((3, 1)), ['GAMMA', 'K'], [0, 2])

    with pytest.raises(ValueError, match='band paths do not match'):
        plot_pw_w90_comparison(
            analyser(pw_bands, 0.0),
            analyser(w90_bands, 0.0, w90=True),
        )
