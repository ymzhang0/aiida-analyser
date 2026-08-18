from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from aiida_analyser.epw import convergence
from aiida_analyser.epw.epw_base import EpwBaseGroup


def _group_with_node(node, *, kpoint='0.30000000001', qpoint='0.5'):
    group = EpwBaseGroup.__new__(EpwBaseGroup)
    group._nested_data = {'HfRuSb': {0.02: {kpoint: {qpoint: node}}}}
    return group


def test_plot_phonon_bands_uses_child_output_and_matches_numeric_extras(monkeypatch):
    bands = object()
    child = SimpleNamespace(pk=2, outputs=SimpleNamespace(ph_band_structure=bands), called=[])
    root = SimpleNamespace(pk=1, called=[child])
    group = _group_with_node(root)
    plotted = []

    def fake_plot_bands(bands_data, *, axis, **_kwargs):
        plotted.append(bands_data)
        axis.plot([0, 1], [0, 1])

    monkeypatch.setattr(convergence, 'plot_bands', fake_plot_bands)

    fig, _ = group.plot_phonon_bands_vs_degauss(
        kpoints_distance=0.3,
        qpoints_distance=0.5,
    )

    assert plotted == [bands]
    plt.close(fig)


def test_plot_phonon_bands_reports_available_parameters_when_no_curve_matches():
    node = SimpleNamespace(pk=1, called=[])
    group = _group_with_node(node, kpoint=0.3, qpoint=0.5)

    with pytest.raises(ValueError, match='Available'):
        group.plot_phonon_bands_vs_degauss(
            kpoints_distance=0.2,
            qpoints_distance=0.5,
        )


def test_plot_phonon_bands_vs_kpoints_uses_all_matching_kpoint_nodes(monkeypatch):
    bands = [object(), object()]
    nodes = [
        SimpleNamespace(pk=index, outputs=SimpleNamespace(ph_band_structure=band), called=[])
        for index, band in enumerate(bands, start=1)
    ]
    group = EpwBaseGroup.__new__(EpwBaseGroup)
    group._nested_data = {
        'HfRuSb': {0.02: {'0.30000000001': {0.5: nodes[0]}, 0.2: {0.5: nodes[1]}}}
    }
    plotted = []

    def fake_plot_bands(bands_data, *, axis, **_kwargs):
        plotted.append(bands_data)
        axis.plot([0, 1], [0, 1])

    monkeypatch.setattr(convergence, 'plot_bands', fake_plot_bands)

    fig, _ = group.plot_phonon_bands_vs_kpoints(
        degauss=0.02,
        qpoints_distance=0.5,
        kpoints_values=[0.3, 0.2],
    )

    assert plotted == bands
    plt.close(fig)


def test_plot_phonon_bands_vs_kpoints_reports_available_parameters_when_no_curve_matches():
    node = SimpleNamespace(pk=1, called=[])
    group = _group_with_node(node, kpoint=0.3, qpoint=0.5)

    with pytest.raises(ValueError, match='Available'):
        group.plot_phonon_bands_vs_kpoints(
            degauss=0.01,
            qpoints_distance=0.5,
        )
