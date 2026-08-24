from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from aiida_analyser.epw import supercon
from aiida_analyser.epw.supercon import SuperConGroup
from aiida_analyser.visualization.convergence import KPOINT_DISTANCE_LABEL


def _make_node(pk=1):
    return SimpleNamespace(
        pk=pk,
        outputs=SimpleNamespace(
            a2f=object(),
            output_parameters=SimpleNamespace(),
        ),
    )


def test_plot_a2f_vs_degauss(monkeypatch):
    node1 = _make_node(1)
    node2 = _make_node(2)
    group = SuperConGroup.__new__(SuperConGroup)
    group.get_a2f_nodes = lambda: {
        'HfRuSb': {
            0.02: {'0.15': {'0.5': {'0.07': node1}}},
            0.01: {'0.15': {'0.5': {'0.07': node2}}},
        }
    }

    plotted = []

    def fake_plot_a2f(a2f_arraydata, output_parameters, *, axis, **_kwargs):
        plotted.append((a2f_arraydata, output_parameters))
        axis.plot([0, 1], [0, 1])

    monkeypatch.setattr(supercon, 'plot_a2f', fake_plot_a2f)

    fig, axes = group.plot_a2f_vs_degauss(
        kpoints_distance=0.15,
        qpoints_distance=0.5,
        qfpoints_distance=0.07,
        xlabel='Custom A2F',
    )

    assert len(plotted) == 2
    assert axes[0].get_xlabel() == 'Custom A2F'
    plt.close(fig)


def test_plot_a2f_vs_kpoints(monkeypatch):
    node1 = _make_node(1)
    node2 = _make_node(2)
    group = SuperConGroup.__new__(SuperConGroup)
    group.get_a2f_nodes = lambda: {
        'HfRuSb': {
            0.02: {
                '0.30000000001': {'0.5': {'0.07': node1}},
                '0.15': {'0.5': {'0.07': node2}},
            }
        }
    }

    plotted = []

    def fake_plot_a2f(a2f_arraydata, output_parameters, *, axis, **_kwargs):
        plotted.append((a2f_arraydata, output_parameters))
        axis.plot([0, 1], [0, 1])

    monkeypatch.setattr(supercon, 'plot_a2f', fake_plot_a2f)

    fig, axes = group.plot_a2f_vs_kpoints(
        degauss=0.02,
        qpoints_distance=0.5,
        qfpoints_distance=0.07,
        kpoints_values=[0.3, 0.15],
    )

    assert len(plotted) == 2
    assert axes[0].get_xlabel() == r'$\alpha^2 F$'
    plt.close(fig)


def test_plot_allen_dynes_tc_convergence_xlabel():
    group = SuperConGroup.__new__(SuperConGroup)
    group.get_allen_dynes_tc = lambda: {
        'HfRuSb': {
            0.02: {
                0.15: {0.5: {0.05: {'Allen_Dynes_Tc': 5.0}}},
            }
        }
    }

    fig1, axes1 = group.plot_allen_dynes_tc_convergence(
        sweep='kpoints',
        qpoints_distance=0.5,
        qfpoints_distance=0.05,
    )
    assert axes1[0].get_xlabel() == KPOINT_DISTANCE_LABEL
    assert axes1[0].get_xscale() == 'function'
    assert axes1[0].get_xlim()[0] > axes1[0].get_xlim()[1]
    plt.close(fig1)

    fig2, axes2 = group.plot_allen_dynes_tc_convergence(
        sweep='qpoints',
        degauss=0.02,
        kpoints_distance=0.15,
        qfpoints_distance=0.05,
    )
    assert r'\Delta_{\mathbf{q}}' in axes2[0].get_xlabel()
    plt.close(fig2)


def test_plot_a2f_vs_kpoints_exclude_and_defaults(monkeypatch):
    node1 = _make_node(1)
    node2 = _make_node(2)
    group = SuperConGroup.__new__(SuperConGroup)
    group.get_a2f_nodes = lambda: {
        'HfRuSb': {
            0.02: {
                '0.3': {'0.5': {'0.07': node1}},
                '0.15': {'0.5': {'0.07': node2}},
            }
        }
    }

    plotted = []

    def fake_plot_a2f(a2f_arraydata, output_parameters, *, axis, **_kwargs):
        plotted.append((a2f_arraydata, output_parameters))
        axis.plot([0, 1], [0, 1])

    monkeypatch.setattr(supercon, 'plot_a2f', fake_plot_a2f)

    fig, axes = group.plot_a2f_vs_kpoints(
        degauss=None,
        qpoints_distance=0.5,
        qfpoints_distance=0.07,
        exclude_kpoints=[0.3],
        ylabel='Custom Omega',
    )

    assert len(plotted) == 1
    assert axes[0].get_ylabel() == 'Custom Omega'
    plt.close(fig)


def test_plot_a2f_raises_for_empty_material():
    group = SuperConGroup.__new__(SuperConGroup)
    group.get_a2f_nodes = lambda: {}

    with pytest.raises(ValueError, match='No A2F nodes match'):
        group.plot_a2f_vs_kpoints()

    with pytest.raises(ValueError, match='No A2F nodes match'):
        group.plot_a2f_vs_degauss()

