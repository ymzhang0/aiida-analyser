from __future__ import annotations

import typing as ty

import numpy
from aiida import orm

from .base import BaseWorkChainAnalyser
from .pw_base import PwBaseWorkChainAnalyser
from .pw_relax import PwRelaxWorkChainAnalyser

from aiida_dislocation.tools.structure_utils import get_strukturbericht


class GSFEWorkChainAnalyserLatest(BaseWorkChainAnalyser):
    """Analyser for the current `GSFEWorkChain` output contract."""

    @property
    def relax(self) -> orm.WorkChainNode:
        if 'relax' not in self.process_tree:
            raise AttributeError('relax is not found')
        return self.process_tree.relax.node

    @property
    def scf(self) -> orm.WorkChainNode:
        if 'scf' not in self.process_tree:
            raise AttributeError('scf is not found')
        return self.process_tree.scf.node

    @property
    def strukturbericht(self) -> str:
        if 'scf' in self.process_tree:
            structure = self.scf.inputs.pw.structure
        else:
            structure = self.node.inputs.structure
        return get_strukturbericht(structure.get_ase())

    @property
    def results_node(self) -> orm.Dict:
        for label in ('results', 'gsfe_results'):
            if label in self.node.outputs:
                return self.node.outputs[label]
        raise AttributeError('GSFE results output is not found')

    @property
    def surface_area(self) -> float | None:
        return self.results_node.get_dict().get('surface_area_angstrom2')

    @property
    def conventional_energy(self) -> float | None:
        return self.results_node.get_dict().get('conventional_energy_ev')

    def get_state(self) -> tuple[str, str, int]:
        """Get the state of the workchain."""
        subprocesses: tuple[tuple[str, type[BaseWorkChainAnalyser]], ...] = (
            ('relax', PwRelaxWorkChainAnalyser),
            ('scf', PwBaseWorkChainAnalyser),
        )
        return self._get_state_from_subprocesses(subprocesses)

    def get_results(self) -> dict[str, dict[str, dict[str, ty.Any]]]:
        """Return the nested GSFE result payload keyed by direction and step."""
        payload = self.results_node.get_dict()
        results = payload.get('results', payload)

        normalized: dict[str, dict[str, dict[str, ty.Any]]] = {}
        for direction, entries in results.items():
            normalized[direction] = {
                str(step): dict(value)
                for step, value in sorted(entries.items(), key=lambda item: int(item[0]))
            }
        return normalized

    def get_direction_results(self, direction_name: str) -> dict[str, dict[str, ty.Any]]:
        """Return all step results for a specific slip direction."""
        results = self.get_results()
        if direction_name not in results:
            raise KeyError(f'direction `{direction_name}` is not present in GSFE results')
        return results[direction_name]

    def get_sfe_energies(self) -> dict[str, dict[int, float | None]]:
        """Return only the SFE values grouped by direction and step."""
        return {
            direction: {
                int(step): entry.get('sfe')
                for step, entry in entries.items()
            }
            for direction, entries in self.get_results().items()
        }

    def get_total_energies(self) -> dict[str, dict[int, float | None]]:
        """Return only the total energies grouped by direction and step."""
        return {
            direction: {
                int(step): entry.get('energy')
                for step, entry in entries.items()
            }
            for direction, entries in self.get_results().items()
        }

    def get_plot_data(
        self,
        x_axis: str = 'step',
    ) -> dict[str, dict[str, list[float | None]]]:
        """Return simple plot-ready x/y arrays for each GSFE direction."""
        plot_data: dict[str, dict[str, list[float | None]]] = {}

        for direction, entries in self.get_results().items():
            xs: list[float] = []
            energies: list[float | None] = []
            sfes: list[float | None] = []

            for step, entry in entries.items():
                if x_axis == 'displacement':
                    xs.append(float(numpy.linalg.norm(entry.get('total_cell_shift', [0.0, 0.0, 0.0]))))
                else:
                    xs.append(float(step))

                energies.append(entry.get('energy'))
                sfes.append(entry.get('sfe'))

            plot_data[direction] = {
                'x': xs,
                'energy': energies,
                'sfe': sfes,
            }

        return plot_data

    def plot(
        self,
        ax=None,
        value: str = 'sfe',
        x_axis: str = 'step',
        **kwargs,
    ):
        """Plot the selected GSFE quantity for each direction."""
        if value not in {'sfe', 'energy'}:
            raise ValueError(f'unsupported value `{value}`, expected `sfe` or `energy`')

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        plot_data = self.get_plot_data(x_axis=x_axis)

        for direction, data in plot_data.items():
            ax.plot(
                data['x'],
                data[value],
                label=kwargs.get('label_prefix', '') + direction,
                marker=kwargs.get('marker', 'o'),
                linestyle=kwargs.get('linestyle', '-'),
                color=kwargs.get('color', None),
                alpha=kwargs.get('alpha', 1.0),
            )

        return ax
