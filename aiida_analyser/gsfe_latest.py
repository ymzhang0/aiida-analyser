from __future__ import annotations

import typing as ty

import numpy
from aiida import orm
from scipy.optimize import curve_fit

from .base import BaseWorkChainAnalyser
from .gsfe import fit_function_map, gamma_esf, gamma_isf, gamma_usf, gamma_usf2
from .pw_base import PwBaseWorkChainAnalyser
from .pw_relax import PwRelaxWorkChainAnalyser

from aiida_dislocation.tools.structure_utils import get_strukturbericht


class GSFEWorkChainAnalyserLatest(BaseWorkChainAnalyser):
    """Analyser for the current `GSFEWorkChain` output contract."""

    @staticmethod
    def _shift_series_to_first_value(values: list[float | None]) -> list[float | None]:
        """Shift a numeric series so the first point becomes the zero reference."""
        if not values or values[0] is None:
            return values

        reference = values[0]
        return [
            None if value is None else value - reference
            for value in values
        ]

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

    @property
    def gliding_plane(self) -> str:
        if 'faulted_structure_data' not in self.node.inputs:
            return fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).default_plane

        faulted_structure_data = self.node.inputs.faulted_structure_data
        gliding_plane = faulted_structure_data.gliding_plane
        if gliding_plane:
            return gliding_plane

        return fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).default_plane

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

    def serialize_faults(self) -> dict[str, numpy.ndarray]:
        """Return the serialized GSFE path coordinate for each direction."""
        gliding_system = fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(
            self.gliding_plane
        )
        nsteps = gliding_system.general.nsteps

        serialized: dict[str, numpy.ndarray] = {}
        for direction, entries in self.get_results().items():
            serialized[direction] = numpy.array(
                [int(step) / nsteps for step in entries],
                dtype=float,
            )

        return serialized

    def get_plot_data(
        self,
        x_axis: str = 'step',
        zero_reference: bool = False,
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

            if zero_reference:
                energies = self._shift_series_to_first_value(energies)
                sfes = self._shift_series_to_first_value(sfes)

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
        zero_reference: bool = False,
        **kwargs,
    ):
        """Plot the selected GSFE quantity for each direction."""
        if value not in {'sfe', 'energy'}:
            raise ValueError(f'unsupported value `{value}`, expected `sfe` or `energy`')

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        plot_data = self.get_plot_data(x_axis=x_axis, zero_reference=zero_reference)

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

    def fit_curve(self, plot: bool = False, axis=None, zero_reference: bool = True, **kwargs):
        """Fit the GSFE curve and extract characteristic fault energies."""
        def get_gradient_shades(hex_color: str, num: int = 5) -> list[str]:
            import matplotlib.colors as mcolors

            cmap = mcolors.LinearSegmentedColormap.from_list("custom", [hex_color, "#ffffff"])
            return [mcolors.to_hex(cmap(i)) for i in numpy.linspace(0, 0.8, num)]

        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x']
        xs = self.serialize_faults()
        energies = self.get_sfe_energies()
        results: dict[str, dict[str, float]] = {}

        gliding_system = fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(
            self.gliding_plane
        )
        func = fit_function_map[self.strukturbericht][self.gliding_plane]
        nsteps = gliding_system.general.nsteps

        def direction_max_energy(direction_name: str) -> float:
            values = list(energies[direction_name].values())
            if zero_reference:
                values = self._shift_series_to_first_value(values)
            non_null_values = [value for value in values if value is not None]
            return max(non_null_values) if non_null_values else float('-inf')

        sorted_keys = sorted(energies, key=direction_max_energy, reverse=True)
        colors = get_gradient_shades(kwargs.get('color', 'black'), num=len(sorted_keys))

        for direction_name, color in zip(sorted_keys, colors):
            y_values = [energy for _, energy in sorted(energies[direction_name].items())]
            if zero_reference:
                y_values = self._shift_series_to_first_value(y_values)

            if any(value is None for value in y_values):
                raise ValueError(
                    f'Cannot fit GSFE direction `{direction_name}` because at least one SFE value is missing.'
                )

            x = xs[direction_name]
            y = numpy.array(y_values, dtype=float)
            x_plot = numpy.linspace(0.0, float(x[-1]), 500)

            results[direction_name] = {}

            if func == gamma_isf:
                (c_g,), _ = curve_fit(
                    lambda x_value, c_g: func(x_value, c_g, y[nsteps]),
                    x,
                    y,
                    maxfev=100000,
                )

                y_fit = func(x_plot, c_g, y[nsteps])
                x_max = numpy.arcsin(-y[nsteps] / numpy.pi / c_g) / 2 * numpy.pi
                usf = float(numpy.atleast_1d(func(x_max, c_g, y[nsteps]))[0])
                isf = float(y[nsteps])

                results[direction_name]['isf'] = isf
                results[direction_name]['isfe'] = isf
                results[direction_name]['usf'] = usf
                results[direction_name]['usfe'] = usf

            elif func == gamma_esf:
                b = y[nsteps]
                c = y[2 * nsteps] - y[nsteps]
                (c_g1, c_g2, c_g3, c_g4), _ = curve_fit(
                    lambda x_value, c_g1, c_g2, c_g3, c_g4: func(x_value, c_g1, c_g2, c_g3, c_g4, b, c),
                    x,
                    y,
                    maxfev=100000,
                )

                y_fit = func(x_plot, c_g1, c_g2, c_g3, c_g4, b, c)
                usf = float(numpy.max(y_fit[:250]))
                isf = float(b)
                ut = float(numpy.max(y_fit[250:]))
                esf = float(b + c)

                results[direction_name]['usf'] = usf
                results[direction_name]['usfe'] = usf
                results[direction_name]['isf'] = isf
                results[direction_name]['isfe'] = isf
                results[direction_name]['ut'] = ut
                results[direction_name]['esf'] = esf
                results[direction_name]['esfe'] = esf

            elif func == gamma_usf:
                (a, b, c, d), _ = curve_fit(func, x, y, maxfev=100000)
                y_fit = func(x_plot, a, b, c, d)
                usf = float(a + b + c + d)

                results[direction_name]['usf'] = usf
                results[direction_name]['usfe'] = usf

            elif func == gamma_usf2:
                (e_usf1, e_usf2), _ = curve_fit(func, x, y, maxfev=100000)
                y_fit = func(x_plot, e_usf1, e_usf2)
                usf = float(numpy.max(y_fit))

                results[direction_name]['usf'] = usf
                results[direction_name]['usfe'] = usf
                results[direction_name]['s'] = float(e_usf2)

            else:
                raise ValueError(
                    f'Unsupported GSFE fit function `{func}` for {self.strukturbericht} ({self.gliding_plane}).'
                )

            if plot:
                if axis is None:
                    import matplotlib.pyplot as plt

                    _, axis = plt.subplots(figsize=(10, 6))

                axis.scatter(
                    x,
                    y,
                    color=color,
                    s=50,
                    zorder=5,
                    marker=markers.pop(0),
                )

                axis.plot(
                    x_plot,
                    y_fit,
                    linestyle=kwargs.get('linestyle', '--'),
                    color=color,
                    lw=kwargs.get('lw', 1.0),
                    label=kwargs.get('label', '') + f' <{direction_name}>',
                )
                axis.grid(True, alpha=0.3)

        return results
