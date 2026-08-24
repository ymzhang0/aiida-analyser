from ..core.base import BaseWorkChainAnalyser
from .pw_base import PwBaseAnalyser
from ..core.groupdata import DegaussKGroup
from ..visualization.convergence import configure_kpoint_distance_axis
from ..visualization.style import figure_size, styled_plot
from loguru import logger

class PwRelaxAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the PwRelaxWorkChain.
    """

    def get_source(self):
        """Get the source of the workchain."""
        source = super().get_source()
        if source is None:
            try:
                source_db, source_id = self.node.inputs.structure.base.extras.get_many(('source_db', 'source_id'))
                source = f"{source_db}-{source_id}"
            except Exception:
                self._log_source_missing()
                return None
        return source

    def get_state(self):
        """Get the state of the workchain."""
        subprocesses = []

        for label in ('init_relax', 'base_init_relax'):
            if label in self.process_tree:
                subprocesses.append((label, PwBaseAnalyser))

        iteration_labels = sorted(
            (
                child_name for child_name, child_tree in self.process_tree.children.items()
                if child_tree.node.process_label == 'PwBaseWorkChain' and child_name.startswith('iteration_')
            ),
            key=lambda label: int(label.split('_')[1]),
        )
        subprocesses.extend((label, PwBaseAnalyser) for label in iteration_labels)

        trailing_pw_bases = [
            child_name for child_name, child_tree in self.process_tree.children.items()
            if child_tree.node.process_label == 'PwBaseWorkChain'
            and child_name not in {name for name, _ in subprocesses}
        ]
        subprocesses.extend((label, PwBaseAnalyser) for label in trailing_pw_bases)

        return self._get_state_from_subprocesses(subprocesses)

class PwRelaxGroup(DegaussKGroup):

    analyser_class = PwRelaxAnalyser
    process_label = 'PwRelaxWorkChain'
    keep_duplicate_nodes = True

    @styled_plot
    def plot_structure_convergence(self, quantity='celldm1', formula=None, ax=None,
                                   degauss_values=None, kpoints_distances=None,
                                   marker='o', legend=True, xlim=None, xticks=None,
                                   xlabel=None, cubic_scale=True, **plot_kwargs):
        """Plot a relaxed cell parameter against k-point distance and degauss.

        Each curve represents one degauss. ``quantity`` accepts celldm1--6
        (or a, b, c, alpha, beta, gamma), mapped to output_structure cell
        lengths and angles. Returns ``(ax, {degauss: {distance: value}})``.
        """
        import matplotlib.pyplot as plt

        quantities = {
            'celldm1': ('cell_lengths', 0, r'$a$ ($\AA$)'),
            'celldm2': ('cell_lengths', 1, r'$b$ ($\AA$)'),
            'celldm3': ('cell_lengths', 2, r'$c$ ($\AA$)'),
            'celldm4': ('cell_angles', 0, r'$\alpha$ (deg)'),
            'celldm5': ('cell_angles', 1, r'$\beta$ (deg)'),
            'celldm6': ('cell_angles', 2, r'$\gamma$ (deg)'),
        }
        aliases = {'a': 'celldm1', 'b': 'celldm2', 'c': 'celldm3',
                   'alpha': 'celldm4', 'beta': 'celldm5', 'gamma': 'celldm6'}
        if isinstance(quantity, int):
            quantity = f'celldm{quantity}'
        quantity_key = aliases.get(str(quantity).lower(), str(quantity).lower())
        if quantity_key not in quantities:
            raise ValueError("quantity must be one of 'celldm1' through 'celldm6' "
                             "(or 'a', 'b', 'c', 'alpha', 'beta', 'gamma').")
        attribute, index, ylabel = quantities[quantity_key]

        formulas = list(self._nested_data)
        if formula is None:
            if len(formulas) != 1:
                raise ValueError(f'formula is required; available materials: {formulas}')
            formula = formulas[0]
        formula_data = self._nested_data.get(formula, {})
        if not formula_data:
            raise ValueError(f'No PwRelaxWorkChain data found for formula {formula!r}.')

        allowed_degauss = set(degauss_values) if degauss_values is not None else None
        allowed_kpoints = set(kpoints_distances) if kpoints_distances is not None else None
        values = {}
        for degauss, kpoints_data in formula_data.items():
            if allowed_degauss is not None and degauss not in allowed_degauss:
                continue
            points = {}
            for kpoints_distance, candidates in kpoints_data.items():
                if allowed_kpoints is not None and kpoints_distance not in allowed_kpoints:
                    continue
                try:
                    distance = float(kpoints_distance)
                except (TypeError, ValueError):
                    logger.warning(f'Skipping non-numeric kpoints_distance: {kpoints_distance!r}')
                    continue
                candidates = [node for node in candidates if getattr(node, 'is_finished_ok', False)]
                if not candidates:
                    continue
                node = max(candidates, key=lambda item: getattr(item, 'pk', -1))
                try:
                    points[distance] = float(getattr(node.outputs.output_structure, attribute)[index])
                except (AttributeError, IndexError, TypeError, ValueError) as exception:
                    logger.warning(f'Could not read {quantity_key} from node<{node.pk}>: {exception}')
            if points:
                values[degauss] = dict(sorted(points.items(), reverse=True))

        if ax is None:
            _, ax = plt.subplots(figsize=figure_size())
        for degauss in sorted(values, key=lambda value: str(value)):
            points = values[degauss]
            ax.plot(list(points), list(points.values()), marker=marker,
                    label=rf'$\sigma$ = {degauss} Ry', **plot_kwargs)
        configure_kpoint_distance_axis(
            ax, xlim=xlim, xticks=xticks, xlabel=xlabel, cubic_scale=cubic_scale,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(f'{formula}: {quantity_key} convergence')
        ax.grid(True, alpha=0.3)
        if legend and values:
            ax.legend(title='degauss')
        return ax, values
