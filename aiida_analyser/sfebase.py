from aiida import orm
from .base import BaseWorkChainAnalyser
from .pw_base import PwBaseWorkChainAnalyser
from .pw_relax import PwRelaxWorkChainAnalyser
from .layer_relax import LayerRelaxWorkChainAnalyser

class SFEBaseWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the SFEBaseWorkChain.
    """
    _RY2eV    = 13.605693122990
    _RYA22Jm2 = 4.3597447222071E-18/2 * 1E+20
    _eVA22Jm2 = 1.602176634E-19 * 1E+20
    
    @property
    def relax(self):
        if 'relax' not in self.process_tree:
            raise AttributeError('relax is not found')
        else:
            return self.process_tree.relax.node

    @property
    def scf(self):
        if 'scf' not in self.process_tree:
            raise AttributeError('scf is not found')
        else:
            return self.process_tree.scf.node

    @property
    def surface_energy(self):
        if 'surface_energy' not in self.process_tree:
            raise AttributeError('sfe is not found')
        else:
            return self.process_tree.surface_energy.node

    def get_source(self):
        """Get the source of the workchain."""
        return super().get_source()


    def fit_spacing_energy(self, plot=False):
        """Fit the spacing energy."""
        import numpy as np
        import matplotlib.pyplot as plt

        energies = self.get_faulted_stacking_fault_energies()

        x = np.array(list(energies.keys()))
        y = np.array(list(energies.values()))

        def fit_and_find_min(degree):
            # fit the spacing energy
            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            
            # find the minimum value in the data range
            # create a fine sampling point to search the minimum value
            x_fine = np.linspace(min(x), max(x), 1000)
            y_fine = poly(x_fine)
            
            idx_min = np.argmin(y_fine)
            return x_fine[idx_min], y_fine[idx_min], poly

        # 执行 4 次和 5 次拟合
        min_x4, min_y4, poly4 = fit_and_find_min(4)


        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, color='black', label='Data Points', zorder=5)

            x_plot = np.linspace(min(x), max(x), 500)
            plt.plot(x_plot, poly4(x_plot), label='4th Degree Fit', linestyle='--')

            plt.scatter([min_x4], [min_y4], color='blue', marker='v', label='Min (4th)')

            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        return min_x4, min_y4, poly4

    def get_energies(self):
        layer_relax_analyser = LayerRelaxWorkChainAnalyser(self.layer_relax)
        return layer_relax_analyser.get_energies()
        
    def get_faulted_stacking_fault_energies(self):
        """Get the faulted stacking fault energies."""
        from ase.formula import Formula
        from aiida_dislocation.tools import calculate_surface_area

        faulted_stacking_fault_energies = {}

        if 'layer_relax' not in self.process_tree or 'scf' not in self.process_tree:
            raise AttributeError('layer_relax or scf is not found')
        
        if not self.scf.is_finished_ok:
            raise AttributeError('scf is not finished ok')

        conventional_structure = self.scf.inputs.pw.structure
        faulted_structure = self.process_tree.layer_relax.relax_1.node.inputs.structure
        total_energy_conventional_geometry = self.scf.outputs.output_parameters.get('energy')

        total_energy_faulted_geometry = self.get_energies()
        faulted_formula = Formula(faulted_structure.get_ase().get_chemical_formula())
        _, faulted_multiplier = faulted_formula.reduce()
        conventional_formula = Formula(conventional_structure.get_ase().get_chemical_formula())
        _, conventional_multiplier = conventional_formula.reduce()
        surface_area = calculate_surface_area(conventional_structure.get_ase().cell)
        for spacing, total_energy_faulted_geometry in total_energy_faulted_geometry.items():
            energy_difference = total_energy_faulted_geometry - total_energy_conventional_geometry / conventional_multiplier * faulted_multiplier
            faulted_stacking_fault_energy = energy_difference / surface_area * self._eVA22Jm2
            faulted_stacking_fault_energies[spacing] = faulted_stacking_fault_energy
        
        return faulted_stacking_fault_energies

    def clean_workchain(self, exempted_states=[], dry_run=True):
        """Clean the workchain."""
        path, process_state, exit_code = self.get_state()
        message = f'Process<{self.node.pk}> is now {process_state} at {path} with exit code {exit_code}. Please check if you really want to clean this workchain.\n'
        if process_state in exempted_states:
            print(message)
            return message, False

        message, success = super().clean_workchain(dry_run=dry_run)
        return message, True

