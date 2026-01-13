from .pw_relax import PwRelaxWorkChainAnalyser
from .base import BaseWorkChainAnalyser
from scipy.optimize import curve_fit
import numpy
from aiida_dislocation.tools import (
    A1GlidingSystem,
    A2GlidingSystem,
    B1GlidingSystem,
    B2GlidingSystem,
    C1bGlidingSystem,
    L21GlidingSystem,
    get_strukturbericht
    )

def gamma_isf(x, cG, g_isf):
    """
    Calculates the value for the first region: 0 <= x <= 1
    Formula: cG * sin^2(pi*x) + gamma_ISF * x
    Note: g_esf is included as an argument for consistency but not used here.
    """
    return cG * numpy.sin(numpy.pi * x)**2 + g_isf * x

def gamma_esf(x, cG, b, c):
    """
    Calculates the value for the second region: 0 < x <= 1
    Formula: cG * sin^2(pi*x) + gamma_ISF * (2 - c) + gamma_ESF * (x - c)
    """
    return numpy.where(
        x <= 1, 
        cG * numpy.sin(numpy.pi * x)**2 + b * x, 
        cG * numpy.sin(numpy.pi * x)**2 + c * x + (b - c)
    )

def gamma_usf(x, e_usf):
    """
    Calculates the value for the third region: 1 < x <= 2
    Formula: gamma_USF * sin^2(pi*x)
    """
    return e_usf * numpy.sin(numpy.pi * x)**2


fit_function_map = {
    'A1': {
        'gliding_system': A1GlidingSystem,
        '100': gamma_usf,
        '011': gamma_usf,
        '111': gamma_esf,
    },
    'A2': {
        'gliding_system': A2GlidingSystem,
        '100': gamma_usf,
        '011': gamma_usf,
        '111': gamma_esf,
    },
    'B1': {
        'gliding_system': B1GlidingSystem,
        '100': gamma_usf,
        '011': gamma_usf,
        '111': gamma_esf,
    },
    'B2': {
        'gliding_system': B2GlidingSystem,
        '100': gamma_usf,
        '011': gamma_usf,
        '111': gamma_isf,
    },
    'C1_b': {
        'gliding_system': C1bGlidingSystem,
        '100': gamma_usf,
        '011': gamma_usf,
        '111': gamma_esf,
    },
    'L2_1': {
        'gliding_system': L21GlidingSystem,
        '100': gamma_usf,
        '011': gamma_usf,
        '111': gamma_esf,
    },
}

class GSFEWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the GsfeWorkChain.
    """
    _RY2eV    = 13.605693122990
    _RYA22Jm2 = 4.3597447222071E-18/2 * 1E+20
    _eVA22Jm2 = 1.602176634E-19 * 1E+20
    
    @property
    def strukturbericht(self):
        return get_strukturbericht(self.scf.inputs.pw.structure.get_ase())

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
            raise AttributeError('surface_energy is not found')
        else:
            return self.process_tree.surface_energy.node
    
    def get_state(self):
        """Get the state of the workchain."""

        if self.node.is_finished_ok:
            return 'ROOT', 'finished_ok', 0
        
        # If all subprocesses are finished but main node is not, use tree traversal
        # to find the actual error in the process tree
        return self._get_state_from_tree()

    @property
    def scf_energy(self):
        """Get the energy of the scf calculation."""

        return self.scf.outputs.output_parameters.get('energy')

    @property
    def pristine_energy(self):
        """Get the pristine energy."""

        return self.process_tree.structure_01.node.outputs.output_parameters.get('energy')
    
    def get_sfe_energies(self):
        """Get the energies of the workchain."""
        from ase.formula import Formula
        from aiida_dislocation.tools import calculate_surface_area
        gliding_plane = self.node.inputs.gliding_plane.value
        gliding_system = fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(gliding_plane)

        nsteps = gliding_system.general.nsteps

        _energies =[]
        energies = {}
        # total_energy_conventional_geometry = self.scf_energy

        # faulted_formula = Formula(self.process_tree.structure_01.node.inputs.pw.structure.get_ase().get_chemical_formula())
        # _, faulted_multiplier = faulted_formula.reduce()
        # conventional_formula = Formula(self.scf.inputs.pw.structure.get_ase().get_chemical_formula())
        # _, conventional_multiplier = conventional_formula.reduce()
        surface_area = calculate_surface_area(self.scf.inputs.pw.structure.get_ase())
        for call_link_label, child in self.process_tree.children.items():
            if call_link_label.startswith('structure_'):
                total_energy_faulted_geometry = child.node.outputs.output_parameters.get('energy')
                # energy_difference = total_energy_faulted_geometry - total_energy_conventional_geometry / conventional_multiplier * faulted_multiplier
                energy_difference = total_energy_faulted_geometry - self.pristine_energy
                faulted_stacking_fault_energy = energy_difference / surface_area * self._eVA22Jm2
                _energies.append(faulted_stacking_fault_energy)

        for slipping_direction, sections in gliding_system.general.burger_vectors.items():
            energies[slipping_direction] = []
            for section in sections:
                energy_section = []
                energy_section.append(_energies.pop(0))
                for _ in section:
                    for _ in range(nsteps):
                        energy_section.append(_energies.pop(0))
                energies[slipping_direction].append(energy_section)
    
        return energies

    def get_surface_energy(self):
        """Get the surface energy."""
        from ase.formula import Formula
        from aiida_dislocation.tools import calculate_surface_area
        from aiida_dislocation.workflows.gsfe import GSFEWorkChain

        if any(
            namespace not in self.process_tree for namespace in (GSFEWorkChain._SCF_NAMESPACE, GSFEWorkChain._SURFACE_ENERGY_NAMESPACE)
        ):
            raise AttributeError(f'{GSFEWorkChain._SCF_NAMESPACE} or {GSFEWorkChain._SURFACE_ENERGY_NAMESPACE} is not found')
        
        conventional_formula = Formula(self.scf.inputs.pw.structure.get_ase().get_chemical_formula())
        _, conventional_multiplier = conventional_formula.reduce()
        
        surface_formula = Formula(self.surface_energy.inputs.pw.structure.get_ase().get_chemical_formula())
        _, surface_multiplier = surface_formula.reduce()
        
        surface_area = calculate_surface_area(self.scf.inputs.pw.structure.get_ase())
        
        total_energy_cleavaged_geometry = self.surface_energy.outputs.output_parameters.get('energy')
        energy_difference = 2*(total_energy_cleavaged_geometry - self.scf_energy * surface_multiplier / conventional_multiplier)
        surface_energy = energy_difference / surface_area * self._eVA22Jm2
        
        return surface_energy

    def serialize_faults(self):
        """Serialize the faults."""
        
        xs = {}

        gliding_plane = self.node.inputs.gliding_plane.value
        gliding_system = fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(gliding_plane)

        nsteps = gliding_system.general.nsteps
        
        for slipping_direction, sections in gliding_system.general.burger_vectors.items():
            xs[slipping_direction] = []
            for section in sections:
                x = numpy.linspace(0, 1, nsteps+1)
                for i in range(len(section)-1):
                    x = numpy.concatenate((x, numpy.linspace(i+1, i+2, nsteps+1)[1:]))
                xs[slipping_direction].append(x)
        
        return xs

    def fit_curve(self, plot=False, axis=None, **kwargs):
        """Fit the curve."""
        xs = self.serialize_faults()
        energies = self.get_sfe_energies()
        results = {}

        gliding_plane = self.node.inputs.gliding_plane.value
        gliding_system = fit_function_map[self.strukturbericht]['gliding_system'](self.strukturbericht).get_plane(gliding_plane)
        
        func = fit_function_map[self.strukturbericht][gliding_plane]
        nsteps = gliding_system.general.nsteps
        print(nsteps)

        for slipping_direction, x_section in xs.items():
            results[slipping_direction] = {}
            for x, y in zip(x_section, energies[slipping_direction]):
                x_plot = numpy.linspace(0, x[-1], 500)

                # y = numpy.array(energies[slipping_direction])

                if func == gamma_isf:
                    b = y[nsteps]
                    cG, pcov = curve_fit(
                        lambda x, cG: func(x, cG, b)
                        , x, y, maxfev=100000)

                    y_fit = func(x_plot, cG, b)

                    x_max = numpy.arcsin(-b / numpy.pi / cG) / 2 * numpy.pi
                    results[slipping_direction]['isf'] = b
                    results[slipping_direction]['usf'] = func(x_max, cG, b)[0]

                if func == gamma_esf:
                    b = y[nsteps]
                    c = y[2*nsteps-1]-y[nsteps]
                    cG, pcov = curve_fit(
                        lambda x, cG: func(x, cG, b, c)
                        , x, y, maxfev=100000)

                    y_fit = func(x_plot, cG, b, c)

                    x_max = numpy.arcsin(-b / numpy.pi / cG) / 2 * numpy.pi
                    results[slipping_direction]['isf'] = b
                    results[slipping_direction]['usf'] = func(x_max, cG, b, c)[0]
                    results[slipping_direction]['esf'] = b+c


                if func == gamma_usf:
                    e_usf, pcov = curve_fit(
                        func, x, y, maxfev=100000)

                    y_fit = func(x_plot, e_usf)

                    results[slipping_direction]['usf'] = e_usf
                    
                if plot:
                    if not axis:
                        import matplotlib.pyplot as plt
                        fig, axis = plt.subplots(figsize=(10, 6))
                    axis.scatter(x, y, color='black', zorder=5)

                    axis.plot(x_plot, y_fit, linestyle='--', label = kwargs.get('label', '$\Gamma_{ISF}$'))

                    axis.legend()
                    axis.grid(True, alpha=0.3)

        return results