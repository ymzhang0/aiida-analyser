from aiida import orm
import numpy
from .base import BaseWorkChainAnalyser

class ThermoPwBaseAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the ThermoPwBaseWorkChain.
    """
    
    def get_source(self):
        """Get the source of the workchain."""
        source = super().get_source()
        if source is None:
            try:
                source_db, source_id = self.node.inputs.thermo_pw.structure.base.extras.get_many(('source_db', 'source_id'))
                source = f"{source_db}-{source_id}"
            except Exception:
                print('Source is not set')
                return None
        return source

    def get_state(self):
        """Get the state of the workchain."""
        return self._get_state_from_tree()

    def print_state(self):
        """Print the state of the workchain."""
        result = self.get_state()
        if not result:
            print(f"Can't check the state of ThermoPwBaseWorkChain<{self.node.pk}>.")
            return
        path, process_state = result
        print(f"ThermoPwBaseWorkChain<{self.node.pk}> is now {process_state} at {path}.")
    
    def get_moduli(self, modulus_type: str):
        """Get the moduli of the workchain."""
        if not self.node.is_finished_ok:
            return None
        moduli = {
            average: self.node.outputs.output_parameters.get('moduli').get(average).get(modulus_type) 
            for average in ['voigt', 'reuss', 'vrh']
            }
        return moduli

    @property
    def code(self):
        """Get the code of the workchain."""
        return self.node.inputs.thermo_pw.code

    @property
    def elastic_constants(self):
        """Get the elastic constants of the workchain."""
        if not self.node.is_finished_ok:
            return None
        return self.node.outputs.elastic_constants.get_array('elastic_constants')

    @property
    def bulk_modulus(self):
        """Get the moduli of the workchain."""
        return self.get_moduli('bulk_modulus_B')

    @property
    def young_modulus(self):
        """Get the Young modulus of the workchain."""
        return self.get_moduli('young_modulus_E')

    @property
    def shear_modulus(self):
        """Get the Shear modulus of the workchain."""
        return self.get_moduli('shear_modulus_G')

    @property
    def poisson_ratio(self):
        """Get the Poisson ratio of the workchain."""
        return self.get_moduli('poisson_ratio_n')

    @property
    def pugh_ratio(self):
        """Get the Pugh ratio of the workchain."""
        return self.get_moduli('pugh_ratio_r')

    @property
    def modified_pettifor_ratio(self):
        """Get the Pettifor ratio of the workchain."""
        bulk_modulus = self.bulk_modulus
        # Note that both the elastic constants and the bulk modulus are in kbar.
        elastic_constants = self.elastic_constants
        if bulk_modulus is None or elastic_constants is None:
            return None
        return {
            average: (elastic_constants[0][1] - elastic_constants[3][3]) / bulk_modulus[average] for average in ['voigt', 'reuss', 'vrh']
        }

    @property
    def pettifor_ratio(self):
        """Get the modified Pettifor ratio of the workchain."""
        young_modulus = self.young_modulus
        elastic_constants = self.elastic_constants
        if young_modulus is None or elastic_constants is None:
            return None
        return {
            average: (elastic_constants[0][1] - elastic_constants[3][3]) / young_modulus[average] for average in ['voigt', 'reuss', 'vrh']
        }

    def clean_workchain(self, dry_run: bool = True):
        """Clean the workchain."""
        message, success = super().clean_workchain(dry_run=dry_run)
        return message

    def get_fitting_coefficients(self):
        """Get the fitting coefficients of the workchain."""
        if not self.node.is_finished_ok:
            return None
        return self.node.outputs.output_parameters.get('elastic_constants_fitting')

    def plot_elastic_fitting(self, axis=None):
        """Plot the elastic fitting of the workchain."""
        if not self.node.is_finished_ok:
            return None
        fitting_coefficients = self.get_fitting_coefficients()
        if not axis:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        else:
            ax = axis

        for x, x_info in fitting_coefficients.items():
            for y, y_info in x_info.items():
                strains = numpy.array(y_info.get('strains'))
                stresses = numpy.array(y_info.get('stresses'))
                coefficients = numpy.array(y_info.get('coefficients'))
                ax.scatter(strains, stresses, color='blue')
                # ax.plot(strains, stresses, color='blue', label=f'{x}-{y}')
                polynomial = numpy.poly1d(coefficients[::-1])
                ax.plot(strains, 147100*polynomial(strains), color='red', label=f'{x}-{y} fitting')
        ax.legend(loc='best')  
        return ax

    def get_RMS_error(self):
        """Get the RMS error of the workchain."""
        if not self.node.is_finished_ok:
            return None

        RMS_errors = {}
        fitting_coefficients = self.get_fitting_coefficients()
        for x, x_info in fitting_coefficients.items():
            RMS_errors[x] = {}
            for y, y_info in x_info.items():
                strains = numpy.array(y_info.get('strains'))
                stresses = numpy.array(y_info.get('stresses'))
                coefficients = numpy.array(y_info.get('coefficients'))
                polynomial = numpy.poly1d(coefficients[::-1])
                errors = stresses - 147100*polynomial(strains)
                RMS_error = numpy.sqrt(numpy.mean(errors**2))
                # print(f'RMS error for {x}-{y} is {RMS_error}')
                RMS_errors[x][y] = RMS_error
        return RMS_errors