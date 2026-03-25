from aiida_analyser.plot import plot_bands, plot_epw_interpolated_bands
import numpy
from collections import defaultdict
from aiida import orm
from ..quantumespresso.ph import check_stability_matdyn_base
from ..base import BaseWorkChainAnalyser
from ..wannier.wannier90 import Wannier90WorkChainAnalyser
from ..quantumespresso.ph_base import PhBaseWorkChainAnalyser
from .epw_base import EpwBaseWorkChainAnalyser

class EpwPrepWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the EpwPrepWorkChain.
    """

    @property
    def w90_intp(self):
        if 'w90_intp' not in self.process_tree:
            raise AttributeError('w90_intp is not found')
        else:
            return self.process_tree.w90_intp.node

    @property
    def ph_base(self):
        if 'ph_base' not in self.process_tree:
            raise AttributeError('ph_base is not found')
        else:
            return self.process_tree.ph_base.node
    @property
    def ph_base_analyser(self):
        if self.ph_base is None:
            raise AttributeError('ph_base is not found')
        else:
            return PhBaseWorkChainAnalyser(self.process_tree.ph_base.node)

    @property
    def epw_base(self):
        if self.process_tree.epw_base.node is None:
            raise ValueError('epw_base is not found')
        else:
            return self.process_tree.epw_base.node

    @property
    def epw_bands(self):
        if 'epw_bands' not in self.process_tree:
            raise ValueError('epw_bands is not found')
        if self.process_tree.epw_bands.node is None:
            raise ValueError('epw_bands is not found')
        return self.process_tree.epw_bands.node

    def get_source(self):
        """Get the source of the workchain."""
        return super().get_source()


    def get_state(self):
        """Get the state of the workchain."""
        return self._get_state_from_subprocesses([
            ('w90_bands', Wannier90WorkChainAnalyser),
            ('ph_base', PhBaseWorkChainAnalyser),
            ('epw_base', EpwBaseWorkChainAnalyser),
            ('epw_bands', EpwBaseWorkChainAnalyser),
        ])

    def check_stability_matdyn_base(self):
        """Get the qpoints and frequencies of the matdyn_base workchain."""
        if 'matdyn_base' not in self.process_tree:
            raise AttributeError('matdyn_base is not found')

        matdyn_base = self.process_tree.matdyn_base.node
        if not matdyn_base.is_finished_ok:
            raise ValueError('matdyn_base is not finished ok')

        return check_stability_matdyn_base(matdyn_base)

    def clean_workchain(self, exempted_states=[], dry_run=True):
        """Clean the workchain."""
        path, process_state, exit_code = self.get_state()
        message = f'Process<{self.node.pk}> is now {process_state} at {path} with exit code {exit_code}. Please check if you really want to clean this workchain.\n'
        if process_state in exempted_states:
            print(message)
            return message, False

        message, success = super().clean_workchain(dry_run=dry_run)
        return message, True


class EpwPrepConvergenceData:

    def __init__(self, groups = []):
        self._groups = groups
        # Data structure: Material -> Degauss -> K_Dist -> {'PwRelaxWorkChain': node, 'q_dist': {Q_Dist -> {'EpwPrepWorkChain': node, 'supercon': node}}}
        self._data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: {
                        'PwRelaxWorkChain': None,
                        'q_dist': defaultdict(
                            lambda: {
                                'EpwPrepWorkChain': None, 
                            }
                        )
                    }
                )
            )
        )
        self.get_data()

    @property
    def groups(self):
        return self._groups

    @property
    def data(self):
        return self._data

    @staticmethod
    def check_protocol(node):
        extras = node.base.extras.all
        if node.process_label in ['PwRelaxWorkChain']:
            for key in ['formula', 'source_db', 'source_id', 'kpoints_distance', 'degauss', ]:
                if key not in extras:
                    raise Warning(f'Extra {key} is not found in node<{node.pk}>')
        else:
            for key in ['formula', 'source_db', 'source_id', 'kpoints_distance', 'degauss', 'qpoints_distance']:
                if key not in extras:
                    raise Warning(f'Extra {key} is not found in node<{node.pk}>')
                
        return True

    def get_data(self):
        for grpname in self._groups:
            group = orm.load_group(grpname)
            for node in group.nodes:
                try:
                    extras = node.base.extras.all
                    self.check_protocol(node)
                    
                    mat_key = f"{extras['source_db']}-{extras['source_id']}-{extras['formula']}"
                    
                    # Structure: Material -> Degauss -> K_Dist -> ...
                    if node.process_label in ['PwRelaxWorkChain']:
                        self._data[mat_key][extras['degauss']][extras['kpoints_distance']]['PwRelaxWorkChain'] = node
                    elif node.process_label in ['EpwPrepWorkChain']:
                        self._data[mat_key][extras['degauss']][extras['kpoints_distance']]['q_dist'][extras['qpoints_distance']]['EpwPrepWorkChain'] = node
                except Exception as e:
                    # Provide more context in error message
                    raise ValueError(f'Node<{node.pk}> processing failed: {e}')

    def get_table(self):
        import pandas as pd
        import numpy as np

        def get_status_string(node):
            if node is None:
                return 'N/A'

            if not node.is_terminated:
                return '⏳'
            if node.is_finished_ok:
                return '✅'
            elif node.is_failed:
                return f'❌ ({node.exit_status})'
            elif node.is_excepted:
                return '⚠️ Excepted'
            elif node.is_killed:
                return '💀 Killed'
            else:
                return f'🏃 {node.process_state.value}'

        flattened_list = []

        # Loop variables matching new dictionary structure:
        # Material -> Degauss -> K_Dist -> {'relax': ..., 'q_dist': ...}
        for material, degauss_dict in self._data.items():
            for degauss, k_dist_dict in degauss_dict.items():
                for k_dist, content in k_dist_dict.items():
                    
                    relax_node = content['PwRelaxWorkChain']
                    q_dist_data = content['q_dist']
                    relax_pk = relax_node.pk if relax_node else None
                    
                    # Base relax row
                    flattened_list.append({
                        'Material': material,
                        'Degauss': degauss,
                        'K_Density': k_dist,
                        'Q_Density': '-', 
                        'Type': 'PwRelaxWorkChain',
                        'Status': get_status_string(relax_node) + f" ({relax_pk})",
                    })
                    
                    if q_dist_data:
                        for q_dist, types_dict in q_dist_data.items():
                            epw_node = types_dict.get('EpwPrepWorkChain')
                            if epw_node:
                                flattened_list.append({
                                    'Material': material,
                                    'Degauss': degauss,
                                    'K_Density': k_dist,
                                    'Q_Density': q_dist,
                                    'Type': 'EpwPrepWorkChain',
                                    'Status': get_status_string(epw_node) + f" ({epw_node.pk})",
                                })
                            

        if not flattened_list:
            return pd.DataFrame()

        df = pd.DataFrame(flattened_list)
        
        pivot_df = df.pivot(
            index=['Degauss', 'K_Density', 'Q_Density', 'Type'],
            columns='Material',
            values='Status'
        )

        pivot_df = pivot_df.fillna('')

        # Sort columns (Materials) alphabetically
        pivot_df = pivot_df.sort_index(axis=1)

        return pivot_df

    def get_epw_bands_nodes(self):
        """
        Get epw bands node.
        """
        # Structure: Material -> Degauss -> K_Dist -> Q_Dist -> AllenDynesTc
        nodes = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: None
                    )
                )
            )
        )
        for material, degauss_dict in self._data.items():
            for degauss, k_dist_dict in degauss_dict.items():
                for k_dist, q_dist_dict in k_dist_dict.items():
                    q_dist_data = q_dist_dict['q_dist']
                    for q_dist, types_dict in q_dist_data.items():
                        node = types_dict.get('EpwPrepWorkChain')
                        if node:
                            analyser = EpwPrepWorkChainAnalyser(node)
                            try:
                                nodes[material][degauss][k_dist][q_dist] = analyser.epw_bands
                            except Exception as e:
                                print(f'Node<{node.pk}> processing failed: {e}')
        
        # Convert nested defaultdict to regular dict for cleaner output
        def default_to_regular(d):
            if isinstance(d, defaultdict):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d

        return default_to_regular(nodes)

    def plot_elbands(self):
        import matplotlib.pyplot as plt

        materials = sorted(self._data.keys())
        num_materials = len(materials)
        if num_materials == 0:
            print("No data to plot.")
            return

        fig, axs = plt.subplots(2, num_materials, figsize=(24, 4*num_materials), squeeze=False)
        # axs = axs.flatten()

        for i, material in enumerate(materials):
            degauss_dict = self._data[material]
            
            cmap = plt.get_cmap('Blues')
            blues = cmap(numpy.linspace(0.3, 1.0, 5)).tolist()
            cmap = plt.get_cmap('Reds')
            reds = cmap(numpy.linspace(0.3, 1.0, 5)).tolist()
            
            for degauss, k_dist_dict in degauss_dict.items():
                for k_dist, content in k_dist_dict.items():
                    # Check q_dist for EpwPrep nodes
                    q_dist_dict = content['q_dist']
                    
                    for q_dist, types in q_dist_dict.items():
                        node = types.get('EpwPrepWorkChain')
                        if node and node.is_finished_ok:
                             # Use Analyser to get bands
                            try:
                                analyser = EpwPrepWorkChainAnalyser(node)

                                plot_epw_interpolated_bands(
                                    analyser.process_tree.epw_bands.node,
                                    axes = axs[:, i], 
                                    label = f'{degauss} {k_dist} {q_dist}',
                                    color_el = blues.pop() if blues else 'blue',
                                    color_ph = reds.pop() if reds else 'red',
                                    )
                            except Exception as e:
                                print(f"Failed to plot {material} {degauss} {k_dist} {q_dist}: {e}")
                                continue # Skip if failing

            axs[0, i].set_title(material)

        # Cleanup unused axes
        for j in range(i + 1, len(axs)):
            axs[:, j].axis('off')

        fig.tight_layout()


    def plot_a2f(
        self,
        axis = None,
        show_data = False,
        **kwargs,
        ):
        from matplotlib import pyplot as plt
        import numpy

        # Identify all unique parameters to set up grid
        materials = sorted(self._data.keys())
        all_k_densities = set()
        for mat in materials:
            for d in self._data[mat].values():
                 all_k_densities.update(d.keys())
                 
        k_densities = sorted(list(all_k_densities), reverse=True) # Sort descending

        num_materials = len(materials)
        num_k_dens = len(k_densities)
        
        if num_materials == 0 or num_k_dens == 0:
            print("No data to plot.")
            return

        fig, axs = plt.subplots(
            num_k_dens, num_materials, 
            figsize=(5*num_materials, 4*num_k_dens),
            squeeze=False 
        )

        cmap_name = kwargs.get('cmap', 'viridis')
        cmap = plt.get_cmap(cmap_name)
        
        for j, material in enumerate(materials):
            degauss_dict = self._data[material]
            
            for i, k_dist in enumerate(k_densities):
                ax = axs[i, j]
                
                # Check if this K_Dist exists for this material (in any degauss group)
                # Structure: Mat -> Degauss -> K_Dist
                
                found_data = False
                sorted_degausses = sorted(degauss_dict.keys())
                colors = cmap(numpy.linspace(0, 1, len(sorted_degausses)))
                
                for d_idx, degauss in enumerate(sorted_degausses):
                    k_dist_dict = degauss_dict[degauss]
                    
                    if k_dist in k_dist_dict:
                        content = k_dist_dict[k_dist]
                        q_dist_dict = content['q_dist']
                        
                        for q_dist, types in q_dist_dict.items():
                            node = types.get('epwprep')
                            
                            if node is None or not node.is_finished_ok:
                                continue
                            
                            try:
                                analyser = EpwPrepWorkChainAnalyser(node)
                                epw_node = analyser.epw_bands
                                if 'a2f_data' in epw_node.outputs:
                                    a2f_data = epw_node.outputs.a2f_data
                                elif 'a2f' in epw_node.outputs:
                                    a2f_data = epw_node.outputs.a2f
                                else:
                                    continue
                                    
                                w = a2f_data.get_array('frequency')
                                spectral = a2f_data.get_array('a2f') 

                                label = f"D={degauss}"
                                if len(q_dist_dict) > 1:
                                    label += f", Q={q_dist}"

                                if kwargs.get('do_a2f', True):
                                    y_val = spectral
                                    if len(spectral.shape) > 1:
                                        if spectral.shape[1] > 9:
                                            y_val = spectral[:, 9]
                                        else:
                                            y_val = spectral[:, 0]
                                    
                                    ax.plot(
                                        y_val,
                                        w,
                                        color=colors[d_idx],
                                        label=label
                                    )
                                found_data = True
                                
                            except Exception as e:
                                print(f"Error extracting/plotting for node {node.pk}: {e}")
                                continue

                if not found_data:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                else:
                    # Formatting
                    ax.set_title(f"{material}\nK={k_dist}")
                    if i == num_k_dens - 1:
                        ax.set_xlabel(r"$\alpha^2F$")
                    if j == 0:
                        ax.set_ylabel(r"$\omega$")
                    ax.legend(fontsize='x-small')

        plt.tight_layout()
        return fig, axs

class EpwPrepData:

    def __init__(self, groups = []):
        self._groups = groups
        # Data structure: Material -> Degauss -> K_Dist -> {'PwRelaxWorkChain': node, 'q_dist': {Q_Dist -> {'EpwPrepWorkChain': node, 'supercon': node}}}
        self._data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: None
                    )
                )
            )
        )
        self.get_data()

    @property
    def groups(self):
        return self._groups

    @property
    def data(self):
        return self._data

    @staticmethod
    def check_protocol(node):
        extras = node.base.extras.all
        if node.process_label in ['EpwPrepWorkChain']:
            for key in ['formula', 'source_db', 'source_id', 'kpoints_distance_scf', 'degauss', 'qpoints_distance']:
                if key not in extras:
                    raise Warning(f'Extra {key} is not found in node<{node.pk}>')
                
        return True

    def get_data(self):
        for grpname in self._groups:
            group = orm.load_group(grpname)
            for node in group.nodes:
                try:
                    extras = node.base.extras.all
                    self.check_protocol(node)
                    
                    mat_key = f"{extras['source_db']}-{extras['source_id']}-{extras['formula']}"
                    
                    # Structure: Material -> Degauss -> K_Dist -> ...
                    if node.process_label in ['EpwPrepWorkChain']:
                        self._data[mat_key][extras['degauss']][extras['kpoints_distance_scf']][extras['qpoints_distance']] = node
                except Exception as e:
                    # Provide more context in error message
                    raise ValueError(f'Node<{node.pk}> processing failed: {e}')

    def get_table(self):
        import pandas as pd
        import numpy as np

        def get_status_string(node):
            if node is None:
                return 'N/A'

            if not node.is_terminated:
                return '⏳'
            if node.is_finished_ok:
                return '✅'
            elif node.is_failed:
                return f'❌ ({node.exit_status})'
            elif node.is_excepted:
                return '⚠️ Excepted'
            elif node.is_killed:
                return '💀 Killed'
            else:
                return f'🏃 {node.process_state.value}'

        flattened_list = []

        # Loop variables matching new dictionary structure:
        # Material -> Degauss -> K_Dist -> {'relax': ..., 'q_dist': ...}
        for material, degauss_dict in self._data.items():
            for degauss, k_dist_dict in degauss_dict.items():
                for k_dist, q_dist_data in k_dist_dict.items():
                    for q_dist, epw_node in q_dist_data.items():
                        if epw_node:
                            flattened_list.append({
                                'Material': material,
                                'Degauss': degauss,
                                'K_Density': k_dist,
                                'Q_Density': q_dist,
                                'Type': 'EpwPrepWorkChain',
                                'Status': get_status_string(epw_node) + f" ({epw_node.pk})",
                            })
                            

        if not flattened_list:
            return pd.DataFrame()

        df = pd.DataFrame(flattened_list)
        
        pivot_df = df.pivot(
            index=['Degauss', 'K_Density', 'Q_Density', 'Type'],
            columns='Material',
            values='Status'
        )

        pivot_df = pivot_df.fillna('')

        # Sort columns (Materials) alphabetically
        pivot_df = pivot_df.sort_index(axis=1)

        return pivot_df

    def get_epw_bands_nodes(self):
        """
        Get epw bands node.
        """
        # Structure: Material -> Degauss -> K_Dist -> Q_Dist -> AllenDynesTc
        nodes = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: None
                    )
                )
            )
        )
        for material, degauss_dict in self._data.items():
            for degauss, k_dist_dict in degauss_dict.items():
                for k_dist, q_dist_dict in k_dist_dict.items():
                    for q_dist, node in q_dist_dict.items():
                        if node:
                            analyser = EpwPrepWorkChainAnalyser(node)
                            try:
                                nodes[material][degauss][k_dist][q_dist] = analyser.epw_bands
                            except Exception as e:
                                print(f'Node<{node.pk}> processing failed: {e}')
        
        # Convert nested defaultdict to regular dict for cleaner output
        def default_to_regular(d):
            if isinstance(d, defaultdict):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d

        return default_to_regular(nodes)

    def plot_elbands(self):
        import matplotlib.pyplot as plt

        materials = sorted(self._data.keys())
        num_materials = len(materials)
        if num_materials == 0:
            print("No data to plot.")
            return

        fig, axs = plt.subplots(2, num_materials, figsize=(24, 4*num_materials), squeeze=False)
        # axs = axs.flatten()

        for i, material in enumerate(materials):
            degauss_dict = self._data[material]
            
            cmap = plt.get_cmap('Blues')
            blues = cmap(numpy.linspace(0.3, 1.0, 5)).tolist()
            cmap = plt.get_cmap('Reds')
            reds = cmap(numpy.linspace(0.3, 1.0, 5)).tolist()
            
            for degauss, k_dist_dict in degauss_dict.items():
                for k_dist, content in k_dist_dict.items():
                    # Check q_dist for EpwPrep nodes
                    q_dist_dict = content['q_dist']
                    
                    for q_dist, types in q_dist_dict.items():
                        node = types.get('EpwPrepWorkChain')
                        if node and node.is_finished_ok:
                             # Use Analyser to get bands
                            try:
                                analyser = EpwPrepWorkChainAnalyser(node)

                                plot_epw_interpolated_bands(
                                    analyser.process_tree.epw_bands.node,
                                    axes = axs[:, i], 
                                    label = f'{degauss} {k_dist} {q_dist}',
                                    color_el = blues.pop() if blues else 'blue',
                                    color_ph = reds.pop() if reds else 'red',
                                    )
                            except Exception as e:
                                print(f"Failed to plot {material} {degauss} {k_dist} {q_dist}: {e}")
                                continue # Skip if failing

            axs[0, i].set_title(material)

        # Cleanup unused axes
        for j in range(i + 1, len(axs)):
            axs[:, j].axis('off')

        fig.tight_layout()


    def plot_a2f(
        self,
        axis = None,
        show_data = False,
        **kwargs,
        ):
        from matplotlib import pyplot as plt
        import numpy

        # Identify all unique parameters to set up grid
        materials = sorted(self._data.keys())
        all_k_densities = set()
        for mat in materials:
            for d in self._data[mat].values():
                 all_k_densities.update(d.keys())
                 
        k_densities = sorted(list(all_k_densities), reverse=True) # Sort descending

        num_materials = len(materials)
        num_k_dens = len(k_densities)
        
        if num_materials == 0 or num_k_dens == 0:
            print("No data to plot.")
            return

        fig, axs = plt.subplots(
            num_k_dens, num_materials, 
            figsize=(5*num_materials, 4*num_k_dens),
            squeeze=False 
        )

        cmap_name = kwargs.get('cmap', 'viridis')
        cmap = plt.get_cmap(cmap_name)
        
        for j, material in enumerate(materials):
            degauss_dict = self._data[material]
            
            for i, k_dist in enumerate(k_densities):
                ax = axs[i, j]
                
                # Check if this K_Dist exists for this material (in any degauss group)
                # Structure: Mat -> Degauss -> K_Dist
                
                found_data = False
                sorted_degausses = sorted(degauss_dict.keys())
                colors = cmap(numpy.linspace(0, 1, len(sorted_degausses)))
                
                for d_idx, degauss in enumerate(sorted_degausses):
                    k_dist_dict = degauss_dict[degauss]
                    
                    if k_dist in k_dist_dict:
                        content = k_dist_dict[k_dist]
                        q_dist_dict = content['q_dist']
                        
                        for q_dist, types in q_dist_dict.items():
                            node = types.get('epwprep')
                            
                            if node is None or not node.is_finished_ok:
                                continue
                            
                            try:
                                analyser = EpwPrepWorkChainAnalyser(node)
                                epw_node = analyser.epw_bands
                                if 'a2f_data' in epw_node.outputs:
                                    a2f_data = epw_node.outputs.a2f_data
                                elif 'a2f' in epw_node.outputs:
                                    a2f_data = epw_node.outputs.a2f
                                else:
                                    continue
                                    
                                w = a2f_data.get_array('frequency')
                                spectral = a2f_data.get_array('a2f') 

                                label = f"D={degauss}"
                                if len(q_dist_dict) > 1:
                                    label += f", Q={q_dist}"

                                if kwargs.get('do_a2f', True):
                                    y_val = spectral
                                    if len(spectral.shape) > 1:
                                        if spectral.shape[1] > 9:
                                            y_val = spectral[:, 9]
                                        else:
                                            y_val = spectral[:, 0]
                                    
                                    ax.plot(
                                        y_val,
                                        w,
                                        color=colors[d_idx],
                                        label=label
                                    )
                                found_data = True
                                
                            except Exception as e:
                                print(f"Error extracting/plotting for node {node.pk}: {e}")
                                continue

                if not found_data:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                else:
                    # Formatting
                    ax.set_title(f"{material}\nK={k_dist}")
                    if i == num_k_dens - 1:
                        ax.set_xlabel(r"$\alpha^2F$")
                    if j == 0:
                        ax.set_ylabel(r"$\omega$")
                    ax.legend(fontsize='x-small')

        plt.tight_layout()
        return fig, axs
