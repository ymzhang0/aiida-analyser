from aiida_analyser.plot import plot_bands, plot_epw_interpolated_bands
import numpy
from collections import defaultdict
import warnings
from aiida import orm
from ..base import BaseWorkChainAnalyser
from pathlib import Path

class ScHubbardWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the ScHubbardWorkChain.
    """

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
    def hubbard(self):
        if 'hubbard' not in self.process_tree:
            raise AttributeError('hubbard is not found')
        else:
            return self.process_tree.hubbard.node

    def get_source(self):
        """Get the source of the workchain."""
        return super().get_source()

    def get_state(self):
        """Get the state of the workchain."""
        return self._get_state_from_subprocesses([
            ('hubbard', SelfConsistentHubbardWorkChainAnalyser),
        ])

    def clean_workchain(self, exempted_states=None, dry_run=True):
        """Clean the workchain."""
        exempted_states = [] if exempted_states is None else exempted_states
        path, process_state, exit_code = self.get_state()
        message = f'Process<{self.node.pk}> is now {process_state} at {path} with exit code {exit_code}. Please check if you really want to clean this workchain.\n'
        if process_state in exempted_states:
            print(message)
            return message, False

        message, success = super().clean_workchain(dry_run=dry_run)
        return message, True

class ScHubbardGroup:
    """
    Analyser for the ScHubbardWorkChain.
    """
    def __init__(self, groups=None):
        self._groups = groups
        self._data = defaultdict(
            lambda: None
        )
        self.get_data()

    @property
    def groups(self):
        return self._groups

    @property
    def data(self):
        return self._data

    def get_data(self):
        for grpname in self._groups:
            group = orm.load_group(grpname)
            for node in group.nodes:
                try:
                    extras = node.base.extras.all
                    self.check_protocol(node)
                    
                    mat_key = f"{extras['source_db']}-{extras['source_id']}-{extras['formula']}"
                    
                    # Structure: Material -> ...
                    if node.process_label in ['SelfConsistentHubbardWorkChain']:
                        self._data[mat_key]= node
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
        for material, node in self._data.items():
            if node:
                flattened_list.append({
                    'Material': material,
                    'Type': 'SelfConsistentHubbardWorkChain',
                    'Status': get_status_string(node) + f" ({node.pk})",
                })
                            

        if not flattened_list:
            return pd.DataFrame()

        df = pd.DataFrame(flattened_list)
        
        pivot_df = df.pivot(
            index=['Type'],
            columns='Material',
            values='Status'
        )

        pivot_df = pivot_df.fillna('')

        # Sort columns (Materials) alphabetically
        pivot_df = pivot_df.sort_index(axis=1)

        return pivot_df

    def dump(self, dest:Path):
        for material, node in self._data.items():
            if node:
                analyser = ScHubbardWorkChainAnalyser(node)
                analyser.copy_tree(
                    dest / material.split("-")[-1] / f"{node.pk}"
                )
