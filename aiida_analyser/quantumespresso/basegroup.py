from collections import defaultdict

class BaseGroupData:
    """
    Base class for group data objects.
    """
    def __init__(self, groups=None):
        self._groups = groups if groups is not None else []
        self._data = defaultdict(lambda: None) # Subclasses should redefine this

    @property
    def groups(self):
        return self._groups

    @property
    def data(self):
        return self._data

    @staticmethod
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

    def get_table(self):
        """
        Generic table generation with pivoting. 
        """
        import pandas as pd
        flattened_list = self._flatten_data()

        if not flattened_list:
            return pd.DataFrame()

        df = pd.DataFrame(flattened_list)

        # Common pivoting for surface and GSFE data
        index_cols = ['Structure', 'Material', 'Layers', 'K_Dist']
        # Check if they exist in the dataframe before pivoting
        available_index = [col for col in index_cols if col in df.columns]
        
        if 'Plane' in df.columns and 'Status' in df.columns:
            pivot_df = df.pivot_table(
                values='Status',
                index=available_index,
                columns='Plane',
                aggfunc='first'
            )
            pivot_df = pivot_df.fillna('')
            pivot_df = pivot_df.sort_index(axis=1)
            return pivot_df
        
        return df

    def _flatten_data(self):
        """
        To be implemented by subclasses.
        """
        raise NotImplementedError
