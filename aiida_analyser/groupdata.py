from collections import defaultdict


class BaseGroupData:
    """Base class for group data objects."""

    def __init__(self, groups=None):
        self._groups = groups if groups is not None else []
        self._data = defaultdict(lambda: None)  # Subclasses should redefine this

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
        if node.is_failed:
            return f'❌ ({node.exit_status})'
        if node.is_excepted:
            return '⚠️ Excepted'
        if node.is_killed:
            return '💀 Killed'
        return f'🏃 {node.process_state.value}'

    def get_table(self):
        """Generic table generation with pivoting."""
        import pandas as pd

        flattened_list = self._flatten_data()
        if not flattened_list:
            return pd.DataFrame()

        df = pd.DataFrame(flattened_list)

        # 1. Define the columns we know are NOT part of the index
        # 'Plane' is our pivot column, 'Status' is our value
        pivot_col = 'Plane'
        value_col = 'Status'

        if pivot_col in df.columns and value_col in df.columns:
            index_cols = [col for col in df.columns if col not in [pivot_col, value_col]]
            pivot_df = df.pivot_table(
                values=value_col,
                index=index_cols,
                columns=pivot_col,
                aggfunc='first',
            )
            pivot_df = pivot_df.fillna('')
            pivot_df = pivot_df.sort_index(axis=1)
            return pivot_df

        return df

    def _flatten_data(self):
        """To be implemented by subclasses."""
        raise NotImplementedError
