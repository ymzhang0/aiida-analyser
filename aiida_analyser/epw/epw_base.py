from aiida import orm
from ..base import BaseWorkChainAnalyser
from .epw_calculation import EpwCalculationAnalyser
from ..groupdata import BaseGroupData

class EpwBaseWorkChainAnalyser(BaseWorkChainAnalyser):
    """
    Analyser for the EpwBaseWorkChain.
    """

    def copy_tree(self, destpath):
        """Copy the tree by delegating each direct EpwCalculation child."""
        return self._copy_tree_for_direct_children(
            destpath,
            lambda _, child: EpwCalculationAnalyser if child.node.process_label == 'EpwCalculation' else None,
        )

    def get_calcjob_paths(self):
        """Get calcjob remote paths by delegating each direct EpwCalculation child."""
        return self._get_calcjob_paths_for_direct_children(
            lambda _, child: EpwCalculationAnalyser if child.node.process_label == 'EpwCalculation' else None,
        )

    def get_source(self):
        """Get the source of the workchain."""
        if all(key in self.node.base.extras for key in ['source_db', 'source_id']):
            return (self.node.base.extras.get('source_db'), self.node.base.extras.get('source_id'))
        elif all(key in self.node.inputs.structure.base.extras for key in ['source_db', 'source_id']):
            return (self.node.inputs.structure.base.extras.get('source_db'), self.node.inputs.structure.base.extras.get('source_id'))
        else:
            raise ValueError('Source is not set')

    def get_state(self):
        """Get the state of the workchain."""
        return self._get_state_from_tree()

    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        message, success = super().clean_workchain(dry_run=dry_run)

        return message


class EpwData(BaseGroupData):
    """
    Data processor for EPW process groups.
    """

    def __init__(self, groups=None):
        super().__init__(groups)
        self._data = {}

    def _flatten_data(self):
        from aiida import orm
        flattened_list = []

        if not self._groups:
            return flattened_list

        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={'label': {'in': self._groups}}, tag='group')
        qb.append(orm.ProcessNode, with_group='group', filters={'attributes.process_label': 'EpwBaseWorkChain'})

        for r in qb.all():
            node = r[0]
            # Material
            formula = 'N/A'
            if 'structure' in node.inputs:
                try:
                    formula = node.inputs.structure.get_formula()
                except Exception:
                    pass
            if formula == 'N/A' and 'formula' in node.base.extras:
                formula = node.base.extras.get('formula')

            # Emojified Status
            status_emoji = self.get_status_string(node)

            # Coarse Grid
            coarse_k = None
            if 'kpoints' in node.inputs:
                try:
                    coarse_k = "x".join(map(str, node.inputs.kpoints.get_kpoints_mesh()[0]))
                except Exception:
                    try:
                        coarse_k = str(len(node.inputs.kpoints.get_kpoints()))
                    except Exception:
                        pass
            coarse_q = None
            if 'qpoints' in node.inputs:
                try:
                    coarse_q = "x".join(map(str, node.inputs.qpoints.get_kpoints_mesh()[0]))
                except Exception:
                    try:
                        coarse_q = str(len(node.inputs.qpoints.get_kpoints()))
                    except Exception:
                        pass

            # Fine Grid
            fine_k = None
            if 'kfpoints' in node.inputs:
                try:
                    fine_k = "x".join(map(str, node.inputs.kfpoints.get_kpoints_mesh()[0]))
                except Exception:
                    try:
                        fine_k = str(len(node.inputs.kfpoints.get_kpoints()))
                    except Exception:
                        pass
            elif 'kfpoints_factor' in node.inputs:
                fine_k = f"f:{node.inputs.kfpoints_factor.value}"

            fine_q = None
            if 'qfpoints' in node.inputs:
                try:
                    fine_q = "x".join(map(str, node.inputs.qfpoints.get_kpoints_mesh()[0]))
                except Exception:
                    try:
                        fine_q = str(len(node.inputs.qfpoints.get_kpoints()))
                    except Exception:
                        pass
            elif 'qfpoints_distance' in node.inputs:
                fine_q = f"d:{node.inputs.qfpoints_distance.value}"

            coarse_str = f"{coarse_k or '?'}/{coarse_q or '?'}"
            fine_str = f"{fine_k or '?'}/{fine_q or '?'}"
            grid_str = f"{coarse_str} -> {fine_str}"

            # Other inputs
            restart_type = node.inputs.restart_type.value if 'restart_type' in node.inputs else '-'
            calculation_type = node.inputs.calculation_type.value if 'calculation_type' in node.inputs else '-'
            momentum_dependence = node.inputs.momentum_dependence.value if 'momentum_dependence' in node.inputs else '-'
            full_bandwidth = node.inputs.full_bandwidth.value if 'full_bandwidth' in node.inputs else '-'
            real_axis = node.inputs.real_axis.value if 'real_axis' in node.inputs else '-'
            analytical_continuation = node.inputs.analytical_continuation.value if 'analytical_continuation' in node.inputs else '-'

            # status string
            if not node.is_terminated:
                status_str = node.process_state.value
            else:
                status_str = f"{node.process_state.value} ({node.exit_status})"

            flattened_list.append({
                'PK': node.pk,
                'Material': formula,
                'Status': status_emoji,
                '粗细网格': grid_str,
                'restart_type': restart_type,
                'calculation_type': calculation_type,
                'momentum_dependence': momentum_dependence,
                'full_bandwidth': full_bandwidth,
                'real_axis': real_axis,
                'analytical_continuation': analytical_continuation,
                'status': status_str,
            })

        return flattened_list

    def get_table(self):
        import pandas as pd
        flattened_list = self._flatten_data()
        if not flattened_list:
            return pd.DataFrame()
        df = pd.DataFrame(flattened_list)
        return df.set_index('PK') if 'PK' in df.columns else df

