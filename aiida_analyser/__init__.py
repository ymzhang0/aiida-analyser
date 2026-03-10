"""
`aiida-analyser`: AiiDA plugin package with postprocessing tools for AiiDA work chains.
"""
__version__ = "0.1.0"

from .dict import (
    NestedDict,
)
from .printer import (
    Printer,
)
from .base import (
    ProcessTree,
    BaseWorkChainAnalyser,
)

from .pw_base import (
    PwBaseWorkChainAnalyser,
)
from .pw_relax import (
    PwRelaxWorkChainAnalyser,
)
from .pw_bands import (
    PwBandsWorkChainAnalyser,
)
from .projwfc_base import (
    ProjwfcBaseWorkChainAnalyser,
)
from .pw2wannier90_base import (
    Pw2Wannier90BaseWorkChainAnalyser,
)
from .wannier90 import (
    Wannier90WorkChainAnalyser,
)
from .wannier90_base import (
    Wannier90BaseWorkChainAnalyser,
)
from .ph_base import (
    PhBaseWorkChainAnalyser,
)
from .q2r_base import (
    Q2rBaseWorkChainAnalyser,
)
from .matdyn_base import (
    MatdynBaseWorkChainAnalyser,
)
from .layer_relax import (
    LayerRelaxWorkChainAnalyser,
)
from .sfebase import (
    SFEBaseWorkChainAnalyser,
)
from .isfe import (
    ISFEWorkChainAnalyser,
)
from .esfe import (
    ESFEWorkChainAnalyser,
)
from .usfe import (
    USFEWorkChainAnalyser,
)
from .gsfe import (
    GSFEWorkChainAnalyser,
    GSFEGroupData,
)

from .surface import (
    SurfaceWorkChainAnalyser,
    SurfaceEnergyData,
)
from .epw_base import (
    EpwBaseWorkChainAnalyser,
)
from .epw_prep import (
    EpwPrepWorkChainAnalyser,
    EpwPrepConvergenceData,
    EpwPrepData
)
from .supercon import (
    SuperConWorkChainAnalyser,
    SuperConData,
)

from .thermo_pw import (
    ThermoPwBaseAnalyser,
)

from .structure import (
    create_structure,
)

__all__ = [
    'NestedDict',
    'deprecated',
    'Printer',
    'ProcessTree',
    'BaseWorkChainAnalyser',
    'PwBaseWorkChainAnalyser',
    'PwRelaxWorkChainAnalyser',
    'PwBandsWorkChainAnalyser',
    'ProjwfcBaseWorkChainAnalyser',
    'Pw2Wannier90BaseWorkChainAnalyser',
    'Wannier90WorkChainAnalyser',
    'Wannier90BaseWorkChainAnalyser',
    'PhBaseWorkChainAnalyser',
    'Q2rBaseWorkChainAnalyser',
    'MatdynBaseWorkChainAnalyser',
    'LayerRelaxWorkChainAnalyser',
    'SFEBaseWorkChainAnalyser',
    'ISFEWorkChainAnalyser',
    'ESFEWorkChainAnalyser',
    'USFEWorkChainAnalyser',
    'GSFEWorkChainAnalyser',
    'GSFEGroupData',
    'SurfaceWorkChainAnalyser',
    'SurfaceEnergyData',
    'EpwBaseWorkChainAnalyser',
    'EpwPrepWorkChainAnalyser',
    'EpwPrepConvergenceData',
    'EpwPrepData',
    'SuperConWorkChainAnalyser',
    'SuperConData',
    'ThermoPwBaseAnalyser',
    'create_structure',
]