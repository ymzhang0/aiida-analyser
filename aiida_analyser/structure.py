from ase.build import bulk
from aiida.orm import StructureData

lattice_constants = {
    # FCC
    'Al': {'fcc': 4.0144}, # https://mc3d.materialscloud.org/#/details/mc3d-29639/pbesol-v1
    'Pb': {'fcc': 4.9191}, # https://mc3d.materialscloud.org/details/mc3d-48868/pbesol-v2
    'Cu': {'fcc': 3.5623}, # https://mc3d.materialscloud.org/#/details/mc3d-24185/pbesol-v1
    'Au': {'fcc': 4.0712}, # https://mc3d.materialscloud.org/#/details/mc3d-66164/pbesol-v1
    'Ag': {'fcc': 4.0496}, # https://mc3d.materialscloud.org/#/details/mc3d-75785/pbesol-v1
    'Ni': {'fcc': 3.4614}, # https://mc3d.materialscloud.org/#/details/mc3d-4988/pbesol-v1
    'Pt': {'fcc': 3.9136}, # https://mc3d.materialscloud.org/#/details/mc3d-31294/pbesol-v1
    'Pd': {'fcc': 3.8731}, # https://mc3d.materialscloud.org/#/details/mc3d-36406/pbesol-v1
    'Rh': {'fcc': 3.7782}, # https://mc3d.materialscloud.org/#/details/mc3d-4421/pbesol-v1
    # BCC
    'Li': {'bcc': 3.4348}, # https://mc3d.materialscloud.org/#/details/mc3d-11543/pbesol-v2
    'Na': {'bcc': 4.1723}, # https://mc3d.materialscloud.org/#/details/mc3d-57514/pbesol-v2
    'V' : {'bcc': 2.9520}, # https://mc3d.materialscloud.org/#/details/mc3d-45322/pbesol-v2
    'Nb': {'bcc': 3.2697}, # https://mc3d.materialscloud.org/#/details/mc3d-10833/pbesol-v2
}

def create_structure(
    element: str,
    crystalstructure: str = 'fcc',
) -> StructureData:
    structure = bulk(element, crystalstructure, a=lattice_constants[element][crystalstructure])
    return StructureData(ase=structure)
