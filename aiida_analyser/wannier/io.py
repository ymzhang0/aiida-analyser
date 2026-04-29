import numpy
from pathlib import Path
from aiida.plugins import DataFactory
BandsData = DataFactory('core.array.bands')

def read_labelinfo(path: str | Path):
    path = Path(path)
    seekpath_parameters = {"explicit_segments": None, "path": None} 
    # First, extract all raw labels and indices
    raw_labels = []
    raw_indices = []

    with path.open('r') as f:
        lines = f.readlines()

        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                raw_labels.append(parts[0])
                # Convert to 0-indexed integer
                raw_indices.append(int(parts[1]) - 1)
                
    # Second, create the paired segments and paths
    explicit_segments = []
    path = []
    
    for i in range(len(raw_labels) - 1):
        # Create segment: [start_index, end_index]
        explicit_segments.append([raw_indices[i], raw_indices[i+1]])
        
        # Create path: ['START_LABEL', 'END_LABEL']
        path.append([raw_labels[i], raw_labels[i+1]])
        
    seekpath_parameters["explicit_segments"] = explicit_segments
    seekpath_parameters["path"] = path
    return seekpath_parameters


def load_bandsdata(path: str | Path):
    path = Path(path)
    seekpath_parameters = read_labelinfo(path / 'aiida_band.labelinfo.dat')

    _, nkpts = seekpath_parameters["explicit_segments"][-1]

    bands_data = BandsData()
    kpoints = numpy.loadtxt(path / 'aiida_band.kpt', skiprows=1)
    bands = numpy.loadtxt(path / 'aiida_band.dat')[:, 1].reshape(-1, nkpts+1).T
    bands_data.set_kpoints(kpoints)
    bands_data.set_bands(bands, units='eV')
    return (bands_data, seekpath_parameters)