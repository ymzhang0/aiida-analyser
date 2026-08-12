# `aiida-analyser`

AiiDA plugin package with workflows for superconductivity research based on the Quantum ESPRESSO software suite and the Electron-Phonon Wannier (EPW) code.

## Project layout

- `aiida_analyser/core/`: shared analyser abstractions, constants, group helpers, logging, and work-chain utilities.
- `aiida_analyser/quantumespresso/`, `epw/`, `wannier/`, `thermo_pw/`, `dislocation/`, and `hubbard/`: workflow-specific analysers.
- `aiida_analyser/data/`, `materials/`, and `visualization/`: data loading, structure creation, and plotting helpers.

## Installation

Install the base package if you only need the common analyser utilities:

```bash
pip install -e .
```

Install the feature-specific extras when needed:

```bash
pip install -e .[dislocation]
pip install -e .[epw]
pip install -e .[thermo]
pip install -e .[structure]
```

Install everything with:

```bash
pip install -e .[all]
```
