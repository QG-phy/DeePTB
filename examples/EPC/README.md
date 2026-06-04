# DeePTB EPC Notebooks

This folder contains DeePTB-native EPC notebooks using local reference assets when available:

- local clean Graphene structure: `examples/EPC/graphene.vasp`
- dftbephy Graphene example: `/Users/aisiqg/Desktop/work/github/dftbephy/examples/Graphene`
- matsci-0-3 SKF files: `/Users/aisiqg/Desktop/work/github/matsci-0-3`

The notebooks use Python APIs directly. They instantiate `DFTBSK` from SKF files, build a `TBSystem`, convert external phonopy data to DeePTB `Phonons` NPZ, then compute EPC, linewidth, transport, mobility, and an Eliashberg-like diagnostic. No `.pth` checkpoint is required.

## Notebooks

- `00_prepare_external_phonons.ipynb`: create `Phonons` NPZ files from the local Graphene structure plus dftbephy/phonopy Graphene data and instantiate the SKF model.
- `01_path_epc_linewidth.ipynb`: fixed-k plus q-path EPC, path linewidth, relaxation time, and coupling summary.
- `02_mesh_transport_mobility.ipynb`: mesh EPC, chunked artifact, linewidth, SERTA transport, SI mobility, and mobility scan.
- `03_eliashberg_like_diagnostic.ipynb`: coupling-strength-weighted phonon spectrum and diagnostic Tc template.

## Scope

These examples are non-SCC EPC v1 examples. DeePTB reads external phonon modes; it does not compute phonons, force constants, forces, stress, or repulsive potentials. SCC EPC, SOC/spinful EPC, polar correction, MPI/CUDA execution, and full superconductivity prediction are outside these notebooks.
