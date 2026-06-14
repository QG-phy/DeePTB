# DeePTB EPC Notebooks

This folder contains DeePTB-native EPC notebooks using local reference assets when available.
Set these environment variables before running the notebooks:

- `DEEPTB_EPC_DFTBEPHY_GRAPHENE`: dftbephy Graphene example directory, for example `<path-to-dftbephy>/examples/Graphene`.
- `DEEPTB_EPC_MATSCI_SK`: matsci-0-3 SKF directory containing `C-C.skf`.
- `DEEPTB_EPH_REFERENCE_ROOT`: dftbephy checkout root, used only by `04_graphene_reference_alignment.ipynb`.

The clean Graphene structure used by default is `examples/EPC/graphene.vasp`.

The notebooks use Python APIs directly. They instantiate `DFTBSK` from SKF files, build a `TBSystem`, convert external phonopy data to DeePTB `Phonons` NPZ, then compute EPC, linewidth, transport, mobility, and an Eliashberg-like diagnostic. No `.pth` checkpoint is required.

## Notebooks

- `00_prepare_external_phonons.ipynb`: create `Phonons` NPZ files from the local Graphene structure plus dftbephy/phonopy Graphene data and instantiate the SKF model.
- `01_path_epc_linewidth.ipynb`: fixed-k plus q-path EPC, path linewidth, relaxation time, and coupling summary.
- `02_mesh_transport_mobility.ipynb`: mesh EPC, chunked artifact, linewidth, SERTA transport, SI mobility, and mobility scan.
- `03_eliashberg_like_diagnostic.ipynb`: scattering maps, phonon DOS, coupling-strength-weighted phonon spectrum, and diagnostic Tc template.
- `04_graphene_reference_alignment.ipynb`: opt-in dftbephy/Graphene numerical reference comparison for `eigenvalues_k`, `eigenvalues_kq`, and `g2` / `coupling_strength`.

## Scope

These examples are non-SCC EPC v1 examples. DeePTB reads external phonon modes; it does not compute phonons, force constants, forces, stress, or repulsive potentials. SCC EPC, SOC/spinful EPC, polar correction, MPI/CUDA execution, and full superconductivity prediction are outside these notebooks.

`04_graphene_reference_alignment.ipynb` reads dftbephy reference data only as a benchmark. The notebooks do not write dftbephy HDF5, do not consume HSD input, do not call dftbephy CLI workflows, and are not a dftbephy drop-in replacement.
