# DeePTB EPC Notebooks

This folder contains DeePTB-native Graphene EPC notebooks with the required
example assets bundled under `examples/EPC/data/`. The notebooks run from this
directory without environment variables or machine-local paths.

The goal is to reproduce the main Graphene chain from the DFTBephy paper using
DeePTB APIs and data contracts, not to clone the dftbephy application shape.
The notebooks instantiate `DFTBSK` from bundled matsci-0-3 `C-C.skf`, build a
`TBSystem`, read bundled phonopy data through `Phonons`, and then compute EPC,
linewidths, SERTA transport, mobility, and reference checks.

## Notebooks

- `00_prepare_external_phonons.ipynb`: paper Fig. 1 style electronic bands and phonon dispersion, plus `Phonons` NPZ preparation.
- `01_path_epc_linewidth.ipynb`: paper Fig. 2 style TA/LA EPC contour, `|g| / ell_q`, for a conduction-band state near `K` and intravalley `q` around Gamma.
- `02_mesh_transport_mobility.ipynb`: paper Fig. 3 style inverse lifetime / scattering-rate curves versus conduction-band energy at `T=300 K`, `mu=100 meV`, and `sigma=3 meV`.
- `03_transport_mobility.ipynb`: paper Fig. 4 style carrier density, conductivity, mobility versus density, and mobility versus temperature.
- `04_graphene_reference_alignment.ipynb`: bundled Graphene numerical reference check for `eigenvalues_k`, `eigenvalues_kq`, and `g2` / `coupling_strength`. This is a benchmark/parity notebook, not a paper figure reproduction.

## Figures

The notebooks generate the following saved figures under `examples/EPC/work/`
when run:

- `figure_00_bands_phonons.png`: electronic bands and phonon dispersion along `G-M-K-G`.
- `figure_01_fig2_epc_contours.png`: Fig. 2 style TA/LA EPC contours around Gamma.
- `figure_02_fig3_scattering_rates.png`: Fig. 3 style inverse lifetime curves versus electronic energy.
- `figure_03_fig4_transport_mobility.png`: Fig. 4 style transport and mobility panels.
- `figure_04_reference_alignment.png`: DeePTB/reference `g2` parity and error histogram.

## Runtime And Convergence

The default meshes are reduced so the notebooks are runnable examples. The paper
used much denser sampling, notably a `200 x 200` intravalley q mesh for
scattering rates and a `400 x 400` k mesh for transport. Increase the parameter
cells in notebooks 01-03 for converged production comparisons.

The generated numerical values from the reduced defaults should be treated as
workflow and convention checks. They are not publication-quality absolute
benchmarks until the meshes are converged.

## Scope

These examples cover non-SCC EPC v1 with external phonopy modes,
finite-difference Hamiltonian/overlap derivatives, path/mesh EPC, linewidths,
relaxation/scattering diagnostics, and SERTA transport/mobility. DeePTB does not
compute phonons, force constants, forces, stress, or repulsive potentials in
these notebooks.

SCC EPC, SOC/spinful EPC, polar correction, MPI/CUDA execution, and production
superconducting Tc workflows are outside this notebook set. The notebooks do not
write dftbephy HDF5, do not consume HSD input, do not call dftbephy CLI
workflows, and are not a dftbephy drop-in replacement.
