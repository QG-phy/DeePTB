# DeePTB EPC v1 Workflow

This document describes the current electron-phonon coupling workflow. DeePTB
EPC v1 reads external phonon mode data; it does not compute phonons, force
constants, forces, stress, or repulsive potentials.

## Data Contracts

The primary DeePTB EPC format is NPZ. HDF5/dftbephy files are reference or
benchmark inputs only, not the public DeePTB output contract.

Python users can import the v1 EPC APIs from either
`dptb.postprocess.unified.eph` or `dptb.postprocess.unified`. Public v1 symbols
include `Phonons`, `EPCData`, `EPCPathData`, `EPCMeshSpec`, `EPCMeshData`,
`LinewidthData`, `LinewidthPathData`, `LinewidthMeshData`,
`RelaxationTimeData`, `RelaxationTimePathData`, `RelaxationTimeMeshData`,
`TransportData`, `MobilityData`, `MobilityScanData`, `SubspaceCouplingData`, `compute_coupling_matrix`,
`compute_linewidth`, `compute_linewidth_path`, `compute_linewidth_mesh`,
`compute_relaxation_time`, `compute_relaxation_time_path`,
`compute_relaxation_time_mesh`, `compute_serta_conductivity`,
`compute_band_velocities_finite_difference`,
`compute_band_velocities_hamiltonian_derivative`, `compute_serta_mobility_si`,
`compute_serta_mobility_scan_si`, `compute_serta_transport_from_epc`,
`find_degenerate_band_groups`, `compute_subspace_coupling_strength`,
`compute_subspace_coupling_data`, `FDProvider`, `SupercellFD`, and the
benchmark-only `DFTBPlusGauge` adapter. EPC unit constants are centralized in
`dptb.utils.constants` and re-exported from the EPC namespace.

### Phonons NPZ

Use `Phonons.save_npz()` / `Phonons.load_npz()`.

Auxiliary Python readers are available for external phonopy data:
`Phonons.from_phonopy(phonopy_obj, qpoints=...)` and
`Phonons.from_phonopy_file("phonopy_disp.yaml", qpoints=..., force_sets_filename=...)`.
These readers delegate parsing to phonopy and are intended to prepare DeePTB
phonon-mode NPZ files; DeePTB still does not compute phonons.

Required arrays:

- `ph_qpoints`: shape `(nq, 3)`, fractional q-points.
- `ph_frequencies`: shape `(nq, nmodes)`, THz.
- `ph_eigenvectors`: shape `(nq, nmodes, natoms, 3)`, Cartesian.
- `ph_masses`: shape `(natoms,)`, amu.
- `metadata_json`: JSON metadata.

Optional arrays:

- `ph_cell`: shape `(3, 3)`.
- `ph_scaled_positions`: shape `(natoms, 3)`.

Q-points and phonon modes must be non-empty. Q-points, frequencies,
eigenvectors, cell, and scaled positions must be finite when present. Masses
must be finite and positive. Imaginary modes are not supported in v1; negative
frequencies are rejected in coupling and linewidth postprocess steps.

### EPCData NPZ

Use `EPCData.save_npz()` / `EPCData.load_npz()`.

Required arrays:

- `el_kpoints`: shape `(nk, 3)`.
- `ph_qpoints`: shape `(nq, 3)`.
- `el_band_indices`: shape `(nbands_selected,)`.
- `ph_frequencies`: shape `(nq, nmodes)`, THz.
- `el_eigenvalues_k`: shape `(nk, nbands_selected)`, eV.
- `el_eigenvalues_kq`: shape `(nq, nk, nbands_selected)`, eV.
- `elph_coupling_matrix`: shape `(nq, nk, nmodes, nbands_selected, nbands_selected)`.
- `elph_coupling_strength`: same shape as `elph_coupling_matrix`.
- `metadata_json`: JSON metadata.

Derivative matrices, full H/S matrices, and full eigenvectors are not part of
the default EPCData NPZ contract.

K-points, q-points, selected bands, and phonon modes must be non-empty.
K-points, q-points, eigenvalues, and coupling matrices must be finite.
Frequencies and coupling strengths must be finite and non-negative. Band
indices must be a one-dimensional non-empty array of non-negative integers.

### EPCPathData NPZ

Use `EPCPathData.save_npz()` / `EPCPathData.load_npz()`.

`EPCPathData` stores the same EPC arrays as `EPCData`, plus path metadata:

- `path_axis`: currently `"q"` for fixed electronic k-points plus q-path.
- `path_coordinates`: cumulative distance in fractional reciprocal
  coordinates.
- optional `path_segments`: shape `(nsegments, 2)`, `[start, stop)` ranges.
- path labels, when used from Python, are stored in `metadata_json`.

The current path workflow supports fixed electronic k-points plus an external
q-path. `path_axis="k"` and full k-path + fixed-q workflows are future work.

### EPCMeshData NPZ

Use `EPCMeshData.save_npz()` / `EPCMeshData.load_npz()`.

`EPCMeshData` stores the same EPC arrays as `EPCData`, plus mesh weights:

- `el_kpoint_weights`: normalized k-point weights, shape `(nk,)`.
- `ph_qpoint_weights`: normalized q-point weights, shape `(nq,)`.

`EPCMeshSpec` can generate electronic k-points from DeePTB's existing k-point
mesh helpers, or accept explicit k-points. The phonon side remains external:
`q_mesh` is only used to validate that the supplied `Phonons.qpoints` match the
expected mesh. DeePTB does not generate phonons for mesh workflows.

### Postprocess NPZ

Use the matching save/load APIs:

- `LinewidthData.save_npz()` / `LinewidthData.load_npz()`.
- `LinewidthPathData.save_npz()` / `LinewidthPathData.load_npz()`.
- `LinewidthMeshData.save_npz()` / `LinewidthMeshData.load_npz()`.
- `RelaxationTimeData.save_npz()` / `RelaxationTimeData.load_npz()`.
- `RelaxationTimePathData.save_npz()` / `RelaxationTimePathData.load_npz()`.
- `RelaxationTimeMeshData.save_npz()` / `RelaxationTimeMeshData.load_npz()`.
- `TransportData.save_npz()` / `TransportData.load_npz()`.
- `MobilityData.save_npz()` / `MobilityData.load_npz()`.
- `MobilityScanData.save_npz()` / `MobilityScanData.load_npz()`.
- `SubspaceCouplingData.save_npz()` / `SubspaceCouplingData.load_npz()`.

`RelaxationTimeData` stores `elph_relaxation_time` in seconds. The v1
convention is `tau = hbar / (2 * linewidth)` with linewidth in eV. Mode-resolved
linewidths preserve their mode axis by default; the CLI can sum the final mode
axis first with `--sum-modes`.

`LinewidthData` stores linewidth, absorption, and emission arrays with shape
`(nk, nbands)` or mode-resolved shape `(nk, nbands, nmodes)`.
`RelaxationTimeData` uses the same 2D or 3D shape convention.

Path postprocess data preserves the path axis:

- `LinewidthPathData`: `(npath, nk, nbands)` or
  `(npath, nk, nbands, nmodes)`.
- `RelaxationTimePathData`: same path shape convention.

Path linewidth is stored as per-path-point contribution. Summing the q-path
axis reproduces the total linewidth for the same EPC path data.

Mesh postprocess data preserves k-point metadata:

- `LinewidthMeshData`: `(nk, nbands)` or `(nk, nbands, nmodes)`, plus
  `el_kpoints`, `el_kpoint_weights`, and `el_band_indices`.
- `RelaxationTimeMeshData`: same mesh shape convention.

Mesh linewidth uses normalized q-point weights in the q summation. Uniform
q-point weights reproduce the current total linewidth convention.

`SubspaceCouplingData` persists contiguous band groups as `[start, stop)`
ranges. Non-contiguous groups are available only through the Python helper
`compute_subspace_coupling_strength()`.

Postprocess data arrays must also be non-empty. In particular, linewidth,
relaxation-time, carrier-density, and subspace group-bound arrays cannot be
empty. Carrier density must be finite and non-negative.

`MobilityData` stores SI SERTA conductivity, carrier density, and mobility.
The Python helper `compute_serta_mobility_si(...)` converts band velocities
from `eV/fractional_reciprocal_coordinate` to `m/s` using an explicit
reciprocal cell in Angstrom^-1. It supports 3D volume normalization
(`conductivity_unit="S/m"`, `carrier_density_unit="m^-3"`) and 2D sheet
normalization (`conductivity_unit="S"`, `carrier_density_unit="m^-2"`).
`temperature` remains kBT in eV, matching the existing EPC postprocess
convention.

`MobilityScanData` stores the same SI quantities over chemical-potential and
temperature axes. The Python helper `compute_serta_mobility_scan_si(...)`
returns arrays with shape `(nmu, ntemperatures, 3, 3)` for conductivity and
mobility, and `(nmu, ntemperatures)` for carrier density.

## CLI Tasks

### Coupling

```bash
dptb eph \
  --task coupling \
  -i model.pth \
  -stu structure.vasp \
  -ph phonons.npz \
  -k kpoints.json \
  -b 0 1 \
  -o epc_data.npz
```

`--task coupling` reads external phonon mode data and electronic k-points, then
writes an `EPCData` NPZ. The CLI does not calculate phonons.

Supported k-point file formats:

- JSON array or object with `kpoints`.
- NPY.
- NPZ with `kpoints`.
- Plain text readable by `numpy.loadtxt`.

Loaded k-points are validated immediately by the CLI: the array must have shape
`(nk, 3)`, be non-empty, and contain only finite values.

### Path Coupling

```bash
dptb eph \
  --task path-coupling \
  -i model.pth \
  -stu structure.vasp \
  -ph phonons_qpath.npz \
  -k fixed_kpoints.json \
  -b 0 1 \
  -o epc_path_data.npz
```

`--task path-coupling` writes an `EPCPathData` NPZ for fixed electronic
k-points plus an external q-path. Path coordinates are generated from the
external q-points in fractional reciprocal coordinates.

### Mesh Coupling

```bash
dptb eph \
  --task mesh-coupling \
  -i model.pth \
  -stu structure.vasp \
  -ph phonons_mesh.npz \
  --k-mesh 4 4 1 \
  --q-mesh 4 4 1 \
  --time-reversal \
  --chunk-size 16 \
  -b 0 1 \
  -o epc_mesh_data.npz
```

`--task mesh-coupling` writes an `EPCMeshData` NPZ. `--k-mesh` generates
electronic k-points with DeePTB's existing k-point mesh helpers. `--q-mesh`
only validates the external phonon q-points in `phonons_mesh.npz`; it does not
calculate phonons. `--chunk-size` enables serial k-point chunk execution and
then concatenates chunks deterministically into one `EPCMeshData` output.

### Linewidth

```bash
dptb eph \
  --task linewidth \
  --epc-data epc_data.npz \
  --chemical-potential 0.15 \
  --temperature 0.025 \
  --sigma 0.01 \
  --broadening gaussian \
  --frequency-floor 1e-5 \
  -o linewidth.npz
```

Temperature is expressed as `kBT` in eV. `--frequency-floor` is in THz and is
used to regularize acoustic zero modes in the Bose occupation.

### Path Linewidth

```bash
dptb eph \
  --task path-linewidth \
  --epc-data epc_path_data.npz \
  --chemical-potential 0.15 \
  --temperature 0.025 \
  --sigma 0.01 \
  --broadening gaussian \
  -o path_linewidth.npz
```

This task reads `EPCPathData` and writes `LinewidthPathData`. It preserves the
q-path axis instead of reducing it.

### Mesh Linewidth

```bash
dptb eph \
  --task mesh-linewidth \
  --epc-data epc_mesh_data.npz \
  --chemical-potential 0.15 \
  --temperature 0.025 \
  --sigma 0.01 \
  --broadening gaussian \
  -o mesh_linewidth.npz
```

This task reads `EPCMeshData` and writes `LinewidthMeshData`, preserving
electronic k-points and k-point weights.

### Relaxation Time

```bash
dptb eph \
  --task relaxation-time \
  --linewidth-data linewidth.npz \
  --sum-modes \
  -o relaxation_time.npz
```

`--task relaxation-time` reads a `LinewidthData` NPZ and writes a
`RelaxationTimeData` NPZ. Use `--sum-modes` when the linewidth input is
mode-resolved and transport should use a total linewidth per k-point and band.

### Path Relaxation Time

```bash
dptb eph \
  --task path-relaxation-time \
  --linewidth-data path_linewidth.npz \
  --sum-modes \
  -o path_relaxation_time.npz
```

This task reads `LinewidthPathData` and writes `RelaxationTimePathData`.

### Mesh Relaxation Time

```bash
dptb eph \
  --task mesh-relaxation-time \
  --linewidth-data mesh_linewidth.npz \
  --sum-modes \
  -o mesh_relaxation_time.npz
```

This task reads `LinewidthMeshData` and writes `RelaxationTimeMeshData`.

### Transport

```bash
dptb eph \
  --task transport \
  -i model.pth \
  -stu structure.vasp \
  --epc-data epc_data.npz \
  --linewidth-data linewidth.npz \
  --chemical-potential 0.15 \
  --temperature 0.025 \
  --kpoint-weights weights.npz \
  --spin-degeneracy 2 \
  --volume 1.0 \
  --velocity-source finite_difference \
  --velocity-delta 1e-4 \
  -o transport.npz
```

Transport v1 supports two band-velocity providers:

- `--velocity-source finite_difference`: central finite differences over
  `system.get_eigenvalues(k_points=...)`. This remains the default.
- `--velocity-source hamiltonian_derivative`: analytic band velocities from
  `system.get_hk(k_points=..., with_derivative=True)`, using
  `<n|dH/dk|n> - E_n <n|dS/dk|n>` when overlap derivatives are present.

Transport velocity metadata is stored as
`eV/fractional_reciprocal_coordinate`; the `TransportData` task remains an
intermediate SERTA conductivity workflow in DeePTB's fractional-k convention.
Use `--task mobility` for SI conductivity, carrier density, and mobility.

Supported k-point weights file formats:

- JSON array or object with `kpoint_weights`.
- NPY.
- NPZ with `kpoint_weights`.
- Plain text readable by `numpy.loadtxt`.

Loaded k-point weights are validated immediately by the CLI: the array must be
one-dimensional, non-empty, finite, non-negative, and have a positive sum.

### Mobility

```bash
dptb eph \
  --task mobility \
  -i model.pth \
  -stu structure.vasp \
  --epc-data epc_data.npz \
  --linewidth-data linewidth.npz \
  --chemical-potential 0.15 \
  --temperature 0.025 \
  --dimension 3d \
  --volume 1.0 \
  --velocity-source hamiltonian_derivative \
  -o mobility.npz
```

`--task mobility` writes a `MobilityData` NPZ with SI conductivity, carrier
density, and mobility. It reuses the same velocity providers as transport.
The reciprocal cell is inferred from the input structure as
`2*pi*atoms.cell.reciprocal()` in Angstrom^-1, consistent with DeePTB's
fractional k-point phase convention.

For 3D systems, use `--dimension 3d --volume <Angstrom^3>`. For 2D sheet
normalization, use `--dimension 2d --area <Angstrom^2>`. Temperature remains
kBT in eV.

For chemical-potential and temperature scans, use plural scan arguments:

```bash
dptb eph \
  --task mobility \
  -i model.pth \
  -stu structure.vasp \
  --epc-data epc_data.npz \
  --linewidth-data linewidth.npz \
  --chemical-potentials 0.10 0.15 0.20 \
  --temperatures 0.025 0.050 \
  --dimension 3d \
  --volume 1.0 \
  -o mobility_scan.npz
```

When `--chemical-potentials` or `--temperatures` is used, the CLI writes a
`MobilityScanData` NPZ. The singular and plural forms for the same axis are
mutually exclusive.

### Subspace Coupling

```bash
dptb eph \
  --task subspace \
  --epc-data epc_data.npz \
  --final-groups 0:2 2:3 \
  --initial-groups 0:2 2:3 \
  -o subspace_coupling.npz
```

This task writes gauge-invariant coupling strength aggregated over contiguous
band subspaces. Groups use `start:stop` ranges and are stored in the NPZ as
`[start, stop)` bounds. CLI-provided group ranges must be non-overlapping.

## Current v1 Limits

- SCC-corrected EPC is not supported. `use_scc=True` raises
  `NotImplementedError`.
- Path workflows currently support fixed electronic k-points plus q-path only.
  k-path plus fixed-q is not implemented.
- Mesh workflows currently use serial in-memory execution. K-point chunking is
  available for `mesh-coupling`, but chunked artifacts, multiprocessing, MPI,
  and CUDA backends are future work.
- SOC/spinful EPC is not implemented.
- Polar correction is not implemented.
- Full degenerate-band gauge fixing and k/q-path continuous gauge tracking are
  not implemented.
- Transport supports finite-difference and Hamiltonian-derivative velocity
  providers and keeps velocities in `eV/fractional_reciprocal_coordinate`.
  SI conversion is handled by the mobility workflow.
- SI mobility is available through the Python helper
  `compute_serta_mobility_si(...)` and `dptb eph --task mobility`; multi-mu /
  multi-temperature scans are available through the Python helper
  `compute_serta_mobility_scan_si(...)` and the CLI scan arguments
  `--chemical-potentials` / `--temperatures`.

## Development Validation

Default EPC tests:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./.venv/bin/python -m pytest dptb/tests/test_electron_phonon.py -q
```

Default tests include a small in-repo synthetic EPC fixture at
`dptb/tests/fixtures/eph/minimal_epc_reference.json` for linewidth reference
regression. The full Graphene case remains opt-in.

Graphene coupling reference:

```bash
DEEPTB_RUN_REFERENCE_EPH=1 \
MPLCONFIGDIR=/private/tmp/matplotlib-cache \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./.venv/bin/python -m pytest \
  dptb/tests/test_electron_phonon.py::test_graphene_reference_case_coupling_strength -q
```

The external reference root can be overridden with:

- `DEEPTB_EPH_REFERENCE_ROOT`: path to the dftbephy checkout.
- `DEEPTB_EPH_SKDATA_ROOT`: path to the matsci SK data used by the slow
  finite-difference reference.

Slow Graphene supercell finite-difference reference:

```bash
DEEPTB_RUN_REFERENCE_EPH=1 DEEPTB_RUN_SLOW_EPH=1 \
MPLCONFIGDIR=/private/tmp/matplotlib-cache \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./.venv/bin/python -m pytest \
  dptb/tests/test_electron_phonon.py::test_graphene_reference_case_supercell_fd_provider -q
```
