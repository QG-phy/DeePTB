# DeePTB EPC v1 Workflow

This document describes the current electron-phonon coupling workflow. DeePTB
EPC v1 reads external phonon mode data; it does not compute phonons, force
constants, forces, stress, or repulsive potentials.

## Data Contracts

The primary DeePTB EPC format is NPZ. HDF5/dftbephy files are reference or
benchmark inputs only, not the public DeePTB output contract.

Python users can import the v1 EPC APIs from either
`dptb.postprocess.unified.eph` or `dptb.postprocess.unified`. Public v1 symbols
include `Phonons`, `EPCData`, `LinewidthData`, `RelaxationTimeData`,
`TransportData`, `SubspaceCouplingData`, `compute_coupling_matrix`,
`compute_linewidth`, `compute_relaxation_time`, `compute_serta_conductivity`,
`compute_band_velocities_finite_difference`, `compute_serta_transport_from_epc`,
`find_degenerate_band_groups`, `compute_subspace_coupling_strength`,
`compute_subspace_coupling_data`, `FDProvider`, `SupercellFD`, and the
benchmark-only `DFTBPlusGauge` adapter.

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

### Postprocess NPZ

Use the matching save/load APIs:

- `LinewidthData.save_npz()` / `LinewidthData.load_npz()`.
- `RelaxationTimeData.save_npz()` / `RelaxationTimeData.load_npz()`.
- `TransportData.save_npz()` / `TransportData.load_npz()`.
- `SubspaceCouplingData.save_npz()` / `SubspaceCouplingData.load_npz()`.

`RelaxationTimeData` stores `elph_relaxation_time` in seconds. The v1
convention is `tau = hbar / (2 * linewidth)` with linewidth in eV. Mode-resolved
linewidths preserve their mode axis by default; the CLI can sum the final mode
axis first with `--sum-modes`.

`LinewidthData` stores linewidth, absorption, and emission arrays with shape
`(nk, nbands)` or mode-resolved shape `(nk, nbands, nmodes)`.
`RelaxationTimeData` uses the same 2D or 3D shape convention.

`SubspaceCouplingData` persists contiguous band groups as `[start, stop)`
ranges. Non-contiguous groups are available only through the Python helper
`compute_subspace_coupling_strength()`.

Postprocess data arrays must also be non-empty. In particular, linewidth,
relaxation-time, carrier-density, and subspace group-bound arrays cannot be
empty. Carrier density must be finite and non-negative.

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
  --velocity-delta 1e-4 \
  -o transport.npz
```

Transport v1 uses a finite-difference velocity bridge over
`system.get_eigenvalues(k_points=...)`. Velocity metadata is stored as
`eV/fractional_reciprocal_coordinate`. Full SI mobility/conductivity unit
conversion is not part of v1.

Supported k-point weights file formats:

- JSON array or object with `kpoint_weights`.
- NPY.
- NPZ with `kpoint_weights`.
- Plain text readable by `numpy.loadtxt`.

Loaded k-point weights are validated immediately by the CLI: the array must be
one-dimensional, non-empty, finite, non-negative, and have a positive sum.

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
- SOC/spinful EPC is not implemented.
- Polar correction is not implemented.
- Full degenerate-band gauge fixing and k/q-path continuous gauge tracking are
  not implemented.
- Transport uses finite-difference velocities and does not perform full SI unit
  conversion.

## Development Validation

Default EPC tests:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./.venv/bin/python -m pytest dptb/tests/test_electron_phonon.py -q
```

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
