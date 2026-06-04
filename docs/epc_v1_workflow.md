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
`TransportData`, `TransportScanData`, `MobilityData`, `MobilityScanData`,
`SubspaceCouplingData`, `compute_coupling_matrix`,
`compute_linewidth`, `compute_linewidth_path`, `compute_linewidth_mesh`,
`compute_relaxation_time`, `compute_relaxation_time_path`,
`compute_relaxation_time_mesh`, `compute_serta_conductivity`,
`compute_band_velocities_finite_difference`,
`compute_band_velocities_hamiltonian_derivative`,
`fractional_band_velocities_to_si`, `compute_serta_mobility_si`,
`compute_serta_mobility_scan_si`, `compute_serta_transport_scan`,
`compute_serta_transport_from_epc`,
`compute_coupling_strength_summary`, `compute_phonon_dos`,
`compute_eliashberg_spectral_function`, `compute_scattering_maps`,
`find_degenerate_band_groups`, `compute_subspace_coupling_strength`,
`compute_subspace_coupling_data`, `FDProvider`, `SupercellFD`, and the
benchmark-only `DFTBPlusGauge` adapter. Mesh chunked artifact helpers
`save_epc_mesh_chunked_artifact(...)` and
`load_epc_mesh_chunked_artifact(...)` are exported for large-output storage
experiments. `compute_linewidth_mesh_chunked_artifact(...)` computes mesh
linewidth from those artifacts one chunk at a time.
`compute_serta_transport_from_epc_mesh_chunked_artifact(...)` computes SERTA
transport from the chunked linewidth reduction plus the existing velocity
providers. `compute_serta_transport_scan_from_epc_mesh_chunked_artifact(...)`
extends this to fixed-linewidth chemical-potential/temperature scans.
`compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact(...)`
exposes the explicit per-scan-point linewidth recomputation convention for
chunked artifacts.
`compute_serta_mobility_si_from_epc_mesh_chunked_artifact(...)` does the same
for SI mobility, and
`compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact(...)` extends it
to fixed-linewidth chemical-potential/temperature scans.
`compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact(...)`
exposes the SI mobility counterpart of the explicit per-scan-point linewidth
recomputation convention. EPC unit constants are centralized in
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

### EPCMeshData Chunked Artifact

Use `save_epc_mesh_chunked_artifact(mesh_data, directory, axis=..., chunk_size=...)`
and `load_epc_mesh_chunked_artifact(directory)`.

The artifact is a directory with:

- `manifest.json`: schema/version, chunk axis, chunk specs, reducer, and mesh
  metadata.
- `weights.npz`: global normalized k/q weights.
- one `EPCData` NPZ per q or k chunk.

Loading validates the manifest, reduces chunk NPZ files with deterministic k/q
reducers, and returns an `EPCMeshData`. This is a storage/reducer contract for
large mesh workflows. It is not a parallel executor, does not require `mpi4py`,
and does not change the public `EPCMeshData` NPZ schema.

`TBSystem.eph.compute_mesh_chunked_artifact(...)` is the first serial streaming
producer for this artifact contract. It computes one q-axis or k-axis chunk at
a time and writes each chunk directly to the artifact directory, instead of
first materializing a full `EPCMeshData` and then splitting it. It is still a
serial Python API, not a multiprocessing/MPI/CUDA executor.

The loader rejects malformed artifacts: inconsistent schema/version, invalid
chunk counts, non-contiguous chunk ranges, wrong reducer names, unsafe
filenames, and invalid weights metadata.

`compute_linewidth_mesh_chunked_artifact(directory, ...)` reads the same
artifact and computes `LinewidthMeshData` without materializing the full
`EPCMeshData`. For q-axis artifacts it accumulates q contributions using the
global q weights; for k-axis artifacts it computes each k chunk and
concatenates the reduced linewidth arrays.

`compute_serta_transport_from_epc_mesh_chunked_artifact(system, directory, ...)`
uses the chunked linewidth reduction, computes band velocities from the
artifact k-points through the existing finite-difference or Hamiltonian-
derivative velocity providers, and returns `TransportData`. It avoids
materializing the full mesh coupling tensor but still evaluates velocities
through the supplied `TBSystem`.

`compute_serta_transport_scan_from_epc_mesh_chunked_artifact(...)` reuses the
same chunked linewidth at the first requested chemical-potential/temperature
point, then applies the non-SI SERTA transport scan helper over the requested
axes. This is a fixed-linewidth scan convention.

`compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact(...)`
is the explicit helper for the alternative convention: it recomputes chunked
linewidth at every requested chemical-potential/temperature point, computes
the selected band velocities once for the artifact k-points, and returns
`TransportScanData` with
`linewidth_scan_convention="per_scan_point_recomputed"`. Existing
fixed-linewidth scan helpers are intentionally not overloaded. The CLI exposes
this artifact workflow through `--epc-artifact` plus
`--linewidth-scan-convention recompute`.

`compute_serta_mobility_si_from_epc_mesh_chunked_artifact(system, directory, ...)`
uses the same chunked linewidth and velocity path, then applies the existing SI
mobility conversion. It supports the same 2D/3D normalization and reciprocal
cell conventions as `compute_serta_mobility_si(...)`.

`compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact(...)` reuses the
same chunked linewidth at the first requested chemical-potential/temperature
point, then applies the existing mobility scan helper over the requested axes.
This matches the current `compute_serta_mobility_scan_si(...)` convention where
linewidth is supplied as a fixed scan input.

`compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact(...)`
is the SI mobility counterpart of the recomputed-linewidth transport scan. It
recomputes chunked linewidth at each scan point and records
`linewidth_scan_convention="per_scan_point_recomputed"`. The CLI exposes this
through `--task mobility --epc-artifact ... --linewidth-scan-convention recompute`.

### Postprocess NPZ

Use the matching save/load APIs:

- `LinewidthData.save_npz()` / `LinewidthData.load_npz()`.
- `LinewidthPathData.save_npz()` / `LinewidthPathData.load_npz()`.
- `LinewidthMeshData.save_npz()` / `LinewidthMeshData.load_npz()`.
- `RelaxationTimeData.save_npz()` / `RelaxationTimeData.load_npz()`.
- `RelaxationTimePathData.save_npz()` / `RelaxationTimePathData.load_npz()`.
- `RelaxationTimeMeshData.save_npz()` / `RelaxationTimeMeshData.load_npz()`.
- `TransportData.save_npz()` / `TransportData.load_npz()`.
- `TransportScanData.save_npz()` / `TransportScanData.load_npz()`.
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

`TransportScanData` stores non-SI SERTA conductivity and carrier density over
chemical-potential and temperature axes. The Python helper
`compute_serta_transport_scan(...)` returns arrays with shape
`(nmu, ntemperatures, 3, 3)` for conductivity and `(nmu, ntemperatures)` for
carrier density. It uses a fixed linewidth input, matching the mobility scan
convention. For chunked EPC mesh artifacts, the explicit helper
`compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact(...)`
can recompute linewidth at each scan point and records
`linewidth_scan_convention="per_scan_point_recomputed"` in metadata.

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
  --q-chunk-size 4 \
  -b 0 1 \
  -o epc_mesh_data.npz
```

`--task mesh-coupling` writes an `EPCMeshData` NPZ. `--k-mesh` generates
electronic k-points with DeePTB's existing k-point mesh helpers. `--q-mesh`
only validates the external phonon q-points in `phonons_mesh.npz`; it does not
calculate phonons. `--chunk-size` enables serial k-point chunk execution and
`--q-chunk-size` enables serial q-point chunk execution. Chunked mesh coupling
still concatenates chunks deterministically into one `EPCMeshData` output.

For direct directory artifacts instead of one in-memory mesh NPZ, use
`--task mesh-artifact`:

```bash
dptb eph \
  --task mesh-artifact \
  -i model.pth \
  -stu structure.vasp \
  -ph phonons_mesh.npz \
  --k-mesh 4 4 1 \
  --q-mesh 4 4 1 \
  --artifact-axis q \
  --q-chunk-size 4 \
  -b 0 1 \
  -o epc_mesh_artifact
```

This task writes a chunked artifact directory containing `manifest.json`,
`weights.npz`, and one `EPCData` NPZ per q or k chunk. `--artifact-axis q`
uses `--q-chunk-size`; `--artifact-axis k` uses `--chunk-size`. The task is a
serial streaming producer and does not require multiprocessing, MPI, or CUDA.

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
electronic k-points and k-point weights. It can also consume a chunked artifact
directory directly:

```bash
dptb eph \
  --task mesh-linewidth \
  --epc-artifact epc_mesh_artifact \
  --chemical-potential 0.15 \
  --temperature 0.025 \
  --sigma 0.01 \
  -o mesh_linewidth.npz
```

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

For chemical-potential and temperature scans, use plural scan arguments:

```bash
dptb eph \
  --task transport \
  -i model.pth \
  -stu structure.vasp \
  --epc-data epc_data.npz \
  --linewidth-data linewidth.npz \
  --chemical-potentials 0.10 0.15 0.20 \
  --temperatures 0.025 0.050 \
  --kpoint-weights weights.npz \
  --spin-degeneracy 2 \
  --volume 1.0 \
  -o transport_scan.npz
```

When `--chemical-potentials` or `--temperatures` is used with
`--task transport`, the CLI writes a `TransportScanData` NPZ. The singular and
plural forms for the same axis are mutually exclusive. The scan uses fixed
linewidth values from `--linewidth-data`.

Per-scan-point linewidth recomputation is exposed through chunked artifacts:

```bash
dptb eph \
  --task transport \
  -i model.pth \
  -stu structure.vasp \
  --epc-artifact epc_mesh_artifact \
  --chemical-potentials 0.10 0.15 0.20 \
  --temperatures 0.025 0.050 \
  --sigma 0.01 \
  --linewidth-scan-convention recompute \
  --spin-degeneracy 2 \
  --volume 1.0 \
  -o transport_recompute_scan.npz
```

For `--epc-artifact`, artifact k-point weights are used directly; do not pass
`--kpoint-weights`, `--epc-data`, or `--linewidth-data` on the same command.

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

For chunked artifacts, per-scan-point linewidth recomputation uses the same
explicit convention flag as transport:

```bash
dptb eph \
  --task mobility \
  -i model.pth \
  -stu structure.vasp \
  --epc-artifact epc_mesh_artifact \
  --chemical-potentials 0.10 0.15 0.20 \
  --temperatures 0.025 0.050 \
  --sigma 0.01 \
  --linewidth-scan-convention recompute \
  --dimension 3d \
  --volume 1.0 \
  -o mobility_recompute_scan.npz
```

As with artifact transport, artifact k-point weights are used directly; do not
pass `--kpoint-weights`, `--epc-data`, or `--linewidth-data` on the same
command.

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

### Coupling Summary

```bash
dptb eph \
  --task coupling-summary \
  --epc-data epc_mesh_data.npz \
  -o coupling_summary.json
```

This task reads an existing `EPCData`, `EPCPathData`, or `EPCMeshData` NPZ and
writes a JSON summary of coupling strength over q-points, k-points, phonon
modes, final bands, initial bands, and band pairs. It does not recompute EPC
and does not introduce a new persistent NPZ schema.

For `EPCMeshData`, summaries use normalized k/q weights by default. Add
`--summary-unweighted` to report raw sums.

### Scattering Map

```bash
dptb eph \
  --task scattering-map \
  --epc-data epc_mesh_data.npz \
  -o scattering_map.json
```

This task reads an existing `EPCData`, `EPCPathData`, or `EPCMeshData` NPZ and
writes JSON q/k/mode/band-resolved coupling-strength maps. These maps are
diagnostics for locating strong EPC channels; they are not energy-conserving
linewidths or full scattering rates. For `EPCMeshData`, maps use normalized
k/q weights by default. Add `--summary-unweighted` to report raw sums.

### Phonon DOS

```bash
dptb eph \
  --task phonon-dos \
  -ph phonons_mesh.npz \
  --dos-grid 0.0 0.5 1.0 1.5 2.0 \
  --dos-sigma 0.05 \
  --broadening gaussian \
  -o phonon_dos.json
```

This task reads external `Phonons` NPZ frequencies and writes a JSON phonon DOS
summary on the requested frequency grid. Frequencies, grid values, and
`--dos-sigma` are in THz. DeePTB does not calculate phonons or force constants
for this task.

### Eliashberg-Like Spectrum

```bash
dptb eph \
  --task eliashberg \
  --epc-data epc_mesh_data.npz \
  --dos-grid 0.0 0.5 1.0 1.5 2.0 \
  --dos-sigma 0.05 \
  --broadening gaussian \
  -o eliashberg.json
```

This task reads an existing `EPCData`, `EPCPathData`, or `EPCMeshData` NPZ and
writes a JSON coupling-strength-weighted phonon frequency spectrum. It is a
DeePTB-native diagnostic, not a claim of full material-specific Eliashberg
theory. For `EPCMeshData`, summaries use normalized k/q weights by default; add
`--summary-unweighted` for raw sums.

## Current v1 Limits

- SCC-corrected EPC is not supported. `use_scc=True` raises
  `NotImplementedError`.
- Path workflows currently support fixed electronic k-points plus q-path only.
  k-path plus fixed-q is not implemented.
- Mesh workflows currently use serial execution. K-point and q-point chunking
  are available for `mesh-coupling`; chunked artifacts are available through
  `TBSystem.eph.compute_mesh_chunked_artifact(...)`, Python artifact helpers,
  and `dptb eph --task mesh-artifact`. Summary-first artifact consumers are
  available for mesh-linewidth, transport, and mobility. Multiprocessing, MPI,
  CUDA backends, and multi-axis streaming artifact production are future work.
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
  `--chemical-potentials` / `--temperatures`. The chunked-artifact API and CLI
  also provide
  `compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact(...)`
  / `--epc-artifact --linewidth-scan-convention recompute` for explicit
  per-scan-point linewidth recomputation.
- Non-SI transport scans are available through
  `compute_serta_transport_scan(...)` and the same CLI scan arguments on
  `--task transport`. The chunked-artifact API and CLI also provide
  `compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact(...)`
  / `--epc-artifact --linewidth-scan-convention recompute` for explicit
  per-scan-point linewidth recomputation.

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
