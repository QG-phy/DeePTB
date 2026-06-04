# DeePTB SCC EPC Design

## Summary

This document defines the boundary for self-consistent-charge electron-phonon
coupling (SCC EPC) in DeePTB. It is intentionally a design document, not an
implementation switch. Current EPC workflows must continue to reject
`use_scc=True` until the formula, reference data, and tests described here are
implemented together.

The key point is that SCC EPC is not obtained by simply passing
`use_scc=True` to the existing non-SCC EPC path. SCC changes the Hamiltonian
through a converged charge-dependent correction, and the derivative with
respect to atomic displacement must define whether charges are held fixed or
allowed to respond self-consistently.

## Current DeePTB SCC Flow

The current SCC implementation is centered on `TBSystem` and `SKSCC`:

- `TBSystem.enable_scc(...)` configures an `SKSCC` engine and stores SCC run
  options.
- `TBSystem.run_scc(...)` runs SCC iterations on the current structure and
  caches an `SCCState`.
- `SCCState` stores:
  - convergence status;
  - `scc_shift`;
  - Mulliken charge;
  - delta charge;
  - Fermi level;
  - total electronic energy.
- `TBSystem.get_hk(..., use_scc=True)` builds the bare model `H(k)` and `S(k)`,
  then adds `SKSCC.cal_scc_hk(...)`.
- `TBSystem.get_hk(..., with_derivative=True, use_scc=True)` currently raises
  `NotImplementedError`.

Inside `SKSCC.run_iters(...)`, the converged SCC correction is built from:

- Mulliken charges;
- reference valence charges;
- `delta_charge`;
- the Gamma interaction matrix;
- `scc_shift = Gamma @ delta_charge`;
- `scc_shift_energy = 0.5 * delta_charge @ scc_shift`;
- `cal_scc_hk(...)`, which applies the SCC shift through the overlap-aware
  SCC Hamiltonian correction.

This means the SCC Hamiltonian depends on both structure and converged charges:

```text
H_scc_total(k, R) = H0(k, R) + H_scc(k, R, delta_q(R))
```

The non-SCC EPC implementation only differentiates `H0` and optional `S`.
It does not define how `delta_q(R)` changes under displacement.

## EPC Formula Decomposition

For a displacement coordinate `u`, SCC EPC should be decomposed as:

```text
dH_scc_total/du =
    dH0/du
  + dH_scc/dS * dS/du
  + dH_scc/dGamma * dGamma/du
  + dH_scc/d(delta_q) * d(delta_q)/du
```

The exact grouping may differ in implementation, but the design must preserve
these physical contributions:

- Bare derivative:
  - the existing non-SCC `dH0/dR` term.
- Overlap derivative:
  - required when SCC correction is overlap-mediated.
  - includes changes in `S(k, R)` used by `cal_scc_hk(...)`.
- Frozen-charge SCC derivative:
  - derivative of the SCC Hamiltonian at fixed converged `delta_q` /
    `scc_shift`.
  - includes explicit structure dependence such as overlap and Gamma, depending
    on the chosen implementation.
- Relaxed-charge SCC derivative:
  - includes the charge response `d(delta_q)/dR`.
  - this is the full self-consistent response term.

The EPC matrix element must also keep the existing generalized-eigenproblem
overlap correction convention:

```text
g_mn ~ <psi_m(k+q)| dH/du - E_reference dS/du |psi_n(k)>
```

The precise choice of `E_reference` and phase/gauge convention must match the
non-SCC EPC convention before SCC is enabled.

## Two Supported Definitions

SCC EPC must expose its definition explicitly in API metadata.

### Frozen-Charge SCC EPC

Frozen-charge SCC EPC differentiates the SCC Hamiltonian while holding the
converged charge state fixed:

```text
delta_q = delta_q_converged(R0)
scc_shift = scc_shift_converged(R0)
```

This is useful as a controlled intermediate and a debugging/reference path, but
it is not full self-consistent SCC EPC. Metadata must state something like:

```text
scc_epc_definition = "frozen_charge"
```

### Relaxed-Charge SCC EPC

Relaxed-charge SCC EPC includes the self-consistent charge response:

```text
d(delta_q)/dR != 0
```

This is the physically complete SCC EPC target. It requires either an analytic
charge-response provider or a finite-difference SCC provider that reruns SCC for
each displaced structure with controlled convergence settings.

Metadata must state:

```text
scc_epc_definition = "relaxed_charge"
```

## Why `use_scc=True` Is Not Enough

Directly enabling `use_scc=True` in existing EPC providers is wrong for three
reasons:

- The current finite-difference providers reset structure through
  `TBSystem.set_atoms(...)`, which invalidates cached SCC state.
- `TBSystem.get_hk(..., with_derivative=True, use_scc=True)` is explicitly
  unsupported, so Hamiltonian-derivative velocity has no SCC derivative path.
- Even if `get_hk(use_scc=True)` returns an SCC-corrected `H(k)`, the derivative
  definition is ambiguous unless the provider states whether charges are frozen
  or re-converged for every displacement.

Therefore current behavior is correct: EPC should raise `NotImplementedError`
for SCC until this design is implemented and tested.

## Provider Boundary

The SCC implementation should be added behind provider interfaces, not by
modifying `OrbitalMapper`, Hamiltonian transforms, SCC charge update, Hubbard U,
or existing non-SCC EPC formulas.

Proposed provider layers:

- `SCCEPCProvider`
  - common protocol for SCC derivative providers.
  - records SCC definition, convergence settings, displacement, and metadata.
- `FrozenChargeSCCProvider`
  - computes SCC Hamiltonian derivatives with fixed `SCCState`.
  - must define whether Gamma is frozen or structure-updated.
- `SCCChargeResponseProvider`
  - analytic or semi-analytic provider for `d(delta_q)/dR`.
  - not required for the first implementation unless the formula is complete.
- `FiniteDifferenceSCCProvider`
  - reruns SCC on plus/minus displaced structures.
  - target provider for relaxed-charge smoke/reference tests.
  - must restore the original structure and SCC state in `finally`.

The existing non-SCC provider API should remain intact:

- `FDProvider`;
- `SupercellFD`;
- Hamiltonian-derivative velocity provider.

SCC providers should be opt-in and should not change default non-SCC behavior.

## Reference Strategy

The existing Graphene reference is non-SCC. It is valuable for non-SCC EPC
regression, but it cannot prove SCC EPC correctness.

Required references before enabling SCC EPC:

- one small SCC smoke fixture;
- explicit record of SCC convergence settings;
- expected frozen-charge or relaxed-charge convention;
- regression arrays for at least one derivative or coupling output;
- metadata proving the SCC EPC definition.

Recommended sequence:

1. Build a tiny hBN-like or similarly polar SCC case if reliable local data
   exists.
2. Use finite-difference SCC reruns as the first relaxed-charge reference.
3. Add dftbephy SCC comparison only after confirming whether its convention is
   frozen-charge, relaxed-charge, or something else.

Do not use a non-SCC Graphene passing result as evidence that SCC EPC works.

## Tests Required Before Implementation Is Enabled

Before any user-facing `use_scc=True` EPC workflow is enabled, tests must cover:

- current SCC rejection remains explicit until implementation lands;
- provider restores original structure after displaced calculations;
- provider restores or invalidates SCC state intentionally after displaced
  calculations;
- frozen-charge provider numerical reference, if implemented;
- relaxed-charge finite-difference reference, if implemented;
- shape and finite-value validation for SCC derivative payloads;
- metadata includes SCC definition and SCC convergence settings;
- `TBSystem.get_hk(..., with_derivative=True, use_scc=True)` remains rejected
  unless SCC derivative semantics are implemented there directly;
- non-SCC EPC tests remain unchanged.

Development references may be hardcoded temporarily, but every such use must be
marked with `TODO(epc-fixture)` and kept out of default CI until it is converted
to a lightweight in-repo fixture.

## Core Modification Boundary

The first SCC EPC implementation should avoid modifying:

- `OrbitalMapper`;
- Hamiltonian transform code;
- SCC mixer logic;
- Hubbard U / Gamma parameter construction;
- charge update semantics.

If implementation proves that one of these core areas must change, write an ADR
or update this design document first. The preferred path is to adapt existing
outputs through an EPC-layer provider.

## Acceptance Criteria

SCC EPC can move from unsupported to implemented only when all of these are
true:

- this design has been reviewed and updated with final formula choices;
- a SCC reference/smoke fixture exists;
- the API names frozen-charge vs relaxed-charge explicitly;
- metadata records SCC definition and convergence settings;
- focused tests pass for SCC and non-SCC EPC;
- default non-SCC EPC behavior is unchanged.

Until then, `use_scc=True` must continue to raise `NotImplementedError` in EPC
workflows.
