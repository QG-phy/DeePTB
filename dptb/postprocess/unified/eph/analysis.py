from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from typing import Union

import numpy as np
from scipy import constants as scipy_constants

from dptb.postprocess.unified.eph.data import EPCData, _merge_metadata, _metadata_from_npz, _metadata_to_json
from dptb.postprocess.unified.eph.utils import (
    as_array,
    validate_finite_nonnegative_scalar,
    validate_finite_scalar,
    normalize_integer_array,
    normalize_integer_indices,
    normalize_kpoints,
    validate_finite_positive_scalar,
)


THZ_TO_EV = scipy_constants.h * scipy_constants.tera / scipy_constants.e
HBAR_EV_S = scipy_constants.hbar / scipy_constants.e
LINEWIDTH_NPZ_SCHEMA_VERSION = 1
RELAXATION_TIME_NPZ_SCHEMA_VERSION = 1
TRANSPORT_NPZ_SCHEMA_VERSION = 1
SUBSPACE_COUPLING_NPZ_SCHEMA_VERSION = 1


@dataclass
class LinewidthData:
    """Electron linewidths from electron-phonon coupling data."""

    linewidth: np.ndarray
    absorption: np.ndarray
    emission: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.linewidth = np.asarray(self.linewidth, dtype=float)
        self.absorption = np.asarray(self.absorption, dtype=float)
        self.emission = np.asarray(self.emission, dtype=float)
        if self.absorption.shape != self.linewidth.shape:
            raise ValueError("absorption must have the same shape as linewidth.")
        if self.emission.shape != self.linewidth.shape:
            raise ValueError("emission must have the same shape as linewidth.")
        if self.linewidth.size == 0:
            raise ValueError("linewidth must be non-empty.")
        if self.linewidth.ndim not in {2, 3}:
            raise ValueError("linewidth must have shape (nk, nbands) or (nk, nbands, nmodes).")
        if not np.all(np.isfinite(self.linewidth)):
            raise ValueError("linewidth must contain finite values.")
        if not np.all(np.isfinite(self.absorption)) or np.any(self.absorption < 0.0):
            raise ValueError("absorption must contain finite non-negative values.")
        if not np.all(np.isfinite(self.emission)) or np.any(self.emission < 0.0):
            raise ValueError("emission must contain finite non-negative values.")
        if np.any(self.linewidth < 0.0):
            raise ValueError("linewidth must contain non-negative values.")
        if not np.allclose(self.linewidth, self.absorption + self.emission):
            raise ValueError("linewidth must equal absorption + emission.")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_linewidth",
                "schema_version": LINEWIDTH_NPZ_SCHEMA_VERSION,
                "linewidth_unit": "eV",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save electron-phonon linewidth postprocess data to NPZ."""
        np.savez_compressed(
            path,
            elph_linewidth=self.linewidth,
            elph_linewidth_absorption=self.absorption,
            elph_linewidth_emission=self.emission,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "LinewidthData":
        """Load electron-phonon linewidth postprocess data from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                linewidth=data["elph_linewidth"],
                absorption=data["elph_linewidth_absorption"],
                emission=data["elph_linewidth_emission"],
                metadata=metadata,
            )


@dataclass
class RelaxationTimeData:
    """Electron relaxation times derived from electron-phonon linewidths."""

    relaxation_time: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.relaxation_time = np.asarray(self.relaxation_time, dtype=float)
        if self.relaxation_time.size == 0:
            raise ValueError("relaxation_time must be non-empty.")
        if self.relaxation_time.ndim not in {2, 3}:
            raise ValueError("relaxation_time must have shape (nk, nbands) or (nk, nbands, nmodes).")
        if not np.all(np.isfinite(self.relaxation_time)) or np.any(self.relaxation_time <= 0.0):
            raise ValueError("relaxation_time must contain finite positive values.")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_relaxation_time",
                "schema_version": RELAXATION_TIME_NPZ_SCHEMA_VERSION,
                "relaxation_time_unit": "s",
                "convention": "hbar_over_2linewidth",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save electron-phonon relaxation-time postprocess data to NPZ."""
        np.savez_compressed(
            path,
            elph_relaxation_time=self.relaxation_time,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "RelaxationTimeData":
        """Load electron-phonon relaxation-time postprocess data from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                relaxation_time=data["elph_relaxation_time"],
                metadata=metadata,
            )


@dataclass
class TransportData:
    """SERTA transport data derived from electron-phonon linewidths."""

    conductivity: np.ndarray
    carrier_density: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.conductivity = np.asarray(self.conductivity, dtype=float)
        self.carrier_density = np.asarray(self.carrier_density, dtype=float)
        if self.conductivity.shape != (3, 3):
            raise ValueError("conductivity must have shape (3, 3).")
        if not np.all(np.isfinite(self.conductivity)):
            raise ValueError("conductivity must contain finite values.")
        if not np.all(np.isfinite(self.carrier_density)):
            raise ValueError("carrier_density must contain finite values.")
        if np.any(self.carrier_density < 0.0):
            raise ValueError("carrier_density must contain non-negative values.")
        if self.carrier_density.ndim > 1:
            raise ValueError("carrier_density must be a scalar or one-dimensional array.")
        if self.carrier_density.size == 0:
            raise ValueError("carrier_density must be non-empty.")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_transport",
                "schema_version": TRANSPORT_NPZ_SCHEMA_VERSION,
                "method": "SERTA",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save SERTA transport postprocess data to NPZ."""
        np.savez_compressed(
            path,
            transport_conductivity=self.conductivity,
            transport_carrier_density=self.carrier_density,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "TransportData":
        """Load SERTA transport postprocess data from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                conductivity=data["transport_conductivity"],
                carrier_density=data["transport_carrier_density"],
                metadata=metadata,
            )


@dataclass
class SubspaceCouplingData:
    """Gauge-invariant EPC strength aggregated over contiguous band subspaces."""

    strength: np.ndarray
    final_group_bounds: np.ndarray
    initial_group_bounds: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.strength = np.asarray(self.strength, dtype=float)
        self.final_group_bounds = normalize_integer_array(self.final_group_bounds, "final_group_bounds")
        self.initial_group_bounds = normalize_integer_array(self.initial_group_bounds, "initial_group_bounds")
        if self.strength.ndim < 2:
            raise ValueError("strength must have at least final and initial subspace axes.")
        if not np.all(np.isfinite(self.strength)) or np.any(self.strength < 0.0):
            raise ValueError("strength must contain finite non-negative values.")
        _validate_group_bounds(self.final_group_bounds, "final_group_bounds")
        _validate_group_bounds(self.initial_group_bounds, "initial_group_bounds")
        if self.strength.shape[-2:] != (self.final_group_bounds.shape[0], self.initial_group_bounds.shape[0]):
            raise ValueError("strength subspace axes must match final_group_bounds and initial_group_bounds.")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_subspace_coupling",
                "schema_version": SUBSPACE_COUPLING_NPZ_SCHEMA_VERSION,
                "coupling_strength_unit": "eV^2",
                "aggregation": "frobenius_norm_squared",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save gauge-invariant subspace EPC strength data to NPZ."""
        np.savez_compressed(
            path,
            elph_subspace_strength=self.strength,
            final_group_bounds=self.final_group_bounds,
            initial_group_bounds=self.initial_group_bounds,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "SubspaceCouplingData":
        """Load gauge-invariant subspace EPC strength data from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                strength=data["elph_subspace_strength"],
                final_group_bounds=data["final_group_bounds"],
                initial_group_bounds=data["initial_group_bounds"],
                metadata=metadata,
            )


def compute_linewidth(
    epc_data: EPCData,
    chemical_potential: float,
    temperature: float,
    sigma: float,
    broadening: str = "gaussian",
    mode_resolved: bool = False,
    frequency_floor: float = 1e-5,
) -> LinewidthData:
    """Compute electron-phonon linewidths from ``EPCData``.

    Frequencies stored in ``EPCData`` are in THz. They are converted to eV for
    the energy-conservation broadening used in this postprocess step.
    """
    chemical_potential = validate_finite_scalar(chemical_potential, "chemical_potential")
    temperature = validate_finite_positive_scalar(temperature, "temperature")
    sigma = validate_finite_positive_scalar(sigma, "sigma")
    frequency_floor = validate_finite_positive_scalar(frequency_floor, "frequency_floor")
    mode_resolved = _validate_bool(mode_resolved, "mode_resolved")
    if np.any(epc_data.frequencies < 0.0):
        raise ValueError("frequencies must be non-negative for linewidth postprocess.")

    if not isinstance(broadening, str):
        raise ValueError("broadening must be either 'gaussian' or 'lorentzian'.")
    broadening = broadening.lower()
    if broadening not in {"gaussian", "lorentzian"}:
        raise ValueError("broadening must be either 'gaussian' or 'lorentzian'.")

    frequencies_ev = np.maximum(epc_data.frequencies, frequency_floor) * THZ_TO_EV
    nq, nk, nmodes, nsel, _ = epc_data.coupling_strength.shape
    shape = (nk, nsel, nmodes) if mode_resolved else (nk, nsel)
    absorption = np.zeros(shape, dtype=float)
    emission = np.zeros(shape, dtype=float)

    for ik in range(nk):
        for initial_band in range(nsel):
            eps_initial = epc_data.eigenvalues_k[ik, initial_band]
            for iq in range(nq):
                for mode in range(nmodes):
                    omega = frequencies_ev[iq, mode]
                    bose_occupation = _bose(omega / temperature)
                    for final_band in range(nsel):
                        eps_final = epc_data.eigenvalues_kq[iq, ik, final_band]
                        strength = epc_data.coupling_strength[iq, ik, mode, final_band, initial_band]
                        absorption_weight = _broadening(
                            eps_initial + omega - eps_final,
                            sigma,
                            broadening,
                        )
                        emission_weight = _broadening(
                            eps_initial - omega - eps_final,
                            sigma,
                            broadening,
                        )
                        absorption_term = strength * (
                            bose_occupation + _fermi((eps_final - chemical_potential) / temperature)
                        ) * absorption_weight
                        emission_term = strength * (
                            bose_occupation
                            + 1.0
                            - _fermi((eps_final - chemical_potential) / temperature)
                        ) * emission_weight

                        if mode_resolved:
                            absorption[ik, initial_band, mode] += absorption_term
                            emission[ik, initial_band, mode] += emission_term
                        else:
                            absorption[ik, initial_band] += absorption_term
                            emission[ik, initial_band] += emission_term

    prefactor = 2.0 * np.pi / nq
    absorption *= prefactor
    emission *= prefactor
    linewidth = absorption + emission
    return LinewidthData(
        linewidth=linewidth,
        absorption=absorption,
        emission=emission,
        metadata={
            "chemical_potential": chemical_potential,
            "temperature": temperature,
            "temperature_unit": "eV",
            "sigma": sigma,
            "sigma_unit": "eV",
            "broadening": broadening,
            "mode_resolved": mode_resolved,
            "linewidth_unit": "eV",
            "frequency_unit_input": epc_data.metadata.get("frequency_unit", "THz"),
            "frequency_unit_internal": "eV",
            "frequency_floor": frequency_floor,
            "frequency_floor_unit": "THz",
            "thz_to_ev": THZ_TO_EV,
        },
    )


def compute_relaxation_time(linewidth_data: LinewidthData, sum_modes: bool = False) -> RelaxationTimeData:
    """Compute relaxation time from EPC linewidths.

    The convention is ``tau = hbar / (2 * linewidth)`` with linewidth in eV,
    giving relaxation time in seconds. Mode-resolved linewidths keep their
    mode axis by default; set ``sum_modes=True`` to sum the last axis first.
    """
    sum_modes = _validate_bool(sum_modes, "sum_modes")
    linewidth = np.asarray(linewidth_data.linewidth, dtype=float)
    if sum_modes:
        if linewidth.ndim < 3:
            raise ValueError("sum_modes=True requires a mode-resolved linewidth array.")
        linewidth = linewidth.sum(axis=-1)
    if not np.all(np.isfinite(linewidth)) or np.any(linewidth <= 0.0):
        raise ValueError("linewidth values must be finite and positive for relaxation time.")

    return RelaxationTimeData(
        relaxation_time=HBAR_EV_S / (2.0 * linewidth),
        metadata={
            "linewidth_schema": linewidth_data.metadata.get("schema"),
            "linewidth_unit": linewidth_data.metadata.get("linewidth_unit", "eV"),
            "relaxation_time_unit": "s",
            "convention": "hbar_over_2linewidth",
            "hbar_eV_s": HBAR_EV_S,
            "mode_resolved_input": linewidth_data.linewidth.ndim == 3,
            "sum_modes": sum_modes,
        },
    )


def compute_serta_conductivity(
    eigenvalues: np.ndarray,
    velocities: np.ndarray,
    linewidth: np.ndarray,
    chemical_potential: float,
    temperature: float,
    kpoint_weights: Optional[np.ndarray] = None,
    spin_degeneracy: int = 1,
    volume: float = 1.0,
) -> TransportData:
    """Compute SERTA conductivity from linewidths and band velocities.

    ``eigenvalues`` has shape ``(nk, nbands)`` and is expressed in eV.
    ``velocities`` has shape ``(nk, nbands, 3)`` and is expected in the same
    velocity convention as DeePTB's electronic postprocess. ``linewidth`` is an
    inverse lifetime in eV as returned by :func:`compute_linewidth`.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    linewidth = np.asarray(linewidth, dtype=float)
    chemical_potential = validate_finite_scalar(chemical_potential, "chemical_potential")
    temperature = validate_finite_positive_scalar(temperature, "temperature")
    volume = validate_finite_positive_scalar(volume, "volume")
    try:
        spin_degeneracy_value = np.asarray(spin_degeneracy)
    except (TypeError, ValueError):
        raise ValueError("spin_degeneracy must be a positive integer.") from None
    if spin_degeneracy_value.shape != ():
        raise ValueError("spin_degeneracy must be a positive integer.")
    if isinstance(spin_degeneracy, (bool, np.bool_)):
        raise ValueError("spin_degeneracy must be a positive integer.")
    if not np.issubdtype(spin_degeneracy_value.dtype, np.number):
        raise ValueError("spin_degeneracy must be a positive integer.")
    try:
        spin_degeneracy_float = float(spin_degeneracy_value)
    except (TypeError, ValueError):
        raise ValueError("spin_degeneracy must be a positive integer.") from None
    if not np.isfinite(spin_degeneracy_float) or spin_degeneracy_float <= 0.0:
        raise ValueError("spin_degeneracy must be a positive integer.")
    if int(spin_degeneracy_float) != spin_degeneracy_float:
        raise ValueError("spin_degeneracy must be a positive integer.")
    spin_degeneracy = int(spin_degeneracy_float)
    if eigenvalues.ndim != 2:
        raise ValueError("eigenvalues must have shape (nk, nbands).")
    if eigenvalues.shape[0] == 0:
        raise ValueError("eigenvalues must contain at least one k-point.")
    if eigenvalues.shape[1] == 0:
        raise ValueError("eigenvalues must contain at least one band.")
    if velocities.shape != (*eigenvalues.shape, 3):
        raise ValueError("velocities must have shape (nk, nbands, 3).")
    if linewidth.shape != eigenvalues.shape:
        raise ValueError("linewidth must have shape (nk, nbands).")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("eigenvalues must contain finite values.")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("velocities must contain finite values.")
    if not np.all(np.isfinite(linewidth)) or np.any(linewidth <= 0.0):
        raise ValueError("linewidth values must be finite and positive for SERTA conductivity.")

    nk = eigenvalues.shape[0]
    if kpoint_weights is None:
        weights = np.full((nk,), 1.0 / nk, dtype=float)
    else:
        weights = np.asarray(kpoint_weights, dtype=float)
        if weights.shape != (nk,):
            raise ValueError("kpoint_weights must have shape (nk,).")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("kpoint_weights must contain finite non-negative values.")
        weight_sum = weights.sum()
        if weight_sum <= 0.0:
            raise ValueError("kpoint_weights must have a positive sum.")
        weights = weights / weight_sum

    conductivity = np.zeros((3, 3), dtype=float)
    carrier_density = 0.0
    for ik in range(nk):
        for iband in range(eigenvalues.shape[1]):
            eps = eigenvalues[ik, iband]
            occupation = _fermi((eps - chemical_potential) / temperature)
            transport_weight = -_dfermi_deps((eps - chemical_potential) / temperature) / temperature
            weight = spin_degeneracy * weights[ik] / volume
            carrier_density += weight * occupation
            conductivity += (
                weight
                * transport_weight
                * np.outer(velocities[ik, iband], velocities[ik, iband])
                / linewidth[ik, iband]
            )

    return TransportData(
        conductivity=conductivity,
        carrier_density=np.asarray(carrier_density),
        metadata={
            "chemical_potential": chemical_potential,
            "temperature": temperature,
            "temperature_unit": "eV",
            "spin_degeneracy": spin_degeneracy,
            "volume": volume,
            "energy_unit": "eV",
            "linewidth_unit": "eV",
            "method": "SERTA",
        },
    )


def compute_band_velocities_finite_difference(
    system,
    kpoints: np.ndarray,
    bands: Optional[Sequence[int]] = None,
    delta: float = 1e-4,
    use_scc: bool = False,
    **solver_kwargs,
) -> np.ndarray:
    """Estimate band velocities by central finite differences in k-space.

    The returned velocity convention is eV per fractional reciprocal-coordinate
    component. This is intentionally a thin workflow bridge over
    ``system.get_eigenvalues``; it does not perform SI unit conversion.
    """
    if use_scc:
        raise NotImplementedError("SCC-corrected electron-phonon transport is not supported in v1.")
    delta = validate_finite_positive_scalar(delta, "delta")

    kpoints = normalize_kpoints(kpoints)
    reference_eigenvalues = _get_system_eigenvalues(system, kpoints, use_scc=use_scc, **solver_kwargs)
    band_indices = _normalize_band_indices(bands, reference_eigenvalues.shape[1])
    velocities = np.zeros((kpoints.shape[0], band_indices.shape[0], 3), dtype=float)

    for axis in range(3):
        shift = np.zeros(3, dtype=float)
        shift[axis] = delta
        eig_plus = _get_system_eigenvalues(system, kpoints + shift, use_scc=use_scc, **solver_kwargs)
        eig_minus = _get_system_eigenvalues(system, kpoints - shift, use_scc=use_scc, **solver_kwargs)
        if eig_plus.shape != reference_eigenvalues.shape or eig_minus.shape != reference_eigenvalues.shape:
            raise ValueError("system.get_eigenvalues must return a consistent eigenvalue shape for shifted kpoints.")
        velocities[:, :, axis] = (eig_plus[:, band_indices] - eig_minus[:, band_indices]) / (2.0 * delta)

    return velocities


def compute_serta_transport_from_epc(
    system,
    epc_data: EPCData,
    linewidth_data: LinewidthData,
    chemical_potential: float,
    temperature: float,
    kpoint_weights: Optional[np.ndarray] = None,
    spin_degeneracy: int = 1,
    volume: float = 1.0,
    velocity_delta: float = 1e-4,
    use_scc: bool = False,
    **solver_kwargs,
) -> TransportData:
    """Compute SERTA transport using EPC linewidths and finite-difference velocities."""
    velocity_delta = validate_finite_positive_scalar(velocity_delta, "velocity_delta")
    linewidth = linewidth_data.linewidth
    if linewidth.ndim == 3:
        linewidth = linewidth.sum(axis=-1)
    if linewidth.shape != epc_data.eigenvalues_k.shape:
        raise ValueError("linewidth_data.linewidth must match EPCData eigenvalues_k shape after mode summation.")

    velocities = compute_band_velocities_finite_difference(
        system=system,
        kpoints=epc_data.kpoints,
        bands=epc_data.band_indices,
        delta=velocity_delta,
        use_scc=use_scc,
        **solver_kwargs,
    )
    result = compute_serta_conductivity(
        eigenvalues=epc_data.eigenvalues_k,
        velocities=velocities,
        linewidth=linewidth,
        chemical_potential=chemical_potential,
        temperature=temperature,
        kpoint_weights=kpoint_weights,
        spin_degeneracy=spin_degeneracy,
        volume=volume,
    )
    result.metadata.update(
        {
            "velocity_source": "finite_difference",
            "velocity_delta": velocity_delta,
            "velocity_unit": "eV/fractional_reciprocal_coordinate",
            "band_indices": epc_data.band_indices,
            "epc_schema": epc_data.metadata.get("schema"),
            "linewidth_schema": linewidth_data.metadata.get("schema"),
        }
    )
    return result


def find_degenerate_band_groups(eigenvalues: np.ndarray, tolerance: float = 1e-5) -> list:
    """Group adjacent bands whose energies are degenerate within ``tolerance``.

    This helper preserves band order and is intended for postprocess diagnostics.
    It does not reorder bands or fix eigenvector gauges.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a one-dimensional band-energy array.")
    tolerance = validate_finite_nonnegative_scalar(tolerance, "tolerance")
    if eigenvalues.size == 0:
        return []

    groups = []
    start = 0
    for iband in range(1, eigenvalues.shape[0]):
        if abs(eigenvalues[iband] - eigenvalues[iband - 1]) > tolerance:
            groups.append(np.arange(start, iband, dtype=int))
            start = iband
    groups.append(np.arange(start, eigenvalues.shape[0], dtype=int))
    return groups


def compute_subspace_coupling_strength(
    coupling_matrix: np.ndarray,
    final_groups: Sequence[Sequence[int]],
    initial_groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """Aggregate EPC strength between band subspaces.

    The aggregation is a Frobenius norm squared over each final/initial band
    block. It is invariant under unitary rotations inside either subspace, which
    makes it useful for degenerate-band diagnostics.
    """
    coupling_matrix = np.asarray(coupling_matrix, dtype=complex)
    if coupling_matrix.ndim < 2:
        raise ValueError("coupling_matrix must have at least final and initial band axes.")
    if not np.all(np.isfinite(coupling_matrix)):
        raise ValueError("coupling_matrix must contain finite values.")
    nfinal, ninitial = coupling_matrix.shape[-2:]
    normalized_final = _normalize_groups(final_groups, nfinal, "final_groups")
    normalized_initial = _normalize_groups(initial_groups, ninitial, "initial_groups")

    out = np.zeros((*coupling_matrix.shape[:-2], len(normalized_final), len(normalized_initial)), dtype=float)
    for ifinal, final_group in enumerate(normalized_final):
        for iinitial, initial_group in enumerate(normalized_initial):
            block = coupling_matrix[..., final_group[:, None], initial_group]
            out[..., ifinal, iinitial] = np.sum(np.abs(block) ** 2, axis=(-2, -1))
    return out


def compute_subspace_coupling_data(
    epc_data: EPCData,
    final_groups: Sequence[Sequence[int]],
    initial_groups: Optional[Sequence[Sequence[int]]] = None,
) -> SubspaceCouplingData:
    """Aggregate ``EPCData.coupling_matrix`` over contiguous band subspaces."""
    if initial_groups is None:
        initial_groups = final_groups
    final_bounds = _groups_to_bounds(final_groups, epc_data.coupling_matrix.shape[-2], "final_groups")
    initial_bounds = _groups_to_bounds(initial_groups, epc_data.coupling_matrix.shape[-1], "initial_groups")
    strength = compute_subspace_coupling_strength(epc_data.coupling_matrix, final_groups, initial_groups)
    return SubspaceCouplingData(
        strength=strength,
        final_group_bounds=final_bounds,
        initial_group_bounds=initial_bounds,
        metadata={
            "epc_schema": epc_data.metadata.get("schema"),
            "source": "EPCData.coupling_matrix",
        },
    )


def _broadening(x: float, sigma: float, broadening: str) -> float:
    if broadening == "gaussian":
        return np.exp(-(x**2) / (2.0 * sigma**2)) / (np.sqrt(2.0 * np.pi) * sigma)
    return (sigma / 2.0) / (x**2 + (sigma / 2.0) ** 2) / np.pi


def _validate_bool(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean.")
    return bool(value)


def _bose(x: float) -> float:
    if x > 700.0:
        return 0.0
    return 1.0 / np.expm1(x)


def _fermi(x: float) -> float:
    if x > 700.0:
        return 0.0
    if x < -700.0:
        return 1.0
    return 1.0 / (np.exp(x) + 1.0)


def _dfermi_deps(x: float) -> float:
    if x > 700.0 or x < -700.0:
        return 0.0
    exp_x = np.exp(x)
    return -exp_x / (exp_x + 1.0) ** 2


def _get_system_eigenvalues(system, kpoints: np.ndarray, use_scc: bool = False, **solver_kwargs) -> np.ndarray:
    _, eigenvalues = system.get_eigenvalues(k_points=kpoints, use_scc=use_scc, **solver_kwargs)
    eigenvalues = as_array(eigenvalues, dtype=float)
    if eigenvalues.ndim != 2:
        raise ValueError("system.get_eigenvalues must return eigenvalues with shape (nk, nbands).")
    if eigenvalues.shape[0] != kpoints.shape[0]:
        raise ValueError("system.get_eigenvalues returned an eigenvalue count inconsistent with kpoints.")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("system.get_eigenvalues must return finite eigenvalues.")
    return eigenvalues


def _normalize_band_indices(bands: Optional[Sequence[int]], nbands: int) -> np.ndarray:
    if nbands <= 0:
        raise ValueError("system.get_eigenvalues must return at least one band.")
    if bands is None:
        return np.arange(nbands, dtype=int)
    band_indices = normalize_integer_indices(bands, "bands")
    if np.any(band_indices < 0) or np.any(band_indices >= nbands):
        raise ValueError("bands contains an index outside the available band range.")
    return band_indices


def _normalize_groups(groups: Sequence[Sequence[int]], size: int, name: str) -> list:
    if len(groups) == 0:
        raise ValueError(f"{name} must contain at least one band group.")
    normalized = []
    for group in groups:
        arr = normalize_integer_indices(group, name)
        if np.any(arr < 0) or np.any(arr >= size):
            raise ValueError(f"{name} contains an index outside the available band range.")
        if np.unique(arr).size != arr.size:
            raise ValueError(f"{name} must not contain duplicate band indices within a group.")
        normalized.append(arr)
    return normalized


def _groups_to_bounds(groups: Sequence[Sequence[int]], size: int, name: str) -> np.ndarray:
    normalized = _normalize_groups(groups, size, name)
    bounds = np.zeros((len(normalized), 2), dtype=int)
    for igroup, group in enumerate(normalized):
        start = int(group[0])
        stop = int(group[-1]) + 1
        if not np.array_equal(group, np.arange(start, stop, dtype=int)):
            raise ValueError(f"{name} must contain contiguous band-index groups for NPZ persistence.")
        bounds[igroup] = (start, stop)
    return bounds


def _validate_group_bounds(bounds: np.ndarray, name: str) -> None:
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"{name} must have shape (ngroups, 2).")
    if bounds.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one [start, stop) range.")
    if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError(f"{name} must contain non-negative [start, stop) ranges.")
