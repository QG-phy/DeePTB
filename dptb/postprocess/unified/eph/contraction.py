from typing import Optional, Sequence, Tuple

import numpy as np

from dptb.postprocess.unified.eph.utils import (
    as_array,
    normalize_integer_indices,
    normalize_kpoints,
    normalize_orbital_slices,
    validate_finite_positive_scalar,
)
from dptb.utils.constants import EPC_PREFAC_AMU_THZ


def compute_coupling_matrix(
    eigenvalues_k: np.ndarray,
    eigenvectors_k: np.ndarray,
    eigenvalues_kq: np.ndarray,
    eigenvectors_kq: np.ndarray,
    h_derivatives_k: np.ndarray,
    h_derivatives_kq: np.ndarray,
    phonon_eigenvectors: np.ndarray,
    masses: np.ndarray,
    overlap_derivatives_k: Optional[np.ndarray] = None,
    overlap_derivatives_kq: Optional[np.ndarray] = None,
    frequencies: Optional[np.ndarray] = None,
    band_indices: Optional[Sequence[int]] = None,
    qpoints: Optional[np.ndarray] = None,
    scaled_positions: Optional[np.ndarray] = None,
    orbital_slices: Optional[Sequence[Tuple[int, int]]] = None,
    derivative_mode: str = "full",
    prefactor_amu_thz: float = EPC_PREFAC_AMU_THZ,
    omega_floor: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute electron-phonon coupling matrices for fixed k and q points."""
    eigenvalues_k = as_array(eigenvalues_k, dtype=float)
    eigenvectors_k = as_array(eigenvectors_k, dtype=complex)
    eigenvalues_kq = as_array(eigenvalues_kq, dtype=float)
    eigenvectors_kq = as_array(eigenvectors_kq, dtype=complex)
    h_derivatives_k = as_array(h_derivatives_k, dtype=complex)
    h_derivatives_kq = as_array(h_derivatives_kq, dtype=complex)
    phonon_eigenvectors = as_array(phonon_eigenvectors, dtype=complex)
    masses = np.atleast_1d(as_array(masses, dtype=float))
    if derivative_mode not in {"full", "row"}:
        raise ValueError("derivative_mode must be either 'full' or 'row'.")
    prefactor_amu_thz = validate_finite_positive_scalar(prefactor_amu_thz, "prefactor_amu_thz")
    omega_floor = validate_finite_positive_scalar(omega_floor, "omega_floor")
    if eigenvalues_k.ndim != 2:
        raise ValueError("eigenvalues_k must have shape (nk, nbands).")
    if eigenvalues_kq.ndim != 3:
        raise ValueError("eigenvalues_kq must have shape (nq, nk, nbands).")
    if eigenvectors_k.ndim != 3:
        raise ValueError("eigenvectors_k must have shape (nk, norb, nbands).")
    if eigenvectors_kq.ndim != 4:
        raise ValueError("eigenvectors_kq must have shape (nq, nk, norb, nbands).")
    if phonon_eigenvectors.ndim != 4 or phonon_eigenvectors.shape[-1] != 3:
        raise ValueError("phonon_eigenvectors must have shape (nq, nmodes, natoms, 3).")
    if not np.all(np.isfinite(eigenvalues_k)):
        raise ValueError("eigenvalues_k must be finite.")
    if not np.all(np.isfinite(eigenvalues_kq)):
        raise ValueError("eigenvalues_kq must be finite.")
    if not np.all(np.isfinite(eigenvectors_k)):
        raise ValueError("eigenvectors_k must be finite.")
    if not np.all(np.isfinite(eigenvectors_kq)):
        raise ValueError("eigenvectors_kq must be finite.")
    if not np.all(np.isfinite(h_derivatives_k)):
        raise ValueError("h_derivatives_k must be finite.")
    if not np.all(np.isfinite(h_derivatives_kq)):
        raise ValueError("h_derivatives_kq must be finite.")
    if not np.all(np.isfinite(phonon_eigenvectors)):
        raise ValueError("phonon_eigenvectors must be finite.")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("masses must be finite and positive.")

    if overlap_derivatives_k is None:
        overlap_derivatives_k = np.zeros_like(h_derivatives_k)
    else:
        overlap_derivatives_k = as_array(overlap_derivatives_k, dtype=complex)
    if overlap_derivatives_kq is None:
        overlap_derivatives_kq = np.zeros_like(h_derivatives_kq)
    else:
        overlap_derivatives_kq = as_array(overlap_derivatives_kq, dtype=complex)
    if overlap_derivatives_k.shape != h_derivatives_k.shape:
        raise ValueError("overlap_derivatives_k must have the same shape as h_derivatives_k.")
    if overlap_derivatives_kq.shape != h_derivatives_kq.shape:
        raise ValueError("overlap_derivatives_kq must have the same shape as h_derivatives_kq.")
    if not np.all(np.isfinite(overlap_derivatives_k)):
        raise ValueError("overlap_derivatives_k must be finite.")
    if not np.all(np.isfinite(overlap_derivatives_kq)):
        raise ValueError("overlap_derivatives_kq must be finite.")

    nk, nbands = eigenvalues_k.shape
    nq, _, nbands_kq = eigenvalues_kq.shape
    nmodes = phonon_eigenvectors.shape[1]
    natoms = masses.shape[0]
    norb = eigenvectors_k.shape[1]
    if nmodes == 0:
        raise ValueError("phonon_eigenvectors must contain at least one phonon mode.")
    if eigenvalues_kq.shape[1] != nk:
        raise ValueError("eigenvalues_kq must agree with eigenvalues_k on nk.")
    if nbands_kq != nbands:
        raise ValueError("eigenvalues_k and eigenvalues_kq must have the same band count.")
    if eigenvectors_k.shape != (nk, norb, nbands):
        raise ValueError("eigenvectors_k must have shape (nk, norb, nbands).")
    if eigenvectors_kq.shape != (nq, nk, norb, nbands):
        raise ValueError("eigenvectors_kq must have shape (nq, nk, norb, nbands).")
    if phonon_eigenvectors.shape[0] != nq:
        raise ValueError("phonon_eigenvectors must agree with eigenvalues_kq on nq.")
    if phonon_eigenvectors.shape[2] != natoms:
        raise ValueError("phonon_eigenvectors natoms dimension must match masses.")
    if derivative_mode == "full":
        if h_derivatives_k.shape != (nk, natoms, 3, norb, norb):
            raise ValueError("full h_derivatives_k must have shape (nk, natoms, 3, norb, norb).")
        if h_derivatives_kq.shape != (nq, nk, natoms, 3, norb, norb):
            raise ValueError("full h_derivatives_kq must have shape (nq, nk, natoms, 3, norb, norb).")
    else:
        if h_derivatives_k.shape != (nk, 3, norb, norb):
            raise ValueError("row h_derivatives_k must have shape (nk, 3, norb, norb).")
        if h_derivatives_kq.shape != (nq, nk, 3, norb, norb):
            raise ValueError("row h_derivatives_kq must have shape (nq, nk, 3, norb, norb).")

    if band_indices is None:
        band_indices = np.arange(nbands, dtype=int)
    else:
        band_indices = normalize_integer_indices(band_indices, "band_indices")
    if np.any(band_indices < 0) or np.any(band_indices >= nbands):
        raise ValueError("band_indices contains an index outside the available band range.")

    nsel = len(band_indices)
    coupling_matrix = np.zeros((nq, nk, nmodes, nsel, nsel), dtype=complex)

    use_block_phase = orbital_slices is not None or scaled_positions is not None or qpoints is not None
    if use_block_phase and (orbital_slices is None or scaled_positions is None or qpoints is None):
        raise ValueError("orbital_slices, scaled_positions and qpoints must be provided together.")
    if derivative_mode == "row" and not use_block_phase:
        raise ValueError("derivative_mode='row' requires orbital_slices, scaled_positions and qpoints.")

    inv_sqrt_mass = 1.0 / np.sqrt(masses)
    if not use_block_phase:
        mode_weights = phonon_eigenvectors * inv_sqrt_mass[None, None, :, None]
    else:
        qpoints = normalize_kpoints(qpoints)
        scaled_positions = np.array(scaled_positions, dtype=float, copy=True)
        if scaled_positions.shape != (len(masses), 3):
            raise ValueError("scaled_positions must have shape (natoms, 3).")
        scaled_positions = scaled_positions - scaled_positions[0]
        orbital_slices = normalize_orbital_slices(orbital_slices)
        if len(orbital_slices) != len(masses):
            raise ValueError("orbital_slices length must match masses.")

    for iq in range(nq):
        for ik in range(nk):
            states_k = np.take(eigenvectors_k[ik], band_indices, axis=-1)
            states_kq = np.take(eigenvectors_kq[iq, ik], band_indices, axis=-1)
            states_kq_dagger = states_kq.conj().T
            band_energies_k = eigenvalues_k[ik, band_indices]
            band_energies_kq = eigenvalues_kq[iq, ik, band_indices]

            for imode in range(nmodes):
                if use_block_phase:
                    (
                        weighted_hamiltonian,
                        weighted_overlap_k,
                        weighted_overlap_kq,
                    ) = _assemble_block_phase_derivatives(
                        phonon_eigenvectors=phonon_eigenvectors[iq, imode],
                        masses=masses,
                        qpoint=qpoints[iq],
                        scaled_positions=scaled_positions,
                        orbital_slices=orbital_slices,
                        h_derivatives_k=h_derivatives_k[ik],
                        h_derivatives_kq=h_derivatives_kq[iq, ik],
                        overlap_derivatives_k=overlap_derivatives_k[ik],
                        overlap_derivatives_kq=overlap_derivatives_kq[iq, ik],
                        derivative_mode=derivative_mode,
                        norb=eigenvectors_k.shape[-2],
                    )
                else:
                    mode_weight = mode_weights[iq, imode]
                    weighted_hamiltonian = np.einsum("sa,samn->mn", mode_weight, h_derivatives_k[ik])
                    weighted_hamiltonian -= np.einsum(
                        "sa,samn->mn", mode_weight, h_derivatives_kq[iq, ik]
                    )
                    weighted_overlap_k = np.einsum(
                        "sa,samn->mn", mode_weight, overlap_derivatives_k[ik]
                    )
                    weighted_overlap_kq = np.einsum(
                        "sa,samn->mn", mode_weight, overlap_derivatives_kq[iq, ik]
                    )

                band_hamiltonian = states_kq_dagger @ weighted_hamiltonian @ states_k
                band_overlap_k = states_kq_dagger @ weighted_overlap_k @ states_k
                band_overlap_kq = states_kq_dagger @ weighted_overlap_kq @ states_k
                coupling_matrix[iq, ik, imode] = (
                    band_hamiltonian
                    - (
                        band_overlap_k * band_energies_k[None, :]
                        - band_energies_kq[:, None] * band_overlap_kq
                    )
                )

    if frequencies is None:
        coupling_strength = np.abs(coupling_matrix) ** 2
    else:
        frequencies = as_array(frequencies, dtype=float)
        if frequencies.shape != (nq, nmodes):
            raise ValueError("frequencies must have shape (nq, nmodes).")
        if not np.all(np.isfinite(frequencies)):
            raise ValueError("frequencies must be finite.")
        if np.any(frequencies < 0.0):
            raise ValueError("frequencies must be non-negative; imaginary modes are not supported in EPC v1.")
        regularized_frequencies = np.maximum(frequencies, omega_floor)
        prefactor = prefactor_amu_thz / (2.0 * regularized_frequencies)
        coupling_matrix = coupling_matrix * np.sqrt(prefactor)[:, None, :, None, None]
        coupling_strength = np.abs(coupling_matrix) ** 2

    return coupling_matrix, coupling_strength


def _assemble_block_phase_derivatives(
    phonon_eigenvectors,
    masses,
    qpoint,
    scaled_positions,
    orbital_slices,
    h_derivatives_k,
    h_derivatives_kq,
    overlap_derivatives_k,
    overlap_derivatives_kq,
    derivative_mode,
    norb,
):
    phase = np.exp(2j * np.pi * (scaled_positions @ qpoint)) / np.sqrt(masses)
    weighted_hamiltonian = np.zeros((norb, norb), dtype=complex)
    weighted_overlap_k = np.zeros((norb, norb), dtype=complex)
    weighted_overlap_kq = np.zeros((norb, norb), dtype=complex)

    for s, sl_s in enumerate(orbital_slices):
        for sp, sl_sp in enumerate(orbital_slices):
            for alpha in range(3):
                weight_s = phase[s] * phonon_eigenvectors[s, alpha]
                weight_sp = phase[sp] * phonon_eigenvectors[sp, alpha]
                if derivative_mode == "row":
                    h_block_k = h_derivatives_k[alpha, sl_s, sl_sp]
                    h_block_kq = h_derivatives_kq[alpha, sl_s, sl_sp]
                    overlap_block_k = overlap_derivatives_k[alpha, sl_s, sl_sp]
                    overlap_block_kq = overlap_derivatives_kq[alpha, sl_s, sl_sp]
                else:
                    h_block_k = h_derivatives_k[s, alpha, sl_s, sl_sp]
                    h_block_kq = h_derivatives_kq[sp, alpha, sl_s, sl_sp]
                    overlap_block_k = overlap_derivatives_k[s, alpha, sl_s, sl_sp]
                    overlap_block_kq = overlap_derivatives_kq[sp, alpha, sl_s, sl_sp]
                weighted_hamiltonian[sl_s, sl_sp] += weight_s * h_block_k - weight_sp * h_block_kq
                weighted_overlap_k[sl_s, sl_sp] += weight_s * overlap_block_k
                weighted_overlap_kq[sl_s, sl_sp] += weight_sp * overlap_block_kq

    return weighted_hamiltonian, weighted_overlap_k, weighted_overlap_kq
