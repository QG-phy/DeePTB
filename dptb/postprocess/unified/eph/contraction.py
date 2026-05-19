from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import constants as scipy_constants

from dptb.postprocess.unified.eph.utils import (
    as_array,
    normalize_kpoints,
    normalize_orbital_slices,
)


# The phonon-mode coupling uses sqrt(hbar / (2 M omega)) to convert Cartesian
# derivatives into normal-mode derivatives. With masses in amu, frequencies in
# THz and derivatives in eV/Angstrom, the unit factor is
# hbar / (amu * THz * Angstrom^2).
EPC_PREFAC_AMU_THZ = (
    scipy_constants.hbar
    / scipy_constants.physical_constants["atomic mass constant"][0]
    / scipy_constants.tera
    / scipy_constants.angstrom**2
)


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
    masses = as_array(masses, dtype=float)

    if overlap_derivatives_k is None:
        overlap_derivatives_k = np.zeros_like(h_derivatives_k)
    else:
        overlap_derivatives_k = as_array(overlap_derivatives_k, dtype=complex)
    if overlap_derivatives_kq is None:
        overlap_derivatives_kq = np.zeros_like(h_derivatives_kq)
    else:
        overlap_derivatives_kq = as_array(overlap_derivatives_kq, dtype=complex)

    nk, nbands = eigenvalues_k.shape
    nq, _, nbands_kq = eigenvalues_kq.shape
    if nbands_kq != nbands:
        raise ValueError("eigenvalues_k and eigenvalues_kq must have the same band count.")

    if band_indices is None:
        band_indices = np.arange(nbands, dtype=int)
    else:
        band_indices = np.asarray(band_indices, dtype=int)

    nmodes = phonon_eigenvectors.shape[1]
    nsel = len(band_indices)
    coupling_matrix = np.zeros((nq, nk, nmodes, nsel, nsel), dtype=complex)

    use_block_phase = orbital_slices is not None or scaled_positions is not None or qpoints is not None
    if use_block_phase and (orbital_slices is None or scaled_positions is None or qpoints is None):
        raise ValueError("orbital_slices, scaled_positions and qpoints must be provided together.")
    if derivative_mode not in {"full", "row"}:
        raise ValueError("derivative_mode must be either 'full' or 'row'.")

    inv_sqrt_mass = 1.0 / np.sqrt(masses)
    if not use_block_phase:
        mode_weights = phonon_eigenvectors * inv_sqrt_mass[None, None, :, None]
    else:
        qpoints = normalize_kpoints(qpoints)
        scaled_positions = as_array(scaled_positions, dtype=float)
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
        prefactor = prefactor_amu_thz / np.maximum(2.0 * frequencies, omega_floor)
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
