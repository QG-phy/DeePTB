import sys
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from scipy import linalg
from scipy import constants as scipy_constants

import dptb.postprocess.unified.eph.providers as eph_providers
from dptb.postprocess.unified.eph import (
    EPCData,
    EPC_MESH_CHUNKED_ARTIFACT_SCHEMA_VERSION,
    EPCMeshData,
    EPCMeshSpec,
    EPCPathData,
    EPC_MESH_NPZ_SCHEMA_VERSION,
    EPC_NPZ_SCHEMA_VERSION,
    EPC_PATH_NPZ_SCHEMA_VERSION,
    EPhAccessor,
    HBAR_EV_S,
    LINEWIDTH_MESH_NPZ_SCHEMA_VERSION,
    LINEWIDTH_NPZ_SCHEMA_VERSION,
    LINEWIDTH_PATH_NPZ_SCHEMA_VERSION,
    LinewidthData,
    LinewidthMeshData,
    LinewidthPathData,
    MOBILITY_NPZ_SCHEMA_VERSION,
    MOBILITY_SCAN_NPZ_SCHEMA_VERSION,
    MobilityData,
    MobilityScanData,
    RELAXATION_TIME_MESH_NPZ_SCHEMA_VERSION,
    RELAXATION_TIME_NPZ_SCHEMA_VERSION,
    RELAXATION_TIME_PATH_NPZ_SCHEMA_VERSION,
    RelaxationTimeData,
    RelaxationTimeMeshData,
    RelaxationTimePathData,
    SUBSPACE_COUPLING_NPZ_SCHEMA_VERSION,
    SubspaceCouplingData,
    THZ_TO_EV,
    PHONON_NPZ_SCHEMA_VERSION,
    Phonons,
    SupercellFD,
    TRANSPORT_NPZ_SCHEMA_VERSION,
    TRANSPORT_SCAN_NPZ_SCHEMA_VERSION,
    TransportData,
    TransportScanData,
    EPCKChunkSpec,
    EPCQChunkSpec,
    build_k_chunk_specs,
    build_q_chunk_specs,
    compute_band_velocities_finite_difference,
    compute_band_velocities_hamiltonian_derivative,
    fractional_band_velocities_to_si,
    compute_eliashberg_spectral_function,
    compute_serta_mobility_si,
    compute_serta_mobility_scan_si,
    compute_linewidth,
    compute_linewidth_mesh_chunked_artifact,
    compute_linewidth_mesh,
    compute_linewidth_path,
    compute_phonon_dos,
    compute_relaxation_time,
    compute_relaxation_time_mesh,
    compute_relaxation_time_path,
    compute_scattering_maps,
    compute_serta_conductivity,
    compute_serta_transport_scan,
    compute_serta_mobility_si_from_epc_mesh_chunked_artifact,
    compute_serta_transport_scan_from_epc_mesh_chunked_artifact,
    compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact,
    compute_serta_transport_from_epc_mesh_chunked_artifact,
    compute_serta_transport_from_epc,
    compute_coupling_strength_summary,
    compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact,
    compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact,
    compute_subspace_coupling_data,
    compute_subspace_coupling_strength,
    concat_epc_k_chunks,
    concat_epc_q_chunks,
    cumulative_path_coordinates,
    find_degenerate_band_groups,
    load_epc_mesh_chunked_artifact,
    save_epc_mesh_chunked_artifact,
)
from dptb.postprocess.unified.eph.contraction import EPC_PREFAC_AMU_THZ, compute_coupling_matrix
from dptb.postprocess.unified.eph.utils import (
    assemble_directed_hk_from_blocks,
    orbital_slices_from_atom_orbs,
    orbital_slices_from_system,
    reshape_phonopy_eigenvectors,
)
from dptb.utils import constants as dptb_constants

MINIMAL_EPC_FIXTURE = Path(__file__).parent / "fixtures" / "eph" / "minimal_epc_reference.json"

def _minimal_fixture_epc_data() -> EPCData:
    with open(MINIMAL_EPC_FIXTURE, "r", encoding="utf-8") as handle:
        fixture = json.load(handle)

    epc_payload = fixture["epc_data"]
    coupling_matrix = np.asarray(epc_payload["coupling_matrix_real"], dtype=float) + 1j * np.asarray(
        epc_payload["coupling_matrix_imag"],
        dtype=float,
    )
    return EPCData(
        kpoints=np.asarray(epc_payload["kpoints"], dtype=float),
        qpoints=np.asarray(epc_payload["qpoints"], dtype=float),
        band_indices=np.asarray(epc_payload["band_indices"], dtype=int),
        frequencies=np.asarray(epc_payload["frequencies"], dtype=float),
        eigenvalues_k=np.asarray(epc_payload["eigenvalues_k"], dtype=float),
        eigenvalues_kq=np.asarray(epc_payload["eigenvalues_kq"], dtype=float),
        coupling_matrix=coupling_matrix,
        coupling_strength=np.asarray(epc_payload["coupling_strength"], dtype=float),
        metadata={"source": "minimal_in_repo_fixture"},
    )

def _minimal_supercell_fd_kwargs():
    return {
        "system": object(),
        "supercell_atoms": Atoms(symbols=["C", "C"], positions=np.zeros((2, 3))),
        "primitive_to_supercell_atom": np.array([0]),
        "supercell_to_primitive_atom": np.array([0, 0]),
        "supercell_atom_to_cell": np.array([0, 1]),
        "primitive_orbital_offsets": np.array([0, 1]),
        "supercell_orbital_offsets": np.array([0, 1, 2]),
        "shortest_vectors": np.zeros((2, 1, 1, 3)),
        "vector_multiplicity": np.ones((2, 1), dtype=int),
    }

def _chunk_artifact_mesh_data():
    strength = np.arange(1, 1 + 2 * 3 * 2 * 2 * 2, dtype=float).reshape(2, 3, 2, 2, 2)
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        band_indices=np.array([0, 1]),
        frequencies=np.array([[1.0, 2.0], [3.0, 4.0]]),
        eigenvalues_k=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        eigenvalues_kq=np.ones((2, 3, 2)),
        coupling_matrix=np.sqrt(strength).astype(complex),
        coupling_strength=strength,
        metadata={"source": "chunk-artifact-test"},
    )
    return EPCMeshData.from_epc_data(
        epc_data,
        kpoint_weights=np.array([1.0, 2.0, 3.0]),
        qpoint_weights=np.array([3.0, 1.0]),
        metadata={"mesh_spec": {"k_mesh": [3, 1, 1], "q_mesh": [2, 1, 1]}},
    )

def _small_linewidth_epc_data():
    return EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        band_indices=np.array([0, 1]),
        frequencies=np.array([[1.0, 2.0]]),
        eigenvalues_k=np.array([[0.10, 0.20]]),
        eigenvalues_kq=np.array([[[0.11, 0.19]]]),
        coupling_matrix=np.ones((1, 1, 2, 2, 2), dtype=complex),
        coupling_strength=np.array(
            [[[[[0.10, 0.20], [0.30, 0.40]], [[0.50, 0.60], [0.70, 0.80]]]]],
            dtype=float,
        ),
    )

def _small_linewidth_epc_path_data():
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        band_indices=np.array([0, 1]),
        frequencies=np.array([[1.0, 2.0], [1.5, 2.5]]),
        eigenvalues_k=np.array([[0.10, 0.20]]),
        eigenvalues_kq=np.array([[[0.11, 0.19]], [[0.12, 0.21]]]),
        coupling_matrix=np.ones((2, 1, 2, 2, 2), dtype=complex),
        coupling_strength=np.array(
            [
                [[[[0.10, 0.20], [0.30, 0.40]], [[0.50, 0.60], [0.70, 0.80]]]],
                [[[[0.15, 0.25], [0.35, 0.45]], [[0.55, 0.65], [0.75, 0.85]]]],
            ],
            dtype=float,
        ),
    )
    return EPCPathData.from_epc_data(
        epc_data,
        path_axis="q",
        path_coordinates=np.array([0.0, 0.25]),
        path_segments=np.array([[0, 2]]),
        metadata={"path_mode": "fixed_k_q_path"},
    )

def _small_linewidth_epc_mesh_data(qpoint_weights=None):
    epc_data = _small_linewidth_epc_data()
    return EPCMeshData.from_epc_data(
        epc_data,
        kpoint_weights=np.array([1.0]),
        qpoint_weights=np.array([1.0]) if qpoint_weights is None else qpoint_weights,
        metadata={"mesh_spec": {"k_mesh": [1, 1, 1]}},
    )

def _manual_linewidth(epc_data, chemical_potential, temperature, sigma, broadening, mode_resolved):
    frequencies_ev = epc_data.frequencies * THZ_TO_EV
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
                    bose = 1.0 / np.expm1(omega / temperature)
                    for final_band in range(nsel):
                        eps_final = epc_data.eigenvalues_kq[iq, ik, final_band]
                        fermi = 1.0 / (np.exp((eps_final - chemical_potential) / temperature) + 1.0)
                        if broadening == "gaussian":
                            absorption_weight = np.exp(
                                -((eps_initial + omega - eps_final) ** 2) / (2.0 * sigma**2)
                            ) / (np.sqrt(2.0 * np.pi) * sigma)
                            emission_weight = np.exp(
                                -((eps_initial - omega - eps_final) ** 2) / (2.0 * sigma**2)
                            ) / (np.sqrt(2.0 * np.pi) * sigma)
                        else:
                            absorption_weight = (
                                (sigma / 2.0)
                                / ((eps_initial + omega - eps_final) ** 2 + (sigma / 2.0) ** 2)
                                / np.pi
                            )
                            emission_weight = (
                                (sigma / 2.0)
                                / ((eps_initial - omega - eps_final) ** 2 + (sigma / 2.0) ** 2)
                                / np.pi
                            )
                        strength = epc_data.coupling_strength[iq, ik, mode, final_band, initial_band]
                        absorption_term = strength * (bose + fermi) * absorption_weight
                        emission_term = strength * (bose + 1.0 - fermi) * emission_weight
                        if mode_resolved:
                            absorption[ik, initial_band, mode] += absorption_term
                            emission[ik, initial_band, mode] += emission_term
                        else:
                            absorption[ik, initial_band] += absorption_term
                            emission[ik, initial_band] += emission_term
    return 2.0 * np.pi * (absorption + emission), 2.0 * np.pi * absorption, 2.0 * np.pi * emission

def _manual_serta_conductivity(
    eigenvalues,
    velocities,
    linewidth,
    chemical_potential,
    temperature,
    kpoint_weights,
    spin_degeneracy,
    volume,
):
    weights = np.asarray(kpoint_weights, dtype=float)
    weights = weights / weights.sum()
    conductivity = np.zeros((3, 3), dtype=float)
    carrier_density = 0.0
    for ik in range(eigenvalues.shape[0]):
        for iband in range(eigenvalues.shape[1]):
            x = (eigenvalues[ik, iband] - chemical_potential) / temperature
            fermi = 1.0 / (np.exp(x) + 1.0)
            dfermi = -np.exp(x) / (np.exp(x) + 1.0) ** 2
            weight = spin_degeneracy * weights[ik] / volume
            carrier_density += weight * fermi
            conductivity += (
                weight
                * (-dfermi / temperature)
                * np.outer(velocities[ik, iband], velocities[ik, iband])
                / linewidth[ik, iband]
            )
    return conductivity, carrier_density

class _LinearBandSystem:
    band_offsets = np.array([0.1, 0.2, 0.3])
    band_slopes = np.array(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 0.0],
            [0.0, -2.0, 1.5],
        ]
    )

    def get_eigenvalues(self, k_points, use_scc=False, **kwargs):
        if use_scc:
            raise AssertionError("test system should not be called with SCC")
        kpoints = np.asarray(k_points, dtype=float)
        return {}, self.band_offsets[None, :] + kpoints @ self.band_slopes.T

class _LinearBandSystemWithAtoms(_LinearBandSystem):
    atoms = Atoms("C", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)

class _NonfiniteBandSystem:
    def get_eigenvalues(self, k_points, use_scc=False, **kwargs):
        return {}, np.full((len(k_points), 2), np.nan)

class _ShapeChangingBandSystem:
    def get_eigenvalues(self, k_points, use_scc=False, **kwargs):
        kpoints = np.asarray(k_points, dtype=float)
        nbands = 3 if np.allclose(kpoints, 0.0) else 2
        return {}, np.zeros((kpoints.shape[0], nbands))

class _EmptyBandSystem:
    def get_eigenvalues(self, k_points, use_scc=False, **kwargs):
        return {}, np.empty((len(k_points), 0))

class _DerivativeBandSystem(_LinearBandSystem):
    def get_hk(self, k_points, use_scc=False, with_derivative=False, **kwargs):
        if use_scc:
            raise AssertionError("test system should not be called with SCC")
        if not with_derivative:
            raise AssertionError("test requires with_derivative=True")
        kpoints = np.asarray(k_points, dtype=float)
        eigenvalues = self.band_offsets[None, :] + kpoints @ self.band_slopes.T
        nk, nbands = eigenvalues.shape
        hk = np.zeros((nk, nbands, nbands), dtype=complex)
        dhdk = np.zeros((nk, nbands, nbands, 3), dtype=complex)
        for ik in range(nk):
            hk[ik] = np.diag(eigenvalues[ik])
            for axis in range(3):
                dhdk[ik, :, :, axis] = np.diag(self.band_slopes[:, axis])
        return hk, dhdk, None, None

class _OverlapDerivativeBandSystem:
    def get_hk(self, k_points, use_scc=False, with_derivative=False, **kwargs):
        if not with_derivative:
            raise AssertionError("test requires with_derivative=True")
        kpoints = np.asarray(k_points, dtype=float)
        nk = kpoints.shape[0]
        hk = np.zeros((nk, 1, 1), dtype=complex)
        dhdk = np.zeros((nk, 1, 1, 3), dtype=complex)
        sk = np.ones((nk, 1, 1), dtype=complex) * 2.0
        dsdk = np.zeros((nk, 1, 1, 3), dtype=complex)
        hk[:, 0, 0] = 2.0 + 6.0 * kpoints[:, 0]
        dhdk[:, 0, 0, 0] = 6.0
        dsdk[:, 0, 0, 0] = 1.0
        return hk, dhdk, sk, dsdk

class _FakeDerivativeProvider:
    def compute(self, kpoints):
        nk = len(kpoints)
        h_derivatives = np.zeros((nk, 1, 3, 2, 2), dtype=complex)
        h_derivatives[:, 0, 0] = np.eye(2)
        return h_derivatives, None

class _NonfiniteDerivativeProvider:
    def compute(self, kpoints):
        nk = len(kpoints)
        h_derivatives = np.zeros((nk, 1, 3, 2, 2), dtype=complex)
        h_derivatives[0, 0, 0, 0, 0] = np.nan
        return h_derivatives, None

class _ShapeChangingDerivativeProvider:
    def compute(self, kpoints):
        nk = len(kpoints)
        natoms = 1 if nk == 1 else 2
        h_derivatives = np.zeros((nk, natoms, 3, 2, 2), dtype=complex)
        return h_derivatives, None

class _BadOverlapDerivativeProvider:
    def compute(self, kpoints):
        nk = len(kpoints)
        h_derivatives = np.zeros((nk, 1, 3, 2, 2), dtype=complex)
        overlap_derivatives = np.zeros((nk, 3, 2, 2), dtype=complex)
        return h_derivatives, overlap_derivatives

class _BadPayloadDerivativeProvider:
    def compute(self, kpoints):
        nk = len(kpoints)
        return np.zeros((nk, 1, 3, 2, 2), dtype=complex)

class _FakeSystem:
    atoms = None

    class calculator:
        @staticmethod
        def get_orbital_info():
            return {"C": ["2s", "2p_y", "2p_z", "2p_x"]}

    def get_eigenstates(self, k_points, use_scc=False):
        nk = len(k_points)
        eigenvalues = np.tile(np.array([[1.0, 2.0]]), (nk, 1))
        eigenvectors = np.tile(np.eye(2, dtype=complex)[None, :, :], (nk, 1, 1))
        return {}, eigenvalues, eigenvectors

    @property
    def eph(self):
        return EPhAccessor(self)

class _BadEigenstatePayloadSystem(_FakeSystem):
    def get_eigenstates(self, k_points, use_scc=False):
        nk = len(k_points)
        return np.zeros((nk, 2), dtype=float), np.zeros((nk, 2, 2), dtype=complex)

class _BadEigenstateArraySystem(_FakeSystem):
    def __init__(self, *, eigenvalues=None, eigenvectors=None):
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors

    def get_eigenstates(self, k_points, use_scc=False):
        nk = len(k_points)
        eigenvalues = self._eigenvalues
        eigenvectors = self._eigenvectors
        if eigenvalues is None:
            eigenvalues = np.tile(np.array([[1.0, 2.0]]), (nk, 1))
        if eigenvectors is None:
            eigenvectors = np.tile(np.eye(2, dtype=complex)[None, :, :], (nk, 1, 1))
        return {}, eigenvalues, eigenvectors

class _KqShapeChangingEigenstateSystem(_FakeSystem):
    def __init__(self, *, kq_norb=2, kq_nbands=2):
        self._calls = 0
        self._kq_norb = kq_norb
        self._kq_nbands = kq_nbands

    def get_eigenstates(self, k_points, use_scc=False):
        self._calls += 1
        nk = len(k_points)
        if self._calls == 1:
            eigenvalues = np.tile(np.array([[1.0, 2.0]]), (nk, 1))
            eigenvectors = np.tile(np.eye(2, dtype=complex)[None, :, :], (nk, 1, 1))
        else:
            eigenvalues = np.tile(np.arange(self._kq_nbands, dtype=float).reshape(1, -1), (nk, 1))
            eigenvectors = np.zeros((nk, self._kq_norb, self._kq_nbands), dtype=complex)
        return {}, eigenvalues, eigenvectors

def _single_mode_phonons():
    return Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        frequencies=np.array([[1.0]]),
        eigenvectors=np.array([[[[1.0, 0.0, 0.0]]]], dtype=complex),
        masses=np.array([1.0]),
    )

def _epc_k_chunk(kpoint_values, *, qpoint_shift=0.0, band_indices=None, frequencies=None, nbands=1):
    kpoint_values = np.asarray(kpoint_values, dtype=float)
    nk = len(kpoint_values)
    if band_indices is None:
        band_indices = np.arange(nbands)
    if frequencies is None:
        frequencies = np.ones((1, 1))
    band_indices = np.asarray(band_indices, dtype=int)
    nbands = len(band_indices)
    values = kpoint_values.reshape(nk, 1) + np.arange(nbands, dtype=float).reshape(1, nbands)
    coupling_strength = np.broadcast_to(values.reshape(1, nk, 1, nbands, 1), (1, nk, 1, nbands, nbands))
    return EPCData(
        kpoints=np.column_stack([kpoint_values, np.zeros((nk, 2))]),
        qpoints=np.array([[qpoint_shift, 0.0, 0.0]]),
        band_indices=band_indices,
        frequencies=np.asarray(frequencies, dtype=float),
        eigenvalues_k=values,
        eigenvalues_kq=values.reshape(1, nk, nbands),
        coupling_matrix=coupling_strength.astype(complex),
        coupling_strength=coupling_strength,
    )

def _epc_q_chunk(qpoint_values, *, kpoint_shift=0.0, band_indices=None, eigenvalues_k=None, nbands=1):
    qpoint_values = np.asarray(qpoint_values, dtype=float)
    nq = len(qpoint_values)
    if band_indices is None:
        band_indices = np.arange(nbands)
    band_indices = np.asarray(band_indices, dtype=int)
    nbands = len(band_indices)
    if eigenvalues_k is None:
        eigenvalues_k = np.arange(nbands, dtype=float).reshape(1, nbands)
    eigenvalues_k = np.asarray(eigenvalues_k, dtype=float)
    values = qpoint_values.reshape(nq, 1, 1) + eigenvalues_k.reshape(1, 1, nbands)
    coupling_strength = np.broadcast_to(values.reshape(nq, 1, 1, nbands, 1), (nq, 1, 1, nbands, nbands))
    return EPCData(
        kpoints=np.array([[kpoint_shift, 0.0, 0.0]]),
        qpoints=np.column_stack([qpoint_values, np.zeros((nq, 2))]),
        band_indices=band_indices,
        frequencies=qpoint_values.reshape(nq, 1) + 1.0,
        eigenvalues_k=eigenvalues_k,
        eigenvalues_kq=values,
        coupling_matrix=coupling_strength.astype(complex),
        coupling_strength=coupling_strength,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
