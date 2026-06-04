import sys
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase import Atoms
from scipy import linalg
from scipy import constants as scipy_constants

import dptb.postprocess.unified.eph.providers as eph_providers
from dptb.postprocess.unified.eph.providers import _length_unit_scale_to_angstrom
from dptb.postprocess import unified as unified_postprocess
from dptb.entrypoints.eph import _load_array, _load_kpoint_weights, _load_kpoints, _parse_band_groups, eph
from dptb.entrypoints.main import parse_args
from dptb.postprocess.unified.eph import (
    EPCData,
    EPCMeshData,
    EPCMeshSpec,
    EPCPathData,
    EPC_MESH_NPZ_SCHEMA_VERSION,
    EPC_NPZ_SCHEMA_VERSION,
    EPC_PATH_NPZ_SCHEMA_VERSION,
    DFTBPlusGauge,
    EPhAccessor,
    FDProvider,
    HBAR_EV_S,
    LINEWIDTH_MESH_NPZ_SCHEMA_VERSION,
    LINEWIDTH_NPZ_SCHEMA_VERSION,
    LINEWIDTH_PATH_NPZ_SCHEMA_VERSION,
    LinewidthData,
    LinewidthMeshData,
    LinewidthPathData,
    MOBILITY_NPZ_SCHEMA_VERSION,
    MobilityData,
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
    TransportData,
    EPCKChunkSpec,
    build_k_chunk_specs,
    compute_band_velocities_finite_difference,
    compute_band_velocities_hamiltonian_derivative,
    compute_serta_mobility_si,
    compute_linewidth,
    compute_linewidth_mesh,
    compute_linewidth_path,
    compute_relaxation_time,
    compute_relaxation_time_mesh,
    compute_relaxation_time_path,
    compute_serta_conductivity,
    compute_serta_transport_from_epc,
    compute_subspace_coupling_data,
    compute_subspace_coupling_strength,
    concat_epc_k_chunks,
    cumulative_path_coordinates,
    find_degenerate_band_groups,
)
from dptb.postprocess.unified.eph.contraction import EPC_PREFAC_AMU_THZ, compute_coupling_matrix
from dptb.postprocess.unified.eph.utils import (
    assemble_directed_hk_from_blocks,
    normalize_orbital_slices,
    orbital_slices_from_atom_orbs,
    orbital_slices_from_system,
    reshape_phonopy_eigenvectors,
)
from dptb.utils import constants as dptb_constants


DEFAULT_EPH_REFERENCE_ROOT = Path("/Users/aisiqg/Desktop/work/github/dftbephy")
DEFAULT_EPH_SKDATA_ROOT = Path("/Users/aisiqg/Desktop/work/github/matsci-0-3")


def _external_reference_root() -> Path:
    return Path(os.environ.get("DEEPTB_EPH_REFERENCE_ROOT", DEFAULT_EPH_REFERENCE_ROOT))


def _external_skdata_root() -> Path:
    return Path(os.environ.get("DEEPTB_EPH_SKDATA_ROOT", DEFAULT_EPH_SKDATA_ROOT))


def test_epc_prefactor_from_standard_constants():
    expected = (
        scipy_constants.hbar
        / scipy_constants.physical_constants["atomic mass constant"][0]
        / scipy_constants.tera
        / scipy_constants.angstrom**2
    )
    np.testing.assert_allclose(EPC_PREFAC_AMU_THZ, expected, rtol=1e-15)
    np.testing.assert_allclose(EPC_PREFAC_AMU_THZ, 6.35078, rtol=1e-6)


def test_epc_public_constants_are_centralized():
    np.testing.assert_allclose(EPC_PREFAC_AMU_THZ, dptb_constants.EPC_PREFAC_AMU_THZ)
    np.testing.assert_allclose(THZ_TO_EV, dptb_constants.THZ_TO_EV)
    np.testing.assert_allclose(HBAR_EV_S, dptb_constants.HBAR_EV_S)
    np.testing.assert_allclose(dptb_constants.ANGSTROM_TO_M, scipy_constants.angstrom)
    np.testing.assert_allclose(dptb_constants.ELECTRON_CHARGE_C, scipy_constants.e)


def test_unified_postprocess_exports_epc_v1_symbols():
    assert unified_postprocess.DFTBPlusGauge is DFTBPlusGauge
    assert unified_postprocess.EPCData is EPCData
    assert unified_postprocess.EPCMeshData is EPCMeshData
    assert unified_postprocess.EPCMeshSpec is EPCMeshSpec
    assert unified_postprocess.EPC_MESH_NPZ_SCHEMA_VERSION == EPC_MESH_NPZ_SCHEMA_VERSION
    assert unified_postprocess.EPCPathData is EPCPathData
    assert unified_postprocess.LinewidthData is LinewidthData
    assert unified_postprocess.LinewidthMeshData is LinewidthMeshData
    assert unified_postprocess.LinewidthPathData is LinewidthPathData
    assert unified_postprocess.MobilityData is MobilityData
    assert unified_postprocess.MOBILITY_NPZ_SCHEMA_VERSION == MOBILITY_NPZ_SCHEMA_VERSION
    assert unified_postprocess.Phonons is Phonons
    assert unified_postprocess.RelaxationTimeData is RelaxationTimeData
    assert unified_postprocess.RelaxationTimeMeshData is RelaxationTimeMeshData
    assert unified_postprocess.RelaxationTimePathData is RelaxationTimePathData
    assert unified_postprocess.TransportData is TransportData
    assert unified_postprocess.SubspaceCouplingData is SubspaceCouplingData
    assert unified_postprocess.EPCKChunkSpec is EPCKChunkSpec
    assert unified_postprocess.build_k_chunk_specs is build_k_chunk_specs
    assert unified_postprocess.concat_epc_k_chunks is concat_epc_k_chunks
    assert unified_postprocess.compute_coupling_matrix is compute_coupling_matrix
    assert unified_postprocess.compute_linewidth is compute_linewidth
    assert unified_postprocess.compute_linewidth_mesh is compute_linewidth_mesh
    assert unified_postprocess.compute_linewidth_path is compute_linewidth_path
    assert unified_postprocess.compute_relaxation_time is compute_relaxation_time
    assert unified_postprocess.compute_relaxation_time_mesh is compute_relaxation_time_mesh
    assert unified_postprocess.compute_relaxation_time_path is compute_relaxation_time_path
    assert unified_postprocess.compute_serta_conductivity is compute_serta_conductivity
    assert unified_postprocess.compute_band_velocities_finite_difference is compute_band_velocities_finite_difference
    assert (
        unified_postprocess.compute_band_velocities_hamiltonian_derivative
        is compute_band_velocities_hamiltonian_derivative
    )
    assert unified_postprocess.compute_serta_mobility_si is compute_serta_mobility_si
    assert unified_postprocess.cumulative_path_coordinates is cumulative_path_coordinates
    assert unified_postprocess.compute_serta_transport_from_epc is compute_serta_transport_from_epc
    assert unified_postprocess.compute_subspace_coupling_strength is compute_subspace_coupling_strength
    assert unified_postprocess.compute_subspace_coupling_data is compute_subspace_coupling_data
    assert unified_postprocess.find_degenerate_band_groups is find_degenerate_band_groups


def test_external_reference_path_helpers_use_environment_overrides(monkeypatch, tmp_path):
    reference_root = tmp_path / "dftbephy"
    skdata_root = tmp_path / "matsci"
    monkeypatch.setenv("DEEPTB_EPH_REFERENCE_ROOT", str(reference_root))
    monkeypatch.setenv("DEEPTB_EPH_SKDATA_ROOT", str(skdata_root))

    assert _external_reference_root() == reference_root
    assert _external_skdata_root() == skdata_root


def test_compute_coupling_matrix_without_overlap():
    eigenvalues_k = np.array([[1.0, 2.0]])
    eigenvalues_kq = eigenvalues_k.reshape(1, 1, 2)
    eigenvectors_k = np.eye(2, dtype=complex).reshape(1, 2, 2)
    eigenvectors_kq = eigenvectors_k.reshape(1, 1, 2, 2)

    h_derivatives_k = np.zeros((1, 1, 3, 2, 2), dtype=complex)
    h_derivatives_kq = np.zeros((1, 1, 1, 3, 2, 2), dtype=complex)
    h_derivatives_k[0, 0, 0] = np.array([[1.0, 2.0], [3.0, 4.0]])

    phonon_eigenvectors = np.zeros((1, 1, 1, 3), dtype=complex)
    phonon_eigenvectors[0, 0, 0, 0] = 1.0

    coupling_matrix, coupling_strength = compute_coupling_matrix(
        eigenvalues_k=eigenvalues_k,
        eigenvectors_k=eigenvectors_k,
        eigenvalues_kq=eigenvalues_kq,
        eigenvectors_kq=eigenvectors_kq,
        h_derivatives_k=h_derivatives_k,
        h_derivatives_kq=h_derivatives_kq,
        phonon_eigenvectors=phonon_eigenvectors,
        masses=np.array([4.0]),
    )

    expected = 0.5 * h_derivatives_k[0, 0, 0]
    assert coupling_matrix.shape == (1, 1, 1, 2, 2)
    np.testing.assert_allclose(coupling_matrix[0, 0, 0], expected)
    np.testing.assert_allclose(coupling_strength[0, 0, 0], np.abs(expected) ** 2)


def test_compute_coupling_matrix_accepts_scalar_mass_for_single_atom():
    eigenvalues_k = np.array([[1.0]])
    eigenvalues_kq = eigenvalues_k.reshape(1, 1, 1)
    eigenvectors_k = np.ones((1, 1, 1), dtype=complex)
    eigenvectors_kq = eigenvectors_k.reshape(1, 1, 1, 1)
    h_derivatives_k = np.ones((1, 1, 3, 1, 1), dtype=complex) * 4.0
    h_derivatives_kq = np.zeros((1, 1, 1, 3, 1, 1), dtype=complex)
    phonon_eigenvectors = np.zeros((1, 1, 1, 3), dtype=complex)
    phonon_eigenvectors[0, 0, 0, 0] = 1.0

    scalar_coupling, scalar_strength = compute_coupling_matrix(
        eigenvalues_k=eigenvalues_k,
        eigenvectors_k=eigenvectors_k,
        eigenvalues_kq=eigenvalues_kq,
        eigenvectors_kq=eigenvectors_kq,
        h_derivatives_k=h_derivatives_k,
        h_derivatives_kq=h_derivatives_kq,
        phonon_eigenvectors=phonon_eigenvectors,
        masses=np.array(4.0),
    )
    array_coupling, array_strength = compute_coupling_matrix(
        eigenvalues_k=eigenvalues_k,
        eigenvectors_k=eigenvectors_k,
        eigenvalues_kq=eigenvalues_kq,
        eigenvectors_kq=eigenvectors_kq,
        h_derivatives_k=h_derivatives_k,
        h_derivatives_kq=h_derivatives_kq,
        phonon_eigenvectors=phonon_eigenvectors,
        masses=np.array([4.0]),
    )

    np.testing.assert_allclose(scalar_coupling, array_coupling)
    np.testing.assert_allclose(scalar_strength, array_strength)


def test_compute_coupling_matrix_with_overlap_and_frequency_prefactor():
    eigenvalues_k = np.array([[2.0]])
    eigenvalues_kq = np.array([[[3.0]]])
    eigenvectors_k = np.ones((1, 1, 1), dtype=complex)
    eigenvectors_kq = np.ones((1, 1, 1, 1), dtype=complex)
    h_derivatives_k = np.ones((1, 1, 3, 1, 1), dtype=complex) * 5.0
    h_derivatives_kq = np.ones((1, 1, 1, 3, 1, 1), dtype=complex)
    overlap_derivatives_k = np.ones_like(h_derivatives_k) * 0.5
    overlap_derivatives_kq = np.ones_like(h_derivatives_kq) * 0.25
    phonon_eigenvectors = np.zeros((1, 1, 1, 3), dtype=complex)
    phonon_eigenvectors[0, 0, 0, 0] = 1.0

    coupling_matrix, coupling_strength = compute_coupling_matrix(
        eigenvalues_k=eigenvalues_k,
        eigenvectors_k=eigenvectors_k,
        eigenvalues_kq=eigenvalues_kq,
        eigenvectors_kq=eigenvectors_kq,
        h_derivatives_k=h_derivatives_k,
        h_derivatives_kq=h_derivatives_kq,
        overlap_derivatives_k=overlap_derivatives_k,
        overlap_derivatives_kq=overlap_derivatives_kq,
        phonon_eigenvectors=phonon_eigenvectors,
        masses=np.array([1.0]),
        frequencies=np.array([[2.0]]),
    )

    raw = (5.0 - 1.0) - (2.0 * 0.5 - 3.0 * 0.25)
    prefactor = EPC_PREFAC_AMU_THZ / (2.0 * 2.0)
    np.testing.assert_allclose(coupling_matrix[0, 0, 0, 0, 0], raw * np.sqrt(prefactor))
    np.testing.assert_allclose(coupling_strength[0, 0, 0, 0, 0], abs(raw) ** 2 * prefactor)


def test_compute_coupling_matrix_frequency_floor_regularizes_frequency():
    eigenvalues_k = np.array([[0.0]])
    eigenvalues_kq = np.array([[[0.0]]])
    eigenvectors_k = np.ones((1, 1, 1), dtype=complex)
    eigenvectors_kq = np.ones((1, 1, 1, 1), dtype=complex)
    h_derivatives_k = np.ones((1, 1, 3, 1, 1), dtype=complex)
    h_derivatives_kq = np.zeros((1, 1, 1, 3, 1, 1), dtype=complex)
    phonon_eigenvectors = np.zeros((1, 1, 1, 3), dtype=complex)
    phonon_eigenvectors[0, 0, 0, 0] = 1.0

    coupling_matrix, coupling_strength = compute_coupling_matrix(
        eigenvalues_k=eigenvalues_k,
        eigenvectors_k=eigenvectors_k,
        eigenvalues_kq=eigenvalues_kq,
        eigenvectors_kq=eigenvectors_kq,
        h_derivatives_k=h_derivatives_k,
        h_derivatives_kq=h_derivatives_kq,
        phonon_eigenvectors=phonon_eigenvectors,
        masses=np.array([1.0]),
        frequencies=np.array([[0.0]]),
        omega_floor=1e-4,
    )

    prefactor = EPC_PREFAC_AMU_THZ / (2.0 * 1e-4)
    np.testing.assert_allclose(coupling_matrix[0, 0, 0, 0, 0], np.sqrt(prefactor))
    np.testing.assert_allclose(coupling_strength[0, 0, 0, 0, 0], prefactor)


def test_compute_coupling_matrix_rejects_invalid_phonon_scalars():
    eigenvalues_k = np.array([[1.0]])
    eigenvalues_kq = np.array([[[1.0]]])
    eigenvectors_k = np.ones((1, 1, 1), dtype=complex)
    eigenvectors_kq = np.ones((1, 1, 1, 1), dtype=complex)
    h_derivatives_k = np.ones((1, 1, 3, 1, 1), dtype=complex)
    h_derivatives_kq = np.ones((1, 1, 1, 3, 1, 1), dtype=complex)
    phonon_eigenvectors = np.ones((1, 1, 1, 3), dtype=complex)

    with pytest.raises(ValueError, match="masses"):
        compute_coupling_matrix(
            eigenvalues_k=eigenvalues_k,
            eigenvectors_k=eigenvectors_k,
            eigenvalues_kq=eigenvalues_kq,
            eigenvectors_kq=eigenvectors_kq,
            h_derivatives_k=h_derivatives_k,
            h_derivatives_kq=h_derivatives_kq,
            phonon_eigenvectors=phonon_eigenvectors,
            masses=np.array([0.0]),
        )
    with pytest.raises(ValueError, match="non-negative"):
        compute_coupling_matrix(
            eigenvalues_k=eigenvalues_k,
            eigenvectors_k=eigenvectors_k,
            eigenvalues_kq=eigenvalues_kq,
            eigenvectors_kq=eigenvectors_kq,
            h_derivatives_k=h_derivatives_k,
            h_derivatives_kq=h_derivatives_kq,
            phonon_eigenvectors=phonon_eigenvectors,
            masses=np.array([1.0]),
            frequencies=np.array([[-1.0]]),
        )
    with pytest.raises(ValueError, match="band_indices"):
        compute_coupling_matrix(
            eigenvalues_k=eigenvalues_k,
            eigenvectors_k=eigenvectors_k,
            eigenvalues_kq=eigenvalues_kq,
            eigenvectors_kq=eigenvectors_kq,
            h_derivatives_k=h_derivatives_k,
            h_derivatives_kq=h_derivatives_kq,
            phonon_eigenvectors=phonon_eigenvectors,
            masses=np.array([1.0]),
            band_indices=[1],
        )
    with pytest.raises(ValueError, match="band_indices"):
        compute_coupling_matrix(
            eigenvalues_k=eigenvalues_k,
            eigenvectors_k=eigenvectors_k,
            eigenvalues_kq=eigenvalues_kq,
            eigenvectors_kq=eigenvectors_kq,
            h_derivatives_k=h_derivatives_k,
            h_derivatives_kq=h_derivatives_kq,
            phonon_eigenvectors=phonon_eigenvectors,
            masses=np.array([1.0]),
            band_indices=[0.5],
        )


def test_compute_coupling_matrix_rejects_invalid_core_shapes_and_values():
    kwargs = {
        "eigenvalues_k": np.array([[1.0]]),
        "eigenvectors_k": np.ones((1, 1, 1), dtype=complex),
        "eigenvalues_kq": np.array([[[1.0]]]),
        "eigenvectors_kq": np.ones((1, 1, 1, 1), dtype=complex),
        "h_derivatives_k": np.ones((1, 1, 3, 1, 1), dtype=complex),
        "h_derivatives_kq": np.ones((1, 1, 1, 3, 1, 1), dtype=complex),
        "phonon_eigenvectors": np.ones((1, 1, 1, 3), dtype=complex),
        "masses": np.array([1.0]),
    }

    bad = dict(kwargs)
    bad["eigenvalues_k"] = np.array([[[1.0]]])
    with pytest.raises(ValueError, match="eigenvalues_k"):
        compute_coupling_matrix(**bad)

    bad = dict(kwargs)
    bad["eigenvectors_kq"] = np.ones((1, 1, 2, 1), dtype=complex)
    with pytest.raises(ValueError, match="eigenvectors_kq"):
        compute_coupling_matrix(**bad)

    bad = dict(kwargs)
    bad["h_derivatives_k"] = np.full((1, 1, 3, 1, 1), np.nan, dtype=complex)
    with pytest.raises(ValueError, match="h_derivatives_k"):
        compute_coupling_matrix(**bad)

    bad = dict(kwargs)
    bad["overlap_derivatives_k"] = np.ones((1, 3, 1, 1), dtype=complex)
    with pytest.raises(ValueError, match="overlap_derivatives_k"):
        compute_coupling_matrix(**bad)

    bad = dict(kwargs)
    bad["phonon_eigenvectors"] = np.ones((1, 1, 2, 3), dtype=complex)
    with pytest.raises(ValueError, match="natoms"):
        compute_coupling_matrix(**bad)

    bad = dict(kwargs)
    bad["phonon_eigenvectors"] = np.empty((1, 0, 1, 3), dtype=complex)
    with pytest.raises(ValueError, match="at least one phonon mode"):
        compute_coupling_matrix(**bad)

    bad = dict(kwargs)
    bad["prefactor_amu_thz"] = 0.0
    with pytest.raises(ValueError, match="prefactor"):
        compute_coupling_matrix(**bad)

    bad = dict(kwargs)
    bad["prefactor_amu_thz"] = "1.0"
    with pytest.raises(ValueError, match="prefactor"):
        compute_coupling_matrix(**bad)

    bad = dict(kwargs)
    bad["omega_floor"] = 0.0
    with pytest.raises(ValueError, match="omega_floor"):
        compute_coupling_matrix(**bad)

    bad = dict(kwargs)
    bad["omega_floor"] = np.array([1e-5])
    with pytest.raises(ValueError, match="omega_floor"):
        compute_coupling_matrix(**bad)

    bad = dict(kwargs)
    bad["h_derivatives_k"] = np.ones((1, 3, 1, 1), dtype=complex)
    bad["h_derivatives_kq"] = np.ones((1, 1, 3, 1, 1), dtype=complex)
    bad["derivative_mode"] = "row"
    with pytest.raises(ValueError, match="requires orbital_slices"):
        compute_coupling_matrix(**bad)


def test_compute_coupling_matrix_block_phase_row_derivatives():
    eigenvalues_k = np.array([[1.0, 2.0]])
    eigenvalues_kq = eigenvalues_k.reshape(1, 1, 2)
    eigenvectors_k = np.eye(2, dtype=complex).reshape(1, 2, 2)
    eigenvectors_kq = eigenvectors_k.reshape(1, 1, 2, 2)

    h_derivatives_k = np.zeros((1, 3, 2, 2), dtype=complex)
    h_derivatives_kq = np.zeros((1, 1, 3, 2, 2), dtype=complex)
    h_derivatives_k[0, 0] = np.array([[1.0, 2.0], [3.0, 4.0]])
    h_derivatives_kq[0, 0, 0] = np.array([[0.5, 1.0], [1.5, 2.0]])

    phonon_eigenvectors = np.zeros((1, 1, 2, 3), dtype=complex)
    phonon_eigenvectors[0, 0, :, 0] = 1.0
    qpoints = np.array([[0.25, 0.0, 0.0]])
    scaled_positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    phase = np.exp(2j * np.pi * (scaled_positions @ qpoints[0]))

    coupling_matrix, coupling_strength = compute_coupling_matrix(
        eigenvalues_k=eigenvalues_k,
        eigenvectors_k=eigenvectors_k,
        eigenvalues_kq=eigenvalues_kq,
        eigenvectors_kq=eigenvectors_kq,
        h_derivatives_k=h_derivatives_k,
        h_derivatives_kq=h_derivatives_kq,
        phonon_eigenvectors=phonon_eigenvectors,
        masses=np.ones(2),
        qpoints=qpoints,
        scaled_positions=scaled_positions,
        orbital_slices=[(0, 1), (1, 2)],
        derivative_mode="row",
    )

    expected = np.empty((2, 2), dtype=complex)
    expected[0, 0] = phase[0] * h_derivatives_k[0, 0, 0, 0] - phase[0] * h_derivatives_kq[0, 0, 0, 0, 0]
    expected[0, 1] = phase[0] * h_derivatives_k[0, 0, 0, 1] - phase[1] * h_derivatives_kq[0, 0, 0, 0, 1]
    expected[1, 0] = phase[1] * h_derivatives_k[0, 0, 1, 0] - phase[0] * h_derivatives_kq[0, 0, 0, 1, 0]
    expected[1, 1] = phase[1] * h_derivatives_k[0, 0, 1, 1] - phase[1] * h_derivatives_kq[0, 0, 0, 1, 1]
    np.testing.assert_allclose(coupling_matrix[0, 0, 0], expected)
    np.testing.assert_allclose(coupling_strength[0, 0, 0], np.abs(expected) ** 2)


def test_coupling_strength_is_invariant_to_orbital_sign_gauge():
    eigenvalues_k = np.array([[1.0, 2.0]])
    eigenvalues_kq = np.array([[[1.2, 2.3]]])
    eigenvectors_k = np.eye(2, dtype=complex).reshape(1, 2, 2)
    eigenvectors_kq = eigenvectors_k.reshape(1, 1, 2, 2)
    h_derivatives_k = np.zeros((1, 1, 3, 2, 2), dtype=complex)
    h_derivatives_kq = np.zeros((1, 1, 1, 3, 2, 2), dtype=complex)
    overlap_derivatives_k = np.zeros_like(h_derivatives_k)
    overlap_derivatives_kq = np.zeros_like(h_derivatives_kq)

    h_derivatives_k[0, 0, 0] = np.array([[0.2, 0.3 + 0.1j], [0.3 - 0.1j, -0.1]])
    h_derivatives_kq[0, 0, 0, 0] = np.array([[0.05, -0.2j], [0.2j, 0.07]])
    overlap_derivatives_k[0, 0, 0] = np.array([[0.01, 0.02], [0.02, -0.03]])
    overlap_derivatives_kq[0, 0, 0, 0] = np.array([[0.02, -0.01j], [0.01j, 0.01]])
    phonon_eigenvectors = np.zeros((1, 1, 1, 3), dtype=complex)
    phonon_eigenvectors[0, 0, 0, 0] = 1.0

    _, reference_strength = compute_coupling_matrix(
        eigenvalues_k=eigenvalues_k,
        eigenvectors_k=eigenvectors_k,
        eigenvalues_kq=eigenvalues_kq,
        eigenvectors_kq=eigenvectors_kq,
        h_derivatives_k=h_derivatives_k,
        h_derivatives_kq=h_derivatives_kq,
        overlap_derivatives_k=overlap_derivatives_k,
        overlap_derivatives_kq=overlap_derivatives_kq,
        phonon_eigenvectors=phonon_eigenvectors,
        masses=np.array([1.0]),
        frequencies=np.array([[5.0]]),
    )

    orbital_signs = np.diag([1.0, -1.0])
    transformed_eigenvectors_k = orbital_signs @ eigenvectors_k
    transformed_eigenvectors_kq = orbital_signs @ eigenvectors_kq
    transformed_h_derivatives_k = orbital_signs @ h_derivatives_k @ orbital_signs
    transformed_h_derivatives_kq = orbital_signs @ h_derivatives_kq @ orbital_signs
    transformed_overlap_derivatives_k = orbital_signs @ overlap_derivatives_k @ orbital_signs
    transformed_overlap_derivatives_kq = orbital_signs @ overlap_derivatives_kq @ orbital_signs

    _, transformed_strength = compute_coupling_matrix(
        eigenvalues_k=eigenvalues_k,
        eigenvectors_k=transformed_eigenvectors_k,
        eigenvalues_kq=eigenvalues_kq,
        eigenvectors_kq=transformed_eigenvectors_kq,
        h_derivatives_k=transformed_h_derivatives_k,
        h_derivatives_kq=transformed_h_derivatives_kq,
        overlap_derivatives_k=transformed_overlap_derivatives_k,
        overlap_derivatives_kq=transformed_overlap_derivatives_kq,
        phonon_eigenvectors=phonon_eigenvectors,
        masses=np.array([1.0]),
        frequencies=np.array([[5.0]]),
    )

    np.testing.assert_allclose(transformed_strength, reference_strength, atol=1e-14, rtol=1e-14)


def test_reshape_phonopy_eigenvectors_column_modes():
    ev = np.arange(2 * 6 * 6).reshape(2, 6, 6)
    reshaped = reshape_phonopy_eigenvectors(ev, natoms=2)
    assert reshaped.shape == (2, 6, 2, 3)
    np.testing.assert_array_equal(reshaped[0, 1], ev[0, :, 1].reshape(2, 3))


def test_phonons_from_phonopy_reads_external_modes():
    class _Primitive:
        masses = np.array([12.0, 12.0])
        cell = np.eye(3)
        scaled_positions = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0]])

    class _Phonopy:
        primitive = _Primitive()

        def __init__(self):
            self.requested_qpoints = None

        def run_qpoints(self, qpoints, with_eigenvectors=False):
            self.requested_qpoints = np.asarray(qpoints, dtype=float)
            self.requested_eigenvectors = with_eigenvectors

        def get_qpoints_dict(self):
            eigenvectors = np.arange(2 * 6 * 6).reshape(2, 6, 6)
            return {
                "qpoints": self.requested_qpoints,
                "frequencies": np.ones((2, 6)),
                "eigenvectors": eigenvectors,
            }

    phonopy_obj = _Phonopy()
    qpoints = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    phonons = Phonons.from_phonopy(phonopy_obj, qpoints=qpoints)

    assert phonopy_obj.requested_eigenvectors is True
    np.testing.assert_allclose(phonons.qpoints, qpoints)
    assert phonons.frequencies.shape == (2, 6)
    assert phonons.eigenvectors.shape == (2, 6, 2, 3)
    np.testing.assert_allclose(phonons.masses, np.array([12.0, 12.0]))
    np.testing.assert_allclose(phonons.cell, np.eye(3))
    assert phonons.metadata["source"] == "phonopy"


def test_phonons_from_phonopy_file_delegates_to_phonopy_load(monkeypatch, tmp_path):
    class _Primitive:
        masses = np.array([12.0])
        cell = np.eye(3)
        scaled_positions = np.array([[0.0, 0.0, 0.0]])

    class _Phonopy:
        primitive = _Primitive()

        def __init__(self):
            self.requested_qpoints = None

        def run_qpoints(self, qpoints, with_eigenvectors=False):
            self.requested_qpoints = np.asarray(qpoints, dtype=float)
            self.requested_eigenvectors = with_eigenvectors

        def get_qpoints_dict(self):
            return {
                "qpoints": self.requested_qpoints,
                "frequencies": np.ones((1, 3)),
                "eigenvectors": np.ones((1, 3, 3), dtype=complex),
            }

    calls = {}

    class _PhonopyModule:
        @staticmethod
        def load(phonopy_yaml, force_sets_filename=None, **kwargs):
            calls["phonopy_yaml"] = phonopy_yaml
            calls["force_sets_filename"] = force_sets_filename
            calls["kwargs"] = kwargs
            return _Phonopy()

    monkeypatch.setitem(sys.modules, "phonopy", _PhonopyModule)

    phonopy_yaml = tmp_path / "phonopy_disp.yaml"
    force_sets = tmp_path / "FORCE_SETS"
    qpoints = np.array([[0.0, 0.0, 0.0]])
    phonons = Phonons.from_phonopy_file(
        phonopy_yaml,
        qpoints=qpoints,
        force_sets_filename=force_sets,
        primitive_matrix="auto",
    )

    assert calls == {
        "phonopy_yaml": str(phonopy_yaml),
        "force_sets_filename": str(force_sets),
        "kwargs": {"primitive_matrix": "auto"},
    }
    np.testing.assert_allclose(phonons.qpoints, qpoints)
    assert phonons.eigenvectors.shape == (1, 3, 1, 3)
    assert phonons.metadata["source"] == "phonopy_file"
    assert phonons.metadata["phonopy_yaml"] == str(phonopy_yaml)
    assert phonons.metadata["force_sets_filename"] == str(force_sets)


def test_phonons_from_phonopy_file_omits_optional_force_sets(monkeypatch, tmp_path):
    class _Primitive:
        masses = np.array([12.0])
        cell = np.eye(3)
        scaled_positions = np.array([[0.0, 0.0, 0.0]])

    class _Phonopy:
        primitive = _Primitive()

        def run_qpoints(self, qpoints, with_eigenvectors=False):
            self.requested_qpoints = np.asarray(qpoints, dtype=float)

        def get_qpoints_dict(self):
            return {
                "qpoints": self.requested_qpoints,
                "frequencies": np.ones((1, 3)),
                "eigenvectors": np.ones((1, 3, 3), dtype=complex),
            }

    calls = {}

    class _PhonopyModule:
        @staticmethod
        def load(phonopy_yaml, **kwargs):
            calls["phonopy_yaml"] = phonopy_yaml
            calls["kwargs"] = kwargs
            return _Phonopy()

    monkeypatch.setitem(sys.modules, "phonopy", _PhonopyModule)

    phonopy_yaml = tmp_path / "phonopy_disp.yaml"
    qpoints = np.array([[0.0, 0.0, 0.0]])
    phonons = Phonons.from_phonopy_file(phonopy_yaml, qpoints=qpoints, primitive_matrix="auto")

    assert calls == {
        "phonopy_yaml": str(phonopy_yaml),
        "kwargs": {"primitive_matrix": "auto"},
    }
    assert "force_sets_filename" not in phonons.metadata
    np.testing.assert_allclose(phonons.qpoints, qpoints)


def test_orbital_slices_from_atom_orbs():
    atom_orbs = ["0-C-2s", "0-C-2p_y", "0-C-2p_z", "0-C-2p_x", "1-C-2s", "1-C-2p_y"]
    assert orbital_slices_from_atom_orbs(atom_orbs) == [(0, 4), (4, 6)]


def test_orbital_slices_from_system_uses_structured_metadata():
    class _System:
        atomic_symbols = ["C", "B", "C"]

        class calculator:
            @staticmethod
            def get_orbital_info():
                return {
                    "C": ["2s", "2p_y", "2p_z", "2p_x"],
                    "B": ["2s"],
                }

    assert orbital_slices_from_system(_System()) == [(0, 4), (4, 5), (5, 9)]


def test_normalize_orbital_slices_rejects_invalid_ranges():
    normalized = normalize_orbital_slices([(0, 1), slice(1, 3)])

    assert [(item.start, item.stop) for item in normalized] == [(0, 1), (1, 3)]
    with pytest.raises(ValueError, match="non-empty"):
        normalize_orbital_slices([])
    with pytest.raises(ValueError, match="explicit"):
        normalize_orbital_slices([slice(None, 1)])
    with pytest.raises(ValueError, match="integer"):
        normalize_orbital_slices([(0.5, 1)])
    with pytest.raises(ValueError, match="integer"):
        normalize_orbital_slices([(False, 1)])
    with pytest.raises(ValueError, match="contiguous"):
        normalize_orbital_slices([(1, 2)])
    with pytest.raises(ValueError, match="contiguous"):
        normalize_orbital_slices([(0, 2), (1, 3)])
    with pytest.raises(ValueError, match="contiguous"):
        normalize_orbital_slices([(0, 0)])


def test_assemble_directed_hk_from_blocks():
    blocks = {
        "0_0_0_0_0": np.array([[1.0]]),
        "0_1_1_0_0": np.array([[2.0]]),
        "1_0_-1_0_0": np.array([[3.0]]),
    }
    kpoints = np.array([[0.25, 0.0, 0.0]])
    hk = assemble_directed_hk_from_blocks(blocks, kpoints, [slice(0, 1), slice(1, 2)], 2)

    assert hk.shape == (1, 2, 2)
    np.testing.assert_allclose(hk[0, 0, 0], 1.0)
    np.testing.assert_allclose(hk[0, 0, 1], 2.0 * np.exp(-0.5j * np.pi))
    np.testing.assert_allclose(hk[0, 1, 0], 3.0 * np.exp(0.5j * np.pi))


def test_assemble_directed_hk_rejects_invalid_orbital_slices_and_block_keys():
    kpoints = np.array([[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="k/q points"):
        assemble_directed_hk_from_blocks({"0_0_0_0_0": np.ones((1, 1))}, np.empty((0, 3)), [slice(0, 1)], 1)
    with pytest.raises(ValueError, match="norb"):
        assemble_directed_hk_from_blocks({"0_0_0_0_0": np.ones((1, 1))}, kpoints, [slice(0, 1)], 1.5)
    with pytest.raises(ValueError, match="norb"):
        assemble_directed_hk_from_blocks({}, kpoints, [slice(0, 1)], 0)
    with pytest.raises(ValueError, match="norb"):
        assemble_directed_hk_from_blocks({"0_0_0_0_0": np.ones((1, 1))}, kpoints, [slice(0, 1)], 2)
    with pytest.raises(ValueError, match="contiguous"):
        assemble_directed_hk_from_blocks(
            {"0_0_0_0_0": np.ones((1, 1))},
            kpoints,
            [slice(0, 1), slice(2, 3)],
            3,
        )
    with pytest.raises(ValueError, match="negative atom index"):
        assemble_directed_hk_from_blocks({"-1_0_0_0_0": np.ones((1, 1))}, kpoints, [slice(0, 1)], 1)
    with pytest.raises(ValueError, match="shape"):
        assemble_directed_hk_from_blocks(
            {"0_1_0_0_0": np.ones((2, 1))},
            kpoints,
            [slice(0, 1), slice(1, 2)],
            2,
        )


def test_supercell_provider_fourier_row_derivative():
    provider = SupercellFD.__new__(SupercellFD)
    provider.primitive_to_supercell_atom = np.array([0])
    provider.supercell_to_primitive_atom = np.array([0, 0])
    provider.supercell_atom_to_cell = np.array([0, 1])
    provider.primitive_orbital_offsets = np.array([0, 1])
    provider.supercell_orbital_offsets = np.array([0, 1, 2])
    provider.shortest_vectors = np.zeros((2, 1, 1, 3))
    provider.shortest_vectors[1, 0, 0] = np.array([1.0, 0.0, 0.0])
    provider.vector_multiplicity = np.ones((2, 1), dtype=int)

    supercell_derivative = np.array([[2.0, 3.0], [5.0, 7.0]])
    rows = provider._fourier_row_derivative(supercell_derivative, primitive_atom_index=0, kpoints=np.array([[0.25, 0.0, 0.0]]))

    assert rows.shape == (1, 1, 1)
    np.testing.assert_allclose(rows[0, 0, 0], 2.0 + 3.0j)


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


def test_supercell_fd_rejects_invalid_mapping_inputs():
    provider = SupercellFD(**_minimal_supercell_fd_kwargs())
    assert provider.displacement == 1e-3

    bad = _minimal_supercell_fd_kwargs()
    bad["primitive_to_supercell_atom"] = np.array([2])
    with pytest.raises(ValueError, match="primitive_to_supercell_atom"):
        SupercellFD(**bad)

    bad = _minimal_supercell_fd_kwargs()
    bad["primitive_to_supercell_atom"] = np.array([0.5])
    with pytest.raises(ValueError, match="primitive_to_supercell_atom"):
        SupercellFD(**bad)

    bad = _minimal_supercell_fd_kwargs()
    bad["primitive_orbital_offsets"] = np.array([0, 0])
    with pytest.raises(ValueError, match="primitive_orbital_offsets"):
        SupercellFD(**bad)

    bad = _minimal_supercell_fd_kwargs()
    bad["vector_multiplicity"] = np.array([[0], [1]])
    with pytest.raises(ValueError, match="vector_multiplicity"):
        SupercellFD(**bad)

    bad = _minimal_supercell_fd_kwargs()
    bad["shortest_vectors"] = np.full((2, 1, 1, 3), np.nan)
    with pytest.raises(ValueError, match="shortest_vectors"):
        SupercellFD(**bad)
    with pytest.raises(ValueError, match="displacement"):
        SupercellFD(**_minimal_supercell_fd_kwargs(), displacement=np.nan)
    with pytest.raises(ValueError, match="displacement"):
        SupercellFD(**_minimal_supercell_fd_kwargs(), displacement="0.1")


def test_length_unit_scale_rejects_invalid_type():
    with pytest.raises(ValueError, match="length_unit"):
        _length_unit_scale_to_angstrom(None)


def test_dftbplus_benchmark_convention_transforms():
    convention = DFTBPlusGauge.from_atom_orbs(
        ["0-C-2s", "0-C-2p_y", "0-C-2p_z", "0-C-2p_x"]
    )

    np.testing.assert_array_equal(convention.orbital_signs, np.array([1.0, -1.0, 1.0, -1.0]))
    np.testing.assert_array_equal(convention.derivative_signs[0], np.array([1.0, -1.0, 1.0, -1.0]))
    np.testing.assert_array_equal(convention.derivative_signs[2], np.array([1.0, -1.0, -1.0, -1.0]))

    eigenvectors = np.eye(4, dtype=complex)[None] * (1.0 + 1.0j)
    transformed = convention.transform_eigenvectors(eigenvectors)
    expected = np.diag([1.0, -1.0, 1.0, -1.0])[None] * (1.0 - 1.0j)
    np.testing.assert_allclose(transformed, expected)

    derivatives = np.ones((1, 3, 4, 4), dtype=complex)
    derivatives[0, 2] *= np.arange(16).reshape(4, 4)
    transformed_derivatives = convention.transform_row_derivatives(derivatives)
    z_signs = convention.derivative_signs[2]
    np.testing.assert_allclose(
        transformed_derivatives[0, 2],
        derivatives[0, 2] * z_signs[:, None] * z_signs[None, :],
    )


def test_epc_data_npz_roundtrip(tmp_path):
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        qpoints=np.array([[0.1, 0.0, 0.0]]),
        band_indices=np.array([0, 1]),
        frequencies=np.array([[1.0, 2.0]]),
        eigenvalues_k=np.array([[1.0, 2.0]]),
        eigenvalues_kq=np.array([[[1.1, 2.1]]]),
        coupling_matrix=np.ones((1, 1, 2, 2, 2), dtype=complex) * (1.0 + 2.0j),
        coupling_strength=np.ones((1, 1, 2, 2, 2)),
        metadata={"use_scc": False, "frequency_unit": "THz"},
    )
    path = tmp_path / "epc_data.npz"
    epc_data.save_npz(path)
    loaded = EPCData.load_npz(path)

    np.testing.assert_allclose(loaded.coupling_matrix, epc_data.coupling_matrix)
    np.testing.assert_allclose(loaded.coupling_strength, epc_data.coupling_strength)
    np.testing.assert_array_equal(loaded.band_indices, epc_data.band_indices)
    assert loaded.metadata["frequency_unit"] == "THz"
    assert loaded.metadata["schema"] == "deeptb.epc_data"
    assert loaded.metadata["schema_version"] == EPC_NPZ_SCHEMA_VERSION
    assert loaded.metadata["coupling_unit"] == "eV"

    with np.load(path, allow_pickle=False) as data:
        assert "elph_coupling_matrix" in data
        assert "metadata_json" in data


def test_epc_mesh_spec_generates_kmesh_and_validates_phonon_qmesh():
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
        frequencies=np.array([[1.0], [2.0]]),
        eigenvectors=np.array([[[[1.0, 0.0, 0.0]]], [[[1.0, 0.0, 0.0]]]], dtype=complex),
        masses=np.array([1.0]),
    )
    spec = EPCMeshSpec(k_mesh=[2, 1, 1], q_mesh=[2, 1, 1])

    kpoints, kpoint_weights = spec.resolve_kpoints_and_weights()
    qpoint_weights = spec.resolve_qpoint_weights(phonons)

    np.testing.assert_allclose(kpoints, np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]]))
    np.testing.assert_allclose(kpoint_weights, np.array([0.5, 0.5]))
    np.testing.assert_allclose(qpoint_weights, np.array([0.5, 0.5]))
    assert spec.metadata_payload()["k_mesh"] == [2, 1, 1]
    assert spec.metadata_payload()["q_mesh"] == [2, 1, 1]

    with pytest.raises(ValueError, match="either kpoints or k_mesh"):
        EPCMeshSpec()
    with pytest.raises(ValueError, match="not both"):
        EPCMeshSpec(kpoints=np.array([[0.0, 0.0, 0.0]]), k_mesh=[1, 1, 1])
    with pytest.raises(ValueError, match="q_mesh"):
        EPCMeshSpec(k_mesh=[1, 1, 1], q_mesh=[1, 1, 1]).resolve_qpoint_weights(phonons)
    with pytest.raises(ValueError, match="k_mesh"):
        EPCMeshSpec(k_mesh=[1, 0, 1])


def test_epc_mesh_data_npz_roundtrip(tmp_path):
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        band_indices=np.array([0, 1]),
        frequencies=np.array([[1.0, 2.0]]),
        eigenvalues_k=np.array([[1.0, 2.0], [1.1, 2.1]]),
        eigenvalues_kq=np.array([[[1.1, 2.1], [1.2, 2.2]]]),
        coupling_matrix=np.ones((1, 2, 2, 2, 2), dtype=complex) * (1.0 + 2.0j),
        coupling_strength=np.ones((1, 2, 2, 2, 2)),
        metadata={"source": "unit-test", "frequency_unit": "THz"},
    )
    mesh_data = EPCMeshData.from_epc_data(
        epc_data,
        kpoint_weights=np.array([1.0, 1.0]),
        qpoint_weights=np.array([2.0]),
        metadata={"mesh_spec": {"k_mesh": [2, 1, 1]}},
    )
    path = tmp_path / "epc_mesh_data.npz"
    mesh_data.save_npz(path)
    loaded = EPCMeshData.load_npz(path)

    np.testing.assert_allclose(loaded.coupling_matrix, mesh_data.coupling_matrix)
    np.testing.assert_allclose(loaded.kpoint_weights, np.array([0.5, 0.5]))
    np.testing.assert_allclose(loaded.qpoint_weights, np.array([1.0]))
    assert loaded.metadata["schema"] == "deeptb.epc_mesh_data"
    assert loaded.metadata["schema_version"] == EPC_MESH_NPZ_SCHEMA_VERSION
    assert loaded.metadata["mesh_spec"]["k_mesh"] == [2, 1, 1]
    assert loaded.epc_data.metadata["schema"] == "deeptb.epc_data"

    with np.load(path, allow_pickle=False) as data:
        assert "el_kpoint_weights" in data
        assert "ph_qpoint_weights" in data
        assert "metadata_json" in data


def test_epc_path_data_npz_roundtrip(tmp_path):
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        band_indices=np.array([0, 1]),
        frequencies=np.array([[1.0, 2.0], [1.5, 2.5]]),
        eigenvalues_k=np.array([[1.0, 2.0]]),
        eigenvalues_kq=np.array([[[1.1, 2.1]], [[1.2, 2.2]]]),
        coupling_matrix=np.ones((2, 1, 2, 2, 2), dtype=complex) * (1.0 + 2.0j),
        coupling_strength=np.ones((2, 1, 2, 2, 2)),
        metadata={"source": "unit-test", "frequency_unit": "THz"},
    )
    path_data = EPCPathData.from_epc_data(
        epc_data,
        path_axis="q",
        path_coordinates=np.array([0.0, 0.25]),
        path_segments=np.array([[0, 2]]),
        metadata={"path_labels": {"G": 0, "X": 1}},
    )
    path = tmp_path / "epc_path_data.npz"
    path_data.save_npz(path)
    loaded = EPCPathData.load_npz(path)

    np.testing.assert_allclose(loaded.coupling_matrix, path_data.coupling_matrix)
    np.testing.assert_allclose(loaded.path_coordinates, np.array([0.0, 0.25]))
    np.testing.assert_array_equal(loaded.path_segments, np.array([[0, 2]]))
    assert loaded.path_axis == "q"
    assert loaded.metadata["schema"] == "deeptb.epc_path_data"
    assert loaded.metadata["schema_version"] == EPC_PATH_NPZ_SCHEMA_VERSION
    assert loaded.metadata["path_labels"] == {"G": 0, "X": 1}
    assert loaded.epc_data.metadata["schema"] == "deeptb.epc_data"

    with np.load(path, allow_pickle=False) as data:
        assert "path_axis" in data
        assert "path_coordinates" in data
        assert "path_segments" in data


def test_epc_path_data_rejects_inconsistent_path_metadata():
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        band_indices=np.array([0]),
        frequencies=np.array([[1.0]]),
        eigenvalues_k=np.array([[0.0]]),
        eigenvalues_kq=np.array([[[0.0]]]),
        coupling_matrix=np.ones((1, 1, 1, 1, 1), dtype=complex),
        coupling_strength=np.ones((1, 1, 1, 1, 1)),
    )

    with pytest.raises(ValueError, match="path_axis"):
        EPCPathData.from_epc_data(epc_data, path_axis="both", path_coordinates=np.array([0.0]))
    with pytest.raises(ValueError, match="path_coordinates length"):
        EPCPathData.from_epc_data(epc_data, path_axis="q", path_coordinates=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="path_segments"):
        EPCPathData.from_epc_data(
            epc_data,
            path_axis="q",
            path_coordinates=np.array([0.0]),
            path_segments=np.array([[0, 2]]),
        )


def test_cumulative_path_coordinates_uses_fractional_distances():
    np.testing.assert_allclose(
        cumulative_path_coordinates(np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.0], [0.3, 0.4, 0.2]])),
        np.array([0.0, 0.5, 0.7]),
    )


def test_phonons_npz_roundtrip(tmp_path):
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        frequencies=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
        eigenvectors=np.ones((2, 3, 1, 3), dtype=complex) * (1.0 + 0.5j),
        masses=np.array([12.0]),
        cell=np.eye(3),
        scaled_positions=np.array([[0.0, 0.0, 0.0]]),
        metadata={"source": "unit-test", "frequency_unit": "THz"},
    )
    path = tmp_path / "phonons.npz"
    phonons.save_npz(path)
    loaded = Phonons.load_npz(path)

    np.testing.assert_allclose(loaded.qpoints, phonons.qpoints)
    np.testing.assert_allclose(loaded.frequencies, phonons.frequencies)
    np.testing.assert_allclose(loaded.eigenvectors, phonons.eigenvectors)
    np.testing.assert_allclose(loaded.masses, phonons.masses)
    np.testing.assert_allclose(loaded.cell, phonons.cell)
    np.testing.assert_allclose(loaded.scaled_positions, phonons.scaled_positions)
    assert loaded.metadata["source"] == "unit-test"
    assert loaded.metadata["schema"] == "deeptb.phonons"
    assert loaded.metadata["schema_version"] == PHONON_NPZ_SCHEMA_VERSION
    assert loaded.metadata["mass_unit"] == "amu"

    with np.load(path, allow_pickle=False) as data:
        assert "ph_eigenvectors" in data
        assert "ph_masses" in data
        assert "metadata_json" in data


def test_npz_metadata_json_must_be_scalar_json_object(tmp_path):
    payload = {
        "ph_qpoints": np.array([[0.0, 0.0, 0.0]]),
        "ph_frequencies": np.array([[1.0, 2.0, 3.0]]),
        "ph_eigenvectors": np.ones((1, 3, 1, 3), dtype=complex),
        "ph_masses": np.array([1.0]),
    }

    missing_metadata = tmp_path / "missing_metadata.npz"
    np.savez(missing_metadata, **payload)
    with pytest.raises(ValueError, match="metadata_json"):
        Phonons.load_npz(missing_metadata)

    array_metadata = tmp_path / "array_metadata.npz"
    np.savez(array_metadata, **payload, metadata_json=np.array(["{}", "{}"]))
    with pytest.raises(ValueError, match="scalar JSON object"):
        Phonons.load_npz(array_metadata)

    invalid_json = tmp_path / "invalid_metadata.npz"
    np.savez(invalid_json, **payload, metadata_json=np.array("{not-json"))
    with pytest.raises(ValueError, match="valid JSON"):
        Phonons.load_npz(invalid_json)

    non_object_json = tmp_path / "non_object_metadata.npz"
    np.savez(non_object_json, **payload, metadata_json=np.array("[]"))
    with pytest.raises(ValueError, match="JSON object"):
        Phonons.load_npz(non_object_json)


def test_phonons_accepts_scalar_mass_for_single_atom():
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        frequencies=np.array([[1.0, 2.0, 3.0]]),
        eigenvectors=np.ones((1, 3, 1, 3), dtype=complex),
        masses=np.array(12.0),
    )

    np.testing.assert_allclose(phonons.masses, np.array([12.0]))


def test_phonons_rejects_inconsistent_mode_shapes():
    with pytest.raises(ValueError, match="non-empty"):
        Phonons(
            qpoints=np.empty((0, 3)),
            frequencies=np.empty((0, 3)),
            eigenvectors=np.empty((0, 3, 1, 3), dtype=complex),
            masses=np.array([1.0]),
        )

    with pytest.raises(ValueError, match="eigenvectors and frequencies"):
        Phonons(
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            frequencies=np.array([[1.0, 2.0]]),
            eigenvectors=np.ones((1, 3, 1, 3), dtype=complex),
            masses=np.array([1.0]),
        )
    with pytest.raises(ValueError, match="at least one phonon mode"):
        Phonons(
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            frequencies=np.empty((1, 0)),
            eigenvectors=np.empty((1, 0, 1, 3), dtype=complex),
            masses=np.array([1.0]),
        )


def test_phonons_rejects_conflicting_schema_metadata():
    with pytest.raises(ValueError, match="frequency_unit"):
        Phonons(
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            frequencies=np.array([[1.0, 2.0, 3.0]]),
            eigenvectors=np.ones((1, 3, 1, 3), dtype=complex),
            masses=np.array([1.0]),
            metadata={"frequency_unit": "cm^-1"},
        )


def test_phonons_rejects_nonpositive_masses():
    with pytest.raises(ValueError, match="masses"):
        Phonons(
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            frequencies=np.array([[1.0, 2.0, 3.0]]),
            eigenvectors=np.ones((1, 3, 1, 3), dtype=complex),
            masses=np.array([0.0]),
        )


def test_phonons_rejects_nonfinite_geometry_or_modes():
    with pytest.raises(ValueError, match="k/q points"):
        Phonons(
            qpoints=np.array([[np.nan, 0.0, 0.0]]),
            frequencies=np.array([[1.0, 2.0, 3.0]]),
            eigenvectors=np.ones((1, 3, 1, 3), dtype=complex),
            masses=np.array([1.0]),
        )
    with pytest.raises(ValueError, match="eigenvectors"):
        Phonons(
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            frequencies=np.array([[1.0, 2.0, 3.0]]),
            eigenvectors=np.array([[[[np.inf, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]], dtype=complex),
            masses=np.array([1.0]),
        )
    with pytest.raises(ValueError, match="scaled_positions"):
        Phonons(
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            frequencies=np.array([[1.0, 2.0, 3.0]]),
            eigenvectors=np.ones((1, 3, 1, 3), dtype=complex),
            masses=np.array([1.0]),
            scaled_positions=np.array([[np.nan, 0.0, 0.0]]),
        )


def test_epc_data_rejects_inconsistent_coupling_shape():
    with pytest.raises(ValueError, match="non-empty"):
        EPCData(
            kpoints=np.empty((0, 3)),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0]),
            frequencies=np.array([[1.0]]),
            eigenvalues_k=np.empty((0, 1)),
            eigenvalues_kq=np.empty((1, 0, 1)),
            coupling_matrix=np.empty((1, 0, 1, 1, 1), dtype=complex),
            coupling_strength=np.empty((1, 0, 1, 1, 1)),
        )
    with pytest.raises(ValueError, match="non-empty"):
        EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            qpoints=np.empty((0, 3)),
            band_indices=np.array([0]),
            frequencies=np.empty((0, 1)),
            eigenvalues_k=np.array([[0.0]]),
            eigenvalues_kq=np.empty((0, 1, 1)),
            coupling_matrix=np.empty((0, 1, 1, 1, 1), dtype=complex),
            coupling_strength=np.empty((0, 1, 1, 1, 1)),
        )
    with pytest.raises(ValueError, match="at least one phonon mode"):
        EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0]),
            frequencies=np.empty((1, 0)),
            eigenvalues_k=np.array([[0.0]]),
            eigenvalues_kq=np.array([[[0.0]]]),
            coupling_matrix=np.empty((1, 1, 0, 1, 1), dtype=complex),
            coupling_strength=np.empty((1, 1, 0, 1, 1)),
        )

    with pytest.raises(ValueError, match="coupling_matrix"):
        EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0]),
            frequencies=np.array([[1.0]]),
            eigenvalues_k=np.array([[0.0]]),
            eigenvalues_kq=np.array([[[0.0]]]),
            coupling_matrix=np.ones((1, 1, 1, 2, 2), dtype=complex),
            coupling_strength=np.ones((1, 1, 1, 1, 1)),
        )


def test_epc_data_accepts_scalar_band_index():
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        band_indices=np.array(0),
        frequencies=np.array([[1.0]]),
        eigenvalues_k=np.array([[0.0]]),
        eigenvalues_kq=np.array([[[0.0]]]),
        coupling_matrix=np.ones((1, 1, 1, 1, 1), dtype=complex),
        coupling_strength=np.ones((1, 1, 1, 1, 1)),
    )

    np.testing.assert_array_equal(epc_data.band_indices, np.array([0]))


def test_epc_data_rejects_invalid_indices_and_nonfinite_values():
    with pytest.raises(ValueError, match="band_indices"):
        EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([-1]),
            frequencies=np.array([[1.0]]),
            eigenvalues_k=np.array([[0.0]]),
            eigenvalues_kq=np.array([[[0.0]]]),
            coupling_matrix=np.ones((1, 1, 1, 1, 1), dtype=complex),
            coupling_strength=np.ones((1, 1, 1, 1, 1)),
        )
    with pytest.raises(ValueError, match="band_indices"):
        EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0.5]),
            frequencies=np.array([[1.0]]),
            eigenvalues_k=np.array([[0.0]]),
            eigenvalues_kq=np.array([[[0.0]]]),
            coupling_matrix=np.ones((1, 1, 1, 1, 1), dtype=complex),
            coupling_strength=np.ones((1, 1, 1, 1, 1)),
        )
    with pytest.raises(ValueError, match="coupling_strength"):
        EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0]),
            frequencies=np.array([[1.0]]),
            eigenvalues_k=np.array([[0.0]]),
            eigenvalues_kq=np.array([[[0.0]]]),
            coupling_matrix=np.ones((1, 1, 1, 1, 1), dtype=complex),
            coupling_strength=np.array([[[[[np.inf]]]]]),
        )
    with pytest.raises(ValueError, match="coupling_strength"):
        EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0]),
            frequencies=np.array([[1.0]]),
            eigenvalues_k=np.array([[0.0]]),
            eigenvalues_kq=np.array([[[0.0]]]),
            coupling_matrix=np.ones((1, 1, 1, 1, 1), dtype=complex),
            coupling_strength=np.array([[[[[-1.0]]]]]),
        )
    with pytest.raises(ValueError, match="non-negative"):
        EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0]),
            frequencies=np.array([[-1.0]]),
            eigenvalues_k=np.array([[0.0]]),
            eigenvalues_kq=np.array([[[0.0]]]),
            coupling_matrix=np.ones((1, 1, 1, 1, 1), dtype=complex),
            coupling_strength=np.ones((1, 1, 1, 1, 1)),
        )
    with pytest.raises(ValueError, match="k/q points"):
        EPCData(
            kpoints=np.array([[0.0, 0.0, np.nan]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0]),
            frequencies=np.array([[1.0]]),
            eigenvalues_k=np.array([[0.0]]),
            eigenvalues_kq=np.array([[[0.0]]]),
            coupling_matrix=np.ones((1, 1, 1, 1, 1), dtype=complex),
            coupling_strength=np.ones((1, 1, 1, 1, 1)),
        )


def test_epc_data_rejects_conflicting_schema_metadata():
    with pytest.raises(ValueError, match="coupling_unit"):
        EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0]),
            frequencies=np.array([[1.0]]),
            eigenvalues_k=np.array([[0.0]]),
            eigenvalues_kq=np.array([[[0.0]]]),
            coupling_matrix=np.ones((1, 1, 1, 1, 1), dtype=complex),
            coupling_strength=np.ones((1, 1, 1, 1, 1)),
            metadata={"coupling_unit": "Hartree"},
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


def test_compute_linewidth_gaussian_matches_manual_reference():
    epc_data = _small_linewidth_epc_data()
    linewidth = compute_linewidth(
        epc_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="Gaussian",
    )
    expected, expected_absorption, expected_emission = _manual_linewidth(
        epc_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="gaussian",
        mode_resolved=False,
    )

    np.testing.assert_allclose(linewidth.linewidth, expected)
    np.testing.assert_allclose(linewidth.absorption, expected_absorption)
    np.testing.assert_allclose(linewidth.emission, expected_emission)
    assert linewidth.metadata["linewidth_unit"] == "eV"
    assert linewidth.metadata["broadening"] == "gaussian"
    np.testing.assert_allclose(linewidth.metadata["thz_to_ev"], THZ_TO_EV)


def test_compute_linewidth_mode_resolved_sums_to_total():
    epc_data = _small_linewidth_epc_data()
    total = compute_linewidth(
        epc_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="lorentzian",
        mode_resolved=False,
    )
    mode_resolved = compute_linewidth(
        epc_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="lorentzian",
        mode_resolved=True,
    )
    expected, _, _ = _manual_linewidth(
        epc_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="lorentzian",
        mode_resolved=True,
    )

    np.testing.assert_allclose(mode_resolved.linewidth, expected)
    np.testing.assert_allclose(mode_resolved.linewidth.sum(axis=-1), total.linewidth)
    assert mode_resolved.metadata["mode_resolved"] is True


def test_compute_linewidth_path_keeps_q_path_axis_and_sums_to_total():
    epc_path_data = _small_linewidth_epc_path_data()
    total = compute_linewidth(
        epc_path_data.epc_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="lorentzian",
        mode_resolved=False,
    )
    path_linewidth = compute_linewidth_path(
        epc_path_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="lorentzian",
        mode_resolved=False,
    )
    path_mode_resolved = compute_linewidth_path(
        epc_path_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="lorentzian",
        mode_resolved=True,
    )

    assert path_linewidth.linewidth.shape == (2, 1, 2)
    assert path_mode_resolved.linewidth.shape == (2, 1, 2, 2)
    np.testing.assert_allclose(path_linewidth.linewidth.sum(axis=0), total.linewidth)
    np.testing.assert_allclose(path_mode_resolved.linewidth.sum(axis=-1), path_linewidth.linewidth)
    np.testing.assert_allclose(path_linewidth.path_coordinates, epc_path_data.path_coordinates)
    np.testing.assert_array_equal(path_linewidth.path_segments, epc_path_data.path_segments)
    assert path_linewidth.metadata["schema"] == "deeptb.epc_path_linewidth"
    assert path_linewidth.metadata["schema_version"] == LINEWIDTH_PATH_NPZ_SCHEMA_VERSION
    assert path_linewidth.metadata["aggregation"] == "per_path_point_contribution"


def test_compute_linewidth_mesh_matches_total_for_uniform_q_weights():
    epc_mesh_data = _small_linewidth_epc_mesh_data()
    total = compute_linewidth(
        epc_mesh_data.epc_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="lorentzian",
        mode_resolved=False,
    )
    mesh_linewidth = compute_linewidth_mesh(
        epc_mesh_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="lorentzian",
        mode_resolved=False,
    )
    mesh_mode_resolved = compute_linewidth_mesh(
        epc_mesh_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="lorentzian",
        mode_resolved=True,
    )

    np.testing.assert_allclose(mesh_linewidth.linewidth, total.linewidth)
    np.testing.assert_allclose(mesh_mode_resolved.linewidth.sum(axis=-1), mesh_linewidth.linewidth)
    np.testing.assert_allclose(mesh_linewidth.kpoint_weights, epc_mesh_data.kpoint_weights)
    np.testing.assert_array_equal(mesh_linewidth.band_indices, epc_mesh_data.band_indices)
    assert mesh_linewidth.metadata["schema"] == "deeptb.epc_mesh_linewidth"
    assert mesh_linewidth.metadata["schema_version"] == LINEWIDTH_MESH_NPZ_SCHEMA_VERSION
    assert mesh_linewidth.metadata["q_weight_convention"] == "normalized_sum_to_one"


def test_linewidth_mesh_data_npz_roundtrip(tmp_path):
    linewidth = LinewidthMeshData(
        linewidth=np.array([[0.01, 0.02], [0.03, 0.04]]),
        absorption=np.array([[0.004, 0.008], [0.012, 0.016]]),
        emission=np.array([[0.006, 0.012], [0.018, 0.024]]),
        kpoints=np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
        kpoint_weights=np.array([1.0, 1.0]),
        band_indices=np.array([0, 1]),
        metadata={"source": "unit-test"},
    )
    path = tmp_path / "mesh_linewidth.npz"
    linewidth.save_npz(path)
    loaded = LinewidthMeshData.load_npz(path)

    np.testing.assert_allclose(loaded.linewidth, linewidth.linewidth)
    np.testing.assert_allclose(loaded.absorption, linewidth.absorption)
    np.testing.assert_allclose(loaded.emission, linewidth.emission)
    np.testing.assert_allclose(loaded.kpoints, linewidth.kpoints)
    np.testing.assert_allclose(loaded.kpoint_weights, np.array([0.5, 0.5]))
    np.testing.assert_array_equal(loaded.band_indices, linewidth.band_indices)
    assert loaded.metadata["schema"] == "deeptb.epc_mesh_linewidth"
    assert loaded.metadata["schema_version"] == LINEWIDTH_MESH_NPZ_SCHEMA_VERSION

    with np.load(path, allow_pickle=False) as data:
        assert "elph_mesh_linewidth" in data
        assert "el_kpoint_weights" in data


def test_linewidth_path_data_npz_roundtrip(tmp_path):
    linewidth = LinewidthPathData(
        linewidth=np.array([[[0.01, 0.02]], [[0.03, 0.04]]]),
        absorption=np.array([[[0.004, 0.008]], [[0.012, 0.016]]]),
        emission=np.array([[[0.006, 0.012]], [[0.018, 0.024]]]),
        path_axis="q",
        path_coordinates=np.array([0.0, 0.25]),
        path_segments=np.array([[0, 2]]),
        band_indices=np.array([0, 1]),
        metadata={"source": "unit-test"},
    )
    path = tmp_path / "path_linewidth.npz"
    linewidth.save_npz(path)
    loaded = LinewidthPathData.load_npz(path)

    np.testing.assert_allclose(loaded.linewidth, linewidth.linewidth)
    np.testing.assert_allclose(loaded.absorption, linewidth.absorption)
    np.testing.assert_allclose(loaded.emission, linewidth.emission)
    np.testing.assert_allclose(loaded.path_coordinates, linewidth.path_coordinates)
    np.testing.assert_array_equal(loaded.path_segments, linewidth.path_segments)
    np.testing.assert_array_equal(loaded.band_indices, linewidth.band_indices)
    assert loaded.metadata["schema"] == "deeptb.epc_path_linewidth"
    assert loaded.metadata["schema_version"] == LINEWIDTH_PATH_NPZ_SCHEMA_VERSION

    with np.load(path, allow_pickle=False) as data:
        assert "elph_path_linewidth" in data
        assert "path_coordinates" in data
        assert "path_segments" in data


def test_compute_linewidth_rejects_invalid_parameters():
    epc_data = _small_linewidth_epc_data()
    with pytest.raises(ValueError, match="chemical_potential"):
        compute_linewidth(epc_data, chemical_potential=np.nan, temperature=0.01, sigma=0.01)
    with pytest.raises(ValueError, match="chemical_potential"):
        compute_linewidth(epc_data, chemical_potential="0.0", temperature=0.01, sigma=0.01)
    with pytest.raises(ValueError, match="temperature"):
        compute_linewidth(epc_data, chemical_potential=0.0, temperature=0.0, sigma=0.01)
    with pytest.raises(ValueError, match="temperature"):
        compute_linewidth(epc_data, chemical_potential=0.0, temperature=np.nan, sigma=0.01)
    with pytest.raises(ValueError, match="temperature"):
        compute_linewidth(epc_data, chemical_potential=0.0, temperature=np.array([0.01]), sigma=0.01)
    with pytest.raises(ValueError, match="sigma"):
        compute_linewidth(epc_data, chemical_potential=0.0, temperature=0.01, sigma=0.0)
    with pytest.raises(ValueError, match="sigma"):
        compute_linewidth(epc_data, chemical_potential=0.0, temperature=0.01, sigma=np.nan)
    with pytest.raises(ValueError, match="broadening"):
        compute_linewidth(
            epc_data,
            chemical_potential=0.0,
            temperature=0.01,
            sigma=0.01,
            broadening="triangle",
        )
    with pytest.raises(ValueError, match="broadening"):
        compute_linewidth(
            epc_data,
            chemical_potential=0.0,
            temperature=0.01,
            sigma=0.01,
            broadening=None,
        )
    with pytest.raises(ValueError, match="frequency_floor"):
        compute_linewidth(epc_data, chemical_potential=0.0, temperature=0.01, sigma=0.01, frequency_floor=0.0)
    with pytest.raises(ValueError, match="frequency_floor"):
        compute_linewidth(epc_data, chemical_potential=0.0, temperature=0.01, sigma=0.01, frequency_floor=np.nan)
    with pytest.raises(ValueError, match="frequency_floor"):
        compute_linewidth(epc_data, chemical_potential=0.0, temperature=0.01, sigma=0.01, frequency_floor="1e-5")
    with pytest.raises(ValueError, match="mode_resolved"):
        compute_linewidth(
            epc_data,
            chemical_potential=0.0,
            temperature=0.01,
            sigma=0.01,
            mode_resolved=1,
        )


def test_compute_linewidth_rejects_negative_frequencies_and_floors_zero_modes():
    epc_data = _small_linewidth_epc_data()
    epc_data.frequencies[0, 0] = 0.0
    linewidth = compute_linewidth(
        epc_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        frequency_floor=1e-4,
    )
    assert np.isfinite(linewidth.linewidth).all()
    assert linewidth.metadata["frequency_floor"] == 1e-4

    negative_frequency_epc = _small_linewidth_epc_data()
    negative_frequency_epc.frequencies[0, 0] = -1e-3
    with pytest.raises(ValueError, match="non-negative"):
        compute_linewidth(
            negative_frequency_epc,
            chemical_potential=0.15,
            temperature=0.025,
            sigma=0.01,
        )


def test_linewidth_data_npz_roundtrip(tmp_path):
    linewidth = LinewidthData(
        linewidth=np.array([[1.0, 2.0], [3.0, 4.0]]),
        absorption=np.array([[0.25, 0.5], [0.75, 1.0]]),
        emission=np.array([[0.75, 1.5], [2.25, 3.0]]),
        metadata={"chemical_potential": 0.1, "linewidth_unit": "eV"},
    )
    path = tmp_path / "linewidth.npz"
    linewidth.save_npz(path)
    loaded = LinewidthData.load_npz(path)

    np.testing.assert_allclose(loaded.linewidth, linewidth.linewidth)
    np.testing.assert_allclose(loaded.absorption, linewidth.absorption)
    np.testing.assert_allclose(loaded.emission, linewidth.emission)
    assert loaded.metadata["schema"] == "deeptb.epc_linewidth"
    assert loaded.metadata["schema_version"] == LINEWIDTH_NPZ_SCHEMA_VERSION
    assert loaded.metadata["linewidth_unit"] == "eV"

    with np.load(path, allow_pickle=False) as data:
        assert "elph_linewidth" in data
        assert "elph_linewidth_absorption" in data
        assert "elph_linewidth_emission" in data
        assert "metadata_json" in data


def test_linewidth_data_rejects_invalid_schema_and_shapes():
    with pytest.raises(ValueError, match="absorption"):
        LinewidthData(
            linewidth=np.ones((1, 2)),
            absorption=np.ones((1, 1)),
            emission=np.ones((1, 2)),
        )
    with pytest.raises(ValueError, match="linewidth_unit"):
        LinewidthData(
            linewidth=np.ones((1, 2)) * 2.0,
            absorption=np.ones((1, 2)),
            emission=np.ones((1, 2)),
            metadata={"linewidth_unit": "meV"},
        )
    with pytest.raises(ValueError, match="finite"):
        LinewidthData(
            linewidth=np.array([[np.nan]]),
            absorption=np.zeros((1, 1)),
            emission=np.zeros((1, 1)),
        )
    with pytest.raises(ValueError, match="non-empty"):
        LinewidthData(
            linewidth=np.empty((0, 1)),
            absorption=np.empty((0, 1)),
            emission=np.empty((0, 1)),
        )
    with pytest.raises(ValueError, match="shape"):
        LinewidthData(
            linewidth=np.array([0.1, 0.2]),
            absorption=np.zeros(2),
            emission=np.array([0.1, 0.2]),
        )
    with pytest.raises(ValueError, match="absorption"):
        LinewidthData(
            linewidth=np.array([[0.1]]),
            absorption=np.array([[-0.1]]),
            emission=np.array([[0.2]]),
        )
    with pytest.raises(ValueError, match="absorption \\+ emission"):
        LinewidthData(
            linewidth=np.array([[0.3]]),
            absorption=np.array([[0.1]]),
            emission=np.array([[0.1]]),
        )


def test_compute_relaxation_time_matches_linewidth_convention():
    linewidth = LinewidthData(
        linewidth=np.array([[0.01, 0.02], [0.04, 0.08]]),
        absorption=np.zeros((2, 2)),
        emission=np.array([[0.01, 0.02], [0.04, 0.08]]),
    )

    relaxation_time = compute_relaxation_time(linewidth)

    np.testing.assert_allclose(relaxation_time.relaxation_time, HBAR_EV_S / (2.0 * linewidth.linewidth))
    assert relaxation_time.metadata["schema"] == "deeptb.epc_relaxation_time"
    assert relaxation_time.metadata["schema_version"] == RELAXATION_TIME_NPZ_SCHEMA_VERSION
    assert relaxation_time.metadata["relaxation_time_unit"] == "s"
    assert relaxation_time.metadata["convention"] == "hbar_over_2linewidth"
    assert relaxation_time.metadata["sum_modes"] is False


def test_compute_relaxation_time_preserves_or_sums_mode_axis():
    linewidth = LinewidthData(
        linewidth=np.array([[[0.01, 0.03], [0.02, 0.06]]]),
        absorption=np.zeros((1, 2, 2)),
        emission=np.array([[[0.01, 0.03], [0.02, 0.06]]]),
    )

    mode_resolved = compute_relaxation_time(linewidth)
    summed = compute_relaxation_time(linewidth, sum_modes=True)

    np.testing.assert_allclose(mode_resolved.relaxation_time, HBAR_EV_S / (2.0 * linewidth.linewidth))
    np.testing.assert_allclose(summed.relaxation_time, HBAR_EV_S / (2.0 * linewidth.linewidth.sum(axis=-1)))
    assert mode_resolved.relaxation_time.shape == (1, 2, 2)
    assert summed.relaxation_time.shape == (1, 2)
    assert summed.metadata["sum_modes"] is True


def test_compute_relaxation_time_path_preserves_path_metadata_and_sums_modes():
    linewidth = LinewidthPathData(
        linewidth=np.array([[[[0.01, 0.03], [0.02, 0.06]]], [[[0.04, 0.08], [0.05, 0.10]]]]),
        absorption=np.zeros((2, 1, 2, 2)),
        emission=np.array([[[[0.01, 0.03], [0.02, 0.06]]], [[[0.04, 0.08], [0.05, 0.10]]]]),
        path_axis="q",
        path_coordinates=np.array([0.0, 0.25]),
        path_segments=np.array([[0, 2]]),
        band_indices=np.array([0, 1]),
    )

    mode_resolved = compute_relaxation_time_path(linewidth)
    summed = compute_relaxation_time_path(linewidth, sum_modes=True)

    np.testing.assert_allclose(mode_resolved.relaxation_time, HBAR_EV_S / (2.0 * linewidth.linewidth))
    np.testing.assert_allclose(summed.relaxation_time, HBAR_EV_S / (2.0 * linewidth.linewidth.sum(axis=-1)))
    np.testing.assert_allclose(summed.path_coordinates, linewidth.path_coordinates)
    np.testing.assert_array_equal(summed.path_segments, linewidth.path_segments)
    assert mode_resolved.relaxation_time.shape == (2, 1, 2, 2)
    assert summed.relaxation_time.shape == (2, 1, 2)
    assert summed.metadata["schema"] == "deeptb.epc_path_relaxation_time"
    assert summed.metadata["schema_version"] == RELAXATION_TIME_PATH_NPZ_SCHEMA_VERSION
    assert summed.metadata["sum_modes"] is True


def test_compute_relaxation_time_mesh_preserves_mesh_metadata_and_sums_modes():
    linewidth = LinewidthMeshData(
        linewidth=np.array([[[0.01, 0.03], [0.02, 0.06]], [[0.04, 0.08], [0.05, 0.10]]]),
        absorption=np.zeros((2, 2, 2)),
        emission=np.array([[[0.01, 0.03], [0.02, 0.06]], [[0.04, 0.08], [0.05, 0.10]]]),
        kpoints=np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
        kpoint_weights=np.array([1.0, 1.0]),
        band_indices=np.array([0, 1]),
        metadata={"mesh_spec": {"k_mesh": [2, 1, 1]}},
    )

    mode_resolved = compute_relaxation_time_mesh(linewidth)
    summed = compute_relaxation_time_mesh(linewidth, sum_modes=True)

    np.testing.assert_allclose(mode_resolved.relaxation_time, HBAR_EV_S / (2.0 * linewidth.linewidth))
    np.testing.assert_allclose(summed.relaxation_time, HBAR_EV_S / (2.0 * linewidth.linewidth.sum(axis=-1)))
    np.testing.assert_allclose(summed.kpoint_weights, np.array([0.5, 0.5]))
    assert mode_resolved.relaxation_time.shape == (2, 2, 2)
    assert summed.relaxation_time.shape == (2, 2)
    assert summed.metadata["schema"] == "deeptb.epc_mesh_relaxation_time"
    assert summed.metadata["schema_version"] == RELAXATION_TIME_MESH_NPZ_SCHEMA_VERSION
    assert summed.metadata["sum_modes"] is True


def test_relaxation_time_mesh_data_npz_roundtrip(tmp_path):
    relaxation_time = RelaxationTimeMeshData(
        relaxation_time=np.array([[1e-13, 2e-13], [3e-13, 4e-13]]),
        kpoints=np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
        kpoint_weights=np.array([1.0, 1.0]),
        band_indices=np.array([0, 1]),
        metadata={"source": "unit-test"},
    )
    path = tmp_path / "mesh_relaxation_time.npz"
    relaxation_time.save_npz(path)
    loaded = RelaxationTimeMeshData.load_npz(path)

    np.testing.assert_allclose(loaded.relaxation_time, relaxation_time.relaxation_time)
    np.testing.assert_allclose(loaded.kpoints, relaxation_time.kpoints)
    np.testing.assert_allclose(loaded.kpoint_weights, np.array([0.5, 0.5]))
    np.testing.assert_array_equal(loaded.band_indices, relaxation_time.band_indices)
    assert loaded.metadata["schema"] == "deeptb.epc_mesh_relaxation_time"
    assert loaded.metadata["schema_version"] == RELAXATION_TIME_MESH_NPZ_SCHEMA_VERSION
    assert loaded.metadata["source"] == "unit-test"


def test_relaxation_time_path_data_npz_roundtrip(tmp_path):
    relaxation_time = RelaxationTimePathData(
        relaxation_time=np.array([[[1e-13, 2e-13]], [[3e-13, 4e-13]]]),
        path_axis="q",
        path_coordinates=np.array([0.0, 0.25]),
        path_segments=np.array([[0, 2]]),
        band_indices=np.array([0, 1]),
        metadata={"source": "unit-test"},
    )
    path = tmp_path / "path_relaxation_time.npz"
    relaxation_time.save_npz(path)
    loaded = RelaxationTimePathData.load_npz(path)

    np.testing.assert_allclose(loaded.relaxation_time, relaxation_time.relaxation_time)
    np.testing.assert_allclose(loaded.path_coordinates, relaxation_time.path_coordinates)
    np.testing.assert_array_equal(loaded.path_segments, relaxation_time.path_segments)
    np.testing.assert_array_equal(loaded.band_indices, relaxation_time.band_indices)
    assert loaded.metadata["schema"] == "deeptb.epc_path_relaxation_time"
    assert loaded.metadata["schema_version"] == RELAXATION_TIME_PATH_NPZ_SCHEMA_VERSION
    assert loaded.metadata["source"] == "unit-test"


def test_relaxation_time_data_npz_roundtrip(tmp_path):
    relaxation_time = RelaxationTimeData(
        relaxation_time=np.array([[1e-13, 2e-13]]),
        metadata={"source": "unit-test"},
    )
    path = tmp_path / "relaxation_time.npz"
    relaxation_time.save_npz(path)
    loaded = RelaxationTimeData.load_npz(path)

    np.testing.assert_allclose(loaded.relaxation_time, relaxation_time.relaxation_time)
    assert loaded.metadata["schema"] == "deeptb.epc_relaxation_time"
    assert loaded.metadata["schema_version"] == RELAXATION_TIME_NPZ_SCHEMA_VERSION
    assert loaded.metadata["source"] == "unit-test"

    with np.load(path, allow_pickle=False) as data:
        assert "elph_relaxation_time" in data
        assert "metadata_json" in data


def test_compute_relaxation_time_rejects_nonpositive_linewidth():
    linewidth = LinewidthData(
        linewidth=np.array([[0.0, 0.01]]),
        absorption=np.zeros((1, 2)),
        emission=np.array([[0.0, 0.01]]),
    )
    with pytest.raises(ValueError, match="positive"):
        compute_relaxation_time(linewidth)
    with pytest.raises(ValueError, match="mode-resolved"):
        compute_relaxation_time(
            LinewidthData(
                linewidth=np.array([[0.01, 0.02]]),
                absorption=np.zeros((1, 2)),
                emission=np.array([[0.01, 0.02]]),
            ),
            sum_modes=True,
        )
    with pytest.raises(ValueError, match="sum_modes"):
        compute_relaxation_time(linewidth, sum_modes=1)
    with pytest.raises(ValueError, match="relaxation_time"):
        RelaxationTimeData(relaxation_time=np.array([[np.inf]]))
    with pytest.raises(ValueError, match="non-empty"):
        RelaxationTimeData(relaxation_time=np.empty((0, 1)))
    with pytest.raises(ValueError, match="shape"):
        RelaxationTimeData(relaxation_time=np.array([1e-13, 2e-13]))
    with pytest.raises(ValueError, match="relaxation_time_unit"):
        RelaxationTimeData(
            relaxation_time=np.array([[1e-13]]),
            metadata={"relaxation_time_unit": "fs"},
        )
    with pytest.raises(ValueError, match="convention"):
        RelaxationTimeData(
            relaxation_time=np.array([[1e-13]]),
            metadata={"convention": "hbar_over_linewidth"},
        )


def test_find_degenerate_band_groups_preserves_band_order():
    groups = find_degenerate_band_groups(np.array([0.0, 1e-6, 0.2, 0.200001, 0.5]), tolerance=1e-5)

    assert [group.tolist() for group in groups] == [[0, 1], [2, 3], [4]]

    with pytest.raises(ValueError, match="one-dimensional"):
        find_degenerate_band_groups(np.ones((1, 2)))
    with pytest.raises(ValueError, match="tolerance"):
        find_degenerate_band_groups(np.array([0.0]), tolerance=-1.0)
    with pytest.raises(ValueError, match="tolerance"):
        find_degenerate_band_groups(np.array([0.0]), tolerance=np.nan)
    with pytest.raises(ValueError, match="tolerance"):
        find_degenerate_band_groups(np.array([0.0]), tolerance=np.array([1e-5]))


def test_subspace_coupling_strength_is_degenerate_gauge_invariant():
    coupling = np.array(
        [
            [0.2 + 0.1j, -0.3j, 0.4],
            [0.5, -0.1 + 0.2j, 0.7j],
            [0.6j, 0.3, -0.2],
        ],
        dtype=complex,
    )
    theta = 0.37
    final_rotation = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]],
        dtype=complex,
    )
    initial_rotation = np.array(
        [[1.0, 0.0], [0.0, np.exp(0.4j)]],
        dtype=complex,
    )

    transformed = coupling.copy()
    transformed[:2, :] = final_rotation.conj().T @ transformed[:2, :]
    transformed[:, :2] = transformed[:, :2] @ initial_rotation

    final_groups = [np.array([0, 1]), np.array([2])]
    initial_groups = [np.array([0, 1]), np.array([2])]
    reference = compute_subspace_coupling_strength(coupling, final_groups, initial_groups)
    rotated = compute_subspace_coupling_strength(transformed, final_groups, initial_groups)

    np.testing.assert_allclose(rotated, reference, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(reference[0, 0], np.sum(np.abs(coupling[:2, :2]) ** 2))


def test_subspace_coupling_strength_rejects_invalid_groups():
    coupling = np.ones((2, 2), dtype=complex)
    with pytest.raises(ValueError, match="coupling_matrix"):
        compute_subspace_coupling_strength(np.array([[np.nan]]), [np.array([0])], [np.array([0])])
    with pytest.raises(ValueError, match="at least one"):
        compute_subspace_coupling_strength(coupling, [], [np.array([0])])
    with pytest.raises(ValueError, match="final_groups"):
        compute_subspace_coupling_strength(coupling, [np.array([2])], [np.array([0])])
    with pytest.raises(ValueError, match="initial_groups"):
        compute_subspace_coupling_strength(coupling, [np.array([0])], [np.array([])])
    with pytest.raises(ValueError, match="final_groups"):
        compute_subspace_coupling_strength(coupling, [np.array([0.5])], [np.array([0])])
    with pytest.raises(ValueError, match="duplicate"):
        compute_subspace_coupling_strength(coupling, [np.array([0, 0])], [np.array([0])])


def test_subspace_coupling_data_from_epc_and_npz_roundtrip(tmp_path):
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        band_indices=np.array([0, 1, 2]),
        frequencies=np.array([[1.0]]),
        eigenvalues_k=np.array([[0.0, 0.0, 1.0]]),
        eigenvalues_kq=np.array([[[0.0, 0.0, 1.0]]]),
        coupling_matrix=np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]], dtype=complex),
        coupling_strength=np.ones((1, 1, 1, 3, 3)),
    )

    data = compute_subspace_coupling_data(
        epc_data,
        final_groups=[np.array([0, 1]), np.array([2])],
        initial_groups=[np.array([0, 1]), np.array([2])],
    )
    expected = np.array([[[[[1.0**2 + 2.0**2 + 4.0**2 + 5.0**2, 3.0**2 + 6.0**2], [7.0**2 + 8.0**2, 9.0**2]]]]])
    np.testing.assert_allclose(data.strength, expected)
    np.testing.assert_array_equal(data.final_group_bounds, np.array([[0, 2], [2, 3]]))
    assert data.metadata["schema"] == "deeptb.epc_subspace_coupling"
    assert data.metadata["schema_version"] == SUBSPACE_COUPLING_NPZ_SCHEMA_VERSION

    path = tmp_path / "subspace_coupling.npz"
    data.save_npz(path)
    loaded = SubspaceCouplingData.load_npz(path)
    np.testing.assert_allclose(loaded.strength, data.strength)
    np.testing.assert_array_equal(loaded.initial_group_bounds, data.initial_group_bounds)
    assert loaded.metadata["aggregation"] == "frobenius_norm_squared"

    with np.load(path, allow_pickle=False) as npz:
        assert "elph_subspace_strength" in npz
        assert "final_group_bounds" in npz
        assert "metadata_json" in npz


def test_subspace_coupling_data_rejects_noncontiguous_persistent_groups():
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        band_indices=np.array([0, 1, 2]),
        frequencies=np.array([[1.0]]),
        eigenvalues_k=np.array([[0.0, 0.1, 0.2]]),
        eigenvalues_kq=np.array([[[0.0, 0.1, 0.2]]]),
        coupling_matrix=np.ones((1, 1, 1, 3, 3), dtype=complex),
        coupling_strength=np.ones((1, 1, 1, 3, 3)),
    )
    with pytest.raises(ValueError, match="at least one"):
        compute_subspace_coupling_data(epc_data, final_groups=[])
    with pytest.raises(ValueError, match="contiguous"):
        compute_subspace_coupling_data(epc_data, final_groups=[np.array([0, 2])])
    with pytest.raises(ValueError, match="final_groups"):
        compute_subspace_coupling_data(epc_data, final_groups=[np.array([0.5])])
    with pytest.raises(ValueError, match="strength"):
        SubspaceCouplingData(
            strength=np.ones((1, 3)),
            final_group_bounds=np.array([[0, 1], [1, 2]]),
            initial_group_bounds=np.array([[0, 1], [1, 2]]),
        )
    with pytest.raises(ValueError, match="strength"):
        SubspaceCouplingData(
            strength=np.array([[np.nan]]),
            final_group_bounds=np.array([[0, 1]]),
            initial_group_bounds=np.array([[0, 1]]),
        )
    with pytest.raises(ValueError, match="strength"):
        SubspaceCouplingData(
            strength=np.array([[-1.0]]),
            final_group_bounds=np.array([[0, 1]]),
            initial_group_bounds=np.array([[0, 1]]),
        )
    with pytest.raises(ValueError, match="final_group_bounds"):
        SubspaceCouplingData(
            strength=np.array([[1.0]]),
            final_group_bounds=np.array([[0.0, 1.5]]),
            initial_group_bounds=np.array([[0, 1]]),
        )
    with pytest.raises(ValueError, match="at least one"):
        SubspaceCouplingData(
            strength=np.ones((0, 1)),
            final_group_bounds=np.empty((0, 2), dtype=int),
            initial_group_bounds=np.array([[0, 1]]),
        )
    with pytest.raises(ValueError, match="aggregation"):
        SubspaceCouplingData(
            strength=np.array([[1.0]]),
            final_group_bounds=np.array([[0, 1]]),
            initial_group_bounds=np.array([[0, 1]]),
            metadata={"aggregation": "sum"},
        )
    with pytest.raises(ValueError, match="coupling_strength_unit"):
        SubspaceCouplingData(
            strength=np.array([[1.0]]),
            final_group_bounds=np.array([[0, 1]]),
            initial_group_bounds=np.array([[0, 1]]),
            metadata={"coupling_strength_unit": "meV^2"},
        )


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


def test_compute_serta_conductivity_matches_manual_reference():
    eigenvalues = np.array([[0.1, 0.2], [0.3, 0.4]])
    velocities = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, 3.0], [1.0, 1.0, 0.0]],
        ]
    )
    linewidth = np.array([[0.01, 0.02], [0.03, 0.04]])
    weights = np.array([2.0, 1.0])
    result = compute_serta_conductivity(
        eigenvalues=eigenvalues,
        velocities=velocities,
        linewidth=linewidth,
        chemical_potential=0.25,
        temperature=0.03,
        kpoint_weights=weights,
        spin_degeneracy=2,
        volume=5.0,
    )
    expected_conductivity, expected_density = _manual_serta_conductivity(
        eigenvalues,
        velocities,
        linewidth,
        chemical_potential=0.25,
        temperature=0.03,
        kpoint_weights=weights,
        spin_degeneracy=2,
        volume=5.0,
    )

    np.testing.assert_allclose(result.conductivity, expected_conductivity)
    np.testing.assert_allclose(result.carrier_density, expected_density)
    assert result.metadata["method"] == "SERTA"
    assert result.metadata["linewidth_unit"] == "eV"


def test_compute_serta_conductivity_uses_uniform_weights_by_default():
    eigenvalues = np.array([[0.1], [0.2]])
    velocities = np.array([[[1.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]])
    linewidth = np.array([[0.01], [0.02]])
    result = compute_serta_conductivity(
        eigenvalues=eigenvalues,
        velocities=velocities,
        linewidth=linewidth,
        chemical_potential=0.15,
        temperature=0.03,
    )
    expected_conductivity, expected_density = _manual_serta_conductivity(
        eigenvalues,
        velocities,
        linewidth,
        chemical_potential=0.15,
        temperature=0.03,
        kpoint_weights=np.array([0.5, 0.5]),
        spin_degeneracy=1,
        volume=1.0,
    )

    np.testing.assert_allclose(result.conductivity, expected_conductivity)
    np.testing.assert_allclose(result.carrier_density, expected_density)


def test_compute_serta_conductivity_rejects_invalid_inputs():
    eigenvalues = np.array([[0.1]])
    velocities = np.array([[[1.0, 0.0, 0.0]]])
    linewidth = np.array([[0.01]])

    with pytest.raises(ValueError, match="temperature"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, chemical_potential=0.0, temperature=0.0)
    with pytest.raises(ValueError, match="chemical_potential"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, chemical_potential=np.nan, temperature=0.01)
    with pytest.raises(ValueError, match="chemical_potential"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, chemical_potential="0.0", temperature=0.01)
    with pytest.raises(ValueError, match="temperature"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, chemical_potential=0.0, temperature=np.nan)
    with pytest.raises(ValueError, match="temperature"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, chemical_potential=0.0, temperature=True)
    with pytest.raises(ValueError, match="volume"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, 0.0, 0.01, volume=np.nan)
    with pytest.raises(ValueError, match="volume"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, 0.0, 0.01, volume=np.array([1.0]))
    with pytest.raises(ValueError, match="spin_degeneracy"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, 0.0, 0.01, spin_degeneracy=np.nan)
    with pytest.raises(ValueError, match="spin_degeneracy"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, 0.0, 0.01, spin_degeneracy=1.5)
    with pytest.raises(ValueError, match="spin_degeneracy"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, 0.0, 0.01, spin_degeneracy=True)
    with pytest.raises(ValueError, match="spin_degeneracy"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, 0.0, 0.01, spin_degeneracy=0)
    with pytest.raises(ValueError, match="spin_degeneracy"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, 0.0, 0.01, spin_degeneracy="2")
    with pytest.raises(ValueError, match="spin_degeneracy"):
        compute_serta_conductivity(eigenvalues, velocities, linewidth, 0.0, 0.01, spin_degeneracy=np.array([2]))
    with pytest.raises(ValueError, match="linewidth"):
        compute_serta_conductivity(eigenvalues, velocities, np.array([[0.0]]), 0.0, 0.01)
    with pytest.raises(ValueError, match="eigenvalues"):
        compute_serta_conductivity(np.array([[np.nan]]), velocities, linewidth, 0.0, 0.01)
    with pytest.raises(ValueError, match="velocities"):
        compute_serta_conductivity(eigenvalues, np.array([[[np.nan, 0.0, 0.0]]]), linewidth, 0.0, 0.01)
    with pytest.raises(ValueError, match="linewidth"):
        compute_serta_conductivity(eigenvalues, velocities, np.array([[np.nan]]), 0.0, 0.01)
    with pytest.raises(ValueError, match="at least one k-point"):
        compute_serta_conductivity(
            np.empty((0, 1)),
            np.empty((0, 1, 3)),
            np.empty((0, 1)),
            0.0,
            0.01,
        )
    with pytest.raises(ValueError, match="at least one band"):
        compute_serta_conductivity(
            np.empty((1, 0)),
            np.empty((1, 0, 3)),
            np.empty((1, 0)),
            0.0,
            0.01,
        )
    with pytest.raises(ValueError, match="velocities"):
        compute_serta_conductivity(eigenvalues, np.ones((1, 3)), linewidth, 0.0, 0.01)
    with pytest.raises(ValueError, match="kpoint_weights"):
        compute_serta_conductivity(
            eigenvalues,
            velocities,
            linewidth,
            chemical_potential=0.0,
            temperature=0.01,
            kpoint_weights=np.array([1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="kpoint_weights"):
        compute_serta_conductivity(
            eigenvalues,
            velocities,
            linewidth,
            chemical_potential=0.0,
            temperature=0.01,
            kpoint_weights=np.array([-1.0]),
        )
    with pytest.raises(ValueError, match="kpoint_weights"):
        compute_serta_conductivity(
            eigenvalues,
            velocities,
            linewidth,
            chemical_potential=0.0,
            temperature=0.01,
            kpoint_weights=np.array([np.nan]),
        )


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


def test_compute_band_velocities_finite_difference_matches_linear_bands():
    system = _LinearBandSystem()
    kpoints = np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.2]])

    velocities = compute_band_velocities_finite_difference(
        system=system,
        kpoints=kpoints,
        bands=[0, 2],
        delta=1e-5,
    )

    expected = np.repeat(system.band_slopes[[0, 2]][None, :, :], kpoints.shape[0], axis=0)
    np.testing.assert_allclose(velocities, expected, atol=1e-11)


def test_compute_band_velocities_hamiltonian_derivative_matches_diagonal_dhdk():
    system = _DerivativeBandSystem()
    kpoints = np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01]])

    velocities = compute_band_velocities_hamiltonian_derivative(
        system=system,
        kpoints=kpoints,
        bands=[0, 2],
    )

    expected = np.repeat(system.band_slopes[[0, 2]][None, :, :], kpoints.shape[0], axis=0)
    np.testing.assert_allclose(velocities, expected)


def test_compute_band_velocities_hamiltonian_derivative_applies_overlap_correction():
    velocities = compute_band_velocities_hamiltonian_derivative(
        system=_OverlapDerivativeBandSystem(),
        kpoints=np.array([[0.0, 0.0, 0.0]]),
    )

    # Generalized eigenvalue is H/S = 1.0 at k=0; normalized eigenvector is 1/sqrt(2).
    # v = <dH> - E <dS> = 0.5 * 6.0 - 1.0 * 0.5 * 1.0.
    np.testing.assert_allclose(velocities, np.array([[[2.5, 0.0, 0.0]]]))


def test_compute_band_velocities_hamiltonian_derivative_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="solver_kwargs"):
        compute_band_velocities_hamiltonian_derivative(
            _DerivativeBandSystem(),
            np.array([[0.0, 0.0, 0.0]]),
            solver="lapack",
        )
    with pytest.raises(ValueError, match="bands"):
        compute_band_velocities_hamiltonian_derivative(
            _DerivativeBandSystem(),
            np.array([[0.0, 0.0, 0.0]]),
            bands=[3],
        )
    with pytest.raises(NotImplementedError, match="SCC-corrected"):
        compute_band_velocities_hamiltonian_derivative(
            _DerivativeBandSystem(),
            np.array([[0.0, 0.0, 0.0]]),
            use_scc=True,
        )


def test_compute_band_velocities_finite_difference_rejects_invalid_inputs():
    system = _LinearBandSystem()
    with pytest.raises(ValueError, match="delta"):
        compute_band_velocities_finite_difference(system, np.array([[0.0, 0.0, 0.0]]), delta=0.0)
    with pytest.raises(ValueError, match="delta"):
        compute_band_velocities_finite_difference(system, np.array([[0.0, 0.0, 0.0]]), delta=np.nan)
    with pytest.raises(ValueError, match="delta"):
        compute_band_velocities_finite_difference(system, np.array([[0.0, 0.0, 0.0]]), delta="1e-4")
    with pytest.raises(ValueError, match="delta"):
        compute_band_velocities_finite_difference(system, np.array([[0.0, 0.0, 0.0]]), delta=np.array([1e-4]))
    with pytest.raises(ValueError, match="bands"):
        compute_band_velocities_finite_difference(system, np.array([[0.0, 0.0, 0.0]]), bands=[3])
    with pytest.raises(ValueError, match="bands"):
        compute_band_velocities_finite_difference(system, np.array([[0.0, 0.0, 0.0]]), bands=[0.5])
    with pytest.raises(ValueError, match="finite"):
        compute_band_velocities_finite_difference(_NonfiniteBandSystem(), np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="consistent eigenvalue shape"):
        compute_band_velocities_finite_difference(_ShapeChangingBandSystem(), np.array([[0.0, 0.0, 0.0]]), bands=[0])
    with pytest.raises(ValueError, match="at least one band"):
        compute_band_velocities_finite_difference(_EmptyBandSystem(), np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(NotImplementedError, match="SCC-corrected"):
        compute_band_velocities_finite_difference(system, np.array([[0.0, 0.0, 0.0]]), use_scc=True)


def test_compute_serta_transport_from_epc_uses_epc_bands_and_linewidth():
    system = _LinearBandSystem()
    kpoints = np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.2]])
    band_indices = np.array([0, 2])
    eigenvalues = system.band_offsets[None, band_indices] + kpoints @ system.band_slopes[band_indices].T
    epc_data = EPCData(
        kpoints=kpoints,
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        band_indices=band_indices,
        frequencies=np.array([[1.0]]),
        eigenvalues_k=eigenvalues,
        eigenvalues_kq=eigenvalues[None, :, :],
        coupling_matrix=np.ones((1, 2, 1, 2, 2), dtype=complex),
        coupling_strength=np.ones((1, 2, 1, 2, 2)),
    )
    linewidth = LinewidthData(
        linewidth=np.array([[[0.01], [0.02]], [[0.03], [0.04]]]),
        absorption=np.zeros((2, 2, 1)),
        emission=np.array([[[0.01], [0.02]], [[0.03], [0.04]]]),
    )

    result = compute_serta_transport_from_epc(
        system=system,
        epc_data=epc_data,
        linewidth_data=linewidth,
        chemical_potential=0.15,
        temperature=0.03,
        kpoint_weights=np.array([2.0, 1.0]),
        spin_degeneracy=2,
        volume=5.0,
        velocity_delta=np.array(1e-4),
    )
    expected = compute_serta_conductivity(
        eigenvalues=eigenvalues,
        velocities=np.repeat(system.band_slopes[band_indices][None, :, :], kpoints.shape[0], axis=0),
        linewidth=linewidth.linewidth.sum(axis=-1),
        chemical_potential=0.15,
        temperature=0.03,
        kpoint_weights=np.array([2.0, 1.0]),
        spin_degeneracy=2,
        volume=5.0,
    )

    np.testing.assert_allclose(result.conductivity, expected.conductivity)
    np.testing.assert_allclose(result.carrier_density, expected.carrier_density)
    assert result.metadata["velocity_source"] == "finite_difference"
    assert result.metadata["velocity_delta"] == 1e-4
    assert result.metadata["velocity_unit"] == "eV/fractional_reciprocal_coordinate"
    np.testing.assert_array_equal(result.metadata["band_indices"], band_indices)
    assert result.metadata["epc_schema"] == "deeptb.epc_data"


def test_compute_serta_transport_from_epc_uses_hamiltonian_derivative_velocity_source():
    system = _DerivativeBandSystem()
    kpoints = np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01]])
    band_indices = np.array([0, 2])
    eigenvalues = system.band_offsets[None, band_indices] + kpoints @ system.band_slopes[band_indices].T
    epc_data = EPCData(
        kpoints=kpoints,
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        band_indices=band_indices,
        frequencies=np.array([[1.0]]),
        eigenvalues_k=eigenvalues,
        eigenvalues_kq=eigenvalues[None, :, :],
        coupling_matrix=np.ones((1, 2, 1, 2, 2), dtype=complex),
        coupling_strength=np.ones((1, 2, 1, 2, 2)),
    )
    linewidth = LinewidthData(
        linewidth=np.array([[0.01, 0.02], [0.03, 0.04]]),
        absorption=np.zeros((2, 2)),
        emission=np.array([[0.01, 0.02], [0.03, 0.04]]),
    )

    result = compute_serta_transport_from_epc(
        system=system,
        epc_data=epc_data,
        linewidth_data=linewidth,
        chemical_potential=0.15,
        temperature=0.03,
        velocity_source="hamiltonian_derivative",
    )
    expected = compute_serta_conductivity(
        eigenvalues=eigenvalues,
        velocities=np.repeat(system.band_slopes[band_indices][None, :, :], kpoints.shape[0], axis=0),
        linewidth=linewidth.linewidth,
        chemical_potential=0.15,
        temperature=0.03,
    )

    np.testing.assert_allclose(result.conductivity, expected.conductivity)
    assert result.metadata["velocity_source"] == "hamiltonian_derivative"
    assert result.metadata["velocity_unit"] == "eV/fractional_reciprocal_coordinate"
    assert result.metadata["velocity_convention"] == "diag_Cdagger_dH_minus_EdS_C"
    assert result.metadata["overlap_correction"] == "dH_minus_E_dS_diagonal"


def test_compute_serta_transport_from_epc_rejects_invalid_velocity_delta():
    system = _LinearBandSystem()
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        band_indices=np.array([0]),
        frequencies=np.array([[1.0]]),
        eigenvalues_k=np.array([[0.1]]),
        eigenvalues_kq=np.array([[[0.1]]]),
        coupling_matrix=np.ones((1, 1, 1, 1, 1), dtype=complex),
        coupling_strength=np.ones((1, 1, 1, 1, 1)),
    )
    linewidth = LinewidthData(
        linewidth=np.array([[0.01]]),
        absorption=np.zeros((1, 1)),
        emission=np.array([[0.01]]),
    )

    with pytest.raises(ValueError, match="velocity_delta"):
        compute_serta_transport_from_epc(
            system,
            epc_data,
            linewidth,
            chemical_potential=0.0,
            temperature=0.01,
            velocity_delta="1e-4",
        )
    with pytest.raises(ValueError, match="velocity_source"):
        compute_serta_transport_from_epc(
            system,
            epc_data,
            linewidth,
            chemical_potential=0.0,
            temperature=0.01,
            velocity_source="unknown",
        )


def test_transport_data_npz_roundtrip(tmp_path):
    transport = TransportData(
        conductivity=np.array([[1.0, 0.1, 0.0], [0.1, 2.0, 0.2], [0.0, 0.2, 3.0]]),
        carrier_density=np.array(0.5),
        metadata={"volume": 10.0, "method": "SERTA"},
    )
    path = tmp_path / "transport.npz"
    transport.save_npz(path)
    loaded = TransportData.load_npz(path)

    np.testing.assert_allclose(loaded.conductivity, transport.conductivity)
    np.testing.assert_allclose(loaded.carrier_density, transport.carrier_density)
    assert loaded.metadata["schema"] == "deeptb.epc_transport"
    assert loaded.metadata["schema_version"] == TRANSPORT_NPZ_SCHEMA_VERSION
    assert loaded.metadata["method"] == "SERTA"

    with np.load(path, allow_pickle=False) as data:
        assert "transport_conductivity" in data
        assert "transport_carrier_density" in data
        assert "metadata_json" in data


def test_compute_serta_mobility_si_matches_manual_3d_reference():
    eigenvalues = np.array([[0.0]])
    velocities = np.array([[[1.0, 0.0, 0.0]]])
    linewidth = np.array([[0.01]])
    reciprocal_cell = np.eye(3)

    result = compute_serta_mobility_si(
        eigenvalues=eigenvalues,
        velocities=velocities,
        linewidth=linewidth,
        reciprocal_cell=reciprocal_cell,
        chemical_potential=0.0,
        temperature=0.1,
        spin_degeneracy=2,
        volume=10.0,
    )

    velocity_si = dptb_constants.ANGSTROM_TO_M / dptb_constants.HBAR_EV_S
    tau = dptb_constants.HBAR_EV_S / (2.0 * 0.01)
    density = 2.0 / (10.0 * dptb_constants.ANGSTROM_TO_M**3)
    occupation = 0.5
    transport_weight_ev = 0.25 / 0.1
    expected_carrier_density = density * occupation
    expected_sigma_xx = (
        density
        * (dptb_constants.ELECTRON_CHARGE_C**2 / dptb_constants.eV2J)
        * transport_weight_ev
        * tau
        * velocity_si**2
    )

    np.testing.assert_allclose(result.carrier_density, expected_carrier_density)
    np.testing.assert_allclose(result.conductivity[0, 0], expected_sigma_xx)
    np.testing.assert_allclose(
        result.mobility[0, 0],
        expected_sigma_xx / (expected_carrier_density * dptb_constants.ELECTRON_CHARGE_C),
    )
    assert result.metadata["conductivity_unit"] == "S/m"
    assert result.metadata["carrier_density_unit"] == "m^-3"
    assert result.metadata["mobility_unit"] == "m^2/V/s"
    assert result.metadata["volume_unit"] == "Angstrom^3"


def test_compute_serta_mobility_si_supports_2d_sheet_normalization():
    result = compute_serta_mobility_si(
        eigenvalues=np.array([[0.0]]),
        velocities=np.array([[[1.0, 0.0, 0.0]]]),
        linewidth=np.array([[0.01]]),
        reciprocal_cell=np.eye(3),
        chemical_potential=0.0,
        temperature=0.1,
        dimension="2d",
        area=5.0,
    )

    assert result.metadata["conductivity_unit"] == "S"
    assert result.metadata["carrier_density_unit"] == "m^-2"
    assert result.metadata["area_unit"] == "Angstrom^2"
    assert result.carrier_density.shape == ()


def test_mobility_data_npz_roundtrip(tmp_path):
    mobility = MobilityData(
        conductivity=np.eye(3),
        mobility=np.eye(3) * 2.0,
        carrier_density=np.array(1.5),
        metadata={"dimension": "3d"},
    )
    path = tmp_path / "mobility.npz"
    mobility.save_npz(path)
    loaded = MobilityData.load_npz(path)

    np.testing.assert_allclose(loaded.conductivity, mobility.conductivity)
    np.testing.assert_allclose(loaded.mobility, mobility.mobility)
    np.testing.assert_allclose(loaded.carrier_density, mobility.carrier_density)
    assert loaded.metadata["schema"] == "deeptb.epc_mobility"
    assert loaded.metadata["schema_version"] == MOBILITY_NPZ_SCHEMA_VERSION


def test_compute_serta_mobility_si_rejects_invalid_inputs():
    kwargs = {
        "eigenvalues": np.array([[0.0]]),
        "velocities": np.array([[[1.0, 0.0, 0.0]]]),
        "linewidth": np.array([[0.01]]),
        "reciprocal_cell": np.eye(3),
        "chemical_potential": 0.0,
        "temperature": 0.1,
        "volume": 10.0,
    }
    with pytest.raises(ValueError, match="dimension"):
        compute_serta_mobility_si(**kwargs, dimension="1d")
    with pytest.raises(ValueError, match="volume"):
        compute_serta_mobility_si(**{**kwargs, "volume": None})
    with pytest.raises(ValueError, match="volume"):
        compute_serta_mobility_si(**kwargs, dimension="2d", area=5.0)
    kwargs_2d = dict(kwargs)
    kwargs_2d.pop("volume")
    with pytest.raises(ValueError, match="area"):
        compute_serta_mobility_si(**kwargs_2d, dimension="2d")
    with pytest.raises(ValueError, match="reciprocal_cell"):
        compute_serta_mobility_si(**{**kwargs, "reciprocal_cell": np.zeros((3, 3))})
    with pytest.raises(ValueError, match="linewidth"):
        compute_serta_mobility_si(**{**kwargs, "linewidth": np.array([[0.0]])})


def test_transport_data_rejects_invalid_schema_and_shapes():
    with pytest.raises(ValueError, match="conductivity"):
        TransportData(
            conductivity=np.ones((2, 2)),
            carrier_density=np.array(1.0),
        )
    with pytest.raises(ValueError, match="method"):
        TransportData(
            conductivity=np.eye(3),
            carrier_density=np.array(1.0),
            metadata={"method": "RTA"},
        )
    with pytest.raises(ValueError, match="finite"):
        TransportData(
            conductivity=np.full((3, 3), np.nan),
            carrier_density=np.array(1.0),
        )
    with pytest.raises(ValueError, match="carrier_density"):
        TransportData(
            conductivity=np.eye(3),
            carrier_density=np.ones((1, 1)),
        )
    with pytest.raises(ValueError, match="carrier_density"):
        TransportData(
            conductivity=np.eye(3),
            carrier_density=np.array(-1.0),
        )
    with pytest.raises(ValueError, match="carrier_density"):
        TransportData(
            conductivity=np.eye(3),
            carrier_density=np.array([]),
        )


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


def test_electron_phonon_accessor_compute_coupling_with_mock_derivatives(tmp_path):
    phonons = _single_mode_phonons()
    accessor = EPhAccessor(_FakeSystem())
    output = tmp_path / "mock_epc.npz"

    epc_data = accessor.compute_coupling(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        phonons=phonons,
        bands=[0],
        derivative_provider=_FakeDerivativeProvider(),
        output_npz=output,
    )

    assert epc_data.coupling_matrix.shape == (1, 1, 1, 1, 1)
    assert epc_data.coupling_strength.shape == (1, 1, 1, 1, 1)
    assert epc_data.metadata["schema"] == "deeptb.epc_data"
    assert epc_data.metadata["source"] == "deeptb.eph.compute_coupling"
    assert epc_data.metadata["derivative_provider"] == "_FakeDerivativeProvider"
    assert epc_data.metadata["phonon_metadata"]["schema"] == "deeptb.phonons"
    assert output.exists()


def test_electron_phonon_accessor_compute_fixed_k_q_path(tmp_path):
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        frequencies=np.array([[1.0], [2.0]]),
        eigenvectors=np.array([[[[1.0, 0.0, 0.0]]], [[[1.0, 0.0, 0.0]]]], dtype=complex),
        masses=np.array([1.0]),
    )
    output = tmp_path / "mock_epc_path.npz"

    path_data = EPhAccessor(_FakeSystem()).compute_path(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        phonons=phonons,
        bands=[0],
        derivative_provider=_FakeDerivativeProvider(),
        path_segments=np.array([[0, 2]]),
        path_labels={"G": 0, "X": 1},
        output_npz=output,
    )
    loaded = EPCPathData.load_npz(output)

    assert path_data.path_axis == "q"
    assert path_data.metadata["source"] == "deeptb.eph.compute_path"
    assert path_data.metadata["path_mode"] == "fixed_k_q_path"
    assert path_data.metadata["base_epc_metadata"]["source"] == "deeptb.eph.compute_coupling"
    assert path_data.metadata["path_labels"] == {"G": 0, "X": 1}
    np.testing.assert_allclose(path_data.path_coordinates, np.array([0.0, 0.25]))
    np.testing.assert_allclose(loaded.coupling_strength, path_data.coupling_strength)
    np.testing.assert_array_equal(loaded.path_segments, np.array([[0, 2]]))


def test_electron_phonon_accessor_compute_mesh_with_generated_kmesh(tmp_path):
    phonons = _single_mode_phonons()
    output = tmp_path / "mock_epc_mesh.npz"

    mesh_data = EPhAccessor(_FakeSystem()).compute_mesh(
        mesh_spec=EPCMeshSpec(k_mesh=[2, 1, 1]),
        phonons=phonons,
        bands=[0],
        derivative_provider=_FakeDerivativeProvider(),
        output_npz=output,
    )
    loaded = EPCMeshData.load_npz(output)

    assert mesh_data.metadata["schema"] == "deeptb.epc_mesh_data"
    assert mesh_data.metadata["source"] == "deeptb.eph.compute_mesh"
    assert mesh_data.metadata["execution"] == "serial_full_mesh"
    assert mesh_data.kpoints.shape == (2, 3)
    assert mesh_data.coupling_strength.shape == (1, 2, 1, 1, 1)
    np.testing.assert_allclose(mesh_data.kpoint_weights, np.array([0.5, 0.5]))
    np.testing.assert_allclose(mesh_data.qpoint_weights, np.array([1.0]))
    np.testing.assert_allclose(loaded.coupling_strength, mesh_data.coupling_strength)


def test_electron_phonon_accessor_compute_mesh_chunked_matches_full_mesh():
    phonons = _single_mode_phonons()
    accessor = EPhAccessor(_FakeSystem())

    full = accessor.compute_mesh(
        mesh_spec=EPCMeshSpec(k_mesh=[3, 1, 1]),
        phonons=phonons,
        bands=[0],
        derivative_provider=_FakeDerivativeProvider(),
    )
    chunked = accessor.compute_mesh(
        mesh_spec=EPCMeshSpec(k_mesh=[3, 1, 1], chunk_size=1),
        phonons=phonons,
        bands=[0],
        derivative_provider=_FakeDerivativeProvider(),
    )

    np.testing.assert_allclose(chunked.kpoints, full.kpoints)
    np.testing.assert_allclose(chunked.kpoint_weights, full.kpoint_weights)
    np.testing.assert_allclose(chunked.eigenvalues_k, full.eigenvalues_k)
    np.testing.assert_allclose(chunked.eigenvalues_kq, full.eigenvalues_kq)
    np.testing.assert_allclose(chunked.coupling_matrix, full.coupling_matrix)
    np.testing.assert_allclose(chunked.coupling_strength, full.coupling_strength)
    assert full.metadata["chunked"] is False
    assert chunked.metadata["chunked"] is True
    assert chunked.metadata["execution"] == "serial_k_chunked"
    assert chunked.metadata["chunk_axis"] == "k"
    assert chunked.metadata["chunks"] == [
        {"chunk_index": 0, "k_start": 0, "k_stop": 1},
        {"chunk_index": 1, "k_start": 1, "k_stop": 2},
        {"chunk_index": 2, "k_start": 2, "k_stop": 3},
    ]


def test_epc_k_chunk_specs_are_deterministic():
    full = build_k_chunk_specs(3, None)
    assert full == [EPCKChunkSpec(chunk_index=0, k_start=0, k_stop=3)]
    assert full[0].slice == slice(0, 3)
    assert full[0].metadata() == {"chunk_index": 0, "k_start": 0, "k_stop": 3}

    chunks = build_k_chunk_specs(5, 2)
    assert chunks == [
        EPCKChunkSpec(chunk_index=0, k_start=0, k_stop=2),
        EPCKChunkSpec(chunk_index=1, k_start=2, k_stop=4),
        EPCKChunkSpec(chunk_index=2, k_start=4, k_stop=5),
    ]


@pytest.mark.parametrize(
    ("nk", "chunk_size", "match"),
    [
        (0, 1, "nk"),
        (3, 0, "chunk_size"),
        (3, True, "chunk_size"),
    ],
)
def test_epc_k_chunk_specs_reject_invalid_inputs(nk, chunk_size, match):
    with pytest.raises(ValueError, match=match):
        build_k_chunk_specs(nk, chunk_size)


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


def test_concat_epc_k_chunks_concatenates_k_axis():
    first = _epc_k_chunk([0.0, 0.5])
    second = _epc_k_chunk([1.0])

    combined = concat_epc_k_chunks([first, second])

    np.testing.assert_allclose(combined.kpoints[:, 0], np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(combined.eigenvalues_k[:, 0], np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(combined.eigenvalues_kq[0, :, 0], np.array([0.0, 0.5, 1.0]))
    assert combined.coupling_matrix.shape == (1, 3, 1, 1, 1)
    assert combined.metadata["chunk_count"] == 2
    assert len(combined.metadata["chunk_sources"]) == 2


@pytest.mark.parametrize(
    ("bad_chunk", "match"),
    [
        (_epc_k_chunk([1.0], qpoint_shift=0.25), "qpoints"),
        (_epc_k_chunk([1.0], band_indices=[1]), "band_indices"),
        (_epc_k_chunk([1.0], frequencies=np.array([[2.0]])), "phonon frequencies"),
    ],
)
def test_concat_epc_k_chunks_rejects_inconsistent_chunks(bad_chunk, match):
    with pytest.raises(ValueError, match=match):
        concat_epc_k_chunks([_epc_k_chunk([0.0]), bad_chunk])


def test_concat_epc_k_chunks_rejects_inconsistent_coupling_trailing_shape():
    bad_chunk = _epc_k_chunk([1.0])
    bad_chunk.coupling_matrix = np.ones((1, 1, 2, 1, 1), dtype=complex)
    bad_chunk.coupling_strength = np.ones((1, 1, 2, 1, 1))

    with pytest.raises(ValueError, match="coupling trailing shape"):
        concat_epc_k_chunks([_epc_k_chunk([0.0]), bad_chunk])


def test_concat_epc_k_chunks_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one EPC chunk"):
        concat_epc_k_chunks([])


def test_electron_phonon_accessor_compute_mesh_rejects_bad_spec():
    with pytest.raises(ValueError, match="EPCMeshSpec"):
        EPhAccessor(_FakeSystem()).compute_mesh(
            mesh_spec={"k_mesh": [1, 1, 1]},
            phonons=_single_mode_phonons(),
            derivative_provider=_FakeDerivativeProvider(),
        )


def test_electron_phonon_accessor_compute_path_rejects_future_k_axis():
    with pytest.raises(NotImplementedError, match="path_axis='q'"):
        EPhAccessor(_FakeSystem()).compute_path(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            phonons=_single_mode_phonons(),
            derivative_provider=_FakeDerivativeProvider(),
            path_axis="k",
        )


def test_electron_phonon_accessor_compute_path_rejects_bad_labels():
    with pytest.raises(ValueError, match="path_labels"):
        EPhAccessor(_FakeSystem()).compute_path(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            phonons=_single_mode_phonons(),
            derivative_provider=_FakeDerivativeProvider(),
            path_labels={"G": 2},
        )


def test_electron_phonon_accessor_rejects_scc_v1():
    phonons = _single_mode_phonons()

    with pytest.raises(NotImplementedError, match="SCC-corrected"):
        EPhAccessor(_FakeSystem()).compute_coupling(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            phonons=phonons,
            use_scc=True,
            derivative_provider=_FakeDerivativeProvider(),
        )


def test_electron_phonon_accessor_rejects_invalid_eigenstates():
    phonons = _single_mode_phonons()
    kpoints = np.array([[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="metadata, eigenvalues, eigenvectors"):
        EPhAccessor(_BadEigenstatePayloadSystem()).compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            derivative_provider=_FakeDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="finite values"):
        EPhAccessor(
            _BadEigenstateArraySystem(eigenvalues=np.array([[1.0, np.nan]]))
        ).compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            derivative_provider=_FakeDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="finite values"):
        bad_eigenvectors = np.tile(np.eye(2, dtype=complex)[None, :, :], (1, 1, 1))
        bad_eigenvectors[0, 0, 0] = np.nan
        EPhAccessor(_BadEigenstateArraySystem(eigenvectors=bad_eigenvectors)).compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            derivative_provider=_FakeDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="inconsistent k-point count"):
        EPhAccessor(
            _BadEigenstateArraySystem(eigenvalues=np.array([[1.0, 2.0], [3.0, 4.0]]))
        ).compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            derivative_provider=_FakeDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="at least one electronic band"):
        EPhAccessor(
            _BadEigenstateArraySystem(
                eigenvalues=np.empty((1, 0)),
                eigenvectors=np.empty((1, 2, 0), dtype=complex),
            )
        ).compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            derivative_provider=_FakeDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="at least one orbital"):
        EPhAccessor(
            _BadEigenstateArraySystem(
                eigenvalues=np.empty((1, 1)),
                eigenvectors=np.empty((1, 0, 1), dtype=complex),
            )
        ).compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            derivative_provider=_FakeDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="inconsistent orbital count"):
        EPhAccessor(_KqShapeChangingEigenstateSystem(kq_norb=3, kq_nbands=2)).compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            derivative_provider=_FakeDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="inconsistent band count"):
        EPhAccessor(_KqShapeChangingEigenstateSystem(kq_norb=2, kq_nbands=3)).compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            derivative_provider=_FakeDerivativeProvider(),
        )


def test_electron_phonon_accessor_rejects_invalid_displacement_and_derivatives():
    accessor = EPhAccessor(_FakeSystem())
    phonons = _single_mode_phonons()
    kpoints = np.array([[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="displacement"):
        accessor.compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            displacement=0.0,
            derivative_provider=_FakeDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="displacement"):
        accessor.compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            displacement=np.nan,
            derivative_provider=_FakeDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="displacement"):
        accessor.compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            displacement=True,
            derivative_provider=_FakeDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="finite"):
        accessor.compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            derivative_provider=_NonfiniteDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="inconsistent derivative shape"):
        accessor.compute_coupling(
            kpoints=kpoints,
            phonons=Phonons(
                qpoints=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
                frequencies=np.array([[1.0], [1.0]]),
                eigenvectors=np.array([[[[1.0, 0.0, 0.0]]], [[[1.0, 0.0, 0.0]]]], dtype=complex),
                masses=np.array([1.0]),
            ),
            derivative_provider=_ShapeChangingDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="overlap_derivatives"):
        accessor.compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            derivative_provider=_BadOverlapDerivativeProvider(),
        )
    with pytest.raises(ValueError, match="h_derivatives, overlap_derivatives"):
        accessor.compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            derivative_provider=_BadPayloadDerivativeProvider(),
        )


def test_fd_provider_rejects_invalid_displacement():
    with pytest.raises(ValueError, match="displacement"):
        FDProvider(_FakeSystem(), displacement=0.0)
    with pytest.raises(ValueError, match="displacement"):
        FDProvider(_FakeSystem(), displacement=np.nan)
    with pytest.raises(ValueError, match="displacement"):
        FDProvider(_FakeSystem(), displacement=np.array([1e-3]))


def test_eph_cli_parser_accepts_external_phonon_mode_inputs():
    args = parse_args(
        [
            "eph",
            "-i",
            "model.pth",
            "-stu",
            "struct.vasp",
            "-ph",
            "phonons.npz",
            "-k",
            "kpoints.json",
            "-b",
            "0",
            "1",
            "-o",
            "epc_data.npz",
        ]
    )

    assert args.command == "eph"
    assert args.init_model == "model.pth"
    assert args.structure == "struct.vasp"
    assert args.phonons == "phonons.npz"
    assert args.kpoints == "kpoints.json"
    assert args.bands == [0, 1]
    assert args.output == "epc_data.npz"


def test_eph_cli_parser_accepts_path_coupling_inputs():
    args = parse_args(
        [
            "eph",
            "--task",
            "path-coupling",
            "-i",
            "model.pth",
            "-stu",
            "struct.vasp",
            "-ph",
            "phonons.npz",
            "-k",
            "kpoints.json",
            "-b",
            "0",
            "-o",
            "epc_path_data.npz",
        ]
    )

    assert args.command == "eph"
    assert args.task == "path-coupling"
    assert args.init_model == "model.pth"
    assert args.structure == "struct.vasp"
    assert args.phonons == "phonons.npz"
    assert args.kpoints == "kpoints.json"
    assert args.bands == [0]
    assert args.output == "epc_path_data.npz"


def test_eph_cli_parser_accepts_mesh_coupling_inputs():
    args = parse_args(
        [
            "eph",
            "--task",
            "mesh-coupling",
            "-i",
            "model.pth",
            "-stu",
            "struct.vasp",
            "-ph",
            "phonons.npz",
            "--k-mesh",
            "2",
            "1",
            "1",
            "--q-mesh",
            "1",
            "1",
            "1",
            "--time-reversal",
            "--chunk-size",
            "1",
            "-b",
            "0",
            "-o",
            "epc_mesh_data.npz",
        ]
    )

    assert args.command == "eph"
    assert args.task == "mesh-coupling"
    assert args.k_mesh == [2, 1, 1]
    assert args.q_mesh == [1, 1, 1]
    assert args.time_reversal is True
    assert args.chunk_size == 1
    assert args.output == "epc_mesh_data.npz"


def test_eph_cli_parser_accepts_linewidth_postprocess_inputs():
    args = parse_args(
        [
            "eph",
            "--task",
            "linewidth",
            "--epc-data",
            "epc_data.npz",
            "--chemical-potential",
            "0.15",
            "--temperature",
            "0.025",
            "--sigma",
            "0.01",
            "--broadening",
            "lorentzian",
            "--mode-resolved",
            "-o",
            "linewidth.npz",
        ]
    )

    assert args.command == "eph"
    assert args.task == "linewidth"
    assert args.epc_data == "epc_data.npz"
    assert args.chemical_potential == 0.15
    assert args.temperature == 0.025
    assert args.sigma == 0.01
    assert args.broadening == "lorentzian"
    assert args.mode_resolved is True
    assert args.init_model is None
    assert args.structure is None


def test_eph_cli_parser_accepts_path_linewidth_postprocess_inputs():
    args = parse_args(
        [
            "eph",
            "--task",
            "path-linewidth",
            "--epc-data",
            "epc_path_data.npz",
            "--chemical-potential",
            "0.15",
            "--temperature",
            "0.025",
            "--sigma",
            "0.01",
            "--mode-resolved",
            "-o",
            "path_linewidth.npz",
        ]
    )

    assert args.command == "eph"
    assert args.task == "path-linewidth"
    assert args.epc_data == "epc_path_data.npz"
    assert args.mode_resolved is True
    assert args.output == "path_linewidth.npz"


def test_eph_cli_parser_accepts_mesh_linewidth_postprocess_inputs():
    args = parse_args(
        [
            "eph",
            "--task",
            "mesh-linewidth",
            "--epc-data",
            "epc_mesh_data.npz",
            "--chemical-potential",
            "0.15",
            "--temperature",
            "0.025",
            "--sigma",
            "0.01",
            "--mode-resolved",
            "-o",
            "mesh_linewidth.npz",
        ]
    )

    assert args.command == "eph"
    assert args.task == "mesh-linewidth"
    assert args.epc_data == "epc_mesh_data.npz"
    assert args.mode_resolved is True
    assert args.output == "mesh_linewidth.npz"


def test_eph_cli_parser_accepts_relaxation_time_postprocess_inputs():
    args = parse_args(
        [
            "eph",
            "--task",
            "relaxation-time",
            "--linewidth-data",
            "linewidth.npz",
            "--sum-modes",
            "-o",
            "relaxation_time.npz",
        ]
    )

    assert args.command == "eph"
    assert args.task == "relaxation-time"
    assert args.linewidth_data == "linewidth.npz"
    assert args.sum_modes is True
    assert args.output == "relaxation_time.npz"

    alias_args = parse_args(["eph", "--task", "relaxation", "--linewidth-data", "linewidth.npz"])
    assert alias_args.task == "relaxation"
    assert alias_args.linewidth_data == "linewidth.npz"


def test_eph_cli_parser_accepts_path_relaxation_time_postprocess_inputs():
    args = parse_args(
        [
            "eph",
            "--task",
            "path-relaxation-time",
            "--linewidth-data",
            "path_linewidth.npz",
            "--sum-modes",
            "-o",
            "path_relaxation_time.npz",
        ]
    )

    assert args.command == "eph"
    assert args.task == "path-relaxation-time"
    assert args.linewidth_data == "path_linewidth.npz"
    assert args.sum_modes is True
    assert args.output == "path_relaxation_time.npz"

    alias_args = parse_args(["eph", "--task", "path-relaxation", "--linewidth-data", "path_linewidth.npz"])
    assert alias_args.task == "path-relaxation"
    assert alias_args.linewidth_data == "path_linewidth.npz"


def test_eph_cli_parser_accepts_mesh_relaxation_time_postprocess_inputs():
    args = parse_args(
        [
            "eph",
            "--task",
            "mesh-relaxation-time",
            "--linewidth-data",
            "mesh_linewidth.npz",
            "--sum-modes",
            "-o",
            "mesh_relaxation_time.npz",
        ]
    )

    assert args.command == "eph"
    assert args.task == "mesh-relaxation-time"
    assert args.linewidth_data == "mesh_linewidth.npz"
    assert args.sum_modes is True
    assert args.output == "mesh_relaxation_time.npz"

    alias_args = parse_args(["eph", "--task", "mesh-relaxation", "--linewidth-data", "mesh_linewidth.npz"])
    assert alias_args.task == "mesh-relaxation"
    assert alias_args.linewidth_data == "mesh_linewidth.npz"


def test_eph_cli_parser_accepts_transport_postprocess_inputs():
    args = parse_args(
        [
            "eph",
            "--task",
            "transport",
            "-i",
            "model.pth",
            "-stu",
            "struct.vasp",
            "--epc-data",
            "epc_data.npz",
            "--linewidth-data",
            "linewidth.npz",
            "--chemical-potential",
            "0.15",
            "--temperature",
            "0.025",
            "--kpoint-weights",
            "weights.npz",
            "--spin-degeneracy",
            "2",
            "--volume",
            "5.0",
            "--velocity-delta",
            "1e-5",
            "--velocity-source",
            "hamiltonian-derivative",
            "-o",
            "transport.npz",
        ]
    )

    assert args.command == "eph"
    assert args.task == "transport"
    assert args.init_model == "model.pth"
    assert args.structure == "struct.vasp"
    assert args.epc_data == "epc_data.npz"
    assert args.linewidth_data == "linewidth.npz"
    assert args.kpoint_weights == "weights.npz"
    assert args.spin_degeneracy == 2
    assert args.volume == 5.0
    assert args.velocity_delta == 1e-5
    assert args.velocity_source == "hamiltonian-derivative"


def test_eph_cli_parser_accepts_subspace_postprocess_inputs():
    args = parse_args(
        [
            "eph",
            "--task",
            "subspace",
            "--epc-data",
            "epc_data.npz",
            "--final-groups",
            "0:2",
            "2:3",
            "--initial-groups",
            "0:1",
            "1:3",
            "-o",
            "subspace.npz",
        ]
    )

    assert args.command == "eph"
    assert args.task == "subspace"
    assert args.epc_data == "epc_data.npz"
    assert args.final_groups == ["0:2", "2:3"]
    assert args.initial_groups == ["0:1", "1:3"]
    assert args.output == "subspace.npz"


def test_load_kpoints_accepts_json_npy_npz_and_text(tmp_path):
    kpoints = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])

    json_path = tmp_path / "kpoints.json"
    json_path.write_text('{"kpoints": [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]}', encoding="utf-8")
    np.testing.assert_allclose(_load_kpoints(str(json_path)), kpoints)

    npy_path = tmp_path / "kpoints.npy"
    np.save(npy_path, kpoints)
    np.testing.assert_allclose(_load_kpoints(str(npy_path)), kpoints)

    npz_path = tmp_path / "kpoints.npz"
    np.savez(npz_path, kpoints=kpoints)
    np.testing.assert_allclose(_load_kpoints(str(npz_path)), kpoints)

    text_path = tmp_path / "kpoints.dat"
    np.savetxt(text_path, kpoints)
    np.testing.assert_allclose(_load_kpoints(str(text_path)), kpoints)

    object_path = tmp_path / "object_kpoints.npy"
    np.save(object_path, np.array([{"kpoints": kpoints}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        _load_kpoints(str(object_path))

    bad_shape_path = tmp_path / "bad_shape.npy"
    np.save(bad_shape_path, np.ones((2, 2)))
    with pytest.raises(ValueError, match="k/q points"):
        _load_kpoints(str(bad_shape_path))


def test_load_array_accepts_npz_key_for_transport_weights(tmp_path):
    weights = np.array([2.0, 1.0])
    path = tmp_path / "weights.npz"
    np.savez(path, kpoint_weights=weights)
    np.testing.assert_allclose(_load_array(str(path), npz_key="kpoint_weights"), weights)
    np.testing.assert_allclose(_load_kpoint_weights(str(path)), weights)

    missing_path = tmp_path / "missing.npz"
    np.savez(missing_path, weights=weights)
    with pytest.raises(KeyError, match="kpoint_weights"):
        _load_array(str(missing_path), npz_key="kpoint_weights")

    bad_shape_path = tmp_path / "bad_weights.npy"
    np.save(bad_shape_path, np.ones((1, 2)))
    with pytest.raises(ValueError, match="one-dimensional"):
        _load_kpoint_weights(str(bad_shape_path))

    empty_path = tmp_path / "empty_weights.npy"
    np.save(empty_path, np.empty((0,)))
    with pytest.raises(ValueError, match="non-empty"):
        _load_kpoint_weights(str(empty_path))

    nonfinite_path = tmp_path / "nonfinite_weights.npy"
    np.save(nonfinite_path, np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="finite"):
        _load_kpoint_weights(str(nonfinite_path))

    negative_path = tmp_path / "negative_weights.npy"
    np.save(negative_path, np.array([1.0, -0.5]))
    with pytest.raises(ValueError, match="non-negative"):
        _load_kpoint_weights(str(negative_path))

    zero_sum_path = tmp_path / "zero_sum_weights.npy"
    np.save(zero_sum_path, np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="positive sum"):
        _load_kpoint_weights(str(zero_sum_path))


def test_parse_band_groups_accepts_start_stop_ranges():
    groups = _parse_band_groups(["0:2", "2:3"])

    assert [group.tolist() for group in groups] == [[0, 1], [2]]
    with pytest.raises(ValueError, match="at least one"):
        _parse_band_groups([])
    with pytest.raises(ValueError, match="start:stop"):
        _parse_band_groups(["0"])
    with pytest.raises(ValueError, match="integers"):
        _parse_band_groups(["0.5:2"])
    with pytest.raises(ValueError, match="non-negative"):
        _parse_band_groups(["-1:2"])
    with pytest.raises(ValueError, match="larger"):
        _parse_band_groups(["2:2"])
    with pytest.raises(ValueError, match="overlap"):
        _parse_band_groups(["0:2", "1:3"])


def test_eph_entrypoint_writes_epc_npz_from_external_phonon_modes(tmp_path):
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        frequencies=np.array([[1.0]]),
        eigenvectors=np.array([[[[1.0, 0.0, 0.0]]]], dtype=complex),
        masses=np.array([1.0]),
    )
    phonon_path = tmp_path / "phonons.npz"
    phonons.save_npz(phonon_path)
    kpoints_path = tmp_path / "kpoints.json"
    kpoints_path.write_text("[[0.0, 0.0, 0.0]]", encoding="utf-8")
    output_path = tmp_path / "epc_data.npz"

    result = eph(
        structure="unused.vasp",
        init_model="unused.pth",
        phonons=str(phonon_path),
        kpoints=str(kpoints_path),
        output=str(output_path),
        bands=[0],
        system=_FakeSystem(),
        derivative_provider=_FakeDerivativeProvider(),
    )
    loaded = EPCData.load_npz(output_path)

    assert output_path.exists()
    np.testing.assert_allclose(loaded.coupling_strength, result.coupling_strength)
    assert loaded.metadata["schema"] == "deeptb.epc_data"
    assert loaded.metadata["phonon_metadata"]["schema"] == "deeptb.phonons"


def test_eph_entrypoint_writes_epc_path_npz_from_external_phonon_modes(tmp_path):
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        frequencies=np.array([[1.0], [2.0]]),
        eigenvectors=np.array([[[[1.0, 0.0, 0.0]]], [[[1.0, 0.0, 0.0]]]], dtype=complex),
        masses=np.array([1.0]),
    )
    phonon_path = tmp_path / "phonons.npz"
    phonons.save_npz(phonon_path)
    kpoints_path = tmp_path / "kpoints.json"
    kpoints_path.write_text("[[0.0, 0.0, 0.0]]", encoding="utf-8")
    output_path = tmp_path / "epc_path_data.npz"

    result = eph(
        task="path-coupling",
        structure="unused.vasp",
        init_model="unused.pth",
        phonons=str(phonon_path),
        kpoints=str(kpoints_path),
        output=str(output_path),
        bands=[0],
        system=_FakeSystem(),
        derivative_provider=_FakeDerivativeProvider(),
    )
    loaded = EPCPathData.load_npz(output_path)

    assert output_path.exists()
    assert result.metadata["schema"] == "deeptb.epc_path_data"
    assert loaded.metadata["schema"] == "deeptb.epc_path_data"
    assert loaded.metadata["path_mode"] == "fixed_k_q_path"
    np.testing.assert_allclose(loaded.path_coordinates, np.array([0.0, 0.25]))
    np.testing.assert_allclose(loaded.coupling_strength, result.coupling_strength)


def test_eph_entrypoint_writes_epc_mesh_npz_from_external_phonon_modes(tmp_path):
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        frequencies=np.array([[1.0]]),
        eigenvectors=np.array([[[[1.0, 0.0, 0.0]]]], dtype=complex),
        masses=np.array([1.0]),
    )
    phonon_path = tmp_path / "phonons.npz"
    phonons.save_npz(phonon_path)
    output_path = tmp_path / "epc_mesh_data.npz"

    result = eph(
        task="mesh-coupling",
        structure="unused.vasp",
        init_model="unused.pth",
        phonons=str(phonon_path),
        k_mesh=[2, 1, 1],
        chunk_size=1,
        output=str(output_path),
        bands=[0],
        system=_FakeSystem(),
        derivative_provider=_FakeDerivativeProvider(),
    )
    loaded = EPCMeshData.load_npz(output_path)

    assert output_path.exists()
    assert result.metadata["schema"] == "deeptb.epc_mesh_data"
    assert loaded.metadata["schema"] == "deeptb.epc_mesh_data"
    assert loaded.metadata["mesh_spec"]["k_mesh"] == [2, 1, 1]
    assert loaded.metadata["mesh_spec"]["chunk_size"] == 1
    assert loaded.metadata["chunked"] is True
    assert loaded.metadata["execution"] == "serial_k_chunked"
    np.testing.assert_allclose(loaded.kpoint_weights, np.array([0.5, 0.5]))
    np.testing.assert_allclose(loaded.qpoint_weights, np.array([1.0]))
    np.testing.assert_allclose(loaded.coupling_strength, result.coupling_strength)


def test_eph_entrypoint_writes_linewidth_npz_from_epc_data(tmp_path):
    epc_data = _small_linewidth_epc_data()
    epc_path = tmp_path / "epc_data.npz"
    epc_data.save_npz(epc_path)
    output_path = tmp_path / "linewidth.npz"

    result = eph(
        task="linewidth",
        epc_data=str(epc_path),
        output=str(output_path),
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="gaussian",
    )
    loaded = LinewidthData.load_npz(output_path)
    expected = compute_linewidth(
        epc_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="gaussian",
    )

    assert output_path.exists()
    np.testing.assert_allclose(result.linewidth, expected.linewidth)
    np.testing.assert_allclose(loaded.linewidth, expected.linewidth)
    assert loaded.metadata["schema"] == "deeptb.epc_linewidth"
    assert loaded.metadata["broadening"] == "gaussian"


def test_eph_entrypoint_writes_path_linewidth_npz_from_epc_path_data(tmp_path):
    epc_path_data = _small_linewidth_epc_path_data()
    epc_path = tmp_path / "epc_path_data.npz"
    output_path = tmp_path / "path_linewidth.npz"
    epc_path_data.save_npz(epc_path)

    result = eph(
        task="path-linewidth",
        epc_data=str(epc_path),
        output=str(output_path),
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="gaussian",
    )
    loaded = LinewidthPathData.load_npz(output_path)
    expected = compute_linewidth_path(
        epc_path_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="gaussian",
    )

    assert output_path.exists()
    np.testing.assert_allclose(result.linewidth, expected.linewidth)
    np.testing.assert_allclose(loaded.linewidth, expected.linewidth)
    np.testing.assert_allclose(loaded.path_coordinates, epc_path_data.path_coordinates)
    assert loaded.metadata["schema"] == "deeptb.epc_path_linewidth"
    assert loaded.metadata["path_mode"] == "fixed_k_q_path"


def test_eph_entrypoint_writes_mesh_linewidth_npz_from_epc_mesh_data(tmp_path):
    epc_mesh_data = _small_linewidth_epc_mesh_data()
    epc_path = tmp_path / "epc_mesh_data.npz"
    output_path = tmp_path / "mesh_linewidth.npz"
    epc_mesh_data.save_npz(epc_path)

    result = eph(
        task="mesh-linewidth",
        epc_data=str(epc_path),
        output=str(output_path),
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="gaussian",
    )
    loaded = LinewidthMeshData.load_npz(output_path)
    expected = compute_linewidth_mesh(
        epc_mesh_data,
        chemical_potential=0.15,
        temperature=0.025,
        sigma=0.01,
        broadening="gaussian",
    )

    assert output_path.exists()
    np.testing.assert_allclose(result.linewidth, expected.linewidth)
    np.testing.assert_allclose(loaded.linewidth, expected.linewidth)
    np.testing.assert_allclose(loaded.kpoint_weights, epc_mesh_data.kpoint_weights)
    assert loaded.metadata["schema"] == "deeptb.epc_mesh_linewidth"
    assert loaded.metadata["mesh_spec"] == {"k_mesh": [1, 1, 1]}


def test_eph_entrypoint_writes_relaxation_time_npz_from_linewidth_data(tmp_path):
    linewidth = LinewidthData(
        linewidth=np.array([[[0.01, 0.03], [0.02, 0.06]]]),
        absorption=np.zeros((1, 2, 2)),
        emission=np.array([[[0.01, 0.03], [0.02, 0.06]]]),
    )
    linewidth_path = tmp_path / "linewidth.npz"
    output_path = tmp_path / "relaxation_time.npz"
    linewidth.save_npz(linewidth_path)

    result = eph(
        task="relaxation-time",
        linewidth_data=str(linewidth_path),
        output=str(output_path),
        sum_modes=True,
    )
    loaded = RelaxationTimeData.load_npz(output_path)
    expected = compute_relaxation_time(linewidth, sum_modes=True)

    assert output_path.exists()
    np.testing.assert_allclose(result.relaxation_time, expected.relaxation_time)
    np.testing.assert_allclose(loaded.relaxation_time, expected.relaxation_time)
    assert loaded.metadata["schema"] == "deeptb.epc_relaxation_time"
    assert loaded.metadata["sum_modes"] is True


def test_eph_entrypoint_writes_path_relaxation_time_npz_from_path_linewidth_data(tmp_path):
    linewidth = LinewidthPathData(
        linewidth=np.array([[[[0.01, 0.03], [0.02, 0.06]]], [[[0.04, 0.08], [0.05, 0.10]]]]),
        absorption=np.zeros((2, 1, 2, 2)),
        emission=np.array([[[[0.01, 0.03], [0.02, 0.06]]], [[[0.04, 0.08], [0.05, 0.10]]]]),
        path_axis="q",
        path_coordinates=np.array([0.0, 0.25]),
        path_segments=np.array([[0, 2]]),
        band_indices=np.array([0, 1]),
        metadata={"path_mode": "fixed_k_q_path"},
    )
    linewidth_path = tmp_path / "path_linewidth.npz"
    output_path = tmp_path / "path_relaxation_time.npz"
    linewidth.save_npz(linewidth_path)

    result = eph(
        task="path-relaxation-time",
        linewidth_data=str(linewidth_path),
        output=str(output_path),
        sum_modes=True,
    )
    loaded = RelaxationTimePathData.load_npz(output_path)
    expected = compute_relaxation_time_path(linewidth, sum_modes=True)

    assert output_path.exists()
    np.testing.assert_allclose(result.relaxation_time, expected.relaxation_time)
    np.testing.assert_allclose(loaded.relaxation_time, expected.relaxation_time)
    np.testing.assert_allclose(loaded.path_coordinates, linewidth.path_coordinates)
    assert loaded.metadata["schema"] == "deeptb.epc_path_relaxation_time"
    assert loaded.metadata["path_mode"] == "fixed_k_q_path"


def test_eph_entrypoint_writes_mesh_relaxation_time_npz_from_mesh_linewidth_data(tmp_path):
    linewidth = LinewidthMeshData(
        linewidth=np.array([[[0.01, 0.03], [0.02, 0.06]], [[0.04, 0.08], [0.05, 0.10]]]),
        absorption=np.zeros((2, 2, 2)),
        emission=np.array([[[0.01, 0.03], [0.02, 0.06]], [[0.04, 0.08], [0.05, 0.10]]]),
        kpoints=np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
        kpoint_weights=np.array([1.0, 1.0]),
        band_indices=np.array([0, 1]),
        metadata={"mesh_spec": {"k_mesh": [2, 1, 1]}},
    )
    linewidth_path = tmp_path / "mesh_linewidth.npz"
    output_path = tmp_path / "mesh_relaxation_time.npz"
    linewidth.save_npz(linewidth_path)

    result = eph(
        task="mesh-relaxation-time",
        linewidth_data=str(linewidth_path),
        output=str(output_path),
        sum_modes=True,
    )
    loaded = RelaxationTimeMeshData.load_npz(output_path)
    expected = compute_relaxation_time_mesh(linewidth, sum_modes=True)

    assert output_path.exists()
    np.testing.assert_allclose(result.relaxation_time, expected.relaxation_time)
    np.testing.assert_allclose(loaded.relaxation_time, expected.relaxation_time)
    np.testing.assert_allclose(loaded.kpoint_weights, np.array([0.5, 0.5]))
    assert loaded.metadata["schema"] == "deeptb.epc_mesh_relaxation_time"
    assert loaded.metadata["mesh_spec"] == {"k_mesh": [2, 1, 1]}


def test_eph_entrypoint_writes_transport_npz_from_epc_and_linewidth_data(tmp_path):
    system = _LinearBandSystem()
    kpoints = np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.2]])
    band_indices = np.array([0, 2])
    eigenvalues = system.band_offsets[None, band_indices] + kpoints @ system.band_slopes[band_indices].T
    epc_data = EPCData(
        kpoints=kpoints,
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        band_indices=band_indices,
        frequencies=np.array([[1.0]]),
        eigenvalues_k=eigenvalues,
        eigenvalues_kq=eigenvalues[None, :, :],
        coupling_matrix=np.ones((1, 2, 1, 2, 2), dtype=complex),
        coupling_strength=np.ones((1, 2, 1, 2, 2)),
    )
    linewidth = LinewidthData(
        linewidth=np.array([[0.01, 0.02], [0.03, 0.04]]),
        absorption=np.zeros((2, 2)),
        emission=np.array([[0.01, 0.02], [0.03, 0.04]]),
    )
    epc_path = tmp_path / "epc_data.npz"
    linewidth_path = tmp_path / "linewidth.npz"
    weights_path = tmp_path / "weights.npz"
    output_path = tmp_path / "transport.npz"
    epc_data.save_npz(epc_path)
    linewidth.save_npz(linewidth_path)
    np.savez(weights_path, kpoint_weights=np.array([2.0, 1.0]))

    result = eph(
        task="transport",
        structure="unused.vasp",
        init_model="unused.pth",
        epc_data=str(epc_path),
        linewidth_data=str(linewidth_path),
        output=str(output_path),
        chemical_potential=0.15,
        temperature=0.03,
        kpoint_weights=str(weights_path),
        spin_degeneracy=2,
        volume=5.0,
        system=system,
    )
    loaded = TransportData.load_npz(output_path)
    expected = compute_serta_transport_from_epc(
        system=system,
        epc_data=epc_data,
        linewidth_data=linewidth,
        chemical_potential=0.15,
        temperature=0.03,
        kpoint_weights=np.array([2.0, 1.0]),
        spin_degeneracy=2,
        volume=5.0,
    )

    assert output_path.exists()
    np.testing.assert_allclose(result.conductivity, expected.conductivity)
    np.testing.assert_allclose(loaded.conductivity, expected.conductivity)
    assert loaded.metadata["schema"] == "deeptb.epc_transport"
    assert loaded.metadata["velocity_source"] == "finite_difference"


def test_eph_entrypoint_writes_subspace_npz_from_epc_data(tmp_path):
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        band_indices=np.array([0, 1, 2]),
        frequencies=np.array([[1.0]]),
        eigenvalues_k=np.array([[0.0, 0.0, 1.0]]),
        eigenvalues_kq=np.array([[[0.0, 0.0, 1.0]]]),
        coupling_matrix=np.ones((1, 1, 1, 3, 3), dtype=complex),
        coupling_strength=np.ones((1, 1, 1, 3, 3)),
    )
    epc_path = tmp_path / "epc_data.npz"
    output_path = tmp_path / "subspace.npz"
    epc_data.save_npz(epc_path)

    result = eph(
        task="subspace",
        epc_data=str(epc_path),
        final_groups=["0:2", "2:3"],
        output=str(output_path),
    )
    loaded = SubspaceCouplingData.load_npz(output_path)

    assert output_path.exists()
    np.testing.assert_allclose(result.strength, loaded.strength)
    np.testing.assert_array_equal(loaded.final_group_bounds, np.array([[0, 2], [2, 3]]))
    assert loaded.metadata["schema"] == "deeptb.epc_subspace_coupling"


def test_eph_entrypoint_rejects_scc_v1(tmp_path):
    phonon_path = tmp_path / "phonons.npz"
    Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        frequencies=np.array([[1.0]]),
        eigenvectors=np.array([[[[1.0, 0.0, 0.0]]]], dtype=complex),
        masses=np.array([1.0]),
    ).save_npz(phonon_path)
    kpoints_path = tmp_path / "kpoints.json"
    kpoints_path.write_text("[[0.0, 0.0, 0.0]]", encoding="utf-8")

    with pytest.raises(NotImplementedError, match="SCC-corrected"):
        eph(
            structure="unused.vasp",
            init_model="unused.pth",
            phonons=str(phonon_path),
            kpoints=str(kpoints_path),
            output=str(tmp_path / "epc_data.npz"),
            use_scc=True,
            system=_FakeSystem(),
        )

    with pytest.raises(NotImplementedError, match="linewidth"):
        eph(
            task="linewidth",
            epc_data=str(tmp_path / "unused_epc.npz"),
            chemical_potential=0.0,
            temperature=0.01,
            sigma=0.01,
            use_scc=True,
        )

    with pytest.raises(NotImplementedError, match="relaxation-time"):
        eph(
            task="relaxation-time",
            linewidth_data=str(tmp_path / "unused_linewidth.npz"),
            use_scc=True,
        )

    with pytest.raises(NotImplementedError, match="subspace"):
        eph(
            task="subspace",
            epc_data=str(tmp_path / "unused_epc.npz"),
            final_groups=["0:1"],
            use_scc=True,
        )
    with pytest.raises(NotImplementedError, match="transport"):
        eph(
            task="transport",
            epc_data=str(tmp_path / "unused_epc.npz"),
            linewidth_data=str(tmp_path / "unused_linewidth.npz"),
            chemical_potential=0.0,
            temperature=0.01,
            use_scc=True,
        )


def test_eph_entrypoint_rejects_invalid_task_type():
    with pytest.raises(ValueError, match="task"):
        eph(task=None)


class _FakePhonopyCell:
    symbols = ["C", "C"]
    scaled_positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    cell = np.eye(3)

    def __len__(self):
        return 2


class _FakePhonopySupercell(_FakePhonopyCell):
    supercell_matrix = np.eye(3, dtype=int)
    u2u_map = np.array([0, 1])
    s2u_map = np.array([0, 1])


class _FakePhonopy:
    primitive = _FakePhonopyCell()
    supercell = _FakePhonopySupercell()


def test_supercell_fd_provider_from_phonopy_mock(monkeypatch):
    def fake_maps(phonopy_obj):
        return (
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0, 0]),
            np.zeros((1, 2, 1, 3)),
            np.ones((1, 2), dtype=int),
        )

    monkeypatch.setattr(eph_providers, "_phonopy_supercell_maps", fake_maps)

    provider = SupercellFD.from_phonopy(
        system=_FakeSystem(),
        phonopy_obj=_FakePhonopy(),
        length_unit="bohr",
    )

    assert isinstance(provider, SupercellFD)
    np.testing.assert_array_equal(provider.primitive_orbital_offsets, np.array([0, 4, 8]))
    np.testing.assert_array_equal(provider.supercell_orbital_offsets, np.array([0, 4, 8]))
    np.testing.assert_allclose(provider.supercell_atoms.cell.array, np.eye(3) * 0.529177249)


def test_graphene_reference_case_coupling_strength():
    if os.environ.get("DEEPTB_RUN_REFERENCE_EPH") != "1":
        pytest.skip("Set DEEPTB_RUN_REFERENCE_EPH=1 to run the external Graphene reference regression.")
    phonopy = pytest.importorskip("phonopy")

    # TODO before merge/release: replace the external Graphene reference with
    # a lightweight in-repo fixture for default CI.
    reference_root = _external_reference_root()
    graphene_root = reference_root / "examples" / "Graphene"
    if not graphene_root.exists():
        pytest.skip("Graphene reference case is not available.")

    if str(reference_root) not in sys.path:
        sys.path.insert(0, str(reference_root))
    dftbephy_fourier = pytest.importorskip("dftbephy.fourier")
    dftbephy_dftb = pytest.importorskip("dftbephy.dftb")
    dftbephy_units = pytest.importorskip("dftbephy.units")

    phonon_yaml = graphene_root / "phonons" / "phonopy_disp.yaml"
    force_sets = graphene_root / "phonons" / "FORCE_SETS"
    reference_npz = graphene_root / "el-ph" / "reference.npz"
    derivatives_npz = graphene_root / "el-ph" / "derivatives.npz"
    reference_hdf5 = graphene_root / "el-ph" / "el-ph-Nq4-K-bandsel.hdf5"
    for path in [phonon_yaml, force_sets, reference_npz, derivatives_npz, reference_hdf5]:
        if not path.exists():
            pytest.skip(f"Missing Graphene reference file: {path}")

    phonons = phonopy.load(str(phonon_yaml), force_sets_filename=str(force_sets))
    primitive = phonons.primitive
    supercell = phonons.supercell
    num_supercells = int(round(abs(np.linalg.det(supercell.supercell_matrix))))
    num_primitive_atoms = len(primitive)
    supercell_atom_to_cell = np.tile(np.arange(num_supercells), num_primitive_atoms)
    primitive_map = supercell.u2u_map
    supercell_to_primitive_atom = np.array([primitive_map[atom] for atom in supercell.s2u_map], dtype=int)
    primitive_to_supercell_atom = np.zeros((num_primitive_atoms,), dtype=int)
    primitive_to_supercell_atom[0] = (num_supercells - 1) // 2
    for atom_index in range(1, num_primitive_atoms):
        primitive_to_supercell_atom[atom_index] = primitive_to_supercell_atom[atom_index - 1] + num_supercells

    angular_momenta = {"C": ["s", "p"]}
    supercell_orbitals = [
        sum([dftbephy_dftb.std_orbital_order[angular] for angular in angular_momenta[symbol]], [])
        for symbol in supercell.symbols
    ]
    supercell_orbital_offsets = np.insert(np.cumsum([len(orbs) for orbs in supercell_orbitals]), 0, 0)
    primitive_orbitals = [supercell_orbitals[index] for index in primitive_to_supercell_atom]
    primitive_orbital_offsets = np.insert(np.cumsum([len(orbs) for orbs in primitive_orbitals]), 0, 0)
    orbital_slices = [
        (int(primitive_orbital_offsets[index]), int(primitive_orbital_offsets[index + 1]))
        for index in range(num_primitive_atoms)
    ]

    shortest_vectors, vector_multiplicity = eph_providers._phonopy_supercell_maps(phonons)[3:]

    with np.load(reference_npz, allow_pickle=False) as data:
        reference_hamiltonian = data["H0"]
        reference_overlap = data["S0"]
    with np.load(derivatives_npz, allow_pickle=False) as data:
        h_derivatives_supercell = data["H_derivs"]
        overlap_derivatives_supercell = data["S_derivs"]
    with h5py.File(reference_hdf5, "r") as data:
        qpoints = data["ph/qpoints"][()]
        frequencies = data["ph/omega"][()] / dftbephy_units.THZ__EV
        reference_strength = data["el-ph/g2_0"][()]
        kpoint = data["el-ph/g2_0"].attrs["kvec"]
        reference_eigenvalues_k = data["el/eps_0"][()]
        reference_eigenvalues_kq = data["el/eps_q_0"][()]

    phonons.run_qpoints(qpoints, with_eigenvectors=True)
    phonon_eigenvectors = reshape_phonopy_eigenvectors(
        phonons.get_qpoints_dict()["eigenvectors"],
        num_primitive_atoms,
    )

    hamiltonian_k = dftbephy_fourier.calculate_lattice_ft(
        reference_hamiltonian,
        kpoint,
        primitive_to_supercell_atom,
        supercell_to_primitive_atom,
        supercell_atom_to_cell,
        primitive_orbital_offsets,
        supercell_orbital_offsets,
        shortest_vectors,
        vector_multiplicity,
    )
    overlap_k = dftbephy_fourier.calculate_lattice_ft(
        reference_overlap,
        kpoint,
        primitive_to_supercell_atom,
        supercell_to_primitive_atom,
        supercell_atom_to_cell,
        primitive_orbital_offsets,
        supercell_orbital_offsets,
        shortest_vectors,
        vector_multiplicity,
    )
    eigenvalues_k, eigenvectors_k = linalg.eigh(hamiltonian_k, b=overlap_k)
    h_derivatives_k = dftbephy_fourier.calculate_lattice_ft_derivative(
        h_derivatives_supercell,
        kpoint,
        primitive_to_supercell_atom,
        supercell_to_primitive_atom,
        supercell_atom_to_cell,
        primitive_orbital_offsets,
        supercell_orbital_offsets,
        shortest_vectors,
        vector_multiplicity,
    )[None]
    overlap_derivatives_k = dftbephy_fourier.calculate_lattice_ft_derivative(
        overlap_derivatives_supercell,
        kpoint,
        primitive_to_supercell_atom,
        supercell_to_primitive_atom,
        supercell_atom_to_cell,
        primitive_orbital_offsets,
        supercell_orbital_offsets,
        shortest_vectors,
        vector_multiplicity,
    )[None]

    eigenvalues_kq = []
    eigenvectors_kq = []
    h_derivatives_kq = []
    overlap_derivatives_kq = []
    for qpoint in qpoints:
        kqpoint = kpoint + qpoint
        hamiltonian_kq = dftbephy_fourier.calculate_lattice_ft(
            reference_hamiltonian,
            kqpoint,
            primitive_to_supercell_atom,
            supercell_to_primitive_atom,
            supercell_atom_to_cell,
            primitive_orbital_offsets,
            supercell_orbital_offsets,
            shortest_vectors,
            vector_multiplicity,
        )
        overlap_kq = dftbephy_fourier.calculate_lattice_ft(
            reference_overlap,
            kqpoint,
            primitive_to_supercell_atom,
            supercell_to_primitive_atom,
            supercell_atom_to_cell,
            primitive_orbital_offsets,
            supercell_orbital_offsets,
            shortest_vectors,
            vector_multiplicity,
        )
        bands_kq, states_kq = linalg.eigh(hamiltonian_kq, b=overlap_kq)
        eigenvalues_kq.append(bands_kq)
        eigenvectors_kq.append(states_kq)
        h_derivatives_kq.append(
            dftbephy_fourier.calculate_lattice_ft_derivative(
                h_derivatives_supercell,
                kqpoint,
                primitive_to_supercell_atom,
                supercell_to_primitive_atom,
                supercell_atom_to_cell,
                primitive_orbital_offsets,
                supercell_orbital_offsets,
                shortest_vectors,
                vector_multiplicity,
            )
        )
        overlap_derivatives_kq.append(
            dftbephy_fourier.calculate_lattice_ft_derivative(
                overlap_derivatives_supercell,
                kqpoint,
                primitive_to_supercell_atom,
                supercell_to_primitive_atom,
                supercell_atom_to_cell,
                primitive_orbital_offsets,
                supercell_orbital_offsets,
                shortest_vectors,
                vector_multiplicity,
            )
        )

    _, coupling_strength = compute_coupling_matrix(
        eigenvalues_k=eigenvalues_k[None],
        eigenvectors_k=eigenvectors_k[None],
        eigenvalues_kq=np.asarray(eigenvalues_kq)[:, None, :],
        eigenvectors_kq=np.asarray(eigenvectors_kq)[:, None, :, :],
        h_derivatives_k=h_derivatives_k,
        h_derivatives_kq=np.asarray(h_derivatives_kq)[:, None, :, :, :],
        overlap_derivatives_k=overlap_derivatives_k,
        overlap_derivatives_kq=np.asarray(overlap_derivatives_kq)[:, None, :, :, :],
        phonon_eigenvectors=phonon_eigenvectors,
        masses=np.asarray(supercell.masses)[primitive_to_supercell_atom],
        frequencies=frequencies,
        qpoints=qpoints,
        scaled_positions=primitive.scaled_positions,
        orbital_slices=orbital_slices,
        derivative_mode="row",
        prefactor_amu_thz=dftbephy_units.hbar_amu_THz,
    )

    np.testing.assert_allclose(eigenvalues_k, reference_eigenvalues_k, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(np.asarray(eigenvalues_kq), reference_eigenvalues_kq, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(coupling_strength[:, 0], reference_strength, atol=5e-12, rtol=1e-12)


def test_graphene_reference_case_supercell_fd_provider():
    if os.environ.get("DEEPTB_RUN_REFERENCE_EPH") != "1" or os.environ.get("DEEPTB_RUN_SLOW_EPH") != "1":
        pytest.skip(
            "Set DEEPTB_RUN_REFERENCE_EPH=1 and DEEPTB_RUN_SLOW_EPH=1 to run the external Graphene "
            "finite-difference regression."
        )
    phonopy = pytest.importorskip("phonopy")

    from dptb.nn.build import build_model
    from dptb.postprocess.unified import TBSystem

    # TODO before merge/release: replace these external references with
    # lightweight in-repo fixtures for default CI.
    reference_root = _external_reference_root()
    graphene_root = reference_root / "examples" / "Graphene"
    skdata = _external_skdata_root()
    for path in [
        graphene_root / "opt" / "graphene.gen",
        graphene_root / "phonons" / "phonopy_disp.yaml",
        graphene_root / "phonons" / "FORCE_SETS",
        graphene_root / "el-ph" / "derivatives.npz",
        skdata,
    ]:
        if not path.exists():
            pytest.skip(f"Missing Graphene finite-difference reference input: {path}")

    common_options = {
        "basis": {"C": ["2s", "2p"]},
        "device": "cpu",
        "dtype": "float64",
        "overlap": True,
    }
    model_options = {
        "dftbsk": {
            "skdata": str(skdata),
            "r_max": {"C": 7.0},
            "smooth_ski": True,
        }
    }
    model = build_model(None, model_options, common_options)
    system = TBSystem(data=str(graphene_root / "opt" / "graphene.gen"), calculator=model)
    phonons = phonopy.load(
        str(graphene_root / "phonons" / "phonopy_disp.yaml"),
        force_sets_filename=str(graphene_root / "phonons" / "FORCE_SETS"),
    )
    provider = SupercellFD.from_phonopy(system, phonons, displacement=1e-3, length_unit="bohr")
    h_derivatives, overlap_derivatives = provider._compute_supercell_derivatives()

    with np.load(graphene_root / "el-ph" / "derivatives.npz", allow_pickle=False) as data:
        reference_h_derivatives = data["H_derivs"]
        reference_overlap_derivatives = data["S_derivs"]

    atom_orbs = []
    orbital_info = system.calculator.get_orbital_info()
    for atom_index, symbol in enumerate(provider.supercell_atoms.symbols):
        for orbital in orbital_info[symbol]:
            atom_orbs.append(f"{atom_index}-{symbol}-{orbital}")
    gauge = DFTBPlusGauge.from_atom_orbs(atom_orbs)
    h_derivatives = gauge.transform_row_derivatives(h_derivatives)
    overlap_derivatives = gauge.transform_row_derivatives(overlap_derivatives)

    h_abs_error = np.abs(h_derivatives - reference_h_derivatives)
    overlap_abs_error = np.abs(overlap_derivatives - reference_overlap_derivatives)
    assert h_abs_error.max() < 2.0e-3
    assert np.sqrt(np.mean(h_abs_error**2)) < 2.0e-5
    assert overlap_abs_error.max() < 1.0e-4
    assert np.sqrt(np.mean(overlap_abs_error**2)) < 1.0e-6
