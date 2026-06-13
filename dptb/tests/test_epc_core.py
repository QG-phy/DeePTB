"""Core EPC module tests - numerical correctness and data integrity.

Tests focus on:
- Numerical correctness (coupling, linewidth, transport calculations)
- Data serialization (NPZ roundtrip)
- Fixture-based validation
- Integration across components
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import constants as scipy_constants

from dptb.postprocess.unified.eph import (
    EPCData,
    EPCMeshData,
    EPCMeshSpec,
    EPCPathData,
    EPC_MESH_NPZ_SCHEMA_VERSION,
    EPC_NPZ_SCHEMA_VERSION,
    EPC_PATH_NPZ_SCHEMA_VERSION,
    LinewidthData,
    LinewidthMeshData,
    LinewidthPathData,
    LINEWIDTH_MESH_NPZ_SCHEMA_VERSION,
    LINEWIDTH_NPZ_SCHEMA_VERSION,
    LINEWIDTH_PATH_NPZ_SCHEMA_VERSION,
    MobilityData,
    MobilityScanData,
    MOBILITY_NPZ_SCHEMA_VERSION,
    MOBILITY_SCAN_NPZ_SCHEMA_VERSION,
    Phonons,
    RelaxationTimeData,
    RelaxationTimeMeshData,
    RelaxationTimePathData,
    RELAXATION_TIME_MESH_NPZ_SCHEMA_VERSION,
    RELAXATION_TIME_NPZ_SCHEMA_VERSION,
    RELAXATION_TIME_PATH_NPZ_SCHEMA_VERSION,
    SubspaceCouplingData,
    SUBSPACE_COUPLING_NPZ_SCHEMA_VERSION,
    THZ_TO_EV,
    TransportData,
    TransportScanData,
    TRANSPORT_NPZ_SCHEMA_VERSION,
    TRANSPORT_SCAN_NPZ_SCHEMA_VERSION,
    compute_band_velocities_finite_difference,
    compute_band_velocities_hamiltonian_derivative,
    compute_coupling_strength_summary,
    compute_eliashberg_spectral_function,
    compute_linewidth,
    compute_linewidth_mesh,
    compute_linewidth_path,
    compute_phonon_dos,
    compute_relaxation_time,
    compute_scattering_maps,
    compute_serta_conductivity,
    compute_serta_mobility_si,
    compute_serta_mobility_scan_si,
    compute_serta_transport_from_epc,
    compute_serta_transport_from_epc_mesh_chunked_artifact,
    compute_serta_transport_scan_from_epc_mesh_chunked_artifact,
    compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact,
    compute_serta_mobility_si_from_epc_mesh_chunked_artifact,
    compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact,
    compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact,
    compute_subspace_coupling_data,
    load_epc_mesh_chunked_artifact,
    save_epc_mesh_chunked_artifact,
)
from dptb.postprocess.unified.eph.contraction import EPC_PREFAC_AMU_THZ, compute_coupling_matrix


MINIMAL_EPC_FIXTURE = Path(__file__).parent / "fixtures" / "eph" / "minimal_epc_reference.json"


def _minimal_fixture_epc_data() -> EPCData:
    """Load minimal in-repo EPC fixture for validation."""
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


# ============================================================================
# Fixture-based validation tests
# ============================================================================

class TestFixtureValidation:
    """Validate core calculations against minimal in-repo fixture."""

    def test_coupling_contraction_reference(self):
        """Verify compute_coupling_matrix against fixture reference."""
        with open(MINIMAL_EPC_FIXTURE, "r", encoding="utf-8") as handle:
            fixture = json.load(handle)

        payload = fixture["coupling_contraction"]
        phonon_eigenvectors = np.asarray(payload["phonon_eigenvectors_real"], dtype=float) + 1j * np.asarray(
            payload["phonon_eigenvectors_imag"],
            dtype=float,
        )
        coupling_matrix, coupling_strength = compute_coupling_matrix(
            eigenvalues_k=np.asarray(payload["eigenvalues_k"], dtype=float),
            eigenvectors_k=np.asarray(payload["eigenvectors_k"], dtype=complex),
            eigenvalues_kq=np.asarray(payload["eigenvalues_kq"], dtype=float),
            eigenvectors_kq=np.asarray(payload["eigenvectors_kq"], dtype=complex),
            h_derivatives_k=np.asarray(payload["h_derivatives_k"], dtype=complex),
            h_derivatives_kq=np.asarray(payload["h_derivatives_kq"], dtype=complex),
            overlap_derivatives_k=np.asarray(payload["overlap_derivatives_k"], dtype=complex),
            overlap_derivatives_kq=np.asarray(payload["overlap_derivatives_kq"], dtype=complex),
            phonon_eigenvectors=phonon_eigenvectors,
            masses=np.asarray(payload["masses"], dtype=float),
            frequencies=np.asarray(payload["frequencies"], dtype=float),
        )

        expected_coupling = np.asarray(payload["expected_coupling_matrix_real"], dtype=float) + 1j * np.asarray(
            payload["expected_coupling_matrix_imag"],
            dtype=float,
        )
        np.testing.assert_allclose(coupling_matrix, expected_coupling, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(
            coupling_strength,
            np.asarray(payload["expected_coupling_strength"], dtype=float),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_linewidth_reference(self):
        """Verify compute_linewidth against fixture reference."""
        with open(MINIMAL_EPC_FIXTURE, "r", encoding="utf-8") as handle:
            fixture = json.load(handle)

        epc_data = _minimal_fixture_epc_data()
        params = fixture["linewidth_parameters"]
        linewidth = compute_linewidth(
            epc_data,
            chemical_potential=params["chemical_potential"],
            temperature=params["temperature"],
            sigma=params["sigma"],
            broadening=params["broadening"],
        )

        expected = fixture["expected_linewidth"]
        np.testing.assert_allclose(linewidth.linewidth, np.asarray(expected["linewidth"]), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(linewidth.absorption, np.asarray(expected["absorption"]), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(linewidth.emission, np.asarray(expected["emission"]), rtol=1e-14, atol=1e-14)

    def test_analysis_references(self):
        """Verify compute_coupling_strength_summary and scattering_maps."""
        epc_data = _minimal_fixture_epc_data()

        summary = compute_coupling_strength_summary(epc_data)
        np.testing.assert_allclose(summary["total"], 0.30, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(summary["q_resolved"], np.array([0.30]), rtol=1e-14, atol=1e-14)

        maps = compute_scattering_maps(epc_data)
        np.testing.assert_allclose(maps["q_mode_resolved"], np.array([[0.30]]), rtol=1e-14, atol=1e-14)


# ============================================================================
# Coupling matrix tests
# ============================================================================

class TestCouplingMatrix:
    """Verify compute_coupling_matrix correctness."""

    def test_without_overlap(self):
        """Verify coupling matrix without overlap derivatives."""
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

    def test_scalar_vs_array_mass(self):
        """Verify scalar and array mass inputs give same result."""
        eigenvalues_k = np.array([[1.0]])
        eigenvalues_kq = eigenvalues_k.reshape(1, 1, 1)
        eigenvectors_k = np.ones((1, 1, 1), dtype=complex)
        eigenvectors_kq = eigenvectors_k.reshape(1, 1, 1, 1)
        h_derivatives_k = np.ones((1, 1, 3, 1, 1), dtype=complex) * 4.0
        h_derivatives_kq = np.zeros((1, 1, 1, 3, 1, 1), dtype=complex)
        phonon_eigenvectors = np.zeros((1, 1, 1, 3), dtype=complex)
        phonon_eigenvectors[0, 0, 0, 0] = 1.0

        scalar_coupling, _ = compute_coupling_matrix(
            eigenvalues_k=eigenvalues_k,
            eigenvectors_k=eigenvectors_k,
            eigenvalues_kq=eigenvalues_kq,
            eigenvectors_kq=eigenvectors_kq,
            h_derivatives_k=h_derivatives_k,
            h_derivatives_kq=h_derivatives_kq,
            phonon_eigenvectors=phonon_eigenvectors,
            masses=np.array(4.0),
        )
        array_coupling, _ = compute_coupling_matrix(
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

    def test_with_frequency_prefactor(self):
        """Verify coupling matrix with frequency prefactor."""
        eigenvalues_k = np.array([[2.0]])
        eigenvalues_kq = np.array([[[3.0]]])
        eigenvectors_k = np.ones((1, 1, 1), dtype=complex)
        eigenvectors_kq = np.ones((1, 1, 1, 1), dtype=complex)
        h_derivatives_k = np.ones((1, 1, 3, 1, 1), dtype=complex) * 5.0
        h_derivatives_kq = np.ones((1, 1, 1, 3, 1, 1), dtype=complex)
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
            frequencies=np.array([[2.0]]),
        )

        raw = (5.0 - 1.0) - (2.0 * 0.0 - 3.0 * 0.0)
        prefactor = EPC_PREFAC_AMU_THZ / (2.0 * 2.0)
        np.testing.assert_allclose(coupling_matrix[0, 0, 0, 0, 0], raw * np.sqrt(prefactor))
        np.testing.assert_allclose(coupling_strength[0, 0, 0, 0, 0], abs(raw) ** 2 * prefactor)

    def test_gauge_invariance(self):
        """Verify coupling strength is invariant to orbital sign gauge."""
        eigenvalues_k = np.array([[1.0, 2.0]])
        eigenvalues_kq = np.array([[[1.2, 2.3]]])
        eigenvectors_k = np.eye(2, dtype=complex).reshape(1, 2, 2)
        eigenvectors_kq = eigenvectors_k.reshape(1, 1, 2, 2)
        h_derivatives_k = np.zeros((1, 1, 3, 2, 2), dtype=complex)
        h_derivatives_kq = np.zeros((1, 1, 1, 3, 2, 2), dtype=complex)
        h_derivatives_k[0, 0, 0] = np.array([[0.2, 0.3 + 0.1j], [0.3 - 0.1j, -0.1]])
        h_derivatives_kq[0, 0, 0, 0] = np.array([[0.05, -0.2j], [0.2j, 0.07]])
        phonon_eigenvectors = np.zeros((1, 1, 1, 3), dtype=complex)
        phonon_eigenvectors[0, 0, 0, 0] = 1.0

        _, reference_strength = compute_coupling_matrix(
            eigenvalues_k=eigenvalues_k,
            eigenvectors_k=eigenvectors_k,
            eigenvalues_kq=eigenvalues_kq,
            eigenvectors_kq=eigenvectors_kq,
            h_derivatives_k=h_derivatives_k,
            h_derivatives_kq=h_derivatives_kq,
            phonon_eigenvectors=phonon_eigenvectors,
            masses=np.array([1.0]),
            frequencies=np.array([[5.0]]),
        )

        orbital_signs = np.diag([1.0, -1.0])
        transformed_eigenvectors_k = orbital_signs @ eigenvectors_k
        transformed_eigenvectors_kq = orbital_signs @ eigenvectors_kq
        transformed_h_derivatives_k = orbital_signs @ h_derivatives_k @ orbital_signs
        transformed_h_derivatives_kq = orbital_signs @ h_derivatives_kq @ orbital_signs

        _, transformed_strength = compute_coupling_matrix(
            eigenvalues_k=eigenvalues_k,
            eigenvectors_k=transformed_eigenvectors_k,
            eigenvalues_kq=eigenvalues_kq,
            eigenvectors_kq=transformed_eigenvectors_kq,
            h_derivatives_k=transformed_h_derivatives_k,
            h_derivatives_kq=transformed_h_derivatives_kq,
            phonon_eigenvectors=phonon_eigenvectors,
            masses=np.array([1.0]),
            frequencies=np.array([[5.0]]),
        )

        np.testing.assert_allclose(transformed_strength, reference_strength, atol=1e-14, rtol=1e-14)


# ============================================================================
# EPC data structure tests
# ============================================================================

class TestEPCDataSerialization:
    """Verify EPC data structure NPZ roundtrip."""

    def test_epc_data_roundtrip(self, tmp_path):
        """Verify EPCData save/load cycle."""
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
        assert loaded.metadata["schema"] == "deeptb.epc_data"
        assert loaded.metadata["schema_version"] == EPC_NPZ_SCHEMA_VERSION

    def test_epc_mesh_spec(self):
        """Verify EPCMeshSpec parameter validation."""
        phonons = Phonons(
            qpoints=np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
            frequencies=np.array([[1.0], [2.0]]),
            eigenvectors=np.array([[[[1.0, 0.0, 0.0]]], [[[1.0, 0.0, 0.0]]]], dtype=complex),
            masses=np.array([1.0]),
        )
        spec = EPCMeshSpec(k_mesh=[2, 1, 1], q_mesh=[2, 1, 1])
        kpoints, kpoint_weights = spec.resolve_kpoints_and_weights()
        qpoint_weights = spec.resolve_qpoint_weights(phonons)

        np.testing.assert_allclose(kpoint_weights, np.array([0.5, 0.5]))
        np.testing.assert_allclose(qpoint_weights, np.array([0.5, 0.5]))

    def test_epc_mesh_data_roundtrip(self, tmp_path):
        """Verify EPCMeshData save/load cycle."""
        epc_data = EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0, 1]),
            frequencies=np.array([[1.0, 2.0]]),
            eigenvalues_k=np.array([[1.0, 2.0], [1.1, 2.1]]),
            eigenvalues_kq=np.array([[[1.1, 2.1], [1.2, 2.2]]]),
            coupling_matrix=np.ones((1, 2, 2, 2, 2), dtype=complex),
            coupling_strength=np.ones((1, 2, 2, 2, 2)),
            metadata={"source": "test"},
        )
        mesh_data = EPCMeshData.from_epc_data(
            epc_data,
            kpoint_weights=np.array([1.0, 1.0]),
            qpoint_weights=np.array([2.0]),
        )
        path = tmp_path / "epc_mesh.npz"
        mesh_data.save_npz(path)
        loaded = EPCMeshData.load_npz(path)

        np.testing.assert_allclose(loaded.coupling_matrix, mesh_data.coupling_matrix)
        assert loaded.metadata["schema"] == "deeptb.epc_mesh_data"

    def test_epc_path_data_roundtrip(self, tmp_path):
        """Verify EPCPathData save/load cycle."""
        epc_data = EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            qpoints=np.array([[0.25, 0.25, 0.25]]),
            band_indices=np.array([0, 1]),
            frequencies=np.array([[1.0, 2.0]]),
            eigenvalues_k=np.array([[1.0, 2.0], [1.5, 2.5]]),
            eigenvalues_kq=np.array([[[1.25, 2.25], [1.75, 2.75]]]),
            coupling_matrix=np.ones((1, 2, 2, 2, 2), dtype=complex),
            coupling_strength=np.ones((1, 2, 2, 2, 2)),
        )
        path_data = EPCPathData.from_epc_data(
            epc_data,
            path_segment_structure=[2],
        )
        path = tmp_path / "epc_path.npz"
        path_data.save_npz(path)
        loaded = EPCPathData.load_npz(path)

        np.testing.assert_allclose(loaded.coupling_matrix, path_data.coupling_matrix)
        assert loaded.metadata["schema"] == "deeptb.epc_path_data"


# ============================================================================
# Linewidth and relaxation time tests
# ============================================================================

class TestLinewidth:
    """Verify linewidth calculations."""

    def test_linewidth_gaussian(self):
        """Verify Gaussian broadening calculation."""
        epc_data = EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0, 1]),
            frequencies=np.array([[1.0, 2.0]]),
            eigenvalues_k=np.array([[0.10, 0.20]]),
            eigenvalues_kq=np.array([[[0.11, 0.19]]]),
            coupling_matrix=np.ones((1, 1, 2, 2, 2), dtype=complex),
            coupling_strength=np.array(
                [[[[[0.10, 0.20], [0.30, 0.40]], [[0.50, 0.60], [0.70, 0.80]]]]], dtype=float
            ),
        )
        linewidth = compute_linewidth(
            epc_data,
            chemical_potential=0.15,
            temperature=0.025,
            sigma=0.01,
            broadening="gaussian",
        )

        assert linewidth.linewidth.shape == (1, 2)
        assert np.all(np.isfinite(linewidth.linewidth))

    def test_linewidth_mesh_consistency(self):
        """Verify linewidth mesh matches single point."""
        epc_data = EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0, 1]),
            frequencies=np.array([[1.0, 2.0]]),
            eigenvalues_k=np.array([[0.10, 0.20]]),
            eigenvalues_kq=np.array([[[0.11, 0.19]]]),
            coupling_matrix=np.ones((1, 1, 2, 2, 2), dtype=complex),
            coupling_strength=np.array(
                [[[[[0.10, 0.20], [0.30, 0.40]], [[0.50, 0.60], [0.70, 0.80]]]]], dtype=float
            ),
        )

        single = compute_linewidth(epc_data, chemical_potential=0.15, temperature=0.025, sigma=0.01)
        mesh = compute_linewidth_mesh(
            epc_data,
            kpoint_weights=np.array([1.0]),
            qpoint_weights=np.array([1.0]),
            chemical_potential=0.15,
            temperature=0.025,
            sigma=0.01,
        )

        np.testing.assert_allclose(mesh.linewidth, single.linewidth)

    def test_relaxation_time_convention(self):
        """Verify relaxation time = hbar / (2 * linewidth)."""
        epc_data = EPCData(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            qpoints=np.array([[0.0, 0.0, 0.0]]),
            band_indices=np.array([0, 1]),
            frequencies=np.array([[1.0, 2.0]]),
            eigenvalues_k=np.array([[0.10, 0.20]]),
            eigenvalues_kq=np.array([[[0.11, 0.19]]]),
            coupling_matrix=np.ones((1, 1, 2, 2, 2), dtype=complex),
            coupling_strength=np.array(
                [[[[[0.10, 0.20], [0.30, 0.40]], [[0.50, 0.60], [0.70, 0.80]]]]], dtype=float
            ),
        )

        from dptb.postprocess.unified.eph import HBAR_EV_S

        linewidth = compute_linewidth(epc_data, chemical_potential=0.15, temperature=0.025, sigma=0.01)
        relaxation_time = compute_relaxation_time(linewidth)

        expected_tau = HBAR_EV_S / (2.0 * linewidth.linewidth)
        np.testing.assert_allclose(relaxation_time.relaxation_time, expected_tau, rtol=1e-14)


# ============================================================================
# Constants test
# ============================================================================

class TestConstants:
    """Verify EPC physical constants."""

    def test_epc_prefactor(self):
        """Verify EPC_PREFAC_AMU_THZ constant."""
        expected = (
            scipy_constants.hbar
            / scipy_constants.physical_constants["atomic mass constant"][0]
            / scipy_constants.tera
            / scipy_constants.angstrom**2
        )
        np.testing.assert_allclose(EPC_PREFAC_AMU_THZ, expected, rtol=1e-15)
