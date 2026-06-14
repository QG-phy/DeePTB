import json

import numpy as np
import pytest

from dptb.tests.epc_test_utils import (
    EPCData,
    EPCMeshData,
    EPCMeshSpec,
    EPC_PREFAC_AMU_THZ,
    MINIMAL_EPC_FIXTURE,
    Phonons,
    SupercellFD,
    _FakeSystem,
    _minimal_fixture_epc_data,
    assemble_directed_hk_from_blocks,
    compute_coupling_matrix,
    compute_coupling_strength_summary,
    compute_eliashberg_spectral_function,
    compute_phonon_dos,
    compute_scattering_maps,
    compute_subspace_coupling_strength,
    cumulative_path_coordinates,
    eph_providers,
    find_degenerate_band_groups,
    orbital_slices_from_atom_orbs,
    orbital_slices_from_system,
    reshape_phonopy_eigenvectors,
    scipy_constants,
)

def test_epc_prefactor_from_standard_constants():
    expected = (
        scipy_constants.hbar
        / scipy_constants.physical_constants["atomic mass constant"][0]
        / scipy_constants.tera
        / scipy_constants.angstrom**2
    )
    np.testing.assert_allclose(EPC_PREFAC_AMU_THZ, expected, rtol=1e-15)
    np.testing.assert_allclose(EPC_PREFAC_AMU_THZ, 6.35078, rtol=1e-6)


def test_minimal_in_repo_epc_fixture_matches_analysis_references():
    epc_data = _minimal_fixture_epc_data()

    summary = compute_coupling_strength_summary(epc_data)
    np.testing.assert_allclose(summary["total"], 0.30, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(summary["q_resolved"], np.array([0.30]), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(summary["k_resolved"], np.array([0.30]), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(summary["mode_resolved"], np.array([0.30]), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(summary["final_band_resolved"], np.array([0.05, 0.25]), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(summary["initial_band_resolved"], np.array([0.10, 0.20]), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        summary["band_pair_resolved"],
        np.array([[0.01, 0.04], [0.09, 0.16]]),
        rtol=1e-14,
        atol=1e-14,
    )
    assert summary["metadata"]["source"] == "EPCData.coupling_strength"
    assert summary["metadata"]["input_schema"] == "deeptb.epc_data"

    maps = compute_scattering_maps(epc_data)
    np.testing.assert_allclose(maps["q_mode_resolved"], np.array([[0.30]]), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(maps["q_initial_band_resolved"], np.array([[0.10, 0.20]]), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(maps["q_final_band_resolved"], np.array([[0.05, 0.25]]), rtol=1e-14, atol=1e-14)
    assert maps["metadata"]["convention"] == "coupling_strength_scattering_proxy"


def test_minimal_in_repo_epc_fixture_matches_coupling_contraction_reference():
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
    with pytest.raises(ValueError, match="q_chunk_size"):
        EPCMeshSpec(k_mesh=[1, 1, 1], q_chunk_size=0)


def test_compute_coupling_strength_summary_from_epc_data():
    strength = np.arange(1, 33, dtype=float).reshape(2, 2, 2, 2, 2)
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        band_indices=np.array([3, 4]),
        frequencies=np.ones((2, 2)),
        eigenvalues_k=np.ones((2, 2)),
        eigenvalues_kq=np.ones((2, 2, 2)),
        coupling_matrix=np.sqrt(strength).astype(complex),
        coupling_strength=strength,
        metadata={"source": "unit-test"},
    )

    summary = compute_coupling_strength_summary(epc_data)

    np.testing.assert_allclose(summary["total"], strength.sum())
    np.testing.assert_allclose(summary["q_resolved"], strength.sum(axis=(1, 2, 3, 4)))
    np.testing.assert_allclose(summary["k_resolved"], strength.sum(axis=(0, 2, 3, 4)))
    np.testing.assert_allclose(summary["mode_resolved"], strength.sum(axis=(0, 1, 3, 4)))
    np.testing.assert_allclose(summary["final_band_resolved"], strength.sum(axis=(0, 1, 2, 4)))
    np.testing.assert_allclose(summary["initial_band_resolved"], strength.sum(axis=(0, 1, 2, 3)))
    np.testing.assert_allclose(summary["band_pair_resolved"], strength.sum(axis=(0, 1, 2)))
    np.testing.assert_array_equal(summary["band_indices"], np.array([3, 4]))
    assert summary["metadata"]["source"] == "EPCData.coupling_strength"
    assert summary["metadata"]["weight_convention"] == "unweighted_sum"


def test_compute_coupling_strength_summary_uses_mesh_weights():
    strength = np.ones((2, 2, 1, 1, 1), dtype=float)
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        band_indices=np.array([0]),
        frequencies=np.ones((2, 1)),
        eigenvalues_k=np.ones((2, 1)),
        eigenvalues_kq=np.ones((2, 2, 1)),
        coupling_matrix=np.ones((2, 2, 1, 1, 1), dtype=complex),
        coupling_strength=strength,
    )
    mesh_data = EPCMeshData.from_epc_data(
        epc_data,
        kpoint_weights=np.array([1.0, 3.0]),
        qpoint_weights=np.array([2.0, 6.0]),
    )

    weighted = compute_coupling_strength_summary(mesh_data)
    unweighted = compute_coupling_strength_summary(mesh_data, weighted=False)

    np.testing.assert_allclose(weighted["total"], 1.0)
    np.testing.assert_allclose(weighted["q_resolved"], np.array([0.25, 0.75]))
    np.testing.assert_allclose(weighted["k_resolved"], np.array([0.25, 0.75]))
    np.testing.assert_allclose(unweighted["total"], 4.0)
    assert weighted["metadata"]["source"] == "EPCMeshData.coupling_strength"
    assert weighted["metadata"]["weight_convention"] == "normalized_qpoint_and_kpoint_weights"
    assert unweighted["metadata"]["weight_convention"] == "unweighted_sum"


def test_compute_scattering_maps_matches_manual_reference():
    strength = np.arange(1, 1 + 2 * 2 * 2 * 2 * 2, dtype=float).reshape(2, 2, 2, 2, 2)
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        band_indices=np.array([3, 4]),
        frequencies=np.array([[1.0, 2.0], [3.0, 4.0]]),
        eigenvalues_k=np.ones((2, 2)),
        eigenvalues_kq=np.ones((2, 2, 2)),
        coupling_matrix=np.sqrt(strength).astype(complex),
        coupling_strength=strength,
    )

    result = compute_scattering_maps(epc_data)

    np.testing.assert_allclose(result["q_mode_resolved"], strength.sum(axis=(1, 3, 4)))
    np.testing.assert_allclose(result["k_mode_resolved"], strength.sum(axis=(0, 3, 4)))
    np.testing.assert_allclose(result["q_final_band_resolved"], strength.sum(axis=(1, 2, 4)))
    np.testing.assert_allclose(result["q_initial_band_resolved"], strength.sum(axis=(1, 2, 3)))
    np.testing.assert_allclose(result["q_band_pair_resolved"], strength.sum(axis=(1, 2)))
    np.testing.assert_allclose(result["k_final_band_resolved"], strength.sum(axis=(0, 2, 4)))
    np.testing.assert_allclose(result["k_initial_band_resolved"], strength.sum(axis=(0, 2, 3)))
    np.testing.assert_allclose(result["k_band_pair_resolved"], strength.sum(axis=(0, 2)))
    np.testing.assert_allclose(result["qpoints"], epc_data.qpoints)
    np.testing.assert_allclose(result["kpoints"], epc_data.kpoints)
    assert result["metadata"]["convention"] == "coupling_strength_scattering_proxy"
    assert result["metadata"]["weight_convention"] == "unweighted_sum"


def test_compute_scattering_maps_uses_mesh_weights():
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        band_indices=np.array([0]),
        frequencies=np.ones((2, 1)),
        eigenvalues_k=np.ones((2, 1)),
        eigenvalues_kq=np.ones((2, 2, 1)),
        coupling_matrix=np.ones((2, 2, 1, 1, 1), dtype=complex),
        coupling_strength=np.ones((2, 2, 1, 1, 1)),
    )
    mesh_data = EPCMeshData.from_epc_data(
        epc_data,
        kpoint_weights=np.array([1.0, 3.0]),
        qpoint_weights=np.array([2.0, 6.0]),
    )

    weighted = compute_scattering_maps(mesh_data)
    unweighted = compute_scattering_maps(mesh_data, weighted=False)

    np.testing.assert_allclose(weighted["q_initial_band_resolved"], [[0.25], [0.75]])
    np.testing.assert_allclose(weighted["k_initial_band_resolved"], [[0.25], [0.75]])
    np.testing.assert_allclose(unweighted["q_initial_band_resolved"], [[2.0], [2.0]])
    assert weighted["metadata"]["weight_convention"] == "normalized_qpoint_and_kpoint_weights"
    assert unweighted["metadata"]["weight_convention"] == "unweighted_sum"


def test_compute_eliashberg_spectral_function_matches_manual_reference():
    strength = np.array([[[[[2.0]]]], [[[[4.0]]]]])
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        band_indices=np.array([0]),
        frequencies=np.array([[1.0], [2.0]]),
        eigenvalues_k=np.ones((1, 1)),
        eigenvalues_kq=np.ones((2, 1, 1)),
        coupling_matrix=np.sqrt(strength).astype(complex),
        coupling_strength=strength,
    )
    grid = np.array([1.0, 2.0])
    sigma = 0.5

    result = compute_eliashberg_spectral_function(epc_data, frequency_grid=grid, sigma=sigma)
    expected = (
        2.0 * np.exp(-((grid - 1.0) ** 2) / (2.0 * sigma**2))
        + 4.0 * np.exp(-((grid - 2.0) ** 2) / (2.0 * sigma**2))
    ) / (np.sqrt(2.0 * np.pi) * sigma)

    np.testing.assert_allclose(result["frequency_grid"], grid)
    np.testing.assert_allclose(result["alpha2f"], expected)
    np.testing.assert_allclose(result["mode_resolved_alpha2f"][0], expected)
    assert result["metadata"]["convention"] == "coupling_strength_weighted_phonon_frequency_spectrum"
    assert result["metadata"]["weight_convention"] == "unweighted_sum"


def test_compute_eliashberg_spectral_function_uses_mesh_weights():
    epc_data = EPCData(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        band_indices=np.array([0]),
        frequencies=np.array([[1.0], [2.0]]),
        eigenvalues_k=np.ones((2, 1)),
        eigenvalues_kq=np.ones((2, 2, 1)),
        coupling_matrix=np.ones((2, 2, 1, 1, 1), dtype=complex),
        coupling_strength=np.ones((2, 2, 1, 1, 1)),
    )
    mesh_data = EPCMeshData.from_epc_data(
        epc_data,
        kpoint_weights=np.array([1.0, 3.0]),
        qpoint_weights=np.array([2.0, 6.0]),
    )
    grid = np.array([1.0, 2.0])
    sigma = 0.5

    weighted = compute_eliashberg_spectral_function(mesh_data, frequency_grid=grid, sigma=sigma)
    unweighted = compute_eliashberg_spectral_function(mesh_data, frequency_grid=grid, sigma=sigma, weighted=False)

    weighted_expected = (
        0.25 * np.exp(-((grid - 1.0) ** 2) / (2.0 * sigma**2))
        + 0.75 * np.exp(-((grid - 2.0) ** 2) / (2.0 * sigma**2))
    ) / (np.sqrt(2.0 * np.pi) * sigma)
    np.testing.assert_allclose(weighted["alpha2f"], weighted_expected)
    assert weighted["metadata"]["weight_convention"] == "normalized_qpoint_and_kpoint_weights"
    assert unweighted["metadata"]["weight_convention"] == "unweighted_sum"


def test_cumulative_path_coordinates_uses_fractional_distances():
    np.testing.assert_allclose(
        cumulative_path_coordinates(np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.0], [0.3, 0.4, 0.2]])),
        np.array([0.0, 0.5, 0.7]),
    )


def test_compute_phonon_dos_matches_manual_gaussian_reference():
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        frequencies=np.array([[1.0, 2.0], [3.0, 4.0]]),
        eigenvectors=np.ones((2, 2, 1, 3), dtype=complex),
        masses=np.array([1.0]),
    )
    grid = np.array([1.0, 2.0, 3.0])
    sigma = 0.5

    result = compute_phonon_dos(phonons, frequency_grid=grid, sigma=sigma)

    expected_mode = np.zeros((2, 3), dtype=float)
    for iq in range(2):
        for imode in range(2):
            expected_mode[imode] += 0.5 * np.exp(-((grid - phonons.frequencies[iq, imode]) ** 2) / (2.0 * sigma**2)) / (
                np.sqrt(2.0 * np.pi) * sigma
            )
    np.testing.assert_allclose(result["frequency_grid"], grid)
    np.testing.assert_allclose(result["mode_resolved_dos"], expected_mode)
    np.testing.assert_allclose(result["dos"], expected_mode.sum(axis=0))
    assert result["metadata"]["frequency_unit"] == "THz"
    assert result["metadata"]["dos_unit"] == "THz^-1"
    assert result["metadata"]["weight_convention"] == "uniform_normalized_qpoint_weights"


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


def test_supercell_fd_provider_from_phonopy_mock(monkeypatch):
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
