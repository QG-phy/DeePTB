import sys
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import linalg
from scipy import constants as scipy_constants

import dptb.postprocess.unified.eph.providers as eph_providers
from dptb.postprocess.unified.eph import (
    EPCData,
    DFTBPlusGauge,
    EPhAccessor,
    Phonons,
    SupercellFD,
)
from dptb.postprocess.unified.eph.contraction import EPC_PREFAC_AMU_THZ, compute_coupling_matrix
from dptb.postprocess.unified.eph.utils import (
    assemble_directed_hk_from_blocks,
    orbital_slices_from_atom_orbs,
    reshape_phonopy_eigenvectors,
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

    with np.load(path, allow_pickle=False) as data:
        assert "elph_coupling_matrix" in data
        assert "metadata_json" in data


class _FakeDerivativeProvider:
    def compute(self, kpoints):
        nk = len(kpoints)
        h_derivatives = np.zeros((nk, 1, 3, 2, 2), dtype=complex)
        h_derivatives[:, 0, 0] = np.eye(2)
        return h_derivatives, None


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


def test_electron_phonon_accessor_compute_coupling_with_mock_derivatives(tmp_path):
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        frequencies=np.array([[1.0]]),
        eigenvectors=np.array([[[[1.0, 0.0, 0.0]]]], dtype=complex),
        masses=np.array([1.0]),
    )
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
    assert output.exists()


def test_electron_phonon_accessor_rejects_scc_v1():
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        frequencies=np.array([[1.0]]),
        eigenvectors=np.array([[[[1.0, 0.0, 0.0]]]], dtype=complex),
        masses=np.array([1.0]),
    )

    with pytest.raises(NotImplementedError, match="SCC-corrected"):
        EPhAccessor(_FakeSystem()).compute_coupling(
            kpoints=np.array([[0.0, 0.0, 0.0]]),
            phonons=phonons,
            use_scc=True,
            derivative_provider=_FakeDerivativeProvider(),
        )


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

    reference_root = Path("/Users/aisiqg/Desktop/work/github/dftbephy")
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

    reference_root = Path("/Users/aisiqg/Desktop/work/github/dftbephy")
    graphene_root = reference_root / "examples" / "Graphene"
    skdata = Path("/Users/aisiqg/Desktop/work/github/matsci-0-3")
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
