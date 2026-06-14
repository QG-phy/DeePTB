import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import linalg

import dptb.postprocess.unified.eph.providers as eph_providers
from dptb.postprocess.unified.eph.providers import _primitive_to_supercell_from_supercell_to_primitive
from dptb.postprocess.unified.eph.contraction import compute_coupling_matrix
from dptb.postprocess.unified.eph.utils import reshape_phonopy_eigenvectors


DEFAULT_EPH_REFERENCE_ROOT = None
DEFAULT_EPH_SKDATA_ROOT = None


def _external_reference_root() -> Path:
    path = os.environ.get("DEEPTB_EPH_REFERENCE_ROOT")
    if path is None:
        pytest.skip("Set DEEPTB_EPH_REFERENCE_ROOT to the external dftbephy checkout.")
    return Path(path)


def _external_skdata_root() -> Path:
    path = os.environ.get("DEEPTB_EPH_SKDATA_ROOT")
    if path is None:
        pytest.skip("Set DEEPTB_EPH_SKDATA_ROOT to the external matsci SK data root.")
    return Path(path)


def test_graphene_reference_case_coupling_strength():
    if os.environ.get("DEEPTB_RUN_REFERENCE_EPH") != "1":
        pytest.skip("Set DEEPTB_RUN_REFERENCE_EPH=1 to run the external Graphene reference regression.")
    import h5py

    phonopy = pytest.importorskip("phonopy")

    # TODO(epc-fixture): keep the full Graphene reference as an opt-in benchmark;
    # default CI should remain self-contained and use lightweight in-repo fixtures.
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
    primitive_to_supercell_atom = _primitive_to_supercell_from_supercell_to_primitive(
        supercell_to_primitive_atom,
        num_primitive_atoms,
        supercell_atom_to_cell=supercell_atom_to_cell,
        preferred_cell=(num_supercells - 1) // 2,
    )

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
    from dptb.postprocess.unified.eph import DFTBPlusGauge, SupercellFD

    # TODO(epc-fixture): keep the full Graphene reference as an opt-in benchmark;
    # default CI should remain self-contained and use lightweight in-repo fixtures.
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
