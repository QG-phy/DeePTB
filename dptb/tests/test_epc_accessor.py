import numpy as np

from dptb.tests.epc_test_utils import (
    EPCKChunkSpec,
    EPCMeshData,
    EPCMeshSpec,
    EPCPathData,
    EPCQChunkSpec,
    EPhAccessor,
    Phonons,
    _FakeDerivativeProvider,
    _FakeSystem,
    _epc_k_chunk,
    _epc_q_chunk,
    _single_mode_phonons,
    build_k_chunk_specs,
    build_q_chunk_specs,
    concat_epc_k_chunks,
    concat_epc_q_chunks,
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


def test_electron_phonon_accessor_compute_mesh_q_chunked_matches_full_mesh():
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        frequencies=np.array([[1.0], [2.0], [3.0]]),
        eigenvectors=np.array(
            [[[[1.0, 0.0, 0.0]]], [[[1.0, 0.0, 0.0]]], [[[1.0, 0.0, 0.0]]]],
            dtype=complex,
        ),
        masses=np.array([1.0]),
    )
    accessor = EPhAccessor(_FakeSystem())

    full = accessor.compute_mesh(
        mesh_spec=EPCMeshSpec(k_mesh=[2, 1, 1]),
        phonons=phonons,
        bands=[0],
        derivative_provider=_FakeDerivativeProvider(),
    )
    chunked = accessor.compute_mesh(
        mesh_spec=EPCMeshSpec(k_mesh=[2, 1, 1], q_chunk_size=1),
        phonons=phonons,
        bands=[0],
        derivative_provider=_FakeDerivativeProvider(),
    )

    np.testing.assert_allclose(chunked.kpoints, full.kpoints)
    np.testing.assert_allclose(chunked.qpoints, full.qpoints)
    np.testing.assert_allclose(chunked.kpoint_weights, full.kpoint_weights)
    np.testing.assert_allclose(chunked.qpoint_weights, full.qpoint_weights)
    np.testing.assert_allclose(chunked.eigenvalues_k, full.eigenvalues_k)
    np.testing.assert_allclose(chunked.eigenvalues_kq, full.eigenvalues_kq)
    np.testing.assert_allclose(chunked.coupling_matrix, full.coupling_matrix)
    np.testing.assert_allclose(chunked.coupling_strength, full.coupling_strength)
    assert chunked.metadata["chunked"] is True
    assert chunked.metadata["execution"] == "serial_q_chunked"
    assert chunked.metadata["chunk_axis"] == "q"
    assert chunked.metadata["chunks"] == [
        {"chunk_index": 0, "q_start": 0, "q_stop": 1},
        {"chunk_index": 1, "q_start": 1, "q_stop": 2},
        {"chunk_index": 2, "q_start": 2, "q_stop": 3},
    ]


def test_electron_phonon_accessor_compute_mesh_qk_chunked_matches_full_mesh():
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        frequencies=np.array([[1.0], [2.0]]),
        eigenvectors=np.array([[[[1.0, 0.0, 0.0]]], [[[1.0, 0.0, 0.0]]]], dtype=complex),
        masses=np.array([1.0]),
    )
    accessor = EPhAccessor(_FakeSystem())

    full = accessor.compute_mesh(
        mesh_spec=EPCMeshSpec(k_mesh=[3, 1, 1]),
        phonons=phonons,
        bands=[0],
        derivative_provider=_FakeDerivativeProvider(),
    )
    chunked = accessor.compute_mesh(
        mesh_spec=EPCMeshSpec(k_mesh=[3, 1, 1], chunk_size=1, q_chunk_size=1),
        phonons=phonons,
        bands=[0],
        derivative_provider=_FakeDerivativeProvider(),
    )

    np.testing.assert_allclose(chunked.kpoints, full.kpoints)
    np.testing.assert_allclose(chunked.qpoints, full.qpoints)
    np.testing.assert_allclose(chunked.eigenvalues_k, full.eigenvalues_k)
    np.testing.assert_allclose(chunked.eigenvalues_kq, full.eigenvalues_kq)
    np.testing.assert_allclose(chunked.coupling_matrix, full.coupling_matrix)
    np.testing.assert_allclose(chunked.coupling_strength, full.coupling_strength)
    assert chunked.metadata["execution"] == "serial_qk_chunked"
    assert chunked.metadata["chunk_axis"] == "q,k"
    assert chunked.metadata["chunks"] == [
        {
            "q_chunk": {"chunk_index": 0, "q_start": 0, "q_stop": 1},
            "k_chunks": [
                {"chunk_index": 0, "k_start": 0, "k_stop": 1},
                {"chunk_index": 1, "k_start": 1, "k_stop": 2},
                {"chunk_index": 2, "k_start": 2, "k_stop": 3},
            ],
        },
        {
            "q_chunk": {"chunk_index": 1, "q_start": 1, "q_stop": 2},
            "k_chunks": [
                {"chunk_index": 0, "k_start": 0, "k_stop": 1},
                {"chunk_index": 1, "k_start": 1, "k_stop": 2},
                {"chunk_index": 2, "k_start": 2, "k_stop": 3},
            ],
        },
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


def test_epc_q_chunk_specs_are_deterministic():
    full = build_q_chunk_specs(3, None)
    assert full == [EPCQChunkSpec(chunk_index=0, q_start=0, q_stop=3)]
    assert full[0].slice == slice(0, 3)
    assert full[0].metadata() == {"chunk_index": 0, "q_start": 0, "q_stop": 3}

    chunks = build_q_chunk_specs(5, 2)
    assert chunks == [
        EPCQChunkSpec(chunk_index=0, q_start=0, q_stop=2),
        EPCQChunkSpec(chunk_index=1, q_start=2, q_stop=4),
        EPCQChunkSpec(chunk_index=2, q_start=4, q_stop=5),
    ]


def test_concat_epc_q_chunks_concatenates_q_axis():
    first = _epc_q_chunk([0.0, 0.5])
    second = _epc_q_chunk([1.0])

    combined = concat_epc_q_chunks([first, second])

    np.testing.assert_allclose(combined.qpoints[:, 0], np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(combined.frequencies[:, 0], np.array([1.0, 1.5, 2.0]))
    np.testing.assert_allclose(combined.eigenvalues_kq[:, 0, 0], np.array([0.0, 0.5, 1.0]))
    assert combined.coupling_matrix.shape == (3, 1, 1, 1, 1)
    assert combined.metadata["chunk_count"] == 2
    assert len(combined.metadata["chunk_sources"]) == 2
