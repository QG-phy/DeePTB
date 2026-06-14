from dptb.tests.epc_test_utils import *

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


def test_phonons_from_phonopy_uses_requested_qpoints_when_result_omits_them():
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

        def get_qpoints_dict(self):
            eigenvectors = np.arange(2 * 6 * 6).reshape(2, 6, 6)
            return {
                "frequencies": np.ones((2, 6)),
                "eigenvectors": eigenvectors,
            }

    qpoints = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    phonons = Phonons.from_phonopy(_Phonopy(), qpoints=qpoints)

    np.testing.assert_allclose(phonons.qpoints, qpoints)
    assert phonons.frequencies.shape == (2, 6)
    assert phonons.eigenvectors.shape == (2, 6, 2, 3)


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


@pytest.mark.parametrize(("axis", "chunk_size", "expected_chunks"), [("k", 2, 2), ("q", 1, 2)])
def test_epc_mesh_chunked_artifact_roundtrip(axis, chunk_size, expected_chunks, tmp_path):
    mesh_data = _chunk_artifact_mesh_data()
    artifact_dir = tmp_path / f"{axis}_artifact"

    save_epc_mesh_chunked_artifact(mesh_data, artifact_dir, axis=axis, chunk_size=chunk_size)
    loaded = load_epc_mesh_chunked_artifact(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "deeptb.epc_mesh_chunked_artifact"
    assert manifest["schema_version"] == EPC_MESH_CHUNKED_ARTIFACT_SCHEMA_VERSION
    assert manifest["axis"] == axis
    assert manifest["chunk_count"] == expected_chunks
    assert (artifact_dir / "weights.npz").exists()
    np.testing.assert_allclose(loaded.kpoints, mesh_data.kpoints)
    np.testing.assert_allclose(loaded.qpoints, mesh_data.qpoints)
    np.testing.assert_allclose(loaded.kpoint_weights, mesh_data.kpoint_weights)
    np.testing.assert_allclose(loaded.qpoint_weights, mesh_data.qpoint_weights)
    np.testing.assert_allclose(loaded.eigenvalues_k, mesh_data.eigenvalues_k)
    np.testing.assert_allclose(loaded.eigenvalues_kq, mesh_data.eigenvalues_kq)
    np.testing.assert_allclose(loaded.coupling_matrix, mesh_data.coupling_matrix)
    np.testing.assert_allclose(loaded.coupling_strength, mesh_data.coupling_strength)
    assert loaded.metadata["chunked_artifact"] is True
    assert loaded.metadata["artifact_axis"] == axis
    assert loaded.metadata["artifact_chunk_count"] == expected_chunks


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


@pytest.mark.parametrize(
    ("data_cls", "payload"),
    [
        pytest.param(
            Phonons,
            {
                "ph_qpoints": np.array([[0.0, 0.0, 0.0]]),
                "ph_frequencies": np.array([[1.0, 2.0, 3.0]]),
                "ph_eigenvectors": np.ones((1, 3, 1, 3), dtype=complex),
                "ph_masses": np.array([1.0]),
            },
            id="phonons",
        ),
        pytest.param(
            TransportData,
            {
                "transport_conductivity": np.eye(3),
                "transport_carrier_density": np.array(1.0),
            },
            id="transport",
        ),
    ],
)
def test_npz_metadata_json_must_be_scalar_json_object(data_cls, payload, tmp_path):

    missing_metadata = tmp_path / "missing_metadata.npz"
    np.savez(missing_metadata, **payload)
    with pytest.raises(ValueError, match="metadata_json"):
        data_cls.load_npz(missing_metadata)

    array_metadata = tmp_path / "array_metadata.npz"
    np.savez(array_metadata, **payload, metadata_json=np.array(["{}", "{}"]))
    with pytest.raises(ValueError, match="scalar JSON object"):
        data_cls.load_npz(array_metadata)

    invalid_json = tmp_path / "invalid_metadata.npz"
    np.savez(invalid_json, **payload, metadata_json=np.array("{not-json"))
    with pytest.raises(ValueError, match="valid JSON"):
        data_cls.load_npz(invalid_json)

    non_object_json = tmp_path / "non_object_metadata.npz"
    np.savez(non_object_json, **payload, metadata_json=np.array("[]"))
    with pytest.raises(ValueError, match="JSON object"):
        data_cls.load_npz(non_object_json)

    object_metadata = tmp_path / "object_metadata.npz"
    np.savez(object_metadata, **payload, metadata_json=np.array({"schema": "object"}, dtype=object))
    with pytest.raises(ValueError, match="metadata_json.*Object arrays cannot be loaded"):
        data_cls.load_npz(object_metadata)


def test_epc_npz_loaders_use_pickle_free_numpy_loading(monkeypatch, tmp_path):
    persistent_data_classes = [
        Phonons,
        EPCData,
        EPCMeshData,
        EPCPathData,
        LinewidthData,
        LinewidthMeshData,
        LinewidthPathData,
        RelaxationTimeData,
        RelaxationTimeMeshData,
        RelaxationTimePathData,
        TransportData,
        TransportScanData,
        MobilityData,
        MobilityScanData,
        SubspaceCouplingData,
    ]

    load_kwargs = []

    def fake_load(*args, **kwargs):
        load_kwargs.append(kwargs)
        raise RuntimeError("stop after checking np.load options")

    monkeypatch.setattr(np, "load", fake_load)

    for data_cls in persistent_data_classes:
        with pytest.raises(RuntimeError, match="stop after checking"):
            data_cls.load_npz(tmp_path / f"{data_cls.__name__}.npz")

    assert len(load_kwargs) == len(persistent_data_classes)
    for kwargs in load_kwargs:
        assert kwargs.get("allow_pickle") is False


def test_epc_npz_loader_rejects_object_arrays_without_pickle(tmp_path):
    payload = {
        "ph_qpoints": np.array([[0.0, 0.0, 0.0]], dtype=object),
        "ph_frequencies": np.array([[1.0]]),
        "el_kpoints": np.array([[0.0, 0.0, 0.0]]),
        "el_band_indices": np.array([0]),
        "el_eigenvalues_k": np.array([[0.0]]),
        "el_eigenvalues_kq": np.array([[[0.0]]]),
        "elph_coupling_matrix": np.ones((1, 1, 1, 1, 1), dtype=complex),
        "elph_coupling_strength": np.ones((1, 1, 1, 1, 1)),
        "metadata_json": np.array('{"schema": "deeptb.epc_data"}'),
    }
    path = tmp_path / "object_array_epc.npz"
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="ph_qpoints.*Object arrays cannot be loaded"):
        EPCData.load_npz(path)


@pytest.mark.parametrize(
    ("data_cls", "object_key", "payload"),
    [
        pytest.param(
            Phonons,
            "ph_qpoints",
            {
                "ph_qpoints": np.array([[0.0, 0.0, 0.0]], dtype=object),
                "ph_frequencies": np.array([[1.0]]),
                "ph_eigenvectors": np.ones((1, 1, 1, 3), dtype=complex),
                "ph_masses": np.array([1.0]),
                "metadata_json": np.array('{"schema": "deeptb.phonons"}'),
            },
            id="phonons",
        ),
        pytest.param(
            LinewidthMeshData,
            "el_kpoints",
            {
                "elph_mesh_linewidth": np.array([[0.1]]),
                "elph_mesh_linewidth_absorption": np.array([[0.0]]),
                "elph_mesh_linewidth_emission": np.array([[0.1]]),
                "el_kpoints": np.array([[0.0, 0.0, 0.0]], dtype=object),
                "el_kpoint_weights": np.array([1.0]),
                "el_band_indices": np.array([0]),
                "metadata_json": np.array('{"schema": "deeptb.epc_mesh_linewidth"}'),
            },
            id="linewidth-mesh",
        ),
        pytest.param(
            TransportScanData,
            "chemical_potentials",
            {
                "transport_scan_conductivity": np.ones((1, 1, 3, 3)),
                "transport_scan_carrier_density": np.ones((1, 1)),
                "chemical_potentials": np.array([0.0], dtype=object),
                "temperatures": np.array([0.1]),
                "metadata_json": np.array('{"schema": "deeptb.epc_transport_scan"}'),
            },
            id="transport-scan",
        ),
        pytest.param(
            MobilityData,
            "mobility_carrier_density",
            {
                "mobility_conductivity": np.eye(3),
                "mobility_tensor": np.eye(3),
                "mobility_carrier_density": np.array(1.0, dtype=object),
                "metadata_json": np.array('{"schema": "deeptb.epc_mobility"}'),
            },
            id="mobility",
        ),
    ],
)
def test_representative_epc_npz_loaders_reject_object_arrays_without_pickle(
    data_cls, object_key, payload, tmp_path
):
    path = tmp_path / f"object_array_{object_key}.npz"
    np.savez(path, **payload)

    with pytest.raises(ValueError, match=f"{object_key}.*Object arrays cannot be loaded"):
        data_cls.load_npz(path)


def test_phonons_accepts_scalar_mass_for_single_atom():
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0]]),
        frequencies=np.array([[1.0, 2.0, 3.0]]),
        eigenvectors=np.ones((1, 3, 1, 3), dtype=complex),
        masses=np.array(12.0),
    )

    np.testing.assert_allclose(phonons.masses, np.array([12.0]))


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


@pytest.mark.parametrize(("axis", "chunk_size"), [("q", 1), ("k", 1)])
def test_electron_phonon_accessor_compute_mesh_chunked_artifact_matches_full_mesh(axis, chunk_size, tmp_path):
    phonons = Phonons(
        qpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        frequencies=np.array([[1.0], [2.0]]),
        eigenvectors=np.array([[[[1.0, 0.0, 0.0]]], [[[1.0, 0.0, 0.0]]]], dtype=complex),
        masses=np.array([1.0]),
    )
    accessor = EPhAccessor(_FakeSystem())
    mesh_spec = EPCMeshSpec(k_mesh=[3, 1, 1])
    artifact_dir = tmp_path / f"{axis}_streaming_artifact"

    full = accessor.compute_mesh(
        mesh_spec=mesh_spec,
        phonons=phonons,
        bands=[0],
        derivative_provider=_FakeDerivativeProvider(),
    )
    manifest = accessor.compute_mesh_chunked_artifact(
        mesh_spec=mesh_spec,
        phonons=phonons,
        directory=artifact_dir,
        axis=axis,
        chunk_size=chunk_size,
        bands=[0],
        derivative_provider=_FakeDerivativeProvider(),
    )
    loaded = load_epc_mesh_chunked_artifact(artifact_dir)

    assert manifest["schema"] == "deeptb.epc_mesh_chunked_artifact"
    assert manifest["axis"] == axis
    assert manifest["mesh_metadata"]["source"] == "deeptb.eph.compute_mesh_chunked_artifact"
    assert manifest["mesh_metadata"]["streaming_artifact"] is True
    assert (artifact_dir / "manifest.json").exists()
    assert (artifact_dir / "weights.npz").exists()
    np.testing.assert_allclose(loaded.kpoints, full.kpoints)
    np.testing.assert_allclose(loaded.qpoints, full.qpoints)
    np.testing.assert_allclose(loaded.kpoint_weights, full.kpoint_weights)
    np.testing.assert_allclose(loaded.qpoint_weights, full.qpoint_weights)
    np.testing.assert_allclose(loaded.eigenvalues_k, full.eigenvalues_k)
    np.testing.assert_allclose(loaded.eigenvalues_kq, full.eigenvalues_kq)
    np.testing.assert_allclose(loaded.coupling_matrix, full.coupling_matrix)
    np.testing.assert_allclose(loaded.coupling_strength, full.coupling_strength)
    assert loaded.metadata["source"] == "deeptb.eph.epc_mesh_chunked_artifact"
    assert loaded.metadata["streaming_artifact"] is True
    assert loaded.metadata["artifact_axis"] == axis
