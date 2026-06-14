import json

import numpy as np
import pytest

from dptb.tests.epc_test_utils import (
    HBAR_EV_S,
    LINEWIDTH_MESH_NPZ_SCHEMA_VERSION,
    LINEWIDTH_NPZ_SCHEMA_VERSION,
    LINEWIDTH_PATH_NPZ_SCHEMA_VERSION,
    LinewidthData,
    LinewidthMeshData,
    LinewidthPathData,
    MINIMAL_EPC_FIXTURE,
    RELAXATION_TIME_MESH_NPZ_SCHEMA_VERSION,
    RELAXATION_TIME_NPZ_SCHEMA_VERSION,
    RELAXATION_TIME_PATH_NPZ_SCHEMA_VERSION,
    RelaxationTimeData,
    RelaxationTimeMeshData,
    RelaxationTimePathData,
    THZ_TO_EV,
    _chunk_artifact_mesh_data,
    _manual_linewidth,
    _minimal_fixture_epc_data,
    _small_linewidth_epc_data,
    _small_linewidth_epc_mesh_data,
    _small_linewidth_epc_path_data,
    compute_linewidth,
    compute_linewidth_mesh,
    compute_linewidth_mesh_chunked_artifact,
    compute_linewidth_path,
    compute_relaxation_time,
    compute_relaxation_time_mesh,
    compute_relaxation_time_path,
    save_epc_mesh_chunked_artifact,
)

def test_minimal_in_repo_epc_fixture_matches_linewidth_reference():
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


@pytest.mark.parametrize("axis", ["q", "k"])
@pytest.mark.parametrize("mode_resolved", [False, True])
def test_compute_linewidth_mesh_chunked_artifact_matches_full_mesh(axis, mode_resolved, tmp_path):
    mesh_data = _chunk_artifact_mesh_data()
    artifact_dir = tmp_path / f"{axis}_artifact"
    save_epc_mesh_chunked_artifact(mesh_data, artifact_dir, axis=axis, chunk_size=1)

    expected = compute_linewidth_mesh(
        mesh_data,
        chemical_potential=0.2,
        temperature=0.03,
        sigma=0.05,
        mode_resolved=mode_resolved,
    )
    actual = compute_linewidth_mesh_chunked_artifact(
        artifact_dir,
        chemical_potential=0.2,
        temperature=0.03,
        sigma=0.05,
        mode_resolved=mode_resolved,
    )

    np.testing.assert_allclose(actual.linewidth, expected.linewidth)
    np.testing.assert_allclose(actual.absorption, expected.absorption)
    np.testing.assert_allclose(actual.emission, expected.emission)
    np.testing.assert_allclose(actual.kpoints, expected.kpoints)
    np.testing.assert_allclose(actual.kpoint_weights, expected.kpoint_weights)
    np.testing.assert_array_equal(actual.band_indices, expected.band_indices)
    assert actual.metadata["summary_first"] is True
    assert actual.metadata["artifact_axis"] == axis
    assert actual.metadata["artifact_chunk_count"] == (2 if axis == "q" else 3)


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
