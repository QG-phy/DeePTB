from dptb.tests.epc_test_utils import *

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


def test_compute_serta_transport_scan_matches_single_point_results():
    eigenvalues = np.array([[0.1, 0.2], [0.3, 0.4]])
    velocities = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, 3.0], [1.0, 1.0, 0.0]],
        ]
    )
    linewidth = np.array([[0.01, 0.02], [0.03, 0.04]])
    chemical_potentials = np.array([0.15, 0.25])
    temperatures = np.array([0.03, 0.05])

    scan = compute_serta_transport_scan(
        eigenvalues=eigenvalues,
        velocities=velocities,
        linewidth=linewidth,
        chemical_potentials=chemical_potentials,
        temperatures=temperatures,
        kpoint_weights=np.array([2.0, 1.0]),
        spin_degeneracy=2,
        volume=5.0,
    )
    point = compute_serta_conductivity(
        eigenvalues=eigenvalues,
        velocities=velocities,
        linewidth=linewidth,
        chemical_potential=chemical_potentials[1],
        temperature=temperatures[0],
        kpoint_weights=np.array([2.0, 1.0]),
        spin_degeneracy=2,
        volume=5.0,
    )

    assert scan.conductivity.shape == (2, 2, 3, 3)
    assert scan.carrier_density.shape == (2, 2)
    np.testing.assert_allclose(scan.conductivity[1, 0], point.conductivity)
    np.testing.assert_allclose(scan.carrier_density[1, 0], point.carrier_density)
    np.testing.assert_allclose(scan.chemical_potentials, chemical_potentials)
    np.testing.assert_allclose(scan.temperatures, temperatures)
    assert scan.metadata["schema"] == "deeptb.epc_transport_scan"
    assert scan.metadata["schema_version"] == TRANSPORT_SCAN_NPZ_SCHEMA_VERSION
    assert scan.metadata["chemical_potential_count"] == 2
    assert scan.metadata["temperature_count"] == 2
    assert scan.metadata["linewidth_scan_convention"] == "fixed_linewidth"


def test_transport_scan_data_npz_roundtrip(tmp_path):
    scan = TransportScanData(
        conductivity=np.ones((2, 1, 3, 3)),
        carrier_density=np.ones((2, 1)) * 3.0,
        chemical_potentials=np.array([0.0, 0.1]),
        temperatures=np.array([0.05]),
        metadata={"source": "unit-test"},
    )
    path = tmp_path / "transport_scan.npz"
    scan.save_npz(path)
    loaded = TransportScanData.load_npz(path)

    np.testing.assert_allclose(loaded.conductivity, scan.conductivity)
    np.testing.assert_allclose(loaded.carrier_density, scan.carrier_density)
    np.testing.assert_allclose(loaded.chemical_potentials, scan.chemical_potentials)
    np.testing.assert_allclose(loaded.temperatures, scan.temperatures)
    assert loaded.metadata["schema"] == "deeptb.epc_transport_scan"
    assert loaded.metadata["schema_version"] == TRANSPORT_SCAN_NPZ_SCHEMA_VERSION
    assert loaded.metadata["source"] == "unit-test"


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


@pytest.mark.parametrize(("axis", "velocity_source", "system_cls"), [
    ("q", "finite_difference", _LinearBandSystem),
    ("k", "hamiltonian_derivative", _DerivativeBandSystem),
])
def test_compute_serta_transport_from_epc_mesh_chunked_artifact_matches_full_mesh(
    axis,
    velocity_source,
    system_cls,
    tmp_path,
):
    system = system_cls()
    mesh_data = _chunk_artifact_mesh_data()
    artifact_dir = tmp_path / f"{axis}_artifact"
    save_epc_mesh_chunked_artifact(mesh_data, artifact_dir, axis=axis, chunk_size=1)

    result = compute_serta_transport_from_epc_mesh_chunked_artifact(
        system=system,
        directory=artifact_dir,
        chemical_potential=0.2,
        temperature=0.03,
        sigma=0.05,
        spin_degeneracy=2,
        volume=5.0,
        velocity_source=velocity_source,
    )
    linewidth = compute_linewidth_mesh(
        mesh_data,
        chemical_potential=0.2,
        temperature=0.03,
        sigma=0.05,
    )
    eigenvalues = (
        system.band_offsets[None, mesh_data.band_indices]
        + mesh_data.kpoints @ system.band_slopes[mesh_data.band_indices].T
    )
    expected = compute_serta_conductivity(
        eigenvalues=eigenvalues,
        velocities=np.repeat(system.band_slopes[mesh_data.band_indices][None, :, :], mesh_data.kpoints.shape[0], axis=0),
        linewidth=linewidth.linewidth,
        chemical_potential=0.2,
        temperature=0.03,
        kpoint_weights=mesh_data.kpoint_weights,
        spin_degeneracy=2,
        volume=5.0,
    )

    np.testing.assert_allclose(result.conductivity, expected.conductivity)
    np.testing.assert_allclose(result.carrier_density, expected.carrier_density)
    assert result.metadata["summary_first"] is True
    assert result.metadata["artifact_axis"] == axis
    assert result.metadata["velocity_source"] == velocity_source
    assert result.metadata["source"] == "deeptb.eph.compute_serta_transport_from_epc_mesh_chunked_artifact"


@pytest.mark.parametrize(("axis", "velocity_source", "system_cls"), [
    ("q", "finite_difference", _LinearBandSystem),
    ("k", "hamiltonian_derivative", _DerivativeBandSystem),
])
def test_compute_serta_transport_scan_from_epc_mesh_chunked_artifact_matches_full_mesh(
    axis,
    velocity_source,
    system_cls,
    tmp_path,
):
    system = system_cls()
    mesh_data = _chunk_artifact_mesh_data()
    artifact_dir = tmp_path / f"{axis}_artifact"
    save_epc_mesh_chunked_artifact(mesh_data, artifact_dir, axis=axis, chunk_size=1)
    chemical_potentials = [0.2, 0.25]
    temperatures = [0.03, 0.04]

    result = compute_serta_transport_scan_from_epc_mesh_chunked_artifact(
        system=system,
        directory=artifact_dir,
        chemical_potentials=chemical_potentials,
        temperatures=temperatures,
        sigma=0.05,
        spin_degeneracy=2,
        volume=5.0,
        velocity_source=velocity_source,
    )
    linewidth = compute_linewidth_mesh(
        mesh_data,
        chemical_potential=chemical_potentials[0],
        temperature=temperatures[0],
        sigma=0.05,
    )
    eigenvalues = (
        system.band_offsets[None, mesh_data.band_indices]
        + mesh_data.kpoints @ system.band_slopes[mesh_data.band_indices].T
    )
    expected = compute_serta_transport_scan(
        eigenvalues=eigenvalues,
        velocities=np.repeat(system.band_slopes[mesh_data.band_indices][None, :, :], mesh_data.kpoints.shape[0], axis=0),
        linewidth=linewidth.linewidth,
        chemical_potentials=chemical_potentials,
        temperatures=temperatures,
        kpoint_weights=mesh_data.kpoint_weights,
        spin_degeneracy=2,
        volume=5.0,
    )

    np.testing.assert_allclose(result.conductivity, expected.conductivity)
    np.testing.assert_allclose(result.carrier_density, expected.carrier_density)
    np.testing.assert_allclose(result.chemical_potentials, expected.chemical_potentials)
    np.testing.assert_allclose(result.temperatures, expected.temperatures)
    assert result.metadata["summary_first"] is True
    assert result.metadata["artifact_axis"] == axis
    assert result.metadata["velocity_source"] == velocity_source
    assert result.metadata["linewidth_reference_chemical_potential"] == chemical_potentials[0]
    assert result.metadata["linewidth_reference_temperature"] == temperatures[0]
    assert result.metadata["source"] == "deeptb.eph.compute_serta_transport_scan_from_epc_mesh_chunked_artifact"


@pytest.mark.parametrize(("axis", "velocity_source", "system_cls"), [
    ("q", "finite_difference", _LinearBandSystem),
    ("k", "hamiltonian_derivative", _DerivativeBandSystem),
])
def test_compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact_matches_full_mesh(
    axis,
    velocity_source,
    system_cls,
    tmp_path,
):
    system = system_cls()
    mesh_data = _chunk_artifact_mesh_data()
    artifact_dir = tmp_path / f"{axis}_artifact"
    save_epc_mesh_chunked_artifact(mesh_data, artifact_dir, axis=axis, chunk_size=1)
    chemical_potentials = [0.2, 0.25]
    temperatures = [0.03, 0.04]

    result = compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact(
        system=system,
        directory=artifact_dir,
        chemical_potentials=chemical_potentials,
        temperatures=temperatures,
        sigma=0.05,
        spin_degeneracy=2,
        volume=5.0,
        velocity_source=velocity_source,
    )

    eigenvalues = (
        system.band_offsets[None, mesh_data.band_indices]
        + mesh_data.kpoints @ system.band_slopes[mesh_data.band_indices].T
    )
    velocities = np.repeat(
        system.band_slopes[mesh_data.band_indices][None, :, :],
        mesh_data.kpoints.shape[0],
        axis=0,
    )
    expected_conductivity = np.zeros((len(chemical_potentials), len(temperatures), 3, 3))
    expected_carrier_density = np.zeros((len(chemical_potentials), len(temperatures)))
    for imu, chemical_potential in enumerate(chemical_potentials):
        for itemperature, temperature in enumerate(temperatures):
            linewidth = compute_linewidth_mesh(
                mesh_data,
                chemical_potential=chemical_potential,
                temperature=temperature,
                sigma=0.05,
            )
            expected = compute_serta_conductivity(
                eigenvalues=eigenvalues,
                velocities=velocities,
                linewidth=linewidth.linewidth,
                chemical_potential=chemical_potential,
                temperature=temperature,
                kpoint_weights=mesh_data.kpoint_weights,
                spin_degeneracy=2,
                volume=5.0,
            )
            expected_conductivity[imu, itemperature] = expected.conductivity
            expected_carrier_density[imu, itemperature] = expected.carrier_density

    np.testing.assert_allclose(result.conductivity, expected_conductivity)
    np.testing.assert_allclose(result.carrier_density, expected_carrier_density)
    np.testing.assert_allclose(result.chemical_potentials, chemical_potentials)
    np.testing.assert_allclose(result.temperatures, temperatures)
    assert result.metadata["summary_first"] is True
    assert result.metadata["artifact_axis"] == axis
    assert result.metadata["velocity_source"] == velocity_source
    assert result.metadata["linewidth_scan_convention"] == "per_scan_point_recomputed"
    assert (
        result.metadata["source"]
        == "deeptb.eph.compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact"
    )


def test_chunked_artifact_transport_scan_linewidth_convention_metadata_differs(tmp_path):
    system = _LinearBandSystem()
    mesh_data = _chunk_artifact_mesh_data()
    artifact_dir = tmp_path / "q_artifact"
    save_epc_mesh_chunked_artifact(mesh_data, artifact_dir, axis="q", chunk_size=1)

    fixed = compute_serta_transport_scan_from_epc_mesh_chunked_artifact(
        system=system,
        directory=artifact_dir,
        chemical_potentials=[0.2, 0.25],
        temperatures=[0.03, 0.04],
        sigma=0.05,
    )
    recomputed = compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact(
        system=system,
        directory=artifact_dir,
        chemical_potentials=[0.2, 0.25],
        temperatures=[0.03, 0.04],
        sigma=0.05,
    )

    assert fixed.metadata["linewidth_scan_convention"] == "fixed_linewidth"
    assert recomputed.metadata["linewidth_scan_convention"] == "per_scan_point_recomputed"
    assert "linewidth_reference_chemical_potential" in fixed.metadata
    assert "linewidth_reference_chemical_potential" not in recomputed.metadata


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
    assert loaded.metadata["conductivity_unit"] == "internal_SERTA_fractional_k"
    assert loaded.metadata["carrier_density_unit"] == "1/input_volume"

    with np.load(path, allow_pickle=False) as data:
        assert "transport_conductivity" in data
        assert "transport_carrier_density" in data
        assert "metadata_json" in data
