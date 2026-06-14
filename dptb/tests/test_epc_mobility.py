import numpy as np
import pytest

from dptb.tests.epc_test_utils import (
    MOBILITY_NPZ_SCHEMA_VERSION,
    MOBILITY_SCAN_NPZ_SCHEMA_VERSION,
    MobilityData,
    MobilityScanData,
    _DerivativeBandSystem,
    _LinearBandSystem,
    _chunk_artifact_mesh_data,
    compute_linewidth_mesh,
    compute_serta_mobility_scan_si,
    compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact,
    compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact,
    compute_serta_mobility_si,
    compute_serta_mobility_si_from_epc_mesh_chunked_artifact,
    dptb_constants,
    fractional_band_velocities_to_si,
    save_epc_mesh_chunked_artifact,
)

@pytest.mark.parametrize(
    ("axis", "velocity_source", "system_cls", "dimension", "normalization_kwargs"),
    [
        ("q", "finite_difference", _LinearBandSystem, "3d", {"volume": 5.0}),
        ("k", "hamiltonian_derivative", _DerivativeBandSystem, "2d", {"area": 7.0}),
    ],
)
def test_compute_serta_mobility_si_from_epc_mesh_chunked_artifact_matches_full_mesh(
    axis,
    velocity_source,
    system_cls,
    dimension,
    normalization_kwargs,
    tmp_path,
):
    system = system_cls()
    mesh_data = _chunk_artifact_mesh_data()
    artifact_dir = tmp_path / f"{axis}_artifact"
    save_epc_mesh_chunked_artifact(mesh_data, artifact_dir, axis=axis, chunk_size=1)
    reciprocal_cell = 2.0 * np.pi * np.eye(3)

    result = compute_serta_mobility_si_from_epc_mesh_chunked_artifact(
        system=system,
        directory=artifact_dir,
        reciprocal_cell=reciprocal_cell,
        chemical_potential=0.2,
        temperature=0.03,
        sigma=0.05,
        spin_degeneracy=2,
        dimension=dimension,
        velocity_source=velocity_source,
        **normalization_kwargs,
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
    expected = compute_serta_mobility_si(
        eigenvalues=eigenvalues,
        velocities=np.repeat(system.band_slopes[mesh_data.band_indices][None, :, :], mesh_data.kpoints.shape[0], axis=0),
        linewidth=linewidth.linewidth,
        reciprocal_cell=reciprocal_cell,
        chemical_potential=0.2,
        temperature=0.03,
        kpoint_weights=mesh_data.kpoint_weights,
        spin_degeneracy=2,
        dimension=dimension,
        **normalization_kwargs,
    )

    np.testing.assert_allclose(result.conductivity, expected.conductivity)
    np.testing.assert_allclose(result.mobility, expected.mobility)
    np.testing.assert_allclose(result.carrier_density, expected.carrier_density)
    assert result.metadata["summary_first"] is True
    assert result.metadata["artifact_axis"] == axis
    assert result.metadata["velocity_source"] == velocity_source
    assert result.metadata["source"] == "deeptb.eph.compute_serta_mobility_si_from_epc_mesh_chunked_artifact"


@pytest.mark.parametrize(("axis", "velocity_source", "system_cls"), [
    ("q", "finite_difference", _LinearBandSystem),
    ("k", "hamiltonian_derivative", _DerivativeBandSystem),
])
def test_compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact_matches_full_mesh(
    axis,
    velocity_source,
    system_cls,
    tmp_path,
):
    system = system_cls()
    mesh_data = _chunk_artifact_mesh_data()
    artifact_dir = tmp_path / f"{axis}_artifact"
    save_epc_mesh_chunked_artifact(mesh_data, artifact_dir, axis=axis, chunk_size=1)
    reciprocal_cell = 2.0 * np.pi * np.eye(3)
    chemical_potentials = [0.2, 0.25]
    temperatures = [0.03, 0.04]

    result = compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact(
        system=system,
        directory=artifact_dir,
        reciprocal_cell=reciprocal_cell,
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
    expected = compute_serta_mobility_scan_si(
        eigenvalues=eigenvalues,
        velocities=np.repeat(system.band_slopes[mesh_data.band_indices][None, :, :], mesh_data.kpoints.shape[0], axis=0),
        linewidth=linewidth.linewidth,
        reciprocal_cell=reciprocal_cell,
        chemical_potentials=chemical_potentials,
        temperatures=temperatures,
        kpoint_weights=mesh_data.kpoint_weights,
        spin_degeneracy=2,
        volume=5.0,
    )

    np.testing.assert_allclose(result.conductivity, expected.conductivity)
    np.testing.assert_allclose(result.mobility, expected.mobility)
    np.testing.assert_allclose(result.carrier_density, expected.carrier_density)
    np.testing.assert_allclose(result.chemical_potentials, expected.chemical_potentials)
    np.testing.assert_allclose(result.temperatures, expected.temperatures)
    assert result.metadata["summary_first"] is True
    assert result.metadata["artifact_axis"] == axis
    assert result.metadata["velocity_source"] == velocity_source
    assert result.metadata["linewidth_reference_chemical_potential"] == chemical_potentials[0]
    assert result.metadata["linewidth_reference_temperature"] == temperatures[0]
    assert result.metadata["source"] == "deeptb.eph.compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact"


@pytest.mark.parametrize(("axis", "velocity_source", "system_cls"), [
    ("q", "finite_difference", _LinearBandSystem),
    ("k", "hamiltonian_derivative", _DerivativeBandSystem),
])
def test_compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact_matches_full_mesh(
    axis,
    velocity_source,
    system_cls,
    tmp_path,
):
    system = system_cls()
    mesh_data = _chunk_artifact_mesh_data()
    artifact_dir = tmp_path / f"{axis}_artifact"
    save_epc_mesh_chunked_artifact(mesh_data, artifact_dir, axis=axis, chunk_size=1)
    reciprocal_cell = 2.0 * np.pi * np.eye(3)
    chemical_potentials = [0.2, 0.25]
    temperatures = [0.03, 0.04]

    result = compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact(
        system=system,
        directory=artifact_dir,
        reciprocal_cell=reciprocal_cell,
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
    expected_mobility = np.zeros((len(chemical_potentials), len(temperatures), 3, 3))
    expected_carrier_density = np.zeros((len(chemical_potentials), len(temperatures)))
    for imu, chemical_potential in enumerate(chemical_potentials):
        for itemperature, temperature in enumerate(temperatures):
            linewidth = compute_linewidth_mesh(
                mesh_data,
                chemical_potential=chemical_potential,
                temperature=temperature,
                sigma=0.05,
            )
            expected = compute_serta_mobility_si(
                eigenvalues=eigenvalues,
                velocities=velocities,
                linewidth=linewidth.linewidth,
                reciprocal_cell=reciprocal_cell,
                chemical_potential=chemical_potential,
                temperature=temperature,
                kpoint_weights=mesh_data.kpoint_weights,
                spin_degeneracy=2,
                volume=5.0,
            )
            expected_conductivity[imu, itemperature] = expected.conductivity
            expected_mobility[imu, itemperature] = expected.mobility
            expected_carrier_density[imu, itemperature] = expected.carrier_density

    np.testing.assert_allclose(result.conductivity, expected_conductivity)
    np.testing.assert_allclose(result.mobility, expected_mobility)
    np.testing.assert_allclose(result.carrier_density, expected_carrier_density)
    np.testing.assert_allclose(result.chemical_potentials, chemical_potentials)
    np.testing.assert_allclose(result.temperatures, temperatures)
    assert result.metadata["summary_first"] is True
    assert result.metadata["artifact_axis"] == axis
    assert result.metadata["velocity_source"] == velocity_source
    assert result.metadata["linewidth_scan_convention"] == "per_scan_point_recomputed"
    assert (
        result.metadata["source"]
        == "deeptb.eph.compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact"
    )


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


def test_fractional_band_velocities_to_si_uses_reciprocal_cell_convention():
    velocities = np.array([[[2.0, 0.0, 0.0]]])
    reciprocal_cell = np.diag([2.0, 4.0, 8.0])

    converted = fractional_band_velocities_to_si(velocities, reciprocal_cell)

    expected = np.array([[[1.0, 0.0, 0.0]]]) * dptb_constants.ANGSTROM_TO_M / dptb_constants.HBAR_EV_S
    np.testing.assert_allclose(converted, expected)


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
    assert result.metadata["velocity_input_unit"] == "eV/fractional_reciprocal_coordinate"
    assert result.metadata["velocity_unit"] == "m/s"
    assert result.metadata["reciprocal_cell_unit"] == "Angstrom^-1"
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


def test_compute_serta_mobility_scan_si_matches_single_point_results():
    eigenvalues = np.array([[0.0]])
    velocities = np.array([[[1.0, 0.0, 0.0]]])
    linewidth = np.array([[0.01]])
    reciprocal_cell = np.eye(3)
    chemical_potentials = np.array([-0.05, 0.0])
    temperatures = np.array([0.05, 0.1])

    scan = compute_serta_mobility_scan_si(
        eigenvalues=eigenvalues,
        velocities=velocities,
        linewidth=linewidth,
        reciprocal_cell=reciprocal_cell,
        chemical_potentials=chemical_potentials,
        temperatures=temperatures,
        volume=10.0,
    )
    point = compute_serta_mobility_si(
        eigenvalues=eigenvalues,
        velocities=velocities,
        linewidth=linewidth,
        reciprocal_cell=reciprocal_cell,
        chemical_potential=chemical_potentials[1],
        temperature=temperatures[0],
        volume=10.0,
    )

    assert scan.conductivity.shape == (2, 2, 3, 3)
    assert scan.mobility.shape == (2, 2, 3, 3)
    assert scan.carrier_density.shape == (2, 2)
    np.testing.assert_allclose(scan.conductivity[1, 0], point.conductivity)
    np.testing.assert_allclose(scan.mobility[1, 0], point.mobility)
    np.testing.assert_allclose(scan.carrier_density[1, 0], point.carrier_density)
    np.testing.assert_allclose(scan.chemical_potentials, chemical_potentials)
    np.testing.assert_allclose(scan.temperatures, temperatures)
    assert scan.metadata["schema"] == "deeptb.epc_mobility_scan"
    assert scan.metadata["chemical_potential_count"] == 2
    assert scan.metadata["temperature_count"] == 2
    assert scan.metadata["velocity_input_unit"] == "eV/fractional_reciprocal_coordinate"
    assert scan.metadata["velocity_unit"] == "m/s"
    assert scan.metadata["reciprocal_cell_unit"] == "Angstrom^-1"
    assert scan.metadata["conductivity_unit"] == "S/m"
    assert scan.metadata["carrier_density_unit"] == "m^-3"
    assert scan.metadata["mobility_unit"] == "m^2/V/s"


def test_mobility_scan_data_npz_roundtrip(tmp_path):
    scan = MobilityScanData(
        conductivity=np.ones((2, 1, 3, 3)),
        mobility=np.ones((2, 1, 3, 3)) * 2.0,
        carrier_density=np.ones((2, 1)) * 3.0,
        chemical_potentials=np.array([0.0, 0.1]),
        temperatures=np.array([0.05]),
        metadata={"dimension": "3d"},
    )
    path = tmp_path / "mobility_scan.npz"
    scan.save_npz(path)
    loaded = MobilityScanData.load_npz(path)

    np.testing.assert_allclose(loaded.conductivity, scan.conductivity)
    np.testing.assert_allclose(loaded.mobility, scan.mobility)
    np.testing.assert_allclose(loaded.carrier_density, scan.carrier_density)
    np.testing.assert_allclose(loaded.chemical_potentials, scan.chemical_potentials)
    np.testing.assert_allclose(loaded.temperatures, scan.temperatures)
    assert loaded.metadata["schema"] == "deeptb.epc_mobility_scan"
    assert loaded.metadata["schema_version"] == MOBILITY_SCAN_NPZ_SCHEMA_VERSION
