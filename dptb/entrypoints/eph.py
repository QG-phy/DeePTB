import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from dptb.postprocess.unified import TBSystem
from dptb.postprocess.unified.eph import (
    EPCData,
    EPCMeshData,
    EPCMeshSpec,
    EPCPathData,
    LinewidthData,
    LinewidthMeshData,
    LinewidthPathData,
    MobilityData,
    MobilityScanData,
    Phonons,
    RelaxationTimeData,
    RelaxationTimeMeshData,
    RelaxationTimePathData,
    SubspaceCouplingData,
    TransportData,
    compute_band_velocities_finite_difference,
    compute_band_velocities_hamiltonian_derivative,
    compute_coupling_strength_summary,
    compute_eliashberg_spectral_function,
    compute_linewidth,
    compute_linewidth_mesh,
    compute_linewidth_path,
    compute_phonon_dos,
    compute_relaxation_time,
    compute_relaxation_time_mesh,
    compute_relaxation_time_path,
    compute_scattering_maps,
    compute_serta_mobility_si,
    compute_serta_mobility_scan_si,
    compute_serta_transport_from_epc,
    compute_subspace_coupling_data,
)
from dptb.postprocess.unified.eph.utils import normalize_kpoints

log = logging.getLogger(__name__)

EPH_PRIMARY_TASKS = (
    "coupling",
    "path-coupling",
    "mesh-coupling",
    "linewidth",
    "path-linewidth",
    "mesh-linewidth",
    "relaxation-time",
    "path-relaxation-time",
    "mesh-relaxation-time",
    "transport",
    "mobility",
    "subspace",
    "coupling-summary",
    "scattering-map",
    "phonon-dos",
    "eliashberg",
)

EPH_TASK_ALIASES = (
    "relaxation",
    "path-relaxation",
    "mesh-relaxation",
)

EPH_TASK_CHOICES = (
    "coupling",
    "path-coupling",
    "mesh-coupling",
    "linewidth",
    "path-linewidth",
    "mesh-linewidth",
    "relaxation-time",
    "relaxation",
    "path-relaxation-time",
    "path-relaxation",
    "mesh-relaxation-time",
    "mesh-relaxation",
    "transport",
    "mobility",
    "subspace",
    "coupling-summary",
    "scattering-map",
    "phonon-dos",
    "eliashberg",
)

EPH_TASK_ERROR_MESSAGE = "task must be one of " + ", ".join(f"'{task}'" for task in EPH_TASK_CHOICES) + "."


def eph(
    structure: Optional[str] = None,
    init_model: Optional[str] = None,
    phonons: Optional[str] = None,
    kpoints: Optional[str] = None,
    output: Optional[str] = None,
    task: str = "coupling",
    epc_data: Optional[str] = None,
    linewidth_data: Optional[str] = None,
    final_groups: Optional[Sequence[str]] = None,
    initial_groups: Optional[Sequence[str]] = None,
    chemical_potential: Optional[float] = None,
    chemical_potentials: Optional[Sequence[float]] = None,
    temperature: Optional[float] = None,
    temperatures: Optional[Sequence[float]] = None,
    sigma: Optional[float] = None,
    broadening: str = "gaussian",
    mode_resolved: bool = False,
    sum_modes: bool = False,
    frequency_floor: float = 1e-5,
    kpoint_weights: Optional[str] = None,
    spin_degeneracy: int = 1,
    volume: float = 1.0,
    area: Optional[float] = None,
    dimension: str = "3d",
    velocity_delta: float = 1e-4,
    velocity_source: str = "finite_difference",
    k_mesh: Optional[Sequence[int]] = None,
    q_mesh: Optional[Sequence[int]] = None,
    chunk_size: Optional[int] = None,
    q_chunk_size: Optional[int] = None,
    time_reversal: bool = False,
    bands: Optional[Sequence[int]] = None,
    displacement: float = 1e-3,
    use_scc: bool = False,
    summary_unweighted: bool = False,
    dos_grid: Optional[Sequence[float]] = None,
    dos_sigma: Optional[float] = None,
    system=None,
    derivative_provider=None,
    **kwargs,
) -> Union[
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
    MobilityData,
    MobilityScanData,
    SubspaceCouplingData,
    dict,
]:
    """Run an electron-phonon workflow task."""
    if not isinstance(task, str):
        raise ValueError(EPH_TASK_ERROR_MESSAGE)
    task = task.lower()
    if use_scc:
        _reject_scc_v1(task)
    if task == "coupling":
        return eph_coupling(
            structure=structure,
            init_model=init_model,
            phonons=phonons,
            kpoints=kpoints,
            output=output or "epc_data.npz",
            bands=bands,
            displacement=displacement,
            use_scc=use_scc,
            system=system,
            derivative_provider=derivative_provider,
        )
    if task == "path-coupling":
        return eph_path_coupling(
            structure=structure,
            init_model=init_model,
            phonons=phonons,
            kpoints=kpoints,
            output=output or "epc_path_data.npz",
            bands=bands,
            displacement=displacement,
            use_scc=use_scc,
            system=system,
            derivative_provider=derivative_provider,
        )
    if task == "mesh-coupling":
        return eph_mesh_coupling(
            structure=structure,
            init_model=init_model,
            phonons=phonons,
            kpoints=kpoints,
            k_mesh=k_mesh,
            q_mesh=q_mesh,
            chunk_size=chunk_size,
            q_chunk_size=q_chunk_size,
            time_reversal=time_reversal,
            output=output or "epc_mesh_data.npz",
            bands=bands,
            displacement=displacement,
            use_scc=use_scc,
            system=system,
            derivative_provider=derivative_provider,
        )
    if task == "linewidth":
        return eph_linewidth(
            epc_data=epc_data,
            output=output or "linewidth.npz",
            chemical_potential=chemical_potential,
            temperature=temperature,
            sigma=sigma,
            broadening=broadening,
            mode_resolved=mode_resolved,
            frequency_floor=frequency_floor,
        )
    if task == "path-linewidth":
        return eph_path_linewidth(
            epc_data=epc_data,
            output=output or "path_linewidth.npz",
            chemical_potential=chemical_potential,
            temperature=temperature,
            sigma=sigma,
            broadening=broadening,
            mode_resolved=mode_resolved,
            frequency_floor=frequency_floor,
        )
    if task == "mesh-linewidth":
        return eph_mesh_linewidth(
            epc_data=epc_data,
            output=output or "mesh_linewidth.npz",
            chemical_potential=chemical_potential,
            temperature=temperature,
            sigma=sigma,
            broadening=broadening,
            mode_resolved=mode_resolved,
            frequency_floor=frequency_floor,
        )
    if task in {"relaxation-time", "relaxation"}:
        return eph_relaxation_time(
            linewidth_data=linewidth_data,
            output=output or "relaxation_time.npz",
            sum_modes=sum_modes,
        )
    if task in {"path-relaxation-time", "path-relaxation"}:
        return eph_path_relaxation_time(
            linewidth_data=linewidth_data,
            output=output or "path_relaxation_time.npz",
            sum_modes=sum_modes,
        )
    if task in {"mesh-relaxation-time", "mesh-relaxation"}:
        return eph_mesh_relaxation_time(
            linewidth_data=linewidth_data,
            output=output or "mesh_relaxation_time.npz",
            sum_modes=sum_modes,
        )
    if task == "transport":
        return eph_transport(
            structure=structure,
            init_model=init_model,
            epc_data=epc_data,
            linewidth_data=linewidth_data,
            output=output or "transport.npz",
            chemical_potential=chemical_potential,
            temperature=temperature,
            kpoint_weights=kpoint_weights,
            spin_degeneracy=spin_degeneracy,
            volume=volume,
            velocity_delta=velocity_delta,
            velocity_source=velocity_source,
            use_scc=use_scc,
            system=system,
        )
    if task == "mobility":
        return eph_mobility(
            structure=structure,
            init_model=init_model,
            epc_data=epc_data,
            linewidth_data=linewidth_data,
            output=output or "mobility.npz",
            chemical_potential=chemical_potential,
            chemical_potentials=chemical_potentials,
            temperature=temperature,
            temperatures=temperatures,
            kpoint_weights=kpoint_weights,
            spin_degeneracy=spin_degeneracy,
            dimension=dimension,
            volume=volume,
            area=area,
            velocity_delta=velocity_delta,
            velocity_source=velocity_source,
            use_scc=use_scc,
            system=system,
        )
    if task == "subspace":
        return eph_subspace(
            epc_data=epc_data,
            output=output or "subspace_coupling.npz",
            final_groups=final_groups,
            initial_groups=initial_groups,
        )
    if task == "coupling-summary":
        return eph_coupling_summary(
            epc_data=epc_data,
            output=output or "coupling_summary.json",
            weighted=not summary_unweighted,
        )
    if task == "scattering-map":
        return eph_scattering_map(
            epc_data=epc_data,
            output=output or "scattering_map.json",
            weighted=not summary_unweighted,
        )
    if task == "phonon-dos":
        return eph_phonon_dos(
            phonons=phonons,
            output=output or "phonon_dos.json",
            frequency_grid=dos_grid,
            sigma=dos_sigma,
            broadening=broadening,
        )
    if task == "eliashberg":
        return eph_eliashberg(
            epc_data=epc_data,
            output=output or "eliashberg.json",
            frequency_grid=dos_grid,
            sigma=dos_sigma,
            broadening=broadening,
            weighted=not summary_unweighted,
        )
    raise ValueError(EPH_TASK_ERROR_MESSAGE)


def eph_coupling(
    structure: Optional[str],
    init_model: Optional[str],
    phonons: str,
    kpoints: str,
    output: str,
    bands: Optional[Sequence[int]] = None,
    displacement: float = 1e-3,
    use_scc: bool = False,
    system=None,
    derivative_provider=None,
) -> EPCData:
    """Calculate electron-phonon coupling from external phonon-mode data."""
    if use_scc:
        _reject_scc_v1("coupling")

    if phonons is None:
        raise ValueError("phonons is required for dptb eph --task coupling.")
    if kpoints is None:
        raise ValueError("kpoints is required for dptb eph --task coupling.")
    if system is None:
        if structure is None:
            raise ValueError("structure is required for dptb eph --task coupling.")
        if init_model is None:
            raise ValueError("init_model is required for dptb eph --task coupling.")
        system = TBSystem(data=structure, calculator=init_model)

    phonon_data = Phonons.load_npz(phonons)
    kpoint_array = _load_kpoints(kpoints)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = system.eph.compute_coupling(
        kpoints=kpoint_array,
        phonons=phonon_data,
        bands=bands,
        displacement=displacement,
        use_scc=use_scc,
        derivative_provider=derivative_provider,
        output_npz=output_path,
    )
    log.info("Electron-phonon coupling data written to %s", output_path)
    return result


def eph_path_coupling(
    structure: Optional[str],
    init_model: Optional[str],
    phonons: str,
    kpoints: str,
    output: str,
    bands: Optional[Sequence[int]] = None,
    displacement: float = 1e-3,
    use_scc: bool = False,
    system=None,
    derivative_provider=None,
) -> EPCPathData:
    """Calculate fixed-k plus q-path electron-phonon coupling data."""
    if use_scc:
        _reject_scc_v1("path-coupling")

    if phonons is None:
        raise ValueError("phonons is required for dptb eph --task path-coupling.")
    if kpoints is None:
        raise ValueError("kpoints is required for dptb eph --task path-coupling.")
    if system is None:
        if structure is None:
            raise ValueError("structure is required for dptb eph --task path-coupling.")
        if init_model is None:
            raise ValueError("init_model is required for dptb eph --task path-coupling.")
        system = TBSystem(data=structure, calculator=init_model)

    phonon_data = Phonons.load_npz(phonons)
    kpoint_array = _load_kpoints(kpoints)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = system.eph.compute_path(
        kpoints=kpoint_array,
        phonons=phonon_data,
        bands=bands,
        displacement=displacement,
        use_scc=use_scc,
        derivative_provider=derivative_provider,
        output_npz=output_path,
    )
    log.info("Electron-phonon path coupling data written to %s", output_path)
    return result


def eph_mesh_coupling(
    structure: Optional[str],
    init_model: Optional[str],
    phonons: str,
    kpoints: Optional[str],
    k_mesh: Optional[Sequence[int]],
    q_mesh: Optional[Sequence[int]],
    chunk_size: Optional[int],
    q_chunk_size: Optional[int],
    time_reversal: bool,
    output: str,
    bands: Optional[Sequence[int]] = None,
    displacement: float = 1e-3,
    use_scc: bool = False,
    system=None,
    derivative_provider=None,
) -> EPCMeshData:
    """Calculate serial full-mesh electron-phonon coupling data."""
    if use_scc:
        _reject_scc_v1("mesh-coupling")

    if phonons is None:
        raise ValueError("phonons is required for dptb eph --task mesh-coupling.")
    if kpoints is None and k_mesh is None:
        raise ValueError("kpoints or k_mesh is required for dptb eph --task mesh-coupling.")
    if system is None:
        if structure is None:
            raise ValueError("structure is required for dptb eph --task mesh-coupling.")
        if init_model is None:
            raise ValueError("init_model is required for dptb eph --task mesh-coupling.")
        system = TBSystem(data=structure, calculator=init_model)

    phonon_data = Phonons.load_npz(phonons)
    explicit_kpoints = _load_kpoints(kpoints) if kpoints is not None else None
    mesh_spec = EPCMeshSpec(
        kpoints=explicit_kpoints,
        k_mesh=k_mesh,
        q_mesh=q_mesh,
        chunk_size=chunk_size,
        q_chunk_size=q_chunk_size,
        time_reversal=time_reversal,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = system.eph.compute_mesh(
        mesh_spec=mesh_spec,
        phonons=phonon_data,
        bands=bands,
        displacement=displacement,
        use_scc=use_scc,
        derivative_provider=derivative_provider,
        output_npz=output_path,
    )
    log.info("Electron-phonon mesh coupling data written to %s", output_path)
    return result


def eph_linewidth(
    epc_data: str,
    output: str,
    chemical_potential: float,
    temperature: float,
    sigma: float,
    broadening: str = "gaussian",
    mode_resolved: bool = False,
    frequency_floor: float = 1e-5,
) -> LinewidthData:
    """Calculate linewidth postprocess data from an EPCData NPZ file."""
    if epc_data is None:
        raise ValueError("epc_data is required for dptb eph --task linewidth.")
    if chemical_potential is None:
        raise ValueError("chemical_potential is required for dptb eph --task linewidth.")
    if temperature is None:
        raise ValueError("temperature is required for dptb eph --task linewidth.")
    if sigma is None:
        raise ValueError("sigma is required for dptb eph --task linewidth.")

    data = EPCData.load_npz(epc_data)
    result = compute_linewidth(
        data,
        chemical_potential=chemical_potential,
        temperature=temperature,
        sigma=sigma,
        broadening=broadening,
        mode_resolved=mode_resolved,
        frequency_floor=frequency_floor,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_npz(output_path)
    log.info("Electron-phonon linewidth data written to %s", output_path)
    return result


def eph_path_linewidth(
    epc_data: str,
    output: str,
    chemical_potential: float,
    temperature: float,
    sigma: float,
    broadening: str = "gaussian",
    mode_resolved: bool = False,
    frequency_floor: float = 1e-5,
) -> LinewidthPathData:
    """Calculate path-resolved linewidth data from an EPCPathData NPZ file."""
    if epc_data is None:
        raise ValueError("epc_data is required for dptb eph --task path-linewidth.")
    if chemical_potential is None:
        raise ValueError("chemical_potential is required for dptb eph --task path-linewidth.")
    if temperature is None:
        raise ValueError("temperature is required for dptb eph --task path-linewidth.")
    if sigma is None:
        raise ValueError("sigma is required for dptb eph --task path-linewidth.")

    result = compute_linewidth_path(
        EPCPathData.load_npz(epc_data),
        chemical_potential=chemical_potential,
        temperature=temperature,
        sigma=sigma,
        broadening=broadening,
        mode_resolved=mode_resolved,
        frequency_floor=frequency_floor,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_npz(output_path)
    log.info("Electron-phonon path linewidth data written to %s", output_path)
    return result


def eph_mesh_linewidth(
    epc_data: str,
    output: str,
    chemical_potential: float,
    temperature: float,
    sigma: float,
    broadening: str = "gaussian",
    mode_resolved: bool = False,
    frequency_floor: float = 1e-5,
) -> LinewidthMeshData:
    """Calculate mesh linewidth data from an EPCMeshData NPZ file."""
    if epc_data is None:
        raise ValueError("epc_data is required for dptb eph --task mesh-linewidth.")
    if chemical_potential is None:
        raise ValueError("chemical_potential is required for dptb eph --task mesh-linewidth.")
    if temperature is None:
        raise ValueError("temperature is required for dptb eph --task mesh-linewidth.")
    if sigma is None:
        raise ValueError("sigma is required for dptb eph --task mesh-linewidth.")

    result = compute_linewidth_mesh(
        EPCMeshData.load_npz(epc_data),
        chemical_potential=chemical_potential,
        temperature=temperature,
        sigma=sigma,
        broadening=broadening,
        mode_resolved=mode_resolved,
        frequency_floor=frequency_floor,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_npz(output_path)
    log.info("Electron-phonon mesh linewidth data written to %s", output_path)
    return result


def eph_relaxation_time(
    linewidth_data: str,
    output: str,
    sum_modes: bool = False,
) -> RelaxationTimeData:
    """Calculate relaxation-time postprocess data from a LinewidthData NPZ file."""
    if linewidth_data is None:
        raise ValueError("linewidth_data is required for dptb eph --task relaxation-time.")

    result = compute_relaxation_time(LinewidthData.load_npz(linewidth_data), sum_modes=sum_modes)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_npz(output_path)
    log.info("Electron-phonon relaxation-time data written to %s", output_path)
    return result


def eph_path_relaxation_time(
    linewidth_data: str,
    output: str,
    sum_modes: bool = False,
) -> RelaxationTimePathData:
    """Calculate path-resolved relaxation-time data from a LinewidthPathData NPZ file."""
    if linewidth_data is None:
        raise ValueError("linewidth_data is required for dptb eph --task path-relaxation-time.")

    result = compute_relaxation_time_path(LinewidthPathData.load_npz(linewidth_data), sum_modes=sum_modes)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_npz(output_path)
    log.info("Electron-phonon path relaxation-time data written to %s", output_path)
    return result


def eph_mesh_relaxation_time(
    linewidth_data: str,
    output: str,
    sum_modes: bool = False,
) -> RelaxationTimeMeshData:
    """Calculate mesh relaxation-time data from a LinewidthMeshData NPZ file."""
    if linewidth_data is None:
        raise ValueError("linewidth_data is required for dptb eph --task mesh-relaxation-time.")

    result = compute_relaxation_time_mesh(LinewidthMeshData.load_npz(linewidth_data), sum_modes=sum_modes)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_npz(output_path)
    log.info("Electron-phonon mesh relaxation-time data written to %s", output_path)
    return result


def eph_transport(
    structure: Optional[str],
    init_model: Optional[str],
    epc_data: str,
    linewidth_data: str,
    output: str,
    chemical_potential: float,
    temperature: float,
    kpoint_weights: Optional[str] = None,
    spin_degeneracy: int = 1,
    volume: float = 1.0,
    velocity_delta: float = 1e-4,
    velocity_source: str = "finite_difference",
    use_scc: bool = False,
    system=None,
) -> TransportData:
    """Calculate SERTA transport data from EPC and linewidth NPZ files."""
    if use_scc:
        _reject_scc_v1("transport")
    if epc_data is None:
        raise ValueError("epc_data is required for dptb eph --task transport.")
    if linewidth_data is None:
        raise ValueError("linewidth_data is required for dptb eph --task transport.")
    if chemical_potential is None:
        raise ValueError("chemical_potential is required for dptb eph --task transport.")
    if temperature is None:
        raise ValueError("temperature is required for dptb eph --task transport.")
    if system is None:
        if structure is None:
            raise ValueError("structure is required for dptb eph --task transport.")
        if init_model is None:
            raise ValueError("init_model is required for dptb eph --task transport.")
        system = TBSystem(data=structure, calculator=init_model)

    weights = _load_kpoint_weights(kpoint_weights) if kpoint_weights is not None else None
    result = compute_serta_transport_from_epc(
        system=system,
        epc_data=EPCData.load_npz(epc_data),
        linewidth_data=LinewidthData.load_npz(linewidth_data),
        chemical_potential=chemical_potential,
        temperature=temperature,
        kpoint_weights=weights,
        spin_degeneracy=spin_degeneracy,
        volume=volume,
        velocity_delta=velocity_delta,
        velocity_source=velocity_source,
        use_scc=use_scc,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_npz(output_path)
    log.info("Electron-phonon transport data written to %s", output_path)
    return result


def eph_mobility(
    structure: Optional[str],
    init_model: Optional[str],
    epc_data: str,
    linewidth_data: str,
    output: str,
    chemical_potential: Optional[float],
    temperature: Optional[float],
    chemical_potentials: Optional[Sequence[float]] = None,
    temperatures: Optional[Sequence[float]] = None,
    kpoint_weights: Optional[str] = None,
    spin_degeneracy: int = 1,
    dimension: str = "3d",
    volume: Optional[float] = 1.0,
    area: Optional[float] = None,
    velocity_delta: float = 1e-4,
    velocity_source: str = "finite_difference",
    use_scc: bool = False,
    system=None,
) -> Union[MobilityData, MobilityScanData]:
    """Calculate SI SERTA mobility data from EPC and linewidth NPZ files."""
    if use_scc:
        _reject_scc_v1("mobility")
    if epc_data is None:
        raise ValueError("epc_data is required for dptb eph --task mobility.")
    if linewidth_data is None:
        raise ValueError("linewidth_data is required for dptb eph --task mobility.")
    chemical_potential_values, use_scan_mu = _resolve_scan_axis(
        single_value=chemical_potential,
        multiple_values=chemical_potentials,
        single_name="chemical_potential",
        multiple_name="chemical_potentials",
        task="mobility",
    )
    temperature_values, use_scan_temperature = _resolve_scan_axis(
        single_value=temperature,
        multiple_values=temperatures,
        single_name="temperature",
        multiple_name="temperatures",
        task="mobility",
        positive=True,
    )
    use_scan = use_scan_mu or use_scan_temperature
    if system is None:
        if structure is None:
            raise ValueError("structure is required for dptb eph --task mobility.")
        if init_model is None:
            raise ValueError("init_model is required for dptb eph --task mobility.")
        system = TBSystem(data=structure, calculator=init_model)

    epc = EPCData.load_npz(epc_data)
    linewidth_result = LinewidthData.load_npz(linewidth_data)
    linewidth_input = linewidth_result.linewidth
    if linewidth_input.ndim == 3:
        linewidth_input = linewidth_input.sum(axis=-1)
    if linewidth_input.shape != epc.eigenvalues_k.shape:
        raise ValueError("linewidth_data.linewidth must match EPCData eigenvalues_k shape after mode summation.")

    weights = _load_kpoint_weights(kpoint_weights) if kpoint_weights is not None else None
    source = _normalize_velocity_source(velocity_source)
    if source == "finite_difference":
        velocities = compute_band_velocities_finite_difference(
            system=system,
            kpoints=epc.kpoints,
            bands=epc.band_indices,
            delta=velocity_delta,
            use_scc=use_scc,
        )
    else:
        velocities = compute_band_velocities_hamiltonian_derivative(
            system=system,
            kpoints=epc.kpoints,
            bands=epc.band_indices,
            use_scc=use_scc,
        )

    reciprocal_cell = _reciprocal_cell_from_system(system)
    if use_scan:
        result = compute_serta_mobility_scan_si(
            eigenvalues=epc.eigenvalues_k,
            velocities=velocities,
            linewidth=linewidth_input,
            reciprocal_cell=reciprocal_cell,
            chemical_potentials=chemical_potential_values,
            temperatures=temperature_values,
            kpoint_weights=weights,
            spin_degeneracy=spin_degeneracy,
            dimension=dimension,
            volume=volume if str(dimension).lower() == "3d" else None,
            area=area,
        )
    else:
        result = compute_serta_mobility_si(
            eigenvalues=epc.eigenvalues_k,
            velocities=velocities,
            linewidth=linewidth_input,
            reciprocal_cell=reciprocal_cell,
            chemical_potential=float(chemical_potential_values[0]),
            temperature=float(temperature_values[0]),
            kpoint_weights=weights,
            spin_degeneracy=spin_degeneracy,
            dimension=dimension,
            volume=volume if str(dimension).lower() == "3d" else None,
            area=area,
        )
    result.metadata.update(
        {
            "velocity_source": source,
            "velocity_unit": "m/s",
            "velocity_input_unit": "eV/fractional_reciprocal_coordinate",
            "reciprocal_cell_source": "2pi_times_ase_cell_reciprocal",
            "band_indices": epc.band_indices,
            "epc_schema": epc.metadata.get("schema"),
            "linewidth_schema": linewidth_result.metadata.get("schema"),
        }
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_npz(output_path)
    log.info("Electron-phonon mobility data written to %s", output_path)
    return result


def eph_subspace(
    epc_data: str,
    output: str,
    final_groups: Sequence[str],
    initial_groups: Optional[Sequence[str]] = None,
) -> SubspaceCouplingData:
    """Calculate gauge-invariant subspace coupling strength from EPCData NPZ."""
    if epc_data is None:
        raise ValueError("epc_data is required for dptb eph --task subspace.")
    if final_groups is None:
        raise ValueError("final_groups is required for dptb eph --task subspace.")

    result = compute_subspace_coupling_data(
        EPCData.load_npz(epc_data),
        final_groups=_parse_band_groups(final_groups),
        initial_groups=_parse_band_groups(initial_groups) if initial_groups is not None else None,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_npz(output_path)
    log.info("Electron-phonon subspace coupling data written to %s", output_path)
    return result


def eph_coupling_summary(
    epc_data: str,
    output: str,
    weighted: bool = True,
) -> dict:
    """Write q/k/mode/band-resolved coupling-strength summaries as JSON."""
    if epc_data is None:
        raise ValueError("epc_data is required for dptb eph --task coupling-summary.")

    result = compute_coupling_strength_summary(_load_epc_summary_data(epc_data), weighted=weighted)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    log.info("Electron-phonon coupling summary written to %s", output_path)
    return result


def eph_scattering_map(
    epc_data: str,
    output: str,
    weighted: bool = True,
) -> dict:
    """Write q/k/band-resolved EPC scattering proxy maps as JSON."""
    if epc_data is None:
        raise ValueError("epc_data is required for dptb eph --task scattering-map.")

    result = compute_scattering_maps(_load_epc_summary_data(epc_data), weighted=weighted)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    log.info("Electron-phonon scattering proxy map written to %s", output_path)
    return result


def eph_phonon_dos(
    phonons: str,
    output: str,
    frequency_grid: Optional[Sequence[float]],
    sigma: Optional[float],
    broadening: str = "gaussian",
) -> dict:
    """Write phonon DOS from external phonon-mode data as JSON."""
    if phonons is None:
        raise ValueError("phonons is required for dptb eph --task phonon-dos.")
    if frequency_grid is None:
        raise ValueError("dos_grid is required for dptb eph --task phonon-dos.")
    if sigma is None:
        raise ValueError("dos_sigma is required for dptb eph --task phonon-dos.")

    result = compute_phonon_dos(
        Phonons.load_npz(phonons),
        frequency_grid=np.asarray(frequency_grid, dtype=float),
        sigma=sigma,
        broadening=broadening,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    log.info("Electron-phonon phonon DOS written to %s", output_path)
    return result


def eph_eliashberg(
    epc_data: str,
    output: str,
    frequency_grid: Optional[Sequence[float]],
    sigma: Optional[float],
    broadening: str = "gaussian",
    weighted: bool = True,
) -> dict:
    """Write an Eliashberg-like coupling-strength spectrum as JSON."""
    if epc_data is None:
        raise ValueError("epc_data is required for dptb eph --task eliashberg.")
    if frequency_grid is None:
        raise ValueError("dos_grid is required for dptb eph --task eliashberg.")
    if sigma is None:
        raise ValueError("dos_sigma is required for dptb eph --task eliashberg.")

    result = compute_eliashberg_spectral_function(
        _load_epc_summary_data(epc_data),
        frequency_grid=np.asarray(frequency_grid, dtype=float),
        sigma=sigma,
        broadening=broadening,
        weighted=weighted,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    log.info("Electron-phonon Eliashberg-like spectrum written to %s", output_path)
    return result


def _load_epc_summary_data(path: str) -> Union[EPCData, EPCMeshData, EPCPathData]:
    with np.load(path, allow_pickle=False) as data:
        metadata = _metadata_json_from_npz(data)
    schema = metadata.get("schema")
    if schema == "deeptb.epc_mesh_data":
        return EPCMeshData.load_npz(path)
    if schema == "deeptb.epc_path_data":
        return EPCPathData.load_npz(path)
    if schema == "deeptb.epc_data":
        return EPCData.load_npz(path)
    raise ValueError("epc_data must contain EPCData, EPCMeshData, or EPCPathData schema metadata.")


def _metadata_json_from_npz(data) -> dict:
    if "metadata_json" not in data:
        raise ValueError("metadata_json is required for DeePTB EPC summary inputs.")
    metadata_raw = data["metadata_json"]
    if np.shape(metadata_raw) != ():
        raise ValueError("metadata_json must be a scalar JSON object.")
    try:
        metadata = json.loads(str(metadata_raw.item()))
    except json.JSONDecodeError:
        raise ValueError("metadata_json must be valid JSON.") from None
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to a JSON object.")
    return metadata


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_kpoints(path: str) -> np.ndarray:
    return normalize_kpoints(_load_array(path, npz_key="kpoints"))


def _load_kpoint_weights(path: str) -> np.ndarray:
    weights = np.asarray(_load_array(path, npz_key="kpoint_weights"), dtype=float)
    if weights.ndim != 1:
        raise ValueError("kpoint_weights must be a one-dimensional array.")
    if weights.size == 0:
        raise ValueError("kpoint_weights must be non-empty.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("kpoint_weights must contain finite values.")
    if np.any(weights < 0.0):
        raise ValueError("kpoint_weights must contain non-negative values.")
    if weights.sum() <= 0.0:
        raise ValueError("kpoint_weights must have a positive sum.")
    return weights


def _normalize_velocity_source(velocity_source: str) -> str:
    if not isinstance(velocity_source, str):
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")
    source = velocity_source.replace("-", "_").lower()
    if source not in {"finite_difference", "hamiltonian_derivative"}:
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")
    return source


def _resolve_scan_axis(
    single_value: Optional[float],
    multiple_values: Optional[Sequence[float]],
    single_name: str,
    multiple_name: str,
    task: str,
    positive: bool = False,
) -> tuple[np.ndarray, bool]:
    if single_value is not None and multiple_values is not None:
        raise ValueError(f"{single_name} and {multiple_name} cannot both be set for dptb eph --task {task}.")
    if multiple_values is not None:
        values = np.asarray(multiple_values, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{multiple_name} must be a one-dimensional non-empty array.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{multiple_name} must contain finite values.")
        if positive and np.any(values <= 0.0):
            raise ValueError(f"{multiple_name} must contain finite positive values.")
        return values, True
    if single_value is None:
        raise ValueError(f"{single_name} is required for dptb eph --task {task}.")
    value = np.asarray(single_value, dtype=float)
    if value.shape != () or not np.isfinite(float(value)):
        raise ValueError(f"{single_name} must be finite.")
    if positive and float(value) <= 0.0:
        raise ValueError(f"{single_name} must be finite and positive.")
    return np.asarray([float(value)], dtype=float), False


def _reciprocal_cell_from_system(system) -> np.ndarray:
    atoms = getattr(system, "atoms", None)
    if atoms is None:
        raise ValueError("system must provide an ASE atoms object for mobility reciprocal cell conversion.")
    reciprocal = 2.0 * np.pi * np.asarray(atoms.cell.reciprocal(), dtype=float)
    if reciprocal.shape != (3, 3) or not np.all(np.isfinite(reciprocal)):
        raise ValueError("system atoms cell must provide a finite 3x3 reciprocal cell.")
    if abs(np.linalg.det(reciprocal)) <= 0.0:
        raise ValueError("system atoms cell must be periodic with an invertible cell for mobility.")
    return reciprocal


def _load_array(path: str, npz_key: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if npz_key not in data:
                raise KeyError(f"NPZ input must contain a {npz_key!r} array.")
            return data[npz_key]
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            value = value.get(npz_key)
        if value is None:
            raise KeyError(f"JSON input must be an array or contain a {npz_key!r} field.")
        return np.asarray(value, dtype=float)
    return np.loadtxt(path, dtype=float)


def _parse_band_groups(values: Sequence[str]) -> Sequence[np.ndarray]:
    if len(values) == 0:
        raise ValueError("band groups must contain at least one start:stop range.")
    groups = []
    seen = set()
    for value in values:
        if ":" not in value:
            raise ValueError("band groups must use 'start:stop' ranges.")
        start_text, stop_text = value.split(":", 1)
        try:
            start = int(start_text)
            stop = int(stop_text)
        except ValueError:
            raise ValueError("band group start and stop must be integers.") from None
        if start < 0 or stop < 0:
            raise ValueError("band group ranges must be non-negative.")
        if stop <= start:
            raise ValueError("band group stop must be larger than start.")
        group = np.arange(start, stop, dtype=int)
        overlap = seen.intersection(group.tolist())
        if overlap:
            raise ValueError("band group ranges must not overlap.")
        seen.update(group.tolist())
        groups.append(group)
    return groups


def _reject_scc_v1(task: str) -> None:
    raise NotImplementedError(f"SCC-corrected electron-phonon {task} is not supported in v1.")
