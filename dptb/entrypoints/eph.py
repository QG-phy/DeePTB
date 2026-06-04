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
    Phonons,
    RelaxationTimeData,
    RelaxationTimeMeshData,
    RelaxationTimePathData,
    SubspaceCouplingData,
    TransportData,
    compute_linewidth,
    compute_linewidth_mesh,
    compute_linewidth_path,
    compute_relaxation_time,
    compute_relaxation_time_mesh,
    compute_relaxation_time_path,
    compute_serta_transport_from_epc,
    compute_subspace_coupling_data,
)
from dptb.postprocess.unified.eph.utils import normalize_kpoints

log = logging.getLogger(__name__)


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
    temperature: Optional[float] = None,
    sigma: Optional[float] = None,
    broadening: str = "gaussian",
    mode_resolved: bool = False,
    sum_modes: bool = False,
    frequency_floor: float = 1e-5,
    kpoint_weights: Optional[str] = None,
    spin_degeneracy: int = 1,
    volume: float = 1.0,
    velocity_delta: float = 1e-4,
    velocity_source: str = "finite_difference",
    k_mesh: Optional[Sequence[int]] = None,
    q_mesh: Optional[Sequence[int]] = None,
    chunk_size: Optional[int] = None,
    time_reversal: bool = False,
    bands: Optional[Sequence[int]] = None,
    displacement: float = 1e-3,
    use_scc: bool = False,
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
    SubspaceCouplingData,
]:
    """Run an electron-phonon workflow task."""
    if not isinstance(task, str):
        raise ValueError(
            "task must be one of 'coupling', 'path-coupling', 'mesh-coupling', 'linewidth', "
            "'path-linewidth', 'mesh-linewidth', 'relaxation-time', 'relaxation', "
            "'path-relaxation-time', 'path-relaxation', 'mesh-relaxation-time', "
            "'mesh-relaxation', 'transport', or 'subspace'."
        )
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
    if task == "subspace":
        return eph_subspace(
            epc_data=epc_data,
            output=output or "subspace_coupling.npz",
            final_groups=final_groups,
            initial_groups=initial_groups,
        )
    raise ValueError(
        "task must be one of 'coupling', 'path-coupling', 'mesh-coupling', 'linewidth', "
        "'path-linewidth', 'mesh-linewidth', 'relaxation-time', 'relaxation', "
        "'path-relaxation-time', 'path-relaxation', 'mesh-relaxation-time', "
        "'mesh-relaxation', 'transport', or 'subspace'."
    )


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
