from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from typing import Union

import numpy as np

from dptb.postprocess.unified.eph.data import (
    EPCData,
    EPCMeshData,
    EPCPathData,
    Phonons,
    _merge_metadata,
    _metadata_from_npz,
    _metadata_to_json,
    _required_npz_array,
)
from dptb.postprocess.unified.eph.executor import read_epc_mesh_chunked_manifest, read_epc_mesh_chunked_weights
from dptb.utils.constants import ANGSTROM_TO_M, ELECTRON_CHARGE_C, HBAR_EV_S, THZ_TO_EV, eV2J
from dptb.postprocess.unified.eph.utils import (
    as_array,
    validate_finite_nonnegative_scalar,
    validate_finite_scalar,
    normalize_integer_array,
    normalize_integer_indices,
    normalize_kpoints,
    validate_finite_positive_scalar,
)


LINEWIDTH_NPZ_SCHEMA_VERSION = 1
LINEWIDTH_MESH_NPZ_SCHEMA_VERSION = 1
LINEWIDTH_PATH_NPZ_SCHEMA_VERSION = 1
RELAXATION_TIME_NPZ_SCHEMA_VERSION = 1
RELAXATION_TIME_MESH_NPZ_SCHEMA_VERSION = 1
RELAXATION_TIME_PATH_NPZ_SCHEMA_VERSION = 1
TRANSPORT_NPZ_SCHEMA_VERSION = 1
TRANSPORT_SCAN_NPZ_SCHEMA_VERSION = 1
MOBILITY_NPZ_SCHEMA_VERSION = 1
MOBILITY_SCAN_NPZ_SCHEMA_VERSION = 1
SUBSPACE_COUPLING_NPZ_SCHEMA_VERSION = 1


def _metadata_unit(metadata: Dict[str, Any], key: str, default: str) -> str:
    value = metadata.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"metadata[{key!r}] must be a non-empty unit string.")
    return value


@dataclass
class LinewidthData:
    """Electron linewidths from electron-phonon coupling data."""

    linewidth: np.ndarray
    absorption: np.ndarray
    emission: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.linewidth = np.asarray(self.linewidth, dtype=float)
        self.absorption = np.asarray(self.absorption, dtype=float)
        self.emission = np.asarray(self.emission, dtype=float)
        if self.absorption.shape != self.linewidth.shape:
            raise ValueError("absorption must have the same shape as linewidth.")
        if self.emission.shape != self.linewidth.shape:
            raise ValueError("emission must have the same shape as linewidth.")
        if self.linewidth.size == 0:
            raise ValueError("linewidth must be non-empty.")
        if self.linewidth.ndim not in {2, 3}:
            raise ValueError("linewidth must have shape (nk, nbands) or (nk, nbands, nmodes).")
        if not np.all(np.isfinite(self.linewidth)):
            raise ValueError("linewidth must contain finite values.")
        if not np.all(np.isfinite(self.absorption)) or np.any(self.absorption < 0.0):
            raise ValueError("absorption must contain finite non-negative values.")
        if not np.all(np.isfinite(self.emission)) or np.any(self.emission < 0.0):
            raise ValueError("emission must contain finite non-negative values.")
        if np.any(self.linewidth < 0.0):
            raise ValueError("linewidth must contain non-negative values.")
        if not np.allclose(self.linewidth, self.absorption + self.emission):
            raise ValueError("linewidth must equal absorption + emission.")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_linewidth",
                "schema_version": LINEWIDTH_NPZ_SCHEMA_VERSION,
                "linewidth_unit": "eV",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save electron-phonon linewidth postprocess data to NPZ."""
        np.savez_compressed(
            path,
            elph_linewidth=self.linewidth,
            elph_linewidth_absorption=self.absorption,
            elph_linewidth_emission=self.emission,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "LinewidthData":
        """Load electron-phonon linewidth postprocess data from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                linewidth=_required_npz_array(data, "elph_linewidth"),
                absorption=_required_npz_array(data, "elph_linewidth_absorption"),
                emission=_required_npz_array(data, "elph_linewidth_emission"),
                metadata=metadata,
            )


@dataclass
class LinewidthMeshData:
    """Mesh linewidths from DeePTB-native EPC mesh data."""

    linewidth: np.ndarray
    absorption: np.ndarray
    emission: np.ndarray
    kpoints: np.ndarray
    kpoint_weights: np.ndarray
    band_indices: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.linewidth = np.asarray(self.linewidth, dtype=float)
        self.absorption = np.asarray(self.absorption, dtype=float)
        self.emission = np.asarray(self.emission, dtype=float)
        self.kpoints = normalize_kpoints(self.kpoints)
        self.kpoint_weights = _normalize_weights(self.kpoint_weights, "kpoint_weights")
        self.band_indices = normalize_integer_indices(self.band_indices, "band_indices")
        if self.absorption.shape != self.linewidth.shape:
            raise ValueError("absorption must have the same shape as linewidth.")
        if self.emission.shape != self.linewidth.shape:
            raise ValueError("emission must have the same shape as linewidth.")
        if self.linewidth.size == 0:
            raise ValueError("linewidth must be non-empty.")
        if self.linewidth.ndim not in {2, 3}:
            raise ValueError("linewidth must have shape (nk, nbands) or (nk, nbands, nmodes).")
        if self.linewidth.shape[:2] != (self.kpoints.shape[0], self.band_indices.shape[0]):
            raise ValueError("linewidth leading axes must match kpoints and band_indices.")
        if self.kpoint_weights.shape != (self.kpoints.shape[0],):
            raise ValueError("kpoint_weights must match kpoints.")
        if not np.all(np.isfinite(self.linewidth)):
            raise ValueError("linewidth must contain finite values.")
        if not np.all(np.isfinite(self.absorption)) or np.any(self.absorption < 0.0):
            raise ValueError("absorption must contain finite non-negative values.")
        if not np.all(np.isfinite(self.emission)) or np.any(self.emission < 0.0):
            raise ValueError("emission must contain finite non-negative values.")
        if np.any(self.linewidth < 0.0):
            raise ValueError("linewidth must contain non-negative values.")
        if not np.allclose(self.linewidth, self.absorption + self.emission):
            raise ValueError("linewidth must equal absorption + emission.")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_mesh_linewidth",
                "schema_version": LINEWIDTH_MESH_NPZ_SCHEMA_VERSION,
                "linewidth_unit": "eV",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        np.savez_compressed(
            path,
            elph_mesh_linewidth=self.linewidth,
            elph_mesh_linewidth_absorption=self.absorption,
            elph_mesh_linewidth_emission=self.emission,
            el_kpoints=self.kpoints,
            el_kpoint_weights=self.kpoint_weights,
            el_band_indices=self.band_indices,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "LinewidthMeshData":
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                linewidth=_required_npz_array(data, "elph_mesh_linewidth"),
                absorption=_required_npz_array(data, "elph_mesh_linewidth_absorption"),
                emission=_required_npz_array(data, "elph_mesh_linewidth_emission"),
                kpoints=_required_npz_array(data, "el_kpoints"),
                kpoint_weights=_required_npz_array(data, "el_kpoint_weights"),
                band_indices=_required_npz_array(data, "el_band_indices"),
                metadata=metadata,
            )


@dataclass
class LinewidthPathData:
    """Path-resolved linewidth contributions from EPC path data."""

    linewidth: np.ndarray
    absorption: np.ndarray
    emission: np.ndarray
    path_axis: str
    path_coordinates: np.ndarray
    band_indices: np.ndarray
    path_segments: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.linewidth = np.asarray(self.linewidth, dtype=float)
        self.absorption = np.asarray(self.absorption, dtype=float)
        self.emission = np.asarray(self.emission, dtype=float)
        self.path_axis = _normalize_path_axis(self.path_axis)
        self.path_coordinates = np.asarray(self.path_coordinates, dtype=float)
        self.band_indices = normalize_integer_indices(self.band_indices, "band_indices")
        if self.absorption.shape != self.linewidth.shape:
            raise ValueError("absorption must have the same shape as linewidth.")
        if self.emission.shape != self.linewidth.shape:
            raise ValueError("emission must have the same shape as linewidth.")
        if self.linewidth.size == 0:
            raise ValueError("linewidth must be non-empty.")
        if self.linewidth.ndim not in {3, 4}:
            raise ValueError("linewidth must have shape (npath, nk, nbands) or (npath, nk, nbands, nmodes).")
        if self.path_coordinates.ndim != 1 or self.path_coordinates.shape[0] != self.linewidth.shape[0]:
            raise ValueError("path_coordinates length must match the path-resolved linewidth axis.")
        if self.linewidth.shape[2] != self.band_indices.shape[0]:
            raise ValueError("band_indices length must match the linewidth band axis.")
        if not np.all(np.isfinite(self.path_coordinates)):
            raise ValueError("path_coordinates must be finite.")
        if not np.all(np.isfinite(self.linewidth)):
            raise ValueError("linewidth must contain finite values.")
        if not np.all(np.isfinite(self.absorption)) or np.any(self.absorption < 0.0):
            raise ValueError("absorption must contain finite non-negative values.")
        if not np.all(np.isfinite(self.emission)) or np.any(self.emission < 0.0):
            raise ValueError("emission must contain finite non-negative values.")
        if np.any(self.linewidth < 0.0):
            raise ValueError("linewidth must contain non-negative values.")
        if not np.allclose(self.linewidth, self.absorption + self.emission):
            raise ValueError("linewidth must equal absorption + emission.")
        if self.path_segments is not None:
            self.path_segments = _normalize_path_segments(self.path_segments, self.linewidth.shape[0])
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_path_linewidth",
                "schema_version": LINEWIDTH_PATH_NPZ_SCHEMA_VERSION,
                "linewidth_unit": "eV",
                "path_coordinate_unit": "fractional_reciprocal_coordinate_distance",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        payload = {
            "elph_path_linewidth": self.linewidth,
            "elph_path_linewidth_absorption": self.absorption,
            "elph_path_linewidth_emission": self.emission,
            "path_axis": np.array(self.path_axis),
            "path_coordinates": self.path_coordinates,
            "el_band_indices": self.band_indices,
            "metadata_json": np.array(_metadata_to_json(self.metadata)),
        }
        if self.path_segments is not None:
            payload["path_segments"] = self.path_segments
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "LinewidthPathData":
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            path_axis = _scalar_string_from_npz(data, "path_axis")
            return cls(
                linewidth=_required_npz_array(data, "elph_path_linewidth"),
                absorption=_required_npz_array(data, "elph_path_linewidth_absorption"),
                emission=_required_npz_array(data, "elph_path_linewidth_emission"),
                path_axis=path_axis,
                path_coordinates=_required_npz_array(data, "path_coordinates"),
                band_indices=_required_npz_array(data, "el_band_indices"),
                path_segments=data["path_segments"] if "path_segments" in data else None,
                metadata=metadata,
            )


@dataclass
class RelaxationTimeData:
    """Electron relaxation times derived from electron-phonon linewidths."""

    relaxation_time: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.relaxation_time = np.asarray(self.relaxation_time, dtype=float)
        if self.relaxation_time.size == 0:
            raise ValueError("relaxation_time must be non-empty.")
        if self.relaxation_time.ndim not in {2, 3}:
            raise ValueError("relaxation_time must have shape (nk, nbands) or (nk, nbands, nmodes).")
        if not np.all(np.isfinite(self.relaxation_time)) or np.any(self.relaxation_time <= 0.0):
            raise ValueError("relaxation_time must contain finite positive values.")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_relaxation_time",
                "schema_version": RELAXATION_TIME_NPZ_SCHEMA_VERSION,
                "relaxation_time_unit": "s",
                "convention": "hbar_over_2linewidth",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save electron-phonon relaxation-time postprocess data to NPZ."""
        np.savez_compressed(
            path,
            elph_relaxation_time=self.relaxation_time,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "RelaxationTimeData":
        """Load electron-phonon relaxation-time postprocess data from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                relaxation_time=_required_npz_array(data, "elph_relaxation_time"),
                metadata=metadata,
            )


@dataclass
class RelaxationTimeMeshData:
    """Mesh relaxation times derived from mesh linewidth data."""

    relaxation_time: np.ndarray
    kpoints: np.ndarray
    kpoint_weights: np.ndarray
    band_indices: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.relaxation_time = np.asarray(self.relaxation_time, dtype=float)
        self.kpoints = normalize_kpoints(self.kpoints)
        self.kpoint_weights = _normalize_weights(self.kpoint_weights, "kpoint_weights")
        self.band_indices = normalize_integer_indices(self.band_indices, "band_indices")
        if self.relaxation_time.size == 0:
            raise ValueError("relaxation_time must be non-empty.")
        if self.relaxation_time.ndim not in {2, 3}:
            raise ValueError("relaxation_time must have shape (nk, nbands) or (nk, nbands, nmodes).")
        if self.relaxation_time.shape[:2] != (self.kpoints.shape[0], self.band_indices.shape[0]):
            raise ValueError("relaxation_time leading axes must match kpoints and band_indices.")
        if self.kpoint_weights.shape != (self.kpoints.shape[0],):
            raise ValueError("kpoint_weights must match kpoints.")
        if not np.all(np.isfinite(self.relaxation_time)) or np.any(self.relaxation_time <= 0.0):
            raise ValueError("relaxation_time must contain finite positive values.")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_mesh_relaxation_time",
                "schema_version": RELAXATION_TIME_MESH_NPZ_SCHEMA_VERSION,
                "relaxation_time_unit": "s",
                "convention": "hbar_over_2linewidth",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        np.savez_compressed(
            path,
            elph_mesh_relaxation_time=self.relaxation_time,
            el_kpoints=self.kpoints,
            el_kpoint_weights=self.kpoint_weights,
            el_band_indices=self.band_indices,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "RelaxationTimeMeshData":
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                relaxation_time=_required_npz_array(data, "elph_mesh_relaxation_time"),
                kpoints=_required_npz_array(data, "el_kpoints"),
                kpoint_weights=_required_npz_array(data, "el_kpoint_weights"),
                band_indices=_required_npz_array(data, "el_band_indices"),
                metadata=metadata,
            )


@dataclass
class RelaxationTimePathData:
    """Path-resolved relaxation times derived from path linewidth data."""

    relaxation_time: np.ndarray
    path_axis: str
    path_coordinates: np.ndarray
    band_indices: np.ndarray
    path_segments: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.relaxation_time = np.asarray(self.relaxation_time, dtype=float)
        self.path_axis = _normalize_path_axis(self.path_axis)
        self.path_coordinates = np.asarray(self.path_coordinates, dtype=float)
        self.band_indices = normalize_integer_indices(self.band_indices, "band_indices")
        if self.relaxation_time.size == 0:
            raise ValueError("relaxation_time must be non-empty.")
        if self.relaxation_time.ndim not in {3, 4}:
            raise ValueError(
                "relaxation_time must have shape (npath, nk, nbands) or (npath, nk, nbands, nmodes)."
            )
        if self.path_coordinates.ndim != 1 or self.path_coordinates.shape[0] != self.relaxation_time.shape[0]:
            raise ValueError("path_coordinates length must match the path-resolved relaxation_time axis.")
        if self.relaxation_time.shape[2] != self.band_indices.shape[0]:
            raise ValueError("band_indices length must match the relaxation_time band axis.")
        if not np.all(np.isfinite(self.path_coordinates)):
            raise ValueError("path_coordinates must be finite.")
        if not np.all(np.isfinite(self.relaxation_time)) or np.any(self.relaxation_time <= 0.0):
            raise ValueError("relaxation_time must contain finite positive values.")
        if self.path_segments is not None:
            self.path_segments = _normalize_path_segments(self.path_segments, self.relaxation_time.shape[0])
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_path_relaxation_time",
                "schema_version": RELAXATION_TIME_PATH_NPZ_SCHEMA_VERSION,
                "relaxation_time_unit": "s",
                "convention": "hbar_over_2linewidth",
                "path_coordinate_unit": "fractional_reciprocal_coordinate_distance",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        payload = {
            "elph_path_relaxation_time": self.relaxation_time,
            "path_axis": np.array(self.path_axis),
            "path_coordinates": self.path_coordinates,
            "el_band_indices": self.band_indices,
            "metadata_json": np.array(_metadata_to_json(self.metadata)),
        }
        if self.path_segments is not None:
            payload["path_segments"] = self.path_segments
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "RelaxationTimePathData":
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            path_axis = _scalar_string_from_npz(data, "path_axis")
            return cls(
                relaxation_time=_required_npz_array(data, "elph_path_relaxation_time"),
                path_axis=path_axis,
                path_coordinates=_required_npz_array(data, "path_coordinates"),
                band_indices=_required_npz_array(data, "el_band_indices"),
                path_segments=data["path_segments"] if "path_segments" in data else None,
                metadata=metadata,
            )


@dataclass
class TransportData:
    """SERTA transport data derived from electron-phonon linewidths."""

    conductivity: np.ndarray
    carrier_density: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.conductivity = np.asarray(self.conductivity, dtype=float)
        self.carrier_density = np.asarray(self.carrier_density, dtype=float)
        if self.conductivity.shape != (3, 3):
            raise ValueError("conductivity must have shape (3, 3).")
        if not np.all(np.isfinite(self.conductivity)):
            raise ValueError("conductivity must contain finite values.")
        if not np.all(np.isfinite(self.carrier_density)):
            raise ValueError("carrier_density must contain finite values.")
        if np.any(self.carrier_density < 0.0):
            raise ValueError("carrier_density must contain non-negative values.")
        if self.carrier_density.ndim > 1:
            raise ValueError("carrier_density must be a scalar or one-dimensional array.")
        if self.carrier_density.size == 0:
            raise ValueError("carrier_density must be non-empty.")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_transport",
                "schema_version": TRANSPORT_NPZ_SCHEMA_VERSION,
                "method": "SERTA",
                "conductivity_unit": "internal_SERTA_fractional_k",
                "carrier_density_unit": "1/input_volume",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save SERTA transport postprocess data to NPZ."""
        np.savez_compressed(
            path,
            transport_conductivity=self.conductivity,
            transport_carrier_density=self.carrier_density,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "TransportData":
        """Load SERTA transport postprocess data from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                conductivity=_required_npz_array(data, "transport_conductivity"),
                carrier_density=_required_npz_array(data, "transport_carrier_density"),
                metadata=metadata,
            )


@dataclass
class TransportScanData:
    """SERTA transport scan over chemical potentials and temperatures."""

    conductivity: np.ndarray
    carrier_density: np.ndarray
    chemical_potentials: np.ndarray
    temperatures: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.conductivity = np.asarray(self.conductivity, dtype=float)
        self.carrier_density = np.asarray(self.carrier_density, dtype=float)
        self.chemical_potentials = np.asarray(self.chemical_potentials, dtype=float)
        self.temperatures = np.asarray(self.temperatures, dtype=float)
        if self.chemical_potentials.ndim != 1 or self.chemical_potentials.size == 0:
            raise ValueError("chemical_potentials must be a one-dimensional non-empty array.")
        if self.temperatures.ndim != 1 or self.temperatures.size == 0:
            raise ValueError("temperatures must be a one-dimensional non-empty array.")
        if not np.all(np.isfinite(self.chemical_potentials)):
            raise ValueError("chemical_potentials must contain finite values.")
        if not np.all(np.isfinite(self.temperatures)) or np.any(self.temperatures <= 0.0):
            raise ValueError("temperatures must contain finite positive values.")
        expected_tensor_shape = (self.chemical_potentials.shape[0], self.temperatures.shape[0], 3, 3)
        expected_scalar_shape = (self.chemical_potentials.shape[0], self.temperatures.shape[0])
        if self.conductivity.shape != expected_tensor_shape:
            raise ValueError("conductivity must have shape (nmu, ntemperatures, 3, 3).")
        if self.carrier_density.shape != expected_scalar_shape:
            raise ValueError("carrier_density must have shape (nmu, ntemperatures).")
        if not np.all(np.isfinite(self.conductivity)):
            raise ValueError("conductivity must contain finite values.")
        if not np.all(np.isfinite(self.carrier_density)) or np.any(self.carrier_density < 0.0):
            raise ValueError("carrier_density must contain finite non-negative values.")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_transport_scan",
                "schema_version": TRANSPORT_SCAN_NPZ_SCHEMA_VERSION,
                "method": "SERTA",
                "chemical_potential_unit": "eV",
                "temperature_unit": "eV",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save SERTA transport scan postprocess data to NPZ."""
        np.savez_compressed(
            path,
            transport_scan_conductivity=self.conductivity,
            transport_scan_carrier_density=self.carrier_density,
            chemical_potentials=self.chemical_potentials,
            temperatures=self.temperatures,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "TransportScanData":
        """Load SERTA transport scan postprocess data from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                conductivity=_required_npz_array(data, "transport_scan_conductivity"),
                carrier_density=_required_npz_array(data, "transport_scan_carrier_density"),
                chemical_potentials=_required_npz_array(data, "chemical_potentials"),
                temperatures=_required_npz_array(data, "temperatures"),
                metadata=metadata,
            )


@dataclass
class MobilityData:
    """SI SERTA mobility data derived from electron-phonon linewidths."""

    conductivity: np.ndarray
    mobility: np.ndarray
    carrier_density: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.conductivity = np.asarray(self.conductivity, dtype=float)
        self.mobility = np.asarray(self.mobility, dtype=float)
        self.carrier_density = np.asarray(self.carrier_density, dtype=float)
        if self.conductivity.shape != (3, 3):
            raise ValueError("conductivity must have shape (3, 3).")
        if self.mobility.shape != (3, 3):
            raise ValueError("mobility must have shape (3, 3).")
        if not np.all(np.isfinite(self.conductivity)):
            raise ValueError("conductivity must contain finite values.")
        if not np.all(np.isfinite(self.mobility)):
            raise ValueError("mobility must contain finite values.")
        if not np.all(np.isfinite(self.carrier_density)):
            raise ValueError("carrier_density must contain finite values.")
        if np.any(self.carrier_density <= 0.0):
            raise ValueError("carrier_density must contain positive values.")
        if self.carrier_density.shape != ():
            raise ValueError("carrier_density must be a scalar.")
        conductivity_unit = _metadata_unit(self.metadata, "conductivity_unit", "S/m")
        carrier_density_unit = _metadata_unit(self.metadata, "carrier_density_unit", "m^-3")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_mobility",
                "schema_version": MOBILITY_NPZ_SCHEMA_VERSION,
                "method": "SERTA",
                "conductivity_unit": conductivity_unit,
                "carrier_density_unit": carrier_density_unit,
                "mobility_unit": "m^2/V/s",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save SI mobility postprocess data to NPZ."""
        np.savez_compressed(
            path,
            mobility_conductivity=self.conductivity,
            mobility_tensor=self.mobility,
            mobility_carrier_density=self.carrier_density,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "MobilityData":
        """Load SI mobility postprocess data from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                conductivity=_required_npz_array(data, "mobility_conductivity"),
                mobility=_required_npz_array(data, "mobility_tensor"),
                carrier_density=_required_npz_array(data, "mobility_carrier_density"),
                metadata=metadata,
            )


@dataclass
class MobilityScanData:
    """SI SERTA mobility scan over chemical potentials and temperatures."""

    conductivity: np.ndarray
    mobility: np.ndarray
    carrier_density: np.ndarray
    chemical_potentials: np.ndarray
    temperatures: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.conductivity = np.asarray(self.conductivity, dtype=float)
        self.mobility = np.asarray(self.mobility, dtype=float)
        self.carrier_density = np.asarray(self.carrier_density, dtype=float)
        self.chemical_potentials = np.asarray(self.chemical_potentials, dtype=float)
        self.temperatures = np.asarray(self.temperatures, dtype=float)
        if self.chemical_potentials.ndim != 1 or self.chemical_potentials.size == 0:
            raise ValueError("chemical_potentials must be a one-dimensional non-empty array.")
        if self.temperatures.ndim != 1 or self.temperatures.size == 0:
            raise ValueError("temperatures must be a one-dimensional non-empty array.")
        if not np.all(np.isfinite(self.chemical_potentials)):
            raise ValueError("chemical_potentials must contain finite values.")
        if not np.all(np.isfinite(self.temperatures)) or np.any(self.temperatures <= 0.0):
            raise ValueError("temperatures must contain finite positive values.")
        expected_tensor_shape = (self.chemical_potentials.shape[0], self.temperatures.shape[0], 3, 3)
        expected_scalar_shape = (self.chemical_potentials.shape[0], self.temperatures.shape[0])
        if self.conductivity.shape != expected_tensor_shape:
            raise ValueError("conductivity must have shape (nmu, ntemperatures, 3, 3).")
        if self.mobility.shape != expected_tensor_shape:
            raise ValueError("mobility must have shape (nmu, ntemperatures, 3, 3).")
        if self.carrier_density.shape != expected_scalar_shape:
            raise ValueError("carrier_density must have shape (nmu, ntemperatures).")
        if not np.all(np.isfinite(self.conductivity)):
            raise ValueError("conductivity must contain finite values.")
        if not np.all(np.isfinite(self.mobility)):
            raise ValueError("mobility must contain finite values.")
        if not np.all(np.isfinite(self.carrier_density)) or np.any(self.carrier_density <= 0.0):
            raise ValueError("carrier_density must contain finite positive values.")
        conductivity_unit = _metadata_unit(self.metadata, "conductivity_unit", "S/m")
        carrier_density_unit = _metadata_unit(self.metadata, "carrier_density_unit", "m^-3")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_mobility_scan",
                "schema_version": MOBILITY_SCAN_NPZ_SCHEMA_VERSION,
                "method": "SERTA",
                "conductivity_unit": conductivity_unit,
                "carrier_density_unit": carrier_density_unit,
                "mobility_unit": "m^2/V/s",
                "chemical_potential_unit": "eV",
                "temperature_unit": "eV",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save SI mobility scan postprocess data to NPZ."""
        np.savez_compressed(
            path,
            mobility_scan_conductivity=self.conductivity,
            mobility_scan_tensor=self.mobility,
            mobility_scan_carrier_density=self.carrier_density,
            chemical_potentials=self.chemical_potentials,
            temperatures=self.temperatures,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "MobilityScanData":
        """Load SI mobility scan postprocess data from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                conductivity=_required_npz_array(data, "mobility_scan_conductivity"),
                mobility=_required_npz_array(data, "mobility_scan_tensor"),
                carrier_density=_required_npz_array(data, "mobility_scan_carrier_density"),
                chemical_potentials=_required_npz_array(data, "chemical_potentials"),
                temperatures=_required_npz_array(data, "temperatures"),
                metadata=metadata,
            )


@dataclass
class SubspaceCouplingData:
    """Gauge-invariant EPC strength aggregated over contiguous band subspaces."""

    strength: np.ndarray
    final_group_bounds: np.ndarray
    initial_group_bounds: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.strength = np.asarray(self.strength, dtype=float)
        self.final_group_bounds = normalize_integer_array(self.final_group_bounds, "final_group_bounds")
        self.initial_group_bounds = normalize_integer_array(self.initial_group_bounds, "initial_group_bounds")
        if self.strength.ndim < 2:
            raise ValueError("strength must have at least final and initial subspace axes.")
        if not np.all(np.isfinite(self.strength)) or np.any(self.strength < 0.0):
            raise ValueError("strength must contain finite non-negative values.")
        _validate_group_bounds(self.final_group_bounds, "final_group_bounds")
        _validate_group_bounds(self.initial_group_bounds, "initial_group_bounds")
        if self.strength.shape[-2:] != (self.final_group_bounds.shape[0], self.initial_group_bounds.shape[0]):
            raise ValueError("strength subspace axes must match final_group_bounds and initial_group_bounds.")
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_subspace_coupling",
                "schema_version": SUBSPACE_COUPLING_NPZ_SCHEMA_VERSION,
                "coupling_strength_unit": "eV^2",
                "aggregation": "frobenius_norm_squared",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save gauge-invariant subspace EPC strength data to NPZ."""
        np.savez_compressed(
            path,
            elph_subspace_strength=self.strength,
            final_group_bounds=self.final_group_bounds,
            initial_group_bounds=self.initial_group_bounds,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "SubspaceCouplingData":
        """Load gauge-invariant subspace EPC strength data from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                strength=_required_npz_array(data, "elph_subspace_strength"),
                final_group_bounds=_required_npz_array(data, "final_group_bounds"),
                initial_group_bounds=_required_npz_array(data, "initial_group_bounds"),
                metadata=metadata,
            )


def compute_linewidth(
    epc_data: EPCData,
    chemical_potential: float,
    temperature: float,
    sigma: float,
    broadening: str = "gaussian",
    mode_resolved: bool = False,
    frequency_floor: float = 1e-5,
) -> LinewidthData:
    """Compute electron-phonon linewidths from ``EPCData``.

    Frequencies stored in ``EPCData`` are in THz. They are converted to eV for
    the energy-conservation broadening used in this postprocess step.
    """
    chemical_potential = validate_finite_scalar(chemical_potential, "chemical_potential")
    temperature = validate_finite_positive_scalar(temperature, "temperature")
    sigma = validate_finite_positive_scalar(sigma, "sigma")
    frequency_floor = validate_finite_positive_scalar(frequency_floor, "frequency_floor")
    mode_resolved = _validate_bool(mode_resolved, "mode_resolved")
    if np.any(epc_data.frequencies < 0.0):
        raise ValueError("frequencies must be non-negative for linewidth postprocess.")

    if not isinstance(broadening, str):
        raise ValueError("broadening must be either 'gaussian' or 'lorentzian'.")
    broadening = broadening.lower()
    if broadening not in {"gaussian", "lorentzian"}:
        raise ValueError("broadening must be either 'gaussian' or 'lorentzian'.")

    frequencies_ev = np.maximum(epc_data.frequencies, frequency_floor) * THZ_TO_EV
    nq, nk, nmodes, nsel, _ = epc_data.coupling_strength.shape
    shape = (nk, nsel, nmodes) if mode_resolved else (nk, nsel)
    absorption = np.zeros(shape, dtype=float)
    emission = np.zeros(shape, dtype=float)

    for ik in range(nk):
        for initial_band in range(nsel):
            eps_initial = epc_data.eigenvalues_k[ik, initial_band]
            for iq in range(nq):
                for mode in range(nmodes):
                    omega = frequencies_ev[iq, mode]
                    bose_occupation = _bose(omega / temperature)
                    for final_band in range(nsel):
                        eps_final = epc_data.eigenvalues_kq[iq, ik, final_band]
                        strength = epc_data.coupling_strength[iq, ik, mode, final_band, initial_band]
                        absorption_weight = _broadening(
                            eps_initial + omega - eps_final,
                            sigma,
                            broadening,
                        )
                        emission_weight = _broadening(
                            eps_initial - omega - eps_final,
                            sigma,
                            broadening,
                        )
                        absorption_term = strength * (
                            bose_occupation + _fermi((eps_final - chemical_potential) / temperature)
                        ) * absorption_weight
                        emission_term = strength * (
                            bose_occupation
                            + 1.0
                            - _fermi((eps_final - chemical_potential) / temperature)
                        ) * emission_weight

                        if mode_resolved:
                            absorption[ik, initial_band, mode] += absorption_term
                            emission[ik, initial_band, mode] += emission_term
                        else:
                            absorption[ik, initial_band] += absorption_term
                            emission[ik, initial_band] += emission_term

    prefactor = 2.0 * np.pi / nq
    absorption *= prefactor
    emission *= prefactor
    linewidth = absorption + emission
    return LinewidthData(
        linewidth=linewidth,
        absorption=absorption,
        emission=emission,
        metadata={
            "chemical_potential": chemical_potential,
            "temperature": temperature,
            "temperature_unit": "eV",
            "sigma": sigma,
            "sigma_unit": "eV",
            "broadening": broadening,
            "mode_resolved": mode_resolved,
            "linewidth_unit": "eV",
            "frequency_unit_input": epc_data.metadata.get("frequency_unit", "THz"),
            "frequency_unit_internal": "eV",
            "frequency_floor": frequency_floor,
            "frequency_floor_unit": "THz",
            "thz_to_ev": THZ_TO_EV,
        },
    )


def compute_linewidth_mesh(
    epc_mesh_data: EPCMeshData,
    chemical_potential: float,
    temperature: float,
    sigma: float,
    broadening: str = "gaussian",
    mode_resolved: bool = False,
    frequency_floor: float = 1e-5,
) -> LinewidthMeshData:
    """Compute linewidths from ``EPCMeshData`` while preserving mesh metadata."""
    chemical_potential = validate_finite_scalar(chemical_potential, "chemical_potential")
    temperature = validate_finite_positive_scalar(temperature, "temperature")
    sigma = validate_finite_positive_scalar(sigma, "sigma")
    frequency_floor = validate_finite_positive_scalar(frequency_floor, "frequency_floor")
    mode_resolved = _validate_bool(mode_resolved, "mode_resolved")
    if np.any(epc_mesh_data.frequencies < 0.0):
        raise ValueError("frequencies must be non-negative for linewidth postprocess.")

    if not isinstance(broadening, str):
        raise ValueError("broadening must be either 'gaussian' or 'lorentzian'.")
    broadening = broadening.lower()
    if broadening not in {"gaussian", "lorentzian"}:
        raise ValueError("broadening must be either 'gaussian' or 'lorentzian'.")

    frequencies_ev = np.maximum(epc_mesh_data.frequencies, frequency_floor) * THZ_TO_EV
    qpoint_weights = _normalize_weights(epc_mesh_data.qpoint_weights, "qpoint_weights")
    nq, nk, nmodes, nsel, _ = epc_mesh_data.coupling_strength.shape
    shape = (nk, nsel, nmodes) if mode_resolved else (nk, nsel)
    absorption = np.zeros(shape, dtype=float)
    emission = np.zeros(shape, dtype=float)

    for ik in range(nk):
        for initial_band in range(nsel):
            eps_initial = epc_mesh_data.eigenvalues_k[ik, initial_band]
            for iq in range(nq):
                q_weight = qpoint_weights[iq]
                for mode in range(nmodes):
                    omega = frequencies_ev[iq, mode]
                    bose_occupation = _bose(omega / temperature)
                    for final_band in range(nsel):
                        eps_final = epc_mesh_data.eigenvalues_kq[iq, ik, final_band]
                        strength = epc_mesh_data.coupling_strength[iq, ik, mode, final_band, initial_band]
                        absorption_weight = _broadening(
                            eps_initial + omega - eps_final,
                            sigma,
                            broadening,
                        )
                        emission_weight = _broadening(
                            eps_initial - omega - eps_final,
                            sigma,
                            broadening,
                        )
                        absorption_term = q_weight * strength * (
                            bose_occupation + _fermi((eps_final - chemical_potential) / temperature)
                        ) * absorption_weight
                        emission_term = q_weight * strength * (
                            bose_occupation
                            + 1.0
                            - _fermi((eps_final - chemical_potential) / temperature)
                        ) * emission_weight

                        if mode_resolved:
                            absorption[ik, initial_band, mode] += absorption_term
                            emission[ik, initial_band, mode] += emission_term
                        else:
                            absorption[ik, initial_band] += absorption_term
                            emission[ik, initial_band] += emission_term

    absorption *= 2.0 * np.pi
    emission *= 2.0 * np.pi
    linewidth = absorption + emission
    return LinewidthMeshData(
        linewidth=linewidth,
        absorption=absorption,
        emission=emission,
        kpoints=epc_mesh_data.kpoints,
        kpoint_weights=epc_mesh_data.kpoint_weights,
        band_indices=epc_mesh_data.band_indices,
        metadata={
            "chemical_potential": chemical_potential,
            "temperature": temperature,
            "temperature_unit": "eV",
            "sigma": sigma,
            "sigma_unit": "eV",
            "broadening": broadening,
            "mode_resolved": mode_resolved,
            "linewidth_unit": "eV",
            "frequency_unit_input": epc_mesh_data.metadata.get("frequency_unit", "THz"),
            "frequency_unit_internal": "eV",
            "frequency_floor": frequency_floor,
            "frequency_floor_unit": "THz",
            "thz_to_ev": THZ_TO_EV,
            "epc_mesh_schema": epc_mesh_data.metadata.get("schema"),
            "qpoint_weights": qpoint_weights,
            "q_weight_convention": "normalized_sum_to_one",
            "mesh_spec": epc_mesh_data.metadata.get("mesh_spec"),
        },
    )


def compute_linewidth_mesh_chunked_artifact(
    directory: Union[str, Path],
    chemical_potential: float,
    temperature: float,
    sigma: float,
    broadening: str = "gaussian",
    mode_resolved: bool = False,
    frequency_floor: float = 1e-5,
) -> LinewidthMeshData:
    """Compute mesh linewidth by reducing an EPC mesh chunked artifact.

    This summary-first helper reads one chunk NPZ at a time. For q-axis
    artifacts it accumulates weighted q contributions without materializing the
    full ``EPCMeshData`` coupling tensor. For k-axis artifacts it computes each
    k chunk independently and concatenates the reduced linewidth arrays.
    """
    directory = Path(directory)
    manifest = read_epc_mesh_chunked_manifest(directory)
    axis = manifest["axis"]
    kpoint_weights, qpoint_weights = read_epc_mesh_chunked_weights(directory, manifest)

    if axis == "q":
        linewidth_result = None
        for entry in manifest["chunks"]:
            spec = entry["spec"]
            q_slice = slice(int(spec["q_start"]), int(spec["q_stop"]))
            chunk = EPCData.load_npz(directory / entry["filename"])
            q_weights = qpoint_weights[q_slice]
            chunk_result = compute_linewidth_mesh(
                EPCMeshData.from_epc_data(
                    chunk,
                    kpoint_weights=kpoint_weights,
                    qpoint_weights=q_weights,
                    metadata=manifest.get("mesh_metadata") or {},
                ),
                chemical_potential=chemical_potential,
                temperature=temperature,
                sigma=sigma,
                broadening=broadening,
                mode_resolved=mode_resolved,
                frequency_floor=frequency_floor,
            )
            q_weight_sum = float(q_weights.sum())
            if linewidth_result is None:
                linewidth_result = chunk_result
                linewidth_result.absorption *= q_weight_sum
                linewidth_result.emission *= q_weight_sum
                linewidth_result.linewidth *= q_weight_sum
            else:
                _validate_linewidth_chunk_compatibility(linewidth_result, chunk_result)
                linewidth_result.absorption += chunk_result.absorption * q_weight_sum
                linewidth_result.emission += chunk_result.emission * q_weight_sum
                linewidth_result.linewidth += chunk_result.linewidth * q_weight_sum
        result = linewidth_result
    else:
        linewidth_chunks = []
        for entry in manifest["chunks"]:
            spec = entry["spec"]
            k_slice = slice(int(spec["k_start"]), int(spec["k_stop"]))
            chunk = EPCData.load_npz(directory / entry["filename"])
            linewidth_chunks.append(
                compute_linewidth_mesh(
                    EPCMeshData.from_epc_data(
                        chunk,
                        kpoint_weights=kpoint_weights[k_slice],
                        qpoint_weights=qpoint_weights,
                        metadata=manifest.get("mesh_metadata") or {},
                    ),
                    chemical_potential=chemical_potential,
                    temperature=temperature,
                    sigma=sigma,
                    broadening=broadening,
                    mode_resolved=mode_resolved,
                    frequency_floor=frequency_floor,
                )
            )
        first = linewidth_chunks[0]
        result = LinewidthMeshData(
            linewidth=np.concatenate([chunk.linewidth for chunk in linewidth_chunks], axis=0),
            absorption=np.concatenate([chunk.absorption for chunk in linewidth_chunks], axis=0),
            emission=np.concatenate([chunk.emission for chunk in linewidth_chunks], axis=0),
            kpoints=np.concatenate([chunk.kpoints for chunk in linewidth_chunks], axis=0),
            kpoint_weights=kpoint_weights,
            band_indices=first.band_indices,
            metadata=dict(first.metadata),
        )

    if result is None:
        raise ValueError("EPC mesh chunked artifact must contain at least one chunk.")
    result.metadata.update(
        {
            "source": "deeptb.eph.compute_linewidth_mesh_chunked_artifact",
            "chunked_artifact": True,
            "artifact_axis": axis,
            "artifact_chunk_count": len(manifest["chunks"]),
            "artifact_schema": manifest.get("schema"),
            "artifact_schema_version": manifest.get("schema_version"),
            "summary_first": True,
        }
    )
    return result


def _validate_linewidth_chunk_compatibility(reference: LinewidthMeshData, chunk: LinewidthMeshData) -> None:
    if not np.array_equal(reference.kpoints, chunk.kpoints):
        raise ValueError("Linewidth chunks must share kpoints for q-axis reduction.")
    if not np.array_equal(reference.band_indices, chunk.band_indices):
        raise ValueError("Linewidth chunks must share band_indices.")
    if reference.linewidth.shape != chunk.linewidth.shape:
        raise ValueError("Linewidth chunks must share linewidth shape.")


def compute_linewidth_path(
    epc_path_data: EPCPathData,
    chemical_potential: float,
    temperature: float,
    sigma: float,
    broadening: str = "gaussian",
    mode_resolved: bool = False,
    frequency_floor: float = 1e-5,
) -> LinewidthPathData:
    """Compute path-resolved linewidth contributions from ``EPCPathData``.

    Unlike :func:`compute_linewidth`, this keeps the EPC path axis instead of
    reducing over it, which makes the result suitable for q-path diagnostics and
    plotting. The current implementation supports the fixed-k + q-path contract
    produced by ``TBSystem.eph.compute_path``.
    """
    if epc_path_data.path_axis != "q":
        raise NotImplementedError("Path linewidth currently supports EPCPathData with path_axis='q' only.")
    chemical_potential = validate_finite_scalar(chemical_potential, "chemical_potential")
    temperature = validate_finite_positive_scalar(temperature, "temperature")
    sigma = validate_finite_positive_scalar(sigma, "sigma")
    frequency_floor = validate_finite_positive_scalar(frequency_floor, "frequency_floor")
    mode_resolved = _validate_bool(mode_resolved, "mode_resolved")
    if np.any(epc_path_data.frequencies < 0.0):
        raise ValueError("frequencies must be non-negative for linewidth postprocess.")

    if not isinstance(broadening, str):
        raise ValueError("broadening must be either 'gaussian' or 'lorentzian'.")
    broadening = broadening.lower()
    if broadening not in {"gaussian", "lorentzian"}:
        raise ValueError("broadening must be either 'gaussian' or 'lorentzian'.")

    frequencies_ev = np.maximum(epc_path_data.frequencies, frequency_floor) * THZ_TO_EV
    nq, nk, nmodes, nsel, _ = epc_path_data.coupling_strength.shape
    shape = (nq, nk, nsel, nmodes) if mode_resolved else (nq, nk, nsel)
    absorption = np.zeros(shape, dtype=float)
    emission = np.zeros(shape, dtype=float)

    for iq in range(nq):
        for ik in range(nk):
            for initial_band in range(nsel):
                eps_initial = epc_path_data.eigenvalues_k[ik, initial_band]
                for mode in range(nmodes):
                    omega = frequencies_ev[iq, mode]
                    bose_occupation = _bose(omega / temperature)
                    for final_band in range(nsel):
                        eps_final = epc_path_data.eigenvalues_kq[iq, ik, final_band]
                        strength = epc_path_data.coupling_strength[iq, ik, mode, final_band, initial_band]
                        absorption_weight = _broadening(
                            eps_initial + omega - eps_final,
                            sigma,
                            broadening,
                        )
                        emission_weight = _broadening(
                            eps_initial - omega - eps_final,
                            sigma,
                            broadening,
                        )
                        absorption_term = strength * (
                            bose_occupation + _fermi((eps_final - chemical_potential) / temperature)
                        ) * absorption_weight
                        emission_term = strength * (
                            bose_occupation
                            + 1.0
                            - _fermi((eps_final - chemical_potential) / temperature)
                        ) * emission_weight

                        if mode_resolved:
                            absorption[iq, ik, initial_band, mode] += absorption_term
                            emission[iq, ik, initial_band, mode] += emission_term
                        else:
                            absorption[iq, ik, initial_band] += absorption_term
                            emission[iq, ik, initial_band] += emission_term

    prefactor = 2.0 * np.pi / nq
    absorption *= prefactor
    emission *= prefactor
    linewidth = absorption + emission
    return LinewidthPathData(
        linewidth=linewidth,
        absorption=absorption,
        emission=emission,
        path_axis=epc_path_data.path_axis,
        path_coordinates=epc_path_data.path_coordinates,
        path_segments=epc_path_data.path_segments,
        band_indices=epc_path_data.band_indices,
        metadata={
            "chemical_potential": chemical_potential,
            "temperature": temperature,
            "temperature_unit": "eV",
            "sigma": sigma,
            "sigma_unit": "eV",
            "broadening": broadening,
            "mode_resolved": mode_resolved,
            "linewidth_unit": "eV",
            "frequency_unit_input": epc_path_data.metadata.get("frequency_unit", "THz"),
            "frequency_unit_internal": "eV",
            "frequency_floor": frequency_floor,
            "frequency_floor_unit": "THz",
            "thz_to_ev": THZ_TO_EV,
            "epc_path_schema": epc_path_data.metadata.get("schema"),
            "path_axis": epc_path_data.path_axis,
            "path_mode": epc_path_data.metadata.get("path_mode"),
            "aggregation": "per_path_point_contribution",
        },
    )


def compute_relaxation_time(linewidth_data: LinewidthData, sum_modes: bool = False) -> RelaxationTimeData:
    """Compute relaxation time from EPC linewidths.

    The convention is ``tau = hbar / (2 * linewidth)`` with linewidth in eV,
    giving relaxation time in seconds. Mode-resolved linewidths keep their
    mode axis by default; set ``sum_modes=True`` to sum the last axis first.
    """
    sum_modes = _validate_bool(sum_modes, "sum_modes")
    linewidth = np.asarray(linewidth_data.linewidth, dtype=float)
    if sum_modes:
        if linewidth.ndim < 3:
            raise ValueError("sum_modes=True requires a mode-resolved linewidth array.")
        linewidth = linewidth.sum(axis=-1)
    if not np.all(np.isfinite(linewidth)) or np.any(linewidth <= 0.0):
        raise ValueError("linewidth values must be finite and positive for relaxation time.")

    return RelaxationTimeData(
        relaxation_time=HBAR_EV_S / (2.0 * linewidth),
        metadata={
            "linewidth_schema": linewidth_data.metadata.get("schema"),
            "linewidth_unit": linewidth_data.metadata.get("linewidth_unit", "eV"),
            "relaxation_time_unit": "s",
            "convention": "hbar_over_2linewidth",
            "hbar_eV_s": HBAR_EV_S,
            "mode_resolved_input": linewidth_data.linewidth.ndim == 3,
            "sum_modes": sum_modes,
        },
    )


def compute_relaxation_time_mesh(
    linewidth_mesh_data: LinewidthMeshData,
    sum_modes: bool = False,
) -> RelaxationTimeMeshData:
    """Compute mesh relaxation time from mesh linewidths."""
    sum_modes = _validate_bool(sum_modes, "sum_modes")
    linewidth = np.asarray(linewidth_mesh_data.linewidth, dtype=float)
    if sum_modes:
        if linewidth.ndim < 3:
            raise ValueError("sum_modes=True requires a mode-resolved mesh linewidth array.")
        linewidth = linewidth.sum(axis=-1)
    if not np.all(np.isfinite(linewidth)) or np.any(linewidth <= 0.0):
        raise ValueError("linewidth values must be finite and positive for relaxation time.")

    return RelaxationTimeMeshData(
        relaxation_time=HBAR_EV_S / (2.0 * linewidth),
        kpoints=linewidth_mesh_data.kpoints,
        kpoint_weights=linewidth_mesh_data.kpoint_weights,
        band_indices=linewidth_mesh_data.band_indices,
        metadata={
            "linewidth_mesh_schema": linewidth_mesh_data.metadata.get("schema"),
            "linewidth_unit": linewidth_mesh_data.metadata.get("linewidth_unit", "eV"),
            "relaxation_time_unit": "s",
            "convention": "hbar_over_2linewidth",
            "hbar_eV_s": HBAR_EV_S,
            "mode_resolved_input": linewidth_mesh_data.linewidth.ndim == 3,
            "sum_modes": sum_modes,
            "mesh_spec": linewidth_mesh_data.metadata.get("mesh_spec"),
        },
    )


def compute_relaxation_time_path(
    linewidth_path_data: LinewidthPathData,
    sum_modes: bool = False,
) -> RelaxationTimePathData:
    """Compute path-resolved relaxation time from path linewidth contributions."""
    sum_modes = _validate_bool(sum_modes, "sum_modes")
    linewidth = np.asarray(linewidth_path_data.linewidth, dtype=float)
    if sum_modes:
        if linewidth.ndim < 4:
            raise ValueError("sum_modes=True requires a mode-resolved path linewidth array.")
        linewidth = linewidth.sum(axis=-1)
    if not np.all(np.isfinite(linewidth)) or np.any(linewidth <= 0.0):
        raise ValueError("linewidth values must be finite and positive for relaxation time.")

    return RelaxationTimePathData(
        relaxation_time=HBAR_EV_S / (2.0 * linewidth),
        path_axis=linewidth_path_data.path_axis,
        path_coordinates=linewidth_path_data.path_coordinates,
        path_segments=linewidth_path_data.path_segments,
        band_indices=linewidth_path_data.band_indices,
        metadata={
            "linewidth_path_schema": linewidth_path_data.metadata.get("schema"),
            "linewidth_unit": linewidth_path_data.metadata.get("linewidth_unit", "eV"),
            "relaxation_time_unit": "s",
            "convention": "hbar_over_2linewidth",
            "hbar_eV_s": HBAR_EV_S,
            "mode_resolved_input": linewidth_path_data.linewidth.ndim == 4,
            "sum_modes": sum_modes,
            "path_axis": linewidth_path_data.path_axis,
            "path_mode": linewidth_path_data.metadata.get("path_mode"),
        },
    )


def compute_serta_conductivity(
    eigenvalues: np.ndarray,
    velocities: np.ndarray,
    linewidth: np.ndarray,
    chemical_potential: float,
    temperature: float,
    kpoint_weights: Optional[np.ndarray] = None,
    spin_degeneracy: int = 1,
    volume: float = 1.0,
) -> TransportData:
    """Compute SERTA conductivity from linewidths and band velocities.

    ``eigenvalues`` has shape ``(nk, nbands)`` and is expressed in eV.
    ``velocities`` has shape ``(nk, nbands, 3)`` and is expected in the same
    velocity convention as DeePTB's electronic postprocess. ``linewidth`` is an
    inverse lifetime in eV as returned by :func:`compute_linewidth`.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    linewidth = np.asarray(linewidth, dtype=float)
    chemical_potential = validate_finite_scalar(chemical_potential, "chemical_potential")
    temperature = validate_finite_positive_scalar(temperature, "temperature")
    volume = validate_finite_positive_scalar(volume, "volume")
    try:
        spin_degeneracy_value = np.asarray(spin_degeneracy)
    except (TypeError, ValueError):
        raise ValueError("spin_degeneracy must be a positive integer.") from None
    if spin_degeneracy_value.shape != ():
        raise ValueError("spin_degeneracy must be a positive integer.")
    if isinstance(spin_degeneracy, (bool, np.bool_)):
        raise ValueError("spin_degeneracy must be a positive integer.")
    if not np.issubdtype(spin_degeneracy_value.dtype, np.number):
        raise ValueError("spin_degeneracy must be a positive integer.")
    try:
        spin_degeneracy_float = float(spin_degeneracy_value)
    except (TypeError, ValueError):
        raise ValueError("spin_degeneracy must be a positive integer.") from None
    if not np.isfinite(spin_degeneracy_float) or spin_degeneracy_float <= 0.0:
        raise ValueError("spin_degeneracy must be a positive integer.")
    if int(spin_degeneracy_float) != spin_degeneracy_float:
        raise ValueError("spin_degeneracy must be a positive integer.")
    spin_degeneracy = int(spin_degeneracy_float)
    if eigenvalues.ndim != 2:
        raise ValueError("eigenvalues must have shape (nk, nbands).")
    if eigenvalues.shape[0] == 0:
        raise ValueError("eigenvalues must contain at least one k-point.")
    if eigenvalues.shape[1] == 0:
        raise ValueError("eigenvalues must contain at least one band.")
    if velocities.shape != (*eigenvalues.shape, 3):
        raise ValueError("velocities must have shape (nk, nbands, 3).")
    if linewidth.shape != eigenvalues.shape:
        raise ValueError("linewidth must have shape (nk, nbands).")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("eigenvalues must contain finite values.")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("velocities must contain finite values.")
    if not np.all(np.isfinite(linewidth)) or np.any(linewidth <= 0.0):
        raise ValueError("linewidth values must be finite and positive for SERTA conductivity.")

    nk = eigenvalues.shape[0]
    if kpoint_weights is None:
        weights = np.full((nk,), 1.0 / nk, dtype=float)
    else:
        weights = np.asarray(kpoint_weights, dtype=float)
        if weights.shape != (nk,):
            raise ValueError("kpoint_weights must have shape (nk,).")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("kpoint_weights must contain finite non-negative values.")
        weight_sum = weights.sum()
        if weight_sum <= 0.0:
            raise ValueError("kpoint_weights must have a positive sum.")
        weights = weights / weight_sum

    conductivity = np.zeros((3, 3), dtype=float)
    carrier_density = 0.0
    for ik in range(nk):
        for iband in range(eigenvalues.shape[1]):
            eps = eigenvalues[ik, iband]
            occupation = _fermi((eps - chemical_potential) / temperature)
            transport_weight = -_dfermi_deps((eps - chemical_potential) / temperature) / temperature
            weight = spin_degeneracy * weights[ik] / volume
            carrier_density += weight * occupation
            conductivity += (
                weight
                * transport_weight
                * np.outer(velocities[ik, iband], velocities[ik, iband])
                / linewidth[ik, iband]
            )

    return TransportData(
        conductivity=conductivity,
        carrier_density=np.asarray(carrier_density),
        metadata={
            "chemical_potential": chemical_potential,
            "temperature": temperature,
            "temperature_unit": "eV",
            "spin_degeneracy": spin_degeneracy,
            "volume": volume,
            "energy_unit": "eV",
            "linewidth_unit": "eV",
            "method": "SERTA",
        },
    )


def compute_serta_transport_scan(
    eigenvalues: np.ndarray,
    velocities: np.ndarray,
    linewidth: np.ndarray,
    chemical_potentials: Sequence[float],
    temperatures: Sequence[float],
    kpoint_weights: Optional[np.ndarray] = None,
    spin_degeneracy: int = 1,
    volume: float = 1.0,
) -> TransportScanData:
    """Compute SERTA transport over chemical-potential and temperature grids.

    The scan uses a fixed linewidth input, matching the current SI mobility
    scan convention. Per-scan-point linewidth recomputation should use a
    separate API to avoid changing this meaning.
    """
    chemical_potentials = np.atleast_1d(np.asarray(chemical_potentials, dtype=float))
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if chemical_potentials.ndim != 1 or chemical_potentials.size == 0:
        raise ValueError("chemical_potentials must be a one-dimensional non-empty array.")
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("temperatures must be a one-dimensional non-empty array.")
    if not np.all(np.isfinite(chemical_potentials)):
        raise ValueError("chemical_potentials must contain finite values.")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
        raise ValueError("temperatures must contain finite positive values.")

    nmu = chemical_potentials.shape[0]
    ntemperatures = temperatures.shape[0]
    conductivity = np.zeros((nmu, ntemperatures, 3, 3), dtype=float)
    carrier_density = np.zeros((nmu, ntemperatures), dtype=float)
    metadata = None
    for imu, chemical_potential in enumerate(chemical_potentials):
        for itemperature, temperature in enumerate(temperatures):
            point = compute_serta_conductivity(
                eigenvalues=eigenvalues,
                velocities=velocities,
                linewidth=linewidth,
                chemical_potential=float(chemical_potential),
                temperature=float(temperature),
                kpoint_weights=kpoint_weights,
                spin_degeneracy=spin_degeneracy,
                volume=volume,
            )
            conductivity[imu, itemperature] = point.conductivity
            carrier_density[imu, itemperature] = point.carrier_density
            if metadata is None:
                metadata = dict(point.metadata)

    scan_metadata = dict(metadata or {})
    scan_metadata.pop("schema", None)
    scan_metadata.pop("schema_version", None)
    scan_metadata.pop("chemical_potential", None)
    scan_metadata.pop("temperature", None)
    scan_metadata.update(
        {
            "chemical_potential_count": int(nmu),
            "temperature_count": int(ntemperatures),
            "scan_axes": ["chemical_potential", "temperature"],
            "linewidth_scan_convention": "fixed_linewidth",
        }
    )
    return TransportScanData(
        conductivity=conductivity,
        carrier_density=carrier_density,
        chemical_potentials=chemical_potentials,
        temperatures=temperatures,
        metadata=scan_metadata,
    )


def fractional_band_velocities_to_si(velocities: np.ndarray, reciprocal_cell: np.ndarray) -> np.ndarray:
    """Convert eV/fractional reciprocal-coordinate velocities to m/s.

    ``reciprocal_cell`` has shape ``(3, 3)`` and maps fractional reciprocal
    coordinates to Cartesian reciprocal coordinates in Angstrom^-1 using
    ``k_cart = k_fractional @ reciprocal_cell``.
    """
    velocities = np.asarray(velocities, dtype=float)
    reciprocal_cell = np.asarray(reciprocal_cell, dtype=float)
    if velocities.ndim != 3 or velocities.shape[-1] != 3:
        raise ValueError("velocities must have shape (nk, nbands, 3).")
    if reciprocal_cell.shape != (3, 3):
        raise ValueError("reciprocal_cell must have shape (3, 3).")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("velocities must contain finite values.")
    if not np.all(np.isfinite(reciprocal_cell)):
        raise ValueError("reciprocal_cell must contain finite values.")
    if abs(np.linalg.det(reciprocal_cell)) <= 0.0:
        raise ValueError("reciprocal_cell must be invertible.")

    cartesian_gradient = velocities @ np.linalg.inv(reciprocal_cell)
    return cartesian_gradient * ANGSTROM_TO_M / HBAR_EV_S


def compute_serta_mobility_si(
    eigenvalues: np.ndarray,
    velocities: np.ndarray,
    linewidth: np.ndarray,
    reciprocal_cell: np.ndarray,
    chemical_potential: float,
    temperature: float,
    kpoint_weights: Optional[np.ndarray] = None,
    spin_degeneracy: int = 1,
    dimension: str = "3d",
    volume: Optional[float] = None,
    area: Optional[float] = None,
) -> MobilityData:
    """Compute SI SERTA conductivity and mobility.

    Energies and linewidths are in eV. ``temperature`` is kBT in eV, matching
    the existing EPC postprocess convention. ``velocities`` are in
    eV/fractional reciprocal-coordinate and are converted to m/s through
    ``reciprocal_cell``.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    linewidth = np.asarray(linewidth, dtype=float)
    velocities_si = fractional_band_velocities_to_si(velocities, reciprocal_cell)
    chemical_potential = validate_finite_scalar(chemical_potential, "chemical_potential")
    temperature = validate_finite_positive_scalar(temperature, "temperature")

    if not isinstance(dimension, str):
        raise ValueError("dimension must be '2d' or '3d'.")
    dimension = dimension.lower()
    if dimension not in {"2d", "3d"}:
        raise ValueError("dimension must be '2d' or '3d'.")
    if dimension == "3d":
        if area is not None:
            raise ValueError("area must not be set for dimension='3d'.")
        normalization = validate_finite_positive_scalar(volume, "volume")
        density_scale = normalization * ANGSTROM_TO_M**3
        conductivity_unit = "S/m"
        carrier_density_unit = "m^-3"
        normalization_metadata = {"volume": normalization, "volume_unit": "Angstrom^3"}
    else:
        if volume is not None:
            raise ValueError("volume must not be set for dimension='2d'.")
        normalization = validate_finite_positive_scalar(area, "area")
        density_scale = normalization * ANGSTROM_TO_M**2
        conductivity_unit = "S"
        carrier_density_unit = "m^-2"
        normalization_metadata = {"area": normalization, "area_unit": "Angstrom^2"}

    try:
        spin_degeneracy_value = np.asarray(spin_degeneracy)
    except (TypeError, ValueError):
        raise ValueError("spin_degeneracy must be a positive integer.") from None
    if spin_degeneracy_value.shape != ():
        raise ValueError("spin_degeneracy must be a positive integer.")
    if isinstance(spin_degeneracy, (bool, np.bool_)):
        raise ValueError("spin_degeneracy must be a positive integer.")
    try:
        spin_degeneracy_float = float(spin_degeneracy_value)
    except (TypeError, ValueError):
        raise ValueError("spin_degeneracy must be a positive integer.") from None
    if not np.isfinite(spin_degeneracy_float) or spin_degeneracy_float <= 0.0:
        raise ValueError("spin_degeneracy must be a positive integer.")
    if int(spin_degeneracy_float) != spin_degeneracy_float:
        raise ValueError("spin_degeneracy must be a positive integer.")
    spin_degeneracy = int(spin_degeneracy_float)

    if eigenvalues.ndim != 2:
        raise ValueError("eigenvalues must have shape (nk, nbands).")
    if eigenvalues.shape[0] == 0 or eigenvalues.shape[1] == 0:
        raise ValueError("eigenvalues must be non-empty.")
    if velocities_si.shape != (*eigenvalues.shape, 3):
        raise ValueError("velocities must have shape (nk, nbands, 3).")
    if linewidth.shape != eigenvalues.shape:
        raise ValueError("linewidth must have shape (nk, nbands).")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("eigenvalues must contain finite values.")
    if not np.all(np.isfinite(linewidth)) or np.any(linewidth <= 0.0):
        raise ValueError("linewidth values must be finite and positive for mobility.")

    nk = eigenvalues.shape[0]
    if kpoint_weights is None:
        weights = np.full((nk,), 1.0 / nk, dtype=float)
    else:
        weights = np.asarray(kpoint_weights, dtype=float)
        if weights.shape != (nk,):
            raise ValueError("kpoint_weights must have shape (nk,).")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("kpoint_weights must contain finite non-negative values.")
        weight_sum = weights.sum()
        if weight_sum <= 0.0:
            raise ValueError("kpoint_weights must have a positive sum.")
        weights = weights / weight_sum

    conductivity = np.zeros((3, 3), dtype=float)
    carrier_density = 0.0
    for ik in range(nk):
        for iband in range(eigenvalues.shape[1]):
            eps = eigenvalues[ik, iband]
            occupation = _fermi((eps - chemical_potential) / temperature)
            transport_weight_ev = -_dfermi_deps((eps - chemical_potential) / temperature) / temperature
            tau = HBAR_EV_S / (2.0 * linewidth[ik, iband])
            density_weight = spin_degeneracy * weights[ik] / density_scale
            carrier_density += density_weight * occupation
            conductivity += (
                density_weight
                * (ELECTRON_CHARGE_C**2 / eV2J)
                * transport_weight_ev
                * tau
                * np.outer(velocities_si[ik, iband], velocities_si[ik, iband])
            )
    if carrier_density <= 0.0:
        raise ValueError("carrier_density must be positive for mobility.")
    mobility = conductivity / (carrier_density * ELECTRON_CHARGE_C)

    metadata = {
        "chemical_potential": chemical_potential,
        "temperature": temperature,
        "temperature_unit": "eV",
        "spin_degeneracy": spin_degeneracy,
        "dimension": dimension,
        "energy_unit": "eV",
        "linewidth_unit": "eV",
        "velocity_input_unit": "eV/fractional_reciprocal_coordinate",
        "velocity_unit": "m/s",
        "reciprocal_cell_unit": "Angstrom^-1",
        "conductivity_unit": conductivity_unit,
        "carrier_density_unit": carrier_density_unit,
        "mobility_unit": "m^2/V/s",
        "method": "SERTA",
        "relaxation_time_convention": "hbar_over_2linewidth",
    }
    metadata.update(normalization_metadata)
    return MobilityData(conductivity=conductivity, mobility=mobility, carrier_density=np.asarray(carrier_density), metadata=metadata)


def compute_serta_mobility_scan_si(
    eigenvalues: np.ndarray,
    velocities: np.ndarray,
    linewidth: np.ndarray,
    reciprocal_cell: np.ndarray,
    chemical_potentials: Sequence[float],
    temperatures: Sequence[float],
    kpoint_weights: Optional[np.ndarray] = None,
    spin_degeneracy: int = 1,
    dimension: str = "3d",
    volume: Optional[float] = None,
    area: Optional[float] = None,
) -> MobilityScanData:
    """Compute SI SERTA mobility over chemical-potential and temperature grids."""
    chemical_potentials = np.atleast_1d(np.asarray(chemical_potentials, dtype=float))
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if chemical_potentials.ndim != 1 or chemical_potentials.size == 0:
        raise ValueError("chemical_potentials must be a one-dimensional non-empty array.")
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("temperatures must be a one-dimensional non-empty array.")
    if not np.all(np.isfinite(chemical_potentials)):
        raise ValueError("chemical_potentials must contain finite values.")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
        raise ValueError("temperatures must contain finite positive values.")

    nmu = chemical_potentials.shape[0]
    ntemperatures = temperatures.shape[0]
    conductivity = np.zeros((nmu, ntemperatures, 3, 3), dtype=float)
    mobility = np.zeros((nmu, ntemperatures, 3, 3), dtype=float)
    carrier_density = np.zeros((nmu, ntemperatures), dtype=float)
    metadata = None
    for imu, chemical_potential in enumerate(chemical_potentials):
        for itemperature, temperature in enumerate(temperatures):
            point = compute_serta_mobility_si(
                eigenvalues=eigenvalues,
                velocities=velocities,
                linewidth=linewidth,
                reciprocal_cell=reciprocal_cell,
                chemical_potential=float(chemical_potential),
                temperature=float(temperature),
                kpoint_weights=kpoint_weights,
                spin_degeneracy=spin_degeneracy,
                dimension=dimension,
                volume=volume,
                area=area,
            )
            conductivity[imu, itemperature] = point.conductivity
            mobility[imu, itemperature] = point.mobility
            carrier_density[imu, itemperature] = point.carrier_density
            if metadata is None:
                metadata = dict(point.metadata)

    scan_metadata = dict(metadata or {})
    scan_metadata.pop("schema", None)
    scan_metadata.pop("schema_version", None)
    scan_metadata.pop("chemical_potential", None)
    scan_metadata.pop("temperature", None)
    scan_metadata.update(
        {
            "chemical_potential_count": int(nmu),
            "temperature_count": int(ntemperatures),
            "scan_axes": ["chemical_potential", "temperature"],
        }
    )
    return MobilityScanData(
        conductivity=conductivity,
        mobility=mobility,
        carrier_density=carrier_density,
        chemical_potentials=chemical_potentials,
        temperatures=temperatures,
        metadata=scan_metadata,
    )


def compute_band_velocities_finite_difference(
    system,
    kpoints: np.ndarray,
    bands: Optional[Sequence[int]] = None,
    delta: float = 1e-4,
    use_scc: bool = False,
    **solver_kwargs,
) -> np.ndarray:
    """Estimate band velocities by central finite differences in k-space.

    The returned velocity convention is eV per fractional reciprocal-coordinate
    component. This is intentionally a thin workflow bridge over
    ``system.get_eigenvalues``; it does not perform SI unit conversion.
    """
    if use_scc:
        raise NotImplementedError("SCC-corrected electron-phonon transport is not supported in v1.")
    delta = validate_finite_positive_scalar(delta, "delta")

    kpoints = normalize_kpoints(kpoints)
    reference_eigenvalues = _get_system_eigenvalues(system, kpoints, use_scc=use_scc, **solver_kwargs)
    band_indices = _normalize_band_indices(bands, reference_eigenvalues.shape[1])
    velocities = np.zeros((kpoints.shape[0], band_indices.shape[0], 3), dtype=float)

    for axis in range(3):
        shift = np.zeros(3, dtype=float)
        shift[axis] = delta
        eig_plus = _get_system_eigenvalues(system, kpoints + shift, use_scc=use_scc, **solver_kwargs)
        eig_minus = _get_system_eigenvalues(system, kpoints - shift, use_scc=use_scc, **solver_kwargs)
        if eig_plus.shape != reference_eigenvalues.shape or eig_minus.shape != reference_eigenvalues.shape:
            raise ValueError("system.get_eigenvalues must return a consistent eigenvalue shape for shifted kpoints.")
        velocities[:, :, axis] = (eig_plus[:, band_indices] - eig_minus[:, band_indices]) / (2.0 * delta)

    return velocities


def compute_band_velocities_hamiltonian_derivative(
    system,
    kpoints: np.ndarray,
    bands: Optional[Sequence[int]] = None,
    use_scc: bool = False,
    **solver_kwargs,
) -> np.ndarray:
    """Compute band velocities from analytic ``dH(k)/dk`` and ``dS(k)/dk``.

    The returned convention matches :func:`compute_band_velocities_finite_difference`:
    eV per fractional reciprocal-coordinate component. This provider does not
    perform SI conversion and does not resolve degenerate-subspace gauge issues.
    """
    if use_scc:
        raise NotImplementedError("SCC-corrected electron-phonon transport is not supported in v1.")
    if solver_kwargs:
        raise ValueError("solver_kwargs are not supported by hamiltonian_derivative velocity provider.")

    kpoints = normalize_kpoints(kpoints)
    hk_payload = system.get_hk(k_points=kpoints, use_scc=use_scc, with_derivative=True)
    if not isinstance(hk_payload, tuple) or len(hk_payload) != 4:
        raise ValueError("system.get_hk(..., with_derivative=True) must return (Hk, dHdk, Sk, dSdk).")
    hk, dhdk, sk, dsdk = hk_payload
    hk = as_array(hk, dtype=complex)
    dhdk = as_array(dhdk, dtype=complex)
    sk = None if sk is None else as_array(sk, dtype=complex)
    dsdk = None if dsdk is None else as_array(dsdk, dtype=complex)

    nk = kpoints.shape[0]
    if hk.ndim != 3 or hk.shape[0] != nk or hk.shape[1] != hk.shape[2]:
        raise ValueError("Hk must have shape (nk, norb, norb).")
    if dhdk.shape != (*hk.shape, 3):
        raise ValueError("dHdk must have shape (nk, norb, norb, 3).")
    if sk is not None and sk.shape != hk.shape:
        raise ValueError("Sk must have shape (nk, norb, norb).")
    if sk is not None and dsdk is None:
        raise ValueError("dSdk is required when overlap Sk is provided.")
    if dsdk is not None and dsdk.shape != (*hk.shape, 3):
        raise ValueError("dSdk must have shape (nk, norb, norb, 3).")
    if not np.all(np.isfinite(hk)) or not np.all(np.isfinite(dhdk)):
        raise ValueError("Hk and dHdk must contain finite values.")
    if sk is not None and (not np.all(np.isfinite(sk)) or not np.all(np.isfinite(dsdk))):
        raise ValueError("Sk and dSdk must contain finite values.")

    all_eigenvalues = np.zeros((nk, hk.shape[1]), dtype=float)
    all_velocities = np.zeros((nk, hk.shape[1], 3), dtype=float)
    for ik in range(nk):
        if sk is None:
            eigenvalues, eigenvectors = np.linalg.eigh(hk[ik])
        else:
            chol = np.linalg.cholesky(sk[ik])
            chol_inv = np.linalg.inv(chol)
            h_orth = chol_inv @ hk[ik] @ chol_inv.conj().T
            eigenvalues, eigenvectors_orth = np.linalg.eigh(h_orth)
            eigenvectors = chol_inv.conj().T @ eigenvectors_orth
        all_eigenvalues[ik] = eigenvalues.real
        for axis in range(3):
            velocity_operator = dhdk[ik, :, :, axis]
            matrix_elements = eigenvectors.conj().T @ velocity_operator @ eigenvectors
            if sk is not None:
                overlap_elements = eigenvectors.conj().T @ dsdk[ik, :, :, axis] @ eigenvectors
                matrix_elements = matrix_elements - eigenvalues[:, None] * overlap_elements
            all_velocities[ik, :, axis] = np.real(np.diag(matrix_elements))

    if not np.all(np.isfinite(all_eigenvalues)) or not np.all(np.isfinite(all_velocities)):
        raise ValueError("Hamiltonian-derivative velocities must be finite.")
    band_indices = _normalize_band_indices(bands, all_eigenvalues.shape[1])
    return all_velocities[:, band_indices, :]


def compute_serta_transport_from_epc(
    system,
    epc_data: EPCData,
    linewidth_data: LinewidthData,
    chemical_potential: float,
    temperature: float,
    kpoint_weights: Optional[np.ndarray] = None,
    spin_degeneracy: int = 1,
    volume: float = 1.0,
    velocity_delta: float = 1e-4,
    velocity_source: str = "finite_difference",
    use_scc: bool = False,
    **solver_kwargs,
) -> TransportData:
    """Compute SERTA transport using EPC linewidths and selected band-velocity provider."""
    if not isinstance(velocity_source, str):
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")
    velocity_source = velocity_source.replace("-", "_").lower()
    if velocity_source not in {"finite_difference", "hamiltonian_derivative"}:
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")
    linewidth = linewidth_data.linewidth
    if linewidth.ndim == 3:
        linewidth = linewidth.sum(axis=-1)
    if linewidth.shape != epc_data.eigenvalues_k.shape:
        raise ValueError("linewidth_data.linewidth must match EPCData eigenvalues_k shape after mode summation.")

    velocity_metadata: Dict[str, Any]
    if velocity_source == "finite_difference":
        velocity_delta = validate_finite_positive_scalar(velocity_delta, "velocity_delta")
        velocities = compute_band_velocities_finite_difference(
            system=system,
            kpoints=epc_data.kpoints,
            bands=epc_data.band_indices,
            delta=velocity_delta,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "finite_difference",
            "velocity_delta": velocity_delta,
            "velocity_convention": "central_difference_band_energy",
        }
    else:
        velocities = compute_band_velocities_hamiltonian_derivative(
            system=system,
            kpoints=epc_data.kpoints,
            bands=epc_data.band_indices,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "hamiltonian_derivative",
            "velocity_convention": "diag_Cdagger_dH_minus_EdS_C",
            "overlap_correction": "dH_minus_E_dS_diagonal",
        }
    result = compute_serta_conductivity(
        eigenvalues=epc_data.eigenvalues_k,
        velocities=velocities,
        linewidth=linewidth,
        chemical_potential=chemical_potential,
        temperature=temperature,
        kpoint_weights=kpoint_weights,
        spin_degeneracy=spin_degeneracy,
        volume=volume,
    )
    result.metadata.update(
        {
            "velocity_unit": "eV/fractional_reciprocal_coordinate",
            "band_indices": epc_data.band_indices,
            "epc_schema": epc_data.metadata.get("schema"),
            "linewidth_schema": linewidth_data.metadata.get("schema"),
        }
    )
    result.metadata.update(velocity_metadata)
    return result


def compute_serta_transport_from_epc_mesh_chunked_artifact(
    system,
    directory: Union[str, Path],
    chemical_potential: float,
    temperature: float,
    sigma: float,
    broadening: str = "gaussian",
    frequency_floor: float = 1e-5,
    spin_degeneracy: int = 1,
    volume: float = 1.0,
    velocity_delta: float = 1e-4,
    velocity_source: str = "finite_difference",
    use_scc: bool = False,
    **solver_kwargs,
) -> TransportData:
    """Compute SERTA transport from an EPC mesh chunked artifact.

    This helper uses summary-first chunked linewidth reduction and then reuses
    the existing band-velocity providers on the artifact k-points. It avoids
    materializing the full mesh coupling tensor.
    """
    linewidth_data = compute_linewidth_mesh_chunked_artifact(
        directory,
        chemical_potential=chemical_potential,
        temperature=temperature,
        sigma=sigma,
        broadening=broadening,
        mode_resolved=False,
        frequency_floor=frequency_floor,
    )
    if not isinstance(velocity_source, str):
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")
    velocity_source = velocity_source.replace("-", "_").lower()
    if velocity_source not in {"finite_difference", "hamiltonian_derivative"}:
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")

    if velocity_source == "finite_difference":
        velocity_delta = validate_finite_positive_scalar(velocity_delta, "velocity_delta")
        velocities = compute_band_velocities_finite_difference(
            system=system,
            kpoints=linewidth_data.kpoints,
            bands=linewidth_data.band_indices,
            delta=velocity_delta,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "finite_difference",
            "velocity_delta": velocity_delta,
            "velocity_convention": "central_difference_band_energy",
        }
    else:
        velocities = compute_band_velocities_hamiltonian_derivative(
            system=system,
            kpoints=linewidth_data.kpoints,
            bands=linewidth_data.band_indices,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "hamiltonian_derivative",
            "velocity_convention": "diag_Cdagger_dH_minus_EdS_C",
            "overlap_correction": "dH_minus_E_dS_diagonal",
        }

    eigenvalues = _get_system_eigenvalues(
        system,
        linewidth_data.kpoints,
        use_scc=use_scc,
        **solver_kwargs,
    )[:, linewidth_data.band_indices]
    result = compute_serta_conductivity(
        eigenvalues=eigenvalues,
        velocities=velocities,
        linewidth=linewidth_data.linewidth,
        chemical_potential=chemical_potential,
        temperature=temperature,
        kpoint_weights=linewidth_data.kpoint_weights,
        spin_degeneracy=spin_degeneracy,
        volume=volume,
    )
    result.metadata.update(
        {
            "source": "deeptb.eph.compute_serta_transport_from_epc_mesh_chunked_artifact",
            "velocity_unit": "eV/fractional_reciprocal_coordinate",
            "band_indices": linewidth_data.band_indices,
            "linewidth_schema": linewidth_data.metadata.get("schema"),
            "chunked_artifact": True,
            "artifact_axis": linewidth_data.metadata.get("artifact_axis"),
            "artifact_chunk_count": linewidth_data.metadata.get("artifact_chunk_count"),
            "summary_first": True,
        }
    )
    result.metadata.update(velocity_metadata)
    return result


def compute_serta_transport_scan_from_epc_mesh_chunked_artifact(
    system,
    directory: Union[str, Path],
    chemical_potentials: Sequence[float],
    temperatures: Sequence[float],
    sigma: float,
    broadening: str = "gaussian",
    frequency_floor: float = 1e-5,
    spin_degeneracy: int = 1,
    volume: float = 1.0,
    velocity_delta: float = 1e-4,
    velocity_source: str = "finite_difference",
    use_scc: bool = False,
    **solver_kwargs,
) -> TransportScanData:
    """Compute SERTA transport scan from an EPC mesh chunked artifact."""
    chemical_potentials = np.atleast_1d(np.asarray(chemical_potentials, dtype=float))
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if chemical_potentials.ndim != 1 or chemical_potentials.size == 0:
        raise ValueError("chemical_potentials must be a one-dimensional non-empty array.")
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("temperatures must be a one-dimensional non-empty array.")

    linewidth_data = compute_linewidth_mesh_chunked_artifact(
        directory,
        chemical_potential=float(chemical_potentials[0]),
        temperature=float(temperatures[0]),
        sigma=sigma,
        broadening=broadening,
        mode_resolved=False,
        frequency_floor=frequency_floor,
    )
    if not isinstance(velocity_source, str):
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")
    velocity_source = velocity_source.replace("-", "_").lower()
    if velocity_source not in {"finite_difference", "hamiltonian_derivative"}:
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")

    if velocity_source == "finite_difference":
        velocity_delta = validate_finite_positive_scalar(velocity_delta, "velocity_delta")
        velocities = compute_band_velocities_finite_difference(
            system=system,
            kpoints=linewidth_data.kpoints,
            bands=linewidth_data.band_indices,
            delta=velocity_delta,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "finite_difference",
            "velocity_delta": velocity_delta,
            "velocity_convention": "central_difference_band_energy",
        }
    else:
        velocities = compute_band_velocities_hamiltonian_derivative(
            system=system,
            kpoints=linewidth_data.kpoints,
            bands=linewidth_data.band_indices,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "hamiltonian_derivative",
            "velocity_convention": "diag_Cdagger_dH_minus_EdS_C",
            "overlap_correction": "dH_minus_E_dS_diagonal",
        }

    eigenvalues = _get_system_eigenvalues(
        system,
        linewidth_data.kpoints,
        use_scc=use_scc,
        **solver_kwargs,
    )[:, linewidth_data.band_indices]
    result = compute_serta_transport_scan(
        eigenvalues=eigenvalues,
        velocities=velocities,
        linewidth=linewidth_data.linewidth,
        chemical_potentials=chemical_potentials,
        temperatures=temperatures,
        kpoint_weights=linewidth_data.kpoint_weights,
        spin_degeneracy=spin_degeneracy,
        volume=volume,
    )
    result.metadata.update(
        {
            "source": "deeptb.eph.compute_serta_transport_scan_from_epc_mesh_chunked_artifact",
            "velocity_unit": "eV/fractional_reciprocal_coordinate",
            "band_indices": linewidth_data.band_indices,
            "linewidth_schema": linewidth_data.metadata.get("schema"),
            "chunked_artifact": True,
            "artifact_axis": linewidth_data.metadata.get("artifact_axis"),
            "artifact_chunk_count": linewidth_data.metadata.get("artifact_chunk_count"),
            "summary_first": True,
            "linewidth_reference_chemical_potential": float(chemical_potentials[0]),
            "linewidth_reference_temperature": float(temperatures[0]),
        }
    )
    result.metadata.update(velocity_metadata)
    return result


def compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact(
    system,
    directory: Union[str, Path],
    chemical_potentials: Sequence[float],
    temperatures: Sequence[float],
    sigma: float,
    broadening: str = "gaussian",
    frequency_floor: float = 1e-5,
    spin_degeneracy: int = 1,
    volume: float = 1.0,
    velocity_delta: float = 1e-4,
    velocity_source: str = "finite_difference",
    use_scc: bool = False,
    **solver_kwargs,
) -> TransportScanData:
    """Compute artifact SERTA scan with linewidth recomputed at each scan point.

    This is intentionally separate from
    :func:`compute_serta_transport_scan_from_epc_mesh_chunked_artifact`, whose
    public meaning is the fixed-linewidth scan convention.
    """
    chemical_potentials = np.atleast_1d(np.asarray(chemical_potentials, dtype=float))
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if chemical_potentials.ndim != 1 or chemical_potentials.size == 0:
        raise ValueError("chemical_potentials must be a one-dimensional non-empty array.")
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("temperatures must be a one-dimensional non-empty array.")
    if not np.all(np.isfinite(chemical_potentials)):
        raise ValueError("chemical_potentials must contain finite values.")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
        raise ValueError("temperatures must contain finite positive values.")
    if not isinstance(velocity_source, str):
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")
    velocity_source = velocity_source.replace("-", "_").lower()
    if velocity_source not in {"finite_difference", "hamiltonian_derivative"}:
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")

    first_linewidth_data = compute_linewidth_mesh_chunked_artifact(
        directory,
        chemical_potential=float(chemical_potentials[0]),
        temperature=float(temperatures[0]),
        sigma=sigma,
        broadening=broadening,
        mode_resolved=False,
        frequency_floor=frequency_floor,
    )

    if velocity_source == "finite_difference":
        velocity_delta = validate_finite_positive_scalar(velocity_delta, "velocity_delta")
        velocities = compute_band_velocities_finite_difference(
            system=system,
            kpoints=first_linewidth_data.kpoints,
            bands=first_linewidth_data.band_indices,
            delta=velocity_delta,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "finite_difference",
            "velocity_delta": velocity_delta,
            "velocity_convention": "central_difference_band_energy",
        }
    else:
        velocities = compute_band_velocities_hamiltonian_derivative(
            system=system,
            kpoints=first_linewidth_data.kpoints,
            bands=first_linewidth_data.band_indices,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "hamiltonian_derivative",
            "velocity_convention": "diag_Cdagger_dH_minus_EdS_C",
            "overlap_correction": "dH_minus_E_dS_diagonal",
        }

    eigenvalues = _get_system_eigenvalues(
        system,
        first_linewidth_data.kpoints,
        use_scc=use_scc,
        **solver_kwargs,
    )[:, first_linewidth_data.band_indices]

    nmu = chemical_potentials.shape[0]
    ntemperatures = temperatures.shape[0]
    conductivity = np.zeros((nmu, ntemperatures, 3, 3), dtype=float)
    carrier_density = np.zeros((nmu, ntemperatures), dtype=float)
    metadata = None
    for imu, chemical_potential in enumerate(chemical_potentials):
        for itemperature, temperature in enumerate(temperatures):
            if imu == 0 and itemperature == 0:
                linewidth_data = first_linewidth_data
            else:
                linewidth_data = compute_linewidth_mesh_chunked_artifact(
                    directory,
                    chemical_potential=float(chemical_potential),
                    temperature=float(temperature),
                    sigma=sigma,
                    broadening=broadening,
                    mode_resolved=False,
                    frequency_floor=frequency_floor,
                )
                if not np.array_equal(linewidth_data.band_indices, first_linewidth_data.band_indices):
                    raise ValueError("chunked artifact linewidth band indices changed during scan.")
                if not np.allclose(linewidth_data.kpoints, first_linewidth_data.kpoints):
                    raise ValueError("chunked artifact kpoints changed during scan.")
                if not np.allclose(linewidth_data.kpoint_weights, first_linewidth_data.kpoint_weights):
                    raise ValueError("chunked artifact kpoint weights changed during scan.")
            point = compute_serta_conductivity(
                eigenvalues=eigenvalues,
                velocities=velocities,
                linewidth=linewidth_data.linewidth,
                chemical_potential=float(chemical_potential),
                temperature=float(temperature),
                kpoint_weights=linewidth_data.kpoint_weights,
                spin_degeneracy=spin_degeneracy,
                volume=volume,
            )
            conductivity[imu, itemperature] = point.conductivity
            carrier_density[imu, itemperature] = point.carrier_density
            if metadata is None:
                metadata = dict(point.metadata)

    scan_metadata = dict(metadata or {})
    scan_metadata.pop("schema", None)
    scan_metadata.pop("schema_version", None)
    scan_metadata.pop("chemical_potential", None)
    scan_metadata.pop("temperature", None)
    scan_metadata.update(
        {
            "chemical_potential_count": int(nmu),
            "temperature_count": int(ntemperatures),
            "scan_axes": ["chemical_potential", "temperature"],
            "linewidth_scan_convention": "per_scan_point_recomputed",
            "source": "deeptb.eph.compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact",
            "velocity_unit": "eV/fractional_reciprocal_coordinate",
            "band_indices": first_linewidth_data.band_indices,
            "linewidth_schema": first_linewidth_data.metadata.get("schema"),
            "chunked_artifact": True,
            "artifact_axis": first_linewidth_data.metadata.get("artifact_axis"),
            "artifact_chunk_count": first_linewidth_data.metadata.get("artifact_chunk_count"),
            "summary_first": True,
        }
    )
    scan_metadata.update(velocity_metadata)
    return TransportScanData(
        conductivity=conductivity,
        carrier_density=carrier_density,
        chemical_potentials=chemical_potentials,
        temperatures=temperatures,
        metadata=scan_metadata,
    )


def compute_serta_mobility_si_from_epc_mesh_chunked_artifact(
    system,
    directory: Union[str, Path],
    reciprocal_cell: np.ndarray,
    chemical_potential: float,
    temperature: float,
    sigma: float,
    broadening: str = "gaussian",
    frequency_floor: float = 1e-5,
    spin_degeneracy: int = 1,
    dimension: str = "3d",
    volume: Optional[float] = None,
    area: Optional[float] = None,
    velocity_delta: float = 1e-4,
    velocity_source: str = "finite_difference",
    use_scc: bool = False,
    **solver_kwargs,
) -> MobilityData:
    """Compute SI SERTA mobility from an EPC mesh chunked artifact."""
    linewidth_data = compute_linewidth_mesh_chunked_artifact(
        directory,
        chemical_potential=chemical_potential,
        temperature=temperature,
        sigma=sigma,
        broadening=broadening,
        mode_resolved=False,
        frequency_floor=frequency_floor,
    )
    if not isinstance(velocity_source, str):
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")
    velocity_source = velocity_source.replace("-", "_").lower()
    if velocity_source not in {"finite_difference", "hamiltonian_derivative"}:
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")

    if velocity_source == "finite_difference":
        velocity_delta = validate_finite_positive_scalar(velocity_delta, "velocity_delta")
        velocities = compute_band_velocities_finite_difference(
            system=system,
            kpoints=linewidth_data.kpoints,
            bands=linewidth_data.band_indices,
            delta=velocity_delta,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "finite_difference",
            "velocity_delta": velocity_delta,
            "velocity_convention": "central_difference_band_energy",
        }
    else:
        velocities = compute_band_velocities_hamiltonian_derivative(
            system=system,
            kpoints=linewidth_data.kpoints,
            bands=linewidth_data.band_indices,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "hamiltonian_derivative",
            "velocity_convention": "diag_Cdagger_dH_minus_EdS_C",
            "overlap_correction": "dH_minus_E_dS_diagonal",
        }

    eigenvalues = _get_system_eigenvalues(
        system,
        linewidth_data.kpoints,
        use_scc=use_scc,
        **solver_kwargs,
    )[:, linewidth_data.band_indices]
    result = compute_serta_mobility_si(
        eigenvalues=eigenvalues,
        velocities=velocities,
        linewidth=linewidth_data.linewidth,
        reciprocal_cell=reciprocal_cell,
        chemical_potential=chemical_potential,
        temperature=temperature,
        kpoint_weights=linewidth_data.kpoint_weights,
        spin_degeneracy=spin_degeneracy,
        dimension=dimension,
        volume=volume,
        area=area,
    )
    result.metadata.update(
        {
            "source": "deeptb.eph.compute_serta_mobility_si_from_epc_mesh_chunked_artifact",
            "band_indices": linewidth_data.band_indices,
            "linewidth_schema": linewidth_data.metadata.get("schema"),
            "chunked_artifact": True,
            "artifact_axis": linewidth_data.metadata.get("artifact_axis"),
            "artifact_chunk_count": linewidth_data.metadata.get("artifact_chunk_count"),
            "summary_first": True,
        }
    )
    result.metadata.update(velocity_metadata)
    return result


def compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact(
    system,
    directory: Union[str, Path],
    reciprocal_cell: np.ndarray,
    chemical_potentials: Sequence[float],
    temperatures: Sequence[float],
    sigma: float,
    broadening: str = "gaussian",
    frequency_floor: float = 1e-5,
    spin_degeneracy: int = 1,
    dimension: str = "3d",
    volume: Optional[float] = None,
    area: Optional[float] = None,
    velocity_delta: float = 1e-4,
    velocity_source: str = "finite_difference",
    use_scc: bool = False,
    **solver_kwargs,
) -> MobilityScanData:
    """Compute SI SERTA mobility scan from an EPC mesh chunked artifact."""
    chemical_potentials = np.atleast_1d(np.asarray(chemical_potentials, dtype=float))
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if chemical_potentials.ndim != 1 or chemical_potentials.size == 0:
        raise ValueError("chemical_potentials must be a one-dimensional non-empty array.")
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("temperatures must be a one-dimensional non-empty array.")

    linewidth_data = compute_linewidth_mesh_chunked_artifact(
        directory,
        chemical_potential=float(chemical_potentials[0]),
        temperature=float(temperatures[0]),
        sigma=sigma,
        broadening=broadening,
        mode_resolved=False,
        frequency_floor=frequency_floor,
    )
    if not isinstance(velocity_source, str):
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")
    velocity_source = velocity_source.replace("-", "_").lower()
    if velocity_source not in {"finite_difference", "hamiltonian_derivative"}:
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")

    if velocity_source == "finite_difference":
        velocity_delta = validate_finite_positive_scalar(velocity_delta, "velocity_delta")
        velocities = compute_band_velocities_finite_difference(
            system=system,
            kpoints=linewidth_data.kpoints,
            bands=linewidth_data.band_indices,
            delta=velocity_delta,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "finite_difference",
            "velocity_delta": velocity_delta,
            "velocity_convention": "central_difference_band_energy",
        }
    else:
        velocities = compute_band_velocities_hamiltonian_derivative(
            system=system,
            kpoints=linewidth_data.kpoints,
            bands=linewidth_data.band_indices,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "hamiltonian_derivative",
            "velocity_convention": "diag_Cdagger_dH_minus_EdS_C",
            "overlap_correction": "dH_minus_E_dS_diagonal",
        }

    eigenvalues = _get_system_eigenvalues(
        system,
        linewidth_data.kpoints,
        use_scc=use_scc,
        **solver_kwargs,
    )[:, linewidth_data.band_indices]
    result = compute_serta_mobility_scan_si(
        eigenvalues=eigenvalues,
        velocities=velocities,
        linewidth=linewidth_data.linewidth,
        reciprocal_cell=reciprocal_cell,
        chemical_potentials=chemical_potentials,
        temperatures=temperatures,
        kpoint_weights=linewidth_data.kpoint_weights,
        spin_degeneracy=spin_degeneracy,
        dimension=dimension,
        volume=volume,
        area=area,
    )
    result.metadata.update(
        {
            "source": "deeptb.eph.compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact",
            "band_indices": linewidth_data.band_indices,
            "linewidth_schema": linewidth_data.metadata.get("schema"),
            "chunked_artifact": True,
            "artifact_axis": linewidth_data.metadata.get("artifact_axis"),
            "artifact_chunk_count": linewidth_data.metadata.get("artifact_chunk_count"),
            "summary_first": True,
            "linewidth_reference_chemical_potential": float(chemical_potentials[0]),
            "linewidth_reference_temperature": float(temperatures[0]),
        }
    )
    result.metadata.update(velocity_metadata)
    return result


def compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact(
    system,
    directory: Union[str, Path],
    reciprocal_cell: np.ndarray,
    chemical_potentials: Sequence[float],
    temperatures: Sequence[float],
    sigma: float,
    broadening: str = "gaussian",
    frequency_floor: float = 1e-5,
    spin_degeneracy: int = 1,
    dimension: str = "3d",
    volume: Optional[float] = None,
    area: Optional[float] = None,
    velocity_delta: float = 1e-4,
    velocity_source: str = "finite_difference",
    use_scc: bool = False,
    **solver_kwargs,
) -> MobilityScanData:
    """Compute SI mobility scan with linewidth recomputed at each scan point."""
    chemical_potentials = np.atleast_1d(np.asarray(chemical_potentials, dtype=float))
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if chemical_potentials.ndim != 1 or chemical_potentials.size == 0:
        raise ValueError("chemical_potentials must be a one-dimensional non-empty array.")
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("temperatures must be a one-dimensional non-empty array.")
    if not np.all(np.isfinite(chemical_potentials)):
        raise ValueError("chemical_potentials must contain finite values.")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
        raise ValueError("temperatures must contain finite positive values.")
    if not isinstance(velocity_source, str):
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")
    velocity_source = velocity_source.replace("-", "_").lower()
    if velocity_source not in {"finite_difference", "hamiltonian_derivative"}:
        raise ValueError("velocity_source must be 'finite_difference' or 'hamiltonian_derivative'.")

    first_linewidth_data = compute_linewidth_mesh_chunked_artifact(
        directory,
        chemical_potential=float(chemical_potentials[0]),
        temperature=float(temperatures[0]),
        sigma=sigma,
        broadening=broadening,
        mode_resolved=False,
        frequency_floor=frequency_floor,
    )
    if velocity_source == "finite_difference":
        velocity_delta = validate_finite_positive_scalar(velocity_delta, "velocity_delta")
        velocities = compute_band_velocities_finite_difference(
            system=system,
            kpoints=first_linewidth_data.kpoints,
            bands=first_linewidth_data.band_indices,
            delta=velocity_delta,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "finite_difference",
            "velocity_delta": velocity_delta,
            "velocity_convention": "central_difference_band_energy",
        }
    else:
        velocities = compute_band_velocities_hamiltonian_derivative(
            system=system,
            kpoints=first_linewidth_data.kpoints,
            bands=first_linewidth_data.band_indices,
            use_scc=use_scc,
            **solver_kwargs,
        )
        velocity_metadata = {
            "velocity_source": "hamiltonian_derivative",
            "velocity_convention": "diag_Cdagger_dH_minus_EdS_C",
            "overlap_correction": "dH_minus_E_dS_diagonal",
        }

    eigenvalues = _get_system_eigenvalues(
        system,
        first_linewidth_data.kpoints,
        use_scc=use_scc,
        **solver_kwargs,
    )[:, first_linewidth_data.band_indices]

    nmu = chemical_potentials.shape[0]
    ntemperatures = temperatures.shape[0]
    conductivity = np.zeros((nmu, ntemperatures, 3, 3), dtype=float)
    mobility = np.zeros((nmu, ntemperatures, 3, 3), dtype=float)
    carrier_density = np.zeros((nmu, ntemperatures), dtype=float)
    metadata = None
    for imu, chemical_potential in enumerate(chemical_potentials):
        for itemperature, temperature in enumerate(temperatures):
            if imu == 0 and itemperature == 0:
                linewidth_data = first_linewidth_data
            else:
                linewidth_data = compute_linewidth_mesh_chunked_artifact(
                    directory,
                    chemical_potential=float(chemical_potential),
                    temperature=float(temperature),
                    sigma=sigma,
                    broadening=broadening,
                    mode_resolved=False,
                    frequency_floor=frequency_floor,
                )
                if not np.array_equal(linewidth_data.band_indices, first_linewidth_data.band_indices):
                    raise ValueError("chunked artifact linewidth band indices changed during scan.")
                if not np.allclose(linewidth_data.kpoints, first_linewidth_data.kpoints):
                    raise ValueError("chunked artifact kpoints changed during scan.")
                if not np.allclose(linewidth_data.kpoint_weights, first_linewidth_data.kpoint_weights):
                    raise ValueError("chunked artifact kpoint weights changed during scan.")
            point = compute_serta_mobility_si(
                eigenvalues=eigenvalues,
                velocities=velocities,
                linewidth=linewidth_data.linewidth,
                reciprocal_cell=reciprocal_cell,
                chemical_potential=float(chemical_potential),
                temperature=float(temperature),
                kpoint_weights=linewidth_data.kpoint_weights,
                spin_degeneracy=spin_degeneracy,
                dimension=dimension,
                volume=volume,
                area=area,
            )
            conductivity[imu, itemperature] = point.conductivity
            mobility[imu, itemperature] = point.mobility
            carrier_density[imu, itemperature] = point.carrier_density
            if metadata is None:
                metadata = dict(point.metadata)

    scan_metadata = dict(metadata or {})
    scan_metadata.pop("schema", None)
    scan_metadata.pop("schema_version", None)
    scan_metadata.pop("chemical_potential", None)
    scan_metadata.pop("temperature", None)
    scan_metadata.update(
        {
            "chemical_potential_count": int(nmu),
            "temperature_count": int(ntemperatures),
            "scan_axes": ["chemical_potential", "temperature"],
            "linewidth_scan_convention": "per_scan_point_recomputed",
            "source": "deeptb.eph.compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact",
            "band_indices": first_linewidth_data.band_indices,
            "linewidth_schema": first_linewidth_data.metadata.get("schema"),
            "chunked_artifact": True,
            "artifact_axis": first_linewidth_data.metadata.get("artifact_axis"),
            "artifact_chunk_count": first_linewidth_data.metadata.get("artifact_chunk_count"),
            "summary_first": True,
        }
    )
    scan_metadata.update(velocity_metadata)
    return MobilityScanData(
        conductivity=conductivity,
        mobility=mobility,
        carrier_density=carrier_density,
        chemical_potentials=chemical_potentials,
        temperatures=temperatures,
        metadata=scan_metadata,
    )


def find_degenerate_band_groups(eigenvalues: np.ndarray, tolerance: float = 1e-5) -> list:
    """Group adjacent bands whose energies are degenerate within ``tolerance``.

    This helper preserves band order and is intended for postprocess diagnostics.
    It does not reorder bands or fix eigenvector gauges.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a one-dimensional band-energy array.")
    tolerance = validate_finite_nonnegative_scalar(tolerance, "tolerance")
    if eigenvalues.size == 0:
        return []

    groups = []
    start = 0
    for iband in range(1, eigenvalues.shape[0]):
        if abs(eigenvalues[iband] - eigenvalues[iband - 1]) > tolerance:
            groups.append(np.arange(start, iband, dtype=int))
            start = iband
    groups.append(np.arange(start, eigenvalues.shape[0], dtype=int))
    return groups


def compute_coupling_strength_summary(
    epc_data: Union[EPCData, EPCMeshData, EPCPathData],
    weighted: bool = True,
) -> Dict[str, Any]:
    """Summarize EPC coupling strength over q/k/mode/band axes.

    The input is an existing DeePTB EPC data object; this helper does not
    recompute coupling and does not introduce a persistent schema. For
    ``EPCMeshData``, ``weighted=True`` applies normalized q-point and k-point
    weights before aggregation. For ``EPCData`` and ``EPCPathData``, no weights
    are available, so aggregation is an unweighted sum.
    """
    weighted = _validate_bool(weighted, "weighted")
    if isinstance(epc_data, EPCMeshData):
        source = "EPCMeshData.coupling_strength"
        strength = np.asarray(epc_data.coupling_strength, dtype=float)
        if weighted:
            qpoint_weights = _normalize_weights(epc_data.qpoint_weights, "qpoint_weights")
            kpoint_weights = _normalize_weights(epc_data.kpoint_weights, "kpoint_weights")
            strength = strength * qpoint_weights[:, None, None, None, None] * kpoint_weights[None, :, None, None, None]
            weight_convention = "normalized_qpoint_and_kpoint_weights"
        else:
            weight_convention = "unweighted_sum"
    elif isinstance(epc_data, EPCPathData):
        source = "EPCPathData.coupling_strength"
        strength = np.asarray(epc_data.coupling_strength, dtype=float)
        weight_convention = "unweighted_sum"
    elif isinstance(epc_data, EPCData):
        source = "EPCData.coupling_strength"
        strength = np.asarray(epc_data.coupling_strength, dtype=float)
        weight_convention = "unweighted_sum"
    else:
        raise ValueError("epc_data must be an EPCData, EPCMeshData, or EPCPathData object.")

    if strength.ndim != 5:
        raise ValueError("coupling_strength must have shape (nq, nk, nmodes, nfinal, ninitial).")
    if strength.size == 0:
        raise ValueError("coupling_strength must be non-empty.")
    if not np.all(np.isfinite(strength)) or np.any(strength < 0.0):
        raise ValueError("coupling_strength must contain finite non-negative values.")

    metadata = {
        "source": source,
        "weighted": weighted,
        "weight_convention": weight_convention,
        "input_schema": epc_data.metadata.get("schema"),
        "coupling_strength_unit": epc_data.metadata.get("coupling_strength_unit", "eV^2"),
    }
    if isinstance(epc_data, EPCPathData):
        metadata["path_axis"] = epc_data.path_axis
    if isinstance(epc_data, EPCMeshData):
        metadata["mesh_spec"] = epc_data.metadata.get("mesh_spec")

    return {
        "total": np.asarray(strength.sum()),
        "q_resolved": strength.sum(axis=(1, 2, 3, 4)),
        "k_resolved": strength.sum(axis=(0, 2, 3, 4)),
        "mode_resolved": strength.sum(axis=(0, 1, 3, 4)),
        "final_band_resolved": strength.sum(axis=(0, 1, 2, 4)),
        "initial_band_resolved": strength.sum(axis=(0, 1, 2, 3)),
        "band_pair_resolved": strength.sum(axis=(0, 1, 2)),
        "band_indices": epc_data.band_indices.copy(),
        "metadata": metadata,
    }


def compute_scattering_maps(
    epc_data: Union[EPCData, EPCMeshData, EPCPathData],
    weighted: bool = True,
) -> Dict[str, Any]:
    """Compute q/k/band-resolved EPC scattering proxy maps.

    The maps are aggregated from existing ``coupling_strength`` arrays. They
    are diagnostics for locating strong EPC channels, not a full linewidth or
    energy-conserving scattering-rate calculation.
    """
    weighted = _validate_bool(weighted, "weighted")
    if isinstance(epc_data, EPCMeshData):
        source = "EPCMeshData.coupling_strength"
        strength = np.asarray(epc_data.coupling_strength, dtype=float)
        if weighted:
            qpoint_weights = _normalize_weights(epc_data.qpoint_weights, "qpoint_weights")
            kpoint_weights = _normalize_weights(epc_data.kpoint_weights, "kpoint_weights")
            strength = strength * qpoint_weights[:, None, None, None, None] * kpoint_weights[None, :, None, None, None]
            weight_convention = "normalized_qpoint_and_kpoint_weights"
        else:
            weight_convention = "unweighted_sum"
    elif isinstance(epc_data, EPCPathData):
        source = "EPCPathData.coupling_strength"
        strength = np.asarray(epc_data.coupling_strength, dtype=float)
        weight_convention = "unweighted_sum"
    elif isinstance(epc_data, EPCData):
        source = "EPCData.coupling_strength"
        strength = np.asarray(epc_data.coupling_strength, dtype=float)
        weight_convention = "unweighted_sum"
    else:
        raise ValueError("epc_data must be an EPCData, EPCMeshData, or EPCPathData object.")

    if strength.ndim != 5:
        raise ValueError("coupling_strength must have shape (nq, nk, nmodes, nfinal, ninitial).")
    if strength.size == 0:
        raise ValueError("coupling_strength must be non-empty.")
    if not np.all(np.isfinite(strength)) or np.any(strength < 0.0):
        raise ValueError("coupling_strength must contain finite non-negative values.")

    metadata = {
        "source": source,
        "weighted": weighted,
        "weight_convention": weight_convention,
        "input_schema": epc_data.metadata.get("schema"),
        "coupling_strength_unit": epc_data.metadata.get("coupling_strength_unit", "eV^2"),
        "convention": "coupling_strength_scattering_proxy",
    }
    if isinstance(epc_data, EPCPathData):
        metadata["path_axis"] = epc_data.path_axis
    if isinstance(epc_data, EPCMeshData):
        metadata["mesh_spec"] = epc_data.metadata.get("mesh_spec")

    return {
        "qpoints": epc_data.qpoints.copy(),
        "kpoints": epc_data.kpoints.copy(),
        "frequencies": epc_data.frequencies.copy(),
        "band_indices": epc_data.band_indices.copy(),
        "q_mode_resolved": strength.sum(axis=(1, 3, 4)),
        "k_mode_resolved": strength.sum(axis=(0, 3, 4)),
        "q_final_band_resolved": strength.sum(axis=(1, 2, 4)),
        "q_initial_band_resolved": strength.sum(axis=(1, 2, 3)),
        "q_band_pair_resolved": strength.sum(axis=(1, 2)),
        "k_final_band_resolved": strength.sum(axis=(0, 2, 4)),
        "k_initial_band_resolved": strength.sum(axis=(0, 2, 3)),
        "k_band_pair_resolved": strength.sum(axis=(0, 2)),
        "metadata": metadata,
    }


def compute_phonon_dos(
    phonons: Phonons,
    frequency_grid: np.ndarray,
    sigma: float,
    broadening: str = "gaussian",
    qpoint_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute phonon DOS from externally supplied phonon frequencies.

    Frequencies and ``sigma`` are in THz. DeePTB does not compute phonons here;
    this helper only broadens the frequencies already stored in ``Phonons``.
    The returned DOS has unit ``THz^-1``.
    """
    if not isinstance(phonons, Phonons):
        raise ValueError("phonons must be a Phonons object.")
    frequency_grid = np.asarray(frequency_grid, dtype=float)
    sigma = validate_finite_positive_scalar(sigma, "sigma")
    if frequency_grid.ndim != 1 or frequency_grid.size == 0:
        raise ValueError("frequency_grid must be a one-dimensional non-empty array.")
    if not np.all(np.isfinite(frequency_grid)):
        raise ValueError("frequency_grid must contain finite values.")
    if np.any(phonons.frequencies < 0.0):
        raise ValueError("phonon DOS does not support negative frequencies.")
    if not isinstance(broadening, str):
        raise ValueError("broadening must be either 'gaussian' or 'lorentzian'.")
    broadening = broadening.lower()
    if broadening not in {"gaussian", "lorentzian"}:
        raise ValueError("broadening must be either 'gaussian' or 'lorentzian'.")

    nq, nmodes = phonons.frequencies.shape
    if qpoint_weights is None:
        weights = np.full((nq,), 1.0 / nq, dtype=float)
        weight_convention = "uniform_normalized_qpoint_weights"
    else:
        weights = _normalize_weights(qpoint_weights, "qpoint_weights")
        if weights.shape != (nq,):
            raise ValueError("qpoint_weights must match phonons.qpoints.")
        weight_convention = "normalized_qpoint_weights"

    mode_resolved_dos = np.zeros((nmodes, frequency_grid.shape[0]), dtype=float)
    for iq in range(nq):
        for imode in range(nmodes):
            mode_resolved_dos[imode] += weights[iq] * _broadening(
                frequency_grid - phonons.frequencies[iq, imode],
                sigma,
                broadening,
            )

    return {
        "frequency_grid": frequency_grid,
        "dos": mode_resolved_dos.sum(axis=0),
        "mode_resolved_dos": mode_resolved_dos,
        "metadata": {
            "source": "Phonons.frequencies",
            "input_schema": phonons.metadata.get("schema"),
            "frequency_unit": phonons.metadata.get("frequency_unit", "THz"),
            "dos_unit": "THz^-1",
            "sigma": sigma,
            "sigma_unit": "THz",
            "broadening": broadening,
            "weight_convention": weight_convention,
        },
    }


def compute_eliashberg_spectral_function(
    epc_data: Union[EPCData, EPCMeshData, EPCPathData],
    frequency_grid: np.ndarray,
    sigma: float,
    broadening: str = "gaussian",
    weighted: bool = True,
) -> Dict[str, Any]:
    """Compute an Eliashberg-like EPC spectral function from existing EPC data.

    This helper broadens the externally supplied phonon frequencies and weights
    them by ``coupling_strength``. It is a DeePTB-native diagnostic, not a claim
    of full material-specific Eliashberg theory.
    """
    weighted = _validate_bool(weighted, "weighted")
    frequency_grid = np.asarray(frequency_grid, dtype=float)
    sigma = validate_finite_positive_scalar(sigma, "sigma")
    if frequency_grid.ndim != 1 or frequency_grid.size == 0:
        raise ValueError("frequency_grid must be a one-dimensional non-empty array.")
    if not np.all(np.isfinite(frequency_grid)):
        raise ValueError("frequency_grid must contain finite values.")
    if not isinstance(broadening, str):
        raise ValueError("broadening must be either 'gaussian' or 'lorentzian'.")
    broadening = broadening.lower()
    if broadening not in {"gaussian", "lorentzian"}:
        raise ValueError("broadening must be either 'gaussian' or 'lorentzian'.")

    if isinstance(epc_data, EPCMeshData):
        source = "EPCMeshData.coupling_strength"
        frequencies = np.asarray(epc_data.frequencies, dtype=float)
        strength = np.asarray(epc_data.coupling_strength, dtype=float)
        if weighted:
            qpoint_weights = _normalize_weights(epc_data.qpoint_weights, "qpoint_weights")
            kpoint_weights = _normalize_weights(epc_data.kpoint_weights, "kpoint_weights")
            strength = strength * qpoint_weights[:, None, None, None, None] * kpoint_weights[None, :, None, None, None]
            weight_convention = "normalized_qpoint_and_kpoint_weights"
        else:
            weight_convention = "unweighted_sum"
    elif isinstance(epc_data, EPCPathData):
        source = "EPCPathData.coupling_strength"
        frequencies = np.asarray(epc_data.frequencies, dtype=float)
        strength = np.asarray(epc_data.coupling_strength, dtype=float)
        weight_convention = "unweighted_sum"
    elif isinstance(epc_data, EPCData):
        source = "EPCData.coupling_strength"
        frequencies = np.asarray(epc_data.frequencies, dtype=float)
        strength = np.asarray(epc_data.coupling_strength, dtype=float)
        weight_convention = "unweighted_sum"
    else:
        raise ValueError("epc_data must be an EPCData, EPCMeshData, or EPCPathData object.")

    if frequencies.ndim != 2:
        raise ValueError("frequencies must have shape (nq, nmodes).")
    if strength.ndim != 5:
        raise ValueError("coupling_strength must have shape (nq, nk, nmodes, nfinal, ninitial).")
    if strength.shape[0] != frequencies.shape[0] or strength.shape[2] != frequencies.shape[1]:
        raise ValueError("coupling_strength q/mode axes must match frequencies.")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        raise ValueError("frequencies must contain finite non-negative values.")
    if not np.all(np.isfinite(strength)) or np.any(strength < 0.0):
        raise ValueError("coupling_strength must contain finite non-negative values.")

    nmodes = frequencies.shape[1]
    mode_resolved = np.zeros((nmodes, frequency_grid.shape[0]), dtype=float)
    for iq in range(frequencies.shape[0]):
        for imode in range(nmodes):
            mode_strength = strength[iq, :, imode].sum()
            mode_resolved[imode] += mode_strength * _broadening(
                frequency_grid - frequencies[iq, imode],
                sigma,
                broadening,
            )

    return {
        "frequency_grid": frequency_grid,
        "alpha2f": mode_resolved.sum(axis=0),
        "mode_resolved_alpha2f": mode_resolved,
        "metadata": {
            "source": source,
            "input_schema": epc_data.metadata.get("schema"),
            "frequency_unit": epc_data.metadata.get("frequency_unit", "THz"),
            "spectral_unit": f"{epc_data.metadata.get('coupling_strength_unit', 'eV^2')} THz^-1",
            "sigma": sigma,
            "sigma_unit": "THz",
            "broadening": broadening,
            "weighted": weighted,
            "weight_convention": weight_convention,
            "convention": "coupling_strength_weighted_phonon_frequency_spectrum",
        },
    }


def compute_subspace_coupling_strength(
    coupling_matrix: np.ndarray,
    final_groups: Sequence[Sequence[int]],
    initial_groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """Aggregate EPC strength between band subspaces.

    The aggregation is a Frobenius norm squared over each final/initial band
    block. It is invariant under unitary rotations inside either subspace, which
    makes it useful for degenerate-band diagnostics.
    """
    coupling_matrix = np.asarray(coupling_matrix, dtype=complex)
    if coupling_matrix.ndim < 2:
        raise ValueError("coupling_matrix must have at least final and initial band axes.")
    if not np.all(np.isfinite(coupling_matrix)):
        raise ValueError("coupling_matrix must contain finite values.")
    nfinal, ninitial = coupling_matrix.shape[-2:]
    normalized_final = _normalize_groups(final_groups, nfinal, "final_groups")
    normalized_initial = _normalize_groups(initial_groups, ninitial, "initial_groups")

    out = np.zeros((*coupling_matrix.shape[:-2], len(normalized_final), len(normalized_initial)), dtype=float)
    for ifinal, final_group in enumerate(normalized_final):
        for iinitial, initial_group in enumerate(normalized_initial):
            block = coupling_matrix[..., final_group[:, None], initial_group]
            out[..., ifinal, iinitial] = np.sum(np.abs(block) ** 2, axis=(-2, -1))
    return out


def compute_subspace_coupling_data(
    epc_data: EPCData,
    final_groups: Sequence[Sequence[int]],
    initial_groups: Optional[Sequence[Sequence[int]]] = None,
) -> SubspaceCouplingData:
    """Aggregate ``EPCData.coupling_matrix`` over contiguous band subspaces."""
    if initial_groups is None:
        initial_groups = final_groups
    final_bounds = _groups_to_bounds(final_groups, epc_data.coupling_matrix.shape[-2], "final_groups")
    initial_bounds = _groups_to_bounds(initial_groups, epc_data.coupling_matrix.shape[-1], "initial_groups")
    strength = compute_subspace_coupling_strength(epc_data.coupling_matrix, final_groups, initial_groups)
    return SubspaceCouplingData(
        strength=strength,
        final_group_bounds=final_bounds,
        initial_group_bounds=initial_bounds,
        metadata={
            "epc_schema": epc_data.metadata.get("schema"),
            "source": "EPCData.coupling_matrix",
        },
    )


def _broadening(x: float, sigma: float, broadening: str) -> float:
    if broadening == "gaussian":
        return np.exp(-(x**2) / (2.0 * sigma**2)) / (np.sqrt(2.0 * np.pi) * sigma)
    return (sigma / 2.0) / (x**2 + (sigma / 2.0) ** 2) / np.pi


def _validate_bool(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean.")
    return bool(value)


def _bose(x: float) -> float:
    if x > 700.0:
        return 0.0
    return 1.0 / np.expm1(x)


def _fermi(x: float) -> float:
    if x > 700.0:
        return 0.0
    if x < -700.0:
        return 1.0
    return 1.0 / (np.exp(x) + 1.0)


def _dfermi_deps(x: float) -> float:
    if x > 700.0 or x < -700.0:
        return 0.0
    exp_x = np.exp(x)
    return -exp_x / (exp_x + 1.0) ** 2


def _get_system_eigenvalues(system, kpoints: np.ndarray, use_scc: bool = False, **solver_kwargs) -> np.ndarray:
    _, eigenvalues = system.get_eigenvalues(k_points=kpoints, use_scc=use_scc, **solver_kwargs)
    eigenvalues = as_array(eigenvalues, dtype=float)
    if eigenvalues.ndim != 2:
        raise ValueError("system.get_eigenvalues must return eigenvalues with shape (nk, nbands).")
    if eigenvalues.shape[0] != kpoints.shape[0]:
        raise ValueError("system.get_eigenvalues returned an eigenvalue count inconsistent with kpoints.")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("system.get_eigenvalues must return finite eigenvalues.")
    return eigenvalues


def _normalize_band_indices(bands: Optional[Sequence[int]], nbands: int) -> np.ndarray:
    if nbands <= 0:
        raise ValueError("system.get_eigenvalues must return at least one band.")
    if bands is None:
        return np.arange(nbands, dtype=int)
    band_indices = normalize_integer_indices(bands, "bands")
    if np.any(band_indices < 0) or np.any(band_indices >= nbands):
        raise ValueError("bands contains an index outside the available band range.")
    return band_indices


def _normalize_groups(groups: Sequence[Sequence[int]], size: int, name: str) -> list:
    if len(groups) == 0:
        raise ValueError(f"{name} must contain at least one band group.")
    normalized = []
    for group in groups:
        arr = normalize_integer_indices(group, name)
        if np.any(arr < 0) or np.any(arr >= size):
            raise ValueError(f"{name} contains an index outside the available band range.")
        if np.unique(arr).size != arr.size:
            raise ValueError(f"{name} must not contain duplicate band indices within a group.")
        normalized.append(arr)
    return normalized


def _groups_to_bounds(groups: Sequence[Sequence[int]], size: int, name: str) -> np.ndarray:
    normalized = _normalize_groups(groups, size, name)
    bounds = np.zeros((len(normalized), 2), dtype=int)
    for igroup, group in enumerate(normalized):
        start = int(group[0])
        stop = int(group[-1]) + 1
        if not np.array_equal(group, np.arange(start, stop, dtype=int)):
            raise ValueError(f"{name} must contain contiguous band-index groups for NPZ persistence.")
        bounds[igroup] = (start, stop)
    return bounds


def _validate_group_bounds(bounds: np.ndarray, name: str) -> None:
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"{name} must have shape (ngroups, 2).")
    if bounds.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one [start, stop) range.")
    if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError(f"{name} must contain non-negative [start, stop) ranges.")


def _normalize_path_axis(path_axis: str) -> str:
    if not isinstance(path_axis, str):
        raise ValueError("path_axis must be 'q' or 'k'.")
    normalized = path_axis.lower()
    if normalized not in {"q", "k"}:
        raise ValueError("path_axis must be 'q' or 'k'.")
    return normalized


def _normalize_path_segments(path_segments: np.ndarray, npath: int) -> np.ndarray:
    path_segments = np.asarray(path_segments)
    if (
        path_segments.ndim != 2
        or path_segments.shape[1] != 2
        or np.issubdtype(path_segments.dtype, np.bool_)
        or not np.issubdtype(path_segments.dtype, np.integer)
    ):
        raise ValueError("path_segments must have shape (nsegments, 2).")
    path_segments = path_segments.astype(int, copy=False)
    if np.any(path_segments < 0):
        raise ValueError("path_segments must be non-negative.")
    if np.any(path_segments[:, 1] <= path_segments[:, 0]):
        raise ValueError("path_segments must contain increasing (start, stop) ranges.")
    if np.any(path_segments > npath):
        raise ValueError("path_segments must stay within the path point count.")
    return path_segments


def _normalize_weights(weights: np.ndarray, name: str) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError(f"{name} must be a one-dimensional non-empty array.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values.")
    total = weights.sum()
    if total <= 0.0:
        raise ValueError(f"{name} must have a positive sum.")
    return weights / total


def _scalar_string_from_npz(data, key: str) -> str:
    value = _required_npz_array(data, key, "DeePTB EPC path NPZ files")
    if np.shape(value) != ():
        raise ValueError(f"{key} must be a scalar string.")
    if hasattr(value, "item"):
        value = value.item()
    return str(value)
