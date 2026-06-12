import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from dptb.kpoints.mesh import kmesh_sampling, time_symmetry_reduce
from dptb.postprocess.unified.eph.utils import (
    as_array,
    normalize_integer_indices,
    normalize_kpoints,
    reshape_phonopy_eigenvectors,
)


PHONON_NPZ_SCHEMA_VERSION = 1
EPC_NPZ_SCHEMA_VERSION = 1
EPC_PATH_NPZ_SCHEMA_VERSION = 1
EPC_MESH_NPZ_SCHEMA_VERSION = 1


@dataclass
class Phonons:
    """External phonon data used by the electron-phonon coupling accessor."""

    qpoints: np.ndarray
    frequencies: np.ndarray
    eigenvectors: np.ndarray
    masses: np.ndarray
    cell: Optional[np.ndarray] = None
    scaled_positions: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.qpoints = normalize_kpoints(self.qpoints)
        self.frequencies = as_array(self.frequencies, dtype=float)
        self.eigenvectors = as_array(self.eigenvectors, dtype=complex)
        self.masses = np.atleast_1d(as_array(self.masses, dtype=float))
        if self.cell is not None:
            self.cell = as_array(self.cell, dtype=float)
        if self.scaled_positions is not None:
            self.scaled_positions = as_array(self.scaled_positions, dtype=float)

        nq = self.qpoints.shape[0]
        if self.frequencies.ndim != 2 or self.frequencies.shape[0] != nq:
            raise ValueError("frequencies must have shape (nq, nmodes).")
        if self.frequencies.shape[1] == 0:
            raise ValueError("frequencies must contain at least one phonon mode.")
        if self.eigenvectors.ndim != 4:
            raise ValueError("eigenvectors must have shape (nq, nmodes, natoms, 3).")
        if self.eigenvectors.shape[:2] != self.frequencies.shape:
            raise ValueError("eigenvectors and frequencies must agree on (nq, nmodes).")
        if self.eigenvectors.shape[2] != self.masses.shape[0]:
            raise ValueError("eigenvectors natoms dimension must match masses.")
        if self.eigenvectors.shape[3] != 3:
            raise ValueError("eigenvectors Cartesian dimension must be 3.")
        if not np.all(np.isfinite(self.frequencies)):
            raise ValueError("frequencies must be finite.")
        if not np.all(np.isfinite(self.eigenvectors)):
            raise ValueError("eigenvectors must be finite.")
        if not np.all(np.isfinite(self.masses)) or np.any(self.masses <= 0.0):
            raise ValueError("masses must be finite and positive.")
        if self.cell is not None and self.cell.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3).")
        if self.cell is not None and not np.all(np.isfinite(self.cell)):
            raise ValueError("cell must be finite.")
        if self.scaled_positions is not None and self.scaled_positions.shape != (self.masses.shape[0], 3):
            raise ValueError("scaled_positions must have shape (natoms, 3).")
        if self.scaled_positions is not None and not np.all(np.isfinite(self.scaled_positions)):
            raise ValueError("scaled_positions must be finite.")

        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.phonons",
                "schema_version": PHONON_NPZ_SCHEMA_VERSION,
                "frequency_unit": "THz",
                "mass_unit": "amu",
                "eigenvector_basis": "cartesian",
            },
            self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save DeePTB phonon-mode data to NPZ."""
        payload = {
            "ph_qpoints": self.qpoints,
            "ph_frequencies": self.frequencies,
            "ph_eigenvectors": self.eigenvectors,
            "ph_masses": self.masses,
            "metadata_json": np.array(_metadata_to_json(self.metadata)),
        }
        if self.cell is not None:
            payload["ph_cell"] = self.cell
        if self.scaled_positions is not None:
            payload["ph_scaled_positions"] = self.scaled_positions
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "Phonons":
        """Load DeePTB phonon-mode data from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                qpoints=_required_npz_array(data, "ph_qpoints"),
                frequencies=_required_npz_array(data, "ph_frequencies"),
                eigenvectors=_required_npz_array(data, "ph_eigenvectors"),
                masses=_required_npz_array(data, "ph_masses"),
                cell=data["ph_cell"] if "ph_cell" in data else None,
                scaled_positions=data["ph_scaled_positions"] if "ph_scaled_positions" in data else None,
                metadata=metadata,
            )

    @classmethod
    def from_phonopy(cls, phonopy_obj, qpoints: Optional[np.ndarray] = None):
        if qpoints is not None:
            requested_qpoints = normalize_kpoints(qpoints)
            phonopy_obj.run_qpoints(requested_qpoints, with_eigenvectors=True)
            result = phonopy_obj.get_qpoints_dict()
            qpts = result.get("qpoints", requested_qpoints)
        else:
            result = phonopy_obj.get_mesh_dict()
            qpts = result["qpoints"]

        primitive = phonopy_obj.primitive
        masses = np.asarray(primitive.masses, dtype=float)
        cell = np.asarray(primitive.cell, dtype=float)
        scaled_positions = np.asarray(primitive.scaled_positions, dtype=float)

        return cls(
            qpoints=qpts,
            frequencies=result["frequencies"],
            eigenvectors=reshape_phonopy_eigenvectors(result["eigenvectors"], len(masses)),
            masses=masses,
            cell=cell,
            scaled_positions=scaled_positions,
            metadata={"source": "phonopy"},
        )

    @classmethod
    def from_phonopy_file(
        cls,
        phonopy_yaml: Union[str, Path],
        qpoints: Optional[np.ndarray] = None,
        force_sets_filename: Optional[Union[str, Path]] = None,
        **load_kwargs,
    ) -> "Phonons":
        """Load external phonon modes from a phonopy file.

        This is an auxiliary reader for preparing DeePTB phonon-mode NPZ data.
        It delegates file parsing to phonopy and does not compute force
        constants or phonon modes inside DeePTB.
        """
        try:
            from phonopy import load as phonopy_load
        except ImportError as exc:
            raise ImportError("phonopy is required to load external phonon files.") from exc

        load_options = dict(load_kwargs)
        if force_sets_filename is not None:
            load_options["force_sets_filename"] = str(force_sets_filename)
        phonopy_obj = phonopy_load(str(phonopy_yaml), **load_options)
        phonons = cls.from_phonopy(phonopy_obj, qpoints=qpoints)
        phonons.metadata.update(
            {
                "source": "phonopy_file",
                "phonopy_yaml": str(phonopy_yaml),
            }
        )
        if force_sets_filename is not None:
            phonons.metadata["force_sets_filename"] = str(force_sets_filename)
        return phonons


@dataclass
class EPCData:
    """Electron-phonon coupling data."""

    kpoints: np.ndarray
    qpoints: np.ndarray
    band_indices: np.ndarray
    frequencies: np.ndarray
    eigenvalues_k: np.ndarray
    eigenvalues_kq: np.ndarray
    coupling_matrix: np.ndarray
    coupling_strength: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.kpoints = normalize_kpoints(self.kpoints)
        self.qpoints = normalize_kpoints(self.qpoints)
        self.band_indices = normalize_integer_indices(self.band_indices, "band_indices")
        self.frequencies = as_array(self.frequencies, dtype=float)
        self.eigenvalues_k = as_array(self.eigenvalues_k, dtype=float)
        self.eigenvalues_kq = as_array(self.eigenvalues_kq, dtype=float)
        self.coupling_matrix = as_array(self.coupling_matrix, dtype=complex)
        self.coupling_strength = as_array(self.coupling_strength, dtype=float)
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_data",
                "schema_version": EPC_NPZ_SCHEMA_VERSION,
                "frequency_unit": "THz",
                "energy_unit": "eV",
                "coupling_unit": "eV",
                "coupling_strength_unit": "eV^2",
            },
            self.metadata,
        )

        nq = self.qpoints.shape[0]
        nk = self.kpoints.shape[0]
        nsel = self.band_indices.shape[0]
        if np.any(self.band_indices < 0):
            raise ValueError("band_indices must be non-negative.")
        if self.frequencies.ndim != 2 or self.frequencies.shape[0] != nq:
            raise ValueError("frequencies must have shape (nq, nmodes).")
        if self.frequencies.shape[1] == 0:
            raise ValueError("frequencies must contain at least one phonon mode.")
        if not np.all(np.isfinite(self.frequencies)):
            raise ValueError("frequencies must be finite.")
        if np.any(self.frequencies < 0.0):
            raise ValueError("frequencies must be non-negative; imaginary modes are not supported in EPC v1.")
        if not np.all(np.isfinite(self.eigenvalues_k)):
            raise ValueError("eigenvalues_k must be finite.")
        if not np.all(np.isfinite(self.eigenvalues_kq)):
            raise ValueError("eigenvalues_kq must be finite.")
        if not np.all(np.isfinite(self.coupling_matrix)):
            raise ValueError("coupling_matrix must be finite.")
        if not np.all(np.isfinite(self.coupling_strength)) or np.any(self.coupling_strength < 0.0):
            raise ValueError("coupling_strength must contain finite non-negative values.")
        if self.eigenvalues_k.shape != (nk, nsel):
            raise ValueError("eigenvalues_k must have shape (nk, nbands_selected).")
        if self.eigenvalues_kq.shape != (nq, nk, nsel):
            raise ValueError("eigenvalues_kq must have shape (nq, nk, nbands_selected).")
        expected_coupling_shape = (nq, nk, self.frequencies.shape[1], nsel, nsel)
        if self.coupling_matrix.shape != expected_coupling_shape:
            raise ValueError("coupling_matrix must have shape (nq, nk, nmodes, nbands_selected, nbands_selected).")
        if self.coupling_strength.shape != expected_coupling_shape:
            raise ValueError("coupling_strength must have shape (nq, nk, nmodes, nbands_selected, nbands_selected).")

    def save_npz(self, path: Union[str, Path]) -> None:
        np.savez_compressed(
            path,
            ph_qpoints=self.qpoints,
            ph_frequencies=self.frequencies,
            el_kpoints=self.kpoints,
            el_band_indices=self.band_indices,
            el_eigenvalues_k=self.eigenvalues_k,
            el_eigenvalues_kq=self.eigenvalues_kq,
            elph_coupling_matrix=self.coupling_matrix,
            elph_coupling_strength=self.coupling_strength,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "EPCData":
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                kpoints=_required_npz_array(data, "el_kpoints"),
                qpoints=_required_npz_array(data, "ph_qpoints"),
                band_indices=_required_npz_array(data, "el_band_indices"),
                frequencies=_required_npz_array(data, "ph_frequencies"),
                eigenvalues_k=_required_npz_array(data, "el_eigenvalues_k"),
                eigenvalues_kq=_required_npz_array(data, "el_eigenvalues_kq"),
                coupling_matrix=_required_npz_array(data, "elph_coupling_matrix"),
                coupling_strength=_required_npz_array(data, "elph_coupling_strength"),
                metadata=metadata,
            )


@dataclass
class EPCMeshSpec:
    """DeePTB-native EPC mesh specification.

    Phonon modes are still read externally through ``Phonons``. ``q_mesh`` is
    only a validation/metadata aid; it does not generate phonons inside DeePTB.
    """

    kpoints: Optional[np.ndarray] = None
    k_mesh: Optional[Sequence[int]] = None
    q_mesh: Optional[Sequence[int]] = None
    gamma_centered: bool = True
    time_reversal: bool = False
    kpoint_weights: Optional[np.ndarray] = None
    qpoint_weights: Optional[np.ndarray] = None
    chunk_size: Optional[int] = None
    q_chunk_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kpoints is None and self.k_mesh is None:
            raise ValueError("EPCMeshSpec requires either kpoints or k_mesh.")
        if self.kpoints is not None and self.k_mesh is not None:
            raise ValueError("EPCMeshSpec accepts either kpoints or k_mesh, not both.")
        self.gamma_centered = _validate_bool(self.gamma_centered, "gamma_centered")
        self.time_reversal = _validate_bool(self.time_reversal, "time_reversal")
        if self.kpoints is not None:
            self.kpoints = normalize_kpoints(self.kpoints)
        if self.k_mesh is not None:
            self.k_mesh = _normalize_mesh(self.k_mesh, "k_mesh")
        if self.q_mesh is not None:
            self.q_mesh = _normalize_mesh(self.q_mesh, "q_mesh")
        if self.kpoint_weights is not None:
            self.kpoint_weights = _normalize_weights(self.kpoint_weights, "kpoint_weights")
        if self.qpoint_weights is not None:
            self.qpoint_weights = _normalize_weights(self.qpoint_weights, "qpoint_weights")
        if self.chunk_size is not None:
            self.chunk_size = _normalize_chunk_size(self.chunk_size, "chunk_size")
        if self.q_chunk_size is not None:
            self.q_chunk_size = _normalize_chunk_size(self.q_chunk_size, "q_chunk_size")

    def resolve_kpoints_and_weights(self) -> tuple:
        if self.kpoints is not None:
            kpoints = self.kpoints
            weights = self.kpoint_weights
            if weights is None:
                weights = np.full(kpoints.shape[0], 1.0 / kpoints.shape[0], dtype=float)
            elif weights.shape != (kpoints.shape[0],):
                raise ValueError("kpoint_weights must match the explicit kpoints count.")
            return kpoints, _normalize_weights(weights, "kpoint_weights")

        if self.time_reversal:
            kpoints, weights = time_symmetry_reduce(self.k_mesh, is_gamma_center=self.gamma_centered)
        else:
            kpoints = kmesh_sampling(self.k_mesh, is_gamma_center=self.gamma_centered)
            weights = np.full(kpoints.shape[0], 1.0 / kpoints.shape[0], dtype=float)
        if self.kpoint_weights is not None:
            if self.kpoint_weights.shape != (kpoints.shape[0],):
                raise ValueError("kpoint_weights must match the generated k mesh point count.")
            weights = self.kpoint_weights
        return normalize_kpoints(kpoints), _normalize_weights(weights, "kpoint_weights")

    def resolve_qpoint_weights(self, phonons: Phonons) -> np.ndarray:
        if self.q_mesh is not None:
            generated = kmesh_sampling(self.q_mesh, is_gamma_center=self.gamma_centered)
            if generated.shape != phonons.qpoints.shape or not np.allclose(generated, phonons.qpoints):
                raise ValueError("q_mesh must generate qpoints matching the external Phonons qpoints.")
        if self.qpoint_weights is None:
            weights = np.full(phonons.qpoints.shape[0], 1.0 / phonons.qpoints.shape[0], dtype=float)
        else:
            weights = self.qpoint_weights
            if weights.shape != (phonons.qpoints.shape[0],):
                raise ValueError("qpoint_weights must match the external Phonons qpoint count.")
        return _normalize_weights(weights, "qpoint_weights")

    def metadata_payload(self) -> Dict[str, Any]:
        payload = dict(self.metadata)
        payload.update(
            {
                "k_mesh": self.k_mesh,
                "q_mesh": self.q_mesh,
                "gamma_centered": self.gamma_centered,
                "time_reversal": self.time_reversal,
                "chunk_size": self.chunk_size,
                "q_chunk_size": self.q_chunk_size,
            }
        )
        return payload


@dataclass
class EPCMeshData:
    """Electron-phonon coupling data on a DeePTB-native k/q mesh."""

    kpoints: np.ndarray
    qpoints: np.ndarray
    band_indices: np.ndarray
    frequencies: np.ndarray
    eigenvalues_k: np.ndarray
    eigenvalues_kq: np.ndarray
    coupling_matrix: np.ndarray
    coupling_strength: np.ndarray
    kpoint_weights: np.ndarray
    qpoint_weights: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_mesh_data",
                "schema_version": EPC_MESH_NPZ_SCHEMA_VERSION,
                "frequency_unit": "THz",
                "energy_unit": "eV",
                "coupling_unit": "eV",
                "coupling_strength_unit": "eV^2",
            },
            self.metadata,
        )
        self._epc_data = EPCData(
            kpoints=self.kpoints,
            qpoints=self.qpoints,
            band_indices=self.band_indices,
            frequencies=self.frequencies,
            eigenvalues_k=self.eigenvalues_k,
            eigenvalues_kq=self.eigenvalues_kq,
            coupling_matrix=self.coupling_matrix,
            coupling_strength=self.coupling_strength,
            metadata={
                key: value
                for key, value in self.metadata.items()
                if key not in {"schema", "schema_version"}
            },
        )
        self.kpoint_weights = _normalize_weights(self.kpoint_weights, "kpoint_weights")
        self.qpoint_weights = _normalize_weights(self.qpoint_weights, "qpoint_weights")
        if self.kpoint_weights.shape != (self.kpoints.shape[0],):
            raise ValueError("kpoint_weights must match kpoints.")
        if self.qpoint_weights.shape != (self.qpoints.shape[0],):
            raise ValueError("qpoint_weights must match qpoints.")

    @property
    def epc_data(self) -> EPCData:
        return self._epc_data

    @classmethod
    def from_epc_data(
        cls,
        epc_data: EPCData,
        kpoint_weights: np.ndarray,
        qpoint_weights: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EPCMeshData":
        mesh_metadata = {
            key: value
            for key, value in epc_data.metadata.items()
            if key not in {"schema", "schema_version"}
        }
        if metadata:
            mesh_metadata.update(metadata)
        return cls(
            kpoints=epc_data.kpoints,
            qpoints=epc_data.qpoints,
            band_indices=epc_data.band_indices,
            frequencies=epc_data.frequencies,
            eigenvalues_k=epc_data.eigenvalues_k,
            eigenvalues_kq=epc_data.eigenvalues_kq,
            coupling_matrix=epc_data.coupling_matrix,
            coupling_strength=epc_data.coupling_strength,
            kpoint_weights=kpoint_weights,
            qpoint_weights=qpoint_weights,
            metadata=mesh_metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        np.savez_compressed(
            path,
            ph_qpoints=self.qpoints,
            ph_frequencies=self.frequencies,
            ph_qpoint_weights=self.qpoint_weights,
            el_kpoints=self.kpoints,
            el_kpoint_weights=self.kpoint_weights,
            el_band_indices=self.band_indices,
            el_eigenvalues_k=self.eigenvalues_k,
            el_eigenvalues_kq=self.eigenvalues_kq,
            elph_coupling_matrix=self.coupling_matrix,
            elph_coupling_strength=self.coupling_strength,
            metadata_json=np.array(_metadata_to_json(self.metadata)),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "EPCMeshData":
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            return cls(
                kpoints=_required_npz_array(data, "el_kpoints"),
                qpoints=_required_npz_array(data, "ph_qpoints"),
                band_indices=_required_npz_array(data, "el_band_indices"),
                frequencies=_required_npz_array(data, "ph_frequencies"),
                eigenvalues_k=_required_npz_array(data, "el_eigenvalues_k"),
                eigenvalues_kq=_required_npz_array(data, "el_eigenvalues_kq"),
                coupling_matrix=_required_npz_array(data, "elph_coupling_matrix"),
                coupling_strength=_required_npz_array(data, "elph_coupling_strength"),
                kpoint_weights=_required_npz_array(data, "el_kpoint_weights"),
                qpoint_weights=_required_npz_array(data, "ph_qpoint_weights"),
                metadata=metadata,
            )


@dataclass
class EPCPathData:
    """Electron-phonon coupling data on a DeePTB-native path workflow."""

    kpoints: np.ndarray
    qpoints: np.ndarray
    band_indices: np.ndarray
    frequencies: np.ndarray
    eigenvalues_k: np.ndarray
    eigenvalues_kq: np.ndarray
    coupling_matrix: np.ndarray
    coupling_strength: np.ndarray
    path_axis: str
    path_coordinates: np.ndarray
    path_segments: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.path_axis = _normalize_path_axis(self.path_axis)
        self.path_coordinates = as_array(self.path_coordinates, dtype=float)
        if self.path_coordinates.ndim != 1 or self.path_coordinates.size == 0:
            raise ValueError("path_coordinates must be a one-dimensional non-empty array.")
        if not np.all(np.isfinite(self.path_coordinates)):
            raise ValueError("path_coordinates must be finite.")

        if self.path_segments is not None:
            self.path_segments = as_array(self.path_segments)
            if (
                self.path_segments.ndim != 2
                or self.path_segments.shape[1] != 2
                or np.issubdtype(self.path_segments.dtype, np.bool_)
                or not np.issubdtype(self.path_segments.dtype, np.integer)
            ):
                raise ValueError("path_segments must have shape (nsegments, 2).")
            self.path_segments = self.path_segments.astype(int, copy=False)
            if np.any(self.path_segments < 0):
                raise ValueError("path_segments must be non-negative.")

        self.metadata = _merge_metadata(
            {
                "schema": "deeptb.epc_path_data",
                "schema_version": EPC_PATH_NPZ_SCHEMA_VERSION,
                "frequency_unit": "THz",
                "energy_unit": "eV",
                "coupling_unit": "eV",
                "coupling_strength_unit": "eV^2",
                "path_coordinate_unit": "fractional_reciprocal_coordinate_distance",
            },
            self.metadata,
        )

        self._epc_data = EPCData(
            kpoints=self.kpoints,
            qpoints=self.qpoints,
            band_indices=self.band_indices,
            frequencies=self.frequencies,
            eigenvalues_k=self.eigenvalues_k,
            eigenvalues_kq=self.eigenvalues_kq,
            coupling_matrix=self.coupling_matrix,
            coupling_strength=self.coupling_strength,
            metadata={
                key: value
                for key, value in self.metadata.items()
                if key not in {"schema", "schema_version", "path_coordinate_unit"}
            },
        )

        npath = self.qpoints.shape[0] if self.path_axis == "q" else self.kpoints.shape[0]
        if self.path_coordinates.shape[0] != npath:
            raise ValueError(f"path_coordinates length must match the {self.path_axis}-path point count.")
        if self.path_segments is not None:
            if np.any(self.path_segments[:, 1] <= self.path_segments[:, 0]):
                raise ValueError("path_segments must contain increasing (start, stop) ranges.")
            if np.any(self.path_segments > npath):
                raise ValueError("path_segments must stay within the path point count.")

    @property
    def epc_data(self) -> EPCData:
        return self._epc_data

    @classmethod
    def from_epc_data(
        cls,
        epc_data: EPCData,
        path_axis: str,
        path_coordinates: np.ndarray,
        path_segments: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EPCPathData":
        path_metadata = {
            key: value
            for key, value in epc_data.metadata.items()
            if key not in {"schema", "schema_version"}
        }
        if metadata:
            path_metadata.update(metadata)
        return cls(
            kpoints=epc_data.kpoints,
            qpoints=epc_data.qpoints,
            band_indices=epc_data.band_indices,
            frequencies=epc_data.frequencies,
            eigenvalues_k=epc_data.eigenvalues_k,
            eigenvalues_kq=epc_data.eigenvalues_kq,
            coupling_matrix=epc_data.coupling_matrix,
            coupling_strength=epc_data.coupling_strength,
            path_axis=path_axis,
            path_coordinates=path_coordinates,
            path_segments=path_segments,
            metadata=path_metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        payload = {
            "ph_qpoints": self.qpoints,
            "ph_frequencies": self.frequencies,
            "el_kpoints": self.kpoints,
            "el_band_indices": self.band_indices,
            "el_eigenvalues_k": self.eigenvalues_k,
            "el_eigenvalues_kq": self.eigenvalues_kq,
            "elph_coupling_matrix": self.coupling_matrix,
            "elph_coupling_strength": self.coupling_strength,
            "path_axis": np.array(self.path_axis),
            "path_coordinates": self.path_coordinates,
            "metadata_json": np.array(_metadata_to_json(self.metadata)),
        }
        if self.path_segments is not None:
            payload["path_segments"] = self.path_segments
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "EPCPathData":
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata_from_npz(data)
            path_axis = _scalar_string_from_npz(data, "path_axis")
            return cls(
                kpoints=_required_npz_array(data, "el_kpoints"),
                qpoints=_required_npz_array(data, "ph_qpoints"),
                band_indices=_required_npz_array(data, "el_band_indices"),
                frequencies=_required_npz_array(data, "ph_frequencies"),
                eigenvalues_k=_required_npz_array(data, "el_eigenvalues_k"),
                eigenvalues_kq=_required_npz_array(data, "el_eigenvalues_kq"),
                coupling_matrix=_required_npz_array(data, "elph_coupling_matrix"),
                coupling_strength=_required_npz_array(data, "elph_coupling_strength"),
                path_axis=path_axis,
                path_coordinates=_required_npz_array(data, "path_coordinates"),
                path_segments=data["path_segments"] if "path_segments" in data else None,
                metadata=metadata,
            )


def cumulative_path_coordinates(points: np.ndarray) -> np.ndarray:
    """Return cumulative distances along fractional reciprocal coordinates."""
    points = normalize_kpoints(points)
    coordinate = np.zeros(points.shape[0], dtype=float)
    if points.shape[0] > 1:
        coordinate[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    return coordinate


def _metadata_to_json(metadata: Dict[str, Any]) -> str:
    return json.dumps(metadata, default=_json_default)


def _metadata_from_npz(data) -> Dict[str, Any]:
    if "metadata_json" not in data:
        raise ValueError("metadata_json is required for DeePTB EPC NPZ files.")
    value = data["metadata_json"]
    if np.shape(value) != ():
        raise ValueError("metadata_json must be a scalar JSON object.")
    if hasattr(value, "item"):
        value = value.item()
    try:
        metadata = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json must contain valid JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to a JSON object.")
    return metadata


def _required_npz_array(data, key: str, context: str = "DeePTB EPC NPZ files"):
    if key not in data:
        raise ValueError(f"{key} is required for {context}.")
    try:
        return data[key]
    except ValueError as exc:
        raise ValueError(f"{key} could not be loaded for {context}: {exc}") from exc


def _scalar_string_from_npz(data, key: str) -> str:
    value = _required_npz_array(data, key, "DeePTB EPC path NPZ files")
    if np.shape(value) != ():
        raise ValueError(f"{key} must be a scalar string.")
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a scalar string.")
    return str(value)


def _merge_metadata(defaults: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(metadata)
    for key, value in defaults.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"metadata[{key!r}] must be {value!r} for this DeePTB EPC NPZ schema.")
    merged = dict(metadata)
    merged.update(defaults)
    return merged


def _normalize_path_axis(path_axis: str) -> str:
    if not isinstance(path_axis, str):
        raise ValueError("path_axis must be 'q' or 'k'.")
    normalized = path_axis.lower()
    if normalized not in {"q", "k"}:
        raise ValueError("path_axis must be 'q' or 'k'.")
    return normalized


def _validate_bool(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean.")
    return bool(value)


def _normalize_mesh(mesh: Sequence[int], name: str) -> list:
    arr = as_array(mesh)
    if arr.ndim != 1 or arr.shape[0] != 3:
        raise ValueError(f"{name} must contain 3 positive integers.")
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must contain 3 positive integers.")
    if not np.all(np.isfinite(arr)) or not np.all(np.equal(arr, arr.astype(int))):
        raise ValueError(f"{name} must contain 3 positive integers.")
    arr = arr.astype(int)
    if np.any(arr <= 0):
        raise ValueError(f"{name} must contain 3 positive integers.")
    return arr.tolist()


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


def _normalize_chunk_size(chunk_size, name: str) -> int:
    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return chunk_size


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")
