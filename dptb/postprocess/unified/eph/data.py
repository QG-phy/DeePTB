import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from dptb.postprocess.unified.eph.utils import (
    as_array,
    normalize_integer_indices,
    normalize_kpoints,
    reshape_phonopy_eigenvectors,
)


PHONON_NPZ_SCHEMA_VERSION = 1
EPC_NPZ_SCHEMA_VERSION = 1


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
                qpoints=data["ph_qpoints"],
                frequencies=data["ph_frequencies"],
                eigenvectors=data["ph_eigenvectors"],
                masses=data["ph_masses"],
                cell=data["ph_cell"] if "ph_cell" in data else None,
                scaled_positions=data["ph_scaled_positions"] if "ph_scaled_positions" in data else None,
                metadata=metadata,
            )

    @classmethod
    def from_phonopy(cls, phonopy_obj, qpoints: Optional[np.ndarray] = None):
        if qpoints is not None:
            phonopy_obj.run_qpoints(qpoints, with_eigenvectors=True)
            result = phonopy_obj.get_qpoints_dict()
            qpts = result["qpoints"]
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
                kpoints=data["el_kpoints"],
                qpoints=data["ph_qpoints"],
                band_indices=data["el_band_indices"],
                frequencies=data["ph_frequencies"],
                eigenvalues_k=data["el_eigenvalues_k"],
                eigenvalues_kq=data["el_eigenvalues_kq"],
                coupling_matrix=data["elph_coupling_matrix"],
                coupling_strength=data["elph_coupling_strength"],
                metadata=metadata,
            )


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


def _merge_metadata(defaults: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(metadata)
    for key, value in defaults.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"metadata[{key!r}] must be {value!r} for this DeePTB EPC NPZ schema.")
    merged = dict(metadata)
    merged.update(defaults)
    return merged


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")
