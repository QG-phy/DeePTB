import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from dptb.postprocess.unified.eph.utils import (
    as_array,
    normalize_kpoints,
    reshape_phonopy_eigenvectors,
)


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
        self.masses = as_array(self.masses, dtype=float)
        if self.cell is not None:
            self.cell = as_array(self.cell, dtype=float)
        if self.scaled_positions is not None:
            self.scaled_positions = as_array(self.scaled_positions, dtype=float)

        nq = self.qpoints.shape[0]
        if self.frequencies.ndim != 2 or self.frequencies.shape[0] != nq:
            raise ValueError("frequencies must have shape (nq, nmodes).")
        if self.eigenvectors.ndim != 4:
            raise ValueError("eigenvectors must have shape (nq, nmodes, natoms, 3).")
        if self.eigenvectors.shape[:2] != self.frequencies.shape:
            raise ValueError("eigenvectors and frequencies must agree on (nq, nmodes).")
        if self.eigenvectors.shape[2] != self.masses.shape[0]:
            raise ValueError("eigenvectors natoms dimension must match masses.")
        if self.eigenvectors.shape[3] != 3:
            raise ValueError("eigenvectors Cartesian dimension must be 3.")

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

    def save_npz(self, path: Union[str, Path]) -> None:
        metadata_json = json.dumps(self.metadata)
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
            metadata_json=np.array(metadata_json),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "EPCData":
        with np.load(path, allow_pickle=False) as data:
            metadata = {}
            if "metadata_json" in data:
                metadata = json.loads(str(data["metadata_json"]))
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
