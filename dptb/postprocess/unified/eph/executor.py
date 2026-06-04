import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from dptb.postprocess.unified.eph.data import EPCData, EPCMeshData, _metadata_to_json, _normalize_weights


EPC_MESH_CHUNKED_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EPCKChunkSpec:
    """Rank-independent k-point chunk spec for serial/future parallel EPC."""

    chunk_index: int
    k_start: int
    k_stop: int

    def __post_init__(self):
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, (int, np.integer)):
            raise ValueError("chunk_index must be an integer.")
        if isinstance(self.k_start, bool) or not isinstance(self.k_start, (int, np.integer)):
            raise ValueError("k_start must be an integer.")
        if isinstance(self.k_stop, bool) or not isinstance(self.k_stop, (int, np.integer)):
            raise ValueError("k_stop must be an integer.")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative.")
        if self.k_start < 0 or self.k_stop <= self.k_start:
            raise ValueError("k chunk ranges must be non-empty [start, stop) ranges.")

    @property
    def slice(self) -> slice:
        return slice(int(self.k_start), int(self.k_stop))

    def metadata(self) -> dict:
        return {
            "chunk_index": int(self.chunk_index),
            "k_start": int(self.k_start),
            "k_stop": int(self.k_stop),
        }


@dataclass(frozen=True)
class EPCQChunkSpec:
    """Rank-independent q-point chunk spec for serial/future parallel EPC."""

    chunk_index: int
    q_start: int
    q_stop: int

    def __post_init__(self):
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, (int, np.integer)):
            raise ValueError("chunk_index must be an integer.")
        if isinstance(self.q_start, bool) or not isinstance(self.q_start, (int, np.integer)):
            raise ValueError("q_start must be an integer.")
        if isinstance(self.q_stop, bool) or not isinstance(self.q_stop, (int, np.integer)):
            raise ValueError("q_stop must be an integer.")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative.")
        if self.q_start < 0 or self.q_stop <= self.q_start:
            raise ValueError("q chunk ranges must be non-empty [start, stop) ranges.")

    @property
    def slice(self) -> slice:
        return slice(int(self.q_start), int(self.q_stop))

    def metadata(self) -> dict:
        return {
            "chunk_index": int(self.chunk_index),
            "q_start": int(self.q_start),
            "q_stop": int(self.q_stop),
        }


def build_k_chunk_specs(nk: int, chunk_size: Optional[int]) -> Sequence[EPCKChunkSpec]:
    """Build deterministic k-axis chunk specs."""
    if isinstance(nk, bool) or not isinstance(nk, (int, np.integer)) or nk <= 0:
        raise ValueError("nk must be a positive integer.")
    nk = int(nk)
    if chunk_size is None:
        return [EPCKChunkSpec(chunk_index=0, k_start=0, k_stop=nk)]
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    chunk_size = int(chunk_size)
    if chunk_size >= nk:
        return [EPCKChunkSpec(chunk_index=0, k_start=0, k_stop=nk)]
    return [
        EPCKChunkSpec(chunk_index=ichunk, k_start=start, k_stop=min(start + chunk_size, nk))
        for ichunk, start in enumerate(range(0, nk, chunk_size))
    ]


def build_q_chunk_specs(nq: int, chunk_size: Optional[int]) -> Sequence[EPCQChunkSpec]:
    """Build deterministic q-axis chunk specs."""
    if isinstance(nq, bool) or not isinstance(nq, (int, np.integer)) or nq <= 0:
        raise ValueError("nq must be a positive integer.")
    nq = int(nq)
    if chunk_size is None:
        return [EPCQChunkSpec(chunk_index=0, q_start=0, q_stop=nq)]
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    chunk_size = int(chunk_size)
    if chunk_size >= nq:
        return [EPCQChunkSpec(chunk_index=0, q_start=0, q_stop=nq)]
    return [
        EPCQChunkSpec(chunk_index=ichunk, q_start=start, q_stop=min(start + chunk_size, nq))
        for ichunk, start in enumerate(range(0, nq, chunk_size))
    ]


def concat_epc_k_chunks(chunks: Sequence[EPCData]) -> EPCData:
    """Deterministically concatenate EPC chunks along the electronic k axis."""
    if len(chunks) == 0:
        raise ValueError("At least one EPC chunk is required.")
    first = chunks[0]
    for chunk in chunks[1:]:
        if not np.array_equal(chunk.qpoints, first.qpoints):
            raise ValueError("EPC chunks must share qpoints.")
        if not np.array_equal(chunk.band_indices, first.band_indices):
            raise ValueError("EPC chunks must share band_indices.")
        if not np.allclose(chunk.frequencies, first.frequencies):
            raise ValueError("EPC chunks must share phonon frequencies.")
        if chunk.coupling_matrix.shape[0] != first.coupling_matrix.shape[0]:
            raise ValueError("EPC chunks must share qpoint count.")
        if chunk.coupling_matrix.shape[2:] != first.coupling_matrix.shape[2:]:
            raise ValueError("EPC chunks must share coupling trailing shape.")

    return EPCData(
        kpoints=np.concatenate([chunk.kpoints for chunk in chunks], axis=0),
        qpoints=first.qpoints,
        band_indices=first.band_indices,
        frequencies=first.frequencies,
        eigenvalues_k=np.concatenate([chunk.eigenvalues_k for chunk in chunks], axis=0),
        eigenvalues_kq=np.concatenate([chunk.eigenvalues_kq for chunk in chunks], axis=1),
        coupling_matrix=np.concatenate([chunk.coupling_matrix for chunk in chunks], axis=1),
        coupling_strength=np.concatenate([chunk.coupling_strength for chunk in chunks], axis=1),
        metadata={
            "source": "deeptb.eph.compute_coupling.k_chunk_concat",
            "frequency_unit": "THz",
            "energy_unit": "eV",
            "coupling_unit": "eV",
            "coupling_strength_unit": "eV^2",
            "chunk_count": len(chunks),
            "chunk_sources": [chunk.metadata for chunk in chunks],
        },
    )


def concat_epc_q_chunks(chunks: Sequence[EPCData]) -> EPCData:
    """Deterministically concatenate EPC chunks along the phonon q axis."""
    if len(chunks) == 0:
        raise ValueError("At least one EPC chunk is required.")
    first = chunks[0]
    for chunk in chunks[1:]:
        if not np.array_equal(chunk.kpoints, first.kpoints):
            raise ValueError("EPC chunks must share kpoints.")
        if not np.array_equal(chunk.band_indices, first.band_indices):
            raise ValueError("EPC chunks must share band_indices.")
        if not np.allclose(chunk.eigenvalues_k, first.eigenvalues_k):
            raise ValueError("EPC chunks must share eigenvalues_k.")
        if chunk.coupling_matrix.shape[1] != first.coupling_matrix.shape[1]:
            raise ValueError("EPC chunks must share kpoint count.")
        if chunk.coupling_matrix.shape[2:] != first.coupling_matrix.shape[2:]:
            raise ValueError("EPC chunks must share coupling trailing shape.")

    return EPCData(
        kpoints=first.kpoints,
        qpoints=np.concatenate([chunk.qpoints for chunk in chunks], axis=0),
        band_indices=first.band_indices,
        frequencies=np.concatenate([chunk.frequencies for chunk in chunks], axis=0),
        eigenvalues_k=first.eigenvalues_k,
        eigenvalues_kq=np.concatenate([chunk.eigenvalues_kq for chunk in chunks], axis=0),
        coupling_matrix=np.concatenate([chunk.coupling_matrix for chunk in chunks], axis=0),
        coupling_strength=np.concatenate([chunk.coupling_strength for chunk in chunks], axis=0),
        metadata={
            "source": "deeptb.eph.compute_coupling.q_chunk_concat",
            "frequency_unit": "THz",
            "energy_unit": "eV",
            "coupling_unit": "eV",
            "coupling_strength_unit": "eV^2",
            "chunk_count": len(chunks),
            "chunk_sources": [chunk.metadata for chunk in chunks],
        },
    )


def save_epc_mesh_chunked_artifact(
    mesh_data: EPCMeshData,
    directory: Union[str, Path],
    axis: str,
    chunk_size: int,
) -> None:
    """Save an ``EPCMeshData`` object as deterministic chunked NPZ artifacts.

    This is a storage/reducer boundary for large mesh workflows. It does not
    introduce parallel execution and does not change the public EPC NPZ schema.
    """
    if not isinstance(mesh_data, EPCMeshData):
        raise ValueError("mesh_data must be an EPCMeshData object.")
    axis = _normalize_chunk_axis(axis)
    chunk_size = _validate_chunk_size(chunk_size)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if axis == "k":
        specs = build_k_chunk_specs(mesh_data.kpoints.shape[0], chunk_size)
    else:
        specs = build_q_chunk_specs(mesh_data.qpoints.shape[0], chunk_size)

    chunks = []
    for spec in specs:
        chunk_data = _slice_mesh_epc_data(mesh_data, axis, spec.slice, spec.metadata())
        filename = f"{axis}_chunk_{spec.chunk_index:06d}.npz"
        chunk_data.save_npz(directory / filename)
        chunks.append(
            {
                "filename": filename,
                "axis": axis,
                "spec": spec.metadata(),
            }
        )

    np.savez_compressed(
        directory / "weights.npz",
        el_kpoint_weights=mesh_data.kpoint_weights,
        ph_qpoint_weights=mesh_data.qpoint_weights,
        metadata_json=np.array(
            json.dumps(
                {
                    "schema": "deeptb.epc_mesh_chunked_artifact.weights",
                    "schema_version": EPC_MESH_CHUNKED_ARTIFACT_SCHEMA_VERSION,
                },
                sort_keys=True,
            )
        ),
    )
    manifest = {
        "schema": "deeptb.epc_mesh_chunked_artifact",
        "schema_version": EPC_MESH_CHUNKED_ARTIFACT_SCHEMA_VERSION,
        "axis": axis,
        "chunk_size": chunk_size,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "weights_filename": "weights.npz",
        "mesh_metadata": mesh_data.metadata,
        "reducer": f"concat_epc_{axis}_chunks",
    }
    (directory / "manifest.json").write_text(
        json.dumps(json.loads(_metadata_to_json(manifest)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_epc_mesh_chunked_manifest(directory: Union[str, Path]) -> dict:
    """Read and validate an EPC mesh chunked artifact manifest."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("manifest.json is required for EPC mesh chunked artifact.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise ValueError("manifest.json must be valid JSON.") from None
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object.")
    if manifest.get("schema") != "deeptb.epc_mesh_chunked_artifact":
        raise ValueError("manifest.json must describe a DeePTB EPC mesh chunked artifact.")
    if manifest.get("schema_version") != EPC_MESH_CHUNKED_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported EPC mesh chunked artifact schema_version.")
    axis = _normalize_chunk_axis(manifest.get("axis"))
    chunk_count = manifest.get("chunk_count")
    if isinstance(chunk_count, bool) or not isinstance(chunk_count, int) or chunk_count <= 0:
        raise ValueError("manifest.json chunk_count must be a positive integer.")
    chunk_size = manifest.get("chunk_size")
    _validate_chunk_size(chunk_size)
    expected_reducer = f"concat_epc_{axis}_chunks"
    if manifest.get("reducer") != expected_reducer:
        raise ValueError(f"manifest.json reducer must be {expected_reducer!r}.")
    weights_filename = manifest.get("weights_filename", "weights.npz")
    _validate_artifact_filename(weights_filename, "weights_filename")
    if "mesh_metadata" in manifest and not isinstance(manifest["mesh_metadata"], dict):
        raise ValueError("manifest.json mesh_metadata must be a JSON object.")
    chunk_entries = manifest.get("chunks")
    if not isinstance(chunk_entries, list) or len(chunk_entries) == 0:
        raise ValueError("EPC mesh chunked artifact must contain at least one chunk.")
    if len(chunk_entries) != chunk_count:
        raise ValueError("manifest.json chunk_count must match chunks length.")

    expected_start = 0
    for expected_index, entry in enumerate(chunk_entries):
        if not isinstance(entry, dict):
            raise ValueError("Each chunk manifest entry must be a JSON object.")
        if entry.get("axis") != axis:
            raise ValueError("Chunk manifest axis must match artifact axis.")
        filename = entry.get("filename")
        _validate_artifact_filename(filename, "chunk filename")
        spec = entry.get("spec")
        if not isinstance(spec, dict):
            raise ValueError("Chunk manifest entry requires a spec object.")
        chunk_index, start, stop = _chunk_bounds_from_spec(axis, spec)
        if chunk_index != expected_index:
            raise ValueError("Chunk manifest entries must be ordered by contiguous chunk_index.")
        if start != expected_start:
            raise ValueError("Chunk manifest ranges must be contiguous and ordered.")
        expected_start = stop
    return manifest


def read_epc_mesh_chunked_weights(directory: Union[str, Path], manifest: Optional[dict] = None) -> tuple:
    """Read and validate global k/q weights from a chunked mesh artifact."""
    directory = Path(directory)
    if manifest is None:
        manifest = read_epc_mesh_chunked_manifest(directory)
    weights_filename = manifest.get("weights_filename", "weights.npz")
    _validate_artifact_filename(weights_filename, "weights_filename")
    with np.load(directory / weights_filename, allow_pickle=False) as weights_payload:
        if "metadata_json" not in weights_payload:
            raise ValueError("weights.npz metadata_json is required.")
        try:
            weights_metadata = json.loads(str(weights_payload["metadata_json"].item()))
        except (AttributeError, json.JSONDecodeError):
            raise ValueError("weights.npz metadata_json must be valid JSON.") from None
        if not isinstance(weights_metadata, dict):
            raise ValueError("weights.npz metadata_json must decode to a JSON object.")
        if weights_metadata.get("schema") != "deeptb.epc_mesh_chunked_artifact.weights":
            raise ValueError("weights.npz schema must be deeptb.epc_mesh_chunked_artifact.weights.")
        if weights_metadata.get("schema_version") != EPC_MESH_CHUNKED_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported weights.npz schema_version.")
        try:
            kpoint_weights = weights_payload["el_kpoint_weights"]
            qpoint_weights = weights_payload["ph_qpoint_weights"]
        except KeyError as exc:
            raise ValueError("weights.npz must contain el_kpoint_weights and ph_qpoint_weights.") from exc
        kpoint_weights = _normalize_weights(kpoint_weights, "el_kpoint_weights")
        qpoint_weights = _normalize_weights(qpoint_weights, "ph_qpoint_weights")
    return kpoint_weights, qpoint_weights


def load_epc_mesh_chunked_artifact(directory: Union[str, Path]) -> EPCMeshData:
    """Load a chunked EPC mesh artifact and reduce it to ``EPCMeshData``."""
    directory = Path(directory)
    manifest = read_epc_mesh_chunked_manifest(directory)
    axis = manifest["axis"]

    chunks = []
    for entry in manifest["chunks"]:
        filename = entry["filename"]
        chunks.append(EPCData.load_npz(directory / filename))

    kpoint_weights, qpoint_weights = read_epc_mesh_chunked_weights(directory, manifest)

    epc_data = concat_epc_k_chunks(chunks) if axis == "k" else concat_epc_q_chunks(chunks)
    metadata = dict(manifest.get("mesh_metadata") or {})
    metadata.update(
        {
            "source": "deeptb.eph.epc_mesh_chunked_artifact",
            "chunked_artifact": True,
            "artifact_axis": axis,
            "artifact_chunk_count": len(chunks),
            "artifact_reducer": manifest.get("reducer"),
        }
    )
    return EPCMeshData.from_epc_data(
        epc_data,
        kpoint_weights=kpoint_weights,
        qpoint_weights=qpoint_weights,
        metadata=metadata,
    )


def _slice_mesh_epc_data(mesh_data: EPCMeshData, axis: str, chunk_slice: slice, chunk_metadata: dict) -> EPCData:
    if axis == "k":
        return EPCData(
            kpoints=mesh_data.kpoints[chunk_slice],
            qpoints=mesh_data.qpoints,
            band_indices=mesh_data.band_indices,
            frequencies=mesh_data.frequencies,
            eigenvalues_k=mesh_data.eigenvalues_k[chunk_slice],
            eigenvalues_kq=mesh_data.eigenvalues_kq[:, chunk_slice, :],
            coupling_matrix=mesh_data.coupling_matrix[:, chunk_slice, :, :, :],
            coupling_strength=mesh_data.coupling_strength[:, chunk_slice, :, :, :],
            metadata=_chunk_metadata(mesh_data, axis, chunk_metadata),
        )
    return EPCData(
        kpoints=mesh_data.kpoints,
        qpoints=mesh_data.qpoints[chunk_slice],
        band_indices=mesh_data.band_indices,
        frequencies=mesh_data.frequencies[chunk_slice],
        eigenvalues_k=mesh_data.eigenvalues_k,
        eigenvalues_kq=mesh_data.eigenvalues_kq[chunk_slice, :, :],
        coupling_matrix=mesh_data.coupling_matrix[chunk_slice, :, :, :, :],
        coupling_strength=mesh_data.coupling_strength[chunk_slice, :, :, :, :],
        metadata=_chunk_metadata(mesh_data, axis, chunk_metadata),
    )


def _chunk_metadata(mesh_data: EPCMeshData, axis: str, chunk_metadata: dict) -> dict:
    return {
        "source": "deeptb.eph.epc_mesh_chunked_artifact.chunk",
        "frequency_unit": mesh_data.metadata.get("frequency_unit", "THz"),
        "energy_unit": mesh_data.metadata.get("energy_unit", "eV"),
        "coupling_unit": mesh_data.metadata.get("coupling_unit", "eV"),
        "coupling_strength_unit": mesh_data.metadata.get("coupling_strength_unit", "eV^2"),
        "artifact_axis": axis,
        "artifact_chunk": chunk_metadata,
        "base_mesh_schema": mesh_data.metadata.get("schema"),
    }


def _normalize_chunk_axis(axis: str) -> str:
    if not isinstance(axis, str):
        raise ValueError("axis must be 'k' or 'q'.")
    axis = axis.lower()
    if axis not in {"k", "q"}:
        raise ValueError("axis must be 'k' or 'q'.")
    return axis


def _validate_chunk_size(chunk_size: int) -> int:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    return int(chunk_size)


def _validate_artifact_filename(filename: str, name: str) -> None:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{name} must be a non-empty relative filename.")
    path = Path(filename)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"{name} must be a relative filename inside the artifact directory.")
    if path.name != filename:
        raise ValueError(f"{name} must not contain directory components.")


def _chunk_bounds_from_spec(axis: str, spec: dict) -> tuple:
    chunk_index = _chunk_index_from_spec(axis, spec)
    if axis == "k":
        return chunk_index, spec["k_start"], spec["k_stop"]
    return chunk_index, spec["q_start"], spec["q_stop"]


def _chunk_index_from_spec(axis: str, spec: dict) -> int:
    try:
        chunk_index = spec["chunk_index"]
    except KeyError:
        raise ValueError("Chunk spec requires chunk_index.") from None
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("Chunk spec chunk_index must be a non-negative integer.")
    if axis == "k":
        EPCKChunkSpec(chunk_index=chunk_index, k_start=spec.get("k_start"), k_stop=spec.get("k_stop"))
    else:
        EPCQChunkSpec(chunk_index=chunk_index, q_start=spec.get("q_start"), q_stop=spec.get("q_stop"))
    return chunk_index
