import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from dptb.postprocess.unified.eph.benchmark import DFTBPlusGauge
from dptb.postprocess.unified.eph.contraction import EPC_PREFAC_AMU_THZ, compute_coupling_matrix
from dptb.postprocess.unified.eph.data import (
    EPCData,
    EPCMeshData,
    EPCMeshSpec,
    EPCPathData,
    Phonons,
    cumulative_path_coordinates,
    _metadata_to_json,
)
from dptb.postprocess.unified.eph.executor import (
    EPC_MESH_CHUNKED_ARTIFACT_SCHEMA_VERSION,
    build_k_chunk_specs,
    build_q_chunk_specs,
    concat_epc_k_chunks,
    concat_epc_q_chunks,
)
from dptb.postprocess.unified.eph.providers import (
    FDProvider,
    SupercellFD,
)
from dptb.postprocess.unified.eph.utils import (
    as_array,
    normalize_integer_indices,
    normalize_kpoints,
    orbital_slices_from_system,
    strip_batch,
    validate_finite_positive_scalar,
)


class EPhAccessor:
    """Electron-phonon coupling calculations for ``TBSystem``."""

    def __init__(self, system):
        self._system = system

    def compute_coupling(
        self,
        kpoints: np.ndarray,
        phonons: Phonons,
        bands: Optional[Sequence[int]] = None,
        displacement: float = 1e-3,
        use_scc: bool = False,
        output_npz: Optional[Union[str, Path]] = None,
        derivative_provider: Optional[FDProvider] = None,
    ) -> EPCData:
        if use_scc:
            raise NotImplementedError("SCC-corrected electron-phonon coupling is not supported in v1.")
        displacement = validate_finite_positive_scalar(displacement, "displacement")

        kpoints = normalize_kpoints(kpoints)
        qpoints = phonons.qpoints
        kqpoints = (kpoints[None, :, :] + qpoints[:, None, :]).reshape(-1, 3)

        _, eigenvalues_k, eigenvectors_k = _unpack_eigenstate_payload(
            self._system.get_eigenstates(k_points=kpoints, use_scc=use_scc),
            name="system.get_eigenstates(k)",
        )
        _, eigenvalues_kq_flat, eigenvectors_kq_flat = _unpack_eigenstate_payload(
            self._system.get_eigenstates(k_points=kqpoints, use_scc=use_scc),
            name="system.get_eigenstates(k+q)",
        )

        eigenvalues_k, eigenvectors_k = _validate_eigenstate_arrays(
            eigenvalues_k,
            eigenvectors_k,
            expected_kpoints=kpoints.shape[0],
            name="system.get_eigenstates(k)",
        )
        eigenvalues_kq_flat, eigenvectors_kq_flat = _validate_eigenstate_arrays(
            eigenvalues_kq_flat,
            eigenvectors_kq_flat,
            expected_kpoints=kqpoints.shape[0],
            name="system.get_eigenstates(k+q)",
            reference_norb=eigenvectors_k.shape[-2],
            reference_nbands=eigenvectors_k.shape[-1],
        )
        eigenvalues_kq = eigenvalues_kq_flat.reshape(qpoints.shape[0], kpoints.shape[0], eigenvalues_k.shape[-1])
        eigenvectors_kq = eigenvectors_kq_flat.reshape(
            qpoints.shape[0], kpoints.shape[0], eigenvectors_k.shape[-2], eigenvectors_k.shape[-1]
        )

        if derivative_provider is None:
            derivative_provider = FDProvider(
                self._system,
                displacement=displacement,
                use_scc=use_scc,
            )
        derivative_provider_name = type(derivative_provider).__name__
        h_derivatives_k, overlap_derivatives_k = _unpack_derivative_payload(
            derivative_provider.compute(kpoints),
            name="derivative_provider.compute(kpoints)",
        )
        h_derivatives_k, overlap_derivatives_k = _validate_derivative_payload(
            h_derivatives_k,
            overlap_derivatives_k,
            expected_kpoints=kpoints.shape[0],
            name="derivative_provider.compute(kpoints)",
        )
        h_derivatives_kq_flat, overlap_derivatives_kq_flat = _unpack_derivative_payload(
            derivative_provider.compute(kqpoints),
            name="derivative_provider.compute(k+q)",
        )
        h_derivatives_kq_flat, overlap_derivatives_kq_flat = _validate_derivative_payload(
            h_derivatives_kq_flat,
            overlap_derivatives_kq_flat,
            expected_kpoints=kqpoints.shape[0],
            name="derivative_provider.compute(k+q)",
            reference_trailing_shape=h_derivatives_k.shape[1:],
        )
        h_derivatives_kq = h_derivatives_kq_flat.reshape(
            qpoints.shape[0], kpoints.shape[0], *h_derivatives_kq_flat.shape[1:]
        )
        overlap_derivatives_kq = None
        if overlap_derivatives_kq_flat is not None:
            overlap_derivatives_kq = overlap_derivatives_kq_flat.reshape(
                qpoints.shape[0], kpoints.shape[0], *overlap_derivatives_kq_flat.shape[1:]
            )

        block_phase_kwargs = self._block_phase_kwargs(phonons, h_derivatives_k)

        coupling_matrix, coupling_strength = compute_coupling_matrix(
            eigenvalues_k=eigenvalues_k,
            eigenvectors_k=eigenvectors_k,
            eigenvalues_kq=eigenvalues_kq,
            eigenvectors_kq=eigenvectors_kq,
            h_derivatives_k=h_derivatives_k,
            h_derivatives_kq=h_derivatives_kq,
            overlap_derivatives_k=overlap_derivatives_k,
            overlap_derivatives_kq=overlap_derivatives_kq,
            phonon_eigenvectors=phonons.eigenvectors,
            masses=phonons.masses,
            frequencies=phonons.frequencies,
            band_indices=bands,
            **block_phase_kwargs,
        )
        if bands is None:
            band_indices = np.arange(eigenvalues_k.shape[-1], dtype=int)
        else:
            band_indices = normalize_integer_indices(bands, "bands")

        result = EPCData(
            kpoints=kpoints,
            qpoints=qpoints,
            band_indices=band_indices,
            frequencies=phonons.frequencies,
            eigenvalues_k=eigenvalues_k[:, band_indices],
            eigenvalues_kq=eigenvalues_kq[:, :, band_indices],
            coupling_matrix=coupling_matrix,
            coupling_strength=coupling_strength,
            metadata={
                "source": "deeptb.eph.compute_coupling",
                "displacement": displacement,
                "use_scc": use_scc,
                "frequency_unit": "THz",
                "epc_prefac_amu_thz": EPC_PREFAC_AMU_THZ,
                "derivative_provider": derivative_provider_name,
                "phonon_metadata": phonons.metadata,
            },
        )
        if output_npz is not None:
            result.save_npz(output_npz)
        return result

    def compute_path(
        self,
        kpoints: np.ndarray,
        phonons: Phonons,
        bands: Optional[Sequence[int]] = None,
        displacement: float = 1e-3,
        use_scc: bool = False,
        output_npz: Optional[Union[str, Path]] = None,
        derivative_provider: Optional[FDProvider] = None,
        path_axis: str = "q",
        path_coordinates: Optional[np.ndarray] = None,
        path_segments: Optional[np.ndarray] = None,
        path_labels: Optional[Dict[str, int]] = None,
    ) -> EPCPathData:
        """Compute EPC data for a path workflow.

        The current path workflow supports the first DeePTB-native slice:
        fixed or explicit electronic k-points combined with an external q-path
        from ``phonons``. The output is an ``EPCPathData`` NPZ contract rather
        than a dftbephy HDF5/JSON layout.
        """
        if path_axis != "q":
            raise NotImplementedError("EPC path workflow currently supports path_axis='q' only.")
        epc_data = self.compute_coupling(
            kpoints=kpoints,
            phonons=phonons,
            bands=bands,
            displacement=displacement,
            use_scc=use_scc,
            derivative_provider=derivative_provider,
        )
        if path_coordinates is None:
            path_coordinates = cumulative_path_coordinates(phonons.qpoints)

        metadata = {
            "source": "deeptb.eph.compute_path",
            "path_mode": "fixed_k_q_path",
            "base_epc_metadata": epc_data.metadata,
        }
        if path_labels is not None:
            metadata["path_labels"] = _validate_path_labels(path_labels, len(path_coordinates))

        result = EPCPathData.from_epc_data(
            epc_data,
            path_axis="q",
            path_coordinates=path_coordinates,
            path_segments=path_segments,
            metadata=metadata,
        )
        if output_npz is not None:
            result.save_npz(output_npz)
        return result

    def compute_mesh(
        self,
        mesh_spec: EPCMeshSpec,
        phonons: Phonons,
        bands: Optional[Sequence[int]] = None,
        displacement: float = 1e-3,
        use_scc: bool = False,
        output_npz: Optional[Union[str, Path]] = None,
        derivative_provider: Optional[FDProvider] = None,
    ) -> EPCMeshData:
        """Compute EPC data on a DeePTB-native k/q mesh.

        This first mesh slice is a serial in-memory reference implementation.
        ``mesh_spec.chunk_size`` and ``mesh_spec.q_chunk_size`` drive
        deterministic serial chunk specs that can be reused by future
        multiprocessing/MPI executors.
        """
        if not isinstance(mesh_spec, EPCMeshSpec):
            raise ValueError("mesh_spec must be an EPCMeshSpec instance.")
        kpoints, kpoint_weights = mesh_spec.resolve_kpoints_and_weights()
        qpoint_weights = mesh_spec.resolve_qpoint_weights(phonons)
        k_chunk_specs = build_k_chunk_specs(kpoints.shape[0], mesh_spec.chunk_size)
        q_chunk_specs = build_q_chunk_specs(phonons.qpoints.shape[0], mesh_spec.q_chunk_size)
        if len(k_chunk_specs) == 1 and len(q_chunk_specs) == 1:
            epc_data = self.compute_coupling(
                kpoints=kpoints,
                phonons=phonons,
                bands=bands,
                displacement=displacement,
                use_scc=use_scc,
                derivative_provider=derivative_provider,
            )
            chunked = False
            execution = "serial_full_mesh"
            chunk_axis = None
            chunk_metadata = []
        else:
            chunked = True
            if len(k_chunk_specs) > 1 and len(q_chunk_specs) > 1:
                execution = "serial_qk_chunked"
                chunk_axis = "q,k"
            elif len(q_chunk_specs) > 1:
                execution = "serial_q_chunked"
                chunk_axis = "q"
            else:
                execution = "serial_k_chunked"
                chunk_axis = "k"
            chunk_metadata = []
            q_chunks = []
            for q_spec in q_chunk_specs:
                chunk_phonons = _slice_phonons(phonons, q_spec.slice, q_spec.metadata())
                k_chunks = []
                k_metadata = []
                for k_spec in k_chunk_specs:
                    chunk_kpoints = kpoints[k_spec.slice]
                    k_chunks.append(
                        self.compute_coupling(
                            kpoints=chunk_kpoints,
                            phonons=chunk_phonons,
                            bands=bands,
                            displacement=displacement,
                            use_scc=use_scc,
                            derivative_provider=derivative_provider,
                        )
                    )
                    k_metadata.append(k_spec.metadata())
                if len(k_chunks) == 1:
                    q_chunks.append(k_chunks[0])
                else:
                    q_chunks.append(concat_epc_k_chunks(k_chunks))
                chunk_metadata.append(
                    {
                        "q_chunk": q_spec.metadata(),
                        "k_chunks": k_metadata,
                    }
                )
            epc_data = q_chunks[0] if len(q_chunks) == 1 else concat_epc_q_chunks(q_chunks)
            if chunk_axis == "k":
                chunk_metadata = chunk_metadata[0]["k_chunks"]
            elif chunk_axis == "q":
                chunk_metadata = [entry["q_chunk"] for entry in chunk_metadata]
        result = EPCMeshData.from_epc_data(
            epc_data,
            kpoint_weights=kpoint_weights,
            qpoint_weights=qpoint_weights,
            metadata={
                "source": "deeptb.eph.compute_mesh",
                "base_epc_metadata": epc_data.metadata,
                "mesh_spec": mesh_spec.metadata_payload(),
                "execution": execution,
                "chunked": chunked,
                "chunk_axis": chunk_axis,
                "chunks": chunk_metadata,
            },
        )
        if output_npz is not None:
            result.save_npz(output_npz)
        return result

    def compute_mesh_chunked_artifact(
        self,
        mesh_spec: EPCMeshSpec,
        phonons: Phonons,
        directory: Union[str, Path],
        axis: str = "q",
        chunk_size: Optional[int] = None,
        bands: Optional[Sequence[int]] = None,
        displacement: float = 1e-3,
        use_scc: bool = False,
        derivative_provider: Optional[FDProvider] = None,
    ) -> dict:
        """Compute serial EPC mesh chunks directly into a directory artifact.

        This is the first streaming artifact producer: it writes each EPC chunk
        as it is computed and only keeps manifest metadata in memory. It reuses
        the same chunked artifact contract as
        ``save_epc_mesh_chunked_artifact(...)``.
        """
        if not isinstance(mesh_spec, EPCMeshSpec):
            raise ValueError("mesh_spec must be an EPCMeshSpec instance.")
        if use_scc:
            raise NotImplementedError("SCC-corrected electron-phonon coupling is not supported in v1.")
        if not isinstance(axis, str):
            raise ValueError("axis must be 'k' or 'q'.")
        axis = axis.lower()
        if axis not in {"k", "q"}:
            raise ValueError("axis must be 'k' or 'q'.")

        kpoints, kpoint_weights = mesh_spec.resolve_kpoints_and_weights()
        qpoint_weights = mesh_spec.resolve_qpoint_weights(phonons)
        if axis == "k":
            effective_chunk_size = chunk_size if chunk_size is not None else mesh_spec.chunk_size
            specs = build_k_chunk_specs(kpoints.shape[0], effective_chunk_size)
            effective_chunk_size = kpoints.shape[0] if effective_chunk_size is None else int(effective_chunk_size)
            reducer = "concat_epc_k_chunks"
        else:
            effective_chunk_size = chunk_size if chunk_size is not None else mesh_spec.q_chunk_size
            specs = build_q_chunk_specs(phonons.qpoints.shape[0], effective_chunk_size)
            effective_chunk_size = phonons.qpoints.shape[0] if effective_chunk_size is None else int(effective_chunk_size)
            reducer = "concat_epc_q_chunks"

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        chunks = []
        for spec in specs:
            if axis == "k":
                epc_chunk = self.compute_coupling(
                    kpoints=kpoints[spec.slice],
                    phonons=phonons,
                    bands=bands,
                    displacement=displacement,
                    use_scc=use_scc,
                    derivative_provider=derivative_provider,
                )
            else:
                epc_chunk = self.compute_coupling(
                    kpoints=kpoints,
                    phonons=_slice_phonons(phonons, spec.slice, spec.metadata()),
                    bands=bands,
                    displacement=displacement,
                    use_scc=use_scc,
                    derivative_provider=derivative_provider,
                )
            epc_chunk.metadata.update(
                {
                    "source": "deeptb.eph.compute_mesh_chunked_artifact.chunk",
                    "artifact_axis": axis,
                    "artifact_chunk": spec.metadata(),
                }
            )
            filename = f"{axis}_chunk_{spec.chunk_index:06d}.npz"
            epc_chunk.save_npz(directory / filename)
            chunks.append(
                {
                    "filename": filename,
                    "axis": axis,
                    "spec": spec.metadata(),
                }
            )

        np.savez_compressed(
            directory / "weights.npz",
            el_kpoint_weights=kpoint_weights,
            ph_qpoint_weights=qpoint_weights,
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
            "chunk_size": effective_chunk_size,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "weights_filename": "weights.npz",
            "mesh_metadata": {
                "source": "deeptb.eph.compute_mesh_chunked_artifact",
                "mesh_spec": mesh_spec.metadata_payload(),
                "execution": "serial_streaming_artifact",
                "artifact_axis": axis,
                "streaming_artifact": True,
            },
            "reducer": reducer,
        }
        (directory / "manifest.json").write_text(
            json.dumps(json.loads(_metadata_to_json(manifest)), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    def _block_phase_kwargs(self, phonons: Phonons, h_derivatives_k: np.ndarray) -> Dict[str, Any]:
        try:
            orbital_slices = orbital_slices_from_system(self._system)
        except RuntimeError:
            return {}
        if len(orbital_slices) != len(phonons.masses):
            return {}

        if phonons.scaled_positions is not None:
            scaled_positions = phonons.scaled_positions
        elif hasattr(self._system, "atoms"):
            scaled_positions = self._system.atoms.get_scaled_positions(wrap=False)
        else:
            return {}

        return {
            "qpoints": phonons.qpoints,
            "scaled_positions": scaled_positions,
            "orbital_slices": orbital_slices,
            "derivative_mode": "row" if h_derivatives_k.ndim == 4 else "full",
        }


def _unpack_derivative_payload(payload, name: str):
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError(f"{name} must return a (h_derivatives, overlap_derivatives) tuple.")
    return payload


def _slice_phonons(phonons: Phonons, q_slice: slice, chunk_metadata: Dict[str, Any]) -> Phonons:
    metadata = dict(phonons.metadata)
    metadata.update(
        {
            "source": "deeptb.eph.phonons.q_chunk",
            "base_phonon_metadata": phonons.metadata,
            "q_chunk": chunk_metadata,
        }
    )
    return Phonons(
        qpoints=phonons.qpoints[q_slice],
        frequencies=phonons.frequencies[q_slice],
        eigenvectors=phonons.eigenvectors[q_slice],
        masses=phonons.masses,
        cell=phonons.cell,
        scaled_positions=phonons.scaled_positions,
        metadata=metadata,
    )


def _unpack_eigenstate_payload(payload, name: str):
    if not isinstance(payload, (tuple, list)) or len(payload) != 3:
        raise ValueError(f"{name} must return a (metadata, eigenvalues, eigenvectors) tuple.")
    return payload


def _validate_eigenstate_arrays(
    eigenvalues,
    eigenvectors,
    expected_kpoints: int,
    name: str,
    reference_norb: Optional[int] = None,
    reference_nbands: Optional[int] = None,
):
    eigenvalues = strip_batch(as_array(eigenvalues, dtype=float))
    eigenvectors = strip_batch(as_array(eigenvectors, dtype=complex))
    if eigenvalues.ndim != 2:
        raise ValueError(f"{name} must return eigenvalues with shape (nk, nbands).")
    if eigenvectors.ndim != 3:
        raise ValueError(f"{name} must return eigenvectors with shape (nk, norb, nbands).")
    if eigenvalues.shape[0] != expected_kpoints:
        raise ValueError(f"{name} returned eigenvalues with an inconsistent k-point count.")
    if eigenvectors.shape[0] != expected_kpoints:
        raise ValueError(f"{name} returned eigenvectors with an inconsistent k-point count.")
    if eigenvalues.shape[1] == 0:
        raise ValueError(f"{name} must return at least one electronic band.")
    if eigenvectors.shape[1] == 0:
        raise ValueError(f"{name} must return at least one orbital.")
    if eigenvectors.shape[2] != eigenvalues.shape[1]:
        raise ValueError(f"{name} eigenvectors and eigenvalues must have the same band count.")
    if reference_norb is not None and eigenvectors.shape[1] != reference_norb:
        raise ValueError(f"{name} returned eigenvectors with an inconsistent orbital count.")
    if reference_nbands is not None and eigenvalues.shape[1] != reference_nbands:
        raise ValueError(f"{name} returned eigenvalues with an inconsistent band count.")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{name} eigenvalues must contain finite values.")
    if not np.all(np.isfinite(eigenvectors)):
        raise ValueError(f"{name} eigenvectors must contain finite values.")
    return eigenvalues, eigenvectors


def _validate_derivative_payload(
    h_derivatives,
    overlap_derivatives,
    expected_kpoints: int,
    name: str,
    reference_trailing_shape=None,
):
    h_derivatives = as_array(h_derivatives, dtype=complex)
    if h_derivatives.ndim not in {4, 5}:
        raise ValueError(f"{name} must return h_derivatives with shape (nk, 3, norb, norb) or (nk, natoms, 3, norb, norb).")
    if h_derivatives.shape[0] != expected_kpoints:
        raise ValueError(f"{name} returned h_derivatives with an inconsistent k-point count.")
    if reference_trailing_shape is not None and h_derivatives.shape[1:] != tuple(reference_trailing_shape):
        raise ValueError(f"{name} returned h_derivatives with an inconsistent derivative shape.")
    if h_derivatives.ndim == 4 and h_derivatives.shape[1] != 3:
        raise ValueError(f"{name} row h_derivatives must have Cartesian axis size 3.")
    if h_derivatives.ndim == 5 and h_derivatives.shape[2] != 3:
        raise ValueError(f"{name} full h_derivatives must have Cartesian axis size 3.")
    if h_derivatives.shape[-1] != h_derivatives.shape[-2]:
        raise ValueError(f"{name} h_derivatives must contain square orbital matrices.")
    if not np.all(np.isfinite(h_derivatives)):
        raise ValueError(f"{name} h_derivatives must contain finite values.")

    if overlap_derivatives is None:
        return h_derivatives, None
    overlap_derivatives = as_array(overlap_derivatives, dtype=complex)
    if overlap_derivatives.shape != h_derivatives.shape:
        raise ValueError(f"{name} overlap_derivatives must have the same shape as h_derivatives.")
    if not np.all(np.isfinite(overlap_derivatives)):
        raise ValueError(f"{name} overlap_derivatives must contain finite values.")
    return h_derivatives, overlap_derivatives


def _validate_path_labels(path_labels: Dict[str, int], npath: int) -> Dict[str, int]:
    if not isinstance(path_labels, dict):
        raise ValueError("path_labels must be a mapping from label string to path index.")
    normalized = {}
    for label, index in path_labels.items():
        if not isinstance(label, str) or not label:
            raise ValueError("path_labels keys must be non-empty strings.")
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("path_labels values must be integer path indices.")
        index = int(index)
        if index < 0 or index >= npath:
            raise ValueError("path_labels values must stay within the path point count.")
        normalized[label] = index
    return normalized


__all__ = [
    "DFTBPlusGauge",
    "EPCData",
    "EPCMeshData",
    "EPCMeshSpec",
    "EPCPathData",
    "EPhAccessor",
    "FDProvider",
    "EPC_PREFAC_AMU_THZ",
    "Phonons",
    "SupercellFD",
    "compute_coupling_matrix",
]
