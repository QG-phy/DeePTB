from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from dptb.postprocess.unified.eph.data import EPCData


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
    if chunk_size is None or chunk_size >= nk:
        return [EPCKChunkSpec(chunk_index=0, k_start=0, k_stop=nk)]
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    chunk_size = int(chunk_size)
    return [
        EPCKChunkSpec(chunk_index=ichunk, k_start=start, k_stop=min(start + chunk_size, nk))
        for ichunk, start in enumerate(range(0, nk, chunk_size))
    ]


def build_q_chunk_specs(nq: int, chunk_size: Optional[int]) -> Sequence[EPCQChunkSpec]:
    """Build deterministic q-axis chunk specs."""
    if isinstance(nq, bool) or not isinstance(nq, (int, np.integer)) or nq <= 0:
        raise ValueError("nq must be a positive integer.")
    nq = int(nq)
    if chunk_size is None or chunk_size >= nq:
        return [EPCQChunkSpec(chunk_index=0, q_start=0, q_stop=nq)]
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    chunk_size = int(chunk_size)
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
