from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from dptb.postprocess.unified.eph.benchmark import DFTBPlusGauge
from dptb.postprocess.unified.eph.contraction import EPC_PREFAC_AMU_THZ, compute_coupling_matrix
from dptb.postprocess.unified.eph.data import EPCData, Phonons
from dptb.postprocess.unified.eph.providers import (
    FDProvider,
    SupercellFD,
)
from dptb.postprocess.unified.eph.utils import (
    as_array,
    normalize_kpoints,
    orbital_slices_from_atom_orbs,
    strip_batch,
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

        kpoints = normalize_kpoints(kpoints)
        qpoints = phonons.qpoints
        kqpoints = (kpoints[None, :, :] + qpoints[:, None, :]).reshape(-1, 3)

        _, eigenvalues_k, eigenvectors_k = self._system.get_eigenstates(k_points=kpoints, use_scc=use_scc)
        _, eigenvalues_kq, eigenvectors_kq = self._system.get_eigenstates(k_points=kqpoints, use_scc=use_scc)

        eigenvalues_k = strip_batch(as_array(eigenvalues_k, dtype=float))
        eigenvectors_k = strip_batch(as_array(eigenvectors_k, dtype=complex))
        eigenvalues_kq = strip_batch(as_array(eigenvalues_kq, dtype=float)).reshape(
            qpoints.shape[0], kpoints.shape[0], -1
        )
        eigenvectors_kq = strip_batch(as_array(eigenvectors_kq, dtype=complex)).reshape(
            qpoints.shape[0], kpoints.shape[0], eigenvectors_k.shape[-2], eigenvectors_k.shape[-1]
        )

        if derivative_provider is None:
            derivative_provider = FDProvider(
                self._system,
                displacement=displacement,
                use_scc=use_scc,
            )
        h_derivatives_k, overlap_derivatives_k = derivative_provider.compute(kpoints)
        h_derivatives_kq_flat, overlap_derivatives_kq_flat = derivative_provider.compute(kqpoints)
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
            band_indices = np.asarray(bands, dtype=int)

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
                "displacement": displacement,
                "use_scc": use_scc,
                "frequency_unit": "THz",
                "epc_prefac_amu_thz": EPC_PREFAC_AMU_THZ,
            },
        )
        if output_npz is not None:
            result.save_npz(output_npz)
        return result

    def _block_phase_kwargs(self, phonons: Phonons, h_derivatives_k: np.ndarray) -> Dict[str, Any]:
        if not hasattr(self._system, "atom_orbs"):
            return {}

        orbital_slices = orbital_slices_from_atom_orbs(self._system.atom_orbs)
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


__all__ = [
    "DFTBPlusGauge",
    "EPCData",
    "EPhAccessor",
    "FDProvider",
    "EPC_PREFAC_AMU_THZ",
    "Phonons",
    "SupercellFD",
    "compute_coupling_matrix",
]
