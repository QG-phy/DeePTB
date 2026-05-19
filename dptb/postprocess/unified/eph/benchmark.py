from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dptb.postprocess.unified.eph.utils import as_array


@dataclass(frozen=True)
class DFTBPlusGauge:
    """Adapter for comparing DeePTB arrays against DFTB+ conventions."""

    orbital_signs: np.ndarray
    derivative_signs: np.ndarray
    conjugate_eigenvectors: bool = True

    @classmethod
    def from_atom_orbs(cls, atom_orbs: Sequence[str]) -> "DFTBPlusGauge":
        orbital_signs = np.asarray([_dftbplus_orbital_sign(label) for label in atom_orbs], dtype=float)
        derivative_signs = np.vstack(
            [
                orbital_signs,
                orbital_signs,
                np.asarray([_dftbplus_derivative_orbital_sign(label, 2) for label in atom_orbs], dtype=float),
            ]
        )
        return cls(orbital_signs=orbital_signs, derivative_signs=derivative_signs)

    def transform_eigenvectors(self, eigenvectors: np.ndarray) -> np.ndarray:
        eigenvectors = as_array(eigenvectors, dtype=complex)
        values = np.conj(eigenvectors) if self.conjugate_eigenvectors else eigenvectors
        return self._left_multiply(values, self.orbital_signs)

    def transform_matrix(self, matrix: np.ndarray, conjugate: bool = True) -> np.ndarray:
        matrix = as_array(matrix, dtype=complex)
        values = np.conj(matrix) if conjugate else matrix
        return self._sandwich(values, self.orbital_signs)

    def transform_row_derivatives(self, derivatives: np.ndarray) -> np.ndarray:
        derivatives = as_array(derivatives, dtype=complex)
        if derivatives.shape[-3] != 3:
            raise ValueError("row derivatives must have Cartesian axis at position -3.")

        out = derivatives.copy()
        for idir in range(3):
            out[..., idir, :, :] = self._sandwich(out[..., idir, :, :], self.derivative_signs[idir])
        return out

    @staticmethod
    def _left_multiply(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
        if values.shape[-2] != len(signs):
            raise ValueError("orbital signs length must match eigenvector orbital axis.")
        return values * signs.reshape((1,) * (values.ndim - 2) + (len(signs), 1))

    @staticmethod
    def _sandwich(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
        if values.shape[-2:] != (len(signs), len(signs)):
            raise ValueError("orbital signs length must match matrix axes.")
        left = signs.reshape((1,) * (values.ndim - 2) + (len(signs), 1))
        right = signs.reshape((1,) * (values.ndim - 2) + (1, len(signs)))
        return values * left * right


def _dftbplus_orbital_sign(atom_orb: str) -> float:
    orbital = atom_orb.rsplit("-", 1)[-1]
    if orbital.endswith("p_y") or orbital.endswith("p_x"):
        return -1.0
    return 1.0


def _dftbplus_derivative_orbital_sign(atom_orb: str, cartesian_direction: int) -> float:
    orbital = atom_orb.rsplit("-", 1)[-1]
    if cartesian_direction == 2 and (
        orbital.endswith("p_y") or orbital.endswith("p_z") or orbital.endswith("p_x")
    ):
        return -1.0
    return _dftbplus_orbital_sign(atom_orb)
