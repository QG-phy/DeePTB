# EPC Example Data

This directory contains the small Graphene assets used by the EPC notebooks so
the examples do not depend on machine-local paths or environment variables.

## Contents

- `skf/matsci-0-3/C-C.skf`: carbon Slater-Koster table from the matsci-0-3
  dataset. The original license is included at `skf/matsci-0-3/LICENSE`.
- `graphene/phonons/phonopy_disp.yaml` and `graphene/phonons/FORCE_SETS`:
  phonopy inputs from the dftbephy Graphene example.
- `graphene/reference/reference.npz`,
  `graphene/reference/derivatives.npz`, and
  `graphene/reference/alignment_reference.npz`: dftbephy Graphene reference
  data used only by `04_graphene_reference_alignment.ipynb`. The
  `alignment_reference.npz` file contains the small subset of arrays extracted
  from the original dftbephy HDF5 benchmark needed by the notebook.

The dftbephy license is included at
`graphene/reference/DFTBEPHY_LICENSE.txt`.
