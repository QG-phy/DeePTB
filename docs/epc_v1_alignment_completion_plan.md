# DeePTB EPC v1 对齐版收尾计划

## Purpose

本文档是当前 EPC PR 的权威收尾计划。当前分支已经具备 EPC v1 的核心
coupling、path/mesh、linewidth、relaxation-time、SERTA transport/mobility、
NPZ 数据对象、chunked artifact、notebook 示例和核心测试。下一步不是重新设计
EPC，而是在现有基础上完成 v1 对齐与发布前收敛。

EPC v1 的对齐原则是：

- 用 DeePTB 原生 API 复现 dftbephy 和论文 Graphene EPC 主链路。
- 对齐物理公式、数值 reference、Graphene 工作流和物理输出。
- 不搬运 dftbephy 的工程形态。
- 不做 dftbephy HDF5 schema、HSD 输入、CLI、目录结构或 drop-in replacement。
- DeePTB 继续使用 `TBSystem`、`Phonons`、`EPCData`、NPZ、chunked artifact 和
  notebook/API workflow。

MPI 加速属于当前唯一 EPC PR 的后续计划阶段，不是后续 PR。本计划不展开 MPI
executor 的实现细节；下一份计划会专门定义 DeePTB-native chunk executor。MPI
未来也必须复用 DeePTB 的 chunk/artifact contract，而不是复刻 `dftbephy-mpi.py`。

## Current Baseline

当前 v1 已经具备：

- 外部 phonopy/phonon mode 输入和 `Phonons` 数据对象。
- `EPhAccessor.compute_coupling(...)`、`compute_path(...)`、`compute_mesh(...)`。
- `compute_mesh_chunked_artifact(...)` 串行 streaming artifact producer。
- `EPCData`、`EPCPathData`、`EPCMeshData` 及 NPZ roundtrip。
- linewidth、mesh/path linewidth、relaxation-time、mesh/path relaxation-time。
- SERTA transport、SI mobility、chemical-potential/temperature scans。
- finite-difference 和 Hamiltonian-derivative velocity providers。
- chunked artifact 的 summary-first linewidth、transport、mobility 后处理。
- coupling summary、scattering maps、phonon DOS、Eliashberg-like diagnostic。
- Graphene/EPC notebook 示例和核心 EPC 测试收敛。

当前 v1 仍明确不支持：

- DeePTB 内部 phonon solver、force constants、forces、stress、total energy 或
  repulsive potential workflow。
- SCC-corrected EPC。
- SOC/spinful EPC。
- polar correction。
- superconducting `Tc` production workflow。
- MPI executor 的具体实现。
- dftbephy HDF5/HSD/CLI/目录结构兼容。

## Completion Gates

### 1. Graphene Reference 对标

目标是证明 DeePTB EPC 公式、单位和相位约定与 dftbephy/论文 reference 一致。

需要完成：

- 增加 opt-in reference benchmark。
- 使用外部 dftbephy Graphene reference 数据对齐：
  - `eigenvalues_k`；
  - `eigenvalues_kq`；
  - `g2` / `coupling_strength`；
  - phonopy eigenvector shape/phase convention；
  - DFTBPlus gauge convention；
  - frequency prefactor 和单位约定。
- Reference 数据保持外部化，不进入默认 CI，不成为 DeePTB 公共数据契约。

验收标准：

```bash
DEEPTB_RUN_REFERENCE_EPH=1 \
DEEPTB_EPH_REFERENCE_ROOT=<path-to-dftbephy> \
uv run pytest dptb/tests/test_epc_reference*.py -q
```

### 2. 串行 DeePTB-native 主链路收敛

目标是先把串行科学结果和 data contract 做稳，再让下一份 MPI 计划接入同一套
chunk/artifact 边界。

需要完成：

- 固化 `TBSystem.eph` API：
  - `compute_coupling(...)`；
  - `compute_path(...)`；
  - `compute_mesh(...)`；
  - `compute_mesh_chunked_artifact(...)`。
- 确认 full mesh、q chunk、k chunk、chunked artifact 后处理结果一致。
- 保持 NPZ 和 chunked artifact 为 DeePTB-native 持久化契约。
- 串行/chunk contract 必须稳定到后续 MPI 计划可以直接接入，不需要重写公共 API
  或数据 schema。

当前计划不实现 MPI/multiprocessing executor，只定义其必须接入的稳定边界。

### 3. Transport / Mobility 物理约定审计

目标是让论文 Eq.15-18 对应的后处理约定足够清楚、可测试、可复现。

需要审计：

- Gaussian / Lorentzian broadening convention。
- Bose/Fermi occupation convention。
- `THz -> eV` 和 linewidth/relaxation-time 单位。
- `tau = hbar / (2 * linewidth)` convention。
- q/k weights normalization。
- fractional reciprocal coordinate 和 `2*pi` reciprocal-cell convention。
- 2D sheet normalization 和 3D volume normalization。
- carrier density convention。
- mobility `mu = sigma / (e |n|)`。

测试策略：

- 只补高价值 numerical/manual reference 测试。
- 不恢复低价值 parser、error-message、重复 `rejects_*` 测试。
- EPC 测试的目标是保护数值和物理契约，不是锁死实现细节。

### 4. Graphene Notebook 对齐

目标是让用户可以通过 DeePTB 原生 notebook/API workflow 产出 dftbephy/论文同类
结果。

需要完成：

- 现有 `examples/EPC` notebook 继续走 DeePTB API。
- 调整或增加 notebook，使其覆盖：
  - Graphene EPC reference 对比；
  - path EPC、linewidth、relaxation-time；
  - mesh linewidth、transport、mobility；
  - scattering maps、phonon DOS、Eliashberg-like diagnostics。
- Notebook 不输出 dftbephy HDF5，不要求 dftbephy 工程兼容。
- Reference 数据缺失时给出清楚提示，而不是让 notebook 静默产生错误结果。

### 5. 文档冻结

目标是把 v1 的能力和边界写清楚，避免继续扩大 scope。

文档必须明确 v1 支持：

- non-SCC EPC；
- external phonopy/phonon mode 输入；
- finite-difference H/S derivatives；
- path/mesh EPC；
- linewidth / relaxation-time；
- SERTA transport / mobility；
- NPZ / chunked artifact；
- Graphene reference benchmark。

文档必须明确当前串行对齐阶段不展开：

- SCC EPC；
- SOC/spinful EPC；
- polar correction；
- superconducting `Tc` production workflow；
- MPI executor 具体实现。

文档必须明确永久 non-goals：

- dftbephy HDF5 schema 兼容；
- dftbephy HSD 输入兼容；
- dftbephy CLI 兼容；
- dftbephy 输出目录结构兼容；
- DeePTB 作为 dftbephy drop-in replacement。

## Relationship To MPI Plan

当前仓库只维护一个 EPC PR；多个计划表示同一 PR 内的不同开发阶段，不是多个 PR。

MPI 加速会在下一份计划中设计和实现，但仍属于当前 PR 的完善工作。下一份 MPI
计划必须遵守以下约束：

- MPI 是 DeePTB-native chunk executor。
- MPI 不改变 EPC 数值内核。
- MPI 不改变 NPZ 或 chunked artifact schema。
- MPI 不引入 dftbephy HDF5/HSD/CLI 兼容层。
- `mpi4py` 必须是 optional dependency。
- 默认测试和默认安装不得依赖 MPI runtime。

## Test And Review Plan

默认核心测试：

```bash
uv run pytest dptb/tests/test_epc_*.py -q
```

Graphene reference opt-in：

```bash
DEEPTB_RUN_REFERENCE_EPH=1 \
DEEPTB_EPH_REFERENCE_ROOT=<path-to-dftbephy> \
uv run pytest dptb/tests/test_epc_reference*.py -q
```

全仓最终回归：

```bash
uv run pytest ./dptb/tests/ -q
```

CodeRabbit 分目录 review：

```bash
coderabbit review --agent --base main --dir dptb/postprocess/unified/eph -c AGENTS.md
coderabbit review --agent --base main --dir dptb/tests -c AGENTS.md
coderabbit review --agent --base main --dir examples/EPC -c AGENTS.md
```

## Assumptions

- 本文档是当前 PR 当前阶段的权威计划。
- 旧计划文档保留为历史和背景，不删除。
- dftbephy 只作为物理公式和数值 reference，不作为 DeePTB API/IO/CLI contract。
- 当前计划不实现 MPI；但 MPI 是当前 PR 的下一计划阶段，不是后续 PR。
- DeePTB EPC v1 先证明串行科学结果，再在同一 PR 后续阶段接入 MPI 加速。
