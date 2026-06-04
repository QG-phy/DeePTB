# DeePTB EPC 下一阶段开发计划

## Summary

当前 EPC v1 已经完成核心 coupling、NPZ 数据契约、`dptb eph` 基础 CLI、linewidth、relaxation time、基础 SERTA transport、有限差分 velocity bridge 和 degenerate-subspace diagnostic。本轮扩展已经进一步补上 DeePTB-native path workflow、mesh workflow、mesh/path linewidth、mesh/path relaxation-time，以及 serial k-point chunk execution 的第一版。

下一阶段的目标不是复刻 dftbephy 的 DFTB+ workflow 外壳，而是把 EPC 能力扩展成 DeePTB-native 的稳定工作流。开发节奏应分成两个层次：

- 第一层：把已经落地的 v1/path/mesh/chunk 能力做稳，确保 API、NPZ schema、CLI、文档和默认测试可以 review/merge。
- 第二层：在稳定 v1 基础上继续扩展 transport/mobility、SCC EPC design、advanced physics 和 scaling backend。

下一阶段的开发主线是：

1. 完成 v1 release hardening：默认 CI fixture、文档收敛、错误信息和 API 稳定性。
2. 稳定 DeePTB-native path workflow：沿 q path 输出 EPC、linewidth、relaxation-time 的可视化数据，并规划 k-path + fixed-q 后续切片。
3. 稳定 DeePTB-native mesh workflow：面向 fine k/q mesh 的批量 EPC、linewidth、relaxation-time 和 transport 输入输出组织。
4. 补齐 transport/mobility 的完整单位和物理量定义。
5. 设计 SCC EPC，不在没有公式和 reference 的情况下直接实现。
6. 把单位、k/q sampling、occupation、velocity provider、executor/backend 等接口对齐到 DeePTB 现有代码，而不是在 EPC 层重复造一套平行体系。
7. 按需推进 SOC/spinful、polar correction、full gauge tracking 和性能优化。

## Design Position

- DeePTB EPC 不计算 phonons，只读取外部 phonon mode data。
- DeePTB EPC 主数据格式继续使用 NPZ；HDF5/dftbephy 只作为 benchmark/reference reader。
- 不复刻 dftbephy 的 `init/bands/epc/ephline/rtline` 目录布局、DFTB+ 文件生态或 HDF5/JSON 字段级格式。
- 需要吸收 dftbephy 中有物理和用户价值的能力切片：path workflow、mesh workflow、mobility workflow、reference validation、mode-resolved scattering analysis。
- 尽量不修改 `OrbitalMapper`、Hamiltonian transform、SCC engine、charge update、Hubbard U 等核心代码；需要桥接信息时优先新增 EPC 层 adapter 或 provider。
- 所有新 workflow 都应先有 Python API，再接 CLI；CLI 不应成为核心逻辑所在。

## Decisions for the Next Development Wave

这些决策是下一阶段实现时的约束，不只是建议：

- Development reference:
  - 开发阶段可以继续使用完整 Graphene reference，并且可以不进入 git 追踪。
  - 当前阶段允许测试里临时硬编码 Graphene reference 路径或数据条件，但必须保留 `TODO(epc-fixture)` 标记。
  - 合并/release 前需要把默认测试收敛成轻量、self-contained、可进入仓库的 fixture；完整 Graphene 只作为 opt-in benchmark。

- Data contract:
  - DeePTB EPC 公共数据契约是 NPZ，不迁移到 dftbephy HDF5。
  - HDF5 reader 只作为 benchmark/reference bridge，不进入核心 workflow contract。
  - 新增 path、mesh、transport 数据对象时必须保留 schema metadata、unit metadata、shape validation 和 pickle-free loading。

- Repo compatibility:
  - 单位常量收敛到 `dptb.utils.constants` 或未来的 `dptb.utils.units`。
  - k/q path 和 mesh 复用 `dptb.kpoints.*`。
  - occupation、Fermi level、smearing 复用已有 postprocess/utils 路径。
  - velocity 不再只依赖有限差分；必须设计 `finite_difference` 和 `hamiltonian_derivative` 两类 provider。

- Scaling boundary:
  - 当前不添加 `mpi4py` 为必需依赖，也不优先写 CUDA 专用 workflow。
  - 先实现 serial executor + deterministic chunk/reducer API。
  - MPI/multiprocessing 是外层 executor，CUDA/torch 是内层 backend；两者不冲突，也不应互相污染接口。
  - 当前实现目标是 serial k-chunk reference path；MPI/CUDA 只通过接口预留进入，不在本轮变成必需依赖或默认路径。

## Stage Gates

EPC 后续开发按 gate 推进，避免在 v1 未稳定时过早扩散：

- Gate A: v1/path/mesh stabilization
  - 当前已实现的 coupling、path、mesh、linewidth、relaxation-time、transport v1、subspace diagnostic、serial k-chunk API 必须完成导出检查、测试、文档和一次 checkpoint commit。
  - 默认测试必须 self-contained；完整 Graphene reference 保持 opt-in，可不进入 git。
  - 所有临时 hardcoded Graphene reference 或开发期路径必须带 `TODO(epc-fixture)`。

- Gate B: transport/mobility completion
  - 先统一速度 provider、单位和 normalization，再扩展 mobility 输出。
  - `finite_difference` velocity 保留为 fallback/reference。
  - `hamiltonian_derivative` velocity 复用 `get_hk(..., with_derivative=True)` 相关路径，但必须先审查 gauge、`2*pi` 因子、overlap correction 和单位 convention。

- Gate C: SCC EPC design
  - 先完成 SCC EPC design doc 和 reference strategy，再实现。
  - 没有 charge response 公式和 reference 前，`use_scc=True` 继续保持 unsupported。

- Gate D: parity and advanced physics
  - 只吸收 dftbephy 中对 DeePTB 用户有价值的能力切片，例如 mode-resolved scattering、path/mesh summaries、mobility scans、reference benchmarks。
  - 不复刻 DFTB+ 生态、目录工作流或 HDF5 字段级 contract。

- Gate E: scaling
  - serial executor 是 reference。
  - chunk spec 和 reducer 先稳定，再接 multiprocessing/MPI。
  - backend/kernel 再决定是否使用 torch CUDA。

## Repository Compatibility Checklist

后续 EPC 开发必须和 DeePTB 现有架构保持兼容，尤其是以下接缝：

- Existing reusable modules:
  - Units/constants: `dptb.utils.constants`，必要时新增 `dptb.utils.units`。
  - K-point paths: `dptb.kpoints.path`。
  - K-point meshes/reduction: `dptb.kpoints.mesh` 和 `dptb.kpoints.reduction`。
  - Fermi level and smearing: `dptb.postprocess.unified.utils` 和 `dptb.utils.occupy`。
  - Structure/data conversion: `AtomicData.from_ase(...)`、`AtomicData.to_AtomicDataDict(...)`、`TBSystem.set_atoms(...)`。
  - Unified property/accessor style: `dptb.postprocess.unified.properties.*`。
  - CPU task count convention: `dptb.utils.multiprocessing.num_tasks`。
  - Small tensor batching helpers: `dptb.utils.batch_ops` where applicable。
  - Existing tests and fixtures under `dptb/tests/` should be reused before adding new ad-hoc fixture conventions。

- `TBSystem` accessor pattern:
  - EPC 继续挂在 `TBSystem.eph`，和 `system.band`、`system.dos`、`system.export` 等 unified postprocess accessor 保持一致。
  - 新能力优先放在 `dptb/postprocess/unified/eph/` 内，由 `EPhAccessor` 或独立 helper 暴露，不把 workflow 逻辑散落到 entrypoint。
  - 新 public API 需要同时从 `dptb.postprocess.unified.eph` 和 `dptb.postprocess.unified` 导出，并添加 import smoke test。

- `TBSystem` electronic API:
  - 电子结构调用优先使用 `TBSystem.get_eigenvalues(...)`、`TBSystem.get_eigenstates(...)`、`TBSystem.get_hk(...)`。
  - 必须保留 `solver`、`nk`、`ill_threshold`、`ill_pad_value` 等现有 solver options 的透传空间，避免 EPC workflow 把 solver 行为写死。
  - 需要注意 `TBSystem` 返回值可能是 torch tensor、nested tensor 或 numpy-like object；EPC 层继续用薄 adapter 做 shape/finite validation。
  - `set_atoms(...)` 会重置 band/DOS/SCC state；EPC finite-difference provider 修改结构时必须保证 `finally` 恢复原结构。

- SCC boundary:
  - 当前 `TBSystem` 已有 `enable_scc/run_scc/get_hk(use_scc=...)` 语义，但 `get_hk(..., with_derivative=True, use_scc=True)` 明确不支持。
  - EPC 后续不能只把 `use_scc=True` 传入现有接口就宣称 SCC EPC；必须处理 SCC charge response 和 derivative 定义。
  - SCC EPC 如需改 `SKSCC`、charge update、Hubbard U 或 SCC Hamiltonian assembly，应先写 design doc/ADR。

- K-point infrastructure:
  - Path workflow 应复用 `dptb.kpoints.path` 的 `ase_kpath`、`abacus_kpath`、`vasp_kpath` 或其 convention，不另造不兼容的 path label/coordinate 格式。
  - Mesh workflow 应复用 `dptb.kpoints.mesh` 的 `kmesh_sampling`、`monkhorst_pack_sampling`、`time_symmetry_reduce` 等函数。
  - 如需要 symmetry-reduced weights，应优先复用 `dptb.kpoints.reduction`，并在 EPC metadata 中保存 reduced/full mesh convention。
  - k/q points 在 EPC NPZ 中继续使用 fractional reciprocal coordinates；如引入 Cartesian/SI velocity，必须在 metadata 中明确转换来源。

- Occupation, Fermi level and smearing:
  - transport/mobility 中的 Fermi-Dirac、Gaussian smearing、Fermi level 估计应优先复用 `dptb.postprocess.unified.utils` 或 `dptb.utils.occupy`。
  - 如果 EPC analysis 需要 Bose/Fermi helper，应避免和已有 Fermi helper 产生符号约定冲突；新增 helper 必须说明输入是 `(E-mu)/kBT`、`kBT` eV 还是 temperature K。
  - transport workflow 要明确 `temperature` 参数到底是 K 还是 `kBT` eV；如果和现有 postprocess convention 不同，必须用参数名区分。

- Data and orbital mapping:
  - 不扩大基于 `atom_orbs` 字符串的生产路径依赖；优先从 structured calculator metadata、`system.atomic_symbols`、`calculator.get_orbital_info()` 和 EPC adapter 获取 atom-resolved orbital slices。
  - 不修改 `OrbitalMapper` 或 Hamiltonian transform 来适配 EPC，除非已有设计证明这是唯一可维护方案。
  - 新 data object 必须保持 pickle-free NPZ loader、schema metadata、shape/finite/non-empty validation。
  - 结构更新和 supercell finite difference 应走 `TBSystem.set_atoms(...)` / `AtomicData` conversion 的现有路径，避免绕过 cutoff、type mapper、overlap override 等初始化逻辑。

- Units and constants:
  - 仓库已有 `dptb/utils/constants.py`，包含 `Bohr2Ang`、`Harte2eV`、`Ryd2eV`、`eV2J`、`kB_eV_per_K` 等单位常量；EPC 后续不应在各模块继续散落新的物理常量。
  - EPC 使用的 `THZ_TO_EV`、`HBAR_EV_S`、`EPC_PREFAC_AMU_THZ` 已集中到 `dptb.utils.constants`；后续新增 transport SI conversion constants 也应继续放在统一位置。
  - 如果 `constants.py` 已经过于混杂，可以新增 `dptb/utils/units.py` 专门放单位转换，并从 `constants.py` re-export 兼容旧用法；不要在 EPC 模块中重复从 `scipy.constants` 现场拼公式。
  - 所有 persistent metadata 里的单位字符串应使用统一常量或 helper 生成，避免 `"THz"`、`"eV"`、`"angstrom"`、`"Ang"` 多种拼写漂移。

- Batching and scaling:
  - Mesh/path EPC 的 chunking 应参考 `dptb.postprocess.unified.properties.optical_conductivity` 中按 k-point batch 调用 `TBSystem.get_eigenstates(...)` 的模式。
  - 并行 worker 数不要自己读 `os.cpu_count()`；如需 CPU 并行，优先使用 `dptb.utils.multiprocessing.num_tasks()` 的仓库约定。
  - 大 mesh 输出策略应先定义 data object 和 metadata，再决定是否使用 chunked artifact；不要把内存策略隐含在 CLI 参数里。
  - 当前阶段不强制引入 `mpi4py`，但 mesh/path/transport API 必须保留 chunk spec、worker/rank-independent task spec 和 deterministic reduce 入口，避免后续 MPI 化时重写核心逻辑。
  - MPI 与 CUDA 不应设计成二选一。外层 executor 负责分发独立 chunks，内层 backend 负责单个 chunk 的数值 kernel；默认 serial executor + numpy/torch CPU backend，后续可插拔 multiprocessing、`mpi4py` 和 torch CUDA backend。
  - 可并行维度应优先按物理独立性切分：EPC coupling 按 q-point 或 k-point blocks，linewidth/scattering 按 initial k/band blocks，transport/mobility 按 chemical potential、temperature 和 k blocks。
  - 每个 chunk 输出必须包含 enough metadata：global shape、chunk indices、k/q indices、band indices、weights、units、schema version，便于后续 MPI ranks 写入后合并。
  - reduction 操作必须明确是 sum、average 还是 concat；不能依赖文件读取顺序或 rank 顺序。
  - CUDA 化只应发生在 backend/kernel 层，例如 batched eigensolve、velocity matrix elements、EPC contraction；不能让 CUDA device assumptions 泄漏到 NPZ data contract 或 CLI。

- Velocity and k-derivatives:
  - 仓库已有解析 k-derivative 路径：`HR2HK(derivative=True)` 会写出 `AtomicDataDict.HAMILTONIAN_DERIV_KEY` / `OVERLAP_DERIV_KEY`，`DeePTBAdapter.get_hk(..., with_derivative=True)` 会返回 `(Hk, dHdk, Sk, dSdk)`。
  - `dptb.postprocess.unified.properties.optical_conductivity` 已经使用该路径计算 velocity-like matrix elements；transport/mobility 后续应优先复用或抽象这个 provider，而不是只依赖 finite-difference band velocity。
  - 在把该路径用于 EPC transport 前，必须先审查 gauge、`2*pi` 因子、fractional-vs-Cartesian k derivative、overlap velocity correction 和单位 metadata；不能默认把当前 finite-difference velocity 与 `dH/dk` velocity 混为同一单位。
  - 后续 velocity provider 应至少包含 `finite_difference` 和 `hamiltonian_derivative` 两种来源，并在 `TransportData` / `MobilityData` metadata 中记录来源、单位和 convention。

- CLI compatibility:
  - `dptb/entrypoints/main.py` 已经集中注册 subcommands；新增 EPC task 应扩展现有 `dptb eph --task ...`，除非有明确理由新增顶层命令。
  - CLI 参数命名应和现有 `train/test/run/export` 风格相容，避免同一概念多套名字。
  - CLI 只负责解析、加载、调用 Python API 和保存结果；不得把核心公式写进 entrypoint。

- Documentation and tests:
  - EPC 用户文档应考虑加入 docs index，避免只存在孤立 Markdown。
  - 新功能默认测试必须自包含；完整 Graphene/dftbephy reference 继续 opt-in。
  - 如果修改 phase convention、Fourier projection、finite difference provider、overlap correction 或 unit prefactor，必须跑 slow Graphene supercell FD reference。

## Non-Goals

- 不实现 DeePTB phonon solver。
- 不生成 force constants、forces、stress 或 repulsive potential。
- 不复刻 DFTB+ 输入输出文件兼容。
- 不承诺 dftbephy HDF5/JSON 字段级兼容。
- 不在当前阶段直接开启 SCC EPC 计算；SCC 必须先完成设计和 reference 对齐。
- 不把完整 SI mobility workflow 混同于当前 v1 的基础 SERTA conductivity。

## Workstream 0: Release Hardening for EPC v1

### Goals

把当前 v1 从“开发完成”收敛到“可以 review/merge/release”的状态。

### Tasks

- 设计轻量 in-repo EPC fixture，替代或补充外部 Graphene reference 在默认 CI 中的角色。
- 保留完整 Graphene reference 作为 opt-in benchmark，不进入 git 追踪。
- 梳理 `docs/epc_v1_workflow.md`，确保所有 CLI 示例和 NPZ schema 与当前实现一致。
- 增加 public API import smoke tests，避免未来导出路径断裂。
- 检查所有 EPC NPZ loader 是否拒绝 pickle/object arrays、空数组、非有限值和 schema metadata conflict。
- 统一 error message 风格：输入 shape、单位、SCC unsupported、phonon boundary 等错误需要可诊断。
- 审查 `CONTEXT.md` 和 docs index 是否需要加入 EPC 文档入口。

### Acceptance

- `uv run pytest ./dptb/tests/` 通过。
- 默认 EPC tests 不依赖外部 checkout。
- Opt-in Graphene coupling reference 通过。
- 修改 phase/Fourier/FD/overlap/prefactor 时，slow Graphene supercell FD reference 通过。
- `docs/epc_v1_workflow.md` 能作为 v1 用户入口文档。

## Workstream 1: DeePTB-Native EPC Path Workflow

### Goals

提供沿 k/q path 的 EPC 和后处理数据，覆盖 dftbephy `ephline/rtline` 中有用户价值的能力，但使用 DeePTB-native API 和 NPZ contract。

### Proposed APIs

- `EPCPathData`
  - path kpoints / qpoints
  - path coordinate / labels / segments
  - selected bands
  - frequencies
  - eigenvalues
  - coupling matrix / strength or selected summaries
  - metadata
- `compute_epc_path(...)`
  - 输入 DeePTB system、external `Phonons` path data、k path、bands、derivative provider。
  - 输出 `EPCPathData`。
- `compute_linewidth_path(...)`
  - 从 path EPC data 计算 path linewidth。
- `compute_relaxation_time_path(...)`
  - 从 path linewidth 计算 path relaxation time。

### Current Implementation Status

- Chosen data contract: standalone `EPCPathData` wrapping the same EPC arrays as `EPCData` plus path metadata.
- Implemented first Python API slice: `TBSystem.eph.compute_path(...)` for fixed electronic k-points plus external q-path phonons.
- Implemented first CLI slice: `dptb eph --task path-coupling` writes `EPCPathData` NPZ for the same fixed-k + q-path workflow.
- Implemented path postprocess slice:
  - `compute_linewidth_path(...)` keeps the q-path axis and writes `LinewidthPathData` contributions.
  - `compute_relaxation_time_path(...)` keeps the q-path axis and writes `RelaxationTimePathData`.
  - `dptb eph --task path-linewidth` and `dptb eph --task path-relaxation-time` are available.
- Implemented NPZ roundtrip fields:
  - `path_axis`
  - `path_coordinates`
  - optional `path_segments`
  - path labels in JSON metadata
- Current path coordinate convention is cumulative distance in fractional reciprocal coordinates.
- Current path linewidth convention is per-path-point contribution; summing the q-path axis reproduces the existing total linewidth for the same `EPCPathData`.
- Current limitation: `path_axis="q"` only; k-path + fixed q remains future work.

### CLI

- `dptb eph --task path-coupling` 已支持 fixed k 点 + q path coupling。
- `dptb eph --task path-linewidth` 已支持从 `EPCPathData` 计算 q-path-resolved linewidth contribution。
- `dptb eph --task path-relaxation-time` 已支持从 `LinewidthPathData` 计算 q-path-resolved relaxation time。
- 输出优先 NPZ；可选提供 JSON summary 或 plot helper，但不要求兼容 dftbephy JSON。

### Open Design Questions

- k-path + fixed-q 是否独立实现为 `path_axis="k"`，还是先通过 mesh/path hybrid workflow 表达。
- 如果未来同时支持 k-path 和 q-path，二者是否允许不同长度；如果允许，如何定义最终 shape。
- path labels/segments 是否应使用 existing band plotting utilities 的 convention。

### Acceptance

- 最小 fake-system path coupling test。
- Path metadata roundtrip test。
- Path linewidth/relaxation-time manual reference test。
- Graphene path opt-in reference 或至少 Graphene smoke test。
- 文档包含 path input/output 示例。
- Public export smoke test 覆盖 `EPCPathData`、`LinewidthPathData`、`RelaxationTimePathData`。

## Workstream 2: DeePTB-Native EPC Mesh Workflow

### Goals

支持 fine k/q mesh 上的 EPC、linewidth、relaxation-time 和 transport 工作流，为后续 mobility 和性能优化打基础。

### Proposed APIs

- `EPCMeshSpec`
  - k mesh / q mesh 或显式 kpoints/qpoints
  - symmetry settings
  - band selection
  - chunk size
  - output policy
- `compute_epc_mesh(...)`
  - 支持 chunked k/q 计算，避免一次性占满内存。
  - 支持 incremental NPZ 或分块 artifact 策略。
- `compute_linewidth_mesh(...)`
  - 支持 mode-resolved 和 band-resolved linewidth。
- `compute_relaxation_time_mesh(...)`
  - 支持 mesh relaxation time。

### Current Implementation Status

- Implemented first mesh data contract:
  - `EPCMeshSpec` accepts explicit k-points or generated `k_mesh`.
  - `q_mesh` is validation/metadata only; external `Phonons.qpoints` remains authoritative.
  - `EPCMeshData` stores EPC arrays plus normalized k/q weights.
- Implemented first Python API slice:
  - `TBSystem.eph.compute_mesh(...)` computes serial full-mesh EPC by reusing existing `compute_coupling(...)`.
  - `mesh_spec.chunk_size` now drives serial k-point chunk execution and deterministic k-axis concatenation.
  - Chunk metadata records chunk index and global `[k_start, k_stop)` ranges.
- Implemented mesh postprocess slice:
  - `compute_linewidth_mesh(...)` computes mesh linewidths from `EPCMeshData` and preserves k-point weights.
  - `compute_relaxation_time_mesh(...)` computes mesh relaxation times from `LinewidthMeshData`.
  - q-point weights are normalized and used in the q summation; uniform q weights reproduce the existing total linewidth convention.
- Implemented first CLI slice:
  - `dptb eph --task mesh-coupling`
  - `dptb eph --task mesh-linewidth`
  - `dptb eph --task mesh-relaxation-time`
  - `--k-mesh n1 n2 n3`
  - optional `--q-mesh n1 n2 n3` for external phonon q-point validation
  - optional `--time-reversal` for generated k-mesh reduction
  - optional `--chunk-size n` for serial k-point chunk execution
- Current limitation: no chunked artifact/reducer yet; chunked mesh coupling still returns a single in-memory `EPCMeshData`.
- Current executor boundary:
  - `EPCKChunkSpec` defines rank-independent k-axis chunk metadata.
  - `build_k_chunk_specs(...)` creates deterministic serial/future-parallel task specs.
  - `concat_epc_k_chunks(...)` performs deterministic k-axis concatenation and rejects inconsistent q-points, bands, frequencies, or coupling trailing shapes.
  - This is an API boundary for future multiprocessing/MPI; it is not yet a chunked artifact format.

### CLI

- `dptb eph --task mesh-coupling` 已支持 serial full-mesh EPC。
- `dptb eph --task mesh-linewidth` 已支持从 `EPCMeshData` 计算 mesh linewidth。
- `dptb eph --task mesh-relaxation-time` 已支持从 `LinewidthMeshData` 计算 mesh relaxation time。

### Open Design Questions

- 大 mesh 数据是否仍用单个 NPZ，还是引入 chunked NPZ/目录 artifact。
- 是否在当前阶段使用 crystal symmetry 降低 q mesh，还是先只对 electronic k mesh 使用已有 reduction。
- 是否需要 lazy loading 的 data object，避免载入完整 coupling matrix。
- 是否需要 summary-only mesh workflow，直接输出 linewidth/transport summary 而不持久化完整 `nq * nk * nmodes * nbands^2` coupling matrix。

### Acceptance

- 小 mesh fake-system integration test。
- Chunked 与 non-chunked 数值一致性测试。
- Serial k-chunked 与 non-chunked coupling parity 已覆盖；chunked artifact/reducer parity 仍待实现。
- Direct executor boundary tests:
  - `build_k_chunk_specs(nk, None)` returns one full chunk.
  - `build_k_chunk_specs(nk, chunk_size)` returns deterministic non-overlapping `[k_start, k_stop)` specs.
  - invalid `chunk_size` and `nk` are rejected.
  - `concat_epc_k_chunks(...)` rejects inconsistent q-points, bands, frequencies, and coupling shapes.
- NPZ roundtrip 或 chunk artifact load test。
- 非空、shape、weights、metadata validation test。
- Graphene small mesh opt-in benchmark。

## Workstream 3: Transport and Mobility Completion

### Goals

把 v1 的基础 SERTA conductivity 扩展为完整 DeePTB transport/mobility workflow，明确单位、速度 provider、carrier density、conductivity 和 mobility 的物理定义。

### Tasks

- 定义 velocity provider interface：
  - finite-difference velocity provider 已实现为默认 transport provider。
  - analytic Hamiltonian-derivative velocity provider 已实现为 `velocity_source="hamiltonian_derivative"`，基于 `get_hk(..., with_derivative=True)`。
  - future model-native velocity provider
- 设计 SI unit conversion：
  - eigenvalue eV
  - k coordinate fractional reciprocal coordinate
  - reciprocal lattice conversion
  - velocity conversion
  - volume/area convention
  - carrier density unit
  - conductivity unit
  - mobility unit
  - all physical constants and unit aliases sourced from `dptb.utils.constants` or `dptb.utils.units`
- 新增 `MobilityData` 或扩展 `TransportData`：
  - conductivity tensor
  - mobility tensor
  - carrier density
  - chemical potentials
  - temperatures
  - unit metadata
- 支持多个 chemical potentials / temperatures 的 vectorized workflow。
- 明确 2D material 的 area normalization 与 3D volume normalization 区别。

### CLI

- `dptb eph --task transport` 已支持 `--velocity-source finite_difference|hamiltonian_derivative`。
- 后续扩展 `dptb eph --task transport` 或新增 `--task mobility`。
- 支持 `--chemical-potentials`、`--temperatures`、`--dimension 2d/3d`、`--area`、`--volume`。

### Current Implementation Status

- Implemented provider selection in `compute_serta_transport_from_epc(...)`:
  - `velocity_source="finite_difference"` keeps the existing central finite-difference behavior and metadata.
  - `velocity_source="hamiltonian_derivative"` computes diagonal band velocities from analytic `dH/dk` and optional `dS/dk`.
- The Hamiltonian-derivative convention is recorded as `diag_Cdagger_dH_minus_EdS_C`.
- The velocity unit remains `eV/fractional_reciprocal_coordinate`; full SI conversion remains future work.
- SCC-corrected velocity remains unsupported in v1.

### Acceptance

- Manual SI conversion tests。
- 2D/3D normalization tests。
- Multi-mu/multi-temperature shape tests。
- Finite-difference velocity unit conversion tests。
- Graphene mobility sanity benchmark。
- 文档中明确当前单位约定和限制。

## Workstream 4: SCC EPC Design

### Goals

先完成 SCC EPC 的理论和工程设计，再实现。当前不能只把 `use_scc=True` 传给 calculator，因为 SCC EPC 需要处理 charge response 对 Hamiltonian 的贡献。

### Design Tasks

- 梳理 DeePTB 当前 SCC engine、charge update、Hubbard U、overlap 和 Hamiltonian correction 的数据流。
- 梳理 dftbephy SCC reference/derivative 方案，确认它是否包含 charge response，如何保存 reference charges。
- 写出 SCC EPC 公式：
  - non-SCC term
  - overlap correction
  - SCC Hamiltonian derivative term
  - charge response / self-consistent response term
  - 是否需要 finite-difference self-consistent charges
- 设计 provider boundary：
  - `SCCEPCProvider`
  - charge-response provider
  - finite-difference SCC derivative provider
- 明确哪些核心代码不能改；如需改 SCC engine，先写 ADR。

### Reference and Tests

- 需要 SCC reference case；不能用当前 non-SCC Graphene reference 证明 SCC EPC。
- 建议先构造小体系 SCC finite-difference smoke test。
- 后续再寻找或生成 dftbephy SCC benchmark。

### Acceptance

- SCC EPC design doc。
- 至少一个 SCC reference/smoke fixture。
- `use_scc=True` 从 NotImplemented 转为受控实现时，必须有公式、数据和测试同时落地。

## Workstream 5: Advanced EPC Physics

### SOC / Spinful EPC

- 明确 spinful eigenvectors shape。
- 明确 spin degeneracy 是否仍作为 scalar metadata，还是进入 explicit spin channel。
- 明确 SOC Hamiltonian derivative 的来源。
- 设计 spin-resolved coupling data contract。

### Polar Correction

- 明确需要的外部输入：
  - Born effective charges
  - dielectric tensor
  - q direction
  - long-range correction convention
- 保持 DeePTB 不计算 phonons 的边界，只读外部 polar metadata。

### Degenerate-Band Gauge Tracking

- 当前 v1 只有 subspace Frobenius diagnostic。
- 后续如需 full gauge tracking，需要设计沿 k/q path 的 eigenvector alignment、parallel transport 或 subspace projection。
- 必须有 degenerate reference test，不能只靠随机 unitary invariance。

## Workstream 6: Analysis Kernels

### Candidate Features

- Eliashberg-like spectral function。
- Phonon DOS from external frequencies。
- Mode-resolved scattering summaries。
- Band-resolved and q-resolved scattering maps。
- Plot helper for path and mesh data。

### Position

- 这些功能按 DeePTB API 组织，不复刻 dftbephy notebook/JSON 格式。
- 优先支持从 existing NPZ data objects 计算，不重新触发 EPC coupling。

### Acceptance

- Manual numerical tests。
- NPZ input/output tests if persistent data is introduced。
- Documentation examples。

## Workstream 7: Performance and Scaling

### Goals

让 mesh EPC 和 transport 能在实际体系上运行，而不只是小测试。

### Candidate Strategies

- Vectorize contraction loops where practical。
- Add chunked k/q execution。
- Cache derivative provider results when safe。
- Use torch backend for batched linear algebra where compatible。
- Prepare rank-independent task specs so multiprocessing or MPI execution can be added after API is stable。
- Keep a serial executor as the reference implementation; optional future executors can include multiprocessing and `mpi4py`。
- Keep backend selection separate from executor selection: executor controls task distribution, backend controls numeric device/kernel。
- Add torch-CUDA backend only after serial chunked API and reducer semantics are stable。
- Keep Cython/CUDA as future option, not first response.

### Parallelization Axes

- Coupling:
  - primary split by q-point chunks when phonon modes are independent.
  - secondary split by k-point chunks when q chunks are too large.
  - derivative provider caching must be keyed by k/q chunk and structure state.
- Linewidth / scattering:
  - split by initial k-point and band groups.
  - q/mode/final-band loops reduce into absorption/emission accumulators.
  - reducers must be associative and deterministic.
- Transport / mobility:
  - split by chemical potential and temperature first when scanning many conditions.
  - split by k-point blocks for large meshes.
  - final reducer sums conductivity and carrier-density contributions with explicit k weights.
- Path workflow:
  - split by path segments only if segment metadata preserves original ordering.
  - final output should concatenate by global path index.

### Executor Design

- Define pure task functions that accept immutable specs and return arrays plus metadata.
- Keep file writing outside worker kernels unless a chunked artifact format is explicitly designed.
- Serial executor should be the default and used in tests.
- Multiprocessing/MPI executors should share the same task spec and reducer.
- Do not make `mpi4py` a required dependency for default EPC workflows.

### Backend Design

- Backend should be explicit metadata, not inferred from input arrays.
- Initial backend can be `numpy` or existing torch CPU flow.
- Torch CUDA backend should reuse DeePTB model device/dtype conventions and avoid unnecessary CPU/GPU transfers.
- Backend code should not decide chunk ownership or reduction order.
- NPZ persistence remains CPU/numpy oriented; GPU tensors must be converted at artifact boundaries.
- Analytic velocity backend should reuse `get_hk(..., with_derivative=True)` where possible.
- Finite-difference velocity remains a fallback/reference backend.

### Acceptance

- Chunked workflow memory does not scale as full `nq * nk * nmodes * nbands^2` when summaries are requested。
- Numerical parity between chunked and non-chunked small cases。
- Benchmark script for Graphene small/medium mesh。
- Serial and future parallel reducers produce bitwise-identical or tolerance-bounded results on small deterministic fixtures。

## Suggested Implementation Order

1. Verify and checkpoint the current v1/path/mesh/chunk implementation.
2. Release hardening and lightweight default fixture.
3. Executor boundary tests and public export smoke tests.
4. Transport/mobility unit design and implementation.
5. Analytic velocity provider design and implementation.
6. SCC EPC design document.
7. SCC EPC implementation only after reference is ready.
8. Advanced physics and mode-resolved analysis.
9. Scaling optimization: chunked artifacts, multiprocessing/MPI executors, then torch CUDA backend.

## Testing Strategy

- Default tests must remain self-contained and not require local dftbephy checkout.
- External Graphene reference remains opt-in through environment variables.
- Every new persistent data object needs:
  - constructor validation tests
  - NPZ roundtrip tests
  - schema metadata conflict tests
  - empty/nonfinite/shape rejection tests
- Every new CLI task needs:
  - parser test
  - fake-system workflow test
  - output NPZ load test
  - missing required input rejection test
- Every new physical formula needs:
  - manual numerical reference
  - unit metadata test
  - at least one regression or benchmark test if available

## Immediate Next Sprint

This sprint is a stabilization sprint for the current implementation, not another feature-expansion sprint.

1. Verify current exports after executor extraction:
   - `dptb.postprocess.unified.eph`
   - `dptb.postprocess.unified`
   - import smoke tests for `EPCKChunkSpec`, `build_k_chunk_specs`, and `concat_epc_k_chunks`
2. Run focused checks:
   - `git diff --check`
   - `uv run pytest dptb/tests/test_electron_phonon.py -q`
3. Add direct executor tests if they are not already present:
   - full single chunk
   - multiple deterministic chunks
   - invalid chunk settings
   - concat rejection for inconsistent chunk inputs
4. Update `docs/epc_v1_workflow.md` only if tests or exports reveal schema/API drift.
5. Create a checkpoint commit for the path/mesh/chunk extension once tests pass.
6. Start the next feature slice only after the checkpoint:
   - transport/mobility units and data contract
   - velocity provider interface
   - `hamiltonian_derivative` velocity review against existing `dH/dk` path

The immediate merge target is: stable EPC v1 plus DeePTB-native path/mesh workflows and serial k-chunk executor boundary. Transport/mobility completion and SCC EPC remain planned follow-up workstreams.
