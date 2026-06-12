# DeePTB EPC 下一阶段开发计划

## Summary

当前 EPC v1 已经完成核心 coupling、NPZ 数据契约、`dptb eph` 基础 CLI、linewidth、relaxation time、基础 SERTA transport、有限差分 velocity bridge、Hamiltonian-derivative velocity bridge、SI mobility、mobility scan 和 degenerate-subspace diagnostic。本轮扩展已经进一步补上 DeePTB-native path workflow、mesh workflow、mesh/path linewidth、mesh/path relaxation-time、serial k/q chunk execution 第一版、chunked mesh artifact reduction，以及 coupling-summary、scattering-map、phonon-DOS 和 Eliashberg-like diagnostic analysis。

下一阶段的目标不是复刻 dftbephy 的 DFTB+ workflow 外壳，而是把 EPC 能力扩展成 DeePTB-native 的稳定工作流。开发节奏应分成两个层次：

- 第一层：把已经落地的 v1/path/mesh/chunk 能力做稳，确保 API、NPZ schema、CLI、文档和默认测试可以 review/merge。
- 第二层：在稳定 v1 基础上继续扩展 transport/mobility、SCC EPC design、advanced physics 和 scaling backend。

下一阶段的开发主线是：

1. 完成 v1 release hardening：默认 CI fixture、文档收敛、错误信息、API 稳定性，以及当前已实现 feature slices 的 checkpoint。
2. 稳定 DeePTB-native path workflow：沿 q path 输出 EPC、linewidth、relaxation-time 的可视化数据，并规划 k-path + fixed-q 后续切片。
3. 稳定 DeePTB-native mesh workflow：面向 fine k/q mesh 的批量 EPC、linewidth、relaxation-time 和 transport 输入输出组织。
4. 固化 transport/mobility 第一版：确认单位、速度 provider、2D/3D normalization、scan axes 和 metadata convention。
5. 设计 SCC EPC，不在没有公式和 reference 的情况下直接实现。
6. 把单位、k/q sampling、occupation、velocity provider、executor/backend 等接口对齐到 DeePTB 现有代码，而不是在 EPC 层重复造一套平行体系。
7. 按需推进 SOC/spinful、polar correction、full gauge tracking、mode-resolved analysis 和性能优化。

本计划的当前执行策略是：先把 serial/chunked v1 做成稳定 reference，再谈并行和 GPU。MPI/multiprocessing 与 CUDA 不是二选一；MPI/multiprocessing 属于外层 executor，负责把独立 k/q/scan chunks 分发到进程或 rank，CUDA 属于内层 backend，负责单个 chunk 内的 batched linear algebra、velocity 和 EPC contraction。当前开发波次只允许预留接口和 metadata，不引入 `mpi4py` 或 CUDA 作为默认依赖。

## Current Roadmap Status

截至当前分支，下一阶段计划已经不是从零开始，而是进入“已有能力稳定化 + 下一层能力设计”的状态：

- Done in code:
  - v1 coupling / linewidth / relaxation-time / transport / subspace diagnostic。
  - DeePTB phonon-mode NPZ reader/writer and EPC NPZ data objects。
  - q-path EPC、path linewidth、path relaxation-time。
  - mesh EPC、mesh linewidth、mesh relaxation-time。
  - serial k-chunk task specs and deterministic k-axis reducer。
  - serial q-chunk task specs and deterministic q-axis reducer。
  - finite-difference and Hamiltonian-derivative velocity providers。
  - SI mobility, 2D/3D normalization, and multi-chemical-potential / multi-temperature mobility scans。
  - coupling-strength summary, scattering proxy maps, phonon DOS, and Eliashberg-like spectral diagnostic from existing NPZ data。
  - chunked mesh artifact save/load/reduce contract。
  - chunked artifact summary-first linewidth, SERTA transport, fixed-linewidth transport scan, SI mobility, and fixed-linewidth SI mobility scan helpers。
  - explicit per-scan-point linewidth recomputation helpers for chunked-artifact transport and SI mobility scans, with CLI exposure through `--linewidth-scan-convention recompute`。
  - first serial streaming mesh artifact producer through `TBSystem.eph.compute_mesh_chunked_artifact(...)`。
- Still needs hardening before merge/release:
  - a minimal in-repo synthetic EPC fixture now covers default linewidth, coupling-contraction, coupling-summary, and scattering-map reference testing; broader FD fixtures still need release hardening。
  - opt-in full Graphene reference kept outside git for development and benchmark。
  - current public API export smoke coverage now asserts every symbol in `dptb.postprocess.unified.eph.__all__` is re-exported from `dptb.postprocess.unified`, so new public symbols must be added to the EPC namespace intentionally。
  - docs index now links the v1 workflow and SCC design docs; CLI task examples, chunk-executor public symbols, parser task choices, and public export drift checks are complete for the current checkpoint and must be rerun after any further EPC API or CLI changes。
  - unit metadata, single-point/scan transport and mobility unit metadata, linewidth/relaxation mesh/path unit metadata, temperature convention, reciprocal-cell convention, and mobility persistent unit-string validation now have focused regression coverage; keep a final physical-convention review before release。
  - artifact metadata validation now covers weights metadata JSON, weights shape/finite/non-negative/positive-sum checks, missing chunk/weights files, unsafe artifact filenames, chunk coverage against global weights, chunk payload range/metadata consistency, fixed-vs-recomputed linewidth scan convention guards, missing required array diagnostics for persistent EPC NPZ loaders, path-axis scalar-string validation, path-segment validation, representative metadata JSON scalar/object validation, representative object-array rejection under pickle-free loading, CLI array-loader missing-field diagnostics, CLI JSON numeric-array diagnostics, CLI NPZ keyed-array diagnostics, summary-loader metadata/schema rejection, strict `EPCMeshSpec` chunk-size type validation, and direct k/q chunk-spec field validation; continue strict NPZ loader audit for remaining edge cases。
  - full repo test pass has been completed for the current checkpoint after the CLI NPZ keyed-array diagnostic hardening commit: `uv run pytest ./dptb/tests/ -q` -> `917 passed, 32 skipped, 12 warnings`; rerun before final merge after any further EPC changes。
- Still design-only:
  - SCC EPC implementation; the design document now lives in `docs/epc_scc_design.md`。
  - multiprocessing/MPI executors。
  - torch CUDA EPC backend。
  - multi-axis streaming artifact production and parallel artifact writers。
  - SOC/spinful EPC, polar correction, and full degenerate-band gauge tracking。

Current sprint posture:

- The branch is in release-hardening mode, not broad parity expansion mode.
- Recent hardening checkpoints have validated required artifact files, strict mesh chunk-size typing, SCC task gating, default minimal-fixture analysis coverage, chunk-executor workflow docs/export drift, and core NPZ loader diagnostics.
- Continue with narrow validation/docs/API-stability slices; do not start MPI, CUDA, SCC EPC, SOC, or polar-correction implementation until the relevant gate is explicitly opened.

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
  - 当前已有最小 in-repo synthetic EPC fixture 覆盖默认 linewidth reference；合并/release 前还需要继续补齐 coupling/FD 相关轻量 fixture。完整 Graphene 只作为 opt-in benchmark。

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
  - 当前实现目标是 serial k/q chunk reference path、chunked artifact contract、summary-first consumers 和清晰 metadata；MPI/CUDA 只通过接口预留进入，不在本轮变成必需依赖或默认路径。
  - 下一步实现时优先把 task spec、chunk metadata、reducer 语义和 backend selection 写清楚；不要先引入 `mpi4py` 或 CUDA kernel。
  - 如果后续需要实际加速，推荐顺序是：serial chunked summaries -> chunked artifact -> multiprocessing/MPI executor -> torch CUDA backend。这样 MPI 和 CUDA 可以组合使用，而不是二选一。
  - 当前开发波次的工程判断是：先不要在默认路径里实现 MPI 或 CUDA；先把 serial/chunk contract 做到足够清晰，使后续 MPI executor 和 CUDA backend 都能接入而不改公共 API。
  - 如果必须在下一波选择一个真实加速方向，优先实现 CPU-side chunked artifact 和 multiprocessing/MPI executor，因为 EPC mesh 的第一瓶颈通常是独立 k/q/chunk 任务数量和输出规模；CUDA 放在 backend/kernel 层，等 contraction/eigensolve/velocity kernel 的数据布局稳定后再做。
  - `mpi4py` 只能作为 optional dependency；默认安装、默认测试和 NPZ 文件格式不得依赖 MPI runtime。
  - CUDA device、torch dtype/device、rank id、process id 等 runtime 信息只能进入 debug metadata 或 benchmark log，不得成为持久化 schema 的必要字段。
  - 后续若进入真实加速实现，优先顺序是：
    1. serial/chunked artifact 和 summary-first reducer 稳定；
    2. optional multiprocessing executor，复用 `dptb.utils.multiprocessing.num_tasks()`；
    3. optional `mpi4py` executor，面向 cluster runs，默认 CI 不启用；
    4. torch CUDA backend，只有 profiling 证明单 chunk kernel 是瓶颈后再实现。
  - 不允许为了实现 MPI 而改变 EPC NPZ schema、data object shape 或 physics API；不允许为了实现 CUDA 而把 device/rank/process 信息变成加载 artifact 的必要条件。

## Stage Gates

EPC 后续开发按 gate 推进，避免在 v1 未稳定时过早扩散：

- Gate A: v1/path/mesh stabilization
  - 当前已实现的 coupling、path、mesh、linewidth、relaxation-time、transport v1、subspace diagnostic、analysis helpers、serial k/q chunk API 必须完成导出检查、测试、文档和一次 checkpoint commit。
  - 默认测试必须 self-contained；完整 Graphene reference 保持 opt-in，可不进入 git。
  - 所有临时 hardcoded Graphene reference 或开发期路径必须带 `TODO(epc-fixture)`。

- Gate B: transport/mobility completion
  - 当前已有 finite-difference velocity、Hamiltonian-derivative velocity、SI mobility 和 mobility scan 第一版。
  - 下一步重点是 release hardening：单位 metadata、`temperature` convention、2D/3D normalization、reciprocal-cell `2*pi` convention、fixed-linewidth/recomputed-linewidth scan convention 和 CLI examples。
  - `finite_difference` velocity 保留为 fallback/reference。
  - `hamiltonian_derivative` velocity 复用 `get_hk(..., with_derivative=True)` 相关路径，但仍需要审查 gauge、`2*pi` 因子、overlap correction 和单位 convention。
  - 已有 `compute_serta_transport_scan(...)` 和固定 linewidth artifact scan helpers 不应被静默改成 per-scan-point linewidth；当前 recomputed-linewidth 能力必须继续通过显式函数名、CLI flag 和 metadata 区分。

- Gate C: SCC EPC design
  - SCC EPC design doc 已创建为 `docs/epc_scc_design.md`；后续需要 review 后再实现。
  - 没有 charge response 公式和 reference 前，`use_scc=True` 继续保持 unsupported。
  - 该 design doc 通过前，不允许为了“功能开关”而简单把 `use_scc=True` 透传给现有 coupling/velocity provider。
  - Entry point coverage now asserts every `dptb eph --task ...` choice rejects `use_scc=True` before task-specific dispatch.

- Gate D: parity and advanced physics
  - 只吸收 dftbephy 中对 DeePTB 用户有价值的能力切片，例如 mode-resolved scattering、path/mesh summaries、mobility scans、reference benchmarks。
  - workflow parity 不是目标；如果某个 dftbephy workflow 外壳只是包装 DFTB+ 文件、目录和 HDF5 字段，则不复刻。
  - analysis parity 只按物理问题吸收：先从 NPZ 对象提供 summary/scattering-map/DOS/Eliashberg-like diagnostic，再考虑 plot helper。
  - 不复刻 DFTB+ 生态、目录工作流或 HDF5 字段级 contract。

- Gate E: scaling
  - serial executor 是 reference。
  - chunk spec、chunked artifact 和 reducer 先稳定，再接 multiprocessing/MPI。
  - summary-first artifact consumers 先覆盖 linewidth/transport/mobility，再推进 true streaming producer。
  - backend/kernel 再决定是否使用 torch CUDA。
  - 本 gate 的完成标准不是“有并行代码”，而是 serial/chunk path 的 schema、metadata、reducer、测试和文档稳定到后续并行化不需要改公共 API。
  - 进入 multiprocessing/MPI 前必须已有：
    - immutable chunk spec；
    - deterministic reducer；
    - artifact metadata/weights validation；
    - serial parity tests；
    - worker 不共享可变 `TBSystem` state 的策略。
  - 进入 CUDA backend 前必须已有：
    - stable per-chunk array layout；
    - explicit backend metadata；
    - CPU reference for every accelerated kernel；
    - artifact boundary 的 CPU/numpy conversion；
    - profiling 证明 per-chunk compute 而非 I/O/reduction 是主要瓶颈。

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
  - 当前测试可以用 `scipy.constants` 校验 DeePTB 常量来源，但生产代码不应在 EPC 模块里重复定义同一常量。
  - transport/mobility metadata 必须明确 temperature convention；当前 EPC transport 使用 temperature as `kBT` in eV convention 时，应在参数名、metadata 或文档中持续显式化，避免和 Kelvin convention 混淆。

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
  - `finite_difference` 是 fallback/reference，不应继续作为唯一生产路径；`hamiltonian_derivative` 是 DeePTB-native 优先方向，但必须保留 overlap correction 和 unit convention 的 guard tests。
  - 任何 velocity provider 新增 backend 时，必须证明和 CPU reference 在小 fixture 上数值一致，并记录 provider/backend/convention metadata。

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
- 当前已新增最小 synthetic EPC fixture，用于默认 linewidth、coupling-contraction、coupling-summary 和 scattering-map reference regression。
- 保留完整 Graphene reference 作为 opt-in benchmark，不进入 git 追踪。
- 梳理 `docs/epc_v1_workflow.md`，确保所有 CLI 示例和 NPZ schema 与当前实现一致。
- 当前 public API import smoke tests 已覆盖主要 EPC data objects、analysis helpers、transport/mobility helpers、velocity helper 和 chunked artifact helpers，并且会自动检查 `dptb.postprocess.unified.eph.__all__` 是否完整 re-export 到 `dptb.postprocess.unified`；新增 public API 时必须先进入 EPC namespace。
- 继续检查所有 EPC NPZ loader 是否拒绝 pickle/object arrays、空数组、非有限值、非 scalar/object `metadata_json`、schema metadata conflict 和缺失 required arrays。
- 统一 error message 风格：输入 shape、单位、SCC unsupported、phonon boundary 等错误需要可诊断。
- docs index 已加入 EPC v1 workflow 和 SCC design 入口；`CONTEXT.md` 是否需要 EPC domain summary 仍可在 release review 时决定。
- 保持 `linewidth_scan_convention="fixed_linewidth"` 与 `"per_scan_point_recomputed"` 两种语义显式分离；任何 transport/mobility scan 新入口都必须记录 metadata 并有 guard test。
- 当前 hardening slice 已跑过全仓测试；后续每个实现切片和 release 前仍需重跑，并把失败项按 EPC 相关/非 EPC 相关分类处理。

### Acceptance

- `uv run pytest ./dptb/tests/` 通过。
- 默认 EPC tests 不依赖外部 checkout。
- Opt-in Graphene coupling reference 通过。
- 修改 phase/Fourier/FD/overlap/prefactor 时，slow Graphene supercell FD reference 通过。
- `docs/epc_v1_workflow.md` 能作为 v1 用户入口文档。
- `docs/index.rst` 保持包含 EPC v1 workflow 和 SCC design 入口。
- 当前所有 persistent EPC NPZ/data artifact loaders 对 metadata JSON、weights、shape、finite values 和 schema mismatch 有明确 rejection 或 documented boundary。

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
  - `dptb eph --task mesh-artifact`
  - `dptb eph --task mesh-linewidth`
  - `dptb eph --task mesh-relaxation-time`
  - `--k-mesh n1 n2 n3`
  - optional `--q-mesh n1 n2 n3` for external phonon q-point validation
  - optional `--time-reversal` for generated k-mesh reduction
  - optional `--chunk-size n` for serial k-point chunk execution
  - optional `--q-chunk-size n` for serial q-point chunk execution
  - optional `--artifact-axis q|k` for streaming chunked artifact output.
- Current limitation: serial chunked `mesh-coupling` still returns a single in-memory `EPCMeshData`; `mesh-artifact` writes a one-axis q or k chunked artifact directly, while multi-axis q/k streaming remains future work.
- Current executor boundary:
  - `EPCKChunkSpec` defines rank-independent k-axis chunk metadata.
  - `EPCQChunkSpec` defines rank-independent q-axis chunk metadata.
  - `build_k_chunk_specs(...)` creates deterministic serial/future-parallel task specs.
  - `build_q_chunk_specs(...)` creates deterministic q-axis task specs for future q-parallel execution.
  - `concat_epc_k_chunks(...)` performs deterministic k-axis concatenation and rejects inconsistent q-points, bands, frequencies, or coupling trailing shapes.
  - `concat_epc_q_chunks(...)` performs deterministic q-axis concatenation and rejects inconsistent k-points, bands, eigenvalues, or coupling trailing shapes.
  - This is an API boundary for future multiprocessing/MPI and for the current chunked artifact reducer.
  - q-axis chunks are now wired into serial `compute_mesh(...)` execution through `EPCMeshSpec.q_chunk_size`, but still return one in-memory `EPCMeshData`; `compute_mesh_chunked_artifact(...)` is the first direct one-axis streaming producer.
  - Both k-axis and q-axis chunk execution are reference serial paths today; they are intentionally structured as future multiprocessing/MPI task specs, not as parallel execution yet.
  - `save_epc_mesh_chunked_artifact(...)` and `load_epc_mesh_chunked_artifact(...)` now provide a directory artifact contract for splitting/reducing materialized `EPCMeshData`.
  - `compute_linewidth_mesh_chunked_artifact(...)` supports summary-first linewidth reduction without loading the full coupling tensor at once.
  - `dptb eph --task mesh-linewidth --epc-artifact ...` now consumes the artifact contract directly for summary-first linewidth output.
  - `compute_serta_transport_from_epc_mesh_chunked_artifact(...)`, `compute_serta_mobility_si_from_epc_mesh_chunked_artifact(...)`, and scan helpers consume the artifact contract and reuse the existing velocity providers.
  - Fixed-linewidth scan helpers compute linewidth at the first requested chemical-potential/temperature point and reuse it across the scan; recomputed-linewidth helpers recompute linewidth at every scan point.

### CLI

- `dptb eph --task mesh-coupling` 已支持 serial full-mesh EPC。
- `dptb eph --task mesh-artifact` 已支持 serial streaming chunked artifact 输出。
- `dptb eph --task mesh-linewidth` 已支持从 `EPCMeshData` 或 chunked artifact 计算 mesh linewidth。
- `dptb eph --task mesh-relaxation-time` 已支持从 `LinewidthMeshData` 计算 mesh relaxation time。

### Open Design Questions

- 大 mesh 数据是否仍用单个 NPZ，还是引入 chunked NPZ/目录 artifact。
- 是否在当前阶段使用 crystal symmetry 降低 q mesh，还是先只对 electronic k mesh 使用已有 reduction。
- 是否需要 lazy loading 的 data object，避免载入完整 coupling matrix。
- 是否需要 summary-only mesh workflow，直接输出 linewidth/transport summary 而不持久化完整 `nq * nk * nmodes * nbands^2` coupling matrix。

### Acceptance

- 小 mesh fake-system integration test。
- Chunked 与 non-chunked 数值一致性测试。
- Serial k-chunked 与 non-chunked coupling parity 已覆盖；chunked artifact/reducer parity 应继续覆盖 linewidth、transport、mobility 和 mobility scan helper。
- Direct executor boundary tests:
  - `build_k_chunk_specs(nk, None)` returns one full chunk.
  - `build_k_chunk_specs(nk, chunk_size)` returns deterministic non-overlapping `[k_start, k_stop)` specs.
  - `build_q_chunk_specs(nq, None)` returns one full q chunk.
  - `build_q_chunk_specs(nq, q_chunk_size)` returns deterministic non-overlapping `[q_start, q_stop)` specs.
  - invalid `chunk_size` and `nk` are rejected.
  - invalid `q_chunk_size` and `nq` are rejected.
  - `concat_epc_k_chunks(...)` rejects inconsistent q-points, bands, frequencies, and coupling shapes.
  - `concat_epc_q_chunks(...)` rejects inconsistent k-points, bands, eigenvalues, and coupling shapes.
- NPZ roundtrip and chunk artifact load/reduce tests。
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
  - reciprocal lattice conversion 已在 `compute_serta_mobility_si(...)` 最小切片中实现。
  - velocity conversion 已支持从 `eV/fractional_reciprocal_coordinate` 到 `m/s`。
  - volume/area convention 已支持 3D volume 和 2D sheet area normalization。
  - carrier density unit 已支持 `m^-3` / `m^-2`。
  - conductivity unit 已支持 `S/m` / sheet `S`。
  - mobility unit 已支持 `m^2/V/s`。
  - all physical constants and unit aliases sourced from `dptb.utils.constants` or `dptb.utils.units`
- 新增 `MobilityData` 或扩展 `TransportData`：
  - `MobilityData` 已新增。
  - `MobilityScanData` 已新增。
  - conductivity tensor 已支持。
  - mobility tensor 已支持。
  - carrier density 已支持。
  - chemical potential / temperature metadata 已支持单点和 scan axes。
  - unit metadata 已支持。
- 支持多个 chemical potentials / temperatures 的 Python scan workflow。
- 明确 2D material 的 area normalization 与 3D volume normalization 区别。

### CLI

- `dptb eph --task transport` 已支持 `--velocity-source finite_difference|hamiltonian_derivative`。
- transport CLI 已支持 `--chemical-potentials`、`--temperatures` scan 参数，并写出 `TransportScanData`。
- transport CLI 已支持 `--epc-artifact --linewidth-scan-convention recompute`，从 chunked artifact 进行 per-scan-point linewidth recomputation scan。
- `dptb eph --task mobility` 已支持单个 chemical potential / temperature 的 SI mobility 输出。
- mobility CLI 已支持 `--dimension 2d/3d`、`--area`、`--volume`。
- mobility CLI 已支持 `--chemical-potentials`、`--temperatures` scan 参数。
- mobility CLI 已支持 `--epc-artifact --linewidth-scan-convention recompute`，从 chunked artifact 进行 per-scan-point linewidth recomputation scan。

### Current Implementation Status

- Implemented provider selection in `compute_serta_transport_from_epc(...)`:
  - `velocity_source="finite_difference"` keeps the existing central finite-difference behavior and metadata.
  - `velocity_source="hamiltonian_derivative"` computes diagonal band velocities from analytic `dH/dk` and optional `dS/dk`.
- The Hamiltonian-derivative convention is recorded as `diag_Cdagger_dH_minus_EdS_C`.
- The `transport` workflow velocity unit remains `eV/fractional_reciprocal_coordinate`; SI conversion is currently exposed through the Python mobility helper.
- Single-point `TransportData` now records explicit non-SI metadata (`conductivity_unit="internal_SERTA_fractional_k"` and `carrier_density_unit="1/input_volume"`) so persisted transport artifacts are not confused with SI mobility outputs.
- Implemented `compute_serta_transport_scan(...)` and `TransportScanData`:
  - scans chemical-potential and temperature axes.
  - stores conductivity shape `(nmu, ntemperatures, 3, 3)`.
  - stores carrier-density shape `(nmu, ntemperatures)`.
  - uses the existing fixed-linewidth convention; per-scan-point linewidth recomputation is only available through the explicit chunked-artifact helper below.
- Implemented `compute_serta_mobility_si(...)` as a Python SI mobility helper:
  - converts fractional reciprocal-coordinate velocities to m/s through an explicit reciprocal cell.
  - computes SI conductivity, carrier density, and mobility.
  - supports 3D volume normalization and 2D sheet normalization.
  - persists results through `MobilityData` NPZ.
- Implemented `dptb eph --task mobility`:
  - reuses finite-difference or Hamiltonian-derivative velocity providers.
  - infers reciprocal cell from structure as `2*pi*atoms.cell.reciprocal()`.
  - writes `MobilityData` NPZ.
- Implemented `compute_serta_mobility_scan_si(...)` and `MobilityScanData`:
  - scans chemical-potential and temperature axes.
  - stores conductivity/mobility shape `(nmu, ntemperatures, 3, 3)`.
  - stores carrier-density shape `(nmu, ntemperatures)`.
  - uses the existing fixed-linewidth convention: the caller supplies one linewidth array, and the scan varies occupation/carrier-density/transport weighting.
- Implemented `compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact(...)`:
  - recomputes chunked linewidth at each scan point.
  - reuses the selected velocity provider once for artifact k-points.
  - stores `linewidth_scan_convention="per_scan_point_recomputed"`.
- Added CLI scan output:
  - `dptb eph --task transport --chemical-potentials ... --temperatures ...` writes `TransportScanData`.
  - `dptb eph --task transport --epc-artifact ... --linewidth-scan-convention recompute` writes recomputed-linewidth `TransportScanData` from chunked artifacts.
  - `dptb eph --task mobility --chemical-potentials ... --temperatures ...` writes `MobilityScanData`.
  - `dptb eph --task mobility --epc-artifact ... --linewidth-scan-convention recompute` writes recomputed-linewidth `MobilityScanData` from chunked artifacts.
  - singular and plural chemical-potential/temperature arguments are mutually exclusive per axis.
- Implemented chunked artifact consumers for transport/mobility:
  - `compute_serta_transport_from_epc_mesh_chunked_artifact(...)` computes chunked linewidth first, then computes transport with the selected velocity provider.
  - `compute_serta_transport_scan_from_epc_mesh_chunked_artifact(...)` computes chunked linewidth at the first scan point and reuses the fixed-linewidth transport scan convention.
  - `compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact(...)` recomputes chunked linewidth at each scan point and records `linewidth_scan_convention="per_scan_point_recomputed"`.
  - `compute_serta_mobility_si_from_epc_mesh_chunked_artifact(...)` computes single-point SI mobility from a chunked mesh artifact.
  - `compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact(...)` computes chunked linewidth at the first scan point and reuses the fixed-linewidth scan convention.
  - `compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact(...)` recomputes chunked linewidth at each scan point and reuses the SI mobility conversion.
- SCC-corrected velocity remains unsupported in v1.
- Existing fixed-linewidth scan helpers keep their public meaning; recomputed-linewidth scan behavior uses an explicit function name and metadata.

### Recomputed-Linewidth Transport Scan Slice

The current transport hardening slice adds explicit per-scan-point linewidth recomputation without changing any existing fixed-linewidth behavior.

Implemented API shape:

- `compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact(...)`
  - reads a chunked EPC mesh artifact;
  - loops over all requested chemical potentials and temperatures;
  - recomputes `compute_linewidth_mesh_chunked_artifact(...)` for each scan point;
  - computes velocities once per artifact k mesh, because velocity is independent of `(mu, temperature)` in the current non-SCC v1 convention;
  - returns `TransportScanData`.
- SI mobility uses the separate explicit helper `compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact(...)` instead of overloading the existing fixed-linewidth helper.

Required metadata:

- `linewidth_scan_convention: "per_scan_point_recomputed"`
- `source` or `producer` with the explicit function name。
- `chunked_artifact: True`
- `artifact_axis`
- `artifact_chunk_count`
- velocity source/convention/unit metadata copied from the selected velocity provider。
- chemical-potential and temperature units; current implementation uses eV temperature/kBT convention, so this must remain explicit。

Implemented tests:

- q-axis artifact plus finite-difference velocity: compare against a manual loop that calls `compute_linewidth_mesh(...)` at every scan point on the full in-memory mesh。
- k-axis artifact plus Hamiltonian-derivative velocity: compare against the same manual full-mesh recomputation。
- metadata test proving fixed-linewidth and recomputed-linewidth scans carry different `linewidth_scan_convention` values。
- import/export smoke test for the new explicit public helper。

Non-goals for this slice:

- Do not modify `compute_serta_transport_scan(...)` semantics。
- Do not add MPI, multiprocessing, or CUDA。
- Do not claim SCC-corrected linewidth recomputation; SCC remains unsupported。

### Acceptance

- Manual SI conversion tests。
- 2D/3D normalization tests。
- Multi-mu/multi-temperature shape tests。
- Finite-difference velocity unit conversion tests。
- Mobility and transport scan outputs assert unit metadata and `linewidth_scan_convention` metadata。
- `linewidth_scan_convention="recompute"` is rejected outside artifact scan workflows so the flag is never silently ignored。
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
  - frozen-charge SCC Hamiltonian derivative term
  - relaxed-charge / self-consistent charge-response term
  - 是否需要 finite-difference self-consistent charges
- 设计 provider boundary：
  - `SCCEPCProvider`
  - `FrozenChargeSCCProvider`
  - charge-response provider
  - finite-difference SCC derivative provider
- 明确哪些核心代码不能改；如需改 SCC engine，先写 ADR。
- 明确 frozen-charge SCC EPC 和 relaxed-charge SCC EPC 是两个不同定义：
  - frozen-charge SCC EPC 固定已收敛的 charge / shift，只对 SCC Hamiltonian 的显式结构依赖求导。
  - relaxed-charge SCC EPC 还包含 `d(delta_q)/dR` 或等价的 self-consistent response。
  - 如果只做 frozen-charge，必须在 API、metadata 和文档中明确不能冒充 full SCC EPC。
- 解释为什么直接调用 `get_hk(..., use_scc=True)` 或 `get_hk(..., with_derivative=True, use_scc=True)` 不足以定义 SCC EPC。
- 确认现有 DeePTB SCC 代码的边界：
  - `TBSystem.enable_scc(...)` / `run_scc(...)` / `get_hk(use_scc=True)` 已存在。
  - `get_hk(..., with_derivative=True, use_scc=True)` 当前明确 unsupported。
  - EPC 当前应继续拒绝 `use_scc=True`，直到公式、reference 和测试一起落地。

### Reference and Tests

- 需要 SCC reference case；不能用当前 non-SCC Graphene reference 证明 SCC EPC。
- 建议先构造小体系 SCC finite-difference smoke test，例如 tiny hBN-like SCC case；如果没有可靠数据，先只写设计，不伪造 reference。
- 后续再寻找或生成 dftbephy SCC benchmark。
- dftbephy SCC benchmark 只有在确认其 SCC convention 后才用于对齐；否则只能作为 exploratory reference。
- 至少需要覆盖：
  - SCC unsupported rejection remains explicit before implementation。
  - frozen-charge SCC derivative fixture, if implemented。
  - relaxed-charge finite-difference response fixture, if implemented。
  - non-SCC Graphene reference 不作为 SCC 通过依据。

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
- Coupling-strength summary and mode-resolved scattering summaries。
- Band-resolved and q-resolved scattering maps。
- Plot helper for path and mesh data。

### Position

- 这些功能按 DeePTB API 组织，不复刻 dftbephy notebook/JSON 格式。
- 优先支持从 existing NPZ data objects 计算，不重新触发 EPC coupling。
- 这些 analysis helpers 是 diagnostic/postprocess 能力，不应反向改变核心 `EPCData` / `EPCMeshData` schema。
- Eliashberg-like spectrum 当前定义为基于已有 `frequencies` 和 `coupling_strength` 的 broadened diagnostic；它不宣称完整材料特定 Eliashberg 理论，也不替代严谨的 alpha^2F 归一化推导。

### Current Implementation Status

- Implemented `compute_coupling_strength_summary(...)` and `dptb eph --task coupling-summary`:
  - reads `EPCData`, `EPCPathData`, or `EPCMeshData`;
  - emits JSON-friendly total, q-resolved, mode-resolved, band-resolved, and metadata summary fields;
  - supports weighted and unweighted mesh behavior.
- Implemented `compute_scattering_maps(...)` and `dptb eph --task scattering-map`:
  - reads `EPCData`, `EPCPathData`, or `EPCMeshData`;
  - emits JSON-friendly q/k/mode/band-resolved coupling-strength proxy maps;
  - supports weighted and unweighted mesh behavior;
  - does not claim to compute energy-conserving linewidths or full scattering rates.
- Implemented `compute_phonon_dos(...)` and `dptb eph --task phonon-dos`:
  - reads external `Phonons`;
  - uses explicit frequency grid, sigma, and broadening;
  - emits JSON-friendly DOS and mode-resolved DOS.
- Implemented current development slice for `compute_eliashberg_spectral_function(...)` and `dptb eph --task eliashberg`:
  - reads existing EPC path/mesh/plain NPZ data;
  - uses explicit frequency grid, sigma, and broadening;
  - emits `alpha2f`, mode-resolved `alpha2f`, and metadata to JSON;
  - uses normalized mesh k/q weights by default, with an unweighted diagnostic option;
  - remains a diagnostic helper, not a full material-specific Eliashberg solver.
- No plot helper is implemented yet.
- No persistent NPZ data object is currently required for these JSON summary helpers.

### Acceptance

- Manual numerical tests。
- Parser and CLI JSON workflow tests。
- Weighted/unweighted mesh behavior tests。
- Public export smoke tests for Python helpers。
- NPZ input/output tests only if a persistent analysis data object is introduced。
- Documentation examples。

## Workstream 7: Performance and Scaling

### Goals

让 mesh EPC 和 transport 能在实际体系上运行，而不只是小测试。

本 workstream 的核心原则是分离三件事：

- data contract: NPZ/chunked artifact 如何表达结果。
- executor: serial/multiprocessing/MPI 如何分发独立任务。
- backend: numpy/torch CPU/CUDA 如何计算单个任务。

这三层不能互相泄漏。尤其是，不能为了 MPI 改 EPCData 的物理 shape，也不能为了 CUDA 改 CLI 或 NPZ schema。

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

### Recommended Scaling Roadmap

Phase 0, already started:

- Keep serial executor as the reference implementation.
- Keep `EPCKChunkSpec`, `build_k_chunk_specs(...)`, and `concat_epc_k_chunks(...)` deterministic.
- Keep chunk metadata rank-independent and process-independent.

Phase 1, current practical scaling step:

- Harden the chunked artifact format for large mesh outputs before adding a parallel runtime.
- Support summary-first workflows that compute linewidth/transport accumulators without persisting the full `nq * nk * nmodes * nbands^2` coupling matrix when the user does not need it.
- Add direct tests proving chunked artifact load/reduce gives the same result as in-memory small cases.

Current Phase 1 status:

- Implemented first Python storage/reducer contract:
  - `save_epc_mesh_chunked_artifact(...)` writes an existing `EPCMeshData` into a directory artifact.
  - `load_epc_mesh_chunked_artifact(...)` validates the manifest and reduces q/k chunks back to `EPCMeshData`.
  - chunks are stored as pickle-free `EPCData` NPZ files plus global k/q weights.
  - artifact reducers reuse the existing deterministic `concat_epc_k_chunks(...)` / `concat_epc_q_chunks(...)` helpers.
  - manifest/weights validation rejects schema drift, unsafe filenames, reducer mismatch, bad chunk counts, non-contiguous ranges, and bad weights metadata.
- Implemented first serial streaming producer:
  - `TBSystem.eph.compute_mesh_chunked_artifact(...)` computes one q-axis or k-axis chunk at a time and writes directly to the artifact directory.
  - this avoids first materializing a full `EPCMeshData`.
  - it remains serial Python execution; no multiprocessing, MPI, or CUDA runtime has been added.
- Implemented first summary-first postprocess helper:
  - `compute_linewidth_mesh_chunked_artifact(...)` reads chunk NPZ files one at a time and returns `LinewidthMeshData`.
  - q-axis artifacts accumulate q contributions with global q weights.
  - k-axis artifacts compute each k chunk and concatenate reduced linewidth arrays.
- Implemented first summary-first transport helper:
  - `compute_serta_transport_from_epc_mesh_chunked_artifact(...)` combines chunked linewidth reduction with existing velocity providers.
  - finite-difference and Hamiltonian-derivative velocity sources remain available.
  - the helper returns `TransportData` without materializing the full mesh coupling tensor.
- Implemented first summary-first transport scan helper:
  - `compute_serta_transport_scan_from_epc_mesh_chunked_artifact(...)` uses chunked linewidth at the first requested scan point and reuses the existing fixed-linewidth scan convention.
  - `compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact(...)` recomputes chunked linewidth at every requested `(chemical_potential, temperature)` point and records `linewidth_scan_convention="per_scan_point_recomputed"`.
- Implemented first summary-first SI mobility helper:
  - `compute_serta_mobility_si_from_epc_mesh_chunked_artifact(...)` combines chunked linewidth reduction, existing velocity providers, and SI mobility conversion.
  - 2D/3D normalization and reciprocal-cell conventions are inherited from `compute_serta_mobility_si(...)`.
- Implemented first summary-first SI mobility scan helper:
  - `compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact(...)` uses chunked linewidth at the first requested scan point and reuses the existing fixed-linewidth scan convention.
  - `compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact(...)` recomputes linewidth at every scan point and reuses the same SI mobility conversion path.
- Implemented artifact CLI exposure:
  - `dptb eph --task mesh-linewidth --epc-artifact ...` writes `LinewidthMeshData` directly from a chunked artifact.
  - `dptb eph --task transport --epc-artifact ... --linewidth-scan-convention fixed|recompute` writes `TransportData` or `TransportScanData`.
  - `dptb eph --task mobility --epc-artifact ... --linewidth-scan-convention fixed|recompute` writes `MobilityData` or `MobilityScanData`.
  - artifact workflows use artifact weights directly and reject conflicting `--epc-data`, `--linewidth-data`, and `--kpoint-weights` inputs.
- Current limitation:
  - summary-first artifact consumers are available, and the first serial streaming producer is available, but multi-axis q/k streaming and parallel writers remain future work.
  - per-scan-point linewidth recomputation is implemented only for explicit chunked-artifact transport/mobility scan helpers; existing fixed-linewidth scan semantics must remain unchanged.
  - no multiprocessing, MPI, or CUDA runtime has been added.

Phase 2, CPU parallel execution:

- Add a multiprocessing executor first if it can reuse `dptb.utils.multiprocessing.num_tasks()` and avoid new required dependencies.
- Add `mpi4py` executor only as an optional backend for cluster runs.
- Both executors must consume the same immutable chunk specs and use the same reducers as serial execution.
- Do not place `mpi4py` imports at module import time for default EPC workflows; MPI imports must be lazy and optional.
- Worker ownership of `TBSystem` must be explicit: either each worker owns an isolated system/calculator instance, or the task is pure array postprocessing and does not mutate system state.

Phase 3, GPU backend:

- Add torch CUDA backend only after the per-chunk arrays, contraction order, velocity provider outputs, and reducer semantics are stable.
- CUDA should accelerate batched eigensolve, velocity matrix elements, and EPC contraction inside one chunk.
- CUDA must not own chunk scheduling, file writing, or persistent schema decisions.
- CUDA backend must respect DeePTB model device/dtype conventions and convert tensors to CPU/numpy at NPZ artifact boundaries.
- MPI and CUDA may be combined later as MPI-over-chunks plus CUDA-inside-rank; this combination must still load and reduce through the same NPZ/chunked artifact contract.

Practical recommendation for this codebase: implement Phase 1 before MPI or CUDA. If the next wave must pick one acceleration feature after Phase 1, pick multiprocessing/MPI executor before CUDA unless profiling shows a single chunk's contraction/eigensolve dominates wall time.

Current recommendation for the next coding sprint: do not add `mpi4py`, multiprocessing, or CUDA yet. Spend the sprint on release hardening the serial/chunked artifact path, because that is the contract future MPI and CUDA paths must preserve. If profiling or user benchmarks later force one acceleration feature first, pick CPU-side multiprocessing/MPI executor before CUDA, unless profiling clearly shows one chunk's eigensolve/contraction dominates wall time.

### MPI and CUDA Position

MPI and CUDA are not competing designs here:

- MPI/multiprocessing answer "which independent chunks run on which rank/process?"
- CUDA/torch backend answers "how does one rank/process compute one chunk efficiently?"
- The public EPC data contract should not depend on either one.
- A future production run may use MPI ranks across q/k chunks and CUDA inside each rank.

For the next implementation wave, the correct preparation is interface-level:

- keep `EPCKChunkSpec` and `EPCQChunkSpec` rank-independent;
- make reducers deterministic and independent of file/rank order;
- preserve backend metadata in outputs;
- avoid putting device objects, MPI rank IDs, or process-local assumptions into NPZ files;
- keep serial execution as the reference path for all tests.

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
- Worker functions must not mutate `TBSystem` shared state without a local copy or a documented restore path; finite-difference displacement workflows are especially sensitive because `set_atoms(...)` resets postprocess state.
- Executor configuration should stay outside physics APIs where possible: physics functions accept specs/data objects, while executor wrappers choose serial/multiprocessing/MPI.
- Reducers must validate global shape, chunk index coverage, k/q/band index consistency, weights, units, schema version, and backend metadata before concatenating or summing.

### Backend Design

- Backend should be explicit metadata, not inferred from input arrays.
- Initial backend can be `numpy` or existing torch CPU flow.
- Torch CUDA backend should reuse DeePTB model device/dtype conventions and avoid unnecessary CPU/GPU transfers.
- Backend code should not decide chunk ownership or reduction order.
- NPZ persistence remains CPU/numpy oriented; GPU tensors must be converted at artifact boundaries.
- Analytic velocity backend should reuse `get_hk(..., with_derivative=True)` where possible.
- Finite-difference velocity remains a fallback/reference backend.
- Backend selection should be orthogonal to velocity source: `velocity_source="finite_difference"` and `velocity_source="hamiltonian_derivative"` can each have CPU implementations first, and only later gain CUDA kernels if profiling justifies it.
- Backend metadata should record enough to reproduce and audit a run, for example backend name, device class, dtype, velocity source, derivative convention, and unit convention; it should not require an MPI rank or CUDA device to load the artifact.

### Acceptance

- Chunked workflow memory does not scale as full `nq * nk * nmodes * nbands^2` when summaries are requested。
- Numerical parity between chunked and non-chunked small cases。
- Benchmark script for Graphene small/medium mesh。
- Serial and future parallel reducers produce bitwise-identical or tolerance-bounded results on small deterministic fixtures。

## Suggested Implementation Order

1. Confirm the current implemented feature set is checkpointed:
   - v1 coupling / linewidth / relaxation-time / transport / subspace。
   - path/mesh workflows。
   - serial k/q chunk executor boundary。
   - chunked mesh artifact save/load/reduce contract。
   - chunked artifact linewidth/transport/mobility/mobility-scan consumers。
   - serial streaming mesh artifact producer and mesh-artifact CLI。
   - Hamiltonian-derivative velocity。
   - SI mobility, mobility scan, and transport scan。
   - coupling-summary / scattering-map / phonon-DOS / Eliashberg-like diagnostic analysis。
2. Keep the explicit per-scan-point linewidth recomputation artifact scans as a completed but protected behavior:
   - existing fixed-linewidth scan helpers must remain unchanged。
   - Python helper, CLI exposure, metadata convention, and fixed-vs-recomputed guard tests are implemented。
   - future scan features must choose fixed-linewidth or per-scan-point recomputation explicitly and record that choice in metadata。
   - single-point artifact entrypoint tests must use synthetic parameters that produce finite positive linewidth; if a temporary hardcoded development fixture is used, keep `TODO(epc-fixture)` nearby。
3. Continue release hardening:
   - lightweight default EPC fixture。
   - public export smoke tests for any newly added public API。
   - CLI/doc schema drift check。
   - strict NPZ validation tests, especially metadata JSON and artifact weights。
   - full repo test pass after recent EPC hardening commits。
4. Review and finalize the SCC EPC design doc:
   - frozen-charge vs relaxed-charge definitions。
   - provider boundary。
   - reference strategy。
   - tests required before enabling `use_scc=True`。
5. Continue Phase 1 scaling:
   - continue hardening the chunked artifact contract as new artifact consumers are added.
   - harden summary-first linewidth/transport/mobility helpers and consider per-point-linewidth accumulators.
   - design multi-axis q/k streaming separately from the current one-axis serial streaming producer.
   - keep executor/backend choices outside persistent NPZ schema and outside core physics functions.
6. Add optional plot helpers from existing NPZ objects.
7. Add multiprocessing executor first, if profiling shows CPU task parallelism is needed and it can reuse the same chunk specs/reducers.
8. Add optional `mpi4py` executor only after multiprocessing/serial reducer semantics are stable and default tests remain MPI-free.
9. Add torch CUDA backend only after serial and CPU/MPI executor semantics are fixed and profiling shows per-chunk kernels dominate.
10. Implement SCC EPC only after design, reference data, and tests are ready.
11. Advance SOC/spinful, polar correction, and full gauge tracking as separate design-backed workstreams.

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

This sprint is a stabilization and design sprint for the current implementation, not another broad feature-expansion sprint.

1. Verify current branch state:
   - `git status --short --branch`
   - confirm no unrelated dirty files before editing implementation。
2. Run the full default test suite after the latest EPC hardening commits:
   - `uv run pytest ./dptb/tests/ -q`
   - if failures appear, separate EPC regressions from unrelated repository failures before editing。
3. Audit strict NPZ validation in the remaining EPC loaders:
   - metadata JSON must be scalar valid JSON object。
   - arrays must be pickle-free/object-free, non-empty where required, shape-consistent, and finite。
   - chunked artifact weights must remain finite, non-negative, shape-consistent, and positive-sum。
   - CLI `.npz` / `.json` helper inputs must report missing required array fields as `ValueError` with the field name。
   - errors should name the bad field and expected convention。
4. Verify current exports whenever a new public helper is added:
   - `dptb.postprocess.unified.eph`
   - `dptb.postprocess.unified`
   - import smoke tests for EPC data objects, linewidth/relaxation/transport/mobility objects, scan helpers, analysis helpers, velocity helpers, and executor helpers。
5. Keep focused checks in the normal loop:
   - `git diff --check`
   - focused tests for any touched EPC helper。
   - existing fixed-linewidth and recomputed-linewidth scan tests when transport/mobility logic changes。
   - `uv run pytest dptb/tests/test_electron_phonon.py -q`
6. Add or extend direct executor tests only where coverage is still missing:
   - full single chunk。
   - multiple deterministic chunks。
   - invalid chunk settings, including wrong types。
   - concat rejection for inconsistent chunk inputs。
   - artifact metadata/weights rejection for malformed files。
7. Add a short scaling design check before implementing new mesh features:
   - identify the intended split axis: q chunk, k chunk, band group, chemical-potential axis, or temperature axis。
   - state whether the work produces a full coupling artifact or summary accumulators。
   - state whether the implementation is executor-only, backend-only, or data-contract work。
   - keep MPI and CUDA out of the default path until Phase 1 chunked artifact/reducer behavior is stable。
8. Audit hardcoded development reference usage:
   - keep full Graphene reference opt-in and untracked if desired。
   - every hardcoded development reference must carry `TODO(epc-fixture)`。
   - default tests must move toward lightweight self-contained fixtures before merge。
9. Update `docs/epc_v1_workflow.md` and docs index if schema/API drift exists.
10. Review and finalize `docs/epc_scc_design.md` before touching SCC implementation.
11. Create a checkpoint commit once docs and focused tests pass.

The immediate merge target is: stable EPC v1 plus DeePTB-native path/mesh workflows, serial k/q chunk executor boundary, chunked mesh artifact reduction, first serial streaming artifact producer, summary-first artifact consumers, fixed-linewidth and explicit recomputed-linewidth transport scan conventions, Hamiltonian-derivative velocity, SI mobility, mobility scan, and JSON analysis helpers from existing NPZ objects. SCC EPC, MPI, CUDA, SOC/spinful, polar correction, and multi-axis/parallel artifact production remain planned follow-up workstreams.
