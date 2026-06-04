# DeePTB EPC 后续功能缺口与开发计划

## Purpose

本文档记录当前 `feat/electron-phonon-coupling` 分支进入 EPC v1
release/PR 收口之后，仍然缺少但不建议混入当前 PR 的功能。

当前 PR 的目标应保持为：稳定 DeePTB-native EPC v1、NPZ 数据契约、path/mesh
workflow、serial/chunked artifact、linewidth/transport/mobility、analysis helpers、
CLI、文档和测试。下面列出的内容是下一阶段开发计划，不是当前 merge
前必须补齐的阻塞项。

## Current Baseline

当前分支已经具备以下 EPC v1 能力：

- 外部 phonon mode NPZ / phonopy reader。
- `EPCData`、`EPCPathData`、`EPCMeshData` 及后处理 NPZ 数据对象。
- fixed-k + q-path EPC、path linewidth、path relaxation time。
- explicit k/q mesh EPC、mesh linewidth、mesh relaxation time。
- serial k/q chunk specs、deterministic reducers、chunked mesh artifact。
- first serial streaming mesh artifact producer。
- summary-first chunked artifact consumers for linewidth、transport、mobility 和 scans。
- finite-difference velocity provider 和 Hamiltonian-derivative velocity provider。
- SERTA transport、SI mobility、transport/mobility scans。
- coupling summary、scattering maps、phonon DOS、Eliashberg-like diagnostic JSON helpers。
- degenerate-subspace gauge-invariant coupling diagnostics。
- `dptb eph --task ...` CLI family。
- strict NPZ/schema/metadata validation 的多轮 hardening。

当前明确不支持：

- DeePTB 内部 phonon solver、force constants、forces、stress、total energy 或
  repulsive potential workflow。
- SCC-corrected EPC。
- SOC/spinful EPC。
- polar correction。
- full degenerate-band gauge tracking。
- multiprocessing/MPI executor。
- torch CUDA backend。
- multi-axis q/k streaming artifact writer。
- dftbephy HDF5/DFTB+ workflow parity。

## Missing Functionality

### 1. Release-Hardening Gaps

这些是当前 PR 合并前最值得继续关注的缺口，但仍属于收口性质，不是新功能扩展。

- Lightweight FD/coupling fixture：
  - 现在已有最小 synthetic fixture 覆盖 linewidth、coupling contraction、
    coupling summary 和 scattering map。
  - 外部 Graphene reference 仍是 opt-in benchmark。
  - 下一步可补一个更轻量的 finite-difference / Fourier projection fixture，
    减少默认测试对外部 Graphene reference 的依赖。
- Final drift audit：
  - 每次新增 public API 后确认 `dptb.postprocess.unified.eph.__all__` 和
    `dptb.postprocess.unified` re-export 一致。
  - 确认 CLI task choices、docs examples、parser help 和 tests 同步。
- Persistent loader audit：
  - 继续抽样检查所有 EPC NPZ/data artifact loader 对 object arrays、空数组、
    非有限值、metadata JSON、schema mismatch、missing required arrays 的错误信息。
- Physical-convention review：
  - 再审一次 temperature-as-kBT-eV、fractional reciprocal k、`2*pi` reciprocal
    cell、2D/3D normalization、transport non-SI metadata 和 SI mobility metadata。
- Final validation：
  - `uv run pytest dptb/tests/test_electron_phonon.py -q`
  - `uv run pytest ./dptb/tests/ -q`
  - `git diff --check`

### 2. K-Path EPC Workflow

当前 path workflow 只支持 fixed electronic k-points + q-path phonons，即
`path_axis="q"`。还缺：

- k-path + fixed q workflow。
- 同时表达 k-path 和 q-path 时的 shape contract。
- path labels/segments 与 DeePTB 现有 band plotting convention 的对齐。
- path-resolved plot-ready summaries。

建议优先级：中。

原因：它对用户可视化有价值，但不影响当前 v1 mesh/path 主契约。

### 3. Plot Helpers

当前已有 JSON diagnostics，但没有官方 plot helper。还缺：

- EPC path coupling / linewidth / relaxation-time plot helper。
- phonon DOS plot helper。
- scattering map plot helper。
- Eliashberg-like diagnostic plot helper。
- 可选的 Matplotlib output，不引入重依赖，不改变 NPZ contract。

建议优先级：中低。

原因：提升用户体验，但应在数据契约稳定后做，避免把 plotting 当核心 API。

### 4. Multi-Axis Streaming Artifact

当前 `mesh-artifact` 支持 q-axis 或 k-axis one-axis serial streaming。还缺：

- 同时按 q 和 k 分块的 artifact layout。
- 二维 chunk manifest。
- q/k chunk reducer 的全局覆盖校验。
- summary-first consumers 对 multi-axis artifact 的支持。
- parallel writer 之前的 serial reference implementation。

建议优先级：高，仅次于 release hardening。

原因：这是进入真实大体系 mesh workflow 的直接扩展，也会为后续 MPI executor
提供稳定 data contract。

### 5. Multiprocessing / MPI Executor

当前只有 serial executor boundary。还缺：

- multiprocessing executor，优先复用 `dptb.utils.multiprocessing.num_tasks()`。
- optional `mpi4py` executor，必须 lazy import，默认安装和默认 CI 不依赖 MPI。
- worker ownership 策略：
  - 每个 worker 独立持有 `TBSystem` / calculator；或
  - worker 只处理纯数组 postprocess task。
- serial / multiprocessing / MPI reducer parity tests。
- rank-independent metadata，不能把 rank/process 信息写成 persistent schema 必需字段。

建议优先级：中高，但必须在 multi-axis/serial artifact contract 稳定后做。

### 6. Torch CUDA Backend

当前无 CUDA EPC backend。还缺：

- per-chunk batched eigensolve / velocity / EPC contraction 的 torch backend 设计。
- CPU reference parity tests。
- device/dtype metadata，仅作 debug/provenance，不作为 NPZ schema 必需字段。
- artifact boundary 统一转 CPU/numpy。
- profiling 证明单 chunk kernel 是瓶颈。

建议优先级：中低。

原因：EPC mesh 的第一瓶颈可能是 q/k task 数量和 I/O/reduction；CUDA 应在
chunk layout 和 profiling 明确后再做。

### 7. SCC EPC

当前 SCC EPC 只有设计文档，不实现。还缺：

- frozen-charge SCC EPC 与 relaxed-charge SCC EPC 的最终公式确认。
- SCC derivative provider interface。
- finite-difference SCC rerun provider。
- SCC reference/smoke fixture，不能用 non-SCC Graphene 证明 SCC EPC。
- metadata 明确记录 SCC definition、charge response convention 和 convergence settings。
- `use_scc=True` 从 reject 改为 opt-in supported 前的完整测试。

建议优先级：高，但必须单独开 workstream。

原因：物理定义复杂，不能通过简单透传 `use_scc=True` 到现有 `get_hk(...)` 实现。

### 8. SOC / Spinful EPC

当前只有 `spin_degeneracy` 标量权重，不支持 spinful Hamiltonian / SOC EPC。
还缺：

- spinor basis / orbital indexing contract。
- spinful Hamiltonian derivative source。
- time-reversal / Kramers degeneracy convention。
- spin-resolved coupling metadata。
- reference case 和 gauge tests。

建议优先级：低，除非用户已有明确 spinful/SOC benchmark 需求。

### 9. Polar Correction

当前没有 polar EPC correction。还缺：

- Born effective charge / dielectric tensor / LO-TO correction metadata contract。
- external polar metadata reader。
- long-range / short-range decomposition convention。
- q -> 0 行为和 unit tests。
- 明确 DeePTB 仍不计算 phonons，只读取外部 polar metadata。

建议优先级：低到中，取决于目标材料体系。

### 10. Full Degenerate-Band Gauge Tracking

当前只提供 gauge-invariant subspace strength diagnostic，不做连续 gauge tracking。
还缺：

- path/mesh eigenvector alignment。
- parallel transport 或 subspace projection strategy。
- band crossing / avoided crossing 处理策略。
- degenerate reference tests。
- 与 EPC matrix phase convention 的明确关系。

建议优先级：中。

原因：对高质量 path analysis 有价值，但不应改变当前 v1 的 coupling matrix contract。

### 11. Reference And Benchmark Suite

当前 Graphene reference 是 opt-in、本地未追踪。还缺：

- 小型 in-repo reference fixture，覆盖 FD derivative、Fourier projection 和 coupling。
- Graphene small mesh benchmark script。
- 可选 dftbephy reference bridge 的文档化输入策略。
- slow reference tests 的运行指南和结果记录。

建议优先级：高。

原因：这是后续改 phase、unit、velocity、provider、backend 时的安全网。

## Recommended Next Development Order

### Phase 0: Finish Current PR

目标：不加新功能，只完成 release hardening。

1. 跑 EPC focused tests 和全仓 tests。
2. 检查 CLI/docs/API drift。
3. 检查 persistent loader edge cases。
4. 检查 external Graphene reference 是否仍 opt-in 且所有开发期硬编码都有
   `TODO(epc-fixture)`。
5. 准备 PR summary，明确 non-goals。

Exit criteria：

- 默认测试通过。
- docs 与 CLI help 不冲突。
- 当前 public API export smoke 通过。
- 当前 PR 没有承诺 SCC、MPI/CUDA、SOC、polar、k-path 或 plot helper。

### Phase 1: Better Fixtures And Benchmarks

目标：补默认测试安全网。

1. 设计轻量 finite-difference / Fourier projection fixture。
2. 补 coupling / FD provider 的 self-contained regression。
3. 保留完整 Graphene 作为 opt-in benchmark。
4. 写 benchmark command 和 expected tolerance。

Exit criteria：

- 默认 CI 不依赖外部 checkout。
- 关键 phase/unit/provider 变化有默认 fixture 或 opt-in benchmark 能兜底。

### Phase 2: Multi-Axis Artifact Contract

目标：把大 mesh artifact contract 扩展到 q/k 双轴。

1. 设计二维 chunk manifest。
2. 实现 serial q/k multi-axis producer。
3. 实现 deterministic reducer。
4. 扩展 linewidth / transport / mobility summary-first consumers。
5. 加 small-case parity tests。

Exit criteria：

- small mesh 下 multi-axis artifact 与 full in-memory result 数值一致。
- artifact schema 不依赖 executor/backend。

### Phase 3: Optional CPU Parallel Executor

目标：在不改变 data contract 的前提下加速。

1. 先实现 multiprocessing executor。
2. 明确 worker owns system 还是 pure array task。
3. 用同一 chunk spec 和 reducer。
4. 再考虑 optional MPI executor。

Exit criteria：

- serial 和 parallel 小 fixture 结果一致。
- 默认安装、默认测试不要求 MPI。

### Phase 4: SCC EPC Workstream

目标：单独推进 SCC EPC，不混入普通 EPC hardening。

1. Review `docs/epc_scc_design.md`。
2. 确认 frozen-charge / relaxed-charge 公式。
3. 准备 SCC smoke/reference fixture。
4. 实现 provider 和 metadata。
5. 最后才允许 `use_scc=True` 进入 supported path。

Exit criteria：

- SCC metadata 明确物理定义。
- SCC 和 non-SCC tests 同时通过。
- `use_scc=True` 不再只是透传现有 non-SCC EPC path。

### Phase 5: User-Facing Analysis Improvements

目标：提升可视化和 workflow 完整度。

1. k-path + fixed-q workflow。
2. plot helpers。
3. full gauge tracking 设计。
4. 根据用户材料需求决定 polar 或 SOC/spinful 优先级。

Exit criteria：

- 新功能从 existing NPZ/API 出发。
- plot/helper 不改变核心 persistent data contract。
- advanced physics 有 reference 或明确 design gate。

## Non-Goals For The Next PR

下一次以 release hardening 为目标的 PR 仍不应包含：

- SCC EPC implementation。
- MPI/CUDA implementation。
- SOC/spinful EPC。
- polar correction。
- DeePTB phonon solver。
- DFTB+ workflow parity。
- HDF5 作为 public EPC data contract。

这些内容应分别开独立 workstream，并在 PR 描述中写清楚物理定义、数据契约、
测试和 non-goals。
