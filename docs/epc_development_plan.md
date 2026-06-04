# DeePTB EPC 功能开发计划

## Summary

目标是形成 DeePTB EPC 的完整开发路线图：当前波次先稳定 EPC v1 的核心计算、数据契约、CLI 工作流和基础后处理；后续再扩展 SCC/SOC/polar correction、degenerate-band gauge 稳定化等更复杂物理模型。EPC 的基本边界是：外部程序提供 phonon qpoints、frequencies、eigenvectors、masses；DeePTB 计算电子结构、有限差分 `dH/dR` / `dS/dR`，并收缩得到 EPC。

功能边界固定为：不计算 force constants，不跑 phonopy 声子计算，不做 total energy、forces、stress，不做 repulsive potential。`phonopy` 只作为读取和解析外部 phonon 数据的可选依赖。

## Current Implementation Status

- EPC v1 core, NPZ data contracts, `dptb eph` workflow entrypoint, linewidth, relaxation-time, basic SERTA transport, finite-difference velocity bridge, and degenerate-subspace diagnostic are implemented in `dptb/postprocess/unified/eph/` and `dptb/entrypoints/eph.py`.
- Public Python APIs are exported from both `dptb.postprocess.unified.eph` and `dptb.postprocess.unified`; CLI and NPZ loaders reject malformed or empty k/q-point inputs early.
- Default EPC tests, Graphene coupling-strength reference, and slow Graphene supercell finite-difference reference are the current acceptance gates for v1 physics and workflow stability.
- Development-stage Graphene reference remains external/untracked. TODO before merge/release: replace or supplement it with a lightweight in-repo fixture or a CI-specific reference strategy.
- SCC-corrected EPC, SOC/spinful EPC, polar correction, full degenerate-band gauge tracking, and full SI mobility/unit conversion remain future milestones, not v1 implementation claims.

## Current Wave: Stabilize EPC v1 Workflow

- Core EPC:
  - 当前 `compute_coupling(kpoints, phonons, ...)` 继续作为核心入口。
  - 第一阶段正式 phonon input contract 固定为 Python `Phonons` object 和 DeePTB phonon-mode NPZ。
  - phonopy object/file 只作为辅助 reader 或 benchmark 输入，不作为 DeePTB EPC 主数据契约。
  - 所有 phonon 数据必须显式包含 qpoints、frequencies、eigenvectors、masses。
  - masses 必须为有限正值；EPC v1 不支持 imaginary phonon modes，negative frequencies 在 coupling/linewidth 后处理中拒绝。
  - acoustic zero-frequency modes 在 linewidth 后处理中通过可配置 `frequency_floor` 正则化，避免 Bose occupation 奇点；该 floor 必须写入 metadata。
  - EPC 扩展不得继续扩大基于 `atom_orbs` 字符串的索引推断；atom orbital slices、basis offsets 和 atom-to-orbital mapping 应复用现有 `OrbitalMapper` 信息或 calculator 明确暴露的 structured metadata。
  - 尽量不修改 `OrbitalMapper`、Hamiltonian transform 等核心索引/矩阵构造代码；如需桥接 EPC 所需 metadata，优先在 unified/EPC 层新增薄适配器。
  - `DFTBPlusGauge.from_atom_orbs()` 可保留为 benchmark adapter，不作为生产 EPC 的索引权威。

- CLI workflow:
  - 当前波次新增 `dptb eph`，但边界固定为 workflow entrypoint，不提供“计算 phonon”子命令。
  - v1 workflow usage 和 NPZ contract 记录在 `docs/epc_v1_workflow.md`。
  - CLI 只做 phonon 数据读取、校验、电子计算、EPC 输出。
  - 输入为 DeePTB 模型/结构、外部 phonon mode NPZ、电子 kpoints/bands、displacement、输出路径。
  - `dptb eph --task linewidth` 支持从 `EPCData` NPZ 计算 linewidth 并输出 `LinewidthData` NPZ。
  - `dptb eph --task relaxation-time` 支持从 `LinewidthData` NPZ 计算 relaxation time 并输出 `RelaxationTimeData` NPZ。
  - `dptb eph --task transport` 支持从 `EPCData` NPZ、`LinewidthData` NPZ 和 DeePTB 模型/结构计算有限差分 velocity，并输出 `TransportData` NPZ。
  - `dptb eph --task subspace` 支持从 `EPCData` NPZ 和 `start:stop` band ranges 输出 `SubspaceCouplingData` NPZ。
  - `use_scc=True` 必须继续抛 `NotImplementedError`，避免误导用户认为 v1 已支持 SCC EPC。

- 输出格式：
  - 继续支持 `EPCData.save_npz()`。
  - DeePTB EPC 主数据格式固定为 NPZ，不把 HDF5 作为主输出格式。
  - `EPCData` NPZ 保存后续后处理所需的最小闭包：kpoints、qpoints、band_indices、frequencies、eigenvalues_k、eigenvalues_kq、coupling_matrix、coupling_strength 和 units/source/displacement/use_scc 等 metadata。
  - `EPCData` NPZ 默认不保存 `dH/dR`、`dS/dR`、full eigenvectors 或 full H/S matrices；这些只作为 debug/reference artifact 另行处理。
  - HDF5 只作为读取或对比 dftbephy reference 的辅助格式；不要求 DeePTB EPC 输出严格兼容 dftbephy HDF5 字段。
  - 公共 API 命名继续用 `coupling_matrix` / `coupling_strength`，不把 `g2` 作为主命名。
  - 后处理结果使用独立 NPZ contract：
    - `LinewidthData.save_npz()/load_npz()` 保存 linewidth、absorption、emission 和 metadata。
    - `RelaxationTimeData.save_npz()/load_npz()` 保存 relaxation time 和 metadata，约定 `tau = hbar / (2 * linewidth)`。
    - `TransportData.save_npz()/load_npz()` 保存 conductivity、carrier_density 和 metadata。

- dftbephy Graphene reference 对齐：
  - 对齐 EPC matrix / strength 的公式、单位、overlap 修正、supercell finite-difference Fourier 投影。
  - 不复制 dftbephy 的 phonon 计算流程；只兼容其 reference/output 用于 benchmark。
  - 当前 dftbephy Graphene reference 是 non-SCC benchmark；虽然 dftbephy 代码支持 SCC reference/derivative 参数，但当前对齐测试不要求 SCC EPC。
  - 不以完整复刻 dftbephy workflow parity 为目标；DFTB+ 文件生态、目录布局、HDF5/JSON 字段级兼容和 `init/bands/ephline/rtline` 外壳不进入 DeePTB EPC 的默认范围。
  - 后续应吸收 dftbephy 中有物理或用户价值的能力切片，并重构成 DeePTB-native workflow：EPC path workflow、EPC mesh workflow、transport/mobility workflow、reference validation 和必要的分析核。

- Basic postprocess:
  - 当前波次实现从 `EPCData` 计算 Gaussian/Lorentzian broadened linewidth。
  - 当前波次实现从 `LinewidthData` 计算 relaxation time，默认保留 mode-resolved 轴，可选先按 mode 求和。
  - linewidth postprocess 对 negative frequencies 抛错，对 zero/near-zero acoustic frequencies 使用 `frequency_floor`。
  - 当前波次实现基础 SERTA conductivity 后处理核心，输入为 eigenvalues、velocities、linewidth、chemical potential、temperature/kBT、kpoint weights、spin degeneracy、volume。
  - 当前波次提供有限差分 band velocity bridge：通过 `system.get_eigenvalues(k_points=...)` 中心差分估计选中 bands 的速度，并可从 `EPCData + LinewidthData` 直接组装 SERTA transport。
  - 当前波次提供 degenerate-band diagnostic：可按简并 band group 对 coupling block 做 Frobenius-norm subspace aggregation，得到对简并子空间内 unitary gauge rotation 不变的强度。
  - `SubspaceCouplingData` NPZ 保存 subspace strength 和 contiguous band-range group bounds；非连续 group 只作为 Python helper 输入，不进入 v1 持久化契约。
  - transport 当前不宣称完整 SI mobility workflow；有限差分速度单位约定为 eV/fractional reciprocal coordinate，必须在 metadata/docstring 中保留。

- 当前波次仍明确不做：
  - 当前开发波次不修改 SCC engine、charge update、Hubbard U、SCC energy/force 相关逻辑；EPC API 继续对 `use_scc=True` 抛 `NotImplementedError`。
  - 当前开发波次不实现 SCC EPC、SOC/spinful EPC、polar correction 或 degenerate-band gauge 稳定化。

## Milestones

- Milestone 1: Stabilize EPC v1
  - 完成 Python `Phonons` object 和 DeePTB phonon-mode NPZ 的读取与校验。
  - 稳定 `TBSystem.eph.compute_coupling(...)`、`Phonons`、`EPCData` 的 Python API。
  - 将 EPC 的 orbital slices / basis offsets 来源收敛到现有 `OrbitalMapper` 信息或 structured calculator metadata，避免改动核心 `OrbitalMapper` 实现。
  - 支持 `EPCData` NPZ 输出和 roundtrip。
  - 验收：默认 EPC 单元测试通过；Graphene coupling-strength reference opt-in 测试通过。

- Milestone 2: Development validation
  - 开发阶段使用完整 dftbephy Graphene reference 做强验证，包括 reference matrices、finite-difference derivatives、phonon modes 和 EPC 输出。
  - Graphene reference 资产不进入 git 追踪；开发阶段通过 `DEEPTB_EPH_REFERENCE_ROOT` 和 `DEEPTB_EPH_SKDATA_ROOT` 指向本机 reference 资产。
  - TODO before merge/release: 设计仓库内轻量 fixture 或 CI 专用 reference 策略，避免默认测试依赖外部 checkout。
  - 功能稳定后再设计 DeePTB 仓库内自包含小 fixture，用于默认单元测试和 CI。

- Milestone 3: CLI and workflow entrypoint
  - 新增 `dptb eph`。
  - 输入为 DeePTB 模型/结构、外部 phonon mode data、电子 kpoints/bands、displacement、输出路径。
  - CLI 不计算 phonons，只读取外部 phonon mode data 并输出 NPZ。
  - 支持 `dptb eph --task linewidth` 从 EPCData NPZ 输出 LinewidthData NPZ。
  - 支持 `dptb eph --task relaxation-time` 从 LinewidthData NPZ 输出 RelaxationTimeData NPZ。
  - 支持 `dptb eph --task transport` 从 EPCData NPZ、LinewidthData NPZ 和有限差分 velocity 输出 TransportData NPZ。
  - 支持 `dptb eph --task subspace` 从 EPCData NPZ 和 contiguous `start:stop` band ranges 输出 SubspaceCouplingData NPZ。
  - 验收：CLI parser、kpoints/weights loader、fake-system EPC output NPZ、linewidth output NPZ、relaxation-time output NPZ、transport output NPZ、subspace output NPZ、SCC rejection 均有测试。

- Milestone 4: Linewidth / relaxation time
  - 从 `EPCData` 计算 Gaussian/Lorentzian broadened linewidth。
  - 支持 mode-resolved relaxation time。
  - 支持 `LinewidthData` NPZ roundtrip。
  - 支持 `RelaxationTimeData` NPZ roundtrip。
  - 验收：Gaussian/Lorentzian manual reference、mode-resolved sum rule、relaxation-time convention、invalid parameter rejection 和 NPZ schema 测试通过。

- Milestone 5: Transport
  - 当前波次先实现 SERTA conductivity 后处理核心。
  - 当前波次接入 DeePTB `system.get_eigenvalues` 的有限差分 band velocity workflow。
  - 后续可替换或扩展为解析 velocity provider，并补充完整 SI mobility/unit conversion 设计。
  - 当前波次提供 transport CLI，但只固化有限差分 velocity bridge 和 NPZ 输入/输出契约；完整 SI mobility/unit conversion 仍后续专项设计。
  - 支持 `TransportData` NPZ roundtrip。
  - 验收：manual SERTA reference、uniform kpoint weights、finite-difference velocity、EPC-to-SERTA workflow、invalid input rejection 和 NPZ schema 测试通过。

- Milestone 6: Advanced EPC physics
  - SCC-corrected EPC 单独设计；如果后续引入 SCC dftbephy reference，必须先完成该设计。
  - SOC/spinful EPC、polar correction、完整 degenerate-band gauge 稳定化作为专项推进。
  - 当前 v1 只提供 degenerate subspace coupling-strength aggregation diagnostic；不做 eigenvector gauge fixing 或沿 k/q 路径的连续 gauge 追踪。
  - 当前 v1 的 subspace diagnostic 支持 NPZ roundtrip，但只固化 contiguous band groups。

- Milestone 7: DeePTB-native workflow completeness
  - 不复刻 dftbephy 的 DFTB+ workflow 外壳；保留 dftbephy reference 只作为 benchmark/validation。
  - 设计 DeePTB-native EPC path workflow：沿 k/q path 输出可画图的 EPC、linewidth 或 relaxation-time 数据，主契约仍优先使用 NPZ，可选提供轻量 plot helper。
  - 设计 DeePTB-native EPC mesh workflow：面向 fine k/q mesh 的 linewidth、relaxation time 和 transport 输入输出组织。
  - 设计完整 transport/mobility workflow：补齐 SI unit conversion、carrier density、conductivity/mobility metadata 和 velocity provider 策略。
  - 根据需要补充 Eliashberg-like functions、phonon DOS、mode-resolved scattering 等分析核，但以 DeePTB API 组织，不追求 dftbephy JSON/HDF5 字段级兼容。

## Test Plan

- Unit tests:
  - Python `Phonons` object 和 DeePTB phonon-mode NPZ 的 shape/unit validation。
  - `compute_coupling_matrix` 的 overlap correction、frequency prefactor、band selection、positive mass/negative frequency/band index validation。
  - `EPCData` NPZ roundtrip，并确认默认不要求 derivative/debug 大矩阵。
  - `LinewidthData` / `RelaxationTimeData` / `TransportData` NPZ roundtrip 和 schema metadata conflict rejection。
  - `dptb eph` CLI parser、kpoints/weights input loader、EPC output NPZ、linewidth output NPZ、relaxation-time output NPZ、transport output NPZ、subspace output NPZ、SCC rejection。
  - linewidth 的 manual-reference numerical tests、negative frequency rejection 和 acoustic zero-mode floor tests。
  - relaxation time 的 `tau = hbar / (2 * linewidth)` convention、mode-resolved shape preservation 和 optional mode summation tests。
  - SERTA transport 的 manual-reference numerical tests。
  - finite-difference band velocity 和 `EPCData + LinewidthData -> TransportData` workflow tests。
  - degenerate-band subspace grouping、gauge-invariant coupling-strength aggregation 和 `SubspaceCouplingData` NPZ roundtrip tests。

- Regression tests:
  - 当前 `dptb/tests/test_electron_phonon.py` 全部保留。
  - 开发阶段以完整 dftbephy Graphene reference 作为主要 opt-in benchmark。
  - 当前开发波次的 EPC 核心验收必须跑 Graphene coupling-strength reference：
    `DEEPTB_RUN_REFERENCE_EPH=1 MPLCONFIGDIR=/private/tmp/matplotlib-cache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv/bin/python -m pytest dptb/tests/test_electron_phonon.py::test_graphene_reference_case_coupling_strength -q`
  - 如果修改 phase convention、Fourier projection、finite difference provider、overlap correction 或 unit prefactor，必须额外跑 slow Graphene supercell finite-difference reference：
    `DEEPTB_RUN_REFERENCE_EPH=1 DEEPTB_RUN_SLOW_EPH=1 MPLCONFIGDIR=/private/tmp/matplotlib-cache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv/bin/python -m pytest dptb/tests/test_electron_phonon.py::test_graphene_reference_case_supercell_fd_provider -q`
  - TODO before merge/release: 审查外部 reference 路径策略，避免默认 CI 依赖本机 checkout。
  - slow supercell finite-difference reference 继续作为 opt-in benchmark。

- CLI tests:
  - 用最小外部 phonon mode fixture 跑 `dptb eph`。
  - 检查输出 shape、单位、metadata、错误信息。

## Assumptions

- DeePTB EPC 不计算 phonons，只读取外部 phonon 数据。
- `phonopy` 是 optional reader，不是 DeePTB 内部 phonon solver。
- v1 不支持 SCC EPC。
- EPC 开发应尽量避开 `OrbitalMapper` 和 Hamiltonian transform 核心代码；需要索引信息时优先通过现有 metadata 或 EPC 薄适配层获取。
- 开发期 Graphene reference 可依赖本地未追踪资产和硬编码路径；最终默认测试数据和路径策略另行收敛。
- 本文档是开发计划；v1 workflow usage 和 NPZ contract 见 `docs/epc_v1_workflow.md`。
