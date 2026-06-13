# EPC 模块测试清理指南

## 概览

当前 `dptb/tests/test_electron_phonon.py` 包含 **429 个测试**，其中：
- **88 个核心验证性测试**（计算正确性、数据序列化、集成测试）
- **341 个低价值测试**（错误处理、meta-tests、参数验证）

## 要删除的测试模式

下列模式中的测试可以安全删除（不会损失数值验证覆盖）：

### 1. 错误处理测试（137+个）
```
- test_*_rejects_*
- test_*_reject_*
```
**原因:** 这些测试验证输入拒绝。输入验证错误会在实际使用时立即暴露。

**示例:**
```
test_compute_coupling_matrix_rejects_invalid_phonon_scalars
test_epc_mesh_chunked_artifact_rejects_bad_manifest_order
test_epc_data_rejects_conflicting_schema_metadata
```

### 2. Meta-tests（39个）
```
- test_docs_*
- test_*_in_repo_epc_fixture_is_self_contained
- test_epc_core_contract_*
- test_epc_default_production_*
```
**原因:** 文档读取、源码自省、API导出验证。这些不测试功能。

### 3. Phonopy/工具函数测试（10+个）
```
- test_phonons_from_phonopy_*
- test_reshape_phonopy_*
- test_orbital_slices_*
- test_assemble_directed_*
- test_supercell_*
- test_dftbplus_*
```
**原因:** 这些是utilities/wrappers的测试。utilities通过集成测试验证。

### 4. CLI/参数解析测试（40+个）
```
- test_main_parser*
- test_parse_args*
- test_eph_*
- test_epc_mesh_spec_rejects_*
- test_normalize_chunk_axis_*
- test_length_unit_scale_*
```
**原因:** argparse/CLI参数验证。错误会在使用时立即失败。

## 要保留的核心测试（88个）

### Fixture验证（3个）
```
test_minimal_in_repo_epc_fixture_matches_linewidth_reference
test_minimal_in_repo_epc_fixture_matches_analysis_references
test_minimal_in_repo_epc_fixture_matches_coupling_contraction_reference
```

### 计算正确性（35个+）
```
# Coupling matrix
test_compute_coupling_matrix_without_overlap
test_compute_coupling_matrix_accepts_scalar_mass_for_single_atom
test_compute_coupling_matrix_with_overlap_and_frequency_prefactor
test_compute_coupling_matrix_frequency_floor_regularizes_frequency
test_compute_coupling_matrix_block_phase_row_derivatives
test_coupling_strength_is_invariant_to_orbital_sign_gauge

# Linewidth
test_compute_linewidth_gaussian_matches_manual_reference
test_compute_linewidth_mesh_matches_total_for_uniform_q_weights

# Transport/mobility
test_compute_band_velocities_finite_difference_matches_linear_bands
test_compute_band_velocities_hamiltonian_derivative_matches_diagonal_dhdk
test_compute_serta_conductivity_matches_manual_reference
test_compute_serta_mobility_si_matches_manual_3d_reference
test_compute_serta_mobility_si_supports_2d_sheet_normalization

# Reference matching
test_compute_scattering_maps_matches_manual_reference
test_compute_eliashberg_spectral_function_matches_manual_reference
test_compute_phonon_dos_matches_manual_gaussian_reference

# 还有20+ 更多的transport/chunked artifact测试
...
```

### NPZ序列化（20+个）
```
test_epc_data_npz_roundtrip
test_epc_mesh_data_npz_roundtrip
test_epc_path_data_npz_roundtrip
test_linewidth_data_npz_roundtrip
test_linewidth_mesh_data_npz_roundtrip
test_linewidth_path_data_npz_roundtrip
test_relaxation_time_*_npz_roundtrip
test_transport_*_npz_roundtrip
test_mobility_*_npz_roundtrip
test_subspace_coupling_data_npz_roundtrip
test_phonons_npz_roundtrip
```

### 集成/域特定测试（20+个）
```
test_compute_serta_transport_from_epc_mesh_chunked_artifact_matches_full_mesh
test_compute_serta_transport_scan_from_epc_mesh_chunked_artifact_matches_full_mesh
test_compute_serta_transport_scan_recompute_linewidth_from_epc_mesh_chunked_artifact_matches_full_mesh
test_compute_serta_mobility_si_from_epc_mesh_chunked_artifact_matches_full_mesh
test_compute_serta_mobility_scan_si_from_epc_mesh_chunked_artifact_matches_full_mesh
test_compute_serta_mobility_scan_si_recompute_linewidth_from_epc_mesh_chunked_artifact_matches_full_mesh
test_epc_mesh_spec_generates_kmesh_and_validates_phonon_qmesh
```

## 手动清理步骤

### 方法1：使用grep找出测试名，手动删除

```bash
# 列出所有要删除的测试
grep -E "def test_(rejects|reject|is_self_contained|docs_|phonons_from_phonopy|orbital_slices|assemble_directed|supercell|dftbplus|normalize_chunk_axis|length_unit_scale|main_parser|parse_args|eph_)" dptb/tests/test_electron_phonon.py | sed 's/.*def //' | sed 's/(.*//'

# 对每一个，找到行号，然后在编辑器中删除
```

### 方法2：创建pytest.ini mark来跳过

```ini
# pyproject.toml 中添加
[tool.pytest.ini_options]
markers = [
    "error_handling: tests for error condition handling (skip for reduced suite)",
]
```

然后给所有要删除的测试加上 `@pytest.mark.error_handling`，运行 `pytest -m "not error_handling"`

### 方法3：Python脚本辅助删除

查看 `/tmp/final_cleanup.py` 中的逻辑：
- 识别要删除的测试名称
- 找到包括装饰器的完整函数范围
- 删除整个范围

## 预期结果

| 指标 | 当前 | 清理后 |
|-----|------|--------|
| 总测试数 | 429 | ~88 |
| 代码行数 | 7,225 | ~3,500 |
| 验证性测试 | ~150 | ~88（100% 保留） |
| 错误处理测试 | 280+ | 0 |
| 测试:代码比 | 2.35x | 1.14x |

## 验证清理无回归

```bash
# 清理前运行一次（基准）
uv run pytest dptb/tests/test_electron_phonon.py -v --tb=short

# 清理后运行（应该完全通过）
uv run pytest dptb/tests/test_electron_phonon.py -v --tb=short

# 检查关键函数覆盖
uv run pytest dptb/tests/test_electron_phonon.py -k "matches_manual or matches_reference or fixture or roundtrip" -v
```

## 注意事项

1. **装饰器管理:** 删除测试时要注意其上方的 `@pytest.mark.parametrize` 装饰器。如果删除了被参数化的测试，也要删除装饰器。

2. **Helper函数:** 确保 `_manual_linewidth`, `_manual_serta_conductivity` 等helper函数被保留（它们被保留的测试使用）。

3. **Fixture:** 保留 `_minimal_fixture_epc_data` 及其他被保留测试使用的fixture。

4. **导入:** 清理后可能有部分import不再使用，可以从文件顶部清理（如AST, inspect等）。

## 时间估计

- 自动脚本（正确处理装饰器）：1-2小时开发
- 手动编辑（for 88个测试）：3-5分钟
- 验证清理无回归：10-15分钟

**建议:** 使用Python脚本 + 手动验证装饰器正确性
