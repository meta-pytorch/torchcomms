# Colltrace Test Migration: Version-Specific Changes

This document lists the version-specific differences found during the
consolidation of colltrace tests from `v2_27/meta/colltrace/tests/`,
`v2_28/meta/colltrace/tests/`, and `v2_29/meta/colltrace/tests/` into the
shared `meta/colltrace/tests/` directory.

## Source of Truth

The v2_29 test files were used as the source of truth (`sl mv` from v2_29).
v2_28 and v2_27 copies were deleted.

## Test Source File Differences

### Identical across all versions (no changes needed)

- `CollTraceWatchdogIbMockTest.cc`
- `DumpNewColltraceUT.cc`
- `MapperTraceDistTest.cc`
- `NewCollTraceUT.cc`
- `ProxyMockUT.cc`
- `ProxyTraceDistTest.cc`
- `SlowCollReporterUT.cc`

### `AlgoStatsTest.cpp`

- **v2_27**: Does not exist (AlgoStats feature not available)
- **v2_28 / v2_29**: Identical
- **Resolution**: Restrict to `versions = ["2.28", "2.29"]` in BUCK

### `CollTraceWatchdogTest.cc`

- **v2_27 / v2_28**: `ncclPutSignal(...)` and `ncclWaitSignal(...)` (unqualified)
- **v2_29**: `ncclx::ncclPutSignal(...)` and `ncclx::ncclWaitSignal(...)` (namespace-qualified)
- **Resolution**: `#if NCCL_MINOR >= 29` guard around the namespace-qualified calls

### `CollTraceDistTest.cc`

- **`ncclx::` namespace**: Same as CollTraceWatchdogTest.cc (v2_29 only). Guarded with `#if NCCL_MINOR >= 29`.
- **`NcclCommRAII` constructor**: v2_28 used 3-arg form (`{this->globalRank, this->numRanks, this->localRank}`), v2_27/v2_29 use 4-arg form (`{globalRank, numRanks, localRank, bootstrap_.get()}`). The shared `NcclCommUtils.h` already uses the 4-arg form, so the v2_29 source works for all versions. No guard needed.
- **`oobBarrier()` vs `MPI_Barrier(MPI_COMM_WORLD)`**: v2_28 used `MPI_Barrier`, v2_27/v2_29 use `oobBarrier()`. Same resolution — v2_29 form works for all. No guard needed.

### `NewCollTraceDistTestNoLocal.cc`

- Same patterns as `CollTraceDistTest.cc`:
  - `ncclx::` namespace guard for PutSignal/WaitSignal
  - `NcclCommRAII` and `oobBarrier` changes work for all versions

### `NewCollTraceDistTestLocal.cc`

- **v2_27**: Missing 4 lines (`ctranWinFree` + `cudaDeviceSynchronize` cleanup). v2_28/v2_29 have them.
- **`NcclCommRAII` / `oobBarrier`**: Same as above, v2_29 form works for all.
- **Resolution**: No guard needed — the v2_29 source is correct for all versions.

### `CollTraceWrapperUT.cc`

- **v2_27 / v2_28**: Missing `CollTraceInitConfigTest` parameterized test fixture (~63 lines)
- **v2_29**: Has the test fixture which tests `newCollTraceInit()` with various NCCL_COLLTRACE config combinations
- **Resolution**: `#if NCCL_MINOR >= 28` guard. The test accesses `comm->algoStats` which doesn't exist in v2_27's `comm.h`.

## BUCK File Changes

The per-version BUCK files (each loading its own `def_build.bzl` with version-specific deps) were replaced by a single shared BUCK file using `def_meta_test.bzl` macros:

- `nccl_cpp_unittest` → `ncclx_meta_unittest` (auto-generates `_v2_27`, `_v2_28`, `_v2_29` targets)
- `nccl_cpp_distributed_unittest` → `ncclx_meta_distributed_unittest`
- Version-pinned deps (`//comms/ncclx:nccl2.XX-internal`) → `nccl_deps = ["internal"]`
- Version-suffixed deps (`scuba_logger_test_mixin_v2_XX`) → `version_deps = ["...:scuba_logger_test_mixin"]`
- `cpp_unittest` for `slow_coll_reporter_ut` → `ncclx_meta_unittest` with `nccl_deps = ["internal"]`
