# MNNVL Numeric Determinism: Baseline Modifications

## Background

On MNNVL systems (GB200/GB300) NCCL groups ranks into an NVLink clique by the `cliqueId` the NVML fabric reports, so the clique layout, and with it the ring/tree reduction order, depends on which racks the scheduler allocated. `NCCL_MNNVL_DETERMINISTIC_COLLECTIVE_ENABLE=1` with `NCCL_MNNVL_CLIQUE_SIZE=N` replaces the hardware `cliqueId` with `globalRank / N`, so the partition is a function of the job configuration and reductions are bitwise reproducible across restarts.

The arithmetic and its guardrails live in shared `meta/MnnvlCliqueId.{h,cc}` (`ncclx::assignMnnvlCliqueIdBasedOnCliqueSize`). CTRAN applies the same override to its own fabric view in `meta/commstate/FactoryCommStateX.cc` (`getLocalGpuFabricInfo`) and in `comms/ctran/commstate/CommStateX.cc`; neither is a baseline file.

## Versions Affected

v2.30, v2.32

## Baseline Files Modified

### `src/init.cc` — `fillInfo()`

**Change**: v2.32 includes `meta/MnnvlCliqueId.h` directly; v2.30 gets it through `meta/commstate/FactoryCommStateX.h`. In the MNNVL block, after the cluster UUID is read and before the upstream `NCCL_MNNVL_CLIQUE_ID` handling:

```cpp
if (NCCL_MNNVL_DETERMINISTIC_COLLECTIVE_ENABLE && NCCL_MNNVL_CLIQUE_SIZE <= 0) {
  WARN("NCCL_MNNVL_CLIQUE_SIZE must be set to a positive integer when NCCL_MNNVL_DETERMINISTIC_COLLECTIVE_ENABLE "
       "is set");
  return ncclInvalidArgument;
}
if (NCCL_MNNVL_DETERMINISTIC_COLLECTIVE_ENABLE && NCCL_MNNVL_CLIQUE_SIZE > 0) {
  int cliqueId = -1;
  ncclx::assignMnnvlCliqueIdBasedOnCliqueSize(&cliqueId);
  info->fabricInfo.cliqueId = cliqueId;
} else if (ncclParamMNNVLCliqueId() == -2) {  // upstream branch, now an else-if
```

**Why in baseline**: `fillInfo()` is where each rank publishes `ncclPeerInfo::fabricInfo`; `ncclMnnvlCheck()` (`mnnvl.cc`) and `ncclTopoCheckMNNVL()` (`graph/paths.cc`) form the clique from the exchanged `cliqueId`, so the override must be applied before the peer-info allgather.

## Build Integration

`meta/MnnvlCliqueId.cc` is listed in each version's `def_build.bzl`, and in `src/Makefile` for versions whose OSS Makefile builds the shared `meta/` sources.

## Revert Checklist

1. `src/init.cc`: remove the two determinism branches and restore `if (ncclParamMNNVLCliqueId() == -2)`; on v2.32 also remove the `meta/MnnvlCliqueId.h` include (v2.30 does not include it directly).
2. `def_build.bzl` / `src/Makefile`: remove `meta/MnnvlCliqueId.cc` once no caller remains (CTRAN's `FactoryCommStateX.cc` still calls it).
