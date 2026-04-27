# Per-Comm Config: Integration Test Plan

## Setup
- Use `noLocal` global hint to force IB transport on single-node devgpu (same pattern as `CommWithNoLocalTest`)
- 1x4 GPU config
- Create parent comm with defaults, then `ncclCommSplit` child with per-comm hints
- `splitShare=0` (default) ensures the child goes through `ncclNetInit()` (not `ncclNetInitFromParent`), which activates the side-channel for per-comm IB config

## Test File
`comms/ncclx/meta/tests/CommSplitIbConfigTest.cc`

## Test Cases

### 1. `NcclBuffSizeOverride`
- Create parent comm with `noLocal` hint (forces IB transport)
- Split child with `{"ncclBuffSize", "8388608"}`
- Assert `child->buffSizes[NCCL_PROTO_SIMPLE] == 8388608`
- Assert parent's `buffSizes[NCCL_PROTO_SIMPLE]` is unchanged
- Run AllReduce on **both** parent and child, verify data correctness on both

### 2. `IbConfigOverride`
- Create parent comm with `noLocal` hint
- Split child with `{"ibQpsPerConnection", "2"}, {"ibSplitDataOnQps", "1"}`
- Cast `child->netContext` to `NcclxIbNetCommConfig*`, assert `ibQpsPerConnection == 2`, `ibSplitDataOnQps == 1`
- Cast `parent->netContext` to `NcclxIbNetCommConfig*`, assert both fields are `NCCL_CONFIG_UNDEF_INT`
- Run AllReduce on **both** parent and child, verify data correctness on both

### 3. `DefaultHintsMatchParent`
- Create parent comm with `noLocal` hint
- Split child with no IB hints
- Verify child's `netContext` IB fields are `NCCL_CONFIG_UNDEF_INT`
- Run AllReduce on child, verify correctness

## Assertions

### Directly assertable
- `comm->buffSizes[NCCL_PROTO_SIMPLE]` — from `comm.h`
- `comm->netContext` cast to `NcclxIbNetCommConfig*` — verifies config propagated through side-channel into IB ctx

### Not easily assertable
- `base.nqps` on actual IB send/recv comms — buried inside proxy connections. `netContext` assertion + AllReduce completing without hang is sufficient: if `qpsPerConn=2` caused a sender/receiver nqps mismatch, the collective would deadlock.

## Key Invariant
- `splitShare=0` → `ncclNetInit()` path → side-channel works ✅
- `splitShare=1` → `ncclNetInitFromParent()` → copies parent's netContext → per-comm IB overrides ignored (correct: shared transport buffers can't have different config)
- `splitShare=1` + `ncclBuffSize` hint → already rejected with `ncclInvalidArgument` in `computeBuffSizes()`
- `splitShare=1` + IB hints → rejected by `ncclxValidatePerCommConfig()` in `computeBuffSizes()`
