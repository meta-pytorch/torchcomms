# Per-Communicator Config Override: Implementation Summary

## Motivation

NCCL's `NCCL_BUFFSIZE`, `NCCL_IB_SPLIT_DATA_ON_QPS`, and `NCCL_IB_QPS_PER_CONNECTION` are process-global env vars. In multi-tenant or multi-workload scenarios (e.g., MoE training with separate communicators for expert vs. non-expert traffic), different communicators benefit from different tuning. This effort makes all three configurable per-communicator via the existing `ncclx::Hints` mechanism.

## User-Facing API

```cpp
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
ncclx::Hints hints({
    {"ncclBuffSize", "8388608"},       // 8 MiB Simple protocol buffer
    {"ibSplitDataOnQps", "1"},         // stripe data across all QPs
    {"ibQpsPerConnection", "2"}        // 2 QPs per IB device
});
config.hints = &hints;
ncclCommSplit(parentComm, color, key, &childComm, &config);
```

All three are hint-only fields — no changes to `ncclConfig_v22800`, `NCCL_CONFIG_INITIALIZER`, or `ncclNetCommConfig_v11_t`. Global env vars continue to work as process-wide defaults.

## Architecture

### Data Flow

```
Hints({"ncclBuffSize", "8388608"})
  -> ncclxParseCommConfig()
    -> ncclx::Config constructor parses hints
    -> stores as std::optional<int> in Config
    -> comm->config.ncclxConfig = new Config(...)

ncclBuffSize path:
  -> initTransportsRank() -> computeBuffSizes(comm)
    -> reads NCCLX_CONFIG_FIELD(comm->config, ncclBuffSize)
    -> overrides comm->buffSizes[NCCL_PROTO_SIMPLE]

IB config path:
  -> commAlloc() -> ncclNetInit(comm)
    -> NcclxCommConfigScope sets static pointer (under netPluginMutex)
    -> ncclIbInit() reads pointer, populates NcclxIbNetCommConfig ctx
    -> scope destructor clears pointer
  -> ncclIbConnect(ctx, ...)
    -> ncclxIbCommInit(comm, ctx)           // applies splitDataOnQps
    -> ibResolveQpsPerConnection(ctx, env)   // resolves qpsPerConn
  -> ncclIbAccept(listenComm, ...)
    -> ncclxIbCommInit(rComm, lComm->ctx)   // applies splitDataOnQps
    -> ibResolveQpsPerConnection(ctx, env)   // resolves qpsPerConn
```

### File Organization

All NCCLX logic lives in `meta/` to minimize baseline exposure:

| File | Role |
|------|------|
| `meta/NcclxConfig.h` | `std::optional<int>` fields: `ncclBuffSize`, `ibSplitDataOnQps`, `ibQpsPerConnection` |
| `meta/NcclxConfig.cc` | Hint parsing with validation (positive-int for buffSize/QPS, 0-or-1 for splitData) |
| `meta/transport/NcclxNetPluginHelper.h` | `NcclxCommConfigScope` RAII class, `ncclxGetCurrentCommConfig()` declaration |
| `meta/transport/NcclxNetPluginHelper.cc` | Static side-channel variable and implementations |
| `meta/transport/NcclxIbNetCommConfig.h` | `NcclxIbNetCommConfig` struct (superset of `ncclNetCommConfig_t`), `ibResolveQpsPerConnection()`, `ibResolveSplitDataOnQps()`, `ncclxIbCommInit()` template |

Baseline modifications are documented in `meta/baseline_modification_docs/PerCommConfig.md`.

### Key Design Decisions

**1. Hint-only fields (no flat `ncclConfig_t` fields)**

Follows the established pattern used by `pipesNvlChunkSize`, `pipesUseDualStateBuffer`, and `vCliqueSize`. Avoids ABI changes and keeps `NCCL_CONFIG_INITIALIZER` unchanged.

**2. Static variable side-channel for IB config**

`ncclIbInit()` receives only `ncclNetCommConfig_t*` (just `trafficClass`). Rather than extending this plugin API struct, we pass the full `ncclx::Config` via a static `const ncclConfig_t*` pointer protected by the existing `netPluginMutex`. A RAII class (`NcclxCommConfigScope`) ensures the pointer is always cleared, even on early returns.

**3. NcclxIbNetCommConfig as extended ctx**

`ncclIbInit()` allocates `NcclxIbNetCommConfig` instead of `ncclNetCommConfig_t`. This struct carries `trafficClass` (for backward compatibility) plus `ibSplitDataOnQps` and `ibQpsPerConnection`. The ctx flows through the existing `void*` plumbing to `ncclIbConnect`, `ncclIbListen`, and `ncclIbAccept`.

**4. Template-based ncclxIbCommInit**

`ncclxIbCommInit<T>()` is a template in `NcclxIbNetCommConfig.h` that applies the `splitDataOnQps` override. Using a template avoids including `common.h` (heavy IB transport header) from `meta/transport/`. The template is instantiated in `connect.cc` where `ncclIbSendComm`/`ncclIbRecvComm` are fully defined.

**5. ncclBuffSize rejects splitShare=1**

With `splitShare=1`, the child comm shares the parent's proxy state and transport buffers. Different buffer sizes would corrupt shared state, so the override returns `ncclInvalidArgument`.

## Validation

**Unit tests** (`meta/hints/tests/ConfigHintsUT.cc`):
- `ncclBuffSize`: valid value, default unset, rejects negative/zero/invalid string (5 tests)
- `ibSplitDataOnQps`: valid 0/1, rejects invalid (>1), default unset (4 tests)
- `ibQpsPerConnection`: valid positive, rejects zero/negative, default unset (4 tests)

**Integration tests** (`meta/tests/CommSplitBuffSizeTest.cc`, 1x4 GPU):
- `PerCommBuffSizeOverride`: child comm gets custom `buffSizes[NCCL_PROTO_SIMPLE]`
- `ParentBuffSizeUnchanged`: parent comm unaffected by child's override
- `DefaultBuffSizeWithoutHint`: child inherits parent's buffer size
- `SplitShareRejectsBuffSizeOverride`: `splitShare=1` returns `ncclInvalidArgument`

IB config integration testing requires multi-node IB hardware and is done via MAST nccl-tests jobs with `NCCL_DEBUG=INFO` log verification.

## What Does NOT Change

- `ncclConfig_v22800` struct / `NCCL_CONFIG_INITIALIZER`
- `ncclNetCommConfig_v11_t` plugin API struct
- Global env vars (`NCCL_BUFFSIZE`, `NCCL_IB_SPLIT_DATA_ON_QPS`, `NCCL_IB_QPS_PER_CONNECTION`)
- LL / LL128 buffer sizes
- `parseCommConfig()` / `envConfigOverride()` / `deepCopyCommConfig()`
- GIN IB transport (only IB net transport is modified)
- `isend` / `irecv` data path
