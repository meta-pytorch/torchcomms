# Per-Comm Config: Baseline Modifications

## Overview

Allow `NCCL_BUFFSIZE`, `NCCL_IB_SPLIT_DATA_ON_QPS`, and `NCCL_IB_QPS_PER_CONNECTION` to be configured per-communicator via `ncclx::Hints` and `ncclx::Config`, instead of process-global env vars only. All NCCLX-specific changes in baseline files are tagged with `[NCCLX-PerCommConfig]` comments.

## Design Principles

1. **Minimize baseline exposure**: NCCLX logic lives in `meta/` and `meta/transport/`. Baseline files contain only thin call-sites with tagged comments.
2. **No changes to `ncclConfig_v22800` or `ncclNetCommConfig_v11_t`**: New fields are hint-only in `ncclx::Config`.
3. **Static variable side-channel under mutex**: IB config passes from `ncclx::Config` to the IB transport's per-comm ctx via a RAII-scoped static pointer, protected by the existing `netPluginMutex`.
4. **Per-comm ctx replacement**: `ncclIbInit` allocates `ncclx::NcclxIbNetCommConfig` (superset of `ncclNetCommConfig_t`) as the ctx, carrying `trafficClass` + IB overrides.

## Baseline Files Modified

### 1. `src/init.cc` — Per-comm NCCL_BUFFSIZE

**Function**: `computeBuffSizes()`

**Change**: After the existing loop that sets `comm->buffSizes[p]` from env/defaults, check `ncclx::Config::ncclBuffSize`. If set and `splitShare == 0`, override `comm->buffSizes[NCCL_PROTO_SIMPLE]`. If `splitShare == 1`, return `ncclInvalidArgument`.

```cpp
// [NCCLX] Per comm buffer size overwrite logic
if (comm->config.ncclxConfig) {
    auto& configBuffSize = NCCLX_CONFIG_FIELD(comm->config, ncclBuffSize);
    if (configBuffSize.has_value()) {
      if (comm->config.splitShare) { ... return ncclInvalidArgument; }
      comm->buffSizes[NCCL_PROTO_SIMPLE] = configBuffSize.value();
    }
}
```

**Why in baseline**: `computeBuffSizes()` runs during `initTransportsRank()` (core NCCL init path). The `comm->buffSizes[]` array is set here and consumed by all transports. Moving this to meta/ would require duplicating the function or adding an awkward hook.

### 2. `src/plugin/net.cc` — Config side-channel RAII scope

**Include added**: `meta/transport/NcclxNetPluginHelper.h`

**Function**: `ncclNetInit()`

**Change**: After acquiring `netPluginMutex`, create an `ncclx::NcclxCommConfigScope` RAII object that makes `&comm->config` available to `ncclIbInit()` via `ncclxGetCurrentCommConfig()`. The scope auto-clears when `ncclNetInit()` returns (including early returns via `NCCLCHECK`).

```cpp
std::lock_guard<std::mutex> lock(netPluginMutex);
// [NCCLX-PerCommConfig] Make comm config available to ncclIbInit via side-channel
ncclx::NcclxCommConfigScope configScope(&comm->config);
```

**Why in baseline**: `ncclNetInit()` is the only place that holds `netPluginMutex` while calling `ncclIbInit()`. The RAII scope is a single line.

### 3. `src/transport/net_ib/init.cc` — Extended IB ctx allocation

**Includes added**: `meta/NcclxConfig.h`, `meta/transport/NcclxIbNetCommConfig.h`, `meta/transport/NcclxNetPluginHelper.h`

**Function**: `ncclIbInit()`

**Change**: Instead of `ncclCalloc(&netCommConfig, 1)` for a plain `ncclNetCommConfig_t`, allocates `ncclx::NcclxIbNetCommConfig` (which is a superset containing `trafficClass` + `ibSplitDataOnQps` + `ibQpsPerConnection`). Reads per-comm IB fields from the side-channel via `ncclxGetCurrentCommConfig()` → `NCCLX_CONFIG_FIELD()`.

```cpp
// [NCCLX-PerCommConfig] Allocate extended ctx with per-comm IB overrides
auto* ncclxConfig = new ncclx::NcclxIbNetCommConfig();
ncclxConfig->trafficClass = config->trafficClass;
const ncclConfig_t* commConfig = ncclxGetCurrentCommConfig();
if (commConfig && commConfig->ncclxConfig) { ... populate overrides ... }
*ctx = (void*)ncclxConfig;
```

**Function**: `ncclIbFinalize()`

**Change**: Uses `delete static_cast<ncclx::NcclxIbNetCommConfig*>(ctx)` instead of `free(ctx)`.

**Why in baseline**: `ncclIbInit` and `ncclIbFinalize` are the IB transport's ctx lifecycle functions. The ctx type change is the core mechanism for carrying per-comm IB config.

### 4. `src/transport/net_ib/common.h` — ctx field on ncclIbListenComm

**Change**: Added `void* ctx` field to `struct ncclIbListenComm`.

```cpp
struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage* stage;
  void* ctx; // [NCCLX-PerCommConfig] per-comm config ctx, set by ncclIbListen
};
```

**Why in baseline**: `ncclIbAccept` receives `listenComm` (not `ctx`). The ctx must be stored in the listen comm so `ncclIbAccept` can read per-comm IB config from it.

### 5. `src/transport/net_ib/connect.cc` — Per-comm IB config application

**Include added**: `meta/transport/NcclxIbNetCommConfig.h`

**5a. `ncclIbListen()`**: Stores `ctx` in the listen comm.

```cpp
comm->ctx = ctx; // [NCCLX-PerCommConfig] store ctx for ncclIbAccept
```

**5b. `ncclIbConnect()`**: Three changes:

1. **splitDataOnQps override** — after `ncclIbSendCommInit(comm)`, calls `ncclx::ncclxIbCommInit(comm, ctx)` to apply the per-comm `splitDataOnQps` override.

2. **qpsPerConn resolution** — replaces `ncclParamIbQpsPerConn()` with `ncclx::ibResolveQpsPerConnection((ncclx::NcclxIbNetCommConfig*)ctx, ncclParamIbQpsPerConn())` for computing `localNqps`, `remoteNqps`, and `cqSize`.

3. **Traffic class ctx cast** — casts `ctx` as `ncclx::NcclxIbNetCommConfig*` instead of `ncclNetCommConfig_t*` when reading `trafficClass` for `meta.sl` / `meta.tc`.

**5c. `ncclIbAccept()`**: Two changes (mirror of 5b.1 and 5b.2):

1. **splitDataOnQps override** — after `ncclIbRecvCommInit(rComm)`, calls `ncclx::ncclxIbCommInit(rComm, lComm->ctx)`.

2. **qpsPerConn resolution** — replaces `ncclParamIbQpsPerConn()` with `ncclx::ibResolveQpsPerConnection((ncclx::NcclxIbNetCommConfig*)lComm->ctx, ncclParamIbQpsPerConn())` for computing `localNqps`, `remoteNqps`, and `cqSize`.

**Why in baseline**: `ncclIbConnect` and `ncclIbAccept` are where QPs are created and `splitDataOnQps`/`nqps`/`cqSize` are computed. The values must be set at connection time.

## NCCLX meta/ Files (not baseline)

These files contain the NCCLX-side logic and are not part of NCCL baseline:

| File | Purpose |
|------|---------|
| `meta/NcclxConfig.h` | `ncclx::Config` fields: `ncclBuffSize`, `ibSplitDataOnQps`, `ibQpsPerConnection` |
| `meta/NcclxConfig.cc` | Hint parsing for all three fields |
| `meta/transport/NcclxNetPluginHelper.h` | `NcclxCommConfigScope` RAII class, `ncclxGetCurrentCommConfig()` declaration |
| `meta/transport/NcclxNetPluginHelper.cc` | Static side-channel variable and accessor implementation |
| `meta/transport/NcclxIbNetCommConfig.h` | `NcclxIbNetCommConfig` struct, `ibResolveQpsPerConnection()`, `ibResolveSplitDataOnQps()`, `ncclxIbCommInit()` template |

## Thread Safety

- `s_ncclxCurrentCommConfig` is accessed only under `netPluginMutex` (set by `NcclxCommConfigScope` in `ncclNetInit()`, read by `ncclIbInit()` synchronously within the same mutex scope).
- `ncclIbConnect`/`ncclIbAccept` do NOT access the static variable — they read from the per-comm ctx stored in `ncclIbListenComm::ctx`.

## Revert Checklist

To remove per-comm config from the baseline:

1. `src/init.cc`: Remove the `[NCCLX] Per comm buffer size overwrite logic` block in `computeBuffSizes()`
2. `src/plugin/net.cc`: Remove the `NcclxNetPluginHelper.h` include and `NcclxCommConfigScope` line
3. `src/transport/net_ib/init.cc`: Revert `ncclIbInit()` to allocate `ncclNetCommConfig_t` via `ncclCalloc`; revert `ncclIbFinalize()` to `free(ctx)`; remove meta/ includes
4. `src/transport/net_ib/common.h`: Remove `void* ctx` from `ncclIbListenComm`
5. `src/transport/net_ib/connect.cc`: Remove `NcclxIbNetCommConfig.h` include; remove `comm->ctx = ctx` in `ncclIbListen`; remove `ncclxIbCommInit` calls; revert `qpsPerConn`/`qpsPerConnAccept` to use `ncclParamIbQpsPerConn()` directly; revert ctx cast to `ncclNetCommConfig_t*`
