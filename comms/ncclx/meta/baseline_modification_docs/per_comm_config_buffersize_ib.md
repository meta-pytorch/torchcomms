# Per-Comm Config: Baseline Modifications

## Overview

Allow `NCCL_BUFFSIZE`, `NCCL_IB_SPLIT_DATA_ON_QPS`, and `NCCL_IB_QPS_PER_CONNECTION` to be configured per-communicator via `ncclx::Hints` and `ncclx::Config`, instead of process-global env vars only. All NCCLX-specific changes in baseline files are tagged with `[NCCLX-PerCommConfig]` comments.

## Design Principles

1. **Minimize baseline exposure**: NCCLX logic lives in `meta/` and `meta/transport/`. Baseline files contain only thin call-sites with tagged comments.
2. **No changes to `ncclConfig_v22800` or `ncclNetCommConfig_v11_t`**: New fields are hint-only in `ncclx::Config`.
3. **Static variable side-channel under mutex**: IB config passes from `ncclx::Config` to the IB transport's per-comm ctx via a RAII-scoped static pointer, protected by the existing `netPluginMutex`.
4. **Per-comm ctx replacement**: `ncclIbInit` allocates `ncclx::NcclxIbNetCommConfig` as the ctx, carrying `trafficClass` + IB overrides. It is not a subtype of `ncclNetCommConfig_t`; it only starts with the same `int trafficClass`, which is all `ncclNetCommConfig_t` holds today. v2_32 `static_assert`s that layout, so a rebase that grows the net comm config fails to compile.

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

**v2_32**: the inline `splitShare` check is replaced by a call into shared code:

```cpp
// [NCCLX-PerCommConfig] Validate and apply per-comm overrides
NCCLCHECK(ncclxValidatePerCommConfig(comm->config));
if (comm->config.ncclxConfig) {
  auto& configBuffSize = NCCLX_CONFIG_FIELD(comm->config, ncclBuffSize);
  if (configBuffSize.has_value()) {
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

**v2_32**: allocates with `NEW_NOTHROW(ncclxConfig, ncclx::NcclxIbNetCommConfig)`, which logs and returns `ncclSystemError` on failure, and `static_assert`s that `NcclxIbNetCommConfig` starts with `ncclNetCommConfig_t`'s layout.

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

**5d. v2_32**: upstream 2.32 splits each entry point into a thin wrapper and an `Impl` that takes the QP count (and, for connect, the traffic class) as arguments. The NCCLX changes follow that split:

- `ncclIbConnect()` / `ncclIbAccept()` wrappers resolve the QP count once with `ncclx::ibResolveQpsPerConnection(..., ncclParamIbQpsPerConn())` (from `ctx` and `lComm->ctx`) and pass it to the `Impl`.
- `ncclIbConnectImpl()` / `ncclIbAcceptImpl()` call `ncclx::ncclxIbCommInit()` after `ncclIbSendCommInit` / `ncclIbRecvCommInit` to apply `splitDataOnQps`. This is the only place the ctx is read as `NcclxIbNetCommConfig`.
- No traffic-class cast change: `ncclIbGetTrafficClass` already reads the ctx as `ncclNetCommConfig_t*` upstream.

**Why in baseline**: `ncclIbConnect` and `ncclIbAccept` are where QPs are created and `splitDataOnQps`/`nqps`/`cqSize` are computed. The values must be set at connection time.

### 6. `src/transport/net_ib/gin.cc` — Extended IB ctx allocation (GIN path)

**Include added**: `meta/transport/NcclxIbNetCommConfig.h`

**Function**: `ncclGinIbInitType()`

**Change**: Same ctx upgrade as `ncclIbInit()`. On v2_30 the GIN backend routes through `ncclIbListen`/`ncclIbConnect`/`ncclIbAccept`, which cast `ctx` to `ncclx::NcclxIbNetCommConfig*`. On v2_32 it calls `ncclIbConnectImpl`/`ncclIbAcceptImpl` directly with 1 QP per device, and the read that needs the extended type is `ncclx::ncclxIbCommInit` inside them. Either way, allocating a plain `ncclNetCommConfig_t` here would read past the allocation, so the ctx must be `NcclxIbNetCommConfig`. The per-comm QP override does not apply to GIN; only `splitDataOnQps` does.

```cpp
// [NCCLX-PerCommConfig] Allocate NcclxIbNetCommConfig (not ncclNetCommConfig_t)
auto* ncclxConfig = new (std::nothrow) ncclx::NcclxIbNetCommConfig();
if (ncclxConfig == nullptr) {
  return ncclSystemError;
}
ncclxConfig->trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
*ctx = ncclxConfig;
```

**Function**: `ncclGinIbFinalize()`

**Change**: Uses `delete static_cast<ncclx::NcclxIbNetCommConfig*>(ctx)` instead of `free(ctx)`.

**Why in baseline**: GIN allocates its own IB ctx and passes it into the shared IB transport entry points, so it must match the ctx type those entry points expect.

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

1. `src/init.cc`: Remove the `[NCCLX] Per comm buffer size overwrite logic` block in `computeBuffSizes()` (v2_32: the `[NCCLX-PerCommConfig] Validate and apply per-comm overrides` block, including the `ncclxValidatePerCommConfig` call)
2. `src/plugin/net.cc`: Remove the `NcclxNetPluginHelper.h` include and `NcclxCommConfigScope` line
3. `src/transport/net_ib/init.cc`: Revert `ncclIbInit()` to allocate `ncclNetCommConfig_t` via `ncclCalloc`; revert `ncclIbFinalize()` to `free(ctx)`; remove meta/ includes
4. `src/transport/net_ib/common.h`: Remove `void* ctx` from `ncclIbListenComm`
5. `src/transport/net_ib/connect.cc`: Remove `NcclxIbNetCommConfig.h` include; remove `comm->ctx = ctx` in `ncclIbListen`; remove `ncclxIbCommInit` calls; revert `qpsPerConn`/`qpsPerConnAccept` to use `ncclParamIbQpsPerConn()` directly; revert ctx cast to `ncclNetCommConfig_t*`. v2_32: remove `comm->ctx = ctx` and the two `ncclxIbCommInit` calls in the `Impl`s, and restore the upstream wrapper bodies:

   ```cpp
   return ncclIbConnectImpl(ctx, dev, opaqueHandle, sendComm, sendDevComm, ncclParamIbQpsPerConn(), ncclParamIbTc());
   return ncclIbAcceptImpl(listenComm, recvComm, recvDevComm, ncclParamIbQpsPerConn());
   ```

   Do steps 4 and 5 together: the wrappers reference `lComm->ctx`.
6. `src/transport/net_ib/gin.cc`: Remove the `NcclxIbNetCommConfig.h` include; revert `ncclGinIbInitType()` to allocate `ncclNetCommConfig_t` via `ncclCalloc`; revert `ncclGinIbFinalize()` to `free(ctx)`
