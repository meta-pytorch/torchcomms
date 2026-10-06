# Structured Logging (spdlog + Scuba): Baseline Modifications

## Background

NCCLX routes every NCCL log line through Meta's shared comms logging stack
(`comms/utils/logger/**`) instead of upstream's `vfprintf(ncclDebugFile, ...)`.
That gives NCCL logs the same spdlog sink, formatter, and `NCCL_DEBUG_FILE`
handling as CTRAN, and fans root-cause errors out to the
`nccl_structured_logging` Scuba table.

Two things are layered on top of the plain redirect:

- A dedicated root-cause error macro `ERR(code, ...)`, which carries the
  `ncclResult_t`. Upstream signals errors with `WARN(...); return code;`, so the
  code is lost at the log site and the root-cause message is indistinguishable
  from ordinary non-fatal warnings. `ERR` records the Scuba error record and
  `ncclGetLastError()` state exactly once, at the origin, instead of at every
  propagating check-macro layer.
- A `NCCL_LOG_ERROR` severity between `NCCL_LOG_VERSION` and `NCCL_LOG_WARN`, so
  oncall automation that keys off ERROR severity can find the real root cause.

Design rationale and the cleanup plan live in
[`comms/ncclx/docs/error_logging.md`](../../docs/error_logging.md).

## Versions Affected

v2_30, v2_32 (and the removed v2_29 / v2_31)

## What is shared, and what has to be forked

Most of the machinery is version-independent and needs no port:

| Concern | Lives in |
|---|---|
| spdlog sink, formatter, async/file routing | `comms/utils/logger/SpdlogLogger.{h,cc}` |
| Scuba error record | `comms/utils/logger/CommsLogging.cc` (`logErrorToScuba`) |
| `ncclGetLastError()` state, native stack | `comms/utils/logger/{CommsLogging,ErrorStackUtil}.cc` |
| `ERR` implementation (`ncclMetaDebugLogError`) | `comms/ncclx/meta/logger/DebugExt.cc` |
| Level mapping + sink entry point (`writeNcclLog`) | `comms/ncclx/meta/logger/NcclDebugLog.h` |

Only the upstream-derived files below are forked per version.

## Baseline Files Modified

### 1. `src/include/nccl_common.h` — new severity

**Change**: Inserted `NCCL_LOG_ERROR = 2` into `ncclDebugLogLevel`, renumbering
`WARN`/`INFO`/`ABORT`/`TRACE` to 3/4/5/6.

**Why in baseline**: `ncclDebugLogLevel` is the type every logging entry point
takes, including the shared `writeNcclLog` switch in `meta/logger/`. The shared
code does not compile against a version that lacks `NCCL_LOG_ERROR`.

**Caveat**: the enumerator values are part of the `ncclDebugLogger_t` ABI passed
to net/tuner plugins. A plugin compiled against pristine upstream headers will
read a shifted level. This is accepted, and matches v2_29/v2_30.

**2.32 note**: upstream 2.32 appended `NCCL_LOG_ATTN = 6` ("logically between
WARN and INFO"). After the `ERROR` insertion `TRACE` takes 6, so v2_32 appends
`ATTN` as 7 instead, and defines `NCCL_LOG_HAS_ATTN` so the shared
`writeNcclLog` switch can map it (to spdlog `warn`) without breaking v2_30,
which has no `ATTN`.

### 2. `src/include/debug.h` — macros and declarations

**Change**: Declared `ncclMetaDebugLog`, `ncclMetaDebugLogError`, and
`ncclSetMyThreadLoggingName`; added the `ERR(code, ...)` macro and the
`NCCL_NAMED_THREAD_START[_EXT]` helpers; changed `VERSION`, `INFO`, `TRACE_CALL`,
and `TRACE` to pass `__FILE__, __func__, __LINE__` where upstream passes
`nullptr, nullptr, 0`.

**Why in baseline**: these are the macros every call site in the tree expands.

**2.31+ note**: upstream 2.31 introduced `ncclDebugLogInternal(level, flags, file,
func, line, fmt, ...)` — the same signature Meta had forked `ncclMetaDebugLog`
for. v2_31 and v2_32 therefore keep upstream's function in the macros and only widens the
arguments, rather than repointing every macro at a parallel symbol as v2_30 does.
`ncclMetaDebugLog` survives as a second entry point into the same funnel purely
because the shared `meta/logger/DebugExt.cc` links against that name in every
version.

**2.32 note**: upstream 2.32 replaced the scalar `ncclDebugLevel` with a
per-level bitmask (`ncclDebugLevelMask`, `ncclDebugShouldLog()`) so
`NCCL_DEBUG_LEVELS` can add levels individually. v2_32 keeps that gate and only
widens the arguments as above; the subsystem mask is read through
`ncclDebugMaskLoad()` (an atomic load, as in v2_30) because runtime
reconfiguration can republish it concurrently. `ncclResetDebugInitInternal()` and
`ncclRefreshDebugInitInternal()` are declared here for the reconfiguration path
below.

### 3. `src/debug.cc` — the funnel

**Change**: Replaced the tail of `ncclDebugLogV()` — timestamp/hostname/pid/tid
prefixing and `vfprintf` — with a `vsnprintf` of the caller's message followed by
`ncclx::logging::writeNcclLog(level, file, func, line, message)`. The sink owns
the line prefix and the output destination. Also: `NCCL_LOG_ERROR` joins
`NCCL_LOG_WARN` in the `ncclDebugNoWarn` downgrade and the `ncclLastError` save;
`ncclDebugInit()` publishes the subsystem mask via
`meta::comms::logger::setSubSystemMask()`; and `ncclMetaDebugLog` /
`ncclSetMyThreadLoggingName` are defined here.

Runtime reconfiguration (v2_30, v2_32): `ncclResetDebugInitInternal()` and the
new `ncclRefreshDebugInitInternal()` both go through `reconfigureDebugInit()`,
which re-runs `ncclDebugInit()` and `initNcclLogger(false)` so the native gate
and the named spdlog logger move together. The refresh reopens the active
`NCCL_DEBUG_FILE` in append mode so lines logged during env-plugin discovery are
not truncated. `ncclDebugInit(false, ...)` skips `setSubSystemMask()`, but
`initNcclLogger(false)` republishes it right after, so every reset and refresh
republishes the shared comms subsystem mask.

`ERROR` is accepted by `NCCL_DEBUG`, `NCCL_DEBUG_TIMESTAMP_LEVELS` and (2.32)
`NCCL_DEBUG_LEVELS`, and is enabled by every level from `ERROR` up.

Sink level (2.32): the native gate is a level mask, `NCCL_DEBUG` expanded to its
level set plus `NCCL_DEBUG_LEVELS`. `initNcclLogger()` sets the sink to the most
verbose level that mask enables, with `ATTN` counted as `WARN`, so the sink never
drops a line the mask passes. A scalar `NCCL_DEBUG` maps as on v2_30. The same
level gates `NCCLX_LOG` and CollTrace, which share the `comms.ncclx` logger.
`NCCL_WARN_ENABLE_DEBUG_INFO` is parsed but not applied, as on v2_30: the sink
level is fixed at configuration, so INFO lines enabled after the first WARN would
be formatted and then dropped.

**Why in baseline**: upstream has no registerable log sink — `ncclDebugLogV`
writes to `ncclDebugFile` directly, and `ncclDebugLogger_t` points the other way
(NCCL to plugins). Redirecting the output requires editing this function.

**Deliberately left in place**: `ncclDebugInit()` still parses
`NCCL_DEBUG_TIMESTAMP_LEVELS` / `NCCL_DEBUG_TIMESTAMP_FORMAT` /
`NCCL_DEBUG_FILE` and still caches `hostname`/`pid`, even though the sink now
supplies those. Keeping the upstream parsing intact keeps the fork to one
function. The timestamp params are accepted but inert: nothing reads
`ncclDebugTimestamp*` on the sink path, so `ERROR` in
`NCCL_DEBUG_TIMESTAMP_LEVELS` is a no-op too.

### 4. `src/include/checks.h` — check macros

**Change**: Added `ncclCodeToString()`; converted the root-cause check macros
(`CUDACHECK[GOTO]`, `SYSCHECK[GOTO]`, `PTHREADCHECK[GOTO]`, `NEQCHECK[GOTO]`,
`EQCHECK[GOTO]`, `CUDACHECKTHREAD`) from `WARN`/`INFO_LOC` to `ERR(code, ...)`;
raised the propagation macros (`NCCLCHECK[GOTO]`, `NCCLWAIT[GOTO]`,
`NCCLCHECKIGNORE`, `NCCLCHECKTHREAD`) from `INFO_LOC` to `WARN` so the
propagation chain stays visible without `NCCL_DEBUG=INFO`; and added
`CHECKABORT` and `CUDACHECKABORT`, which shared `meta/` code calls and so must
exist in every version dir. `v2_30`'s `SYSCHECKVAL` is deliberately not ported:
nothing in `v2_32/src`, `meta/`, or any consumer references it.

**Why in baseline**: `checks.h` is where the bulk of NCCL's error reporting
actually happens, so converting it here covers most call sites without touching
them individually.

**2.31+ note**: upstream 2.31 added `INFO_LOC`, which prepends `file:line (func)`
to the message. Since the sink now carries source location in the log record,
converted macros use the shorter 2.31+ message text rather than v2_30's explicit
`"%s:%d -> %d"` spelling.

### 5. `src/misc/param.cc`, `src/include/param.h`, `src/plugin/env.cc` — sink registration

**Change**: Added `initNcclLogger()` and called it from `initEnv()`, after
`meta::comms::initFolly()`, `ncclCvarInit()` and the `NCCL_CONF_FILE` / home /
`/etc/nccl.conf` processing, so the logger sees the same `NCCL_DEBUG*` values
native NCCL does. It reads them through the env plugin once that is published
(`getNcclLoggerEnv()`), and records success in `ncclLoggerInitialized()`.
`ncclGetEnv()` queries a published plugin directly rather than re-entering
`ncclInitEnv()`, so a logger reset during env init cannot recurse into the
active `call_once`. `plugin/env.cc` calls `ncclRefreshDebugInitInternal()` right
after the plugin is published, so plugin-supplied `NCCL_DEBUG*` settings take
effect.

**Why in baseline**: `initEnv()` is the one-time init hook upstream already runs
before any logging is configured. Registration has to happen there or the first
log lines are lost.

**Why it is not covered by CTRAN or MCCL initializing spdlog**:
`getSpdlogLogger(name)` is a name-keyed registry. CTRAN registers
`"comms.ctran"`; `writeNcclLog` looks up `"comms.ncclx"`. An unregistered name
is lazily default-constructed — no file sink, no `"NCCL"` prefix, no cudaDev
thread context, no `setLastError` hook — so the logs go nowhere quietly. Only
`initCommLogging()` is genuinely shared, and it is `folly::once`-guarded.

**Known duplication**: `initNcclLogger()` is near-identical to
`comms/ctran/utils/LogInit.cc`, differing only in the logger name and prefix
(one copy per version dir plus ctran). Hoisting a
`configureDomainLogger(name, prefix)` helper into `comms/utils/logger` would
collapse all four; that is a shared-code refactor, tracked separately.

### 6. `src/init.cc` — `ncclGetLastError()`

**Change**: returns the Meta last-error store (`getLastCommsError()`), which
`ERR` writes at the origin, instead of upstream's `ncclLastError[]`, which every
propagation `WARN` overwrites with the outermost `"-> %d"` frame.

**Why in baseline**: `ncclGetLastError()` is the upstream API callers read the
root cause from, and upstream fills its buffer from every `WARN`.

## Upstreaming

Two asks would retire almost all of the above, and both still hold against
upstream 2.32:

1. A result-code-carrying error macro (`ERR(code, ...)` / `WARNRET(code, ...)`),
   which removes the `debug.h` and `checks.h` forks and the per-rebase call-site
   conversions.
2. A registerable logging callback in `ncclDebugLog`, which removes the
   `debug.cc` fork entirely.
