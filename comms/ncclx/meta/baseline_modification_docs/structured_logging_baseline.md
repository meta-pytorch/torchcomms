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

v2_29, v2_30, v2_31

## What is shared, and what has to be forked

Most of the machinery is version-independent and needs no port:

| Concern | Lives in |
|---|---|
| spdlog sink, formatter, async/file routing | `comms/utils/logger/SpdlogLogger.{h,cc}` |
| Scuba error record | `comms/utils/logger/LoggingFormat.cc` (`logErrorToScuba`) |
| `ncclGetLastError()` state, native stack | `comms/utils/logger/{LoggingFormat,ErrorStackUtil}.cc` |
| `ERR` implementation (`ncclMetaDebugLogError`) | `comms/ncclx/meta/logger/DebugExt.cc` |
| Level mapping + sink entry point (`writeNcclLog`) | `comms/ncclx/meta/logger/NcclDebugLog.h` |

Only the five upstream-derived files below are forked per version.

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

### 2. `src/include/debug.h` — macros and declarations

**Change**: Declared `ncclMetaDebugLog`, `ncclMetaDebugLogError`, and
`ncclSetMyThreadLoggingName`; added the `ERR(code, ...)` macro and the
`NCCL_NAMED_THREAD_START[_EXT]` helpers; changed `VERSION`, `INFO`, `TRACE_CALL`,
and `TRACE` to pass `__FILE__, __func__, __LINE__` where upstream passes
`nullptr, nullptr, 0`.

**Why in baseline**: these are the macros every call site in the tree expands.

**2.31 note**: upstream 2.31 introduced `ncclDebugLogInternal(level, flags, file,
func, line, fmt, ...)` — the same signature Meta had forked `ncclMetaDebugLog`
for. v2_31 therefore keeps upstream's function in the macros and only widens the
arguments, rather than repointing every macro at a parallel symbol as v2_30 does.
`ncclMetaDebugLog` survives as a second entry point into the same funnel purely
because the shared `meta/logger/DebugExt.cc` links against that name across all
three versions.

### 3. `src/debug.cc` — the funnel

**Change**: Replaced the tail of `ncclDebugLogV()` — timestamp/hostname/pid/tid
prefixing and `vfprintf` — with a `vsnprintf` of the caller's message followed by
`ncclx::logging::writeNcclLog(level, file, func, line, message)`. The sink owns
the line prefix and the output destination. Also: `NCCL_LOG_ERROR` joins
`NCCL_LOG_WARN` in the `ncclDebugNoWarn` downgrade and the `ncclLastError` save;
`ncclDebugInit()` publishes the subsystem mask via
`meta::comms::logger::setSubSystemMask()`; and `ncclMetaDebugLog` /
`ncclSetMyThreadLoggingName` are defined here.

**Why in baseline**: upstream has no registerable log sink — `ncclDebugLogV`
writes to `ncclDebugFile` directly, and `ncclDebugLogger_t` points the other way
(NCCL to plugins). Redirecting the output requires editing this function.

**Deliberately left in place**: `ncclDebugInit()` still parses
`NCCL_DEBUG_TIMESTAMP_LEVELS` / `NCCL_DEBUG_TIMESTAMP_FORMAT` /
`NCCL_DEBUG_FILE` and still caches `hostname`/`pid`, even though the sink now
supplies those. Keeping the upstream parsing intact keeps the fork to one
function and keeps the published params honest.

### 4. `src/include/checks.h` — check macros

**Change**: Added `ncclCodeToString()`; converted the root-cause check macros
(`CUDACHECK[GOTO]`, `SYSCHECK[GOTO]`, `PTHREADCHECK[GOTO]`, `NEQCHECK[GOTO]`,
`EQCHECK[GOTO]`, `CUDACHECKTHREAD`) from `WARN`/`INFO_LOC` to `ERR(code, ...)`;
raised the propagation macros (`NCCLCHECK[GOTO]`, `NCCLWAIT[GOTO]`,
`NCCLCHECKIGNORE`, `NCCLCHECKTHREAD`) from `INFO_LOC` to `WARN` so the
propagation chain stays visible without `NCCL_DEBUG=INFO`; and added
`CHECKABORT`, `CUDACHECKABORT`, and `SYSCHECKVAL`.

**Why in baseline**: `checks.h` is where the bulk of NCCL's error reporting
actually happens, so converting it here covers most call sites without touching
them individually.

**2.31 note**: upstream 2.31 added `INFO_LOC`, which prepends `file:line (func)`
to the message. Since the sink now carries source location in the log record,
converted macros use 2.31's shorter message text rather than v2_30's explicit
`"%s:%d -> %d"` spelling.

### 5. `src/misc/param.cc`, `src/include/param.h` — sink registration

**Change**: Added `initNcclLogger()` and called it from `initEnv()`, alongside
`meta::comms::initFolly()` and `ncclCvarInit()`.

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
(four copies across ctran + v2_29 + v2_30 + v2_31). Hoisting a
`configureDomainLogger(name, prefix)` helper into `comms/utils/logger` would
collapse all four; that is a shared-code refactor, tracked separately.

## Upstreaming

Two asks would retire almost all of the above, and both still hold against
upstream 2.31:

1. A result-code-carrying error macro (`ERR(code, ...)` / `WARNRET(code, ...)`),
   which removes the `debug.h` and `checks.h` forks and the per-rebase call-site
   conversions.
2. A registerable logging callback in `ncclDebugLog`, which removes the
   `debug.cc` fork entirely.
