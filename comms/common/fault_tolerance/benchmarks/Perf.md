# Abort Benchmark Results

Collected for `D113372521` after `Abort` moved to shared CUDA-mapped pinned
state and `AbortDevice` was added.

H100 was refreshed on 2026-08-19 on the same host as the previous run
(`devgpu012.mwg1`), against the rebased stack. The one material movement is the
device polling row, and it is worth calling out because it is the whole reason
the poll-interval gate exists:

| Row | 2026-08-12 | 2026-08-19 | |
|---|---:|---:|---|
| `AbortDeviceIsAbortedLoadLoop` | 225.93ms / 100K polls | **4.87ms / 100K polls** | 46x faster |
| `AbortDeviceIsAbortedWithDeadlineLoadLoop` | 226.28ms / 100K polls | **4.88ms / 100K polls** | 46x faster |

`AbortState` lives in mapped pinned host memory, so an ungated poll is an
uncached PCIe round trip — about 2.26us each, which the unchanged
`CudaAtomicDeviceLoadLoop` row (1.11us per mapped load) corroborates.
`checkExpired()` now gates that shared read behind the free device clock, so a
steady-state poll costs a register compare and only reaches memory once per
poll interval. An abort is still observed within one interval.

Everything else moved by a few percent or less. A confirming second run put the
two device polling rows at 4.96ms, so the 46x is not a one-off; the sub-nanosecond
host rows disagree between runs by more than they differ from each other and
remain noise-limited, as noted below.

GB200 numbers below are from the 2026-08-12 run and were not re-collected.

Both runs were taken after tightening the benchmark methodology on 2026-08-12:

- `LegacyAbortModel::Test()` now matches the old no-timeout fast path shape:
  enabled branch, abort acquire load, and `hasTimeout` acquire load.
- `MappedPinnedAbortFlagLoad` isolates a single mapped pinned abort-flag load.
- `AbortDeviceDefaultTimeoutLoadLoop` is documented as a one-shot timeout read,
  and `AbortDeviceIsAborted*LoadLoop` rows measure the device polling API.
- The many-block device row uses 128 blocks x 32 threads x 1024 loads and is
  reported as aggregate throughput, not serialized load latency.
- Signal rows use persistent kernels, include requester/responder backoff
  (`std::this_thread::yield()` on host and `__nanosleep(64)` on device), and
  are upper bounds for this benchmark protocol. Device waits are bounded by a
  1s device-clock budget; host waits use a longer 5s steady-clock guard.

GB300 was not refreshed in this run: the CUDA 13.0 / `b300a_native` build
passed, but remote access failed before authentication because agent SSH
identities were unavailable for the GB300 hosts tried: `rtptest1000.mwg2`,
`rtptest1004.mwg2`, `rtptest1007.mwg2`, `rtptest1010.mwg2`, and
`rtptest1012.mwg2`. Old GB300 numbers are not included below because the
benchmark set and device iteration counts changed.

## Commands

H100:

```bash
CUDA_VISIBLE_DEVICES=0 buck run --flagfile fbcode//mode/opt \
  fbcode//comms/common/fault_tolerance/benchmarks:abort_bench -- \
  --bm_slice_usec=100000
```

GB200:

```bash
buck build --show-full-output @fbcode//mode/opt \
  -c fbcode.arch=aarch64 \
  -c fbcode.enable_gpu_sections=true \
  -c fbcode.nvcc_arch=b200 \
  -c fbcode.platform010-aarch64_clang=17 \
  -m ovr_config//third-party/cuda/constraints:12.8 \
  fbcode//comms/common/fault_tolerance/benchmarks:abort_bench

suscp --reason 'copy abort benchmark for GB200 perf collection after reviewer fixes' \
  <abort_bench> rtptest546.kcm2:/tmp/has_abort_bench_gb200/abort_bench
sush2 --stdin-null rtptest546.kcm2 \
  '/tmp/has_abort_bench_gb200/abort_bench --bm_slice_usec=100000'
```

GB300 build-only check, plus attempted remote hosts:

```bash
buck build --show-full-output @fbcode//mode/opt \
  -c fbcode.arch=aarch64 \
  -c fbcode.enable_gpu_sections=true \
  -c fbcode.nvcc_arch=b300a_native \
  -c fbcode.platform010-aarch64_clang=17 \
  -m ovr_config//third-party/cuda/constraints:13.0 \
  fbcode//comms/common/fault_tolerance/benchmarks:abort_bench

# Remote access attempts failed before authentication on:
# rtptest1000.mwg2, rtptest1004.mwg2, rtptest1007.mwg2,
# rtptest1010.mwg2, and rtptest1012.mwg2.
```

## Hosts

| Platform | Host | GPU | Refresh status |
|---|---|---|---|
| H100 | `devgpu012.mwg1.facebook.com` | `NVIDIA H100` | Refreshed 2026-08-19; run passed |
| GB200 | `rtptest546.kcm2.facebook.com` | `NVIDIA GB200` | Last run 2026-08-12; not re-collected |
| GB300 | `rtptest1000.mwg2`, `rtptest1004.mwg2`, `rtptest1007.mwg2`, `rtptest1010.mwg2`, `rtptest1012.mwg2` | `NVIDIA GB300` | Build passed; remote copy/run blocked by agent SSH identity |

## Host API Latency

The host `Abort::isAborted()` row is not a pure mapped-memory comparison. The
legacy no-timeout fast path performs an enabled branch, one abort acquire load,
and one `hasTimeout` acquire load. The current no-timeout path performs three
mapped abort-reason loads plus one `hasTimeout` load. The default-timeout
set/get row is a cold-path shared-state update/read, not a spin-loop cost.

### H100

| Scenario | Std / legacy baseline | Mapped pinned / after | Delta | What this row measures |
|---|---:|---:|---:|---|
| Host atomic load | `StdAtomicHostLoad`: 1.22ns | `MappedPinnedHostLoad`: 1.17ns | Not reported | Sub-nanosecond load-only rows are noise-limited; use absolute values only. |
| Mapped abort flag load | N/A | `MappedPinnedAbortFlagLoad`: 1.09ns | N/A | Single mapped pinned abort-flag acquire load. |
| `Abort::isAborted()` no timeout | `LegacyAbortTestNoTimeout`: 1.07ns | `AbortTestNoTimeout`: 1.96ns | 1.83x | Host abort fast path with no active deadline. |
| `Abort::isAborted()` with future timeout | N/A | `AbortTestWithFutureTimeout`: 21.21ns | N/A | Host abort polling when an active deadline must also be checked. |
| Default timeout set/get | `LegacyDefaultTimeoutSetGet`: 1.55ns | `AbortDefaultTimeoutSetGet`: 2.52ns | 1.63x | Cold-path shared default-timeout update and optional read. |
| `Abort::getTimeRemaining()` | N/A | `AbortTimeRemaining`: 20.81ns | N/A | Host remaining-time computation for an active deadline. |
| `Abort::startTimeout()` + `cancelTimeout()` | N/A | `AbortSetTimeoutCancel`: 20.96ns | N/A | Host active-deadline lifecycle update cost. |
| Host store+load mapped atomic | N/A | `MappedPinnedHostStoreLoad`: 1.05ns | N/A | Host store followed by host load on mapped pinned memory. |

### GB200

| Scenario | Std / legacy baseline | Mapped pinned / after | Delta | What this row measures |
|---|---:|---:|---:|---|
| Host atomic load | `StdAtomicHostLoad`: 1.24ns | `MappedPinnedHostLoad`: 1.22ns | Not reported | Sub-nanosecond load-only rows are noise-limited; use absolute values only. |
| Mapped abort flag load | N/A | `MappedPinnedAbortFlagLoad`: 1.22ns | N/A | Single mapped pinned abort-flag acquire load. |
| `Abort::isAborted()` no timeout | `LegacyAbortTestNoTimeout`: 1.23ns | `AbortTestNoTimeout`: 2.76ns | 2.24x | Host abort fast path with no active deadline. |
| `Abort::isAborted()` with future timeout | N/A | `AbortTestWithFutureTimeout`: 31.00ns | N/A | Host abort polling when an active deadline must also be checked. |
| Default timeout set/get | `LegacyDefaultTimeoutSetGet`: 2.75ns | `AbortDefaultTimeoutSetGet`: 4.28ns | 1.56x | Cold-path shared default-timeout update and optional read. |
| `Abort::getTimeRemaining()` | N/A | `AbortTimeRemaining`: 31.32ns | N/A | Host remaining-time computation for an active deadline. |
| `Abort::startTimeout()` + `cancelTimeout()` | N/A | `AbortSetTimeoutCancel`: 32.89ns | N/A | Host active-deadline lifecycle update cost. |
| Host store+load mapped atomic | N/A | `MappedPinnedHostStoreLoad`: 3.43ns | N/A | Host store followed by host load on mapped pinned memory. |

## Serialized Device Latency

These rows run one CUDA thread and amortize launch overhead over 100K loop
iterations. `AbortDeviceDefaultTimeoutLoadLoop` measures only
`AbortDevice::getTimeoutMs()`, which is the one-shot setup read used by
`AbortDevice::startTimeout()`. The `AbortDeviceIsAborted*LoadLoop` rows measure
the polling API a wait loop would call.

| Scenario | H100 | GB200 | What this row measures |
|---|---:|---:|---|
| Device atomic load loop | `CudaAtomicDeviceLoadLoop`: 110.58ms / 100K loads | `CudaAtomicDeviceLoadLoop`: 100.10ms / 100K loads | Single CUDA thread polling mapped pinned memory with `cuda::atomic_ref`. |
| Device default-timeout loads | `AbortDeviceDefaultTimeoutLoadLoop`: 110.66ms / 100K loads | `AbortDeviceDefaultTimeoutLoadLoop`: 100.25ms / 100K loads | One-shot shared default-timeout read. |
| `AbortDevice::isAborted()` no deadline | `AbortDeviceIsAbortedLoadLoop`: 4.87ms / 100K polls | `AbortDeviceIsAbortedLoadLoop`: 205.83ms / 100K polls | Device polling API with no active local deadline. H100 is post-gate; the GB200 figure predates it. |
| `AbortDevice::isAborted()` armed deadline | `AbortDeviceIsAbortedWithDeadlineLoadLoop`: 4.88ms / 100K polls | `AbortDeviceIsAbortedWithDeadlineLoadLoop`: 205.83ms / 100K polls | Device polling API after `startTimeout()` arms a future deadline. Arming costs nothing measurable against the no-deadline row: the deadline is a register compare on the same gated path. |
| Device atomic store loop | `CudaAtomicDeviceStoreLoop`: 156.61ms / 100K stores | `CudaAtomicDeviceStoreLoop`: 109.90ms / 100K stores | Single CUDA thread storing to mapped pinned memory with `cuda::atomic_ref`. |

## Aggregate Device Throughput

This row measures many concurrent threads loading the same mapped address. It is
an aggregate throughput measurement with warp-uniform accesses, not the
serialized latency a single waiting warp would experience.

| Scenario | H100 | GB200 | What this row measures |
|---|---:|---:|---|
| Many-block device atomic load loop | `CudaAtomicManyBlockDeviceLoadLoop`: 128.74ms / 4.19M loads | `CudaAtomicManyBlockDeviceLoadLoop`: 1.86ms / 4.19M loads | 128 blocks x 32 threads each polling mapped pinned memory 1024 times. |

The ~69x H100/GB200 gap on identical work is the most decision-relevant number
here, so it should not be read as a measurement artifact. The likely cause is the
interconnect: on GB200 the mapped host allocation is reached over C2C and is
GPU-L2-cacheable, so 4096 warp-uniform threads hammering one address mostly hit
cache, while on H100 the same pattern is an uncached PCIe round trip. Per
`goelayush` on this diff, that caching applies to ordinary host memory on a C2C
platform and needs no special allocation API.

Treat this as attribution, not decomposition: nothing here isolates the
interconnect from the other platform differences, and no transaction counts were
profiled. The practical consequence for sizing a device-side abort poll is that
the ungated many-thread cost does not transfer from one platform to the other,
which is a further argument for the poll-interval gate rather than for relying on
the aggregate figure.

## Signal Protocol Latency

Each benchmark iteration completes one full exchange, and folly normalizes its
primary metric per iteration, so the reported figure is a **full** round trip or
ping-pong. That is what the row names say and what ServiceLab publishes; the
one-way half is derived here in parentheses and is only meaningful if the two
directions are symmetric, which this protocol does not establish. The
`ops` counter records two logical hops per iteration but does not change folly's
normalization.

These are upper bounds for this bounded benchmark protocol because unsuccessful
polls include the backoff noted above.

| Scenario | H100 | GB200 | What this row measures |
|---|---:|---:|---|
| Host/device signal RTT | `AbortSignalHostDeviceRoundTrip`: 4.24us (2.12us one way) | `AbortSignalHostDeviceRoundTrip`: 3.08us (1.54us one way) | CPU requester to persistent GPU responder and back over mapped pinned atomics. |
| Device/device signal ping-pong | `AbortSignalDeviceDevicePingPong`: 6.34us (3.17us one way) | `AbortSignalDeviceDevicePingPong`: 4.61us (2.31us one way) | Two CUDA blocks exchanging request/response over mapped pinned atomics. |
