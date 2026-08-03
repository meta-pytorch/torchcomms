# Abort Benchmark Results

Collected for `D113372521` after `Abort` moved to shared CUDA-mapped pinned
state and `AbortDevice` was added. H100, GB200, and GB300 were refreshed on
2026-08-03 after switching host mapped-pinned access to `std::atomic_ref`,
start-gating the device/device ping-pong benchmark, and adding a 128-block
device polling benchmark.

Signal benchmarks use persistent kernels so per-iteration CUDA launch overhead
is outside the measured loop. Host/device and device/device signal rows report
half round-trip latency first, with the raw benchmark round-trip or ping-pong
latency shown in parentheses.

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
  -c fbcode.platform010_cuda_version=12.8 \
  -c fbcode.platform010-aarch64_clang=17 \
  -m ovr_config//third-party/cuda/constraints:12.8 \
  fbcode//comms/common/fault_tolerance/benchmarks:abort_bench

suscp --reason 'copy abort benchmark for GB200 perf collection' \
  <abort_bench> rtptest546.kcm2:/tmp/has_abort_bench_gb200/abort_bench
sush2 --stdin-null rtptest546.kcm2 \
  '/tmp/has_abort_bench_gb200/abort_bench --bm_slice_usec=100000'
```

GB300:

```bash
buck build --show-full-output @fbcode//mode/opt \
  -c fbcode.arch=aarch64 \
  -c fbcode.enable_gpu_sections=true \
  -c fbcode.nvcc_arch=b300a_native \
  -c fbcode.platform010_cuda_version=13.0 \
  -c fbcode.platform010-aarch64_clang=17 \
  -m ovr_config//third-party/cuda/constraints:13.0 \
  fbcode//comms/common/fault_tolerance/benchmarks:abort_bench

suscp --reason 'copy abort benchmark for GB300 perf collection' \
  <abort_bench> rtptest1000.mwg2:/tmp/has_abort_bench_gb300/abort_bench
sush2 --stdin-null rtptest1000.mwg2 \
  '/tmp/has_abort_bench_gb300/abort_bench --bm_slice_usec=100000'
```

## Hosts

| Platform | Host | GPU |
|---|---|---|
| H100 | `devgpu012.mwg1.facebook.com` | `NVIDIA H100` |
| GB200 | `rtptest546.kcm2.facebook.com` | `NVIDIA GB200` |
| GB300 | `rtptest1000.mwg2.facebook.com` | `NVIDIA GB300` |

## H100

| Scenario | Std / legacy baseline | Mapped pinned / after | Delta | What this row measures |
|---|---:|---:|---:|---|
| Host atomic load | `StdAtomicHostLoad`: 1.11ns | `MappedPinnedHostLoad`: 936.16ps | 0.84x | Host polling of mapped pinned memory through `std::atomic_ref` versus `std::atomic`. |
| `Abort::isAborted()` no timeout | `LegacyAbortTestNoTimeout`: 1.00ns | `AbortTestNoTimeout`: 2.44ns | 2.44x | Common host abort fast path with no active deadline. |
| `Abort::isAborted()` with future timeout | N/A | `AbortTestWithFutureTimeout`: 22.33ns | N/A | Host abort polling when an active deadline must also be checked. |
| Default timeout set/get | `LegacyDefaultTimeoutSetGet`: 1.43ns | `AbortDefaultTimeoutSetGet`: 3.35ns | 2.34x | Host update/read of shared default timeout state. |
| `Abort::getTimeRemaining()` | N/A | `AbortTimeRemaining`: 21.88ns | N/A | Host remaining-time computation for an active deadline. |
| `Abort::startTimeout()` + `cancelTimeout()` | N/A | `AbortSetTimeoutCancel`: 22.44ns | N/A | Host active-deadline lifecycle update cost. |
| Device atomic load loop | N/A | `CudaAtomicDeviceLoadLoop`: 1.33s / 1.00M loads | N/A | Single CUDA thread polling mapped pinned memory with `cuda::atomic_ref`. |
| Many-block device atomic load loop | N/A | `CudaAtomicManyBlockDeviceLoadLoop`: 8.47ms / 262K loads | N/A | 128 blocks x 32 threads each polling mapped pinned memory 64 times. |
| Device atomic store loop | N/A | `CudaAtomicDeviceStoreLoop`: 1.57s / 1.00M stores | N/A | Single CUDA thread storing to mapped pinned memory with `cuda::atomic_ref`. |
| Device default-timeout loads | N/A | `AbortDeviceDefaultTimeoutLoadLoop`: 1.17s / 1.00M loads | N/A | Device read cost for shared `Abort` timeout state. |
| Host store+load mapped atomic | N/A | `MappedPinnedHostStoreLoad`: 914.19ps | N/A | Host store followed by host load on mapped pinned memory. |
| Host/device signal half RTT | N/A | `AbortSignalHostDeviceHalfRoundTrip`: 2.27us (4.53us RTT) | N/A | CPU requester to persistent GPU responder and back over mapped pinned atomics. |
| Device/device signal half RTT | N/A | `AbortSignalDeviceDeviceHalfRoundTrip`: 3.19us (6.37us ping-pong) | N/A | Two CUDA blocks exchanging request/response over mapped pinned atomics. |

## GB200

| Scenario | Std / legacy baseline | Mapped pinned / after | Delta | What this row measures |
|---|---:|---:|---:|---|
| Host atomic load | `StdAtomicHostLoad`: 1.26ns | `MappedPinnedHostLoad`: 1.22ns | 0.97x | Host polling of mapped pinned memory through `std::atomic_ref` versus `std::atomic`. |
| `Abort::isAborted()` no timeout | `LegacyAbortTestNoTimeout`: 1.23ns | `AbortTestNoTimeout`: 2.77ns | 2.25x | Common host abort fast path with no active deadline. |
| `Abort::isAborted()` with future timeout | N/A | `AbortTestWithFutureTimeout`: 31.18ns | N/A | Host abort polling when an active deadline must also be checked. |
| Default timeout set/get | `LegacyDefaultTimeoutSetGet`: 2.75ns | `AbortDefaultTimeoutSetGet`: 4.27ns | 1.55x | Host update/read of shared default timeout state. |
| `Abort::getTimeRemaining()` | N/A | `AbortTimeRemaining`: 31.02ns | N/A | Host remaining-time computation for an active deadline. |
| `Abort::startTimeout()` + `cancelTimeout()` | N/A | `AbortSetTimeoutCancel`: 32.89ns | N/A | Host active-deadline lifecycle update cost. |
| Device atomic load loop | N/A | `CudaAtomicDeviceLoadLoop`: 1.03s / 1.00M loads | N/A | Single CUDA thread polling mapped pinned memory with `cuda::atomic_ref`. |
| Many-block device atomic load loop | N/A | `CudaAtomicManyBlockDeviceLoadLoop`: 133.22us / 262K loads | N/A | 128 blocks x 32 threads each polling mapped pinned memory 64 times. |
| Device atomic store loop | N/A | `CudaAtomicDeviceStoreLoop`: 1.15s / 1.00M stores | N/A | Single CUDA thread storing to mapped pinned memory with `cuda::atomic_ref`. |
| Device default-timeout loads | N/A | `AbortDeviceDefaultTimeoutLoadLoop`: 1.03s / 1.00M loads | N/A | Device read cost for shared `Abort` timeout state. |
| Host store+load mapped atomic | N/A | `MappedPinnedHostStoreLoad`: 3.61ns | N/A | Host store followed by host load on mapped pinned memory. |
| Host/device signal half RTT | N/A | `AbortSignalHostDeviceHalfRoundTrip`: 1.54us (3.08us RTT) | N/A | CPU requester to persistent GPU responder and back over mapped pinned atomics. |
| Device/device signal half RTT | N/A | `AbortSignalDeviceDeviceHalfRoundTrip`: 2.31us (4.61us ping-pong) | N/A | Two CUDA blocks exchanging request/response over mapped pinned atomics. |

## GB300

| Scenario | Std / legacy baseline | Mapped pinned / after | Delta | What this row measures |
|---|---:|---:|---:|---|
| Host atomic load | `StdAtomicHostLoad`: 1.26ns | `MappedPinnedHostLoad`: 1.24ns | 0.98x | Host polling of mapped pinned memory through `std::atomic_ref` versus `std::atomic`. |
| `Abort::isAborted()` no timeout | `LegacyAbortTestNoTimeout`: 1.25ns | `AbortTestNoTimeout`: 2.84ns | 2.27x | Common host abort fast path with no active deadline. |
| `Abort::isAborted()` with future timeout | N/A | `AbortTestWithFutureTimeout`: 31.57ns | N/A | Host abort polling when an active deadline must also be checked. |
| Default timeout set/get | `LegacyDefaultTimeoutSetGet`: 2.80ns | `AbortDefaultTimeoutSetGet`: 4.34ns | 1.55x | Host update/read of shared default timeout state. |
| `Abort::getTimeRemaining()` | N/A | `AbortTimeRemaining`: 31.55ns | N/A | Host remaining-time computation for an active deadline. |
| `Abort::startTimeout()` + `cancelTimeout()` | N/A | `AbortSetTimeoutCancel`: 33.39ns | N/A | Host active-deadline lifecycle update cost. |
| Device atomic load loop | N/A | `CudaAtomicDeviceLoadLoop`: 876.69ms / 1.00M loads | N/A | Single CUDA thread polling mapped pinned memory with `cuda::atomic_ref`. |
| Many-block device atomic load loop | N/A | `CudaAtomicManyBlockDeviceLoadLoop`: 126.45us / 262K loads | N/A | 128 blocks x 32 threads each polling mapped pinned memory 64 times. |
| Device atomic store loop | N/A | `CudaAtomicDeviceStoreLoop`: 951.83ms / 1.00M stores | N/A | Single CUDA thread storing to mapped pinned memory with `cuda::atomic_ref`. |
| Device default-timeout loads | N/A | `AbortDeviceDefaultTimeoutLoadLoop`: 859.17ms / 1.00M loads | N/A | Device read cost for shared `Abort` timeout state. |
| Host store+load mapped atomic | N/A | `MappedPinnedHostStoreLoad`: 3.66ns | N/A | Host store followed by host load on mapped pinned memory. |
| Host/device signal half RTT | N/A | `AbortSignalHostDeviceHalfRoundTrip`: 1.35us (2.69us RTT) | N/A | CPU requester to persistent GPU responder and back over mapped pinned atomics. |
| Device/device signal half RTT | N/A | `AbortSignalDeviceDeviceHalfRoundTrip`: 2.20us (4.40us ping-pong) | N/A | Two CUDA blocks exchanging request/response over mapped pinned atomics. |
