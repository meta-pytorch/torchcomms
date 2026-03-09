# Unified Device Window: Per-Peer Signal + Barrier Design

## Goal

Extend the window system to support **both NVL and IBGDA peers** writing to the
same logical signal space, while keeping the two transport domains physically
separate to avoid cross-transport atomicity hazards.

Provide **per-peer signals** (point-to-point, `wait_signal_from()` support) and
**flat barriers** (N-way sync with per-peer-type accumulation).

---

## Background & Design Rationale

### Cross-Transport Atomicity Hazard

NVLink atomics (GPU `atomicAdd_system` via SM → L2 → NVSwitch) and IBGDA RDMA
atomics (NIC-initiated `OPCODE_ATOMIC_FA` via NIC → PCIe → GPU memory controller)
target memory through **different coherence domains**. There is no architectural
guarantee these two hardware paths are mutually atomic on the same address.

### NCCLX GIN's Solution (Prior Art)

NCCLX GIN avoids this entirely via **hierarchical separation**:
- **LSA barrier** (NVLink only): GPU multicast or unicast stores
- **GIN barrier** (RDMA only): NIC RDMA atomic fetch-add
- **Composite barrier**: LSA sync → GIN sync (sequential, never mixed)

### Our Design: Separate Buffers, Unified API

Follow GIN's separation but expose unified APIs that dispatch internally:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  DeviceWindow (unified device-side API)                         │
  │  ├── MultiPeerDeviceHandle (transports + myRank + nRanks)       │
  │  ├── signal_peer() / signal_all()                               │
  │  ├── wait_signal() / wait_signal_from()                         │
  │  ├── wait_counter() / read_counter() / reset_counter()          │
  │  ├── barrier()                                                  │
  │  └── Data APIs: put() / send() / recv() / put_signal()          │
  ├─────────────────────────────────────────────────────────────────┤
  │  NVL backing buffers          IBGDA backing buffers             │
  │  (GpuMemHandler, IPC)         (cudaMalloc + ibv_mr registered)  │
  │  Written by NVL peers ONLY    Written by IBGDA peers ONLY       │
  │  via NVLink st_release        via NIC RDMA atomic fetch-add     │
  └─────────────────────────────────────────────────────────────────┘
```

No single `uint64_t` is ever written by both NVL and RDMA — the hazard is
eliminated by construction.

---

## Architecture

### Ownership Model

```
User creates:
  1. MultiPeerTransport transport(myRank, nRanks, deviceId, bootstrap, config);
     transport.exchange();

  2. HostWindow window(transport, windowConfig, userBuffer, bufferSize);
     window.exchange();

  3. DeviceWindow dw = window.getDeviceWindow();  // single call
     myKernel<<<...>>>(dw, ...);
```

- **MultiPeerTransport**: Pure transport service. Knows nothing about windows.
  Provides topology queries, IBGDA buffer registration/exchange, NVL bootstrap.
- **HostWindow**: Standalone RAII object. Takes `MultiPeerTransport&` as a service
  reference (NOT owned by transport). Manages all signal/counter/barrier buffers.
  Optionally registers/exchanges a user-provided data buffer.
- **DeviceWindow**: Lightweight device-side handle. All signal, barrier, counter,
  and data transfer state is held directly (no sub-objects). Passed by value to
  kernels.

### Key: No Circular Ownership

```
HostWindow ──uses──▸ MultiPeerTransport   (one-way reference)
                     │
                     ├── registerIbgdaBuffer()
                     ├── exchangeIbgdaBuffer()
                     ├── deregisterIbgdaBuffer()
                     ├── nvl_bootstrap()
                     ├── get_transport_type(rank)
                     ├── get_device_handle()
                     └── nvl_peer_ranks() / ibgda_peer_ranks()
```

---

## Signal Design: Per-Peer Inbox Model

### Per-Peer Signal

- **One slot per (peer, signal_id)** — each peer has a dedicated row
- **Use case**: Point-to-point "has rank X specifically signaled?"
- **Wait (all)**: Sums across all peer rows (NVL + IBGDA) → warp-parallel reduction
- **Wait (from)**: Reads one specific peer's slot → **O(1)**

### Memory Layout (Per Rank)

```
Per-peer NVL inbox (SignalState, 128-byte aligned):
  [nvlPeer0_sig0] [nvlPeer0_sig1] ... [nvlPeer0_sig_{peerSignalCount-1}]
  [nvlPeer1_sig0] [nvlPeer1_sig1] ...
  ...

Per-peer IBGDA inbox (packed uint64_t):
  [ibgdaPeer0_sig0] [ibgdaPeer0_sig1] ...
  [ibgdaPeer1_sig0] [ibgdaPeer1_sig1] ...
  ...
```

### Signal vs Counter Semantics

| | Signal | Counter |
|---|---|---|
| **What** | Remote data-arrival notification | Local NIC completion notification |
| **Direction** | Sender's NIC → Remote peer's memory | Sender's NIC → Sender's own memory |
| **Who writes** | NVL: GPU st_release; IBGDA: NIC RDMA atomic | IBGDA companion QP loopback |
| **Meaning** | "Data arrived at peer's buffer" | "NIC finished processing my WQEs" |
| **Who waits** | Receiver polls local signal inbox | Sender polls local counter buffer |
| **Transport** | Both NVL and IBGDA | **IBGDA only** |

**Why NVL doesn't need counters**: NVLink stores are synchronous — when
`st_release_sys_global` retires, the data is globally visible.

**Why IBGDA needs counters**: RDMA operations are asynchronous — the companion QP
writes to sender's own local counter via loopback RDMA atomic fetch-add when the
main QP's WQEs complete. Cheaper than CQ polling.

---

## Configuration

```cpp
// window/HostWindow.h

struct WindowConfig {
    std::size_t peerSignalCount{0};    // per-peer signals (wait_signal_from)
    std::size_t peerCounterCount{0};   // IBGDA-only: per-peer local NIC completion
    std::size_t barrierCount{0};       // flat barriers (per-peer-type accumulation)
};
```

All types default to 0 — set to the desired count to enable allocation.
Set to 0 to skip allocation for unused features.

---

## Detailed Design

### 1. Host-Side: `HostWindow`

#### Constructor

```cpp
HostWindow(
    MultiPeerTransport& transport,
    const WindowConfig& config,
    void* userBuffer = nullptr,     // optional user-allocated data buffer
    std::size_t userBufferSize = 0  // size of user buffer
);
```

If a user buffer is provided, HostWindow registers and exchanges it on **both**
NVL and IBGDA sides:
- **NVL**: IPC exchange via `GpuMemHandler` so NVL peers can access it via NVLink
- **IBGDA**: RDMA registration + exchange so IBGDA peers can do RDMA writes to it

After exchange(), the caller retrieves per-transport buffer info for kernels:

```cpp
HostWindow window(transport, config, myGpuBuffer, bufferSize);
window.exchange();

// For IBGDA peers: RDMA-registered local buffer + remote peer buffer info
auto localBuf = window.getUserLocalBuffer();      // IbgdaLocalBuffer (lkey)
auto remoteBufs = window.getUserRemoteBuffers();   // DeviceSpan<IbgdaRemoteBuffer>

// For NVL peers: IPC-mapped peer pointers (direct NVLink access)
auto peerPtrs = window.getUserNvlPeerPtrs();       // DeviceSpan<void*>
```

Extracts from transport: `my_rank()`, `n_ranks()`, `nvl_peer_ranks()`,
`ibgda_peer_ranks()`, `nvl_local_rank()`, `nvl_n_ranks()`, `nvl_bootstrap()`,
`get_transport_type(rank)`.

#### Allocation Strategy

**NVL buffers** — allocated via `GpuMemHandler` (IPC-shared for NVLink):
- Uses `transport.nvl_bootstrap()` scoped to NVL peers
- Exchanged via `GpuMemHandler::exchangeMemPtrs()` within NVL group

**IBGDA buffers** — allocated via `cudaMalloc` then RDMA-registered:
- Registered with `transport.registerIbgdaBuffer(ptr, size)` → `IbgdaLocalBuffer`
- Exchanged with `transport.exchangeIbgdaBuffer(localBuf)` → `vector<IbgdaRemoteBuffer>`

#### IBGDA Buffer Lifecycle

For each IBGDA buffer, there are 3 stages:
1. `void* buffer` — raw GPU memory from `cudaMalloc` (**temporary local variable**)
2. `IbgdaLocalBuffer localBuf` — after `registerIbgdaBuffer()`, adds lkey.
   Stores `ptr` internally, so `localBuf.ptr` replaces the raw `void*`.
3. `vector<IbgdaRemoteBuffer> remoteBufs` — after `exchangeIbgdaBuffer()`, peer
   rkeys. **Temporary host-side** — only the device copy is kept.

**Key**: raw `void*` and host-side `vector<IbgdaRemoteBuffer>` are local
variables in constructor/exchange(), NOT class fields. Only `IbgdaLocalBuffer`
(for deregister + lkey) and `DeviceBuffer` (for device access) are members.

#### Private Members

```cpp
class HostWindow {
    MultiPeerTransport& transport_;
    int myRank_, nRanks_;
    WindowConfig config_;
    std::vector<int> nvlPeerRanks_, ibgdaPeerRanks_;
    int nvlLocalRank_, nvlNRanks_;

    // --- Pre-computed peer index maps (device-accessible, O(1) lookup) ---
    // rankToNvlPeerIndex_[rank] = NVL peer index, or -1 if not NVL peer
    // rankToIbgdaPeerIndex_[rank] = IBGDA peer index, or -1 if not IBGDA peer
    std::unique_ptr<DeviceBuffer> peerIndexMapsDevice_;

    // --- Barrier buffers (flat, per-peer-type) ---
    std::unique_ptr<GpuMemHandler> nvlBarrierHandler_;
    std::unique_ptr<DeviceBuffer> nvlBarrierPeerPtrsDevice_;
    IbgdaLocalBuffer ibgdaBarrierLocalBuf_;        // .ptr for cudaFree
    std::unique_ptr<DeviceBuffer> ibgdaBarrierRemoteBufsDevice_;

    // --- Per-peer signals ---
    std::unique_ptr<GpuMemHandler> nvlPeerSignalHandler_;
    std::unique_ptr<DeviceBuffer> nvlPeerSignalSpansDevice_;
    IbgdaLocalBuffer ibgdaPeerSignalLocalBuf_;
    std::unique_ptr<DeviceBuffer> ibgdaPeerSignalRemoteBufsDevice_;

    // --- Per-peer counters (IBGDA-only, local — no exchange) ---
    IbgdaLocalBuffer ibgdaPeerCounterLocalBuf_;

    // --- User data buffer (optional) ---
    IbgdaLocalBuffer userLocalBuf_;
    std::unique_ptr<DeviceBuffer> userRemoteBufsDevice_;
    std::vector<void*> userNvlMappedPtrs_;
    std::unique_ptr<DeviceBuffer> userNvlPeerPtrsDevice_;
};
```

#### Exchange Flow

```
exchange():
  1. NVL barrier:       nvlBarrierHandler_->exchangeMemPtrs()
  2. NVL peer signal:   nvlPeerSignalHandler_->exchangeMemPtrs()
  3. IBGDA barrier:     transport_.registerIbgdaBuffer() + exchangeIbgdaBuffer()
  4. IBGDA peer signal: transport_.registerIbgdaBuffer() + exchangeIbgdaBuffer()
  5. IBGDA counter:     transport_.registerIbgdaBuffer() (local only, no exchange)
  6. User buffer:       register + exchange (IBGDA) and/or exchangeNvlBuffer (NVL)
  7. Pre-compute peer index maps: build rankToNvlPeerIndex_ / rankToIbgdaPeerIndex_
  8. exchanged_ = true
```

### 2–4. Device-Side: `DeviceWindow` (flattened)

All signal, barrier, and counter state is held directly in `DeviceWindow`.
No separate `DeviceWindowSignal` or `DeviceWindowBarrier` classes — they are
unnecessary indirection. `DeviceWindow` owns a `MultiPeerDeviceHandle` which
provides `get_type(rank)` and `get_ibgda(rank)` for transport dispatch.

```cpp
class DeviceWindow {
    MultiPeerDeviceHandle handle_;

    // --- Per-peer signal buffers ---
    int peerSignalCount_;
    int nNvlPeers_, nIbgdaPeers_;
    DeviceSpan<SignalState> nvlPeerSignalInbox_;
    DeviceSpan<DeviceSpan<SignalState>> nvlPeerSignalSpans_;
    uint64_t* ibgdaPeerSignalInbox_;
    DeviceSpan<IbgdaRemoteBuffer> ibgdaPeerSignalRemoteBufs_;

    // --- Pre-computed peer index maps (O(1) rank → peer index lookup) ---
    DeviceSpan<int> rankToNvlPeerIndex_;    // [nRanks], -1 if not NVL peer
    DeviceSpan<int> rankToIbgdaPeerIndex_;  // [nRanks], -1 if not IBGDA peer

    // --- Per-peer counter buffers (IBGDA-only, local) ---
    int peerCounterCount_;
    uint64_t* ibgdaPeerCounterBuffer_;

    // --- Barrier buffers (flat, per-peer-type) ---
    int barrierCount_;
    SignalState* nvlBarrierInbox_;
    DeviceSpan<SignalState*> nvlBarrierPeerPtrs_;
    uint64_t* ibgdaBarrierInbox_;
    DeviceSpan<IbgdaRemoteBuffer> ibgdaBarrierRemoteBufs_;
    uint32_t barrierExpected_{0};
};
```

**Dispatch uses `handle_.get_type(rank)`. Peer index lookup is O(1):**

```cuda
__device__ void signal_peer(ThreadGroup& group, int target_rank,
                            int signal_id, SignalOp op, uint64_t value) {
    if (handle_.get_type(target_rank) == TransportType::P2P_NVL) {
        int nvlIdx = rankToNvlPeerIndex_[target_rank];  // O(1)
        nvlPeerSignalSpans_[nvlIdx][signal_id].signal(group, op, value);
    } else {
        int ibgdaIdx = rankToIbgdaPeerIndex_[target_rank];  // O(1)
        handle_.get_ibgda(target_rank).signal_remote(
            ibgdaPeerSignalRemoteBufs_[ibgdaIdx], signal_id, value);
    }
}
```

#### Full DeviceWindow API

```cpp
class DeviceWindow {
public:
    // Metadata
    int rank(), n_ranks(), num_peers(), num_nvl_peers(), num_ibgda_peers();
    int peer_index_to_rank(int index);
    int rank_to_peer_index(int rank);

    // Transport access
    TransportType get_type(int rank);
    P2pNvlTransportDevice& get_nvl(int rank);
    P2pIbgdaTransportDevice& get_ibgda(int rank);

    // Signal operations
    void signal_peer(ThreadGroup& group, int target_rank, int signal_id, ...);
    void signal_all(ThreadGroup& group, int signal_id, ...);
    void wait_signal(ThreadGroup& group, int signal_id, CmpOp, uint64_t);
    void wait_signal_from(ThreadGroup& group, int source_rank, int signal_id, ...);
    uint64_t read_signal(int signal_id);
    uint64_t read_signal_from(int source_rank, int signal_id);

    // Counter operations (IBGDA-only)
    void wait_counter(int peer_rank, int counter_id, CmpOp, uint64_t);
    uint64_t read_counter(int peer_rank, int counter_id);
    void reset_counter(int peer_rank, int counter_id);

    // Barrier
    void barrier(ThreadGroup& group, int barrier_id);

    // Data transfer
    void send(int rank, ...);   // NVL only, traps on IBGDA
    void recv(int rank, ...);   // NVL only, traps on IBGDA
    void put(int rank, ...);    // NVL only for now (void* buffers)
    void put_signal(int rank, ...);  // put + signal_peer

    // Direct handle access
    const MultiPeerDeviceHandle& get_handle();
};
```

### 5. `P2pIbgdaTransportDevice` — Signal/Counter Methods

The window owns signal/counter buffers. The transport owns QPs. To post RDMA
atomics to window-owned buffers, `P2pIbgdaTransportDevice` provides:

```cpp
IbgdaWork signal_remote(const IbgdaRemoteBuffer& remoteBuf, int signalId, uint64_t value);
IbgdaWork signal_remote_with_fence(const IbgdaRemoteBuffer& remoteBuf, int signalId, uint64_t value);

void put_signal_counter_remote(
    const IbgdaLocalBuffer& localData, const IbgdaRemoteBuffer& remoteData, size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignal, int signalId, uint64_t signalVal,
    const IbgdaLocalBuffer& localCounter, int counterId, uint64_t counterVal);

void signal_counter_remote(...);
void wait_local(const IbgdaWork& work, Timeout timeout = Timeout());
```

---

## Naming Conventions

- `MultiPeerTransport` public API: **snake_case** (`nvl_bootstrap`, `is_nvl_peer`)
- `HostWindow` public API: **camelCase** (`getDeviceWindow`, `getDeviceWindowMemory`)
- `DeviceWindow` device API: **snake_case** (`signal_peer`, `barrier`, `put`)
- Internal helpers: **snake_case** (`rank_to_peer_index`)

---

## Files

| File | Description |
|---|---|
| `window/HostWindow.h/.cc` | Host-side RAII manager (replaces WindowMemory) |
| `window/DeviceWindow.cuh` | Unified device-side handle — all signal/barrier/counter/data ops |
| `window/DeviceWindowSignal.cuh` | **(deprecated, kept for backward compat)** |
| `window/DeviceWindowBarrier.cuh` | **(deprecated, kept for backward compat)** |
| `window/DeviceWindowMemory.cuh` | **(deprecated, kept for backward compat)** |
| `MultiPeerTransport.h/.cc` | Pure transport service (no window ownership) |
| `MultiPeerDeviceHandle.cuh` | Lightweight transport handle for kernels |
| `P2pIbgdaTransportDevice.cuh` | IBGDA device transport with signal_remote, wait_local |
| `MultipeerIbgdaTransport.h/.cc` | Host IBGDA transport with companion QP + sink buffer |
| ~~`MultiPeerDeviceTransport.cuh`~~ | **Removed** — superseded by DeviceWindow |

---

## Diff Stack

### Diff 1 (bottom): Signal ownership — buffers move from transport to caller

Pure refactoring — all signal APIs now take explicit buffer args. No new
capabilities added.

**Files:**
- `P2pIbgdaTransportDevice.cuh` — Remove signal buffer members. `put_signal()`,
  `wait_signal()`, `read_signal()`, `reset_signal()` take explicit
  `IbgdaLocalBuffer`/`IbgdaRemoteBuffer` args. `put_signal()` uses high-level
  `doca_gpu_dev_verbs_put_signal` API. `reset_signal()` uses
  `doca_gpu_dev_verbs_p<uint64_t>` RDMA inline write. Add
  `kDefaultDeviceTimeoutCycles`.
- `MultipeerIbgdaTransport.h` — Remove signal buffer ownership from host transport
- `MultipeerIbgdaDeviceTransport.cuh` — Adapt kernel wrappers to pass explicit
  signal buffers
- `MultipeerIbgdaTransportCuda.cu/.cuh` — Adapt CUDA launch wrappers
- `tests/MultipeerIbgdaTransportTest.cc/.cu/.cuh/.h` — **Modify** existing test
  cases to pass explicit signal buffer args
- `tests/MultiPeerTransportIntegrationTest.cc` — Adapt to new signal API
- `tests/MultiPeerTransportTest.cc` — Adapt to new signal API
- `benchmarks/IbgdaBenchmark.cc/.cu/.cuh/.h` — **Modify** existing benchmarks to
  pass explicit signal buffer args

### Diff 2: Companion QP + counter infrastructure

New capabilities — companion QP for NIC completion tracking, new compound RDMA
operations.

**Files:**
- `MultipeerIbgdaTransport.cc` — Companion QP creation (`core_direct=true`),
  sink buffer allocation (`ibv_reg_mr_iova2` with `iova=0`)
- `MultipeerIbgdaTransport.h` — Add companion QP + sink to constructor/members
- `P2pIbgdaTransportDevice.cuh` — Add `companionQp_`, `sinkLkey_` members.
  **New methods:** `signal_remote()`, `signal_remote_with_fence()`,
  `put_signal_counter_remote()`, `signal_counter_remote()`, `wait_local()` with
  timeout. **Upgrade** `put_signal()` from high-level DOCA API to
  `put()` + `signal_remote_with_fence()` for NIC fence support.
- `tests/MultipeerIbgdaTransportTest.cc/.cu/.cuh/.h` — **Add** new test cases
  for `signal_remote`, counter operations, `wait_local` timeout
- `benchmarks/IbgdaBenchmark.cc/.cu/.cuh/.h` — **Add** new counter-based latency
  benchmark modes
- `benchmarks/BUCK` — Add counter benchmark targets

### Diff 3: MultiPeerTransport — buffer exchange + topology APIs

Make MultiPeerTransport a standalone service with all APIs that HostWindow needs.

**Files:**
- `MultiPeerTransport.h/.cc` — `registerIbgdaBuffer()`, `exchangeIbgdaBuffer()`,
  `deregisterIbgdaBuffer()`, `nvl_bootstrap()`, `exchangeNvlBuffer()`,
  `unmapNvlBuffers()`, `get_device_handle()`
- `MultiPeerNvlTransport.h/.cc` — Remove `getMultiPeerDeviceTransport()`
- `BUCK` — Dependency updates
- `tests/MultiPeerTransportTest.cc` — Test new APIs
- `tests/MultiPeerTransportMultiNodeTest.cc` — Multi-node tests

### Diff 4: Unified DeviceWindow + HostWindow

New window abstraction layer with device-side unified API.

**Files:**
- `window/DeviceWindow.cuh` — **New** unified device-side handle
- `window/HostWindow.h/.cc` — **New** host-side RAII manager
- `window/BUCK` — New targets
- `window/DeviceWindowSignal.cuh`, `DeviceWindowBarrier.cuh`,
  `DeviceWindowMemory.cuh`, `WindowMemory.h/.cc` — **Delete** (deprecated)
- `MultiPeerDeviceTransport.cuh` — **Delete** (superseded by DeviceWindow)
- `tests/DeviceWindowTest.cc/.cu/.cuh` — **New** window unit tests
- `tests/HostWindowTest.cc` — **New** host window tests
- `tests/MultiPeerDeviceTransportTest.cc/.cu/.cuh`,
  `tests/WindowMemoryTest.cc` — **Delete**
- `tests/BUCK` — Update targets

### Diff 5 (top): Migrate callers to DeviceWindow

All integration tests and benchmarks now use DeviceWindow.

**Files:**
- `tests/MultipeerIbgdaTransportTest.cc/.cu/.cuh/.h` — Kernels take
  `DeviceWindow`, callers pass `window.getDeviceWindow()`
- `tests/MultiPeerNvlTransportIntegrationTest.cc/.cu/.cuh` — Same migration
- `benchmarks/IbgdaBenchmark.cc/.cu/.cuh/.h` — Benchmark kernels take
  `DeviceWindow`
- `benchmarks/MultiPeerBenchmark.cc/.cu/.cuh` — Same migration
- `benchmarks/BUCK` — Target updates

### Diff stack design notes

The test/benchmark files (`MultipeerIbgdaTransportTest.*`,
`IbgdaBenchmark.*`) are touched at most twice across the stack:

| | Diff 1 (ownership) | Diff 2 (companion QP) | Diff 5 (migration) |
|---|---|---|---|
| **Test changes** | Modify existing tests (pass explicit buffer args) | Add new test cases (counters, signal_remote, timeout) | Migrate kernels to DeviceWindow |
| **Benchmark changes** | Modify existing benchmarks (pass explicit buffer args) | Add new benchmark modes (counter latency) | Migrate kernels to DeviceWindow |

Diffs 3 and 4 have zero file overlap with any other diff.

---

## Resolved Design Decisions

1. **No flat signals**: Barriers use their own flat per-peer-type accumulation
   model directly inside DeviceWindow. Per-peer signals handle all point-to-point
   needs. The flat_signal_* API was dropped.

2. **HostWindow is standalone**: NOT owned by MultiPeerTransport. This avoids
   circular ownership. Users create both objects and manage their lifetimes.

3. **signal_remote() on P2pIbgdaTransportDevice**: Window owns signal/counter
   buffers, transport owns QPs. Clean ownership separation.

4. **IBGDA buffer alignment**: Packed uint64_t layout (no 128-byte alignment),
   matching GIN's approach. RDMA atomics operate on naturally-aligned uint64_t.

5. **Barrier is per-peer-type, not per-peer**: Each BarrierState accumulates
   arrivals from ALL peers of one transport type. Total = 2 × barrierCount.

6. **Dispatch info lives in MultiPeerDeviceHandle**: DeviceWindow does NOT
   carry separate transport type arrays. It uses `handle_.get_type(rank)` and
   `handle_.get_ibgda(rank)` for all NVL vs IBGDA dispatch.

7. **No separate DeviceWindowSignal/DeviceWindowBarrier classes**: All signal,
   barrier, and counter state is flattened into `DeviceWindow` directly.

8. **Pre-computed peer index maps**: O(1) `rankToNvlPeerIndex_[rank]` and
   `rankToIbgdaPeerIndex_[rank]` lookups instead of computing indices on the
   critical path.

9. **User buffer in HostWindow constructor**: HostWindow optionally accepts a
   user-allocated data buffer, handles registration + exchange under the hood.

10. **Raw void* removed from HostWindow fields**: `IbgdaLocalBuffer.ptr` replaces
    raw pointers. Host-side `vector<IbgdaRemoteBuffer>` are local variables in
    exchange(), not class members.

11. **No MultiPeerDeviceTransport**: DeviceWindow is the single unified device-side
    API. It owns a `MultiPeerDeviceHandle` for transport dispatch and provides all
    signal/barrier/counter/data transfer APIs. No separate transport wrapper needed.
