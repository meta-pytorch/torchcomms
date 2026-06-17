# IBRC Tile Send/Recv Design

A CPU-initiated IB Reliable Connection (RC) transport with `send_tile` /
`recv_tile` APIs for pipelined point-to-point transfers, callable from CUDA and
Triton kernels. Extends the pipes transport layer alongside the existing NVLink
and IBGDA transports.

**Target:** New `P2P_IBRC` transport type in pipes, with the same user-facing
tile API as NVLink (`P2pNvlTransportDevice`) and IBGDA
(`P2pIbgdaTransportDevice`).

**Motivation:** IBGDA requires DOCA GPUNetIO and GPU-mapped QPs, which are not
available on all platforms (e.g., ConnectX-6 without DOCA, non-NVIDIA DPUs, ROCm
environments). IBRC uses standard `libibverbs` with CPU-initiated RDMA, making
it portable across all IB/RoCE hardware. The CPU proxy adds latency per
operation, but the pipelined tile architecture hides it for large transfers.

**Reference implementations:**
- ctran IB RC backend (`comms/ctran/backends/ib/`) — CPU-initiated RC QPs with
  spray/DQPLB modes, epoch locking, ibverbx RAII wrappers
- NCCL net IB transport (`comms/ncclx/v2_29/src/transport/net_ib/`) — CTS FIFO
  protocol, proxy threads, GDR/bounce-buffer paths, multi-QP striping
- IBGDA tile API (`comms/pipes/P2pIbgdaTransportDevice.cuh`, D101055842) — the
  GPU-initiated tile protocol this design mirrors

**In scope:** Per-block tile send/recv with CPU-proxy-mediated RDMA, cooperative
staging memcpy, pipelined ring buffer, transport-type dispatch in the Triton
extern layer.

**Out of scope:** Multi-stream concurrency on the same transport, adaptive
routing, traffic class configuration, QP resiliency/failover.

---

## 1. Why IBRC is Fundamentally Different

The core difference from IBGDA is **who posts RDMA operations**:

| Aspect | IBGDA (GPU-initiated) | IBRC (CPU-initiated) |
|--------|----------------------|---------------------|
| QP access | GPU-mapped via DOCA GPUNetIO | Kernel-mode QP, CPU-only |
| RDMA posting | GPU thread writes WQE + rings doorbell | CPU thread calls `ibv_post_send()` |
| CQ polling | GPU thread via `doca_gpu_dev_verbs_poll_one_cq_at` | CPU thread via `ibv_poll_cq()` |
| GPU→NIC signaling | Direct (GPU writes to NIC SQ) | Indirect (GPU→FIFO→CPU→NIC) |
| Dependencies | DOCA GPUNetIO, ConnectX-7+ | Standard libibverbs, any RC-capable HCA |
| Per-op latency | ~1-2µs (GPU doorbell) | ~5-10µs (GPU→CPU notification + `ibv_post_send`) |

Because the GPU cannot call `ibv_post_send()`, we introduce a **CPU proxy
thread** and a **GPU→CPU FIFO** — a shared-memory ring buffer where GPU threads
write work descriptors that the CPU proxy drains.

---

## 2. Architecture

```
  GPU (device code)                    CPU (proxy thread)
  ═══════════════                      ══════════════════

  ┌──────────┐                         ┌──────────────┐
  │ user src │                         │  Proxy Loop  │
  └────┬─────┘                         │              │
       │ memcpy (cooperative)          │  1. Poll FIFO│
       ▼                               │  2. ibv_post │
  ┌────────────┐    FIFO descriptor    │     _send()  │
  │sendStaging │ ──────────────────▶   │  3. Poll CQ  │
  │  (GPU mem) │    (shared memory)    │  4. Update   │
  └────────────┘                       │     counters │
       ▲                               └──────┬───────┘
       │                                      │ RDMA Write
       │ GPU polls                            ▼
       │ completion                    ┌────────────┐
  ┌─────────────┐                     │recvStaging │
  │ NIC_DONE    │◀── volatile write   │ (remote    │
  │ counter     │    from CPU proxy   │  GPU mem)  │
  └─────────────┘                     └────────────┘
```

### End-to-End Data Flow (one chunk, sender block 0 → receiver block 0)

```
  SENDER RANK A                                              RECEIVER RANK B
  ══════════════                                             ═══════════════

  GPU block 0                    CPU proxy A                 CPU proxy B                    GPU block 0
  ───────────                    ───────────                 ───────────                    ───────────
  │                              │                           │                              │
  │ (1) memcpy src→staging       │                           │                              │
  │     group.sync()             │                           │                              │
  │     __threadfence_system()   │                           │                              │
  │                              │                           │                              │
  │ (2) write FIFO entry ──────▶│                           │                              │
  │     atomicAdd(fifoTail)      │ (3) read FIFO entry       │                              │
  │     set ready = seq+1        │     (acquire fence)       │                              │
  │                              │                           │                              │
  │                              │ (4) paramWrite ──────────────▶ signal QP ──────────▶ paramSlot
  │                              │     (signal QP, 8B)       │                              │
  │                              │                           │                              │
  │                              │ (5) ibv_post_send ───────────▶ data QP ──────────▶ recvStaging
  │                              │     (RDMA Write,          │     (NIC DMA)                │
  │                              │      data QP)             │                              │
  │                              │                           │                              │
  │                              │ (6) ibv_poll_cq           │                              │
  │                              │     (data CQE)            │                              │
  │                              │                           │                              │
  │ (7) GPU polls ◀─────────────│ (6a) nicDoneCounter =     │                              │
  │     nicDoneCounter           │      step (WC fence)      │                              │
  │                              │                           │                              │
  │                              │ (6b) postAtomicFetchAdd ─────▶ signal QP ──────────▶ signalPad
  │                              │     (DATA_READY,          │     (RDMA Atomic)      [block_id]
  │                              │      signal QP)           │                              │
  │                              │                           │                     (8) GPU polls
  │                              │                           │                         DATA_READY
  │                              │                           │                              │
  │                              │                           │                     (9) validate
  │                              │                           │                         paramSlot
  │                              │                           │                              │
  │                              │                           │                     (10) __threadfence
  │                              │                           │                          _system()
  │                              │                           │                              │
  │                              │                           │                     (11) memcpy
  │                              │                           │                     staging→dst
  │                              │                           │                              │
  │                              │                           │ ◀──────────────── (12) SLOT_FREE
  │                              │                           │     FIFO entry        FIFO entry
  │                              │                           │                              │
  │                              │          signalPad ◀──────│ (13) postAtomicFetchAdd
  │ (14) GPU polls ◀───── [maxBlocks+block_id]              │     (RDMA Atomic,
  │      SLOT_FREE               │                           │      signal QP)
  │                              │                           │
```

### Proxy Loop Phases (per iteration)

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                        PROXY MAIN LOOP                             │
  │                                                                     │
  │  Phase 0: RETRY REPLAY                                             │
  │  ┌──────────────────────────────────────────────────┐              │
  │  │ for each peer p:                                  │              │
  │  │   if peerAborted_[p]: clear queues, skip          │              │
  │  │   replay retryQueue_[p] (param+data pairs)        │              │
  │  │   replay signalRetryQueue_[p] (atomics)           │              │
  │  └──────────────────────────────────────────────────┘              │
  │                          ▼                                          │
  │  Phase 1: FIFO DRAIN                                               │
  │  ┌──────────────────────────────────────────────────┐              │
  │  │ for each peer p (up to maxEntries per peer):      │              │
  │  │   acquire fence → read entry                      │              │
  │  │   if SLOT_FREE: postAtomicFetchAdd(signalQP)      │              │
  │  │   if DATA:                                        │              │
  │  │     if paramWord≠0: postParamWrite(signalQP)      │              │
  │  │       on ENOMEM: defer entire entry to retryQueue  │              │
  │  │     postRdmaWrite(dataQP) → pendingCompletions    │              │
  │  │   release fence → clear ready → advance head      │              │
  │  └──────────────────────────────────────────────────┘              │
  │                          ▼                                          │
  │  Phase 1.5: FORCED-SIGNAL FLUSH                                    │
  │  ┌──────────────────────────────────────────────────┐              │
  │  │ for each peer p:                                  │              │
  │  │   if unsignaledCount_[p] > 0 or pendingFlush_[p]: │              │
  │  │     post zero-byte signaled Write (dataQP)        │              │
  │  │     push FLUSH_SENTINEL to pendingCompletions      │              │
  │  └──────────────────────────────────────────────────┘              │
  │                          ▼                                          │
  │  Phase 2: DATA CQ POLL                                             │
  │  ┌──────────────────────────────────────────────────┐              │
  │  │ ibv_poll_cq(dataCq_)                              │              │
  │  │ for each CQE:                                     │              │
  │  │   decode peer from wr_id                          │              │
  │  │   drain pendingCompletions_[peer] up to wr_id:    │              │
  │  │     if FLUSH_SENTINEL: skip (no counter/signal)   │              │
  │  │     else: nicDoneCounter += step (WC fence)       │              │
  │  │           postAtomicFetchAdd(DATA_READY)          │              │
  │  └──────────────────────────────────────────────────┘              │
  │                          ▼                                          │
  │  Phase 3: SIGNAL CQ POLL                                           │
  │  ┌──────────────────────────────────────────────────┐              │
  │  │ ibv_poll_cq(signalCq_)                            │              │
  │  │ (SQ slot reclamation + error detection only)      │              │
  │  └──────────────────────────────────────────────────┘              │
  └─────────────────────────────────────────────────────────────────────┘
```

### Signal Pad Layout (per peer)

```
  signalPad (host pinned, RDMA-registered)
  ┌─────────────────────────────────────────────────────────────────┐
  │ [0]          [1]          ...   [maxBlocks-1]                   │
  │ DATA_READY   DATA_READY         DATA_READY                     │
  │ (block 0)    (block 1)          (block N-1)                    │
  │ ← written by sender's proxy via RDMA Atomic Fetch-Add →       │
  ├─────────────────────────────────────────────────────────────────┤
  │ [maxBlocks]  [maxBlocks+1] ... [2*maxBlocks-1]                 │
  │ SLOT_FREE    SLOT_FREE          SLOT_FREE                      │
  │ (block 0)    (block 1)          (block N-1)                    │
  │ ← written by receiver's proxy via RDMA Atomic Fetch-Add →     │
  ├─────────────────────────────────────────────────────────────────┤
  │ [2*maxBlocks]                                                   │
  │ PARAM VALIDATION                                                │
  │ (chunk_size << 32 | active_blocks)                             │
  │ ← written by sender's proxy via RDMA Write →                  │
  └─────────────────────────────────────────────────────────────────┘
```

### Pipeline Timeline (pipelineDepth=2, steps_per_slot=1, 4 chunks)

```
  Block 0 timeline:     step=0        step=1        step=2        step=3
                        slot 0        slot 1        slot 0        slot 1
                        ──────        ──────        ──────        ──────
  send_tile:
    memcpy→staging     [████████]
    FIFO enqueue         │
    NIC RDMA Write        ╲──────▶[████████]
    memcpy→staging                 [████████]
    FIFO enqueue                     │
    NIC RDMA Write                    ╲──────▶ wait NIC_DONE(0)
    memcpy→staging                             [████████]
    ...                                                  ╲──────▶ wait NIC_DONE(1)

  recv_tile:
    wait DATA_READY    [........]
    memcpy staging→dst [████████]
    SLOT_FREE signal     │──────▶
    wait DATA_READY              [........]
    memcpy staging→dst           [████████]
    SLOT_FREE signal               │──────▶ unblocks sender's step=2

  Legend: [████████] = data in flight/processing
          [........] = waiting for signal
          ──────▶    = signal/dependency
```

### Components

1. **`P2pIbrcTransportDevice`** (device-side handle) — holds FIFO pointers,
   staging buffer handles, signal/counter buffers, step state. No QP handles.

2. **`IbrcProxyThread`** (CPU-side) — one thread per transport instance. Drains
   the GPU→CPU FIFO, posts `ibv_post_send()`, polls CQs, updates GPU-visible
   completion counters.

3. **`IbrcProxyFifo`** (shared memory) — ring buffer in device-mapped host
   memory (`cudaHostAlloc` with `cudaHostAllocMapped`). GPU threads write
   descriptors; CPU thread reads them.

4. **`MultipeerIbrcTransport`** (host-side manager) — creates RC QPs, registers
   buffers, exchanges rkeys, allocates staging/signal/counter buffers, launches
   proxy thread.

---

## 3. Transport Setup

### 3.1 Config (`MultipeerIbrcTransportConfig`)

```cpp
struct MultipeerIbrcTransportConfig {
  int cudaDevice{0};

  // Tile pipeline config
  std::size_t dataBufferSize{8 * 1024 * 1024};  // 8 MiB per slot
  int pipelineDepth{2};                          // slots in staging ring
  int tileMaxBlocks{128};                        // upper bound on concurrent blocks
  std::size_t minSignalBytes{0};                 // minimum max_signal_bytes callers will pass
                                                 // at send_tile/recv_tile time. 0 = per_block_slot
                                                 // (no sub-slot signaling). Used at construction
                                                 // to derive maxStepsPerSlot for QP/FIFO sizing.
                                                 // Runtime send_tile/recv_tile traps if
                                                 // max_signal_bytes < minSignalBytes.

  // IB connection config — depths auto-sized from pipeline config above
  uint32_t dataQpDepth{0};       // max outstanding WQEs on data QP. 0 = auto-size:
                                 //   pipelineDepth * dataBufferSize / minSignalBytes
                                 //   (or pipelineDepth * tileMaxBlocks if minSignalBytes == 0)
  uint32_t signalQpDepth{0};     // max outstanding WQEs on signal QP. 0 = auto-size:
                                 //   dataQpDepth + pipelineDepth * tileMaxBlocks
  uint8_t ibPort{1};              // IB port number
  uint8_t timeout{20};            // IB ACK timeout (4.096µs × 2^timeout)
  uint8_t retryCount{7};          // transport error retries
  uint8_t rnrRetry{7};            // RNR NAK retries (7 = infinite)
  uint8_t trafficClass{0};        // IB traffic class
  uint8_t serviceLevel{0};        // IB service level

  // Proxy config
  int proxyFifoDepth{2048};       // FIFO entries per peer (must be power of 2)
                                  // Default sized for tileMaxBlocks=128, pipelineDepth=2,
                                  // steps_per_slot up to 4. Auto-sized at construction if 0.
  int proxyMaxEntriesPerPeerPerPoll{16};  // fairness: max entries drained per peer per iteration
  bool proxyBusyPoll{true};       // busy-poll vs epoll for proxy thread
  int proxySignalEveryN{16};      // selective signaling: signal every Nth data WQE
  int proxySignalQpSignalEveryN{16}; // selective signaling for signal QP (same mechanism)

  // GDR config
  // auto: try DMA-BUF, then nvidia_peermem, then fall back to host bounce buffers
  // true: require GDR (fail if unavailable)
  // false: always use host bounce buffers
  enum class GdrMode { AUTO, REQUIRE, DISABLE };
  GdrMode gdrMode{GdrMode::AUTO};
};
```

### 3.2 Validation (throws at construction)

- `pipelineDepth >= 1`
- `tileMaxBlocks >= 1`
- `(dataBufferSize % 16) == 0` — must be 16-byte aligned so `slot_off = slot * dataBufferSize` produces aligned addresses for vectorized memcpy across all pipeline slots
- `(dataBufferSize / tileMaxBlocks) >= 16` — per-block slot must fit 16-byte vectorized memcpy
- `proxyFifoDepth` is 0 (auto-size) or a power of 2. If 0, computed as
  `nextPow2(tileMaxBlocks * pipelineDepth * (maxStepsPerSlot + 1))`
- `proxyFifoDepth >= tileMaxBlocks` — hard requirement. If `numBlocks >
  proxyFifoDepth`, all blocks claim FIFO slots via `atomicAdd` but blocks
  beyond `proxyFifoDepth` spin forever on `ready != 0` because the proxy
  cannot drain their slots until earlier blocks finish writing descriptors,
  which they may not do if SMs are occupied by spinning blocks (deadlock)
- Compute `maxStepsPerSlot`:
  - If `minSignalBytes == 0`: `maxStepsPerSlot = 1`
  - If `minSignalBytes > 0`:
    `maxStepsPerSlot = dataBufferSize / minSignalBytes`
    (worst case: `active_blocks=1`, `per_block_slot ≈ dataBufferSize`)
  - If `minSignalBytes > 0`: `minSignalBytes` must be a power of 2 and >= 16.
    **Note:** this does NOT guarantee that `minSignalBytes` (or any runtime
    `chunk_size`) divides `per_block_slot` for all `active_blocks` values.
    `per_block_slot = (dataBufferSize / active_blocks) & ~15ULL` depends on the
    runtime `active_blocks`, and integer division + alignment masking can produce
    values not divisible by `minSignalBytes` (e.g., `dataBufferSize=8MiB`,
    `active_blocks=3` → `per_block_slot=2796192`, not divisible by 4096).
    **Divisibility is enforced at runtime** by the `per_block_slot % chunk_size
    != 0` trap in Section 7.1. Callers must choose `active_blocks` values that
    produce compatible `per_block_slot` values, or use `max_signal_bytes = 0`
    (which sets `chunk_size = per_block_slot`, always divisible).
- **Key insight:** the total outstanding WQEs across all blocks is
  `pipelineDepth * steps_per_slot * active_blocks`
  = `pipelineDepth * (dataBufferSize / active_blocks / chunk_size) * active_blocks`
  = `pipelineDepth * dataBufferSize / chunk_size`.
  The `active_blocks` factor cancels out — fewer blocks means more WQEs per
  block but fewer blocks, so the total is constant. The worst case is thus
  `pipelineDepth * dataBufferSize / minSignalBytes` (when `minSignalBytes > 0`)
  or `pipelineDepth * tileMaxBlocks` (when `minSignalBytes == 0`).
- Auto-size QP depths (if set to 0):
  - `dataQpDepth = pipelineDepth * dataBufferSize / minSignalBytes`
    (or `pipelineDepth * tileMaxBlocks` if `minSignalBytes == 0`)
  - `signalQpDepth = dataQpDepth + pipelineDepth * tileMaxBlocks +
    ceil(tileMaxBlocks / maxEntriesPerPeerPerPoll)`
    (DATA_READY = same as data WQEs; SLOT_FREE = one per slot boundary per
    block; param writes = at most one per proxy iteration, with up to
    `ceil(tileMaxBlocks / maxEntriesPerPeerPerPoll)` iterations needed to
    drain all blocks' s=0 entries)
- Validation (if QP depths are explicitly set):
  - If `minSignalBytes > 0`:
    `dataQpDepth >= pipelineDepth * dataBufferSize / minSignalBytes`
  - If `minSignalBytes == 0`:
    `dataQpDepth >= pipelineDepth * tileMaxBlocks`
  - `signalQpDepth >= dataQpDepth + pipelineDepth * tileMaxBlocks +
    ceil(tileMaxBlocks / maxEntriesPerPeerPerPoll)`

**Runtime `max_signal_bytes` check (in `send_tile`/`recv_tile`):**
The device code traps if the runtime `chunk_size` would produce a
`steps_per_slot` exceeding the construction-time provisioning:
- If `minSignalBytes == 0` (default): sub-slot signaling is disallowed. Any
  `max_signal_bytes` value that makes `chunk_size < per_block_slot` is rejected.
  To use sub-slot signaling, set `minSignalBytes` at construction.
- If `minSignalBytes > 0`: any `chunk_size < minSignalBytes` is rejected.
- `proxySignalEveryN >= 1` — 0 would prevent CQE generation, causing deadlock
- `proxySignalEveryN <= dataQpDepth / 2` — SQ must not fill before a signaled WQE completes
- `proxySignalQpSignalEveryN >= 1` — same constraint for signal QP
- `proxySignalQpSignalEveryN <= signalQpDepth / 2` — same depth bound for signal QP
- `proxyMaxEntriesPerPeerPerPoll >= 1` — 0 would prevent FIFO drain, causing deadlock

**Runtime FIFO depth check (at `send_tile`/`recv_tile` time):**
`proxyFifoDepth` must be large enough for the worst-case enqueue burst. With
`steps_per_slot = perBlockSlotSize / chunk_size`, each `send_tile` call enqueues
`pipelineDepth * steps_per_slot` FIFO entries per block, and each `recv_tile`
enqueues `pipelineDepth` SLOT_FREE entries per block. Both sender and receiver
share the same per-peer FIFO, so worst-case total is:
`tileMaxBlocks * (pipelineDepth * steps_per_slot + pipelineDepth)`. With
`steps_per_slot=4`, `pipelineDepth=2`, `tileMaxBlocks=128`, this is
`128 * (8 + 2) = 1280`. The default `proxyFifoDepth=2048` covers this case, but
with higher `steps_per_slot` or `tileMaxBlocks` the default may be insufficient.
**Recommendation:** Size `proxyFifoDepth` conservatively at construction as
`nextPow2(tileMaxBlocks * pipelineDepth * (maxStepsPerSlot + 1))`, or validate
at runtime when `max_signal_bytes` is first used and trap with a diagnostic if
insufficient.

---

## 4. Internal State

### 4.1 `IbrcTileState` (device-side, per-peer)

```cpp
struct IbrcTileState {
  // Staging ring buffers (pipelineDepth × dataBufferSize bytes each)
  char* sendStagingPtr{nullptr};   // local staging (GPU memcpy target)
  char* recvStagingPtr{nullptr};   // local recv staging (RDMA write target)

  // RDMA addressing for staging buffers (needed by GPU to fill FIFO entries)
  uint32_t sendStagingLkey{0};     // MR lkey for local sendStaging
  uint64_t remoteRecvStagingAddr{0}; // peer's recvStaging base address
  uint32_t remoteRecvStagingRkey{0}; // peer's recvStaging MR rkey

  // Signal pad: uint64_t slots, updated by CPU proxy (DATA_READY)
  // or by remote's CPU proxy via RDMA atomic (SLOT_FREE)
  // [0..maxBlocks)           = DATA_READY (sender→receiver)
  // [maxBlocks..2*maxBlocks) = SLOT_FREE  (receiver→sender)
  volatile uint64_t* localSignalPad{nullptr};    // polled by GPU
  // Remote signal pad address + rkey (peer's signalPad, for RDMA atomics).
  // On rank A's tile state for peer B: points to B's signalPad.
  //   - Sender (rank A) uses this for DATA_READY to B's signalPad[block_id]
  //   - Receiver (rank B) uses its own tile state for peer A, which points to
  //     A's signalPad, for SLOT_FREE to A's signalPad[maxBlocks + block_id]
  uint64_t remoteSignalAddr{0};
  uint32_t remoteSignalRkey{0};

  // NIC completion counters: uint64_t slots, written by CPU proxy
  // after ibv_poll_cq confirms the RDMA Write completed.
  // [0..maxBlocks) = NIC_DONE
  volatile uint64_t* nicDoneCounter{nullptr};    // polled by GPU

  // Per-block step counters (persistent across kernel launches)
  // [0..maxBlocks)           = sender steps
  // [maxBlocks..2*maxBlocks) = receiver steps
  int64_t* stepState{nullptr};

  // GPU→CPU FIFO for RDMA work requests
  IbrcProxyFifoEntry* fifoEntries{nullptr};  // ring buffer (device-mapped host mem)
  uint64_t* fifoTail{nullptr};                // in GPU device memory (fast atomicAdd, uint64_t to prevent wrap)
  uint32_t fifoMask{0};                       // fifoDepth - 1 (for & masking, works with uint64_t via truncation)

  // Abort flag: set by CPU proxy on CQ errors, checked by GPU wait loops
  volatile uint32_t* abortFlag{nullptr};      // device-mapped host memory

  // Config values needed on device
  int maxBlocks{0};
  int pipelineDepth{0};
  std::size_t dataBufferSize{0};
  std::size_t minSignalBytes{0};   // from MultipeerIbrcTransportConfig; used by
                                   // runtime guard in send_tile/recv_tile to reject
                                   // chunk_size values that exceed QP provisioning
};
```

**Implementation note: `volatile` vs `cuda::atomic_ref`.**
The struct above uses `volatile` pointers for GPU-polled host-mapped memory
(signalPad, nicDoneCounter, abortFlag). While `volatile` works on current
NVIDIA hardware (PTX compiles it to `ld.cg`/`st.cg` — cache-global,
non-caching loads), it is a C/C++ compiler directive, not a hardware memory
model directive. The formally correct, future-proof approach is
`cuda::atomic_ref<uint64_t, cuda::thread_scope_system>` for system-scope
loads from host-mapped memory. Implementations should use `cuda::atomic_ref`
with `memory_order_relaxed` for polling loads and pair with explicit
`cuda::atomic_thread_fence(memory_order_acquire, thread_scope_system)` where
ordering is needed (e.g., after DATA_READY observation, before reading
recvStaging). The `volatile` declarations in the struct are retained for
pseudocode clarity but should be replaced with raw pointers + `atomic_ref`
wrappers in the implementation.

### 4.2 `IbrcProxyFifoEntry` (shared memory)

Each FIFO entry describes one RDMA operation for the CPU proxy to execute.

**Multi-producer safety:** Multiple blocks enqueue concurrently via `atomicAdd`
on `fifoTail`. Because blocks may complete their descriptor writes out of order,
each entry has a `ready` flag. The CPU proxy must spin on `entry.ready` before
reading the descriptor — checking only `fifoTail` is insufficient (see Section
5 for the full protocol).

```cpp
struct alignas(64) IbrcProxyFifoEntry {
  // Ready flag: set by GPU producer AFTER writing all other fields +
  // __threadfence_system(). CPU proxy spins on this before consuming.
  // Uses uint64_t (not uint32_t) to prevent ABA wrap-around: with uint32_t,
  // seq + 1 wraps to 0 after ~4B entries, matching the "empty" sentinel
  // and silently deadlocking the FIFO (~72 min at 1M entries/sec).
  // With uint64_t, wrap takes ~584 years at 1 entry/ns.
  volatile uint64_t ready;  // 0 = empty/in-progress, seq + 1 = ready

  // Source (local staging)
  uint64_t localAddr;       // sendStaging + offset
  uint32_t localLkey;       // MR lkey for local staging

  // Destination (remote recvStaging)
  uint64_t remoteAddr;      // peer's recvStaging + offset
  uint32_t remoteRkey;      // MR rkey for peer's recvStaging

  // Transfer
  uint32_t nbytes;          // chunk size

  // Signaling: what the CPU proxy should do after the RDMA Write completes
  uint8_t signalType;       // 0=none, 1=DATA_READY, 2=SLOT_FREE, 3=FLUSH_SENTINEL
  uint32_t signalId;        // signal pad slot index
  uint64_t signalVal;       // value to atomic-add (1 for DATA_READY, chunks_in_slot for SLOT_FREE)
  uint64_t remoteSignalAddr; // remote signal pad address (for RDMA atomic)
  uint32_t remoteSignalRkey; // remote signal pad rkey

  // Completion: which NIC_DONE counter to increment
  uint32_t counterId;       // nicDoneCounter slot index
  int64_t step;             // pipeline step value (written to nicDoneCounter on completion)

  // Sequencing (uint64_t to match ready flag and prevent ABA wrap)
  uint64_t seq;             // monotonic sequence number (for FIFO ordering)

  // Cross-rank parameter validation (always-on, see Section 7.6).
  // Set by GPU on the first data entry of each send_tile call (s == 0).
  // Packs active_blocks (lower 32 bits) and chunk_size (upper 32 bits).
  // The CPU proxy writes this to the receiver's param validation slot
  // via RDMA Write before posting the first DATA_READY atomic.
  // Zero for non-first entries and SLOT_FREE/FLUSH_SENTINEL entries.
  uint64_t paramWord;       // (chunk_size << 32) | active_blocks
};
```

The `alignas(64)` ensures each entry occupies a full cache line, preventing
false sharing between concurrent producers writing adjacent slots.

### 4.3 `wr_id` Encoding

All `ibv_post_send` calls encode a 64-bit `wr_id` that the CQ handlers use to
identify the peer and distinguish data, signal, and flush WQEs. The bit layout:

```text
Bit 63   Bit 62   Bits [48..61]   Bits [0..47]
──────   ──────   ────────────    ────────────
 type     type      peer_id         seq
```

| Function | Bits [63:62] | Bits [61:48] | Bits [47:0] | Used on |
|----------|-------------|-------------|------------|---------|
| `encodeWrId(peer, seq)` | `00` | `peer` | `seq` | Data QP |
| `encodeSignalWrId(peer, seq)` | `01` | `peer` | `seq` | Signal QP |
| `encodeFlushWrId(peer, flushSeq)` | `10` | `peer` | `flushSeq` | Data QP (flush) |

`decodePeer(wr_id)` extracts bits [61:48] — this is the same for all three
formats, so a single decoder works for both data and signal CQ handlers.

The type bits [63:62] are used only in Phase 2's drain loop to distinguish
flush sentinels from data entries (via the `FLUSH_SENTINEL` signalType in
`pendingCompletions_`, not by inspecting the CQE's wr_id type bits directly).

Constraints: `peer < 2^14 = 16384`, `seq < 2^48 ≈ 281T`. Both are comfortably
within range for any practical deployment.

### 4.4 Memory Layout

**Staging buffers** (per peer, each `pipelineDepth × dataBufferSize` bytes):

```
slot k (= step % pipelineDepth):
┌──────────────┬──────────────┬─────┬────────────────────┐
│ block 0 row  │ block 1 row  │ ... │ block (N-1) row    │
└──────────────┴──────────────┴─────┴────────────────────┘
  each row = perBlockSlotSize = (dataBufferSize / active_blocks) & ~15ULL
```

**Signal pad** (per peer, `(2 × maxBlocks + 1) × sizeof(uint64_t)`):
- `[0, maxBlocks)` = DATA_READY inbox (written by sender's CPU proxy via RDMA
  atomic after put completes)
- `[maxBlocks, 2×maxBlocks)` = SLOT_FREE inbox (written by receiver's CPU proxy
  via RDMA atomic after receiver GPU signals completion)
- `[2×maxBlocks]` = param validation slot (written by sender's CPU proxy via
  RDMA Write before first DATA_READY; read by receiver GPU for cross-rank
  `active_blocks`/`chunk_size` validation)

**NIC_DONE counters** (per peer, `maxBlocks × sizeof(uint64_t)`):
- Written by the **local** CPU proxy (simple volatile store, not RDMA) after
  `ibv_poll_cq()` confirms the RDMA Write completed. No RDMA loopback needed
  (unlike IBGDA's companion QP).

**Step state** (per peer, `2 × maxBlocks × sizeof(int64_t)`):
- `[0, maxBlocks)` = sender step counters
- `[maxBlocks, 2×maxBlocks)` = receiver step counters

---

## 5. GPU→CPU FIFO Protocol

The FIFO is a **multi-producer** (GPU block leader threads), **single-consumer**
(CPU proxy thread) ring buffer in device-mapped host memory.

Multiple blocks enqueue concurrently via `atomicAdd` on `fifoTail`. Because
blocks may complete their descriptor writes out of order, the CPU proxy cannot
simply read entries sequentially up to `fifoTail` — it must check a per-entry
`ready` flag to ensure the descriptor is fully written before consuming it.
This follows the same pattern as NCCL's `connFifo` (which uses `size != -1` as
the ready sentinel).

### Memory Ordering Contract (GPU producer ↔ CPU consumer)

```
  GPU producer (block leader)              CPU consumer (proxy)
  ═══════════════════════════              ═══════════════════

  (1) atomicAdd(fifoTail)
      → claim slot seq

  (2) spin: ready != 0?  ◀───────────── ready = 0 (cleared by CPU)

  (3) write descriptor fields
      (localAddr, remoteAddr,
       nbytes, signalType, ...)

  (4) __threadfence_system()  ─ ─ ─ ─┐
      (all fields → system scope)     │  paired fences
                                      │
  (5) ready = seq + 1  ─ ─ ─ ─ ─ ─ ─▶│
                                      │
                            ┌─ ─ ─ ─ ┘
                            ▼
                    (A) observe ready != 0

                    (B) acquire fence  ← pairs with (4)
                        (ensures descriptor
                         reads see GPU writes)

                    (C) read descriptor fields

                    (D) dispatch (post RDMA, etc.)

                    (E) release fence  ← ensures (C,D)
                        complete before (F)

                    (F) ready = 0      → GPU may see slot free
```

### Producer (GPU, `send_tile`/`recv_tile` leader thread)

```text
// (1) Claim a slot atomically. fifoTail is uint64_t in GPU device memory
//     (not host) to avoid slow PCIe atomics and prevent ABA wrap.
seq = atomicAdd(fifoTail, 1)   // uint64_t atomicAdd
entry_idx = seq & fifoMask     // truncates to ring index

// (2) Spin until slot is free (consumer has cleared the ready flag)
while (fifoEntries[entry_idx].ready != 0) {
    if (*abortFlag):
        printf("IBRC ABORT: FIFO slot spin (seq=%llu, idx=%u)\n",
               (unsigned long long)seq, entry_idx)
        __trap()
}

// (3) Write descriptor fields (ready flag is still 0)
fifoEntries[entry_idx].localAddr = ...
fifoEntries[entry_idx].remoteAddr = ...
fifoEntries[entry_idx].nbytes = ...
fifoEntries[entry_idx].signalType = ...
fifoEntries[entry_idx].seq = seq
// ... all other fields ...

// (4) System fence: ensure ALL descriptor fields are visible to CPU
//     before the ready flag is set
__threadfence_system()

// (5) Set ready flag — CPU proxy may now consume this entry
fifoEntries[entry_idx].ready = seq + 1  // nonzero = ready
```

**fifoTail location:** `fifoTail` is in **GPU device memory** (not host pinned
memory). GPU `atomicAdd` on device memory takes ~10ns (vs ~1-2µs for host
memory over PCIe). The CPU proxy reads the tail via the device-mapped pointer
(`cudaHostGetDevicePointer` reverse mapping or a separate host-side copy updated
by the GPU). In practice, the CPU proxy does NOT need to read `fifoTail` at
all — it simply scans entries and checks `ready` flags.

### Consumer (CPU proxy thread)

```text
while (running):
  // Phase 0: Replay retry queues BEFORE draining new FIFO entries.
  // Retries must be re-posted in original order before any newer work
  // for the same peer, or QP posting order would be violated (IB requires
  // WQEs on the same QP to complete in posting order; reordering causes
  // NIC_DONE/DATA_READY to fire for the wrong staging offsets).
  // Skip aborted peers — their QPs are in an error state; drop queued entries.
  for (each peer p):
      if (peerAborted_[p]):
          retryQueue_[p].clear()
          signalRetryQueue_[p].clear()
          continue
      // Phase 0 uses tryPost* helpers that do NOT enqueue on ENOMEM.
      // Data retry entries may have paramWord != 0 (deferred param+data
      // pair from Phase 1). The param write must succeed before the data
      // write is attempted, preserving the ordering guarantee.
      while (!retryQueue_[p].empty()):
          auto& r = retryQueue_[p].front()
          // If this entry has a deferred param write, post it first.
          if (r.paramWord != 0):
              int pret = tryPostParamWrite(signalQp_[p], r)
              if (pret == ENOMEM): break  // param still can't post; stop
              if (pret != 0):
                  peerAborted_[p] = true; *abortFlag_ = 1; break
              r.paramWord = 0  // param posted; clear so we don't re-post
          int ret = tryPostRdmaWrite(dataVQp_[p], r)
          if (ret == 0):
              pendingCompletions_[p].push(r)
              retryQueue_[p].pop()
          elif (ret == ENOMEM):
              break  // still SQ full; entry stays at front, retry after Phase 2/3
          else:
              peerAborted_[p] = true; *abortFlag_ = 1; break
      while (!signalRetryQueue_[p].empty()):
          auto& r = signalRetryQueue_[p].front()
          int ret = tryPostAtomicFetchAdd(signalQp_[p], r)
          if (ret == 0):
              signalRetryQueue_[p].pop()
          elif (ret == ENOMEM):
              break
          else:
              peerAborted_[p] = true; *abortFlag_ = 1; break

  // Phase 1: Drain FIFO — scan entries by ready flag, NOT by tail
  for (each peer p):
    entries_this_peer = 0
    paramWrittenThisStep_[p] = false  // reset per-iteration dedup flag
    while (entries_this_peer < maxEntriesPerPeerPerPoll):
      idx = localHead_[p] & fifoMask

      // Spin-check ready flag (NOT fifoTail)
      if (fifoEntries[p][idx].ready == 0):
          break  // no more ready entries for this peer

      // Acquire fence: pairs with the GPU producer's __threadfence_system()
      // + ready-flag store. On weakly ordered CPUs (ARM), without this fence
      // the CPU could speculatively read descriptor fields before the ready
      // flag store is globally visible, seeing stale/partial data. On x86
      // (TSO), loads are not reordered past loads, so this is a compiler
      // barrier only. The acquire fence ensures all subsequent reads of the
      // descriptor see the values written by the GPU before it set ready.
      std::atomic_thread_fence(std::memory_order_acquire)
      entry = fifoEntries[p][idx]

      // Dispatch based on entry type.
      // Skip posting for aborted peers — consume the entry (clear ready,
      // advance head) to unblock the FIFO, but do not post to errored QPs.
      if (!peerAborted_[p]):
          if (entry.signalType == SLOT_FREE):
              postAtomicFetchAdd(signalQp_[p], entry)
          else:
              // If this is the first data entry of a send_tile call
              // (paramWord != 0), post a param-slot RDMA Write on the
              // SIGNAL QP before the data Write. This must go on the
              // signal QP (not data QP) because DATA_READY atomics are
              // also on the signal QP, and IB same-QP ordering guarantees
              // the param write is visible at the receiver before any
              // DATA_READY atomic lands. The param write is a small
              // IBV_WR_RDMA_WRITE (8 bytes, unsignaled) to
              // remoteSignalAddr + 2*maxBlocks*sizeof(uint64_t).
              // Post param-slot RDMA Write for cross-rank validation.
              // Every block's s=0 entry carries paramWord, but the proxy
              // deduplicates: posts at most once per peer per step.
              //
              // CRITICAL: if the param write fails (ENOMEM), the data Write
              // for this entry must also be deferred — otherwise Phase 2
              // could post DATA_READY before the param write lands, breaking
              // the validation ordering. Both are pushed to retryQueue_ and
              // replayed together next iteration (Phase 0 replays param
              // writes before data writes).
              bool paramNeeded = (entry.paramWord != 0 and !paramWrittenThisStep_[p])
              bool paramOk = true
              if (paramNeeded):
                  paramOk = postParamWrite(signalQp_[p], entry)
                  if (paramOk): paramWrittenThisStep_[p] = true

              if (paramOk):
                  // Param write succeeded (or not needed) — post data Write.
                  if (postRdmaWrite(dataVQp_[p], entry)):
                      pendingCompletions_[p].push(entry)
              else:
                  // Param write failed (ENOMEM) — defer the ENTIRE entry
                  // (param + data) to retryQueue_. Phase 0 will replay the
                  // param write before the data write next iteration,
                  // preserving the ordering guarantee.
                  retryQueue_[p].push(entry)

      // Release fence BEFORE clearing ready: ensures all CPU reads of the
      // descriptor payload (and any ibv_post_send using those values) are
      // complete before the GPU sees ready=0 and may overwrite the slot.
      // On x86 (TSO) this is a no-op for loads, but on ARM the fence
      // prevents the ready=0 store from being reordered before prior loads.
      std::atomic_thread_fence(std::memory_order_release)
      fifoEntries[p][idx].ready = 0  // GPU may now reuse this slot
      localHead_[p]++
      entries_this_peer++

  // Phase 1.5: Force-signal flush for each peer's data QP.
  //
  // Ensures every iteration that posted data WQEs produces at least one
  // CQE for Phase 2 to drain. Also retries any deferred flushes from
  // prior iterations (pendingFlush_[p] set on ENOMEM).
  //
  // The zero-byte signaled Write is a no-op on the wire (no data moved)
  // but produces a CQE that triggers the Phase 2 batch-drain. IB spec
  // guarantees it completes after all prior WQEs on the same QP.
  for (each peer p):
      if (!peerAborted_[p] and (unsignaledCount_[p] > 0 or pendingFlush_[p])):
          // Post a zero-byte signaled RDMA Write and push a FLUSH_SENTINEL
          // entry into pendingCompletions_[p] so Phase 2's drain loop has a
          // matching entry to stop at. Without the sentinel, a flush CQE
          // would not match any queued data entry's wr_id, causing the drain
          // to pop entries belonging to WQEs posted AFTER the flush.
          //
          // Flush wr_ids use encodeFlushWrId(peer, flushSeq), which sets
          // bit 63 to distinguish from data wr_ids (encodeWrId uses bits
          // [0..62]). This prevents collisions: data seq values come from
          // the GPU's monotonic fifoTail counter, while flushSeq is a
          // separate per-peer counter in the proxy. Without disjoint
          // namespaces, a flush CQE could match a data entry or vice versa.
          uint64_t flushSeq = nextFlushSeq_[p]++
          uint64_t flushWrId = encodeFlushWrId(p, flushSeq)  // bit 63 set
          int ret = postZeroByteSignaledWrite(dataVQp_[p], flushWrId)
          if (ret == 0):
              // Push sentinel with matching wr_id so Phase 2 can match it.
              IbrcProxyFifoEntry sentinel{}
              sentinel.signalType = FLUSH_SENTINEL  // no NIC_DONE update, no DATA_READY
              sentinel.seq = flushSeq
              pendingCompletions_[p].push(sentinel)
              unsignaledCount_[p] = 0
              pendingFlush_[p] = false
          elif (ret == ENOMEM):
              pendingFlush_[p] = true  // retry next iteration
          else:
              peerAborted_[p] = true
              *abortFlag_ = 1
              LOG(ERROR) << "Flush WQE failed (peer=" << p << "): "
                         << strerror(ret)

  // Phase 2: Poll CQ for data write completions
  //
  // With selective signaling (proxySignalEveryN), only every Nth WQE
  // produces a CQE. The IB ordering guarantee means ALL preceding
  // unsignaled WQEs on the same QP are also complete. We must drain
  // pendingCompletions_ up to the signaled wr_id and update EACH
  // block's nicDoneCounter individually (different blocks have
  // different counterId values).
  // Poll via IbvVirtualCq — handles multi-QP fragment reassembly.
  // Only fires completions when all fragments of a striped Write are done.
  n = dataVirtualCq_->pollCq(kPollBatch, wcArray_)
  for (int i = 0; i < n; i++):
      if (wcArray_[i].status != IBV_WC_SUCCESS):
          int errPeer = decodePeer(wcArray_[i].wr_id)
          peerAborted_[errPeer] = true
          *abortFlag_ = 1
          LOG(ERROR) << "RDMA Write failed (peer=" << errPeer << "): "
                     << ibv_wc_status_str(wcArray_[i].status)
          continue

      // Decode peer from wr_id and drain THAT PEER's pending queue only.
      // pendingCompletions_ is per-peer (pendingCompletions_[peer]) because
      // a shared CQ serves all peers' QPs. IB ordering guarantees only hold
      // within a single QP — a CQE from peer A's QP says nothing about peer
      // B's QP. Draining a single global queue would incorrectly mark peer B's
      // entries as complete when peer A's CQE arrives.
      int peer = decodePeer(wcArray_[i].wr_id)
      while (!pendingCompletions_[peer].empty()):
          auto& c = pendingCompletions_[peer].front()

          // FLUSH_SENTINEL entries are no-ops — no NIC_DONE update, no
          // DATA_READY signal. They exist only to give Phase 2 a matching
          // wr_id to stop the drain at the correct position.
          if (c.signalType != FLUSH_SENTINEL):
              // Update this block's NIC_DONE counter.
              // cudaHostAlloc memory may be WC-mapped on the CPU. Use a
              // volatile store + arch-appropriate fence to flush WC buffers.
              volatile_store(&nicDoneCounters_[peer][c.counterId], c.step)
              wc_store_fence()  // _mm_sfence() on x86, __dmb(ST) on ARM

              // Post DATA_READY signal for this entry.
              if (c.signalType == DATA_READY):
                  postAtomicFetchAdd(signalQp_[peer], c)

          // Match against the CQE's wr_id. Data entries use encodeWrId
          // (bits [0..62]), flush sentinels use encodeFlushWrId (bit 63 set).
          // The disjoint namespaces prevent cross-matching.
          uint64_t entryWrId = (c.signalType == FLUSH_SENTINEL)
              ? encodeFlushWrId(peer, c.seq)
              : encodeWrId(peer, c.seq)
          bool is_signaled = (entryWrId == wcArray_[i].wr_id)
          pendingCompletions_[peer].pop()
          if (is_signaled): break

  // Phase 3: Poll signal CQ for SQ slot reclamation
  //
  // The IB spec requires CQ polling to reclaim SQ slots — unsignaled WQE
  // slots are NOT freed by the HCA on wire ACK alone. Without this, the
  // signal QP's SQ fills after signalQpDepth total posts and ibv_post_send
  // fails with ENOMEM. Signal CQEs don't need batch-draining of pending
  // entries — they are polled only for SQ slot reclamation and error detection.
  n = ibv_poll_cq(signalCq_, kPollBatch, wcArray_)
  for (int i = 0; i < n; i++):
      if (wcArray_[i].status != IBV_WC_SUCCESS):
          int errPeer = decodePeer(wcArray_[i].wr_id)
          peerAborted_[errPeer] = true
          *abortFlag_ = 1
          LOG(ERROR) << "Signal atomic failed (peer=" << errPeer << "): "
                     << ibv_wc_status_str(wcArray_[i].status)
```

### Separate QPs for Data and Signals

The proxy uses **two QPs per peer**: a data QP for RDMA Writes and a signal QP
for RDMA Atomics (DATA_READY + SLOT_FREE). This prevents signal atomics from
consuming data QP depth, which was identified as a correctness issue when using
a single QP (SQ exhaustion when `data_WQEs + signal_WQEs > qpDepth`).

This mirrors ctran's architecture, which uses separate data QPs, notify QP, and
atomic QP per virtual connection.

### Selective Signaling for Data WQEs

Not every data RDMA Write needs `IBV_SEND_SIGNALED`. The proxy signals every
`proxySignalEveryN`-th data WQE. On CQ completion of a signaled WQE, ALL
preceding unsignaled WQEs on the same QP are implicitly complete (IB ordering
guarantee).

**Critical:** The proxy must drain `pendingCompletions_[peer]` (per-peer queue)
up to the signaled `wr_id` and update `nicDoneCounter[counterId]` for **each**
entry individually.
Different blocks have different `counterId` values (consecutive FIFO entries
typically come from different blocks), so a single counter update for the
signaled WQE is insufficient. The proxy also posts DATA_READY RDMA atomics for
each drained entry. See the Phase 2 pseudocode in Section 5 for the batch-drain
loop.

This reduces CQ polling pressure from O(tileMaxBlocks) to
O(tileMaxBlocks / proxySignalEveryN) per pipeline step.

**Forced-signaled flush (Phase 1.5):** After Phase 0 and Phase 1 complete for
all peers, the proxy checks each peer's `unsignaledCount_`. If any data WQEs
were posted this iteration without a signaled WQE among them, the proxy posts
a zero-byte signaled RDMA Write on that peer's data QP. This no-op Write
produces a CQE that triggers the Phase 2 batch-drain, ensuring
`pendingCompletions_[peer]` is never stranded. Without this:
- Short transfers (fewer than `proxySignalEveryN` data WQEs) produce no CQE.
- Data WQEs followed only by signal-QP SLOT_FREE entries in the FIFO are not
  detectable as "last data WQE" by inspecting the FIFO alone.
- Retried data WQEs from Phase 0 also need this guarantee.

The zero-byte Write consumes one SQ slot and one CQE but moves no data on the
wire. IB spec guarantees it completes after all prior WQEs on the same QP,
so the resulting CQE correctly represents completion of all preceding data
Writes. The `unsignaledCount_` is reset to 0 after the flush.

### Signal QP Completion Handling

Signal QP WQEs (RDMA Atomic Fetch-Add for DATA_READY and SLOT_FREE) use
**selective signaling**, mirroring the data QP approach. The proxy signals every
`proxySignalQpSignalEveryN`-th signal WQE and polls the signal CQ in the main
loop (Phase 3, after Phase 2's data CQ poll).

This is required because the IB specification mandates CQ polling for SQ slot
reclamation: unsignaled WQE slots are NOT freed by the HCA on wire ACK alone.
Without periodic signaled WQEs + CQ polling, the signal QP's SQ fills after
`signalQpDepth` total posts, and subsequent `ibv_post_send` calls fail with
`ENOMEM`. This would manifest as a transport failure after just 1-2 pipeline
steps if `signalQpDepth` is undersized relative to `steps_per_slot`.

```text
// Phase 3 (in proxy main loop): Poll signal CQ
n = ibv_poll_cq(signalCq_, kPollBatch, wcArray_)
for (int i = 0; i < n; i++):
    if (wcArray_[i].status != IBV_WC_SUCCESS):
        int errPeer = decodePeer(wcArray_[i].wr_id)
        peerAborted_[errPeer] = true
        *abortFlag_ = 1
        LOG(ERROR) << "Signal atomic failed (peer=" << errPeer << "): "
                   << ibv_wc_status_str(wcArray_[i].status)
```

The proxy maintains a separate `unsignaledSignalCount_[peer]` counter and
signals every Nth signal WQE, just like data WQEs. Signal CQEs don't require
batch-draining of pending entries — they are polled only for SQ slot reclamation
and error detection.

### Key Design Decisions

**Why not use the companion QP loopback for NIC_DONE (like IBGDA)?**
IBGDA uses companion QPs because only the GPU can see the main QP's completion
(the CQ is GPU-mapped). In IBRC, the CPU proxy polls a standard `ibv_cq` and
can simply write the counter to GPU-visible memory via a volatile store +
WC fence (see Section 5 for why an explicit fence is required). No loopback
RDMA needed. This is simpler and avoids consuming an extra QP.

**Why use RDMA atomics for DATA_READY signals (not just a volatile write)?**
DATA_READY is written to the **remote** GPU's signal pad. The CPU proxy cannot
do a volatile GPU memory write to a remote machine. It must use an RDMA atomic
(Fetch-And-Add) to atomically increment the remote signal counter. This is the
same mechanism IBGDA uses, just CPU-initiated instead of GPU-initiated.

**Why is SLOT_FREE also an RDMA atomic?**
Same reason — the receiver's CPU proxy writes SLOT_FREE to the sender's signal
pad, which is on a remote machine's GPU memory. RDMA atomic is required.

**Why separate data and signal QPs?**
With a single QP, data WQEs + signal atomics compete for the same SQ depth.
With `tileMaxBlocks=128` and `pipelineDepth=2`, up to 256 data WQEs + 256
signal atomics could be outstanding simultaneously, requiring `qpDepth >= 512`.
Separate QPs allow independent depth sizing and prevent signal atomics from
blocking data writes (or vice versa).

**Why an explicit fence after CPU counter writes?**
`cudaHostAlloc` memory may be mapped as write-combining (WC) on the CPU side.
WC stores can sit in CPU write-combine buffers indefinitely — a plain store
(even with `memory_order_release` on x86, which compiles to `MOV`) does not
flush them. The GPU would poll stale values. We use a volatile store followed
by an architecture-appropriate fence:
- **x86:** `_mm_sfence()` — flushes WC buffers. Lighter than `MFENCE`
  (`memory_order_seq_cst`), which also serializes loads unnecessarily.
- **ARM (GB200 Grace-Blackwell):** `__dmb(ST)` or `STLR` — drains the store
  buffer. ARM does not have WC in the same sense, but the fence ensures
  visibility.

We intentionally avoid relying on `std::memory_order_seq_cst` because the C++
memory model does not define WC buffer flush semantics — compiler
implementations vary in what instruction they emit for `seq_cst` stores, and
some may not flush WC buffers on all architectures. The explicit fence is
portable and unambiguous. This follows NCCL's approach of using
`__sync_synchronize()` (full barrier) for WC memory writes, but is more
targeted (store fence only, not a full barrier).

### Error Handling and Abort

The CPU proxy checks `wc.status` on every CQ completion. On any error
(`IBV_WC_RETRY_EXC_ERR`, `IBV_WC_RNR_RETRY_EXC_ERR`, etc.), it sets an
`abortFlag` in device-mapped host memory. GPU-side `wait_volatile()` loops
check `*abortFlag` on each iteration and trap immediately if set, providing a
diagnostic rather than hanging forever. This matches NCCL's `comm->abortFlag`
pattern.

---

## 6. API Surface

### 6.1 C++ (P2pIbrcTransportDevice)

```cpp
class P2pIbrcTransportDevice {
 public:
  __device__ void send_tile(
      ThreadGroup& group,
      const void* src,
      std::size_t nbytes,
      int active_blocks,              // REQUIRED — must match sender and receiver
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout());

  __device__ void recv_tile(
      ThreadGroup& group,
      void* dst,
      std::size_t nbytes,
      int active_blocks,              // REQUIRED — must match sender and receiver
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout());

  __device__ void drain_tile(
      ThreadGroup& group,
      const Timeout& timeout = Timeout());
};
```

`active_blocks` is **required** (no default) — matching NVL's `numBlocks` and
IBGDA's `numBlocks`. When one side passes an explicit value and the other
defaults to `tileMaxBlocks`, staging offsets diverge silently causing data
corruption. Making it required forces the caller to always be explicit.

Signature otherwise matches the unified Triton extern API (D101445960). The
existing NVL/IBGDA C++ APIs use older parameter names (`numBlocks`,
`chunksPerSlot`) — they will be aligned in a follow-up.

### 6.2 Triton (via existing extern layer)

The existing `torchcomms_transport_send_tile` / `recv_tile` (from D101445960)
dispatch via `handle->get_type(peer)`. A new `torchcomms_transport_drain_tile`
extern is required for IBRC (send_tile does NOT drain internally — see Section
7.2):

```cpp
__device__ __noinline__ int torchcomms_transport_send_tile(
    void* handle_ptr, int peer, void* src_ptr,
    unsigned long long nbytes, int block_id,
    int active_blocks, unsigned long long max_signal_bytes) {
  auto* handle = reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr);
  auto group = make_block_group();
  group.group_id = block_id;
  auto type = handle->get_type(peer);
  if (type == TransportType::P2P_NVL) {
    handle->get_nvl(peer).send_tile(group, src_ptr, nbytes, active_blocks, max_signal_bytes);
  } else if (type == TransportType::P2P_IBGDA) {
    handle->get_ibgda(peer).send_tile(group, src_ptr, nbytes, active_blocks, max_signal_bytes);
  } else if (type == TransportType::P2P_IBRC) {
    handle->get_ibrc(peer).send_tile(group, src_ptr, nbytes, active_blocks, max_signal_bytes);
  }
  return 0;
}

// drain_tile: MUST be called after send_tile on ALL transports.
// For IBRC, this waits for SLOT_FREE confirmation. For NVL and IBGDA,
// this is a no-op (they drain internally in send_tile), but callers
// MUST still call it so that user code is transport-agnostic. Code
// written for NVL that omits drain_tile will silently corrupt data
// when run on IBRC. A debug-mode assert can verify drain_tile is
// called by checking a per-transport "pending" flag.
__device__ __noinline__ int torchcomms_transport_drain_tile(
    void* handle_ptr, int peer, int block_id) {
  auto* handle = reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr);
  auto group = make_block_group();
  group.group_id = block_id;
  auto type = handle->get_type(peer);
  if (type == TransportType::P2P_IBRC) {
    handle->get_ibrc(peer).drain_tile(group);
  }
  // NVL and IBGDA: no-op (drain is internal to send_tile).
  return 0;
}
```

### 6.3 Parameter Table

Same as the unified tile API (see `tile_sendrecv.md`):

| Param | Required | Default | Meaning |
|-------|----------|---------|---------|
| `group` / `block_id` | yes | — | Identifies calling block for slot routing |
| `src` / `dst` | yes | — | Pre-sliced data pointer |
| `nbytes` | yes | — | This block's data size |
| `active_blocks` | yes | — | Number of concurrent blocks. Must match sender and receiver. |
| `max_signal_bytes` | no | `0` → `perBlockSlotSize` | Signaling granularity hint |
| `timeout` | no | `Timeout()` | Per-wait timeout. Caller must call `timeout.start()` before passing to `send_tile`/`recv_tile`/`drain_tile` — `checkExpired()` returns false until `start()` is called (see `Timeout.cuh`). `Timeout()` (default) = infinite wait. |

---

## 7. Algorithm

### 7.1 Common Precomputation (same as NVL/IBGDA)

```text
block_id        = group.group_id
trap if active_blocks <= 0          // required parameter, no default
trap if active_blocks > maxBlocks   // signal/step arrays sized to maxBlocks
trap if block_id >= active_blocks

per_block_slot  = (dataBufferSize / active_blocks) & ~15ULL
trap if per_block_slot == 0
chunk_size      = min(max_signal_bytes > 0 ? max_signal_bytes : per_block_slot,
                      per_block_slot)
// Runtime guard: chunk_size must not produce a steps_per_slot exceeding what
// the QP/FIFO depths were provisioned for at construction.
// - If minSignalBytes == 0: construction assumed steps_per_slot = 1, so
//   chunk_size must equal per_block_slot (no sub-slot signaling allowed).
// - If minSignalBytes > 0: chunk_size must be >= minSignalBytes.
if minSignalBytes == 0:
    trap if chunk_size != per_block_slot
        // QP/FIFO depths were provisioned for steps_per_slot = 1.
        // Sub-slot signaling (max_signal_bytes < per_block_slot) requires
        // setting minSignalBytes at construction to provision adequate depth.
else:
    trap if chunk_size < minSignalBytes
        // chunk_size (from max_signal_bytes) is smaller than the minimum
        // the QP/FIFO depths were provisioned for.
trap if per_block_slot % chunk_size != 0
    // chunk_size must divide per_block_slot exactly. Non-divisible values
    // would waste staging space and make steps_per_slot inaccurate (truncating
    // integer division). The slot/signal logic assumes steps_per_slot *
    // chunk_size == per_block_slot. Callers should use power-of-2 values for
    // max_signal_bytes (e.g., 16384, 32768) to satisfy this naturally.
total_chunks    = ceil(nbytes / chunk_size)
steps_per_slot  = per_block_slot / chunk_size  // exact (no truncation)
```

### 7.2 `send_tile` (IBRC)

```text
// Guard: detect forgotten drain_tile from a previous call.
// Bit 63 of stepState[block_id] is the "pending sends" flag.
// send_tile sets it; drain_tile clears it. If set on entry,
// the previous send_tile was not drained — staging may be in use.
// This check runs BEFORE the nbytes == 0 early return so that a
// zero-byte call after a missed drain is still diagnosed.
raw_step = stepState[block_id]
if (raw_step & (1LL << 63)) != 0:
    printf("IBRC FATAL: send_tile called without drain_tile "
           "(block_id=%d, step=%lld)\n", block_id, raw_step & ~(1LL << 63))
    __trap()
step = raw_step  // bit 63 is 0 here (cleared by drain or initial zero)

if nbytes == 0: return

for s in [0, total_chunks):
    slot        = (s / steps_per_slot) % pipelineDepth
    sub_step    = s % steps_per_slot
    slot_off    = slot * dataBufferSize
    chunk_off   = sub_step * chunk_size
    staging_off = slot_off + block_id * per_block_slot + chunk_off
    data_off    = s * chunk_size
    bytes_this  = min(chunk_size, nbytes - data_off)

    // (1) Wait NIC_DONE: local staging safe to overwrite.
    //     CPU proxy has confirmed RDMA Write completed from this slot.
    if step >= pipelineDepth * steps_per_slot:
        wait_volatile_with_abort(nicDoneCounter[block_id],
                      step - pipelineDepth * steps_per_slot + 1,
                      abortFlag, timeout)

    // (2) Cooperative memcpy: src chunk → local sendStaging.
    memcpy_vectorized(sendStagingPtr + staging_off, src + data_off,
                      bytes_this, group)
    group.sync()             // all threads complete their memcpy portion
    __threadfence_system()   // ALL threads execute: each thread's writes
                             // are promoted to system scope (visible to NIC
                             // over PCIe). The fence only orders the CALLING
                             // thread's writes, so every thread that
                             // participated in the memcpy must execute it.

    // (3) Wait SLOT_FREE: remote recvStaging safe for new RDMA Write.
    //     Only at slot boundaries, only after pipeline is filled.
    //     SLOT_FREE counter is in step-space (recv adds chunks_in_slot per
    //     slot boundary), so the formula matches NIC_DONE's formula exactly.
    if sub_step == 0 and step >= pipelineDepth * steps_per_slot:
        wait_volatile_with_abort(localSignalPad[maxBlocks + block_id],
                      step - pipelineDepth * steps_per_slot + 1,
                      abortFlag, timeout)

    // (4) Enqueue RDMA Work to CPU proxy via FIFO.
    //     Leader claims a slot, writes descriptor, fences, sets ready flag.
    //     CPU proxy will:
    //       a) ibv_post_send() RDMA Write on data QP (staging → remote recvStaging)
    //       b) ibv_poll_cq(), then volatile-store + wc_store_fence() nicDoneCounter[block_id]
    //       c) ibv_post_send() RDMA Atomic Fetch-Add on signal QP to
    //          remote signalPad[block_id] (DATA_READY signal)
    if group.is_leader():
        if *abortFlag:
            printf("IBRC ABORT: send_tile pre-enqueue "
                   "(block_id=%d, step=%lld)\n", block_id, (long long)step)
            __trap()

        seq = atomicAdd(fifoTail, 1)
        idx = seq & fifoMask
        while (fifoEntries[idx].ready != 0) {
            if (*abortFlag):
                printf("IBRC ABORT: send_tile FIFO slot spin "
                       "(block_id=%d, step=%lld, seq=%llu, idx=%u)\n",
                       block_id, (long long)step, (unsigned long long)seq, idx)
                __trap()
        }
        fifoEntries[idx].localAddr  = sendStagingPtr + staging_off
        fifoEntries[idx].localLkey  = sendStagingLkey
        fifoEntries[idx].remoteAddr = remoteRecvStagingAddr + staging_off
        fifoEntries[idx].remoteRkey = remoteRecvStagingRkey
        fifoEntries[idx].nbytes     = bytes_this
        fifoEntries[idx].signalType = DATA_READY
        fifoEntries[idx].signalId   = block_id
        fifoEntries[idx].signalVal  = 1  // DATA_READY adds 1 per chunk
        fifoEntries[idx].remoteSignalAddr = remoteSignalAddr
        fifoEntries[idx].remoteSignalRkey = remoteSignalRkey
        fifoEntries[idx].counterId  = block_id
        fifoEntries[idx].step       = step + 1  // nicDoneCounter value after this put
        fifoEntries[idx].seq        = seq
        // Pack params for cross-rank validation. Every block's first entry
        // (s == 0) carries the value. The proxy deduplicates and posts the
        // param-slot RDMA Write at most once per step per peer (see Phase 1).
        // Having every block carry it ensures the param write is posted in
        // the same Phase 1 batch as that block's data entry, guaranteeing
        // same-QP ordering with that block's DATA_READY (posted in Phase 2).
        fifoEntries[idx].paramWord  = (s == 0)
            ? ((uint64_t)chunk_size << 32) | (uint64_t)active_blocks
            : 0
        __threadfence_system()      // all fields visible before ready
        fifoEntries[idx].ready = seq + 1  // nonzero = ready

    step++

// Set bit 63 to mark "pending sends — drain required"
stepState[block_id] = step | (1LL << 63)
group.sync()
```

**No internal drain.** Unlike IBGDA (which does `wait_counter(nic_done, step)`
at the end), IBRC's `send_tile` does NOT drain. The drain is deferred to
`drain_tile()`, which the caller invokes after `send_tile` completes. This
matches the D101055842 IBGDA pattern. **Note:** consecutive `send_tile` calls
to the **same peer** without an intervening `drain_tile` will trap (bit 63
guard) because staging buffers may still be in use by in-flight RDMA writes.
To send to **different peers** before draining, call `send_tile` on each peer's
transport, then `drain_tile` on each — the drain is per-peer and per-transport.

### 7.3 `recv_tile` (IBRC)

```text
if nbytes == 0: return

step = stepState[maxBlocks + block_id]

for s in [0, total_chunks):
    slot        = (s / steps_per_slot) % pipelineDepth
    sub_step    = s % steps_per_slot
    slot_off    = slot * dataBufferSize
    chunk_off   = sub_step * chunk_size
    staging_off = slot_off + block_id * per_block_slot + chunk_off
    data_off    = s * chunk_size
    bytes_this  = min(chunk_size, nbytes - data_off)

    // (1) Wait DATA_READY: data has arrived in local recvStaging.
    //     Sender's CPU proxy wrote this via RDMA atomic after put completed.
    wait_volatile_with_abort(localSignalPad[block_id], step + 1,
                             abortFlag, timeout)

    // Cross-rank parameter validation (always-on, see Section 7.6).
    // Every block validates on its first chunk. The sender's proxy wrote
    // the param slot via RDMA Write BEFORE any DATA_READY atomic (same QP
    // ordering), so it is visible to all blocks when they observe DATA_READY.
    if s == 0:
        param = localSignalPad[2 * maxBlocks]  // param slot
        remote_active = param & 0xFFFFFFFF
        remote_cs = (param >> 32) & 0xFFFFFFFF
        if remote_active != active_blocks or remote_cs != chunk_size:
            printf("IBRC FATAL: cross-rank parameter mismatch "
                   "(block_id=%d, local active_blocks=%d, remote=%u, "
                   "local chunk_size=%llu, remote=%u)\n",
                   block_id, active_blocks, remote_active,
                   (unsigned long long)chunk_size, remote_cs)
            __trap()

    // IB guarantees the data Write landed before the DATA_READY atomic was
    // visible. This system fence establishes a happens-before so subsequent
    // SM loads from recvStaging see the NIC-written data. Same pattern as NCCL.
    __threadfence_system()

    // (2) Cooperative memcpy: local recvStaging → user dst.
    memcpy_vectorized(dst + data_off, recvStagingPtr + staging_off,
                      bytes_this, group)
    group.sync()

    // (3) Signal SLOT_FREE to sender.
    //     Only at slot boundaries (last sub-step or last step).
    //     GPU enqueues a FIFO entry asking CPU proxy to RDMA atomic
    //     Fetch-Add of chunks_in_slot (= sub_step + 1) to sender's
    //     signalPad[maxBlocks + block_id]. This ensures the SLOT_FREE
    //     counter uses the same step-space as NIC_DONE and drain_tile,
    //     so drain_tile can wait on `step` directly.
    bool last_in_slot = (sub_step == steps_per_slot - 1) or (s == total_chunks - 1)
    if last_in_slot and group.is_leader():
        if *abortFlag:
            printf("IBRC ABORT: recv_tile pre-enqueue "
                   "(block_id=%d, step=%lld)\n", block_id, (long long)step)
            __trap()

        // Compute how many chunks were actually consumed in this slot.
        // For full slots: sub_step == steps_per_slot - 1, so chunks_in_slot
        // = steps_per_slot. For the final partial slot (s == total_chunks - 1
        // with sub_step < steps_per_slot - 1): chunks_in_slot = sub_step + 1.
        // SLOT_FREE must add exactly this count to stay in the same step-space
        // as stepState. Adding steps_per_slot unconditionally would overcount
        // on partial final slots, inflating the sender's free counter and
        // letting drain_tile/send_tile proceed before staging is truly safe.
        int chunks_in_slot = sub_step + 1

        seq = atomicAdd(fifoTail, 1)
        idx = seq & fifoMask
        while (fifoEntries[idx].ready != 0) {
            if (*abortFlag):
                printf("IBRC ABORT: recv_tile FIFO slot spin "
                       "(block_id=%d, step=%lld, seq=%llu, idx=%u)\n",
                       block_id, (long long)step, (unsigned long long)seq, idx)
                __trap()
        }
        fifoEntries[idx].signalType = SLOT_FREE
        fifoEntries[idx].signalId   = maxBlocks + block_id
        fifoEntries[idx].signalVal  = chunks_in_slot  // exact count, not steps_per_slot
        fifoEntries[idx].remoteSignalAddr = remoteSignalAddr  // peer's signalPad
        fifoEntries[idx].remoteSignalRkey = remoteSignalRkey
        fifoEntries[idx].seq = seq
        __threadfence_system()
        fifoEntries[idx].ready = seq + 1

    step++

stepState[maxBlocks + block_id] = step
group.sync()
```

### 7.4 `drain_tile`

```text
raw_step = stepState[block_id]
step = raw_step & ~(1LL << 63)  // mask off the pending flag
if step > 0:
    // Wait for all SLOT_FREE signals from the receiver.
    // This confirms the receiver has consumed all data.
    // SLOT_FREE counter is in the same step-space as NIC_DONE because
    // recv_tile adds chunks_in_slot per slot boundary (sub_step + 1).
    // After all chunks, SLOT_FREE counter == step.
    wait_volatile_with_abort(localSignalPad[maxBlocks + block_id], step,
                             abortFlag, timeout)

// Clear bit 63 — drain is complete, send_tile can be called again
stepState[block_id] = step
```

**`wait_volatile_with_abort`** is a polling loop that checks the target counter
(`*counter >= expected`) on each iteration and the abort flag (`*abortFlag != 0`)
periodically (e.g., every `kAbortCheckInterval` iterations — see backoff
guidance below). Checking abort on every iteration is unnecessary since abort
is a rare event, and both the counter and abort flag reside in host-pinned
memory (each read is a ~1-2µs PCIe round-trip). Amortizing abort checks
trades worst-case abort detection latency (~1-2ms at 1024-iteration intervals)
for reduced PCIe traffic. On abort or timeout, the loop prints a diagnostic
with the wait type, block_id, step, expected value, and current counter value
before calling `__trap()`.

**Polling backoff (implementation tuning knob):** Since the counter and abort
flag reside in host-pinned memory, each poll is a PCIe round-trip (~1-2µs).
With 128 blocks polling concurrently, tight spin-loops generate measurable
PCIe read traffic and waste SM cycles. Implementations should consider a
bounded spin-then-backoff policy: spin for a short burst (e.g., 64-128
iterations) for low-latency hits, then fall back to `__nanosleep()`-gated
polls (e.g., 200-1000ns pause between reads) to reduce PCIe pressure. The
backoff parameters are workload-dependent and should be tunable, not
hard-coded. Abort flag checks can be amortized (e.g., every 1024 iterations,
matching NCCL's `testAbort()` pattern) since abort is a rare event:

```text
while (*counter < expected):
    if (++spins > kSpinBurst):
        __nanosleep(kBackoffNs)   // SM 70+; reduces PCIe poll pressure
    if (spins % kAbortCheckInterval == 0):
        if (*abortFlag): trap_with_diagnostic()
    if (timeout.checkExpired()): trap_with_diagnostic()
```

```cpp
printf("IBRC ABORT: wait_%s (block_id=%d, step=%lld, "
       "expected=%llu, current=%llu)\n",
       wait_type, block_id, (long long)step,
       (unsigned long long)expected, (unsigned long long)*counter);
__trap();
```

This follows the pipes convention (`TIMEOUT_TRAP_IF_EXPIRED_SINGLE` in
`Timeout.cuh`) where every `__trap()` site is preceded by a context-rich
`printf`. The wait_type string is one of `"NIC_DONE"`, `"SLOT_FREE"`, or
`"DATA_READY"`, identifying which protocol step hung.

**Why SLOT_FREE adds `chunks_in_slot` (not 1):** With sub-slot signaling
(`max_signal_bytes < per_block_slot`), there are `steps_per_slot` chunks per
full pipeline slot. SLOT_FREE fires once per slot boundary and adds the actual
number of chunks consumed in that slot (`chunks_in_slot = sub_step + 1`). For
full slots this equals `steps_per_slot`; for the final partial slot it equals
the remainder. This keeps the SLOT_FREE counter in the same step-space as
NIC_DONE and `stepState`:
- `send_tile`'s SLOT_FREE wait uses `step - pipelineDepth * steps_per_slot + 1`
  (same formula as NIC_DONE wait — no division needed)
- `drain_tile` can wait for `step` directly — after all chunks, the cumulative
  SLOT_FREE counter equals the final `step` value exactly

### 7.5 Why These Waits are Placed Where They Are

| Step | Why needed |
|------|-----------|
| `send_tile (1)` NIC_DONE | CPU proxy may still be posting from this staging slot. Memcpying new data would corrupt the in-flight RDMA source. |
| `send_tile (3)` SLOT_FREE | Receiver may still be reading remote staging. New RDMA Write would corrupt receiver's in-progress memcpy. |
| `send_tile (4)` FIFO enqueue | Must happen after `__threadfence_system()` so NIC reads coherent data. |
| `recv_tile (1)` DATA_READY | Cannot consume staging until RDMA Write has fully landed. |
| `recv_tile (1)` `__threadfence_system()` after wait | Ensures RDMA-written data is visible to SM loads. Correctness relies on the combination of: (1) IB ordering guarantee — data Write is committed to responder memory before the subsequent DATA_READY atomic is visible, (2) the GPU observes DATA_READY via a system-scope load from host-mapped memory, and (3) `__threadfence_system()` establishes a happens-before relationship so subsequent device-memory loads see the NIC-written data. This is the same pattern NCCL uses, validated on V100/A100/H100/GB200. |
| `recv_tile (3)` SLOT_FREE | Sender's NIC_DONE + SLOT_FREE waits rely on this signal. |
| `drain_tile` SLOT_FREE | Confirms receiver consumed all data before sender modifies source. |

### 7.6 Cross-Rank Coordination Contract

These invariants are **critical for correctness** — mismatched values silently
cause data corruption or deadlock:

1. **Sender and receiver must agree on `active_blocks`.** The staging layout
   uses `perBlockSlotSize = (dataBufferSize / active_blocks) & ~15ULL`. Mismatched
   `active_blocks` causes staging offset divergence — blocks on opposite sides
   read/write different memory regions, corrupting data.

2. **Sender and receiver must agree on `max_signal_bytes`.** This determines
   `steps_per_slot`, which controls signaling granularity. Mismatched values
   cause the SLOT_FREE counter to drift out of sync — the sender waits for a
   SLOT_FREE value the receiver never produces (deadlock) or proceeds before the
   receiver finishes (corruption).

3. **Both sides must use the same `dataBufferSize` and `pipelineDepth`.**
   These are transport construction parameters and must match for staging offset
   calculations and pipeline wrap-around to be consistent.

4. **If `active_blocks` or `max_signal_bytes` changes between consecutive calls,
   `drain_tile` must be called first (and a cross-rank barrier for
   `active_blocks`).** Changing `active_blocks` alters `perBlockSlotSize` and
   the slot row layout. Changing `max_signal_bytes` alters `steps_per_slot`,
   which changes the NIC_DONE wait formula's pipeline depth — without a drain,
   the new formula may not correctly protect staging buffers from the previous
   call's in-flight RDMA writes. Both sides must use matching values.

5. **Both sides must call `drain_tile` (sender) / complete `recv_tile`
   (receiver) before destroying the transport.** Outstanding FIFO entries and
   in-flight RDMA operations must complete before resources are freed.

**Runtime cross-rank validation (always-on):** Construction-time config
validation (Section 9.2 step 1) catches `dataBufferSize`/`pipelineDepth`
mismatches, but `active_blocks` and `max_signal_bytes` are per-call runtime
parameters that can still silently diverge. Since mismatches cause silent
data corruption (not just deadlock), this check must be always-on, not
debug-only.

**Mechanism: separate param validation slot in the signal pad.** The signal
pad is extended by one slot per peer at index `2 * maxBlocks` (the "param
slot"). **Every block's** first data FIFO entry (`s == 0`) carries
`paramWord = (chunk_size << 32) | active_blocks`; non-first entries set
`paramWord = 0`. The sender's CPU proxy detects `paramWord != 0` and posts
a param-slot RDMA Write (8 bytes, deduplicated per-iteration) on the
**signal QP** before posting any DATA_READY atomic.

**Ordering argument:** Every block's `s=0` FIFO entry carries `paramWord`.
When the proxy processes any block's `s=0` entry in Phase 1, it posts the
param write (if not already posted this iteration) on the signal QP. That
same block's data Write is posted on the data QP, also in Phase 1. The
data Write's CQE triggers DATA_READY posting in Phase 2 (on the signal QP).
Since Phase 1 completes before Phase 2 in each iteration, and both the
param write and DATA_READY go on the same signal QP, IB same-QP ordering
guarantees the param write is visible at the receiver before that block's
DATA_READY. This holds per-block independently: each block's param write
and DATA_READY are in the same iteration's Phase 1 and Phase 2 respectively.
Blocks whose `s=0` entries land in different proxy iterations each get
their own param write (deduplicated but harmless — same values).

The **receiver** validates on first DATA_READY for **every block** (not just
block 0), because all blocks independently observe DATA_READY and proceed:
```text
// In recv_tile, after wait_volatile_with_abort(DATA_READY, ...):
if s == 0:
    param = localSignalPad[2 * maxBlocks]  // param slot
    remote_active = param & 0xFFFFFFFF
    remote_cs = (param >> 32) & 0xFFFFFFFF
    trap if remote_active != active_blocks
    trap if remote_cs != chunk_size
```

This validates **before any block reads staging data** (the check happens
between each block's DATA_READY wait and its memcpy), so mismatched
parameters are caught before corruption occurs. The cost is one extra
8-byte RDMA Write per `send_tile` call — negligible. The param slot is
written before the first DATA_READY, so it is visible to all blocks when
they observe their respective DATA_READY signals.

Uses raw `active_blocks` (uint32_t) and `chunk_size` (uint32_t) — no
log2 encoding, no power-of-2 assumption for the validation path.

---

## 8. CPU Proxy Thread

### 8.1 Lifecycle

Three threads are managed per transport instance:

1. **Proxy thread** — drains FIFO, posts RDMA, polls CQs. Created by
   `MultipeerIbrcTransport::exchange()` after QPs are connected. Pinned to a
   CPU core near the GPU's NUMA node (via `pthread_setaffinity_np`).
2. **Watchdog thread** — monitors proxy heartbeat, sets `abortFlag` on stall.
   Created alongside the proxy thread.
3. **IB async event thread** — calls `ibv_get_async_event()` to detect QP/port/HCA
   errors that don't produce CQEs. Created by `initIbDevice()`.

All three check a shared `running_` flag. **Destroyed** by
`MultipeerIbrcTransport` destructor: sets `running_ = false`, joins all three
threads (see Section 9.4 steps 2–3b for ordering and async thread wake-up).

### 8.2 Main Loop

See Section 5 for the detailed protocol pseudocode. The proxy loop has five
phases per iteration (Phase 0: retry replay, Phase 1: FIFO drain, Phase 1.5:
forced-signaled flush, Phase 2: data CQ poll, Phase 3: signal CQ poll), with
fairness limits per peer, per-peer abort tracking, and full error checking.

**Heartbeat and watchdog:** The proxy writes a monotonic timestamp to a shared
`lastHeartbeat` location (device-mapped host memory) on each loop iteration.
A **separate watchdog thread** (launched alongside the proxy) monitors staleness.
If the heartbeat stalls beyond a configurable threshold (e.g., 10s), the
watchdog sets `*abortFlag = 1`, unblocking GPU wait loops. The watchdog must be
an independent thread — the transport destructor cannot serve this role because
teardown starts with `cudaStreamSynchronize(stream)`, which itself blocks if
GPU kernels are spinning in device waits. Without an independent watchdog, a
proxy crash (segfault, OOM kill) leaves GPU blocks and the destructor deadlocked.

**ENOMEM retry:** If `ibv_post_send` returns `ENOMEM` (SQ full — a transient
condition), the proxy does NOT set `abortFlag`. Instead, it pushes the entry
onto a per-peer retry queue and falls through to Phase 2/3 to poll CQEs and
reclaim SQ slots. Retried entries are re-posted on the next loop iteration.
Only persistent ENOMEM (e.g., N consecutive failures across K iterations)
triggers `abortFlag`.

**Per-peer abort tracking:** On any CQ error, the proxy marks `peerAborted_[peer]
= true`. In Phase 1, FIFO entries for aborted peers are consumed (ready flag
cleared to unblock the FIFO) but not posted. This prevents posting to errored
QPs and produces cleaner diagnostics.

### 8.3 RDMA Write Posting

```cpp
// Returns true if posted successfully, false on ENOMEM (entry in retryQueue_).
bool IbrcProxyThread::postRdmaWrite(int peer, const IbrcProxyFifoEntry& entry) {
  ibv_sge sge{
      .addr = entry.localAddr,
      .length = entry.nbytes,
      .lkey = entry.localLkey,
  };
  // Decide signaling BEFORE post, but only commit the counter AFTER success.
  // This ensures the counter tracks actual posts, not attempts — an ENOMEM
  // on the would-be signaled WQE must not reset the counter, or the SQ could
  // go more than signalEveryN_ posts without a real CQE.
  bool shouldSignal = (unsignaledCount_[peer] + 1 >= signalEveryN_);

  ibv_send_wr wr{
      .wr_id = encodeWrId(peer, entry.seq),
      .sg_list = &sge,
      .num_sge = 1,
      .opcode = IBV_WR_RDMA_WRITE,
      .send_flags = shouldSignal ? IBV_SEND_SIGNALED : 0u,
      .wr.rdma = {
          .remote_addr = entry.remoteAddr,
          .rkey = entry.remoteRkey,
      },
  };
  ibv_send_wr* bad_wr;
  // Post via IbvVirtualQp — handles multi-QP striping transparently.
  // For single-QP config (N=1), this is a direct ibv_post_send passthrough.
  auto result = dataVirtualQps_[peer]->postSend(wr);
  if (!result.hasValue()) {
    if (result.error().errNum == ENOMEM) {
      // SQ full — transient. Counter NOT updated (post didn't happen).
      retryQueue_[peer].push(entry);
      return false;
    }
    *abortFlag_ = 1;
    LOG(ERROR) << "ibv_post_send failed: " << result.error().errStr;
    return false;
  }
  // Post succeeded — now commit the counter.
  if (shouldSignal) unsignaledCount_[peer] = 0;
  else ++unsignaledCount_[peer];
  return true;
}
```

### 8.4 Signal Atomic Posting

```cpp
// Returns true if posted successfully, false on ENOMEM (entry in signalRetryQueue_).
bool IbrcProxyThread::postAtomicFetchAdd(int peer, const IbrcProxyFifoEntry& entry) {
  // Same counter-after-success pattern as postRdmaWrite.
  bool shouldSignal = (unsignaledSignalCount_[peer] + 1 >= signalQpSignalEveryN_);

  // IB Fetch-And-Add returns the previous remote value into a local buffer.
  // We don't need the result, but the verbs API requires a valid SGE pointing
  // to a registered 8-byte buffer for the HCA to write into.
  // atomicScratch_ is a per-proxy, pre-registered 8-byte host buffer (allocated
  // once at construction, ibv_reg_mr'd with LOCAL_WRITE). The returned value
  // is discarded. See ctran/backends/ib/CtranIbVc.cc:289 for the same pattern.
  ibv_sge sge{
      .addr = reinterpret_cast<uint64_t>(atomicScratch_),
      .length = sizeof(uint64_t),
      .lkey = atomicScratchLkey_,
  };
  ibv_send_wr wr{
      .wr_id = encodeSignalWrId(peer, entry.seq),
      .sg_list = &sge,
      .num_sge = 1,
      .opcode = IBV_WR_ATOMIC_FETCH_AND_ADD,
      .send_flags = shouldSignal ? IBV_SEND_SIGNALED : 0u,
      .wr.atomic = {
          .remote_addr = entry.remoteSignalAddr +
                         entry.signalId * sizeof(uint64_t),
          .compare_add = entry.signalVal,
          .rkey = entry.remoteSignalRkey,
      },
  };
  ibv_send_wr* bad_wr;
  int ret = signalQps_[peer]->postSend(&wr, &bad_wr);
  if (ret == ENOMEM) {
    // SQ full — transient. Counter NOT updated.
    signalRetryQueue_[peer].push(entry);
    return false;
  }
  if (ret != 0) {
    *abortFlag_ = 1;
    LOG(ERROR) << "ibv_post_send (signal atomic) failed: " << strerror(ret);
    return false;
  }
  // Post succeeded — commit the counter.
  if (shouldSignal) unsignaledSignalCount_[peer] = 0;
  else ++unsignaledSignalCount_[peer];
  return true;
}
```

**`tryPostRdmaWrite` / `tryPostAtomicFetchAdd`:** Used exclusively by Phase 0
retry replay. Identical to `postRdmaWrite` / `postAtomicFetchAdd` except:
- Return the raw `ibv_post_send` result (`0`, `ENOMEM`, or other error)
  instead of `bool`.
- Do **NOT** enqueue into `retryQueue_` / `signalRetryQueue_` on ENOMEM.
- Do **NOT** set `abortFlag` on fatal errors (Phase 0 handles this).
- Still apply the same counter-after-success signaling logic.

This separation prevents the duplicate-enqueue bug: Phase 1 callers use
`postRdmaWrite` (which enqueues on ENOMEM for first-time failures), while
Phase 0 callers use `tryPostRdmaWrite` (which leaves the entry at the front
of the retry queue on ENOMEM, never re-appending it).

**`postParamWrite`:** Returns `bool` (true = success, false = failed).
Posts an 8-byte unsignaled RDMA Write on the **signal QP** carrying
`entry.paramWord` to the receiver's param validation slot
(`remoteSignalAddr + 2 * maxBlocks * sizeof(uint64_t)`). Posted on the
signal QP (not data QP) because DATA_READY atomics are also on the signal
QP — IB same-QP ordering guarantees the param write completes at the
receiver before any subsequent DATA_READY atomic on that QP. The write
uses `atomicScratch_` as the local source buffer (copy `paramWord` into it
before posting). On ENOMEM, returns `false` — the **caller** is
responsible for deferring both the param write and the associated data
Write together (Phase 1 pushes the entire FIFO entry to `retryQueue_`).
This avoids a separate `paramRetryQueue_` and guarantees the param write
and data Write are always replayed as a pair. On fatal error, sets
`peerAborted_` + `abortFlag` and returns `false`. Contributes to the
signal QP's `unsignaledSignalCount_` for selective signaling.

### 8.5 Performance Considerations

- **Batch posting:** The proxy can chain multiple WQEs into a linked list and
  post them in a single `ibv_post_send()` call (like ctran's `iputBatch`).
- **CQ batching:** Poll multiple CQEs per `ibv_poll_cq()` call (batch of 16).
- **Busy-poll vs. event-driven:** Default is busy-poll (lowest latency). Can
  switch to `ibv_get_cq_event()` + `ibv_req_notify_cq()` for lower CPU usage.
- **NUMA affinity:** Pin proxy thread to the same NUMA node as the GPU to
  minimize PCIe latency for device-mapped memory access.
- **Selective signaling:** Not every RDMA Write needs `IBV_SEND_SIGNALED`. The
  proxy can signal every Nth WQE and batch-complete the intervening ones. This
  reduces CQ pressure. Both data QP and signal QP use selective signaling
  with periodic CQ polling (required by IB spec for SQ slot reclamation).
- **Single-thread scaling concern:** With 128 blocks × 8 peers = 1024 entries
  per pipeline step, each requiring ~300-500ns for `ibv_post_send`, the proxy
  spends 360-500µs just posting WQEs. For pipeline steps targeting ~1ms, this
  is 35-50% of the budget. Mitigations: (a) WQE batch posting via linked WR
  lists in a single `ibv_post_send` call, (b) peer-sharding across 2-4 proxy
  threads for high block counts, (c) adaptive peer scheduling to skip empty
  FIFOs. See Open Question 8.
- **Warp-leader signal pad polling:** Signal pad and NIC_DONE counters reside
  in host-pinned memory, so each GPU load is a ~1-2µs PCIe round-trip. With
  128 blocks polling concurrently (128+ warps), this generates significant
  PCIe read traffic. Optimization: have one thread per warp (lane 0) perform
  the host-memory load, then broadcast the result to all 31 other lanes via
  `__shfl_sync(0xFFFFFFFF, value, 0)`. This cuts PCIe read pressure by 32×
  while maintaining the same polling responsiveness. The `wait_volatile_with_
  abort` implementation should use this pattern internally.
- **Minimum effective chunk size:** IBRC's proxy overhead of ~5-15µs per
  operation dominates for small messages. The transport is practical for
  per-block chunk sizes >= 32-64KB where data transfer time exceeds control
  plane overhead. Below this threshold, IBGDA's ~1-2µs GPU-initiated path is
  preferable.

---

## 9. Host-Side Transport (`MultipeerIbrcTransport`)

**Implementation requirement:** All IB resource management must use `ibverbx`
RAII wrappers (`comms/ctran/ibverbx/`) — not raw `libibverbs` — as mandated by
pipes CLAUDE.md. This provides RAII cleanup (eliminating most of the manual
ordering in Section 9.4), move-only semantics, and `folly::Expected` error
handling. Use `IbvQpUtils::createRcQp()`, `initQp()`, `rtrQp()`, `rtsQp()` for
QP state machine transitions.

### QP and CQ Topology (per transport instance, 2 peers shown)

```
  MultipeerIbrcTransport (Rank A)
  ═══════════════════════════════════════════════════════════════════

  ibverbx::IbvDevice → ibverbx::IbvPd
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  Peer 0                              Peer 1                  │
  │  ┌──────────────────┐ ┌────────┐    ┌──────────────────┐ ┌────────┐
  │  │ IbvVirtualQp     │ │IbvQp   │    │ IbvVirtualQp     │ │IbvQp   │
  │  │ (data, N phys)   │ │(signal)│    │ (data, N phys)   │ │(signal)│
  │  │ ┌────┐ ┌────┐   │ │Atomic  │    │ ┌────┐ ┌────┐   │ │Atomic  │
  │  │ │QP 0│ │QP 1│...│ │+param  │    │ │QP 0│ │QP 1│...│ │+param  │
  │  │ │Wrt │ │Wrt │   │ │Write   │    │ │Wrt │ │Wrt │   │ │Write   │
  │  │ └──┬─┘ └──┬─┘   │ └───┬────┘    │ └──┬─┘ └──┬─┘   │ └───┬────┘
  │  └────┼──────┼──────┘     │         └────┼──────┼──────┘     │
  │       │      │            │              │      │            │
  │       ▼      ▼            ▼              ▼      ▼            ▼
  │  ┌───────────────────────────┐     ┌──────────────────────────┐
  │  │  IbvVirtualCq (dataCq_)  │     │     IbvCq (signalCq_)   │
  │  │  aggregates across all    │     │  shared across all       │
  │  │  peers' physical data QPs │     │  peers' signal QPs       │
  │  │  + fragment reassembly    │     │                          │
  │  └───────────────────────────┘     └──────────────────────────┘
  │                                                              │
  │  atomicScratch_ (8B, ibv_reg_mr LOCAL_WRITE)                │
  │  — shared result buffer for all Fetch-And-Add WQEs          │
  └──────────────────────────────────────────────────────────────┘

  Data VirtualQP: RDMA Writes (striped across N physical QPs)
                  + zero-byte flush WQEs (IBV_SEND_SIGNALED)
  Signal QP:      DATA_READY atomics, SLOT_FREE atomics,
                  param validation writes

  wr_id encoding: [63:62]=type  [61:48]=peer  [47:0]=seq
    00=data(dataQP)  01=signal(signalQP)  10=flush(dataQP)
```

### 9.1 Construction Flow

1. **`initIbDevice()`** — Open IB device (selected by GPU-to-NIC PCIe affinity,
   matching the GPU's NUMA node) via `ibverbx::IbvDevice`, allocate
   `ibverbx::IbvPd`. Create an `ibverbx::IbvVirtualCq` for data completions
   (aggregates across all peers' physical data QPs, handles fragment
   reassembly for multi-QP virtual QPs) and a regular `ibverbx::IbvCq` for
   signal completions.
   Start an **IB async event thread** (`ibv_get_async_event()` loop) to detect
   QP errors, port state changes, and HCA catastrophic errors that don't produce
   CQEs. On async error, set `*abortFlag = 1`. This matches NCCL's
   `ncclIbAsyncThreadMain` pattern.
2. **`createQps()`** — Create QPs per peer using `ibverbx` RAII wrappers
   (as required by pipes CLAUDE.md — do NOT use raw `libibverbs`).
   - **Data path:** Use `ibverbx::IbvVirtualQp` per peer, backed by N physical
     `IbvQp` objects (configurable, default 1). The virtual QP provides:
     - Automatic message fragmentation across physical QPs
     - SPRAY or DQPLB load balancing for multi-path bandwidth
     - Completion reassembly via `IbvVirtualCq`
     - Zero-overhead passthrough when N=1
     This resolves Open Question 1 (multi-QP striping) — the ibverbx layer
     handles it transparently. Physical QPs use `sq_sig_all=0`,
     `max_inline_data=0`, `qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
     IBV_ACCESS_REMOTE_WRITE`.
   - **Signal QP:** A single `ibverbx::IbvQp` per peer (not virtualized —
     atomics and param writes are 8-byte operations that don't benefit from
     striping). `sq_sig_all=0`, `max_inline_data=16` (required by Broadcom
     HCAs for atomic operations), `qp_access_flags = IBV_ACCESS_LOCAL_WRITE
     | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC`.
   - **RTR transition:** `min_rnr_timer=12` (0.01ms, matching NCCL/ctran),
     `max_dest_rd_atomic=1` (required for RDMA Atomic support).
   - **RTS transition:** `max_rd_atomic=1`.
   Use `IbvQpUtils::createRcQp()`, `initQp()`, `rtrQp()`, `rtsQp()` from
   `comms/ctran/ibverbx/`. Exchange QP metadata via
   `IbvVirtualQpBusinessCard` for multi-QP connection setup.
3. **`probeGdr()`** — Detect GDR support (DMA-BUF or nvidia_peermem). If
   `gdrMode=AUTO`, try DMA-BUF first, then nvidia_peermem. If neither is
   available, set `useHostBounce=true`.
4. **`allocateTileBuffers()`** — Bulk allocations:
   - sendStaging: GDR path: `cudaMalloc` + `ibv_reg_mr`. Host bounce path:
     `cudaHostAlloc(cudaHostAllocMapped)` + `ibv_reg_mr`.
   - recvStaging: Same as sendStaging. Exchanged with peers.
   - signalPad: `cudaHostAlloc(cudaHostAllocMapped)`, `ibv_reg_mr`, exchanged
   - nicDoneCounter: `cudaHostAlloc(cudaHostAllocMapped)` (no RDMA reg)
   - stepState: `cudaMalloc`, zero-initialized (GPU-only, no RDMA)
   - proxyFifo: `cudaHostAlloc(cudaHostAllocMapped)`, zero-initialized
   - fifoTail: `cudaMalloc` (GPU device memory, for fast atomicAdd)
   - abortFlag: `cudaHostAlloc(cudaHostAllocMapped)`, zero-initialized
   - atomicScratch: 8-byte host buffer, `ibv_reg_mr` with `LOCAL_WRITE`.
     Used as the result buffer for `IBV_WR_ATOMIC_FETCH_AND_ADD` WQEs
     (verbs requires a local SGE even though we discard the fetched value).
     One buffer per proxy thread (not per peer — atomics are serialized
     through the single-threaded proxy loop)

### 9.2 `exchange()`

1. **Cross-rank config validation** — compute a hash of the **construction-time**
   transport config parameters that must match across ranks (`dataBufferSize`,
   `pipelineDepth`, `tileMaxBlocks`, `minSignalBytes`) and `allGather` it. If
   any peer's hash differs, throw with a diagnostic listing the mismatched
   values. This catches mismatched buffer sizing and pipeline depth at setup
   time. **Note:** this does NOT validate `active_blocks` or `max_signal_bytes`,
   which are per-call runtime parameters. Those are validated always-on at
   runtime via the param validation slot (see Section 7.6).
2. Bootstrap QP metadata exchange (QPN, GID/LID, port) — uses `allGather`.
3. Connect QPs: INIT → RTR → RTS (same as ctran/NCCL).
4. Exchange recvStaging buffer info (address + rkey) — uses `allGather`.
5. Exchange signalPad buffer info (address + rkey) — uses `allGather`.
6. Compute per-peer remote offsets using `remotePeerIndex` formula.
7. **Launch proxy thread.**

### 9.3 Buffer Memory Decisions

| Buffer | GDR Path | Host Bounce Path | RDMA Reg | Exchange |
|--------|---------|------------------|----------|----------|
| sendStaging | `cudaMalloc` (GPU HBM) | `cudaHostAlloc(Mapped)` (pinned host) | `ibv_reg_mr` (lkey) | No |
| recvStaging | `cudaMalloc` (GPU HBM) | `cudaHostAlloc(Mapped)` (pinned host) | `ibv_reg_mr` (rkey) | Yes |
| signalPad | `cudaHostAlloc(Mapped)` | same | `ibv_reg_mr` (rkey, `IBV_ACCESS_REMOTE_ATOMIC \| LOCAL_WRITE \| REMOTE_WRITE`) | Yes |
| nicDoneCounter | `cudaHostAlloc(Mapped)` | same | None (CPU→GPU only) | No |
| stepState | `cudaMalloc` (GPU HBM) | same | None (GPU-only) | No |
| proxyFifo | `cudaHostAlloc(Mapped)` | same | None (GPU→CPU shared) | No |
| fifoTail | `cudaMalloc` (GPU) | same | None (GPU atomicAdd) | No |
| abortFlag | `cudaHostAlloc(Mapped)` | same | None (CPU→GPU flag) | No |

**GDR vs. host bounce buffer path:**
With GDR (nvidia_peermem or DMA-BUF), staging buffers are in GPU HBM. The NIC
reads/writes GPU memory directly over PCIe, achieving maximum bandwidth. Without
GDR, staging uses host pinned memory. The GPU does cooperative memcpy to/from
host pinned staging, and the NIC reads/writes host memory. This adds one extra
GPU↔host memcpy per chunk but requires no GDR kernel module — fulfilling IBRC's
portability goal. The `gdrMode` config controls the behavior.

**Why signalPad is in host pinned memory (not GPU HBM):**
The remote NIC writes RDMA atomics to the signalPad. Host pinned memory is
directly addressable by both the NIC (for RDMA) and the GPU (via
`cudaHostGetDevicePointer`). GPU polling of host memory is slower than GPU
memory (~1-2µs per PCIe read vs ~100ns for HBM), but pipelining hides this
latency since the GPU polls while the cooperative memcpy for the next chunk
runs concurrently. GPU loads from `cudaHostAllocMapped` memory are system-scope
transactions that traverse PCIe to host DRAM, bypassing GPU L1 cache. The
`volatile` qualifier prevents register caching, ensuring each poll re-reads from
host memory and observes NIC-written RDMA atomic values. Signal pad fields must
be 8-byte aligned (`alignas(8)` or `static_assert`) to satisfy RDMA atomic
alignment requirements.

**Why nicDoneCounter does NOT need RDMA registration:**
Unlike IBGDA where the companion QP writes the counter via RDMA loopback, in
IBRC the local CPU proxy writes the counter directly via a volatile store +
`wc_store_fence()` (`_mm_sfence()` on x86, `__dmb(ST)` on ARM) to flush WC
buffers. See Section 5 for details. No RDMA involved.

**NUMA-aware allocation for host-pinned buffers:**
`cudaHostAlloc` does NOT guarantee NUMA placement — it allocates on whatever
NUMA node the calling thread runs on, which may not be the GPU's NUMA node.
Cross-NUMA access adds ~100-200ns per read/write (UPI/QPI traversal), which
compounds to ~150µs of additional latency per step when the proxy drains 1024
FIFO entries. For performance-critical host-pinned buffers (proxyFifo,
signalPad, nicDoneCounter, abortFlag), prefer NUMA-local allocation:

```cpp
// Instead of: cudaHostAlloc(&fifo, size, cudaHostAllocMapped);
// Do:
int gpuNumaNode = cycleclub::getGpuNumaNode(cudaDevice);
void* fifo = numa_alloc_onnode(size, gpuNumaNode);
cudaHostRegister(fifo, size, cudaHostRegisterMapped);
```

Verify placement with `numactl --hardware` and `nvidia-smi topo -m`. The proxy
thread, FIFO memory, and NIC should all be on the same NUMA node.

### 9.4 Cleanup and Destruction Order

Resource cleanup must follow a strict order to avoid use-after-free and hangs:

```text
 1. cudaStreamSynchronize(stream) — ensure no GPU threads are mid-write to FIFO.
                                    Use the specific stream tile kernels were
                                    launched on, NOT cudaDeviceSynchronize()
                                    (which would block on unrelated GPU work).
                                    The transport must track the kernel stream
                                    or require the caller to pass it at teardown.
 2. Set running_ = false          — signal proxy thread, watchdog, and async
                                    event thread to exit
 3. proxy_thread_.join()          — wait for proxy loop to finish
 3a. watchdog_thread_.join()      — wait for watchdog to finish (checks
                                    running_ flag on each heartbeat poll)
 3b. ibv_cmd_async_fd + write to  — wake async event thread from blocking
     pipe / eventfd                 ibv_get_async_event() call, then join.
                                    (Some implementations use a self-pipe or
                                    eventfd alongside the async FD in epoll
                                    to enable clean shutdown.)
 4. ibv_modify_qp(IBV_QPS_ERR)   — transition ALL QPs (data + signal) to Error
                                    state. This causes the HCA to flush all
                                    outstanding WQEs and generate error CQEs.
 5. Drain CQs                    — poll all remaining CQEs (data + signal CQs),
                                    including flush CQEs from the Error transition
 6. ibv_destroy_qp()             — destroy all data + signal QPs per peer
 7. ibv_destroy_cq()             — destroy data CQ and signal CQ (must be after
                                    QP destruction, since QPs reference CQs)
 8. ibv_dereg_mr()               — deregister all RDMA-registered MRs
                                    (sendStaging, recvStaging, signalPad)
 9. cudaFree / cudaFreeHost      — free underlying memory
                                    (DeviceBuffer RAII handles this)
10. ibv_dealloc_pd()             — free protection domain
11. ibv_close_device()           — close IB device context
```

**Why this order matters:**
- Step 1 before step 2: GPU threads may still be writing to the FIFO. Stopping
  the proxy while the GPU is mid-enqueue leaves orphaned entries. Must use
  `cudaStreamSynchronize` (not `cudaDeviceSynchronize`) to avoid blocking on
  unrelated GPU work on other streams.
- Step 3 before step 4: The proxy thread may be mid-`ibv_post_send`. Joining
  ensures all posts are complete before QP state transition.
- Step 4 before step 5: Moving QPs to Error state flushes all outstanding WQEs
  and generates error CQEs. Without this, draining the CQ while QPs are in RTS
  may miss in-flight WQEs that generate CQEs after the drain.
- Step 5 before step 6: Outstanding WQEs (now flushed as errors) must be drained
  before destroying QPs. `ibv_destroy_qp` with undrained CQEs may return `EBUSY`.
- Step 6 before step 7: QPs must be destroyed before their associated CQs.
- Step 8 before step 9: `ibv_dereg_mr` must complete before the underlying
  memory is freed. Freeing memory with active MRs is undefined behavior.

---

## 10. Transport Type Integration

### 10.1 `TransportType` Enum

```cpp
enum class TransportType : uint8_t {
  SELF,
  P2P_NVL,
  P2P_IBGDA,
  P2P_IBRC,      // NEW
};
```

### 10.2 `Transport` Union

```cpp
struct Transport {
  TransportType type;
  union {
    P2pSelfTransportDevice self;
    P2pNvlTransportDevice p2p_nvl;
    P2pIbgdaTransportDevice* p2p_ibgda;
    P2pIbrcTransportDevice* p2p_ibrc;   // NEW — pointer (same reason as IBGDA)
  };
};
```

### 10.3 `MultiPeerDeviceHandle`

Add `get_ibrc(rank)` accessor:

```cpp
__device__ P2pIbrcTransportDevice& get_ibrc(int rank) {
  return *transports[rank].p2p_ibrc;
}
```

---

## 11. Comparison of All Three Tile Transports

| Aspect | NVL | IBGDA | IBRC |
|--------|-----|-------|------|
| Data path | NVLink memcpy to remote staging | GPU RDMA Write via DOCA QP | CPU `ibv_post_send()` RDMA Write |
| NIC wait | N/A | Companion QP loopback counter | CPU proxy volatile write |
| DATA_READY signal | NVLink `SignalState::signal(SET)` | GPU RDMA atomic via DOCA | CPU `ibv_post_send()` RDMA atomic |
| SLOT_FREE signal | NVLink `SignalState::signal(SET)` | GPU `signal_remote()` RDMA atomic | CPU `ibv_post_send()` RDMA atomic |
| Drain | None (NVLink coherent) | `wait_signal(nicDone, step)` | `wait_volatile(SLOT_FREE, step)` |
| sendStaging | Not used (direct remote write) | GPU HBM (RDMA-registered) | GPU HBM (RDMA-registered) |
| CPU involvement | None | None | Proxy thread (FIFO drain + CQ poll) |
| GPU→NIC latency | ~0 (NVLink MMIO) | ~1-2µs (GPU doorbell) | ~5-10µs (GPU→FIFO→CPU→NIC) |
| HW requirements | NVLink/NVSwitch | DOCA GPUNetIO, ConnectX-7+ | Any IB/RoCE HCA with RC |

---

## 12. Worked Example

Bidirectional send/recv kernel using `partition(2)`:

```cpp
// Host-side launch — use transport.launchKernel() to enforce single-stream:
//   transport.launchKernel(stream, bidirectional_ibrc_tile_kernel,
//       numBlocks, threadsPerBlock, sharedMem, stream,
//       transport.getDeviceTransport(), src, dst, totalBytes, numBlocks, timeout);
// Direct <<<...>>> launches bypass stream enforcement and are unsupported.

__global__ void bidirectional_ibrc_tile_kernel(
    P2pIbrcTransportDevice* transport,
    char* src, char* dst,
    std::size_t totalBytes,
    int numBlocks,
    Timeout timeout) {
  auto group = make_block_group();
  auto [role, sub] = group.partition(2);
  const bool is_sender = (role == 0);

  // Timeout requires explicit start() before any wait loop checks it.
  // See Timeout.cuh:20 — checkExpired() returns false until start() is called.
  timeout.start();

  if (totalBytes == 0) {
    if (is_sender) transport->drain_tile(sub, timeout);
    return;
  }
  const std::size_t sectionBytes =
      min(transport->tile_state().dataBufferSize, totalBytes);
  const std::size_t totalSections =
      (totalBytes + sectionBytes - 1) / sectionBytes;  // ceiling division

  for (std::size_t s = 0; s < totalSections; ++s) {
    const std::size_t offset = s * sectionBytes;
    const std::size_t thisSection = min(sectionBytes, totalBytes - offset);
    TiledBuffer<char> tiles(
        is_sender ? src + offset : dst + offset, thisSection, sub);

    if (is_sender) {
      transport->send_tile(sub, tiles.data(), tiles.bytes(), numBlocks, 0, timeout);
      // drain_tile MUST be called after each send_tile before the next one
      // (Section 7.2: consecutive send_tile to the same peer without drain
      // traps on the bit-63 guard).
      transport->drain_tile(sub, timeout);
    } else {
      transport->recv_tile(sub, tiles.data(), tiles.bytes(), numBlocks, 0, timeout);
    }
  }

}
```

---

## 13. Implementation Phases

| Phase | Files | Complexity | Depends on |
|-------|-------|-----------|-----------|
| 1. Transport type + device struct | `Transport.cuh`, `MultiPeerDeviceHandle.cuh`, new `P2pIbrcTransportDevice.cuh`, `IbrcTileState` in `IbgdaBuffer.h` | Low | None |
| 2. CPU proxy thread + FIFO | New `IbrcProxy.h/.cc`, `IbrcProxyFifo.h` | High | Phase 1 |
| 3. Host transport (QPs, buffers, exchange) | New `MultipeerIbrcTransport.h/.cc` | High | Phase 2 |
| 4. Device-side `send_tile`/`recv_tile`/`drain_tile` | `P2pIbrcTransportDevice.cuh` | Medium | Phase 1 |
| 5. Triton dispatch | `device_transport.cu/.h` (add IBRC branch) | Low | Phase 4 + D101445960 |
| 6. C++ tests + benchmark | `IbrcTileSendRecvBenchmark.cc` | Medium | Phase 3-4 |
| 7. Triton E2E tests | `test_e2e.py` extensions | Low | Phase 5 |

---

## 14. Resolved Design Decisions

These were originally open questions, now resolved based on expert review:

1. **GDR vs. bounce buffer for staging:** RESOLVED — support both via
   `gdrMode={AUTO, REQUIRE, DISABLE}` config. AUTO tries DMA-BUF, then
   nvidia_peermem, then falls back to host bounce buffers. This fulfills the
   portability goal (Section 3.1, 9.1, 9.3).

2. **Error handling and abort:** RESOLVED — CPU proxy sets `abortFlag` in
   device-mapped memory on any CQ error. GPU wait loops check it and trap with
   a diagnostic. Proxy checks `wc.status` on every CQ completion (Section 5,
   8.2, 8.3).

3. **SQ exhaustion:** RESOLVED — separate data QP and signal QP per peer
   prevent data WQEs and signal atomics from competing for the same SQ depth.
   Both QPs use selective signaling (every Nth WQE is `IBV_SEND_SIGNALED`) with
   periodic CQ polling to reclaim SQ slots. The IB spec requires CQ polling for
   SQ slot reclamation — unsignaled WQE slots are NOT freed by the HCA on wire
   ACK alone (Section 5, 8.3, 9.1).

4. **Multi-producer FIFO race:** RESOLVED — per-entry `ready` flag with
   `__threadfence_system()` before setting it. CPU proxy checks `ready` before
   consuming entries, never reads based solely on `fifoTail` (Section 4.2, 5).

5. **fifoTail PCIe atomic bottleneck:** RESOLVED — `fifoTail` is in GPU device
   memory for fast `atomicAdd` (~10ns). CPU proxy scans by `ready` flags, does
   not need to read `fifoTail` (Section 5).

6. **TransportType dispatch site checklist:** RESOLVED — adding `P2P_IBRC`
   requires explicit handling in these files (use if-else/switch for all four
   types, no brittle `else` catch-alls):
   - `Transport.cuh`: move constructor, move assignment, destructor, `transport_type_name()`
   - `MultiPeerDeviceHandle.cuh`: forward declaration + `get_ibrc()` accessor
   - `MultiPeerTransport.cc`: `build_device_handle()` case, type assignment logic
   - `device_transport.cu`: transport-type dispatch in `send_tile`/`recv_tile`/`drain_tile` externs
   - `transport.py`: `TRANSPORT_P2P_IBRC = 3` constant
   - `device_transport.h`: document `3=P2P_IBRC` in comment

---

## 15. Open Questions

1. **Multi-QP striping:** RESOLVED — use `ibverbx::IbvVirtualQp` for the data
   path. The virtual QP wraps N physical QPs with automatic message
   fragmentation and SPRAY/DQPLB load balancing. Default N=1 (zero-overhead
   passthrough); increase N for higher per-peer bandwidth. Completions are
   reassembled by `IbvVirtualCq`. Connection metadata exchanged via
   `IbvVirtualQpBusinessCard`. No custom striping code needed — ibverbx
   handles it transparently (Section 9.1).

2. **CUDA graph compatibility:** The CPU proxy polls FIFO entries independently.
   CUDA graph replay re-executes GPU `atomicAdd` and descriptor writes on live
   device-mapped memory, which is correct (not captured snapshots). The proxy
   must be running before graph replay starts. Verify with CUDA graph capture
   and replay benchmarks.

3. **Proxy thread count:** One proxy thread per `MultipeerIbrcTransport`
   instance. With 8 GPUs × multiple communicators, this could mean many threads.
   Consider sharing a proxy thread pool across communicators (similar to ctran's
   `CtranIbSingleton`). For v1, one thread per instance is acceptable.

4. **RDMA Write + Atomic chaining:** ~~An optimization would be to chain the
   Write and Atomic as back-to-back WQEs with `IBV_SEND_FENCE`.~~ **INVALID:**
   `IBV_SEND_FENCE` only orders WQEs within the **same QP**. Since data Writes
   and signal Atomics are on separate QPs, the fence has no effect across them.
   The current approach (CQ completion of Write on data QP, then post Atomic on
   signal QP) is correct. To use fencing, both operations would need to be on
   the same QP, losing the separate QP architecture's benefits (Section 5).

5. **Transport selection policy:** When should a peer be assigned `P2P_IBRC` vs
   `P2P_IBGDA`? Options: (a) config-driven (user explicitly chooses), (b)
   capability-driven (fallback when DOCA is unavailable), (c) per-communicator.
   Needs team input.

6. **DeviceWindow support:** Should IBRC peers be usable through the
   `DeviceWindow` API (signal/wait_signal/barrier/put), or is IBRC exclusively a
   tile-API transport? This significantly affects the scope of dispatch-site
   updates in `DeviceWindow.cuh`. Recommendation: tile-only for v1.

7. **Register pressure in 3-way dispatch (implementation requirement):** The
   Triton extern `torchcomms_transport_send_tile` must NOT inline all three
   transport code paths. The compiler unions register footprints of all inlined
   paths, drastically reducing occupancy. Each transport's `send_tile` /
   `recv_tile` / `drain_tile` must be called through per-transport
   `__noinline__` helpers (e.g., `ibrc_send_tile_impl`, `nvl_send_tile_impl`)
   to isolate register allocation. Monitor with `--ptxas-options=-v`.

8. **Proxy thread scaling:** With 128 blocks × 8 peers, a single proxy thread
   spends 360-500µs per step just on `ibv_post_send` calls. Options: (a) WQE
   batch posting via linked WR lists (reduces per-call overhead), (b) multi-
   thread proxy with peer-sharding (e.g., 2 threads × 4 peers), (c) shared
   proxy thread pool across communicators (like ctran's `CtranIbSingleton`).
   Recommendation: start with WQE batch posting in v1; add thread sharding
   if benchmarks show the proxy is the bottleneck.

9. **Bit 63 drain guard and `__trap()` behavior:** All protocol violations
   (forgotten drain, parameter mismatch, abort flag) result in `__trap()`, which
   is **process-fatal** in CUDA/Triton — it kills the kernel, invalidates the
   CUDA context, and subsequent CUDA API calls return `cudaErrorAssert`. There is
   no in-process transport recovery path after `__trap()`. The intended behavior
   is **fail-fast with diagnostic**: the `printf` before each `__trap()` provides
   root-cause information, and the process must be restarted. This matches NCCL's
   approach where `__trap()` is a terminal error. Do not design for in-process
   recovery from device-side traps.

10. **Multi-stream enforcement (part of the transport contract):** The
    transport uses a single shared FIFO per peer and non-atomic bit 63
    stepState read-modify-write in `send_tile`/`drain_tile`. Two streams
    calling `send_tile` concurrently on the same transport will race on
    stepState, corrupting the pending flag and step counter.

    **Enforcement:** `MultipeerIbrcTransport` exposes a `launchKernel()`
    method (or a `getStream()` + `CHECK` pattern) as the **only supported
    kernel launch path**. This method:
    - On first call: records `boundStream_` via `compare_exchange`.
    - On subsequent calls: asserts `stream == boundStream_`, throws if not.
    - Passes a `streamToken` (monotonic ID derived from the stream pointer)
      into `IbrcTileState` so device code can optionally cross-validate.

    Direct `<<<...>>>` kernel launches bypassing this method are
    **unsupported** — the transport cannot enforce single-stream safety if
    the caller controls the launch site. The worked example (Section 12)
    shows the correct host-side launch pattern.
