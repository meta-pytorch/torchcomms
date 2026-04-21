# Tile Send/Recv Design

A **tile** is the smallest unit of work that all threads in a block process concurrently at one time. A user is free to choose how big or small a tile is. Smaller tiles allow more pipelining but incur more signaling overhead; larger tiles amortize signaling costs. Data is divided amongst blocks by creating tiles of data — each block may handle multiple tiles sequentially, and different blocks may handle different tiles in parallel.

A unified `send_tile` / `recv_tile` API for pipelined point-to-point transfers
on `P2pNvlTransportDevice` and `P2pIbgdaTransportDevice`, callable from CUDA
and Triton kernels. Composable building blocks for collectives (allgather,
alltoall, sendrecv) without users needing to manage staging, signals, slot
rotation, or pipeline depth.

**Target backends (all 4 share this contract):** NVLink cpp, NVLink Triton, IB
cpp, IB Triton.

**In scope:** per-block tile send/recv with cooperative memcpy + pipelined
staging.
**Out of scope:** explicit user-visible drain (handled internally), multi-stream
concurrency on the same transport, buffer registration (transport-owned), and
cross-rank rendezvous (separate barrier primitive).

---

## 1. Transport Setup

The tile API reuses the existing per-peer buffer settings already present in
each transport's config. No new `TileConfig` sub-struct is introduced — the
three knobs the tile algorithm needs are drawn from fields that already exist
(or are added minimally where missing).

### NVL (`MultiPeerNvlTransportConfig`)

The existing config already carries all three knobs:

| Existing field | Role in tile API |
|---|---|
| `data_buffer_size` | Bytes per pipeline slot, per peer, per direction. Tile staging is allocated from this. |
| `pipeline_depth` | Number of slots in the pipeline ring. |
| `tile_max_blocks` (renamed → `tile_max_groups`) | Upper bound on the number of groups that may call `send_tile`/`recv_tile`. Sizes signal pad and step state arrays. |

No new fields are needed on the NVL side.

### IB (`MultipeerIbgdaTransportConfig`)

The existing config has `data_buffer_size` but lacks explicit pipeline depth and
tile group count. Two fields are added:

```cpp
struct MultipeerIbgdaTransportConfig {
  // ... existing fields (data_buffer_size, qpDepth, etc.) ...

  // Number of pipeline slots for tile send/recv staging.
  // Default 2 (double-buffered). Must be >= 1.
  std::size_t tile_pipeline_depth{2};

  // Upper bound on the number of groups calling send_tile/recv_tile.
  // Sizes signal pad and step state arrays. Must be >= 1.
  int tile_max_groups{128};
};
```

### Validation (throws at construction, both transports)

- `pipeline_depth` (NVL) / `tile_pipeline_depth` (IB) `>= 1`
- `tile_max_groups >= 1`
- `(data_buffer_size / tile_max_groups) >= 16` — per-group slot must fit at least
  one 16-byte vectorized memcpy.

**Defaults rationale:** matches NVL benchmarks and Triton (H100) practice. With
`tile_max_groups=128`, `per_block_slot_size = data_buffer_size / tile_max_groups = 64 KiB`
— a non-trivial chunk on which RDMA puts amortize NIC setup cost.

---

## 2. Internal State

Owned by the transport, allocated and registered at construction. **Invisible
to users** — referenced here only for implementer reference.

The NVLink and IB transports use separate tile state structs because the IB
transport requires additional fields for NIC completion tracking and local
send staging that NVLink does not need.

### `NvlinkTransportTileState`

```cpp
struct NvlinkTransportTileState {
  // Per-block step counters. Persistent across send_tile/recv_tile calls.
  // Required for monotonic signal values and slot-rotation continuity.
  DeviceSpan<int64_t> step_state;   // [2 * max_groups]
                                    //   [0..max_groups)         = sender step per block
                                    //   [max_groups..2*max_groups) = receiver step per block

  int tile_max_groups{0};

  // Signal pad (using SignalState). Receiver inbox + sender ack inbox.
  DeviceSpan<SignalState> local_signals;   // [2 * max_groups]
  DeviceSpan<SignalState> remote_signals;  // [2 * max_groups]
                                           //   [0..max_groups)         = DATA_READY (sender→receiver, "tail")
                                           //   [max_groups..2*max_groups) = SLOT_FREE (receiver→sender, "head")
};
```

### `IbTransportTileState`

```cpp
struct IbTransportTileState {
  // Per-block step counters. Persistent across send_tile/recv_tile calls.
  DeviceSpan<int64_t> step_state;   // [2 * max_groups]
                                    //   [0..max_groups)         = sender step per block
                                    //   [max_groups..2*max_groups) = receiver step per block

  int tile_max_groups{0};

  // Signal pad. Receiver inbox + sender ack inbox.
  DeviceSpan<uint64_t> signal_pad;  // [2 * max_groups]
                                    //   [0..max_groups)         = DATA_READY (sender→receiver, "tail")
                                    //   [max_groups..2*max_groups) = SLOT_FREE (receiver→sender, "head")

  // NIC completion counters.
  // Bumped by the companion-QP loopback when an RDMA put is delivered.
  DeviceSpan<uint64_t> nic_done_counter;  // [max_groups]

  // Local send staging buffer (registered MR for RDMA source).
  DeviceSpan<std::byte> send_staging;  // [pipeline_depth * data_buffer_size]
};
```

**Per-slot layout** (one slot is `data_buffer_size` bytes, partitioned across
the calling blocks):

```
slot k  (= step / chunks_per_slot % pipeline_depth):
┌──────────────┬──────────────┬─────┬────────────────────┐
│ block 0 row  │ block 1 row  │ ... │ block (N-1) row    │
└──────────────┴──────────────┴─────┴────────────────────┘
   N = max_groups (the construction-time upper bound, NOT active_blocks)
   each row = per_block_slot_size = (data_buffer_size / active_blocks) & ~15ULL
```

`active_blocks` is per-call; `per_block_slot_size` is internal to the algorithm
and never exposed to users.

**Construction responsibilities (host):**
- Allocate `step_state`, `signal_pad`, `nic_done_counter`, `send_staging`,
  `recv_staging`.
- IB: register MRs for staging; exchange `recv_staging` rkeys + `signal_pad`
  rkeys with each peer.
- NVL: P2P-enable `recv_staging` access; exchange device pointers.
- Zero-init `step_state`, `signal_pad`, `nic_done_counter`.

**Destruction:** deregister MRs (IB), free buffers. Outstanding ops are the
caller's responsibility (kernel must finish before the host destructor runs).

---

## 3. API Surface

### Cpp (both transports, identical signature)

```cpp
class P2pIbgdaTransportDevice {
 public:
  __device__ void send_tile(
      ThreadGroup& group,
      const void* src,
      size_t nbytes,
      int active_blocks = 0,
      size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout());

  __device__ void recv_tile(
      ThreadGroup& group,
      void* dst,
      size_t nbytes,
      int active_blocks = 0,
      size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout());
};
// P2pNvlTransportDevice exposes the same two methods with the same signature.
```

### Triton (both transports, identical signature)

```python
@core.extern
def send_tile(
    src_ptr, nbytes,
    block_id, active_blocks,
    max_signal_bytes,
    timeout_ns,
    # Constexpr handles plumbed from host transport.
    # Bundles staging ptrs, signal pad ptrs, step state ptr, transport config values.
    # Exact constexpr/runtime split is an impl-time decision.
    transport_handle: tl.constexpr,
):
    ...

@core.extern
def recv_tile(dst_ptr, nbytes, block_id, active_blocks, max_signal_bytes, timeout_ns,
              transport_handle: tl.constexpr):
    ...
```

### Parameter table

| Param | Required | Default | Meaning |
|---|---|---|---|
| `group` (cpp) / `block_id` (Triton) | yes | — | Identifies this calling block. Slot routing uses `group.group_id` (cpp) or the `block_id` arg (Triton). |
| `src` / `dst` | yes | — | This block's pre-sliced data pointer. Caller computes per-block offset (see `TiledBuffer`). |
| `nbytes` | yes | — | This block's data size. May exceed `per_block_slot_size` — chunked internally over pipeline slots. |
| `active_blocks` | no | `0` → `tile_max_groups` | Number of blocks calling `send_tile`/`recv_tile` concurrently. Determines `per_block_slot_size = data_buffer_size / active_blocks`. |
| `max_signal_bytes` | no | `0` → `per_block_slot_size` | Hint for the maximum number of bytes between consecutive DATA_READY signals. The transport may signal more frequently if too many blocks share the data buffer. Capped at `per_block_slot_size` if larger (sub-slot signaling only). |
| `timeout` | no | `Timeout()` (no limit) | Per-wait timeout. Reuses `comms::pipes::Timeout`. On expiry: `__trap()`. |

### Special values

- **`nbytes == 0`** — block participates in convergent control flow but does no
  copy and no signal; `step_state` does not advance. Sender and receiver MUST
  both pass `nbytes==0` for the same `block_id` (per-block matching rule below).
- **`max_signal_bytes > per_block_slot_size`** — silently capped to `per_block_slot_size`.
  The protocol never signals less frequently than once per slot fill (sub-slot
  signaling only).
- **`active_blocks > tile_max_groups`** — `__trap()`. The precondition check
  at the top of the algorithm catches this and traps immediately rather than
  silently aliasing staging rows.

---

## 4. Coordination Contract

### Per-call contract

1. **CTA-cooperative.** All threads in `group` MUST call `send_tile` /
   `recv_tile` convergently. Cooperative memcpy across the block; leader thread
   issues signals and RDMA puts.
2. **Slot routing index = `group.group_id`** (cpp) / `block_id` extern arg
   (Triton). The *logical index within the calling group*, not raw `blockIdx.x`.
   So a kernel that does `auto [role, sub] = group.partition(2)` passes `sub`
   to `send_tile` / `recv_tile`, and `sub.group_id` (range `[0, sub.total_groups)`)
   is the slot row index.
3. **Trap precondition (debug-mode `__trap`):**
   `group.group_id < (active_blocks > 0 ? active_blocks : tile_max_groups)`.
   Violating this would alias two blocks onto the same staging row and silently
   corrupt data — trap converts that into an immediate, locatable failure.

### Cross-rank coordination

- For each `group_id k`: sender block_k's `(nbytes, active_blocks, max_signal_bytes)`
  MUST equal receiver block_k's. The protocol routes data through slot row `k`
  on both sides; mismatched values cause deadlock (receiver waits for more
  signals) or silent drop (receiver consumes too few).
- Across blocks within the same call: `nbytes` may differ per block (uneven tile
  partitions are supported as long as both sides agree per-block).

### Changing `active_blocks` between calls

If `active_blocks` changes between consecutive `send_tile`/`recv_tile` calls on the same transport, a **cross-rank barrier** is required between the two calls. Changing `active_blocks` alters `per_block_slot_size` and therefore the slot row layout; without a barrier, the receiver may still be draining the old layout while the sender begins writing the new one, corrupting staging data.

### Concurrency

- **Single-stream sequential calls on the same transport are supported** —
  internal `step_state` and `signal_pad` survive across calls; the next call
  resumes the protocol monotonically.
- **Multiple kernels on the same transport via different CUDA streams =
  undefined behavior** — they would race on `step_state` and `signal_pad`.

---

## 5. Algorithm

Both backends share the same precomputation and slot-rotation logic. The key
differences are in how data reaches the remote side and what synchronization
primitives are used.

| Aspect | NVL | IB |
|---|---|---|
| Data path | Direct P2P memcpy to **remote** `recv_staging` via NVLink | Cooperative memcpy to **local** `send_staging`, then fused RDMA put to remote `recv_staging` |
| NIC wait | None — P2P writes complete in-order | `wait_counter(nic_done_counter)` before reusing local staging |
| Signaling | `SignalState.signal(SIGNAL_SET, step)` via NVLink remote write | Fused RDMA put-with-signal (`put_signal_counter_remote`) |
| Drain | None — no outstanding async ops after memcpy + sync | Internal drain at end: `wait_counter(nic_done_counter, step)` |
| `send_staging` | Not used (`nullptr`) | Required (registered MR for RDMA source) |

### Common precomputation (both backends)

```text
block_id        = group.group_id
eff_active      = active_blocks > 0 ? active_blocks : tile_max_groups
trap if block_id >= eff_active

per_block_slot  = (data_buffer_size / eff_active) & ~15ULL
trap if per_block_slot == 0
chunk_size      = min(max_signal_bytes > 0 ? max_signal_bytes : per_block_slot,
                      per_block_slot)
chunks_per_slot = per_block_slot / chunk_size      // sub-slot signaling factor
total_chunks    = ceil(nbytes / chunk_size)

tail_signal_id  = block_id                         // DATA_READY (sender → receiver)
head_signal_id  = tile_max_groups + block_id         // SLOT_FREE  (receiver → sender)
```

### `send_tile` (NVL)

```text
if nbytes == 0: return

step = step_state.sender[block_id]

for s in [0, total_chunks):
    slot_step     = s / chunks_per_slot
    sub_step      = s % chunks_per_slot
    slot          = slot_step % pipeline_depth
    slot_off      = slot * data_buffer_size
    chunk_off     = sub_step * chunk_size
    data_off      = s * chunk_size
    bytes_this    = min(chunk_size, nbytes - data_off)

    // (1) Backpressure: wait for receiver to free this slot.
    //     Only at slot boundary (first sub-step) and only after the pipeline
    //     is fully filled.
    if sub_step == 0 and step >= chunks_per_slot * pipeline_depth:
        local_signals[head_signal_id].wait_until(
            group, CMP_GE,
            step - chunks_per_slot * pipeline_depth + 1,
            timeout)

    // (2) Cooperative P2P memcpy: src chunk -> remote recv_staging via NVLink.
    //     No local staging needed — NVLink writes go directly to the peer's
    //     staging buffer.
    memcpy_vectorized(
        remote_recv_staging + slot_off + staging_off + chunk_off,
        src + data_off,
        bytes_this, group)

    // (3) Barrier + signal DATA_READY to receiver.
    group.sync()
    if group.is_leader():
        remote_signals[tail_signal_id].signal(SIGNAL_SET, step + 1)

    step++

step_state.sender[block_id] = step
group.sync()
```

**Key difference from IB:** no NIC-done wait (step 1 in IB) and no drain (step
5 in IB). The P2P memcpy writes directly to remote memory — once `group.sync()`
completes, all threads have finished their stores, so the data is visible on the
remote side and local `src` is immediately safe. No outstanding async operations
remain.

### `recv_tile` (NVL)

```text
if nbytes == 0: return

step = step_state.receiver[block_id]

for s in [0, total_chunks):
    slot_step     = s / chunks_per_slot
    sub_step      = s % chunks_per_slot
    slot          = slot_step % pipeline_depth
    slot_off      = slot * data_buffer_size
    chunk_off     = sub_step * chunk_size
    data_off      = s * chunk_size
    bytes_this    = min(chunk_size, nbytes - data_off)

    // (1) Wait for sender's DATA_READY signal.
    local_signals[tail_signal_id].wait_until(
        group, CMP_GE, step + 1, timeout)

    // (2) Cooperative memcpy: local recv_staging -> dst.
    //     Sender wrote here via NVLink; we read from local memory (fast).
    memcpy_vectorized(
        dst + data_off,
        local_recv_staging + slot_off + staging_off + chunk_off,
        bytes_this, group)

    // (3) Barrier + conditional SLOT_FREE signal to sender.
    //     Signal only at slot boundaries (last sub-step in a slot or the
    //     very last step) to release the entire slot for reuse.
    group.sync()
    bool last_in_slot = (sub_step == chunks_per_slot - 1)
                        or (s == total_chunks - 1)
    if last_in_slot and group.is_leader():
        remote_signals[head_signal_id].signal(SIGNAL_SET, step + 1)

    step++

step_state.receiver[block_id] = step
group.sync()
```

### `send_tile` (IB)

```text
if nbytes == 0: return

step = step_state.sender[block_id]

for s in [0, total_chunks):
    slot_step     = s / chunks_per_slot
    sub_step      = s % chunks_per_slot
    slot          = slot_step % pipeline_depth
    slot_off      = slot * data_buffer_size
    chunk_off     = sub_step * chunk_size
    staging_off   = slot_off + block_id * per_block_slot + chunk_off
    data_off      = s * chunk_size
    bytes_this    = min(chunk_size, nbytes - data_off)

    // (1) Wait for prior NIC use of this slot to drain (local staging is safe).
    if step >= pipeline_depth * chunks_per_slot:
        wait_counter(group,
                     nic_done_counter[block_id],
                     step - pipeline_depth * chunks_per_slot + 1,
                     timeout)

    // (2) Cooperative memcpy: src chunk -> local send_staging.
    memcpy_vectorized(send_staging + staging_off,
                      src + data_off,
                      bytes_this, group)
    group.sync()

    // (3) Wait for receiver to free this slot — only at slot boundary.
    if sub_step == 0 and step >= pipeline_depth * chunks_per_slot:
        wait_signal(group,
                    signal_pad[tile_max_groups + block_id],       // SLOT_FREE row
                    step / chunks_per_slot - pipeline_depth + 1,
                    timeout)

    // (4) Fused RDMA put + remote DATA_READY signal + local NIC_DONE bump.
    if group.is_leader():
        put_signal_counter_remote(
            local_src     = send_staging        + staging_off,
            remote_dst    = recv_staging_remote + staging_off,
            nbytes        = bytes_this,
            remote_signal = signal_pad_remote[block_id],       // DATA_READY row
            signal_inc    = 1,
            local_counter = nic_done_counter[block_id],
            counter_inc   = 1)

    step++

step_state.sender[block_id] = step
group.sync()

// (5) Internal drain: wait for all RDMA puts on this block to complete.
wait_counter(group, nic_done_counter[block_id], step, timeout)
group.sync()
```

### `recv_tile` (IB)

```text
if nbytes == 0: return

step = step_state.receiver[block_id]

for s in [0, total_chunks):
    slot_step     = s / chunks_per_slot
    sub_step      = s % chunks_per_slot
    slot          = slot_step % pipeline_depth
    slot_off      = slot * data_buffer_size
    chunk_off     = sub_step * chunk_size
    staging_off   = slot_off + block_id * per_block_slot + chunk_off
    data_off      = s * chunk_size
    bytes_this    = min(chunk_size, nbytes - data_off)

    // (1) Wait for sender's data.
    wait_signal(group,
                signal_pad[block_id],                          // DATA_READY row
                step + 1,
                timeout)

    // (2) Cooperative memcpy: local recv_staging -> dst.
    memcpy_vectorized(dst + data_off,
                      recv_staging + staging_off,
                      bytes_this, group)
    group.sync()

    // (3) Tell sender slot is free — only at slot boundary.
    bool last_in_slot = (sub_step == chunks_per_slot - 1)
                        or (s == total_chunks - 1)
    if last_in_slot and group.is_leader():
        signal_remote(signal_pad_remote[tile_max_groups + block_id],  // SLOT_FREE row
                      increment = 1)

    step++

step_state.receiver[block_id] = step
group.sync()
```

### Why these waits are placed where they are

| Step | Backend | Why it is needed |
|---|---|---|
| `send_tile (1)` backpressure | NVL | Receiver may still be reading its local staging. Writing new data via NVLink would corrupt the receiver's in-progress memcpy. Slot-boundary only — sub-steps within a slot share the same slot. |
| `send_tile (1)` NIC drain | IB | NIC may still be reading local staging from a prior put. Memcpying new data would corrupt the in-flight RDMA. |
| `send_tile (3)` backpressure | IB | Receiver may still be reading remote staging. Putting new data would corrupt the receiver's read. Slot-boundary only. |
| `send_tile (5)` drain | IB | Without the drain, returning from `send_tile` would leave outstanding RDMA puts in flight. Internal drain makes the postcondition crisp: on return, all RDMA is delivered. |
| `recv_tile (1)` | Both | Receiver cannot consume staging until the sender has signaled DATA_READY. |
| `recv_tile (3)` | Both | Sender's backpressure relies on this signal. Slot-boundary only, matching sender's wait granularity. |

---

## 6. Worked Example

A bidirectional same-rank-pair send/recv kernel using `partition(2)` to split
blocks into senders and receivers.

```cpp
#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/TiledBuffer.cuh"
#include "comms/pipes/Timeout.cuh"

using namespace comms::pipes;

__global__ void bidirectional_send_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src, char* dst,
    size_t total_bytes,
    Timeout timeout) {
  auto group = make_block_group();
  auto [role, sub] = group.partition(2);
  const bool is_sender = (role == 0);

  // Each side partitions its own data evenly across its half of the blocks.
  TiledBuffer<char> tiles(is_sender ? src : dst, total_bytes, sub);

  if (is_sender) {
    transport->send_tile(
        sub,
        tiles.data(),
        tiles.bytes(),
        /*active_blocks=*/sub.total_groups,   // explicit — matches the partition size
        /*max_signal_bytes=*/  0,                   // default = one signal per slot fill
        timeout);
  } else {
    transport->recv_tile(
        sub,
        tiles.data(),
        tiles.bytes(),
        /*active_blocks=*/sub.total_groups,
        /*max_signal_bytes=*/  0,
        timeout);
  }
}

// Host side:
MultipeerIbgdaTransportConfig cfg{
    .cudaDevice = local_rank,
    // ... existing IB fields (qpDepth, etc.) ...
    .data_buffer_size    = 8 * 1024 * 1024,
    .tile_pipeline_depth = 2,
    .tile_max_groups     = 64,    // we launch 128 blocks total = 64 senders + 64 receivers
};
MultipeerIbgdaTransport transport(global_rank, world_size, bootstrap, cfg);
transport.exchange();

auto* device_xport = transport.get_p2p_transport_device(peer_rank);
bidirectional_send_recv_kernel<<<128, 256, 0, stream>>>(
    device_xport, send_buf, recv_buf, total_bytes, Timeout::ms(5000));
```

Notes on the example:
- The `partition(2)` renumbers `sub.group_id` to `[0, 64)` for both senders and
  receivers. The trap precondition is `sub.group_id < active_blocks=64`,
  which is satisfied.
- `TiledBuffer` partitions `total_bytes` evenly across the 64 sub-blocks; each
  block's `tiles.data()` is its own pre-sliced pointer, `tiles.bytes()` is its
  per-block byte count. Last block may be smaller (handled by `TiledBuffer`).
- `max_signal_bytes=0` keeps the default of one DATA_READY signal per slot fill —
  optimal for IB (amortizes RDMA atomic cost). To get sub-slot signaling, pass
  e.g., `max_signal_bytes = 16384` for finer-grained pipelining.

---

## 7. Out of Scope / Future Work

- **Explicit `drain_tile` API.** Drain is currently internal to `send_tile`
  (step 5). May be exposed if users want to overlap NIC drain with other
  device-side work between consecutive `send_tile` calls.
- **Multi-stream concurrency on the same transport.** Currently undefined.
  Would require per-stream `step_state` and `signal_pad` arenas.
- **Per-call config overrides.** `pipeline_depth` and `data_buffer_size` are
  construction-time only. Per-call overrides would require dynamic staging
  re-allocation.
- **Cross-rank rendezvous.** Use a separate barrier primitive; not coupled to
  send/recv_tile.
- **Error reporting on timeout.** Currently traps. Future: device flag +
  host-readable error code for graceful recovery in long-running services.
