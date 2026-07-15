# Tile Send/Recv Design

A **tile** is the smallest unit of work that all threads in a block process concurrently at one time. A user is free to choose how big or small a tile is. Smaller tiles allow more pipelining but incur more signaling overhead; larger tiles amortize signaling costs. Data is divided amongst blocks by creating tiles of data — each block may handle multiple tiles sequentially, and different blocks may handle different tiles in parallel.

A unified `send` / `recv` API for pipelined point-to-point transfers
on `P2pNvlTransportDevice` and `P2pIbgdaTransportDevice`, callable from CUDA
and Triton kernels. Composable building blocks for collectives (allgather,
alltoall, sendrecv) without users needing to manage staging, signals, slot
rotation, or pipeline depth.

**Target backends:** NVLink cpp, NVLink Triton, IB cpp, IB Triton. Both NVL
and IB use fixed channel geometry chosen at host init. Callers select a
channel with `group.group_id`; per-call arguments control transfer size and
optional sub-chunk signaling, not staging layout.

**In scope:** per-block tile send/recv with cooperative memcpy + pipelined staging.
**Out of scope:** explicit user-visible drain (handled internally), multi-stream
concurrency on the same transport, buffer registration (transport-owned), and
cross-rank rendezvous (separate barrier primitive).

---

## 1. Transport Setup

The tile API reuses the existing per-peer buffer settings already present in
each transport's config. No new `TileConfig` sub-struct is introduced — the
knobs the tile algorithm needs are drawn from the transport config.

### NVL (`MultiPeerNvlTransportConfig`)

The config carries the fixed-channel geometry explicitly:

| Field | Role in tile API |
|---|---|
| `perChannelSize` | Bytes owned by one channel in one pipeline slot. |
| `pipelineDepth` | Number of slots in the pipeline ring. |
| `maxNumChannels` | Number of **channels** allocated per peer. Each channel owns one fixed `perChannelSize` staging slice in every pipeline slot, plus one `NvlChannelState` for cursors + signals. |

The host derives `data_buffer_size = perChannelSize * maxNumChannels` for
the pipeline slot. Channel sizing is fixed at init.

### IB (`MultipeerIbTransportConfig`)

IB uses the same fixed-channel shape. The config exposes the first-order
geometry and derives the total staging slot size:

```cpp
struct MultipeerIbTransportConfig {
  // ... existing fields (qpDepth, qpsPerConnection, etc.) ...

  // Raw put()/signal() buffer size. For send()/recv(), this is derived as
  // perChannelSize * max_num_channels when perChannelSize is set.
  std::size_t dataBufferSize{0};

  // Bytes owned by one channel in one pipeline slot.
  std::size_t perChannelSize{0};

  // Number of channels allocated per peer.
  int max_num_channels{64};

  // Number of pipeline slots for send/recv staging.
  int pipelineDepth{2};
};
```

### Validation (throws at construction, both transports)

- `pipelineDepth >= 1`
- `maxNumChannels` / `max_num_channels >= 1`
- per-channel size is `>= 16` and 16-byte aligned, so each channel slot fits
  at least one 16-byte vectorized memcpy.

**Defaults rationale:** NVL and IB default to channel counts that cover the
largest NCCL P2P channel counts we expect to mirror. With
`perChannelSize=128 KiB` and `maxNumChannels=64`, one pipeline slot is 8 MiB.

---

## 2. Internal State

Owned by the transport, allocated and registered at construction. **Invisible
to users** — referenced here only for implementer reference.

The NVLink and IB transports use separate per-channel state structs because
the IB transport requires additional fields for NIC completion tracking and
local send staging that NVLink does not need.

### NVL: `NvlChannelState`

One `NvlChannelState` per channel per peer (array length = `maxNumChannels`).
The host allocates the array via an IPC-shared `GpuMemHandler` so the remote
sender / receiver can write the `data_ready` / `slot_free` signals into this
rank's local endpoint. Cursors are local-only (remote never touches them).

```cpp
struct alignas(128) NvlChannelState {
  int64_t send_cursor{0};  // bytes sent from this local endpoint (persistent)
  int64_t recv_cursor{0};  // bytes received by this local endpoint (persistent)
  SignalState data_ready;  // remote sender bumps via NVLink; local recv waits
  SignalState slot_free;   // remote recv bumps via NVLink; local send waits
};
```

Layout: cursors and pads share one cache line; each of the two signals gets
its own 128-byte cache line to keep concurrent NVLink writes by the remote
sender (to `data_ready`) and the remote receiver (to `slot_free`) from
false-sharing. The signals deliberately are not packed into the cursor line:
that would put local cursor updates and remote NVLink signal writes on the same
hot line. `sizeof(NvlChannelState) == 384` (one cursor line + two signal lines).

The device transport stores two pointers into this layout:

```cpp
NvlChannelState* local_channels_;   // this rank's endpoint; remote rank signals here
NvlChannelState* remote_channels_;  // remote rank's endpoint via IPC; this rank signals here
```

Both arrays have length `options_.max_num_channels`. Both
the channel index and the per-channel staging slice index are `group.group_id`.

### IB: `IbLocalChannel`, `IbRemoteChannel`, and `IbChannelLayout`

```cpp
struct IbLocalChannel {
  IbChannelProgress sendProgress;
  IbChannelProgress recvProgress;
  // Local DATA_READY, SLOT_FREE, and NIC_DONE endpoints.
  // Local send staging and channel-owned QP state.
};

struct IbRemoteChannel {
  // Peer DATA_READY and SLOT_FREE endpoints.
  // Peer recv staging slice and registration handles.
};

struct IbChannelLayout {
  std::byte* sendStaging;
  std::byte* recvStaging;
  DeviceSpan<IbLocalChannel> localChannels;
  DeviceSpan<IbRemoteChannel> remoteChannels;
  int max_num_channels;
  int pipelineDepth;
  std::size_t perChannelSize;
};
```

**Per-slot layout** (one slot is `data_buffer_size` bytes, partitioned across
fixed channels):

```
slot k  (= step / chunks_per_slot % pipeline_depth):
┌──────────────┬──────────────┬─────┬────────────────────┐
│ channel 0 row│ channel 1 row│ ... │ channel (N-1) row  │
└──────────────┴──────────────┴─────┴────────────────────┘
   N = maxNumChannels / max_num_channels (fixed at init).
   each row = per_channel_slot = per_channel_size
```

Using fewer-than-max channels wastes the unused channels' slices but does not
change live channels' bandwidth.

**Construction responsibilities (host):**
- NVL: allocate one IPC-shared `NvlChannelState[nPeers * maxNumChannels]`
  buffer; exchange via `GpuMemHandler::exchangeMemPtrs()`. P2P-enable
  `recv_staging` access; exchange device pointers. Zero-init the channel
  buffer (zeros cursors and signals).
- IB: allocate local channels, remote channel descriptors, send staging, and
  recv staging. Register MRs for staging and signal/counter storage; exchange
  peer channel descriptors and rkeys. Zero-init channel progress, signals, and
  counters.

**Destruction:** deregister MRs (IB), free buffers. Outstanding ops are the
caller's responsibility (kernel must finish before the host destructor runs).

---

## 3. API Surface

### Cpp

NVL and IB use the same call shape for blocking send/recv. Channel count is
fixed at init.

```cpp
class P2pNvlTransportDevice {
 public:
  __device__ void send(
      ThreadGroup& group,
      const void* src,
      size_t nbytes,
      size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout());

  __device__ void recv(
      ThreadGroup& group,
      void* dst,
      size_t nbytes,
      size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout());

  __device__ void forward(
      ThreadGroup& group,
      void* dst,
      size_t nbytes,
      P2pNvlTransportDevice& successor,
      size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout());
};

class P2pIbgdaTransportDevice {
 public:
  __device__ void send(
      ThreadGroup& group,
      const void* src,
      size_t nbytes,
      size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout());

  __device__ void recv(
      ThreadGroup& group,
      void* dst,
      size_t nbytes,
      size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout());
};
```

### Triton (both transports, identical signature)

```python
@core.extern
def send(
    src_ptr, nbytes,
    block_id,
    max_signal_bytes,
    timeout_ns,
    # Constexpr handles plumbed from host transport.
    # Bundles staging pointers, channel endpoints, and transport config values.
    # Exact constexpr/runtime split is an impl-time decision.
    transport_handle: tl.constexpr,
):
    ...

@core.extern
def recv(dst_ptr, nbytes, block_id, max_signal_bytes, timeout_ns,
              transport_handle: tl.constexpr):
    ...
```

### Parameter table

| Param | Required | Default | Meaning |
|---|---|---|---|
| `group` (cpp) / `block_id` (Triton) | yes | — | Identifies this calling block. Slot routing uses `group.group_id` (cpp) or the `block_id` arg (Triton). |
| `src` / `dst` | yes | — | This block's pre-sliced data pointer. Caller computes per-block offset (see `TiledBuffer`). |
| `nbytes` | yes | — | This block's data size. May exceed `per_channel_size` — chunked internally over pipeline slots. |
| `max_signal_bytes` | no | `0` → `per_channel_size` | Hint for the maximum number of bytes between consecutive DATA_READY signals. Capped at `per_channel_size` if larger (sub-slot signaling only). |
| `timeout` | no | `Timeout()` (no limit) | Per-wait timeout. Reuses `comms::prims::Timeout`. On expiry: `__trap()`. |

### Special values

- **`nbytes == 0`** — block participates in convergent control flow but does no
  copy and no signal; channel progress does not advance. Sender and receiver MUST
  both pass `nbytes==0` for the same `block_id` (per-block matching rule below).
- **`max_signal_bytes > per_channel_size`** — silently capped to `per_channel_size`.
  The protocol never signals less frequently than once per slot fill (sub-slot
  signaling only).
- **`group.group_id >= max_num_channels`** — `__trap()`. Catches a kernel
  selecting a channel the host did not allocate, which would alias two groups
  onto the same channel or corrupt adjacent state.

---

## 4. Coordination Contract

### Per-call contract

1. **CTA-cooperative.** All threads in `group` MUST call `send` /
   `recv` convergently. Cooperative memcpy across the block; leader thread
   issues signals and RDMA puts.
2. **Slot routing index = `group.group_id`** (cpp) / `block_id` extern arg
   (Triton). The *logical index within the calling group*, not raw `blockIdx.x`.
   So a kernel that does `auto [role, sub] = group.partition(2)` passes `sub`
   to `send` / `recv`, and `sub.group_id` (range `[0, sub.total_groups)`)
   is the slot row index.
3. **Trap precondition (debug-mode `__trap`):**
   - `group.group_id < max_num_channels`. The channel index is
     `group.group_id`; selecting a channel outside the allocated range would
     alias state or corrupt adjacent staging.

### Cross-rank coordination

- For each `group_id k`: sender block_k's `(nbytes, max_signal_bytes)` MUST
  equal receiver block_k's. The protocol routes data through slot row `k` on
  both sides; mismatched values cause deadlock (receiver waits for more
  signals) or silent drop (receiver consumes too few).
- Across blocks within the same call: `nbytes` may differ per block (uneven tile
  partitions are supported as long as both sides agree per-block).

### Changing per-call sizing between calls

The per-channel slot size is fixed at host init, so consecutive calls cannot
change the staging layout. `max_signal_bytes` may vary between calls because
progress is tracked in bytes rather than in a layout-dependent step count.

### Concurrency

- **Single-stream sequential calls on the same transport are supported** —
  internal channel progress and signal/counter state survive across calls; the
  next call resumes the protocol monotonically.
- **Multiple kernels on the same transport via different CUDA streams =
  undefined behavior** — they would race on channel progress and signals.

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

### Common precomputation

**NVL:**
```text
channel         = group.group_id
trap if group.total_groups > options.max_num_channels

per_channel_slot = options.per_channel_slot            // fixed at host init
trap if per_channel_slot == 0
chunk_size      = min(max_signal_bytes > 0 ? max_signal_bytes : per_channel_slot,
                      per_channel_slot)
chunks_per_slot = per_channel_slot / chunk_size       // sub-slot signaling factor
total_chunks    = ceil(nbytes / chunk_size)

local_ch        = local_channels_[channel]            // remote rank writes here via NVLink
remote_ch       = remote_channels_[channel]           // this rank writes here via NVLink
```

**IB:**
```text
channel         = group.group_id
trap if channel >= max_num_channels

per_block_slot  = perChannelSize & ~15ULL
trap if per_block_slot == 0
chunk_size      = min(max_signal_bytes > 0 ? max_signal_bytes : per_block_slot,
                      per_block_slot)
chunks_per_slot = per_block_slot / chunk_size      // sub-slot signaling factor
total_chunks    = ceil(nbytes / chunk_size)

local_ch        = local_channels_[channel]
remote_ch       = remote_channels_[channel]
```

### `send` (NVL)

The actual implementation uses a byte-cursor (`send_cursor`) rather than a
step counter — wait/signal values are byte positions, which lets `chunk_size`
vary between calls without losing monotonicity.

```text
if nbytes == 0: return

base_byte    = local_ch.send_cursor
staging_off  = channel * per_channel_slot
pipeline_bytes = per_channel_slot * pipeline_depth

for data_off in [0, protocol_bytes):           // protocol_bytes = align16(nbytes)
    stream_start    = base_byte + data_off
    pipeline_off    = stream_start % pipeline_bytes
    slot            = pipeline_off / per_channel_slot
    slot_off        = slot * data_buffer_size
    chunk_off       = pipeline_off - slot * per_channel_slot
    copy_bytes      = min(chunk_size, protocol_bytes - data_off,
                          per_channel_slot - chunk_off)
    stream_end      = stream_start + copy_bytes

    // (1) Backpressure: only once the pipeline has wrapped.
    if stream_end > pipeline_bytes:
        local_ch.slot_free.wait_until(
            group, CMP_GE, stream_end - pipeline_bytes, timeout)

    // (2) Cooperative P2P memcpy: src chunk -> remote staging via NVLink.
    memcpy_vectorized(
        remote_recv_staging + slot_off + staging_off + chunk_off,
        src + data_off,
        valid_payload(copy_bytes, nbytes, data_off),
        group)

    // (3) Barrier + signal DATA_READY to the remote endpoint.
    group.sync()
    if group.is_leader():
        remote_ch.data_ready.signal(SIGNAL_SET, stream_end)

    data_off += copy_bytes

if group.is_leader():
    local_ch.send_cursor = base_byte + protocol_bytes
group.sync()
```

**Key difference from IB:** no NIC-done wait (step 1 in IB) and no drain (step
5 in IB). The P2P memcpy writes directly to remote memory — once `group.sync()`
completes, all threads have finished their stores, so the data is visible on the
remote side and local `src` is immediately safe. No outstanding async operations
remain.

### `recv` (NVL)

```text
if nbytes == 0: return

base_byte    = local_ch.recv_cursor
staging_off  = channel * per_channel_slot
pipeline_bytes = per_channel_slot * pipeline_depth

for data_off in [0, protocol_bytes):
    stream_start    = base_byte + data_off
    pipeline_off    = stream_start % pipeline_bytes
    slot            = pipeline_off / per_channel_slot
    slot_off        = slot * data_buffer_size
    chunk_off       = pipeline_off - slot * per_channel_slot
    copy_bytes      = min(chunk_size, protocol_bytes - data_off,
                          per_channel_slot - chunk_off)
    stream_end      = stream_start + copy_bytes

    // (1) Wait for DATA_READY from the remote endpoint.
    local_ch.data_ready.wait_until(group, CMP_GE, stream_end, timeout)

    // (2) Cooperative memcpy: local recv_staging -> dst.
    memcpy_vectorized(
        dst + data_off,
        local_recv_staging + slot_off + staging_off + chunk_off,
        valid_payload(copy_bytes, nbytes, data_off),
        group)

    // (3) Barrier + conditional SLOT_FREE signal to peer.
    //     Signal only at slot boundaries (chunk hits end of slot, or last
    //     chunk overall) to release the entire slot for reuse.
    group.sync()
    last_in_slot = (chunk_off + copy_bytes == per_channel_slot) or
                   (data_off + copy_bytes == protocol_bytes)
    if last_in_slot and group.is_leader():
        remote_ch.slot_free.signal(SIGNAL_SET, stream_end)

    data_off += copy_bytes

if group.is_leader():
    local_ch.recv_cursor = base_byte + protocol_bytes
group.sync()
```

### `send` (IB)

```text
if nbytes == 0: return

base_byte = local_ch.sendProgress.cursor
pipeline_bytes = per_block_slot * pipeline_depth

for s in [0, total_chunks):
    slot_step     = s / chunks_per_slot
    sub_step      = s % chunks_per_slot
    slot          = slot_step % pipeline_depth
    slot_off      = slot * data_buffer_size
    chunk_off     = sub_step * chunk_size
    staging_off   = slot_off + channel * per_block_slot + chunk_off
    data_off      = s * chunk_size
    bytes_this    = min(chunk_size, nbytes - data_off)
    stream_end    = base_byte + data_off + bytes_this

    // (1) Wait for prior NIC use of this slot to drain (local staging is safe).
    if stream_end > pipeline_bytes:
        wait_counter(group,
                     local_ch.nic_done_counter,
                     stream_end - pipeline_bytes,
                     timeout)

    // (2) Cooperative memcpy: src chunk -> local send_staging.
    memcpy_vectorized(send_staging + staging_off,
                      src + data_off,
                      bytes_this, group)
    group.sync()

    // (3) Wait for receiver to free this slot — only at slot boundary.
    if sub_step == 0 and stream_end > pipeline_bytes:
        wait_signal(group,
                    local_ch.slot_free,
                    stream_end - pipeline_bytes,
                    timeout)

    // (4) Fused RDMA put + remote DATA_READY signal + local NIC_DONE bump.
    if group.is_leader():
        put_signal_counter_remote(
            local_src     = send_staging        + staging_off,
            remote_dst    = recv_staging_remote + staging_off,
            nbytes        = bytes_this,
            remote_signal = remote_ch.data_ready,
            signal_val    = stream_end,
            local_counter = local_ch.nic_done_counter,
            counter_val   = stream_end)

local_ch.sendProgress.cursor = base_byte + protocol_bytes
group.sync()

// (5) Internal drain: wait for all RDMA puts on this channel to complete.
wait_counter(group, local_ch.nic_done_counter, base_byte + protocol_bytes, timeout)
group.sync()
```

### `recv` (IB)

```text
if nbytes == 0: return

base_byte = local_ch.recvProgress.cursor

for s in [0, total_chunks):
    slot_step     = s / chunks_per_slot
    sub_step      = s % chunks_per_slot
    slot          = slot_step % pipeline_depth
    slot_off      = slot * data_buffer_size
    chunk_off     = sub_step * chunk_size
    staging_off   = slot_off + channel * per_block_slot + chunk_off
    data_off      = s * chunk_size
    bytes_this    = min(chunk_size, nbytes - data_off)
    stream_end    = base_byte + data_off + bytes_this

    // (1) Wait for sender's data.
    wait_signal(group,
                local_ch.data_ready,
                stream_end,
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
        signal_remote(remote_ch.slot_free,
                      signal_val = stream_end)

local_ch.recvProgress.cursor = base_byte + protocol_bytes
group.sync()
```

### Why these waits are placed where they are

| Step | Backend | Why it is needed |
|---|---|---|
| `send (1)` backpressure | NVL | Receiver may still be reading its local staging. Writing new data via NVLink would corrupt the receiver's in-progress memcpy. Slot-boundary only — sub-steps within a slot share the same slot. |
| `send (1)` NIC drain | IB | NIC may still be reading local staging from a prior put. Memcpying new data would corrupt the in-flight RDMA. |
| `send (3)` backpressure | IB | Receiver may still be reading remote staging. Putting new data would corrupt the receiver's read. Slot-boundary only. |
| `send (5)` drain | IB | Without the drain, returning from `send` would leave outstanding RDMA puts in flight. Internal drain makes the postcondition crisp: on return, all RDMA is delivered. |
| `recv (1)` | Both | Receiver cannot consume staging until the sender has signaled DATA_READY. |
| `recv (3)` | Both | Sender's backpressure relies on this signal. Slot-boundary only, matching sender's wait granularity. |

---

## 6. Worked Example

A bidirectional same-rank-pair send/recv kernel using `partition(2)` to split
blocks into senders and receivers.

```cpp
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/core/Timeout.cuh"

using namespace comms::prims;

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
    transport->send(
        sub,
        tiles.data(),
        tiles.bytes(),
        /*max_signal_bytes=*/0,  // default = one signal per slot fill
        timeout);
  } else {
    transport->recv(
        sub,
        tiles.data(),
        tiles.bytes(),
        /*max_signal_bytes=*/0,
        timeout);
  }
}

// Host side:
MultipeerIbTransportConfig cfg{
    .cudaDevice = local_rank,
    // ... existing IB fields (qpDepth, etc.) ...
    .perChannelSize   = 128 * 1024,
    .max_num_channels = 64, // 128 blocks total = 64 senders + 64 receivers
    .pipelineDepth    = 2,
};
MultipeerIbgdaTransport transport(global_rank, world_size, bootstrap, cfg);
transport.exchange();

auto* device_xport = transport.get_p2p_transport_device(peer_rank);
bidirectional_send_recv_kernel<<<128, 256, 0, stream>>>(
    device_xport, send_buf, recv_buf, total_bytes, Timeout::ms(5000));
```

Notes on the example:
- The `partition(2)` renumbers `sub.group_id` to `[0, 64)` for both senders
  and receivers. The trap precondition is `sub.group_id < max_num_channels`,
  which is satisfied.
- `TiledBuffer` partitions `total_bytes` evenly across the 64 sub-blocks; each
  block's `tiles.data()` is its own pre-sliced pointer, `tiles.bytes()` is its
  per-block byte count. Last block may be smaller (handled by `TiledBuffer`).
- `max_signal_bytes=0` keeps the default of one DATA_READY signal per slot fill —
  optimal for IB (amortizes RDMA atomic cost). To get sub-slot signaling, pass
  e.g., `max_signal_bytes = 16384` for finer-grained pipelining.

---

## 7. Out of Scope / Future Work

- **Explicit `drain_tile` API.** Drain is currently internal to `send`
  (step 5). May be exposed if users want to overlap NIC drain with other
  device-side work between consecutive `send` calls.
- **Multi-stream concurrency on the same transport.** Currently undefined.
  Would require per-stream channel progress and signal/counter arenas.
- **Per-call config overrides.** `pipelineDepth` and `perChannelSize` are
  construction-time only. Per-call overrides would require dynamic staging
  re-allocation.
- **Cross-rank rendezvous.** Use a separate barrier primitive; not coupled to
  send/recv.
- **Error reporting on timeout.** Currently traps. Future: device flag +
  host-readable error code for graceful recovery in long-running services.
