# MCCL AllReduce IBGDA LL128 Design

Status: Proposed

## Summary

This document proposes an IBGDA LL128 protocol for Phase 2 of MCCL fused
AllReduce. It preserves the current IBGDA staging, RDMA put, local completion,
`SLOT_FREE` backpressure, and Tree/Ring scheduling machinery while replacing
the current LL wire format:

```text
LL:     8B =   4B payload + 4B flag
LL128: 128B = 120B payload + 8B flag
```

The main decisions are:

- Use the NCCL LL128 register organization: eight `uint64_t` wire words per
  thread, 2048B of wire data per warp slice, 16 LL128 lines, and 1920B of
  payload.
- Use warp lanes 7, 15, 23, and 31 as flag lanes. Each owns the final 16B slot
  of four lines; its low 8B is payload and its high 8B is the generation flag.
- Keep buffer ownership out of the packet flag. The flag means data-ready;
  existing `SLOT_FREE` signals continue to control ring-slot reuse.
- Provide separate wait-and-decode and already-ready decode functions so the
  progress path does not rescan flags after `all_flags_set()` succeeds.
- Give LL128 a distinct protocol slot. Simple, LL, and LL128 must not share
  persistent staging/progress state.
- Initially support staged Ring and Tree Phase 2 only. Registered/zero-copy,
  warp proxy, Pipes trace, and Direct are out of scope for the first version.
- Treat NIC-to-GPU ordered visibility of one 128B line as an enablement
  blocker, not an assumption inferred from a passing stress test.

The data-path model comes from
`comms/ncclx/v2_31/src/device/prims_ll128.h`. This design intentionally does
not use `comms/prims/transport/ll128/Ll128Ops.cuh` as a reference.

## Background

MCCL fused AllReduce uses:

```text
640 CUDA threads/block = 20 warps/block
```

`MCCL_MAX_NBLOCKS` limits the number of blocks; it does not change
`blockDim.x`. With `MCCL_MAX_NBLOCKS=1`, the normal fused launch is one
640-thread CTA.

The fused algorithm has three phases:

1. NVL ReduceScatter within the node.
2. Algorithm-specific IB AllReduce across nodes.
3. NVL AllGather within the node.

Only Phase 2 changes for LL128.

The current IB transport already supplies the mechanisms LL128 should reuse:

- protocol-specific staging and persistent progress slots;
- NIC local-completion protection for send-staging reuse;
- remote recv-staging backpressure through `SLOT_FREE`;
- blocking `send`, `recv`, and `forward`;
- resumable progress send/recv;
- Ring reduce-forward CopyOps;
- Tree receive-reduce CopyOps;
- generation-based ABA prevention and abort handling.

LL128 is therefore primarily a packet-codec and protocol-geometry change, plus
new compile-time AllReduce kernel variants. It should not create a second
transport implementation.

## Goals

1. Provide an IBGDA LL128 Phase-2 format for MCCL Ring and Tree AllReduce.
2. Reuse existing WQE, completion, `SLOT_FREE`, channel, and fault-tolerance
   behavior.
3. Efficiently support the 640-thread kernel and the 320/160/64-thread groups
   created by Tree and Ring striping.
4. Fuse receive, reduction, copy, and forwarding in registers without a
   chunk-sized intermediate global-memory round trip.
5. Preserve correctness for every legal AllReduce alignment and tail;
   16-byte user-buffer alignment remains a fast path, not an API requirement.
6. Keep Simple, LL, and LL128 persistent resources isolated.
7. Select the protocol on the host before launch; do not add a device-side
   runtime protocol branch to existing kernels.

## Non-goals

- Changing NVL Phase 1 or Phase 3.
- Implementing LL128 for AllReduce Direct in the first version.
- Registered or zero-copy LL128 sends.
- Ring warp-proxy LL128 kernels.
- Combined Pipes trace and LL128 kernels.
- Automatic protocol tuning in the first version.
- Reproducing NCCL's connection head/tail state machine.

## Wire Format

### Packet layout

```text
bytes   0..119: payload
bytes 120..127: uint64_t generation flag

uint64 word  0..14: payload
uint64 word 15:     flag
```

Introduce an explicit IBGDA geometry rather than simply removing the current
8-byte-only guard from `LlxPacket`:

```cpp
struct Ll128PacketGeometry {
  using FlagType = uint64_t;

  static constexpr int kData = 120;
  static constexpr int kFlag = 8;
  static constexpr int kPacketBytes = 128;
  static constexpr int kSlotBytes = 16;
  static constexpr int kThreadsPerPacket = 8;
  static constexpr int kPacketsPerWarp = 4;
  static constexpr int kFlagLane = 7;
};
```

The separate geometry makes the correctness contracts explicit:

- LL relies on one atomic 64-bit `{data, flag}` transfer.
- LL128 relies on ordered NIC-to-GPU visibility within one aligned 128B line.

### Payload/wire conversion

```text
packet_count(payload) = ceil(payload / 120)
wire_bytes(payload)   = packet_count(payload) * 128
max_payload(wire)     = floor(wire / 128) * 120
```

Wire efficiency is `120 / 128 = 93.75%`, versus 50% for current LL. LL128 is
therefore a candidate for the range between tiny LL messages and Simple's
larger-message range.

### Flag lifecycle

Reuse current IBGDA LL generation semantics:

```text
flagVal = streamPayload / pipelineBytesPayload + 1
```

The receiver accepts only the expected generation for the current ring pass,
so stale packets from a previous use of the staging slot do not satisfy
readiness.

The embedded flag is not an ACK:

```text
sender:   pack(flagVal) -> RDMA put
receiver: wait(flagVal) -> consume/reduce -> signal SLOT_FREE
```

`SLOT_FREE` remains the only remote recv-staging reuse permission.

## Warp and Register Mapping

### One 128B line

Eight lanes each own one 16B wire slot:

| Lane within eight | Line bytes | Meaning |
|---:|---:|---|
| 0 | 0-15 | 16B payload |
| 1 | 16-31 | 16B payload |
| 2 | 32-47 | 16B payload |
| 3 | 48-63 | 16B payload |
| 4 | 64-79 | 16B payload |
| 5 | 80-95 | 16B payload |
| 6 | 96-111 | 16B payload |
| 7 | 112-127 | 8B payload + 8B flag |

The flag lanes are therefore 7, 15, 23, and 31. Lane 7 is the eighth thread
under zero-based indexing; lane 8 starts the next line.

### Recommended warp slice

Use eight wire words per thread:

```text
32 threads * 8 uint64 words = 256 words = 2048B wire
256 / 16 words-per-line      = 16 LL128 lines
16 lines * 120B              = 1920B payload/warp/slice
```

Each lane performs four 16B wire operations. Each flag lane owns four flags.

### MCCL group widths

Use group-local warp IDs and support any warp-aligned group:

| Threads | Warps | Lines/slice | Payload/slice | Wire/slice | Consumer |
|---:|---:|---:|---:|---:|---|
| 64 | 2 | 32 | 3,840B | 4,096B | Ring, 10 stripes |
| 160 | 5 | 80 | 9,600B | 10,240B | Ring, 4 stripes; Direct send |
| 320 | 10 | 160 | 19,200B | 20,480B | Ring/Tree, 2 stripes |
| 480 | 15 | 240 | 28,800B | 30,720B | Future Direct recv |
| 640 | 20 | 320 | 38,400B | 40,960B | Full Tree/Ring block |

```cpp
const int warpInGroup = group.thread_id_in_group / 32;
const int lane = group.thread_id_in_group % 32;
const int numWarps = group.group_size / 32;
const bool flagLane = lane % 8 == 7;

constexpr int kWordsPerWarpSlice = 32 * 8;
size_t wireWord = warpInGroup * kWordsPerWarpSlice + 2 * lane;

for (; wireWord < chunkWireWords;
     wireWord += numWarps * kWordsPerWarpSlice) {
  // Process one warp slice.
}
```

Do not use a kernel-global warp ID. `ThreadGroup::group_id` selects the logical
IB channel or stripe; it is not the warp index inside the codec group.

## Contiguous Payload to Wire Registers

Direct packet-centric addressing:

```text
payload + packet * 120 + laneInPacket * 16
```

makes every odd packet start eight bytes off a 16-byte boundary. The fast path
should instead keep contiguous 16B user-buffer accesses and insert/remove flag
holes in registers, following NCCL's mapping.

### Aligned fast path

Each thread owns `uint64_t regs[8]`. For `g` in `[0, 4)`, the contiguous
user-buffer 16B vector index is:

```cpp
ix = g * 32 - 4 * (g / 2) + lane - (g % 2) * (lane / 8);
```

All lanes load for even `g`; flag lanes skip the original load for odd `g`.
After independent memory operations have been issued, flag lanes perform:

```cpp
for (int g = 1; g < 4; g += 2) {
  if (flagLane) {
    regs[2 * g] = regs[2 * g - 1];
  }
}
```

This converts 120 contiguous 16B vectors, or 1920B, into a 2048B register
layout containing 16 eight-byte flag holes.

For `u` in `{0, 2, 4, 6}`, a lane's wire address in `uint64_t` units is:

```cpp
wire + sliceBase + u * 32 + 2 * lane
```

For `g = u/2`, subgroup `q = lane/8`, and lane-within-subgroup `r = lane%8`,
this is line `4*g+q`, words `2*r` and `2*r+1`.

### Misaligned path

AllReduce guarantees only element alignment, and correctness tests exercise
buffer offsets. LL128 cannot require 16-byte user-buffer alignment.

Provide:

1. A 16-byte-aligned vector fast path.
2. A misaligned/tail fallback preserving the same logical register layout.

The preferred fallback is per-warp shared-memory staging. One warp needs
2048B; 20 warps need approximately 40KiB. Audit this against current Tree/Ring
shared-memory use. If that cost is unacceptable, a byte-safe scalar fallback
is valid; only performance may change.

## Codec Architecture

Keep `LLImpl<P>` as the external CopyOp facade, but dispatch internally to
packet-specific codecs:

```cpp
template <typename P>
struct PacketCodec;

template <>
struct PacketCodec<LlPacketGeometry>;

template <>
struct PacketCodec<Ll128PacketGeometry>;
```

The current LL implementation assigns a complete packet to one thread, and its
reduction/repack helpers assume a 4B payload. LL128 requires new cooperative
implementations; simply instantiating `LlxPacket<120,8>` is insufficient.

### `pack`

1. Load one contiguous user slice into registers.
2. Apply compact-to-wire permutation in flag lanes.
3. Insert `flagVal` for each active line.
4. Have every lane perform four aligned 16B staging stores.
5. Zero invalid final-line payload bytes while always writing the flag.
6. Synchronize the codec group before returning.
7. The transport leader then executes `__threadfence_system()` and posts the
   RDMA WQE.

```cpp
for (int u = 0; u < 8; u += 2) {
  store128(
      wire + sliceBase + u * 32 + 2 * lane,
      regs[u],
      flagLane ? flagVal : regs[u + 1]);
}
```

### `all_flags_set`

This is the non-blocking progress readiness probe:

- Only flag lanes load flags; other lanes begin with a true verdict.
- Each flag lane checks its four lines in every assigned slice.
- Combine within each warp using ballot/all-vote operations.
- Combine across the group with `group.all()`.
- Return one group-uniform result without copying payload or spinning.

### `unpack_wait`

Use NCCL's conservative blocking baseline:

1. Every lane loads its four 16B wire slots on each polling iteration.
2. Flag lanes compare the 16 flags owned by the warp.
3. `__any_sync()` keeps wait and abort decisions warp-uniform.
4. Once every flag matches, reload the complete wire slice so non-flag lanes
   do not retain payload read before the final flag became ready.
5. Remove flag holes and store contiguous payload.

A later IBGDA-specific optimization may poll only flags and load payload once
after readiness. It requires separate ordered-visibility proof and benchmark
evidence and is not the initial baseline.

### `unpack_ready`

The progress path calls this after `all_flags_set()` succeeds. It performs only
the full wire load, register compaction, and destination store; it does not
rescan flags.

### `unpack_reduce_wait` and `unpack_reduce_ready`

```text
load local accumulator registers
             |
             +-- overlap with blocking readiness wait
             v
load incoming wire registers
remove or ignore flag words
reduce in registers
store contiguous accumulator
```

The 120B payload divides evenly across all current fused element sizes:

| Element size | Normal-lane elements | Flag-lane elements | Elements/line |
|---:|---:|---:|---:|
| 1B | 16 | 8 | 120 |
| 2B | 8 | 4 | 60 |
| 4B | 4 | 2 | 30 |
| 8B | 2 | 1 | 15 |

fp64 and int64 never cross LL128 line boundaries. No cross-lane element
reassembly is needed. Monitor register pressure: receive-reduce may hold
`local[8]` and `incoming[8]` simultaneously.

### `repack`

For an already-ready upstream chunk:

1. Load upstream wire registers without polling.
2. Optionally compact/store the local destination.
3. Replace flags with the downstream `fwdFlagVal`.
4. Store downstream staging.

Do not copy upstream packets verbatim because upstream and downstream rings
normally expect different generations.

### `repack_reduce`

```text
incoming wire -> registers
local operand -> registers
reduce
stamp downstream generation
write downstream staging
```

No intermediate contiguous global-memory buffer is created.

## IBGDA Transport Integration

### Protocol slot

Add:

```cpp
namespace protocol {
struct LL128 {
  static constexpr int kProtoSlot = 2;
  using Packet = Ll128PacketGeometry;
  static constexpr size_t kData = 120;
  static constexpr size_t kPacketBytes = 128;
  // max_payload() and wire_bytes()
};
}
```

Simple, LL, and LL128 need separate slots because these values persist across
launches and use protocol-specific units/layouts:

- send/recv progress cursors;
- staging-ring position;
- generation values;
- `DATA_READY`, `SLOT_FREE`, and counter state;
- send-completion slot generations.

Increase `kNumProtoSlots` from two to three. At the current default geometry,
one extra bank costs approximately 8MiB of send staging and 8MiB of recv
staging per peer, or 16MiB combined. It does not allocate more QPs.

### Blocking path

Add LL128 dispatches parallel to current LL:

- `prepareSendBuf(protocol::LL128)` calls LL128 `sendLL`.
- `consumeRecvBuf(protocol::LL128)` calls wait/decode or wait/reduce.
- `prepareForwardBuf(protocol::LL128)` calls ready repack/repack-reduce.

The put carries no `DATA_READY`:

1. Wait for local completion of the prior send-staging use.
2. Pack LL128 staging.
3. Wait for remote `SLOT_FREE` when the pipeline wraps.
4. Synchronize the group.
5. Leader executes `__threadfence_system()`.
6. Submit one RDMA put.
7. Record local completion.

The receiver publishes existing `SLOT_FREE` credit after consume/reduce.

### Progress path

The core progress state machine already uses protocol `wire_bytes()` and
`max_payload()`. Generalize the LL-specific seams that hard-code
`protocol::LL` and `LlxPacket<4,4>`:

```cpp
template <typename Proto>
using PacketFor = typename Proto::Packet;
```

LL128 progress receive becomes:

1. `all_flags_set()`: false returns `Waiting` without spinning.
2. `unpack_ready()` or `unpack_reduce_ready()`.
3. Publish recv credit.

An LL128 put still advances the shared send-QP cursor even without
`DATA_READY`. Increment `recvDataReadyLaneCursor` exactly once per successfully
consumed chunk or a later Simple operation can wait on the wrong QP lane.

### Geometry fixes

Current blocking and progress geometry assumes `Proto::kData` is a power of
two and uses `max(kData, 8)` as an LCM. LL128 needs a real LCM:

```text
lcm(120, 8) = 120
```

The current tail-padding helper also contains fixed 16B rounding assumptions.
It must advance from protocol-rounded payload bytes rather than applying a
second fixed rounding that would move the LL128 cursor incorrectly.

Required invariants:

- staging base is 128B aligned;
- `perChannelSize` is a multiple of 128;
- `perChannelSize / pipelineDepth` is a multiple of 128;
- every slot/chunk wire offset is 128B aligned;
- every transport or QP split is 128B aligned;
- one line never crosses independent QPs or write domains;
- `perBlockSlotPayload = floor(perBlockSlotWire / 128) * 120`;
- non-final chunks contain integral 120B payload quanta;
- reduction chunks contain whole elements;
- final RDMA writes cover complete packets while user accesses touch only
  valid bytes;
- tail padding and cursors convert explicitly between payload and wire units.

NCCL net-IB also uses 128B chunk alignment for LL/LL128 and keeps multi-QP
split points aligned. Prims needs the same line-preservation rule.

## MCCL AllReduce Integration

### Launch shape

Keep `blockDim.x=640` with no LL128 proxy warp. With
`MCCL_MAX_NBLOCKS=1`, the launch is one 640-thread block.

### Ring

Ring is the first integration target because it exercises the full codec:

- initial send: `pack`;
- reduce-forward: `repack_reduce`;
- final reduce-scatter receive: `unpack_reduce`;
- all-gather forwarding/copy: `repack`;
- final receive: `unpack`.

Ring groups are 640/320/160/64 threads for 1/2/4/10 stripes. The codec must
not assume a full block.

### Tree

Tree's two logical lanes are data halves, not thread halves. A uniform progress
scheduler alternates them over the same physical group.

- One virtual stripe: a full 640-thread group.
- Two virtual stripes: two independent 320-thread groups; each still drives
  both logical tree lanes.

LL128 Tree needs packetized progress send, `all_flags_set()` receive,
internal-node receive-reduce, and broadcast receive/copy.

LL128 is initially IBGDA-only. Tree validation currently permits IBRC in the
non-registered, single-stripe case, so host selection must add an all-peer
IBGDA gate. Never select LL128 over an IBRC edge.

### Direct

Direct remains Simple initially. Its Phase 2 uses 480 recv threads and 160 send
threads with a different rail schedule. Design it separately after Ring/Tree.

### Typed kernel ownership

Bind the protocol into the kernel symbol at compile time. Add independent
owners such as:

```text
AllReduceIbRingFloat16Ll128.cu
AllReduceIbRingFloat32Ll128.cu
AllReduceIbTreeFloat16Ll128.cu
AllReduceIbTreeFloat32Ll128.cu
```

Each defines:

```cpp
#define MCCL_ALLREDUCE_*_PROTO comms::prims::protocol::LL128
```

Initially match current LL datatypes: fp16, bf16, fp32, fp64, and int64. This
is a scope choice, not a wire-format limitation.

Preserve one-kernel-per-owner and per-algorithm/per-datatype sharding. Do not
add a monolithic CUDA TU, device link, or runtime device dispatcher.

### Host selection

Every rank must make the same choice from collective-uniform inputs. A split
decision uses incompatible packet layouts and protocol banks and will hang.

Add an explicit LL128 gate and kill switch, disabled by default. Future auto
selection should define:

```text
explicit protocol override
    > LL in its measured tiny-message range
    > LL128 in its measured medium-message range
    > Simple for larger messages
```

Measure crossover by algorithm, topology, datatype, and group width.

## Memory Ordering and Hardware Correctness

### Sender staging

```text
group.sync()
leader __threadfence_system()
leader posts RDMA WQE
```

This establishes visibility of encoded send staging to the NIC.

### Receiver ordered visibility

LL128 requires:

> If the GPU observes the final 8B flag of an aligned 128B line equal to the
> expected generation, the preceding 120B payload is already visible.

The writer is NIC DMA, so this is not ordinary CUDA 128B-store atomicity.

Current staging memory registration enables PCIe Relaxed Ordering, while
control-signal memory uses stricter ordering. LL128 embeds readiness inside
staging and cannot inherit the control-signal guarantee. One RDMA-write WQE
also does not by itself prove remote 128B atomicity or payload-before-flag
visibility.

Before enablement, choose and validate one approach:

1. Establish that the target GPU/NIC/Data Direct path supplies aligned 128B
   ordered visibility.
2. Register LL128 staging in a dedicated MR without Relaxed Ordering and
   validate the resulting behavior.
3. Disable LL128 on unsupported backends and fall back to LL or Simple.

A receiver acquire fence can order later GPU loads; it cannot repair a NIC
that made the flag visible first.

Additional invariants:

- payload and flag remain in one ordered write domain;
- a line never crosses a QP or transport fragment;
- flag and payload use volatile/non-caching loads;
- baseline blocking polling reloads the full slice after all flags match.

### Visibility stress test

Derive every payload word from generation, packet index, and word index. The
receiver reads payload only after observing the matching flag. Run many ring
wraps and detect mixed-generation or torn lines while sweeping:

- Relaxed Ordering enabled/disabled;
- supported NIC backends;
- multiple QP counts;
- line and chunk boundaries;
- blocking and progress paths.

The stress test is necessary but not a substitute for a backend contract.

## Fault Tolerance

All lanes participate in baseline wire loads. Flag lanes determine readiness
and perform poll-loop abort checks, but decisions are warp-uniform:

```text
ready / waiting / aborted
        -> ballot or __any_sync
        -> all 32 lanes take the same path
```

Across a multi-warp group:

- each warp may wait on its own slice;
- no lane exits before a barrier that still references it;
- a detecting warp updates shared abort state;
- other warps observe it on their next poll;
- every thread reaches final group synchronization;
- no downstream packet or false `SLOT_FREE` is published after abort.

`repack` and `repack_reduce` start only after the complete input chunk is
ready. A partially written downstream encoding cannot safely retry readiness.

## Implementation Plan

### Phase 0: visibility qualification

- Build a standalone LL128 line stress test.
- Test every target GPU/NIC/backend combination.
- Verify alignment, QP splitting, Relaxed Ordering, volatile loads, and
  generation semantics.
- Do not connect LL128 to AllReduce before resolving this gate.

### Phase 1: geometry and codec

- Add `Ll128PacketGeometry` and the specialized codec.
- Implement pack, flag probe, wait/ready unpack, reduction, repack, and
  repack-reduce.
- Add aligned fast and misaligned/tail paths.
- Add host/device geometry and codec tests.

### Phase 2: blocking transport

- Add `protocol::LL128` and a third protocol slot.
- Generalize blocking send/recv/forward seams.
- Fix 120B geometry and tail-padding assumptions.
- Enforce 128B staging and QP split alignment.
- Add transport benchmarks and wrap-generation tests.

### Phase 3: Ring AllReduce

- Add typed owners and host selection.
- Exercise 640/320/160/64-thread groups.
- Validate reduce-forward and all-gather forwarding.
- Compare with MCCL Simple, MCCL LL, and NCCL LL128.

### Phase 4: progress and Tree

- Generalize progress inline-flag seams.
- Connect `all_flags_set()` to ready-only consume functions.
- Add Tree typed owners and the all-peer IBGDA gate.
- Exercise dual-tree scheduling, 320-thread stripes, and abort behavior.

### Phase 5: tuning and expansion

- Set thresholds from measured data.
- Add remaining datatypes.
- Separately evaluate trace, warp proxy, registered sends, Direct, and
  packet-level progress.

## Validation Plan

### Codec and geometry

- Payload sizes 0, 1, 7, 8, 15, 16, 119, 120, 121, 127, 128, and multi-slice
  sizes.
- Packet count and payload/wire conversion.
- Flag round trip and individual packet corruption.
- Aligned and element-aligned buffer offsets.
- Group sizes 64, 160, 320, 480, and 640.
- Tail zero-fill with no out-of-bounds user access.
- fp16, bf16, fp32, fp64, and int64 reduction.
- Repack with a different downstream generation.
- Abort without further downstream writes.
- Non-power-of-two `kData=120`, true `lcm(120,8)`, slot boundaries, and tail
  cursor advancement.

### Transport

- Blocking send/recv and forward.
- Progress send/recv.
- Pipeline depths 1, 2, and 8.
- Repeated ring wraps and multi-QP boundaries.
- `Simple -> LL -> LL128 -> Simple` and
  `Simple -> LL128 -> LL -> Simple`.
- Shared `recvDataReadyLaneCursor` alignment.
- Lost WQE, missing flag, and communicator abort.

### AllReduce

- Ring and Tree.
- Mandatory `MCCL_MAX_NBLOCKS=1` coverage.
- Ring stripe counts 1, 2, 4, and 10.
- Tree stripe counts 1 and 2.
- IB_ONLY 1x2, 1x4, 1x7, and 1x8.
- HYBRID 2x4 and 2x8.
- In-place/out-of-place and buffer offsets.
- Initial datatypes and supported reduction operations.
- Explicitly disabled graph behavior until capture/replay is supported; do not
  silently downgrade the requested protocol.

### Performance

Record latency, algorithm bandwidth, codec cycles, WQE/completion time, poll
iterations, registers/thread, spills, shared memory, occupancy, and L2/global
transactions.

Compare:

```text
MCCL Simple
MCCL LL
MCCL LL128
NCCL LL128
```

Sweep message size, node count, `MCCL_MAX_NBLOCKS`, Ring/Tree, datatype,
`kWordsPerThread={2,4,8}`, and full-slice versus qualified flag-only polling.

Use `kWordsPerThread=8` as the primary implementation because it matches
NCCL's 2048B wire/warp slice. Keep 2 and 4 as tuning experiments. Final choice
depends on `ptxas` resources and end-to-end results.

## Success Criteria

1. No regressions in existing Simple or LL tests.
2. Correct results for every supported group width and alignment.
3. No mixed-generation lines in qualification stress tests.
4. No unacceptable spills or launch-resource failures at 640 threads.
5. LL128 beats LL in its measured range and beats Simple in at least one
   useful small-to-medium range.
6. Disabled LL128 adds no runtime branch or register cost to existing kernels.
7. Typed owners retain single kernel ownership with no device-link/LTO
   boundary.

## Risks and Open Questions

1. **Remote 128B visibility:** the primary enablement blocker.
2. **Relaxed Ordering:** determine whether LL128 needs a strict-order staging
   MR.
3. **Misalignment:** choose between approximately 40KiB of block scratch and a
   slower scalar fallback.
4. **Register pressure:** reduce-forward holds incoming and local register
   sets; register reuse may be needed.
5. **Polling:** full-slice polling is baseline; flag-only needs proof and data.
6. **Progress granularity:** whole-chunk readiness may lose packet streaming.
7. **Protocol selection:** crossover varies by algorithm, topology, datatype,
   and group width.
8. **Backend coverage:** NIC backends may need separate enablement policies.
9. **Build cost:** preserve per-algorithm/per-datatype CUDA sharding.

## Related Code

- `comms/mccl/collectives/allreduce/AllReduce.md`
- `comms/mccl/collectives/allreduce/AllReduceFusedTypes.h`
- `comms/mccl/collectives/allreduce/AllReduceFusedCommon.cuh`
- `comms/mccl/collectives/allreduce/AllReduceIbRingImpl.cuh`
- `comms/mccl/collectives/allreduce/AllReduceIbTreeImpl.cuh`
- `comms/prims/core/LlxPacket.cuh`
- `comms/prims/core/LLImpl.cuh`
- `comms/prims/core/MemcpyCopyOp.cuh`
- `comms/prims/transport/P2pIbTransportDeviceImpl.cuh`
- `comms/prims/transport/P2pIbTransportProgressImpl.cuh`
- `comms/prims/transport/ibgda/IbgdaBuffer.h`
- `comms/prims/transport/MultiPeerIbTransport.cc`
- `comms/ncclx/v2_31/src/device/prims_ll128.h`
- `comms/ncclx/v2_31/src/include/device.h`
- `comms/ncclx/v2_31/src/transport/net_ib/p2p.cc`
