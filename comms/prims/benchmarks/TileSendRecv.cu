// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Tile send/recv kernels — caller partitions data across blocks,
// each block calls P2pNvlTransportDevice::send/recv.

#include "comms/prims/benchmarks/TileSendRecv.cuh"

namespace comms::prims::benchmark {

__global__ __launch_bounds__(512, 1) void p2pTileSendRecv(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    std::size_t max_signal_bytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  auto [role, sub] = group.partition(2);

  const int blockId = sub.group_id;

  if (role == 0) {
    p2p.send(
        sub,
        sendTiles.tile_data(blockId),
        sendTiles.tile_bytes(blockId),
        max_signal_bytes,
        timeout);
  } else {
    p2p.recv(
        sub,
        recvTiles.tile_data(blockId),
        recvTiles.tile_bytes(blockId),
        max_signal_bytes,
        timeout);
  }
}

// =============================================================================
// Dynamic block count variant — uses transport-internal tile state
// =============================================================================
//
// Requires maxNumChannels > 0 and p2pBarrierCount >= maxNumChannels in
// transport config.
//
// DYNAMIC BLOCK COUNT: BARRIER CORRECTNESS
// =========================================
// When numBlocks changes between kernel launches, the staging buffer
// layout shifts (perBlockSlotSize = dataBufferSize / numBlocks). This
// creates a cross-GPU race: the new sender on GPU A may overwrite
// staging positions that the old receiver on GPU B is still reading.
//
// The per-block barrier_sync prevents this race:
//
//   Stream ordering guarantee:
//     Both kernels execute on the same CUDA stream per GPU. So on each
//     GPU individually, kernel N completes before kernel N+1 starts.
//     But GPU A's kernel N+1 can start while GPU B's kernel N is still
//     running (no cross-GPU stream ordering).
//
//   What the barrier provides:
//     Each block in the NEW kernel barriers with its same-numbered peer
//     block on the remote GPU. Since the peer block can only reach the
//     barrier AFTER its kernel N+1 starts, and kernel N+1 can only start
//     after kernel N completed (stream ordering), the barrier guarantees
//     that ALL of the peer's kernel N work is done — including reads
//     from the staging buffer.
//
//   Higher → Lower (e.g., 16 → 8 blocks):
//     Only blocks 0-7 are launched. Blocks 8-15 don't barrier, but
//     that's safe: stream ordering on the peer ensures kernel N (which
//     used blocks 8-15) completed before kernel N+1 started on that GPU.
//     Barrier counters for blocks 8-15 remain consistent because BOTH
//     GPUs skip them (same kernel launch parameters on both sides).
//
//   Lower → Higher (e.g., 8 → 16 blocks):
//     Blocks 8-15 are new on both GPUs. Both sides launch them, so both
//     call barrier(8..15). The barrier counters may have stale values
//     from earlier kernels, but they're symmetric (both sides did the
//     same number of arrive/wait cycles), so the monotonic arrive/wait
//     succeeds correctly.
//
//   Why the receiver doesn't need protection:
//     The new receiver waits for TAIL signals from the new sender before
//     reading. The old sender on the remote GPU completed (stream
//     ordering). So the receiver never reads stale data.
//
// CUDA graph compatible: all device-side (no host synchronization).

__global__ __launch_bounds__(512, 1) void p2pTileSendRecvDynamic(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    bool needsBarrier,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  auto [role, sub] = group.partition(2);
  const int blockId = sub.group_id;

  if (needsBarrier) {
    p2p.barrier_sync(sub, blockId, timeout);
  }

  if (role == 0) {
    p2p.send(
        sub,
        sendTiles.tile_data(blockId),
        sendTiles.tile_bytes(blockId),
        /*max_signal_bytes=*/0,
        timeout);
  } else {
    p2p.recv(
        sub,
        recvTiles.tile_data(blockId),
        recvTiles.tile_bytes(blockId),
        /*max_signal_bytes=*/0,
        timeout);
  }
}

// =============================================================================
// Tile-style fused recv+forward kernel.
// =============================================================================
//
// Each block calls P2pNvlTransportDevice::forward(), which fuses a recv from
// the predecessor staging buffer with a send to the successor staging buffer
// (plus a local copy to the user dst). All blocks run forward (no role
// partition); block group_id ∈ [0, numBlocks) drives the per-block tile
// assignment and signal slots.
//
// In a 2-rank ring test (rank 0 ↔ rank 1):
//   - Rank 0 launches p2pTileSendRecv (sends src → rank 1 staging,
//     receives forwarded data ← rank 1 staging).
//   - Rank 1 launches p2pTileForward with p2p_pred == p2p_succ ==
//     the single transport to rank 0; forward reads from rank 0's send
//     into local staging and writes to rank 0's recv staging.
//
// Signal slots are paired correctly because forward.recv uses [tail=i,
// head=i+max_groups] on this transport, while forward.send uses
// [tail=i, head=i+max_groups] on successor transport. When this ==
// successor, the recv-side and send-side touch DIFFERENT halves of the
// signal/step arrays (recv → step_state[max_groups+i]; send →
// step_state[i]). Rank 0's send/recv similarly use distinct halves on
// the same transport.

// =============================================================================
// Bidir-CTA variant — one block does BOTH send and recv via multiwarp groups.
// =============================================================================
//
// Each block creates two half-block multiwarp groups and interleaves them:
//   - role 0 (first half-block): cooperative send on this block's channel
//   - role 1 (second half-block): cooperative recv on the same channel
//
// make_multiwarp_group(blockDim.x / 2) gives two groups per CTA. Then
// partition_interleaved(2) maps group 0 to role 0 / channel k and group 1
// to role 1 / channel k, so both halves use the same channel index. Send and
// recv touch INDEPENDENT signals (remote_ch.data_ready vs
// remote_ch.slot_free)
// and different memory regions (remote staging vs local staging), so the same
// channel handles bidir traffic without conflict.
//
// Compared to the 2-role partition() variant (`p2pTileSendRecv`), this
// kernel uses HALF the blocks (gridDim = numSendBlocks instead of
// 2 * numSendBlocks) but each block has HALF the threads per role, so
// per-tile threadcount is the same. The point is to test whether fewer
// CTAs with thread-level role split gives better NVLink BW on H100 (per
// the redesign brainstorm — see ~/gdrive/workstreams/ctranfoundation/
// channels/design.md).

__global__ __launch_bounds__(512, 1) void p2pTileSendRecvBidirCta(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    std::size_t max_signal_bytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_multiwarp_group(blockDim.x / 2);
  auto [role, sub] = group.partition_interleaved(2);

  const int blockId = sub.group_id;

  if (role == 0) {
    p2p.send(
        sub,
        sendTiles.tile_data(blockId),
        sendTiles.tile_bytes(blockId),
        max_signal_bytes,
        timeout);
  } else {
    p2p.recv(
        sub,
        recvTiles.tile_data(blockId),
        recvTiles.tile_bytes(blockId),
        max_signal_bytes,
        timeout);
  }
}

__global__ __launch_bounds__(512, 1) void p2pTileForward(
    P2pNvlTransportDevice p2p_pred,
    P2pNvlTransportDevice p2p_succ,
    TiledBuffer<char> dstTiles,
    std::size_t max_signal_bytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;

  p2p_pred.forward(
      group,
      dstTiles.tile_data(blockId),
      dstTiles.tile_bytes(blockId),
      p2p_succ,
      max_signal_bytes,
      timeout);
}

} // namespace comms::prims::benchmark
