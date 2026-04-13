// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Tile send/recv kernels — caller partitions data across blocks,
// each block calls P2pNvlTransportDevice::send_tile/recv_tile.

#include "comms/pipes/benchmarks/TileSendRecv.cuh"

namespace comms::pipes::benchmark {

__global__ __launch_bounds__(512, 1) void p2pTileSendRecv(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int numBlocks,
    int chunksPerSlot,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  auto [role, sub] = group.partition(2);

  const int blockId = sub.group_id;

  if (role == 0) {
    p2p.send_tile(
        sub,
        sendTiles.tile_data(blockId),
        sendTiles.tile_bytes(blockId),
        numBlocks,
        timeout,
        chunksPerSlot);
  } else {
    p2p.recv_tile(
        sub,
        recvTiles.tile_data(blockId),
        recvTiles.tile_bytes(blockId),
        numBlocks,
        timeout,
        chunksPerSlot);
  }
}

// =============================================================================
// Dynamic block count variant — uses transport-internal tile state
// =============================================================================
//
// Requires tileMaxBlocks > 0 and p2pBarrierCount >= tileMaxBlocks in
// transport config.
//
// DYNAMIC BLOCK COUNT: BARRIER CORRECTNESS
// =========================================
// When numBlocks changes between kernel launches, the staging buffer
// layout shifts (perBlockSlotSize = dataBufferSize / numBlocks). This
// creates a cross-GPU race: the new sender on GPU A may overwrite
// staging positions that the old receiver on GPU B is still reading.
//
// The per-block barrier_sync_threadgroup prevents this race:
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
    int numBlocks,
    bool needsBarrier,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  auto [role, sub] = group.partition(2);
  const int blockId = sub.group_id;

  // If block count changed, each block barriers with its peer.
  // Since kernels are on the same stream, the peer's current kernel can't
  // start until its previous kernel finished. So when any peer block
  // reaches the barrier, ALL of the peer's old-round work is done.
  // Each block uses its own barrier slot — all barriers complete in parallel.
  // Requires p2pBarrierCount >= tileMaxBlocks.
  if (needsBarrier) {
    p2p.barrier_sync_threadgroup(sub, blockId, timeout);
  }

  // Uses transport-internal tile signals, stepState, and maxBlocks
  if (role == 0) {
    p2p.send_tile(
        sub,
        sendTiles.tile_data(blockId),
        sendTiles.tile_bytes(blockId),
        numBlocks,
        timeout);
  } else {
    p2p.recv_tile(
        sub,
        recvTiles.tile_data(blockId),
        recvTiles.tile_bytes(blockId),
        numBlocks,
        timeout);
  }
}

} // namespace comms::pipes::benchmark
