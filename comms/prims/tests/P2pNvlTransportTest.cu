// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/tests/Checks.h"
#include "comms/prims/tests/P2pNvlTransportTest.cuh"

#include "comms/prims/core/TiledBuffer.cuh"

namespace comms::prims::test {

// Helper to create the appropriate thread group based on type
__device__ inline ThreadGroup make_group(GroupType groupType) {
  switch (groupType) {
    case GroupType::WARP:
      return make_warp_group();
    case GroupType::BLOCK:
      return make_block_group();
    default:
      return make_warp_group();
  }
}

__global__ void testSendKernel(
    P2pNvlTransportDevice* p2p,
    void* src_d,
    size_t nbytes,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p->send_group(group, src_d, nbytes);
}

__global__ void testRecvKernel(
    P2pNvlTransportDevice* p2p,
    void* dst_d,
    size_t nbytes,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p->recv_group(group, dst_d, nbytes);
}

__global__ void testTileSendKernel(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    int activeBlocks,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();
  auto group = make_block_group();
  TiledBuffer<char> tiles(reinterpret_cast<char*>(src_d), nbytes, group);
  p2p.send(
      group,
      tiles.data(),
      tiles.bytes(),
      activeBlocks,
      maxSignalBytes,
      timeout);
}

__global__ void testTileRecvKernel(
    P2pNvlTransportDevice p2p,
    void* dst_d,
    size_t nbytes,
    int activeBlocks,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();
  auto group = make_block_group();
  TiledBuffer<char> tiles(reinterpret_cast<char*>(dst_d), nbytes, group);
  p2p.recv(
      group,
      tiles.data(),
      tiles.bytes(),
      activeBlocks,
      maxSignalBytes,
      timeout);
}

// Kernel that performs multiple sequential sends within a single kernel launch
__global__ void testMultiSendKernel(
    P2pNvlTransportDevice* p2p,
    void* src_d,
    size_t nbytes,
    int numSends,
    GroupType groupType) {
  auto group = make_group(groupType);
  char* src = reinterpret_cast<char*>(src_d);
  for (int i = 0; i < numSends; i++) {
    p2p->send_group(group, src + i * nbytes, nbytes);
  }
}

// Kernel that performs multiple sequential recvs within a single kernel launch
__global__ void testMultiRecvKernel(
    P2pNvlTransportDevice* p2p,
    void* dst_d,
    size_t nbytes,
    int numRecvs,
    GroupType groupType) {
  auto group = make_group(groupType);
  char* dst = reinterpret_cast<char*>(dst_d);
  for (int i = 0; i < numRecvs; i++) {
    p2p->recv_group(group, dst + i * nbytes, nbytes);
  }
}

// Kernel that performs both send and recv within a single kernel launch
// Used for pipelined bidirectional communication
__global__ void testSendRecvKernel(
    P2pNvlTransportDevice* p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p->send_group(group, send_d, nbytes);
  p2p->recv_group(group, recv_d, nbytes);
}

// Kernel that performs recv then send within a single kernel launch
// Paired with testSendRecvKernel for bidirectional tests
__global__ void testRecvSendKernel(
    P2pNvlTransportDevice* p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p->recv_group(group, recv_d, nbytes);
  p2p->send_group(group, send_d, nbytes);
}

// Kernel that performs weighted partition send/recv
// Groups are partitioned according to weights, partition 0 sends, partition 1
// recvs
__global__ void testWeightedSendRecvKernel(
    P2pNvlTransportDevice* p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    uint32_t sendWeight,
    uint32_t recvWeight,
    GroupType groupType) {
  auto group = make_group(groupType);
  uint32_t weights[] = {sendWeight, recvWeight};
  auto [partition_id, subgroup] = group.partition(make_device_span(weights, 2));
  if (partition_id == 0) {
    p2p->send_group(subgroup, send_d, nbytes);
  } else {
    p2p->recv_group(subgroup, recv_d, nbytes);
  }
}

// Kernel that performs weighted partition recv/send
// Groups are partitioned according to weights, partition 0 recvs, partition 1
// sends
__global__ void testWeightedRecvSendKernel(
    P2pNvlTransportDevice* p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    uint32_t recvWeight,
    uint32_t sendWeight,
    GroupType groupType) {
  auto group = make_group(groupType);
  uint32_t weights[] = {recvWeight, sendWeight};
  auto [partition_id, subgroup] = group.partition(make_device_span(weights, 2));
  if (partition_id == 0) {
    p2p->recv_group(subgroup, recv_d, nbytes);
  } else {
    p2p->send_group(subgroup, send_d, nbytes);
  }
}

__device__ void wait_for_second_call_signal(
    P2pNvlTransportDevice& p2p,
    ThreadGroup& group,
    int blockId,
    int activeBlocks,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool enabled,
    const Timeout& timeout) {
  if (!enabled) {
    return;
  }
  const size_t perBlockSlotSize =
      (p2p.options().dataBufferSize / activeBlocks) & ~15ULL;
  const size_t chunkSize =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlotSize
      ? (maxSignalBytes & ~15ULL)
      : perBlockSlotSize;
  const size_t effectiveChunk = chunkSize > 0 ? chunkSize : perBlockSlotSize;
  const uint64_t secondCallStarted = bytesPerCall +
      (bytesPerCall < effectiveChunk ? bytesPerCall : effectiveChunk);
  p2p.tile_state().local_signals[blockId].wait_until(
      group, CmpOp::CMP_GE, secondCallStarted, timeout);
}

__global__ void testTileMultiCallSendRecvKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool waitForSecondCallSignal,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  auto [role, sub] = group.partition(2);
  const int blockId = sub.group_id;

  if (role == 0) {
    char* sendTile = sendTiles.tile_data(blockId);
    for (int i = 0; i < numCalls; ++i) {
      p2p.send(
          sub,
          sendTile + i * bytesPerCall,
          bytesPerCall,
          activeBlocks,
          maxSignalBytes,
          timeout);
    }
  } else {
    wait_for_second_call_signal(
        p2p,
        sub,
        blockId,
        activeBlocks,
        bytesPerCall,
        maxSignalBytes,
        waitForSecondCallSignal,
        timeout);
    char* recvTile = recvTiles.tile_data(blockId);
    for (int i = 0; i < numCalls; ++i) {
      p2p.recv(
          sub,
          recvTile + i * bytesPerCall,
          bytesPerCall,
          activeBlocks,
          maxSignalBytes,
          timeout);
    }
  }
}

__global__ void testTileTwoCallVariableSignalSendRecvKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t firstMaxSignalBytes,
    size_t secondMaxSignalBytes,
    bool waitForSecondCallSignal,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  auto [role, sub] = group.partition(2);
  const int blockId = sub.group_id;

  if (role == 0) {
    char* sendTile = sendTiles.tile_data(blockId);
    p2p.send(
        sub,
        sendTile,
        firstCallBytes,
        activeBlocks,
        firstMaxSignalBytes,
        timeout);
    p2p.send(
        sub,
        sendTile + firstCallBytes,
        secondCallBytes,
        activeBlocks,
        secondMaxSignalBytes,
        timeout);
  } else {
    wait_for_second_call_signal(
        p2p,
        sub,
        blockId,
        activeBlocks,
        firstCallBytes,
        secondMaxSignalBytes,
        waitForSecondCallSignal,
        timeout);
    char* recvTile = recvTiles.tile_data(blockId);
    p2p.recv(
        sub,
        recvTile,
        firstCallBytes,
        activeBlocks,
        firstMaxSignalBytes,
        timeout);
    p2p.recv(
        sub,
        recvTile + firstCallBytes,
        secondCallBytes,
        activeBlocks,
        secondMaxSignalBytes,
        timeout);
  }
}

__global__ void testTileMultiCallSendOnlyKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* sendTile = sendTiles.tile_data(blockId);
  for (int i = 0; i < numCalls; ++i) {
    p2p.send(
        group,
        sendTile + i * bytesPerCall,
        bytesPerCall,
        activeBlocks,
        maxSignalBytes,
        timeout);
  }
}

__global__ void testTileTwoCallSendOnlyKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* sendTile = sendTiles.tile_data(blockId);
  p2p.send(
      group, sendTile, firstCallBytes, activeBlocks, maxSignalBytes, timeout);
  p2p.send(
      group,
      sendTile + firstCallBytes,
      secondCallBytes,
      activeBlocks,
      maxSignalBytes,
      timeout);
}

__device__ void check_wrapped_substep_with_existing_signals(
    P2pNvlTransportDevice& p2p,
    ThreadGroup& group,
    int activeBlocks,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    const Timeout& timeout) {
  const size_t slotSize = p2p.options().dataBufferSize;
  const size_t perBlockSlotSize = (slotSize / activeBlocks) & ~15ULL;
  const size_t effectiveChunk =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlotSize
      ? (maxSignalBytes & ~15ULL)
      : perBlockSlotSize;
  const size_t pipelineBytes = perBlockSlotSize * p2p.options().pipelineDepth;
  const uint64_t streamStart =
      static_cast<uint64_t>(p2p.tile_state().step_state[0]);
  const uint64_t firstStreamEnd = streamStart + effectiveChunk;
  const uint64_t firstAckValue = firstStreamEnd - pipelineBytes;
  const uint64_t targetStreamStart = firstStreamEnd;
  const size_t targetPipelineOff =
      static_cast<size_t>(targetStreamStart % pipelineBytes);
  const size_t targetSlot = targetPipelineOff / perBlockSlotSize;
  const size_t targetChunkOff =
      targetPipelineOff - targetSlot * perBlockSlotSize;
  const size_t targetOffset = targetSlot * slotSize + targetChunkOff;
  const uint64_t targetStreamEnd = targetStreamStart + effectiveChunk;
  const uint64_t targetAckValue = targetStreamEnd - pipelineBytes;
  const int tailSignalId = 0;
  const int headSignalId = p2p.tile_state().tile_max_groups;

  // Drive the existing head signal to the minimum threshold that releases only
  // the first wrapped chunk. Real recv() may coalesce head updates at slot
  // boundaries; this test is intentionally isolating the sender/forwarder wait
  // predicate for the following nonzero wrapped substep.
  p2p.tile_state().local_signals[headSignalId].signal(
      group, SignalOp::SIGNAL_SET, firstAckValue);
  p2p.tile_state().remote_signals[tailSignalId].wait_until(
      group, CmpOp::CMP_GE, firstStreamEnd, timeout);

  if (group.is_leader()) {
    const auto observed =
        static_cast<unsigned char>(p2p.remote_state().dataBuffer[targetOffset]);
    *observedEarlyOverwrite = observed == sentinel ? 0 : 1;
    p2p.tile_state().local_signals[headSignalId].signal(
        SignalOp::SIGNAL_SET, targetAckValue);
  }
}

__global__ void testTileSendWaitsForWrappedSubstepAckKernel(
    P2pNvlTransportDevice p2p,
    const char* sendData,
    int activeBlocks,
    size_t nbytes,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  if (blockIdx.x == 0) {
    p2p.send(group, sendData, nbytes, activeBlocks, maxSignalBytes, timeout);
  } else {
    check_wrapped_substep_with_existing_signals(
        p2p,
        group,
        activeBlocks,
        maxSignalBytes,
        sentinel,
        observedEarlyOverwrite,
        timeout);
  }
}

__global__ void testTileForwardWaitsForWrappedSubstepAckKernel(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    char* dst,
    int activeBlocks,
    size_t nbytes,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  if (blockIdx.x == 0) {
    pred.forward(
        group, dst, nbytes, succ, activeBlocks, maxSignalBytes, timeout);
  } else {
    check_wrapped_substep_with_existing_signals(
        succ,
        group,
        activeBlocks,
        maxSignalBytes,
        sentinel,
        observedEarlyOverwrite,
        timeout);
  }
}

__global__ void testPrepareTileStagingKernel(
    P2pNvlTransportDevice p2p,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    int sourceRank,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* staging = p2p.local_state().dataBuffer;

  const size_t slotSize = p2p.options().dataBufferSize;
  const size_t perBlockSlotSize = (slotSize / activeBlocks) & ~15ULL;
  const size_t chunkSize =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlotSize
      ? (maxSignalBytes & ~15ULL)
      : perBlockSlotSize;
  const size_t effectiveChunk = chunkSize > 0 ? chunkSize : perBlockSlotSize;
  const size_t pipelineBytes = perBlockSlotSize * p2p.options().pipelineDepth;
  const size_t stagingOff = blockId * perBlockSlotSize;

  uint64_t baseByte = 0;
  for (int call = 0; call < numCalls; ++call) {
    const char pattern = static_cast<char>(0x30 + sourceRank * 0x20 + call);
    for (size_t dataOff = 0; dataOff < bytesPerCall;) {
      const uint64_t streamStart = baseByte + dataOff;
      const size_t pipelineOff =
          static_cast<size_t>(streamStart % pipelineBytes);
      const size_t slot = pipelineOff / perBlockSlotSize;
      const size_t chunkOff = pipelineOff - slot * perBlockSlotSize;
      const size_t slotRemaining = perBlockSlotSize - chunkOff;
      const size_t dataRemaining = bytesPerCall - dataOff;
      size_t copyBytes =
          effectiveChunk < dataRemaining ? effectiveChunk : dataRemaining;
      copyBytes = copyBytes < slotRemaining ? copyBytes : slotRemaining;
      const size_t bufferOff = slot * slotSize + stagingOff + chunkOff;
      for (size_t idx = group.thread_id_in_group; idx < copyBytes;
           idx += group.group_size) {
        staging[bufferOff + idx] = pattern;
      }
      dataOff += copyBytes;
    }
    baseByte += bytesPerCall;
  }

  group.sync();
  if (group.is_leader()) {
    p2p.tile_state().local_signals[blockId].signal(
        SignalOp::SIGNAL_SET, baseByte);
  }
}

__global__ void testPrepareTileTwoCallStagingKernel(
    P2pNvlTransportDevice p2p,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int sourceRank,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* staging = p2p.local_state().dataBuffer;

  const size_t slotSize = p2p.options().dataBufferSize;
  const size_t perBlockSlotSize = (slotSize / activeBlocks) & ~15ULL;
  const size_t effectiveChunk =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlotSize
      ? (maxSignalBytes & ~15ULL)
      : perBlockSlotSize;
  const size_t pipelineBytes = perBlockSlotSize * p2p.options().pipelineDepth;
  const size_t stagingOff = blockId * perBlockSlotSize;

  uint64_t baseByte = 0;
  for (int call = 0; call < 2; ++call) {
    const size_t callBytes = call == 0 ? firstCallBytes : secondCallBytes;
    const char pattern = static_cast<char>(0x30 + sourceRank * 0x20 + call);
    for (size_t dataOff = 0; dataOff < callBytes;) {
      const uint64_t streamStart = baseByte + dataOff;
      const size_t pipelineOff =
          static_cast<size_t>(streamStart % pipelineBytes);
      const size_t slot = pipelineOff / perBlockSlotSize;
      const size_t chunkOff = pipelineOff - slot * perBlockSlotSize;
      const size_t slotRemaining = perBlockSlotSize - chunkOff;
      const size_t dataRemaining = callBytes - dataOff;
      size_t copyBytes =
          effectiveChunk < dataRemaining ? effectiveChunk : dataRemaining;
      copyBytes = copyBytes < slotRemaining ? copyBytes : slotRemaining;
      const size_t bufferOff = slot * slotSize + stagingOff + chunkOff;
      for (size_t idx = group.thread_id_in_group; idx < copyBytes;
           idx += group.group_size) {
        staging[bufferOff + idx] = pattern;
      }
      dataOff += copyBytes;
    }
    baseByte += callBytes;
  }

  group.sync();
  if (group.is_leader()) {
    p2p.tile_state().local_signals[blockId].signal(
        SignalOp::SIGNAL_SET, baseByte);
  }
}

__global__ void testTileMultiCallRecvOnlyKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* recvTile = recvTiles.tile_data(blockId);
  for (int i = 0; i < numCalls; ++i) {
    p2p.recv(
        group,
        recvTile + i * bytesPerCall,
        bytesPerCall,
        activeBlocks,
        maxSignalBytes,
        timeout);
  }
}

__global__ void testTileTwoCallRecvOnlyKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* recvTile = recvTiles.tile_data(blockId);
  p2p.recv(
      group, recvTile, firstCallBytes, activeBlocks, maxSignalBytes, timeout);
  p2p.recv(
      group,
      recvTile + firstCallBytes,
      secondCallBytes,
      activeBlocks,
      maxSignalBytes,
      timeout);
}

__global__ void testTileMultiCallForwardKernel(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool waitForSecondCallSignal,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  wait_for_second_call_signal(
      pred,
      group,
      blockId,
      activeBlocks,
      bytesPerCall,
      maxSignalBytes,
      waitForSecondCallSignal,
      timeout);

  char* dstTile = dstTiles.tile_data(blockId);
  for (int i = 0; i < numCalls; ++i) {
    pred.forward(
        group,
        dstTile + i * bytesPerCall,
        bytesPerCall,
        succ,
        activeBlocks,
        maxSignalBytes,
        timeout);
  }
}

__global__ void testTileTwoCallForwardKernel(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* dstTile = dstTiles.tile_data(blockId);
  pred.forward(
      group,
      dstTile,
      firstCallBytes,
      succ,
      activeBlocks,
      maxSignalBytes,
      timeout);
  pred.forward(
      group,
      dstTile + firstCallBytes,
      secondCallBytes,
      succ,
      activeBlocks,
      maxSignalBytes,
      timeout);
}

__global__ void testTileTwoCallVariableSignalForwardKernel(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t firstMaxSignalBytes,
    size_t secondMaxSignalBytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* dstTile = dstTiles.tile_data(blockId);
  pred.forward(
      group,
      dstTile,
      firstCallBytes,
      succ,
      activeBlocks,
      firstMaxSignalBytes,
      timeout);
  pred.forward(
      group,
      dstTile + firstCallBytes,
      secondCallBytes,
      succ,
      activeBlocks,
      secondMaxSignalBytes,
      timeout);
}

__global__ void testCopyLocalStagingKernel(
    P2pNvlTransportDevice p2p,
    void* dst,
    size_t nbytes) {
  auto group = make_block_group();
  memcpy_vectorized(
      static_cast<char*>(dst), p2p.local_state().dataBuffer, nbytes, group);
}

void testSend(
    P2pNvlTransportDevice* p2p,
    void* src_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    int /*blocksPerGroup*/,
    cudaStream_t stream) {
  testSendKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p, src_d, nbytes, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testRecv(
    P2pNvlTransportDevice* p2p,
    void* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    int /*blocksPerGroup*/,
    cudaStream_t stream) {
  testRecvKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p, dst_d, nbytes, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileSend(
    const P2pNvlTransportDevice& p2p,
    void* src_d,
    size_t nbytes,
    int activeBlocks,
    size_t maxSignalBytes,
    Timeout timeout,
    int numBlocks,
    int blockSize,
    cudaStream_t stream) {
  testTileSendKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p, src_d, nbytes, activeBlocks, maxSignalBytes, timeout);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileRecv(
    const P2pNvlTransportDevice& p2p,
    void* dst_d,
    size_t nbytes,
    int activeBlocks,
    size_t maxSignalBytes,
    Timeout timeout,
    int numBlocks,
    int blockSize,
    cudaStream_t stream) {
  testTileRecvKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p, dst_d, nbytes, activeBlocks, maxSignalBytes, timeout);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testMultiSend(
    P2pNvlTransportDevice* p2p,
    void* src_d,
    size_t nbytes,
    int numSends,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    int /*blocksPerGroup*/) {
  testMultiSendKernel<<<numBlocks, blockSize>>>(
      p2p, src_d, nbytes, numSends, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testMultiRecv(
    P2pNvlTransportDevice* p2p,
    void* dst_d,
    size_t nbytes,
    int numRecvs,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    int /*blocksPerGroup*/) {
  testMultiRecvKernel<<<numBlocks, blockSize>>>(
      p2p, dst_d, nbytes, numRecvs, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileMultiCallSendRecv(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool waitForSecondCallSignal,
    int blockSize,
    cudaStream_t stream) {
  testTileMultiCallSendRecvKernel<<<activeBlocks * 2, blockSize, 0, stream>>>(
      p2p,
      sendTiles,
      recvTiles,
      activeBlocks,
      numCalls,
      bytesPerCall,
      maxSignalBytes,
      waitForSecondCallSignal,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileTwoCallVariableSignalSendRecv(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t firstMaxSignalBytes,
    size_t secondMaxSignalBytes,
    bool waitForSecondCallSignal,
    int blockSize,
    Timeout timeout,
    cudaStream_t stream) {
  testTileTwoCallVariableSignalSendRecvKernel<<<
      activeBlocks * 2,
      blockSize,
      0,
      stream>>>(
      p2p,
      sendTiles,
      recvTiles,
      activeBlocks,
      firstCallBytes,
      secondCallBytes,
      firstMaxSignalBytes,
      secondMaxSignalBytes,
      waitForSecondCallSignal,
      timeout);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileMultiCallSendOnly(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream) {
  testTileMultiCallSendOnlyKernel<<<activeBlocks, blockSize, 0, stream>>>(
      p2p,
      sendTiles,
      activeBlocks,
      numCalls,
      bytesPerCall,
      maxSignalBytes,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileTwoCallSendOnly(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream) {
  testTileTwoCallSendOnlyKernel<<<activeBlocks, blockSize, 0, stream>>>(
      p2p,
      sendTiles,
      activeBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileSendWaitsForWrappedSubstepAck(
    P2pNvlTransportDevice p2p,
    const char* sendData,
    int activeBlocks,
    size_t nbytes,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    int blockSize,
    cudaStream_t stream) {
  testTileSendWaitsForWrappedSubstepAckKernel<<<2, blockSize, 0, stream>>>(
      p2p,
      sendData,
      activeBlocks,
      nbytes,
      maxSignalBytes,
      sentinel,
      observedEarlyOverwrite,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileForwardWaitsForWrappedSubstepAck(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    char* dst,
    int activeBlocks,
    size_t nbytes,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    int blockSize,
    cudaStream_t stream) {
  testTileForwardWaitsForWrappedSubstepAckKernel<<<2, blockSize, 0, stream>>>(
      pred,
      succ,
      dst,
      activeBlocks,
      nbytes,
      maxSignalBytes,
      sentinel,
      observedEarlyOverwrite,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testPrepareTileStaging(
    P2pNvlTransportDevice p2p,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    int sourceRank,
    int blockSize,
    cudaStream_t stream) {
  testPrepareTileStagingKernel<<<activeBlocks, blockSize, 0, stream>>>(
      p2p,
      activeBlocks,
      numCalls,
      bytesPerCall,
      maxSignalBytes,
      sourceRank,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testPrepareTileTwoCallStaging(
    P2pNvlTransportDevice p2p,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int sourceRank,
    int blockSize,
    cudaStream_t stream) {
  testPrepareTileTwoCallStagingKernel<<<activeBlocks, blockSize, 0, stream>>>(
      p2p,
      activeBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      sourceRank,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileMultiCallRecvOnly(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream) {
  testTileMultiCallRecvOnlyKernel<<<activeBlocks, blockSize, 0, stream>>>(
      p2p,
      recvTiles,
      activeBlocks,
      numCalls,
      bytesPerCall,
      maxSignalBytes,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileTwoCallRecvOnly(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream) {
  testTileTwoCallRecvOnlyKernel<<<activeBlocks, blockSize, 0, stream>>>(
      p2p,
      recvTiles,
      activeBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileMultiCallForward(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool waitForSecondCallSignal,
    int blockSize,
    cudaStream_t stream) {
  testTileMultiCallForwardKernel<<<activeBlocks, blockSize, 0, stream>>>(
      pred,
      succ,
      dstTiles,
      activeBlocks,
      numCalls,
      bytesPerCall,
      maxSignalBytes,
      waitForSecondCallSignal,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileTwoCallForward(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream) {
  testTileTwoCallForwardKernel<<<activeBlocks, blockSize, 0, stream>>>(
      pred,
      succ,
      dstTiles,
      activeBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileTwoCallVariableSignalForward(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t firstMaxSignalBytes,
    size_t secondMaxSignalBytes,
    int blockSize,
    cudaStream_t stream) {
  testTileTwoCallVariableSignalForwardKernel<<<
      activeBlocks,
      blockSize,
      0,
      stream>>>(
      pred,
      succ,
      dstTiles,
      activeBlocks,
      firstCallBytes,
      secondCallBytes,
      firstMaxSignalBytes,
      secondMaxSignalBytes,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testCopyLocalStaging(
    P2pNvlTransportDevice p2p,
    void* dst,
    size_t nbytes,
    int blockSize,
    cudaStream_t stream) {
  testCopyLocalStagingKernel<<<1, blockSize, 0, stream>>>(p2p, dst, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testSendRecv(
    P2pNvlTransportDevice* p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    int /*blocksPerGroup*/) {
  testSendRecvKernel<<<numBlocks, blockSize>>>(
      p2p, send_d, recv_d, nbytes, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testRecvSend(
    P2pNvlTransportDevice* p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    int /*blocksPerGroup*/) {
  testRecvSendKernel<<<numBlocks, blockSize>>>(
      p2p, recv_d, send_d, nbytes, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testWeightedSendRecv(
    P2pNvlTransportDevice* p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t sendWeight,
    uint32_t recvWeight,
    GroupType groupType) {
  testWeightedSendRecvKernel<<<numBlocks, blockSize>>>(
      p2p, send_d, recv_d, nbytes, sendWeight, recvWeight, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testWeightedRecvSend(
    P2pNvlTransportDevice* p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t recvWeight,
    uint32_t sendWeight,
    GroupType groupType) {
  testWeightedRecvSendKernel<<<numBlocks, blockSize>>>(
      p2p, recv_d, send_d, nbytes, recvWeight, sendWeight, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// forward_group() test kernel and wrapper
// =============================================================================

__global__ void testForwardKernel(
    P2pNvlTransportDevice* pred,
    P2pNvlTransportDevice* succ,
    void* dst_d,
    size_t nbytes,
    GroupType groupType) {
  auto group = make_group(groupType);
  pred->forward_group(group, dst_d, nbytes, *succ);
}

void testForward(
    P2pNvlTransportDevice* pred,
    P2pNvlTransportDevice* succ,
    void* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    cudaStream_t stream) {
  testForwardKernel<<<numBlocks, blockSize, 0, stream>>>(
      pred, succ, dst_d, nbytes, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// write() test kernel and wrapper
// =============================================================================

__global__ void testPutWithSignalKernel(
    P2pNvlTransportDevice* p2p,
    char* dst_d,
    const char* src_d,
    uint64_t signal_id,
    size_t nbytes,
    GroupType groupType) {
  auto group = make_group(groupType);
  auto writtenBytes = p2p->put_group(group, dst_d, src_d, nbytes);
  p2p->signal(group, signal_id, SignalOp::SIGNAL_ADD, writtenBytes);
}

void testPutWithSignal(
    P2pNvlTransportDevice* p2p,
    char* dst_d,
    const char* src_d,
    uint64_t signal_id,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testPutWithSignalKernel<<<numBlocks, blockSize>>>(
      p2p, dst_d, src_d, signal_id, nbytes, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// wait() test kernel and wrapper
// =============================================================================

__global__ void testWaitKernel(
    P2pNvlTransportDevice* p2p,
    CmpOp op,
    uint64_t signal_id,
    uint64_t expected,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p->wait_signal_until(group, signal_id, op, expected);
}

void testWait(
    P2pNvlTransportDevice* p2p,
    CmpOp op,
    uint64_t signal_id,
    uint64_t expected,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testWaitKernel<<<numBlocks, blockSize>>>(
      p2p, op, signal_id, expected, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
