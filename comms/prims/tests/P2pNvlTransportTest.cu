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

__device__ inline size_t align_protocol_bytes(size_t nbytes) {
  return (nbytes + 15ULL) & ~15ULL;
}

__device__ inline uint64_t round_up_to_multiple(
    uint64_t value,
    size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const uint64_t alignment64 = static_cast<uint64_t>(alignment);
  return ((value + alignment64 - 1) / alignment64) * alignment64;
}

__device__ inline size_t signal_alignment(
    size_t maxSignalBytes,
    size_t perBlockSlotSize) {
  const bool usesPartialSlot =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlotSize;
  size_t alignment =
      usesPartialSlot ? (maxSignalBytes & ~15ULL) : perBlockSlotSize;
  return alignment == 0 ? perBlockSlotSize : alignment;
}

__device__ inline size_t protocol_step_bytes(
    uint64_t baseByte,
    size_t payloadBytes,
    size_t maxSignalBytes,
    size_t perBlockSlotSize) {
  const size_t protocolBytes = align_protocol_bytes(payloadBytes);
  const size_t alignment = signal_alignment(maxSignalBytes, perBlockSlotSize);
  const uint64_t payloadEnd = baseByte + protocolBytes;
  return protocolBytes +
      static_cast<size_t>(
             round_up_to_multiple(payloadEnd, alignment) - payloadEnd);
}

__global__ void testTileSendKernel(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();
  auto group = make_block_group();
  TiledBuffer<char> tiles(reinterpret_cast<char*>(src_d), nbytes, group);
  p2p.send(group, tiles.data(), tiles.bytes(), maxSignalBytes, timeout);
}

__global__ void testTileRecvKernel(
    P2pNvlTransportDevice p2p,
    void* dst_d,
    size_t nbytes,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();
  auto group = make_block_group();
  TiledBuffer<char> tiles(reinterpret_cast<char*>(dst_d), nbytes, group);
  p2p.recv(group, tiles.data(), tiles.bytes(), maxSignalBytes, timeout);
}

__device__ void wait_for_second_call_signal(
    P2pNvlTransportDevice& p2p,
    ThreadGroup& group,
    int blockId,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool enabled,
    const Timeout& timeout) {
  if (!enabled) {
    return;
  }
  const size_t protocolBytes = (bytesPerCall + 15ULL) & ~15ULL;
  const size_t perBlockSlotSize = p2p.options().per_channel_slot;
  const size_t chunkSize =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlotSize
      ? (maxSignalBytes & ~15ULL)
      : perBlockSlotSize;
  const size_t effectiveChunk = chunkSize > 0 ? chunkSize : perBlockSlotSize;
  const uint64_t secondCallStarted = protocolBytes +
      (protocolBytes < effectiveChunk ? protocolBytes : effectiveChunk);
  p2p.local_channel_at(blockId).data_ready.wait_until(
      group, CmpOp::CMP_GE, secondCallStarted, timeout);
}

__global__ void testTileMultiCallSendRecvKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
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
          maxSignalBytes,
          timeout);
    }
  } else {
    wait_for_second_call_signal(
        p2p,
        sub,
        blockId,
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
          maxSignalBytes,
          timeout);
    }
  }
}

__global__ void testTileTwoCallVariableSignalSendRecvKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
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
    p2p.send(sub, sendTile, firstCallBytes, firstMaxSignalBytes, timeout);
    p2p.send(
        sub,
        sendTile + firstCallBytes,
        secondCallBytes,
        secondMaxSignalBytes,
        timeout);
  } else {
    wait_for_second_call_signal(
        p2p,
        sub,
        blockId,
        firstCallBytes,
        secondMaxSignalBytes,
        waitForSecondCallSignal,
        timeout);
    char* recvTile = recvTiles.tile_data(blockId);
    p2p.recv(sub, recvTile, firstCallBytes, firstMaxSignalBytes, timeout);
    p2p.recv(
        sub,
        recvTile + firstCallBytes,
        secondCallBytes,
        secondMaxSignalBytes,
        timeout);
  }
}

__global__ void testTileTwoCallSendThenRecvKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* sendTile = sendTiles.tile_data(blockId);
  char* recvTile = recvTiles.tile_data(blockId);

  p2p.send(group, sendTile, firstCallBytes, maxSignalBytes, timeout);
  p2p.recv(group, recvTile, firstCallBytes, maxSignalBytes, timeout);
  p2p.send(
      group,
      sendTile + firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      timeout);
  p2p.recv(
      group,
      recvTile + firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      timeout);
}

__global__ void testTileMultiCallSendOnlyKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
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
        maxSignalBytes,
        timeout);
  }
}

__global__ void testTileTwoCallSendOnlyKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* sendTile = sendTiles.tile_data(blockId);
  p2p.send(group, sendTile, firstCallBytes, maxSignalBytes, timeout);
  p2p.send(
      group,
      sendTile + firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      timeout);
}

__device__ void check_wrapped_substep_with_existing_signals(
    P2pNvlTransportDevice& p2p,
    ThreadGroup& group,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    const Timeout& timeout) {
  const size_t slotSize = p2p.options().dataBufferSize;
  const size_t perBlockSlotSize = p2p.options().per_channel_slot;
  const size_t effectiveChunk =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlotSize
      ? (maxSignalBytes & ~15ULL)
      : perBlockSlotSize;
  const size_t pipelineBytes = perBlockSlotSize * p2p.options().pipelineDepth;
  // blockId 0 by construction in this test (kernel launched with 2 blocks,
  // block 0 is sender/forwarder, block 1 is this checker).
  const uint64_t streamStart =
      static_cast<uint64_t>(p2p.local_channel_at(0).send_cursor);
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

  // Drive the existing head signal to the minimum threshold that releases only
  // the first wrapped chunk. Real recv() may coalesce head updates at slot
  // boundaries; this test is intentionally isolating the sender/forwarder wait
  // predicate for the following nonzero wrapped substep.
  p2p.local_channel_at(0).slot_free.signal(
      group, SignalOp::SIGNAL_SET, firstAckValue);
  p2p.remote_channel_at(0).data_ready.wait_until(
      group, CmpOp::CMP_GE, firstStreamEnd, timeout);

  if (group.is_leader()) {
    const auto observed =
        static_cast<unsigned char>(p2p.remote_state().dataBuffer[targetOffset]);
    *observedEarlyOverwrite = observed == sentinel ? 0 : 1;
    p2p.local_channel_at(0).slot_free.signal(
        SignalOp::SIGNAL_SET, targetAckValue);
  }
}

__global__ void testTileSendWaitsForWrappedSubstepAckKernel(
    P2pNvlTransportDevice p2p,
    const char* sendData,
    size_t nbytes,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  if (blockIdx.x == 0) {
    p2p.send(group, sendData, nbytes, maxSignalBytes, timeout);
  } else {
    check_wrapped_substep_with_existing_signals(
        p2p, group, maxSignalBytes, sentinel, observedEarlyOverwrite, timeout);
  }
}

__global__ void testTileForwardWaitsForWrappedSubstepAckKernel(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    char* dst,
    size_t nbytes,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  if (blockIdx.x == 0) {
    pred.forward(group, dst, nbytes, succ, maxSignalBytes, timeout);
  } else {
    check_wrapped_substep_with_existing_signals(
        succ, group, maxSignalBytes, sentinel, observedEarlyOverwrite, timeout);
  }
}

__global__ void testPrepareTileStagingKernel(
    P2pNvlTransportDevice p2p,
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
  const size_t perBlockSlotSize = p2p.options().per_channel_slot;
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
    const size_t protocolBytes = align_protocol_bytes(bytesPerCall);
    for (size_t dataOff = 0; dataOff < protocolBytes;) {
      const uint64_t streamStart = baseByte + dataOff;
      const size_t pipelineOff =
          static_cast<size_t>(streamStart % pipelineBytes);
      const size_t slot = pipelineOff / perBlockSlotSize;
      const size_t chunkOff = pipelineOff - slot * perBlockSlotSize;
      const size_t slotRemaining = perBlockSlotSize - chunkOff;
      const size_t dataRemaining = protocolBytes - dataOff;
      size_t copyBytes =
          effectiveChunk < dataRemaining ? effectiveChunk : dataRemaining;
      copyBytes = copyBytes < slotRemaining ? copyBytes : slotRemaining;
      size_t validBytes = 0;
      if (dataOff < bytesPerCall) {
        const size_t remaining = bytesPerCall - dataOff;
        validBytes = copyBytes < remaining ? copyBytes : remaining;
      }
      const size_t bufferOff = slot * slotSize + stagingOff + chunkOff;
      for (size_t idx = group.thread_id_in_group; idx < validBytes;
           idx += group.group_size) {
        staging[bufferOff + idx] = pattern;
      }
      dataOff += copyBytes;
    }
    baseByte += protocol_step_bytes(
        baseByte, bytesPerCall, maxSignalBytes, perBlockSlotSize);
  }

  group.sync();
  if (group.is_leader()) {
    p2p.local_channel_at(blockId).data_ready.signal(
        SignalOp::SIGNAL_SET, baseByte);
  }
}

__global__ void testPrepareTileTwoCallStagingKernel(
    P2pNvlTransportDevice p2p,
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
  const size_t perBlockSlotSize = p2p.options().per_channel_slot;
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
    const size_t protocolBytes = align_protocol_bytes(callBytes);
    for (size_t dataOff = 0; dataOff < protocolBytes;) {
      const uint64_t streamStart = baseByte + dataOff;
      const size_t pipelineOff =
          static_cast<size_t>(streamStart % pipelineBytes);
      const size_t slot = pipelineOff / perBlockSlotSize;
      const size_t chunkOff = pipelineOff - slot * perBlockSlotSize;
      const size_t slotRemaining = perBlockSlotSize - chunkOff;
      const size_t dataRemaining = protocolBytes - dataOff;
      size_t copyBytes =
          effectiveChunk < dataRemaining ? effectiveChunk : dataRemaining;
      copyBytes = copyBytes < slotRemaining ? copyBytes : slotRemaining;
      size_t validBytes = 0;
      if (dataOff < callBytes) {
        const size_t remaining = callBytes - dataOff;
        validBytes = copyBytes < remaining ? copyBytes : remaining;
      }
      const size_t bufferOff = slot * slotSize + stagingOff + chunkOff;
      for (size_t idx = group.thread_id_in_group; idx < validBytes;
           idx += group.group_size) {
        staging[bufferOff + idx] = pattern;
      }
      dataOff += copyBytes;
    }
    baseByte += protocol_step_bytes(
        baseByte, callBytes, maxSignalBytes, perBlockSlotSize);
  }

  group.sync();
  if (group.is_leader()) {
    p2p.local_channel_at(blockId).data_ready.signal(
        SignalOp::SIGNAL_SET, baseByte);
  }
}

__global__ void testTileMultiCallRecvOnlyKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> recvTiles,
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
        maxSignalBytes,
        timeout);
  }
}

__global__ void testTileTwoCallRecvOnlyKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> recvTiles,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* recvTile = recvTiles.tile_data(blockId);
  p2p.recv(group, recvTile, firstCallBytes, maxSignalBytes, timeout);
  p2p.recv(
      group,
      recvTile + firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      timeout);
}

__global__ void testTileMultiCallForwardKernel(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
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
        maxSignalBytes,
        timeout);
  }
}

__global__ void testTileTwoCallForwardKernel(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    Timeout timeout) {
  timeout.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* dstTile = dstTiles.tile_data(blockId);
  pred.forward(group, dstTile, firstCallBytes, succ, maxSignalBytes, timeout);
  pred.forward(
      group,
      dstTile + firstCallBytes,
      secondCallBytes,
      succ,
      maxSignalBytes,
      timeout);
}

__global__ void testTileTwoCallVariableSignalForwardKernel(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
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
      group, dstTile, firstCallBytes, succ, firstMaxSignalBytes, timeout);
  pred.forward(
      group,
      dstTile + firstCallBytes,
      secondCallBytes,
      succ,
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

void testTileSend(
    const P2pNvlTransportDevice& p2p,
    void* src_d,
    size_t nbytes,
    size_t maxSignalBytes,
    Timeout timeout,
    int numBlocks,
    int blockSize,
    cudaStream_t stream) {
  testTileSendKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p, src_d, nbytes, maxSignalBytes, timeout);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileRecv(
    const P2pNvlTransportDevice& p2p,
    void* dst_d,
    size_t nbytes,
    size_t maxSignalBytes,
    Timeout timeout,
    int numBlocks,
    int blockSize,
    cudaStream_t stream) {
  testTileRecvKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p, dst_d, nbytes, maxSignalBytes, timeout);
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
      firstCallBytes,
      secondCallBytes,
      firstMaxSignalBytes,
      secondMaxSignalBytes,
      waitForSecondCallSignal,
      timeout);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileTwoCallSendThenRecv(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int blockSize,
    Timeout timeout,
    cudaStream_t stream) {
  testTileTwoCallSendThenRecvKernel<<<activeBlocks, blockSize, 0, stream>>>(
      p2p,
      sendTiles,
      recvTiles,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
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
      p2p, sendTiles, numCalls, bytesPerCall, maxSignalBytes, Timeout());
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
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      Timeout());
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileSendWaitsForWrappedSubstepAck(
    P2pNvlTransportDevice p2p,
    const char* sendData,
    size_t nbytes,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    int blockSize,
    cudaStream_t stream) {
  testTileSendWaitsForWrappedSubstepAckKernel<<<2, blockSize, 0, stream>>>(
      p2p,
      sendData,
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
      p2p, numCalls, bytesPerCall, maxSignalBytes, sourceRank, Timeout());
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
      p2p, recvTiles, numCalls, bytesPerCall, maxSignalBytes, Timeout());
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
