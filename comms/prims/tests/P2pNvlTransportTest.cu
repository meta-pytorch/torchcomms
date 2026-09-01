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
    AbortDevice abortDevice) {
  abortDevice.start();
  auto group = make_block_group();
  TiledBuffer<char> tiles(reinterpret_cast<char*>(src_d), nbytes, group);
  p2p.send(group, tiles.data(), tiles.bytes(), maxSignalBytes, abortDevice);
}

__global__ void testTileRecvKernel(
    P2pNvlTransportDevice p2p,
    void* dst_d,
    size_t nbytes,
    size_t maxSignalBytes,
    AbortDevice abortDevice) {
  abortDevice.start();
  auto group = make_block_group();
  TiledBuffer<char> tiles(reinterpret_cast<char*>(dst_d), nbytes, group);
  p2p.recv(group, tiles.data(), tiles.bytes(), maxSignalBytes, abortDevice);
}

__device__ void wait_for_second_call_signal(
    P2pNvlTransportDevice& p2p,
    ThreadGroup& group,
    int blockId,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool enabled,
    const AbortDevice& abortDevice) {
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
      group, CmpOp::CMP_GE, secondCallStarted, abortDevice);
}

__global__ void testTileMultiCallSendRecvKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool waitForSecondCallSignal,
    AbortDevice abortDevice) {
  abortDevice.start();

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
          abortDevice);
    }
  } else {
    wait_for_second_call_signal(
        p2p,
        sub,
        blockId,
        bytesPerCall,
        maxSignalBytes,
        waitForSecondCallSignal,
        abortDevice);
    char* recvTile = recvTiles.tile_data(blockId);
    for (int i = 0; i < numCalls; ++i) {
      p2p.recv(
          sub,
          recvTile + i * bytesPerCall,
          bytesPerCall,
          maxSignalBytes,
          abortDevice);
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
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  auto [role, sub] = group.partition(2);
  const int blockId = sub.group_id;

  if (role == 0) {
    char* sendTile = sendTiles.tile_data(blockId);
    p2p.send(sub, sendTile, firstCallBytes, firstMaxSignalBytes, abortDevice);
    p2p.send(
        sub,
        sendTile + firstCallBytes,
        secondCallBytes,
        secondMaxSignalBytes,
        abortDevice);
  } else {
    wait_for_second_call_signal(
        p2p,
        sub,
        blockId,
        firstCallBytes,
        secondMaxSignalBytes,
        waitForSecondCallSignal,
        abortDevice);
    char* recvTile = recvTiles.tile_data(blockId);
    p2p.recv(sub, recvTile, firstCallBytes, firstMaxSignalBytes, abortDevice);
    p2p.recv(
        sub,
        recvTile + firstCallBytes,
        secondCallBytes,
        secondMaxSignalBytes,
        abortDevice);
  }
}

__global__ void testTileTwoCallSendThenRecvKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* sendTile = sendTiles.tile_data(blockId);
  char* recvTile = recvTiles.tile_data(blockId);

  p2p.send(group, sendTile, firstCallBytes, maxSignalBytes, abortDevice);
  p2p.recv(group, recvTile, firstCallBytes, maxSignalBytes, abortDevice);
  p2p.send(
      group,
      sendTile + firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      abortDevice);
  p2p.recv(
      group,
      recvTile + firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      abortDevice);
}

__global__ void testTileMultiCallSendOnlyKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* sendTile = sendTiles.tile_data(blockId);
  for (int i = 0; i < numCalls; ++i) {
    p2p.send(
        group,
        sendTile + i * bytesPerCall,
        bytesPerCall,
        maxSignalBytes,
        abortDevice);
  }
}

__global__ void testTileTwoCallSendOnlyKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* sendTile = sendTiles.tile_data(blockId);
  p2p.send(group, sendTile, firstCallBytes, maxSignalBytes, abortDevice);
  p2p.send(
      group,
      sendTile + firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      abortDevice);
}

__device__ void check_wrapped_substep_with_existing_signals(
    P2pNvlTransportDevice& p2p,
    ThreadGroup& group,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    const AbortDevice& abortDevice) {
  const size_t perChannelBuffer = p2p.options().per_channel_buffer;
  const size_t perBlockSlotSize = p2p.options().per_channel_slot;
  const size_t effectiveChunk =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlotSize
      ? (maxSignalBytes & ~15ULL)
      : perBlockSlotSize;
  const size_t pipelineBytes = perChannelBuffer;
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
  const size_t targetOffset = targetSlot * perBlockSlotSize + targetChunkOff;
  const uint64_t targetStreamEnd = targetStreamStart + effectiveChunk;
  const uint64_t targetAckValue = targetStreamEnd - pipelineBytes;

  // Drive the existing head signal to the minimum threshold that releases only
  // the first wrapped chunk. Real recv() may coalesce head updates at slot
  // boundaries; this test is intentionally isolating the sender/forwarder wait
  // predicate for the following nonzero wrapped substep.
  p2p.local_channel_at(0).slot_free.signal(
      group, SignalOp::SIGNAL_SET, firstAckValue);
  p2p.remote_channel_at(0).data_ready.wait_until(
      group, CmpOp::CMP_GE, firstStreamEnd, abortDevice);

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
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  if (blockIdx.x == 0) {
    p2p.send(group, sendData, nbytes, maxSignalBytes, abortDevice);
  } else {
    check_wrapped_substep_with_existing_signals(
        p2p,
        group,
        maxSignalBytes,
        sentinel,
        observedEarlyOverwrite,
        abortDevice);
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
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  if (blockIdx.x == 0) {
    pred.forward(group, dst, nbytes, succ, maxSignalBytes, abortDevice);
  } else {
    check_wrapped_substep_with_existing_signals(
        succ,
        group,
        maxSignalBytes,
        sentinel,
        observedEarlyOverwrite,
        abortDevice);
  }
}

__global__ void testPrepareTileStagingKernel(
    P2pNvlTransportDevice p2p,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    int sourceRank,
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* staging = p2p.local_state().dataBuffer;

  const size_t perChannelBuffer = p2p.options().per_channel_buffer;
  const size_t perBlockSlotSize = p2p.options().per_channel_slot;
  const size_t chunkSize =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlotSize
      ? (maxSignalBytes & ~15ULL)
      : perBlockSlotSize;
  const size_t effectiveChunk = chunkSize > 0 ? chunkSize : perBlockSlotSize;
  const size_t pipelineBytes = perChannelBuffer;
  const size_t stagingOff = blockId * perChannelBuffer;

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
      const size_t bufferOff = stagingOff + slot * perBlockSlotSize + chunkOff;
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
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* staging = p2p.local_state().dataBuffer;

  const size_t perChannelBuffer = p2p.options().per_channel_buffer;
  const size_t perBlockSlotSize = p2p.options().per_channel_slot;
  const size_t effectiveChunk =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlotSize
      ? (maxSignalBytes & ~15ULL)
      : perBlockSlotSize;
  const size_t pipelineBytes = perChannelBuffer;
  const size_t stagingOff = blockId * perChannelBuffer;

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
      const size_t bufferOff = stagingOff + slot * perBlockSlotSize + chunkOff;
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
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* recvTile = recvTiles.tile_data(blockId);
  for (int i = 0; i < numCalls; ++i) {
    p2p.recv(
        group,
        recvTile + i * bytesPerCall,
        bytesPerCall,
        maxSignalBytes,
        abortDevice);
  }
}

__global__ void testTileTwoCallRecvOnlyKernel(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> recvTiles,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* recvTile = recvTiles.tile_data(blockId);
  p2p.recv(group, recvTile, firstCallBytes, maxSignalBytes, abortDevice);
  p2p.recv(
      group,
      recvTile + firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      abortDevice);
}

__global__ void testTileMultiCallForwardKernel(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool waitForSecondCallSignal,
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  wait_for_second_call_signal(
      pred,
      group,
      blockId,
      bytesPerCall,
      maxSignalBytes,
      waitForSecondCallSignal,
      abortDevice);

  char* dstTile = dstTiles.tile_data(blockId);
  for (int i = 0; i < numCalls; ++i) {
    pred.forward(
        group,
        dstTile + i * bytesPerCall,
        bytesPerCall,
        succ,
        maxSignalBytes,
        abortDevice);
  }
}

__global__ void testTileTwoCallForwardKernel(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* dstTile = dstTiles.tile_data(blockId);
  pred.forward(
      group, dstTile, firstCallBytes, succ, maxSignalBytes, abortDevice);
  pred.forward(
      group,
      dstTile + firstCallBytes,
      secondCallBytes,
      succ,
      maxSignalBytes,
      abortDevice);
}

__global__ void testTileTwoCallVariableSignalForwardKernel(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t firstMaxSignalBytes,
    size_t secondMaxSignalBytes,
    AbortDevice abortDevice) {
  abortDevice.start();

  auto group = make_block_group();
  const int blockId = group.group_id;
  char* dstTile = dstTiles.tile_data(blockId);
  pred.forward(
      group, dstTile, firstCallBytes, succ, firstMaxSignalBytes, abortDevice);
  pred.forward(
      group,
      dstTile + firstCallBytes,
      secondCallBytes,
      succ,
      secondMaxSignalBytes,
      abortDevice);
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
    AbortDevice abortDevice,
    int numBlocks,
    int blockSize,
    cudaStream_t stream) {
  testTileSendKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p, src_d, nbytes, maxSignalBytes, abortDevice);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testTileRecv(
    const P2pNvlTransportDevice& p2p,
    void* dst_d,
    size_t nbytes,
    size_t maxSignalBytes,
    AbortDevice abortDevice,
    int numBlocks,
    int blockSize,
    cudaStream_t stream) {
  testTileRecvKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p, dst_d, nbytes, maxSignalBytes, abortDevice);
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
      AbortDevice());
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
    AbortDevice abortDevice,
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
      abortDevice);
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
    AbortDevice abortDevice,
    cudaStream_t stream) {
  testTileTwoCallSendThenRecvKernel<<<activeBlocks, blockSize, 0, stream>>>(
      p2p,
      sendTiles,
      recvTiles,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      abortDevice);
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
      p2p, sendTiles, numCalls, bytesPerCall, maxSignalBytes, AbortDevice());
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
      AbortDevice());
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
      AbortDevice());
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
      AbortDevice());
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
      p2p, numCalls, bytesPerCall, maxSignalBytes, sourceRank, AbortDevice());
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
      AbortDevice());
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
      p2p, recvTiles, numCalls, bytesPerCall, maxSignalBytes, AbortDevice());
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
      AbortDevice());
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
      AbortDevice());
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
      AbortDevice());
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
      AbortDevice());
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

/*
 * `sendDone` / `recvDone` are per-thread but derive only from the group-uniform
 * status each progress call returns, so every thread in the block agrees on
 * them. That matters: progress_*_once syncs the group internally, so a diverged
 * skip would strand part of the block at the next barrier.
 */
__global__ void testProgressSendRecvKernel(
    P2pNvlTransportDevice p2p,
    void* src_d,
    void* dst_d,
    size_t nbytes,
    size_t maxSignalBytes,
    AbortDevice abort,
    ProgressCounters* counters) {
  abort.start();
  auto group = make_block_group();

  TiledBuffer<char> sendTiles(static_cast<char*>(src_d), nbytes, group);
  TiledBuffer<char> recvTiles(static_cast<char*>(dst_d), nbytes, group);

  p2p.init_send_progress(group, sendTiles.bytes(), maxSignalBytes);
  p2p.init_recv_progress(group, recvTiles.bytes(), maxSignalBytes);

  bool sendDone = sendTiles.bytes() == 0;
  bool recvDone = recvTiles.bytes() == 0;
  int sendWaiting = 0;
  int sendProgressed = 0;
  int sendAborted = 0;
  int recvWaiting = 0;
  int recvProgressed = 0;
  int recvAborted = 0;

  while (!sendDone || !recvDone) {
    if (!sendDone) {
      const auto status = p2p.progress_send_once(
          group, sendTiles.data(), sendTiles.bytes(), maxSignalBytes, abort);
      if (status == NvlSendRecvProgressStatus::Done) {
        sendDone = true;
      } else if (status == NvlSendRecvProgressStatus::Aborted) {
        sendDone = true;
        ++sendAborted;
      } else if (status == NvlSendRecvProgressStatus::Waiting) {
        ++sendWaiting;
      } else {
        ++sendProgressed;
      }
    }
    if (!recvDone) {
      const auto status = p2p.progress_recv_once(
          group, recvTiles.data(), recvTiles.bytes(), maxSignalBytes, abort);
      if (status == NvlSendRecvProgressStatus::Done) {
        recvDone = true;
      } else if (status == NvlSendRecvProgressStatus::Aborted) {
        recvDone = true;
        ++recvAborted;
      } else if (status == NvlSendRecvProgressStatus::Waiting) {
        ++recvWaiting;
      } else {
        ++recvProgressed;
      }
    }
  }

  if (counters != nullptr && blockIdx.x == 0 && group.is_leader()) {
    counters->sendWaiting = sendWaiting;
    counters->sendProgressed = sendProgressed;
    counters->recvWaiting = recvWaiting;
    counters->recvProgressed = recvProgressed;
    counters->sendAborted = sendAborted;
    counters->recvAborted = recvAborted;
  }
}

void testProgressSendRecv(
    P2pNvlTransportDevice p2p,
    void* src_d,
    void* dst_d,
    size_t nbytes,
    size_t maxSignalBytes,
    AbortDevice abort,
    ProgressCounters* counters,
    int numBlocks,
    int blockSize,
    cudaStream_t stream) {
  testProgressSendRecvKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p, src_d, dst_d, nbytes, maxSignalBytes, abort, counters);
  PIPES_KERNEL_LAUNCH_CHECK();
}

/*
 * Two peers driven concurrently from one block.
 *
 * Group construction matters here. `make_block_group().partition(2)` does not
 * work on a single-block launch: partition() splits *groups*, and one block
 * group means total_groups == 1, so it rejects num_partitions == 2. Instead
 * make two half-block multiwarp groups and interleave them, the same shape
 * p2pTileSendRecvBidirCta uses. partition_interleaved(2) renumbers both
 * subgroups to group_id 0, so each drives channel 0 of its own transport --
 * and channel indices are per-transport, so that is not a collision.
 *
 * Each subgroup runs the same alternating send/recv loop as the single-peer
 * case. The status is group-uniform within a subgroup, so every thread of a
 * subgroup leaves the loop together.
 */
__global__ void testProgressTwoPeerSendRecvKernel(
    P2pNvlTransportDevice p2pPred,
    P2pNvlTransportDevice p2pSucc,
    void* predSrc,
    void* predDst,
    void* succSrc,
    void* succDst,
    size_t nbytes,
    size_t maxSignalBytes,
    AbortDevice abort) {
  abort.start();
  auto group = make_multiwarp_group(blockDim.x / 2);
  auto [role, sub] = group.partition_interleaved(2);

  P2pNvlTransportDevice& p2p = (role == 0) ? p2pPred : p2pSucc;
  char* src = static_cast<char*>((role == 0) ? predSrc : succSrc);
  char* dst = static_cast<char*>((role == 0) ? predDst : succDst);

  p2p.init_send_progress(sub, nbytes, maxSignalBytes);
  p2p.init_recv_progress(sub, nbytes, maxSignalBytes);

  bool sendDone = nbytes == 0;
  bool recvDone = nbytes == 0;
  while (!sendDone || !recvDone) {
    if (!sendDone) {
      const auto status =
          p2p.progress_send_once(sub, src, nbytes, maxSignalBytes, abort);
      sendDone = status == NvlSendRecvProgressStatus::Done ||
          status == NvlSendRecvProgressStatus::Aborted;
    }
    if (!recvDone) {
      const auto status =
          p2p.progress_recv_once(sub, dst, nbytes, maxSignalBytes, abort);
      recvDone = status == NvlSendRecvProgressStatus::Done ||
          status == NvlSendRecvProgressStatus::Aborted;
    }
  }
}

void testProgressTwoPeerSendRecv(
    P2pNvlTransportDevice p2pPred,
    P2pNvlTransportDevice p2pSucc,
    void* predSrc,
    void* predDst,
    void* succSrc,
    void* succDst,
    size_t nbytes,
    size_t maxSignalBytes,
    AbortDevice abort,
    int blockSize,
    cudaStream_t stream) {
  testProgressTwoPeerSendRecvKernel<<<1, blockSize, 0, stream>>>(
      p2pPred,
      p2pSucc,
      predSrc,
      predDst,
      succSrc,
      succDst,
      nbytes,
      maxSignalBytes,
      abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

/*
 * Sender with nobody draining the far end, so the pipeline necessarily fills
 * and stays full. Bounded iteration count: the point is to observe Waiting and
 * non-completion, not to finish. In a symmetric exchange whether Waiting is
 * ever observed depends on how fast the peer drains, so this one-sided shape is
 * the only way to make backpressure deterministic.
 */
__global__ void testProgressSendBackpressureKernel(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    size_t maxSignalBytes,
    int maxIterations,
    AbortDevice abort,
    ProgressCounters* counters) {
  abort.start();
  auto group = make_block_group();

  TiledBuffer<char> sendTiles(static_cast<char*>(src_d), nbytes, group);
  p2p.init_send_progress(group, sendTiles.bytes(), maxSignalBytes);

  bool done = false;
  bool aborted = false;
  int waiting = 0;
  int progressed = 0;

  for (int i = 0; i < maxIterations && !done && !aborted; ++i) {
    const auto status = p2p.progress_send_once(
        group, sendTiles.data(), sendTiles.bytes(), maxSignalBytes, abort);
    if (status == NvlSendRecvProgressStatus::Done) {
      done = true;
    } else if (status == NvlSendRecvProgressStatus::Aborted) {
      aborted = true;
    } else if (status == NvlSendRecvProgressStatus::Waiting) {
      ++waiting;
    } else {
      ++progressed;
    }
  }

  if (counters != nullptr && blockIdx.x == 0 && group.is_leader()) {
    counters->sendWaiting = waiting;
    counters->sendProgressed = progressed;
    counters->sendCompleted = done ? 1 : 0;
    counters->sendAborted = aborted ? 1 : 0;
    counters->recvWaiting = 0;
    counters->recvProgressed = 0;
  }
}

void testProgressSendBackpressure(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    size_t maxSignalBytes,
    int maxIterations,
    AbortDevice abort,
    ProgressCounters* counters,
    int numBlocks,
    int blockSize,
    cudaStream_t stream) {
  testProgressSendBackpressureKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p, src_d, nbytes, maxSignalBytes, maxIterations, abort, counters);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// Drives one progress operation to completion on this block's channel.
__device__ inline void drainProgressPair(
    P2pNvlTransportDevice& p2p,
    ThreadGroup& group,
    char* src,
    char* dst,
    size_t bytes,
    size_t maxSignalBytes,
    const AbortDevice& abort) {
  p2p.init_send_progress(group, bytes, maxSignalBytes);
  p2p.init_recv_progress(group, bytes, maxSignalBytes);

  bool sendDone = bytes == 0;
  bool recvDone = bytes == 0;
  while (!sendDone || !recvDone) {
    if (!sendDone) {
      const auto status =
          p2p.progress_send_once(group, src, bytes, maxSignalBytes, abort);
      sendDone = status == NvlSendRecvProgressStatus::Done ||
          status == NvlSendRecvProgressStatus::Aborted;
    }
    if (!recvDone) {
      const auto status =
          p2p.progress_recv_once(group, dst, bytes, maxSignalBytes, abort);
      recvDone = status == NvlSendRecvProgressStatus::Done ||
          status == NvlSendRecvProgressStatus::Aborted;
    }
  }
}

__global__ void testProgressTwoCallSendRecvKernel(
    P2pNvlTransportDevice p2p,
    void* src_d,
    void* dst_d,
    size_t firstBytes,
    size_t secondBytes,
    size_t firstMaxSignalBytes,
    size_t secondMaxSignalBytes,
    AbortDevice abort) {
  abort.start();
  auto group = make_block_group();

  TiledBuffer<char> firstSend(static_cast<char*>(src_d), firstBytes, group);
  TiledBuffer<char> firstRecv(static_cast<char*>(dst_d), firstBytes, group);
  drainProgressPair(
      p2p,
      group,
      firstSend.data(),
      firstRecv.data(),
      firstSend.bytes(),
      firstMaxSignalBytes,
      abort);

  // Second operation starts where the first left the cursor. Offsetting the
  // user buffers keeps the two payloads distinguishable at verification time.
  TiledBuffer<char> secondSend(
      static_cast<char*>(src_d) + firstBytes, secondBytes, group);
  TiledBuffer<char> secondRecv(
      static_cast<char*>(dst_d) + firstBytes, secondBytes, group);
  drainProgressPair(
      p2p,
      group,
      secondSend.data(),
      secondRecv.data(),
      secondSend.bytes(),
      secondMaxSignalBytes,
      abort);
}

void testProgressTwoCallSendRecv(
    P2pNvlTransportDevice p2p,
    void* src_d,
    void* dst_d,
    size_t firstBytes,
    size_t secondBytes,
    size_t firstMaxSignalBytes,
    size_t secondMaxSignalBytes,
    AbortDevice abort,
    int numBlocks,
    int blockSize,
    cudaStream_t stream) {
  testProgressTwoCallSendRecvKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p,
      src_d,
      dst_d,
      firstBytes,
      secondBytes,
      firstMaxSignalBytes,
      secondMaxSignalBytes,
      abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

__global__ void testProgressAbortThenReinitKernel(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    int maxIterations,
    AbortDevice abort,
    ProgressCounters* counters) {
  abort.start();
  auto group = make_block_group();

  TiledBuffer<char> sendTiles(static_cast<char*>(src_d), nbytes, group);

  // Phase 1: stall with no receiver, until the abort concludes the operation.
  // Completion is impossible here, so Aborted is the only terminal status the
  // loop can reach; `done` is tracked separately to keep that assertable.
  p2p.init_send_progress(group, sendTiles.bytes(), 0);
  bool done = false;
  bool aborted = false;
  for (int i = 0; i < maxIterations && !done && !aborted; ++i) {
    const auto status = p2p.progress_send_once(
        group, sendTiles.data(), sendTiles.bytes(), 0, abort);
    done = status == NvlSendRecvProgressStatus::Done;
    aborted = status == NvlSendRecvProgressStatus::Aborted;
  }

  if (counters != nullptr && blockIdx.x == 0 && group.is_leader()) {
    counters->sendCompleted = done ? 1 : 0;
    counters->sendAborted = aborted ? 1 : 0;
  }

  // Phase 2: re-init the same channel with no kernel boundary. If the abort
  // cleanup were not published to every thread, this traps ("already has an
  // in-flight send") or strands part of the group at a barrier.
  p2p.init_send_progress(group, sendTiles.bytes(), 0);
  group.sync();
}

void testProgressAbortThenReinit(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    int maxIterations,
    AbortDevice abort,
    ProgressCounters* counters,
    int numBlocks,
    int blockSize,
    cudaStream_t stream) {
  testProgressAbortThenReinitKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p, src_d, nbytes, maxIterations, abort, counters);
  PIPES_KERNEL_LAUNCH_CHECK();
}

/*
 * Receiver-side mirror of testProgressSendBackpressureKernel. With no sender,
 * DATA_READY never advances and every poll must report Waiting.
 */
__global__ void testProgressRecvBackpressureKernel(
    P2pNvlTransportDevice p2p,
    void* dst_d,
    size_t nbytes,
    size_t maxSignalBytes,
    int maxIterations,
    AbortDevice abort,
    ProgressCounters* counters) {
  abort.start();
  auto group = make_block_group();

  TiledBuffer<char> recvTiles(static_cast<char*>(dst_d), nbytes, group);
  p2p.init_recv_progress(group, recvTiles.bytes(), maxSignalBytes);

  bool done = false;
  bool aborted = false;
  int waiting = 0;
  int progressed = 0;

  for (int i = 0; i < maxIterations && !done && !aborted; ++i) {
    const auto status = p2p.progress_recv_once(
        group, recvTiles.data(), recvTiles.bytes(), maxSignalBytes, abort);
    if (status == NvlSendRecvProgressStatus::Done) {
      done = true;
    } else if (status == NvlSendRecvProgressStatus::Aborted) {
      aborted = true;
    } else if (status == NvlSendRecvProgressStatus::Waiting) {
      ++waiting;
    } else {
      ++progressed;
    }
  }

  if (counters != nullptr && blockIdx.x == 0 && group.is_leader()) {
    counters->recvWaiting = waiting;
    counters->recvProgressed = progressed;
    counters->recvCompleted = done ? 1 : 0;
    counters->recvAborted = aborted ? 1 : 0;
    counters->sendWaiting = 0;
    counters->sendProgressed = 0;
    counters->sendCompleted = 0;
    counters->sendAborted = 0;
  }
}

void testProgressRecvBackpressure(
    P2pNvlTransportDevice p2p,
    void* dst_d,
    size_t nbytes,
    size_t maxSignalBytes,
    int maxIterations,
    AbortDevice abort,
    ProgressCounters* counters,
    int numBlocks,
    int blockSize,
    cudaStream_t stream) {
  testProgressRecvBackpressureKernel<<<numBlocks, blockSize, 0, stream>>>(
      p2p, dst_d, nbytes, maxSignalBytes, maxIterations, abort, counters);
  PIPES_KERNEL_LAUNCH_CHECK();
}

__global__ void testReadSlotFreeCounterKernel(
    P2pNvlTransportDevice p2p,
    int channel,
    unsigned long long* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = static_cast<unsigned long long>(
        p2p.local_channel_at(channel).slot_free.load());
  }
}

void testReadSlotFreeCounter(
    P2pNvlTransportDevice p2p,
    int channel,
    unsigned long long* out,
    cudaStream_t stream) {
  testReadSlotFreeCounterKernel<<<1, 32, 0, stream>>>(p2p, channel, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
