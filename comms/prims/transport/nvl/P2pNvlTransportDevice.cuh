// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <cuda.h>

#include <cuda_runtime.h>
#include <cstddef>
#include <cstring>
#include "comms/prims/core/BarrierState.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/DeviceMacros.cuh"
#include "comms/prims/core/MemcpyCopyOp.cuh"
#include "comms/prims/core/SignalState.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/amd/HipHostCompat.h"
#include "comms/prims/transport/ll/LlOps.cuh"
#include "comms/prims/transport/ll128/Ll128Ops.cuh"
#include "comms/prims/transport/nvl/NvlChannelState.cuh"

namespace comms::prims {

/**
 * LocalState - Pointers to local GPU's buffers
 *
 * With REMOTE-WRITE pattern (tile send/recv/forward):
 * - Sender writes to RemoteState (peer's local buffers via NVLink)
 * - Receiver reads from LocalState (own local buffers)
 *
 * This means LocalState buffers are the DESTINATION for incoming data.
 * Per-channel synchronization state is tracked separately by
 * NvlChannelState, not by these buffers.
 */
struct LocalState {
  char* dataBuffer;
  DeviceSpan<SignalState> signalBuffer;
  DeviceSpan<BarrierState> barrierBuffer;
  Ll128Packet* ll128Buffer{nullptr};
  LlLine* llBuffer{nullptr};
};

/**
 * RemoteState - Pointers to peer GPU's buffers (via NVLink peer mapping)
 *
 * With REMOTE-WRITE pattern:
 * - Sender writes directly to these buffers (peer's local memory)
 * - This allows receiver to read from local memory (faster)
 *
 * These pointers are obtained via IPC and point to peer's LocalState buffers.
 */
struct RemoteState {
  char* dataBuffer;
  DeviceSpan<SignalState> signalBuffer;
  DeviceSpan<BarrierState> barrierBuffer;
  Ll128Packet* ll128Buffer{nullptr};
  LlLine* llBuffer{nullptr};
};

/**
 * P2pNvlTransportOptions - Configuration for P2P NVLink transport
 *
 * Defines the derived buffer sizes for staged transfers.
 * - dataBufferSize: Size of ONE pipeline slot across all fixed channels
 * - pipelineDepth: Number of buffer slots for pipelining (typically 2-8)
 *
 * Total memory allocated = pipelineDepth × dataBufferSize
 */
struct P2pNvlTransportOptions {
  std::size_t dataBufferSize{0};
  std::size_t pipelineDepth{0};
  std::size_t ll128BufferNumPackets{0}; // 0 = no chunking
  std::size_t llBufferNumLines{0}; // 0 = no chunking

  // ---- Tile (per-channel) protocol fields. Populated by the host transport
  // from MultiPeerNvlTransportConfig. Used by send/recv/forward (the tile
  // path).
  //
  // Slot-major staging layout: within each pipeline slot (size =
  // dataBufferSize) each channel owns a fixed slice of size per_channel_slot.
  // Channel c at pipeline slot s reads/writes
  //   staging_base + s * dataBufferSize + c * per_channel_slot
  //
  //   per_channel_slot = MultiPeerNvlTransportConfig.perChannelSize
  //   dataBufferSize >= maxNumChannels * per_channel_slot
  //   max_num_channels = MultiPeerNvlTransportConfig.maxNumChannels
  //
  // max_num_channels must equal the array length of the per-peer
  // NvlChannelState arrays passed into the device transport.
  std::size_t per_channel_slot{0};
  int max_num_channels{0};
};

/**
 * P2pNvlTransportDevice - High-Performance GPU-to-GPU Data Transfer over NVLink
 *
 * Provides per-channel pipelined data transfer between GPUs using NVLink.
 * The tile send/recv/forward path uses NvlChannelState (cursors + signal
 * objects) for synchronization; remote-write semantics put data directly
 * into the peer's local staging buffer so the receiver reads locally.
 */
class P2pNvlTransportDevice {
 public:
  __host__ P2pNvlTransportDevice() = default;

  __host__ P2pNvlTransportDevice(
      int myRank,
      int peerRank,
      const P2pNvlTransportOptions& options,
      const LocalState& localState,
      const RemoteState& remoteState,
      NvlChannelState* localChannels = nullptr,
      NvlChannelState* remoteChannels = nullptr)
      : myRank_(myRank),
        peerRank_(peerRank),
        options_(options),
        localState_(localState),
        remoteState_(remoteState),
        local_channels_(localChannels),
        remote_channels_(remoteChannels) {}

  __host__ __device__ ~P2pNvlTransportDevice() = default;

  // Largest single call that can be pipelined within the staging ring without a
  // send()/recv() wrapping the ring and blocking on backpressure mid-call. In
  // the fixed-channel (tile) model each channel pipelines through its own
  // per_channel_slot ring (independent of the active block count), so the
  // window must be bounded by that ring's capacity (per_channel_slot *
  // safeDepth). A larger window lets a single call wrap the ring and wait for a
  // slot_free the peer only produces after its own recv() — deadlocking the
  // send-before-recv AllReduce pattern.
  __host__ __device__ std::size_t pipeline_window() const {
    const std::size_t safeDepth =
        options_.pipelineDepth > 1 ? options_.pipelineDepth - 1 : 1;
    return options_.per_channel_slot * safeDepth;
  }

  // Getters for testing
  __host__ const LocalState& getLocalState() const {
    return localState_;
  }

  __host__ const RemoteState& getRemoteState() const {
    return remoteState_;
  }

  __host__ __device__ size_t get_ll128_buffer_num_packets() const {
    return options_.ll128BufferNumPackets;
  }

  /**
   * put_group - Cooperative local memory copy using vectorized operations
   *
   * Performs a high-performance vectorized copy from src_d to dst_d.
   * Multiple groups collaborate on the same src/dst/nbytes — work is
   * distributed across all calling groups via for_each_item_contiguous
   * by global group_id.
   *
   * All calling groups must pass the same src/dst/nbytes. Unlike put(),
   * which has each group independently copy its own partition of data, this
   * version has all groups cooperate on the entire buffer.
   *
   * Contrast with send(): send() writes to the peer GPU's staging buffer
   * via NVLink with pipelined flow control. put_group() copies within local
   * memory without any signaling or flow control.
   *
   * @param group ThreadGroup for cooperative processing
   * @param dst_d Destination pointer (device memory)
   * @param src_d Source pointer (device memory)
   * @param nbytes Number of bytes to copy
   *
   * @return Number of bytes written by the current thread group
   */
  __device__ __forceinline__ std::size_t put_group(
      [[maybe_unused]] ThreadGroup& group,
      [[maybe_unused]] char* dst_d,
      [[maybe_unused]] const char* src_d,
      [[maybe_unused]] std::size_t nbytes) {
#if PIPES_IS_DEVICE_COMPILE
    if (nbytes == 0) {
      return 0;
    }

    // Compute chunk size: aim for nbytes / total_groups per chunk,
    // aligned to 16 bytes (uint4 size) for efficient vectorized access
    constexpr std::size_t kAlignment = 16;
    const std::size_t targetChunkSize = nbytes / group.total_groups;
    // Round up to nearest 16-byte boundary, minimum 16 bytes
    const std::size_t chunkSize =
        ((targetChunkSize + kAlignment - 1) / kAlignment) * kAlignment;
    // Ensure minimum chunk size
    const std::size_t alignedChunkSize = chunkSize > 0 ? chunkSize : kAlignment;

    const std::size_t numChunks =
        (nbytes + alignedChunkSize - 1) / alignedChunkSize;

    // Distribute chunks across all groups using for_each_item_contiguous
    // Each group processes its assigned contiguous range of chunks
    std::size_t totalBytesWritten = 0;
    group.for_each_item_contiguous(numChunks, [&](uint32_t chunkIdx) {
      const std::size_t chunkOffset = chunkIdx * alignedChunkSize;
      const std::size_t chunkBytes = (chunkOffset + alignedChunkSize <= nbytes)
          ? alignedChunkSize
          : nbytes - chunkOffset;

      if (chunkBytes > 0) {
        memcpy_vectorized(
            dst_d + chunkOffset, // dst_base
            src_d + chunkOffset, // src_base
            chunkBytes, // chunk_bytes
            group);
        totalBytesWritten += chunkBytes;
      }
    });
    return totalBytesWritten;
#endif
    return 0;
  }

  /**
   * put - Independent per-group local memory copy
   *
   * Performs a vectorized copy from src_d to dst_d using only threads within
   * the calling group. Each group operates independently on its own data,
   * so different groups can call put() with different src/dst/nbytes.
   *
   * Unlike put_group(), which has all groups cooperate on the same buffer,
   * put() has each group work on its own partition independently.
   *
   * Contrast with send(): send() writes to the peer GPU's staging
   * buffer via NVLink with pipelined flow control and signaling. put()
   * copies within local memory without any signaling or flow control.
   *
   * @param group ThreadGroup for cooperative processing (group-local)
   * @param dst_d Destination pointer (device memory)
   * @param src_d Source pointer (device memory)
   * @param nbytes Number of bytes to copy
   */
  __device__ __forceinline__ void
  put(ThreadGroup& group, char* dst_d, const char* src_d, std::size_t nbytes) {
#if PIPES_IS_DEVICE_COMPILE
    if (nbytes == 0 || dst_d == src_d) {
      return;
    }
    assert_buffer_non_overlap(dst_d, src_d, nbytes, group);
    memcpy_vectorized(dst_d, src_d, nbytes, group);
#endif
  }

  /**
   * signal - Signal peer GPU via NVLink
   *
   * Sends a signal to the peer's Signal object at the specified index.
   * Only the group leader performs the signal after synchronizing all threads.
   *
   * MEMORY SEMANTICS:
   * - Uses release semantics: all prior memory operations from all threads
   *   in the group are guaranteed to be visible to the peer after the signal.
   * - Uses .sys scope for cross-GPU NVLink coherence.
   *
   * @param group ThreadGroup for cooperative processing (leader signals)
   * @param signal_id Index into the signalBuffer array
   * @param op SIGNAL_SET to store value, SIGNAL_ADD to atomically add value
   * @param value The value to set or add to peer's signal counter
   */
  __device__ __forceinline__ void
  signal(ThreadGroup& group, uint64_t signal_id, SignalOp op, uint64_t value) {
    remoteState_.signalBuffer[signal_id].signal(group, op, value);
  }

  /**
   * wait_signal_until - Wait for signal from peer GPU
   *
   * Waits until the local Signal object at the specified index satisfies
   * the given condition. All threads in the group poll the signal.
   *
   * MEMORY SEMANTICS:
   * - Uses acquire semantics: all subsequent memory operations are guaranteed
   *   to see the peer's writes that occurred before their signal.
   * - Uses .sys scope for cross-GPU NVLink coherence.
   *
   * @param group ThreadGroup for cooperative processing
   * @param signal_id Index into the signalBuffer array
   * @param op The comparison operation (CMP_EQ, CMP_GE, etc.)
   * @param value The value to compare against
   */
  __device__ __forceinline__ void wait_signal_until(
      ThreadGroup& group,
      uint64_t signal_id,
      CmpOp op,
      uint64_t value,
      const Timeout& timeout = Timeout()) {
    localState_.signalBuffer[signal_id].wait_until(group, op, value, timeout);
  }

  /**
   * reset_signal - Reset a local signal slot to zero
   *
   * Resets the local signal counter at the specified index to zero.
   * This is safe to call from the receiver side after processing the signal,
   * since the receiver owns the local inbox buffer.
   *
   * The caller must ensure the signal has already been consumed (waited on)
   * before resetting, and that no peer is concurrently signaling the same slot.
   *
   * @param group ThreadGroup for cooperative thread synchronization
   * @param signal_id Index into the signalBuffer array
   */
  __device__ __forceinline__ void reset_signal(
      ThreadGroup& group,
      uint64_t signal_id) {
    if (group.is_leader()) {
      localState_.signalBuffer[signal_id].store(0);
    }
    group.sync();
  }

  /**
   * barrier_sync - Two-sided barrier synchronization with peer GPU
   *
   * Performs a full barrier synchronization between this GPU and the peer GPU
   * over NVLink. Both sides must call this function to complete the barrier.
   *
   * Synchronization protocol:
   * 1. group.sync() - Ensure all local threads have completed prior work
   * 2. Leader signals peer - Writes to peer's barrier state via NVLink
   * 3. Leader waits for peer - Polls local barrier until peer signals
   * 4. group.sync() - Broadcast completion to all threads in the group
   *
   * This provides a full memory fence: all memory operations before the barrier
   * on both GPUs are visible to all threads after the barrier completes.
   *
   * @param group ThreadGroup for cooperative thread synchronization
   * @param barrier_id Index of the barrier to use (must be < numBarriers)
   *
   * All threads in the group must call this function (collective operation).
   * Both GPUs must call with the same barrier_id to synchronize.
   */
  __device__ __forceinline__ void barrier_sync(
      ThreadGroup& group,
      uint64_t barrier_id,
      const Timeout& timeout = Timeout()) {
    // Ensure all prior memory operations are complete
    group.sync();

    // Only global leader performs barrier operations to avoid races where
    // different threads read different counter values.
    if (group.is_leader()) {
      // Signal peer - write to peer's local barrier state via NVLink
      remoteState_.barrierBuffer[barrier_id].arrive();

      // Wait for peer - poll local barrier state until peer signals
      localState_.barrierBuffer[barrier_id].wait(timeout);
    }

    // Ensure all threads wait for leader to complete barrier
    group.sync();
  }

  // ===========================================================================
  // LL128 Protocol Operations
  // ===========================================================================

  /**
   * ll128_send_group — Send data to peer's LL128 buffer via NVLink.
   *
   * Packs user data into LL128 packets and volatile-stores them to the
   * peer's LL128 buffer with inline flag signaling.
   *
   * PRECONDITION: ll128BufferSize > 0 in transport config.
   *
   * @param group   ThreadGroup (auto-converted to warp scope)
   * @param src     Local source buffer (16-byte aligned)
   * @param nbytes  Total bytes (must be a multiple of 16)
   * @param timeout Timeout for flag polling
   */
  __device__ __forceinline__ void ll128_send_group(
      const ThreadGroup& group,
      const char* src,
      size_t nbytes,
      const Timeout& timeout = Timeout()) {
#if PIPES_IS_DEVICE_COMPILE
    PIPES_DEVICE_CHECK(remoteState_.ll128Buffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll128(src, nbytes));

    comms::prims::ll128_send(
        group,
        src,
        nbytes,
        remoteState_.ll128Buffer,
        timeout,
        options_.ll128BufferNumPackets);
#endif
  }

  /**
   * ll128_recv_group — Receive data from local LL128 buffer.
   *
   * Polls the local LL128 buffer (written remotely by peer), reads
   * payload to output buffer, and ACKs with READY_TO_WRITE.
   *
   * PRECONDITION: ll128BufferSize > 0 in transport config.
   *
   * @param group   ThreadGroup (auto-converted to warp scope)
   * @param dst     Local output buffer (16-byte aligned)
   * @param nbytes  Total bytes (must be a multiple of 16)
   * @param timeout Timeout for flag polling
   */
  __device__ __forceinline__ void ll128_recv_group(
      const ThreadGroup& group,
      char* dst,
      size_t nbytes,
      const Timeout& timeout = Timeout()) {
#if PIPES_IS_DEVICE_COMPILE
    PIPES_DEVICE_CHECK(localState_.ll128Buffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll128(dst, nbytes));

    comms::prims::ll128_recv(
        group,
        dst,
        nbytes,
        localState_.ll128Buffer,
        timeout,
        options_.ll128BufferNumPackets);
#endif
  }

  /**
   * ll128_forward_group — Receive from predecessor and forward to successor.
   *
   * Reads from this transport's local LL128 buffer (predecessor wrote here),
   * forwards to successor_transport's remote LL128 buffer, copies payload
   * to local output, and ACKs predecessor.
   *
   * PRECONDITION: ll128BufferSize > 0 in both this and successor transport.
   *
   * @param group                ThreadGroup (auto-converted to warp scope)
   * @param dst                  Local output buffer (16-byte aligned)
   * @param nbytes               Total bytes (must be a multiple of 16)
   * @param successor_transport  Transport for the successor peer
   * @param timeout              Timeout for flag polling
   */
  __device__ __forceinline__ void ll128_forward_group(
      const ThreadGroup& group,
      char* dst,
      size_t nbytes,
      const P2pNvlTransportDevice& successor_transport,
      const Timeout& timeout = Timeout()) {
#if PIPES_IS_DEVICE_COMPILE
    PIPES_DEVICE_CHECK(localState_.ll128Buffer != nullptr);
    PIPES_DEVICE_CHECK(successor_transport.remoteState_.ll128Buffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll128(dst, nbytes));

    // Use the minimum packet count of local and successor buffers.
    // 0 means uncapped (legacy path where buffer is pre-sized to fit).
    const size_t my_packets = options_.ll128BufferNumPackets;
    const size_t succ_packets =
        successor_transport.options_.ll128BufferNumPackets;
    size_t effective_packets = 0;
    if (my_packets > 0 && succ_packets > 0) {
      effective_packets =
          (my_packets < succ_packets) ? my_packets : succ_packets;
    } else if (my_packets > 0) {
      effective_packets = my_packets;
    } else {
      effective_packets = succ_packets;
    }

    comms::prims::ll128_forward(
        group,
        dst,
        nbytes,
        localState_.ll128Buffer,
        successor_transport.remoteState_.ll128Buffer,
        timeout,
        effective_packets);
#endif
  }

  /**
   * send - Independent per-group transfer to peer GPU over NVLink
   *
   * Each group independently sends its own tile of data to the peer GPU's
   * staging buffer via NVLink, with per-group pipelined flow control and
   * signaling. Different groups can call send() with different src/nbytes.
   *
   * @param max_signal_bytes Hint for max bytes between DATA_READY signals.
   *   0 means one signal per slot fill. Capped at per_channel_slot.
   */
  __device__ __forceinline__ void send(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout()) {
#if PIPES_IS_DEVICE_COMPILE
    if (nbytes == 0) {
      return;
    }

    const NvlChannelLayout layout =
        make_channel_layout(group, max_signal_bytes, "send");
    const uint32_t channel = group.group_id;

    NvlChannelState& local_ch = local_channels_[channel];
    NvlChannelState& remote_ch = remote_channels_[channel];

    const char* __restrict__ srcPtr = reinterpret_cast<const char*>(src);
    char* __restrict__ stagBuf = remoteState_.dataBuffer;

    const uint64_t baseByte = static_cast<uint64_t>(local_ch.send_cursor);

    const std::size_t payloadProtocolBytes = align_tile_protocol_bytes(nbytes);
    const std::size_t protocolTailPadding = tail_padding_for_signal_granularity(
        baseByte, max_signal_bytes, layout.perChannelSlot, nbytes);
    const std::size_t protocolBytes =
        payloadProtocolBytes + protocolTailPadding;
    for (std::size_t dataOff = 0; dataOff < payloadProtocolBytes;) {
      const uint64_t streamStart = baseByte + dataOff;
      const NvlPipelinePosition position = layout.position(streamStart);
      const std::size_t dataRemaining = payloadProtocolBytes - dataOff;
      std::size_t copyBytes = layout.effectiveChunk < dataRemaining
          ? layout.effectiveChunk
          : dataRemaining;
      copyBytes = copyBytes < position.slotRemaining ? copyBytes
                                                     : position.slotRemaining;
      const bool isFinalChunk = dataOff + copyBytes >= payloadProtocolBytes;
      const uint64_t streamEnd = streamStart + copyBytes;
      const uint64_t protocolStreamEnd =
          streamEnd + (isFinalChunk ? protocolTailPadding : 0);

      if (protocolStreamEnd > layout.pipelineBytes) {
        local_ch.slot_free.wait_until(
            group,
            CmpOp::CMP_GE,
            protocolStreamEnd - layout.pipelineBytes,
            timeout);
      }

      const std::size_t validBytes =
          valid_payload_bytes(dataOff, copyBytes, nbytes);
      if (validBytes > 0) {
        memcpy_vectorized(
            layout.staging_ptr(stagBuf, position),
            srcPtr + dataOff,
            validBytes,
            group);
      }

      group.sync();
      if (group.is_leader()) {
        remote_ch.data_ready.signal(SignalOp::SIGNAL_SET, protocolStreamEnd);
      }
      dataOff += copyBytes;
    }

    if (group.is_leader()) {
      local_ch.send_cursor = static_cast<int64_t>(baseByte + protocolBytes);
    }
    group.sync();
#endif
  }

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void recv(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      [[maybe_unused]] const Timeout& timeout = Timeout(),
      [[maybe_unused]] Args... args) {
#if PIPES_IS_DEVICE_COMPILE
    if (nbytes == 0) {
      return;
    }

    const NvlChannelLayout layout =
        make_channel_layout(group, max_signal_bytes, "recv");
    const uint32_t channel = group.group_id;

    NvlChannelState& local_ch = local_channels_[channel];
    NvlChannelState& remote_ch = remote_channels_[channel];

    char* __restrict__ dstPtr = reinterpret_cast<char*>(dst);
    char* __restrict__ stagBuf = localState_.dataBuffer;

    const uint64_t baseByte = static_cast<uint64_t>(local_ch.recv_cursor);

    const std::size_t payloadProtocolBytes = align_tile_protocol_bytes(nbytes);
    const std::size_t protocolTailPadding = tail_padding_for_signal_granularity(
        baseByte, max_signal_bytes, layout.perChannelSlot, nbytes);
    const std::size_t protocolBytes =
        payloadProtocolBytes + protocolTailPadding;
    for (std::size_t dataOff = 0; dataOff < payloadProtocolBytes;) {
      const uint64_t streamStart = baseByte + dataOff;
      const NvlPipelinePosition position = layout.position(streamStart);
      const std::size_t dataRemaining = payloadProtocolBytes - dataOff;
      std::size_t copyBytes = layout.effectiveChunk < dataRemaining
          ? layout.effectiveChunk
          : dataRemaining;
      copyBytes = copyBytes < position.slotRemaining ? copyBytes
                                                     : position.slotRemaining;
      const bool isFinalChunk = dataOff + copyBytes >= payloadProtocolBytes;
      const uint64_t streamEnd = streamStart + copyBytes;
      const uint64_t protocolStreamEnd =
          streamEnd + (isFinalChunk ? protocolTailPadding : 0);

      local_ch.data_ready.wait_until(group, CmpOp::CMP_GE, streamEnd, timeout);

      const std::size_t validBytes =
          valid_payload_bytes(dataOff, copyBytes, nbytes);
      if (validBytes > 0) {
        CopyOp::recv(
            dstPtr + dataOff,
            layout.staging_ptr(stagBuf, position),
            validBytes,
            group,
            dataOff,
            args...);
      }

      group.sync();
      if (group.is_leader()) {
        if (position.chunkOff + copyBytes == layout.perChannelSlot ||
            isFinalChunk) {
          remote_ch.slot_free.signal(SignalOp::SIGNAL_SET, protocolStreamEnd);
        }
      }
      dataOff += copyBytes;
    }

    if (group.is_leader()) {
      local_ch.recv_cursor = static_cast<int64_t>(baseByte + protocolBytes);
    }
    group.sync();
#endif
  }

  /**
   * forward - Independent per-channel fused receive-and-forward (tile-style)
   *
   * Each group reads its own channel from this transport's predecessor staging
   * buffer and writes to two destinations simultaneously: the local user
   * buffer (dst) and the successor's remote staging buffer. Halves read
   * bandwidth vs sequential recv + send.
   *
   * PRECONDITIONS:
   * - `this` transport is connected to the predecessor (data arrives in
   *   this->localState_.dataBuffer)
   * - `successor` transport is connected to the next rank (data forwarded
   *   to successor.remoteState_.dataBuffer)
   * - Both transports must have matching options (dataBufferSize,
   *   per_channel_slot, max_num_channels, pipelineDepth).
   *
   * @param group ThreadGroup for cooperative processing (group-local)
   * @param dst Local user buffer to copy data into
   * @param nbytes Number of bytes to forward
   * @param successor Transport to the next rank in the ring
   * @param max_signal_bytes Hint for max bytes between signals.
   *   0 means one signal per slot fill. Capped at per_channel_slot.
   */
  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void forward(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      P2pNvlTransportDevice& successor,
      std::size_t max_signal_bytes = 0,
      [[maybe_unused]] const Timeout& timeout = Timeout(),
      [[maybe_unused]] Args... args) {
#if PIPES_IS_DEVICE_COMPILE
    if (nbytes == 0) {
      return;
    }

    const NvlChannelLayout layout =
        make_channel_layout(group, max_signal_bytes, "forward");
    const uint32_t channel = group.group_id;

    // Recv side: this transport (predecessor → me).
    NvlChannelState& recv_local_ch = local_channels_[channel];
    NvlChannelState& recv_remote_ch = remote_channels_[channel];
    // Send side: successor transport (me → successor).
    NvlChannelState& send_local_ch = successor.local_channels_[channel];
    NvlChannelState& send_remote_ch = successor.remote_channels_[channel];

    char* __restrict__ dstPtr = reinterpret_cast<char*>(dst);
    // Predecessor's staging buffer (local read)
    char* __restrict__ recvBuf = localState_.dataBuffer;
    // Successor's staging buffer (NVLink write)
    char* __restrict__ sendBuf = successor.remoteState_.dataBuffer;

    const uint64_t recvBaseByte =
        static_cast<uint64_t>(recv_local_ch.recv_cursor);
    const uint64_t sendBaseByte =
        static_cast<uint64_t>(send_local_ch.send_cursor);

    const std::size_t payloadProtocolBytes = align_tile_protocol_bytes(nbytes);
    const std::size_t recvProtocolTailPadding =
        tail_padding_for_signal_granularity(
            recvBaseByte, max_signal_bytes, layout.perChannelSlot, nbytes);
    const std::size_t sendProtocolTailPadding =
        tail_padding_for_signal_granularity(
            sendBaseByte, max_signal_bytes, layout.perChannelSlot, nbytes);
    const std::size_t recvProtocolBytes =
        payloadProtocolBytes + recvProtocolTailPadding;
    const std::size_t sendProtocolBytes =
        payloadProtocolBytes + sendProtocolTailPadding;
    for (std::size_t dataOff = 0; dataOff < payloadProtocolBytes;) {
      const uint64_t recvStreamStart = recvBaseByte + dataOff;
      const NvlPipelinePosition recvPosition = layout.position(recvStreamStart);
      const uint64_t sendStreamStart = sendBaseByte + dataOff;
      const NvlPipelinePosition sendPosition = layout.position(sendStreamStart);
      const std::size_t dataRemaining = payloadProtocolBytes - dataOff;
      std::size_t copyBytes = layout.effectiveChunk < dataRemaining
          ? layout.effectiveChunk
          : dataRemaining;
      copyBytes = copyBytes < recvPosition.slotRemaining
          ? copyBytes
          : recvPosition.slotRemaining;
      copyBytes = copyBytes < sendPosition.slotRemaining
          ? copyBytes
          : sendPosition.slotRemaining;
      const bool isFinalChunk = dataOff + copyBytes >= payloadProtocolBytes;
      const uint64_t recvStreamEnd = recvStreamStart + copyBytes;
      const uint64_t sendStreamEnd = sendStreamStart + copyBytes;
      const uint64_t recvProtocolStreamEnd =
          recvStreamEnd + (isFinalChunk ? recvProtocolTailPadding : 0);
      const uint64_t sendProtocolStreamEnd =
          sendStreamEnd + (isFinalChunk ? sendProtocolTailPadding : 0);

      // 1. Wait for predecessor data to be ready (recv side, every step).
      recv_local_ch.data_ready.wait_until(
          group, CmpOp::CMP_GE, recvStreamEnd, timeout);

      // 2. Wait for successor's staging slot to be free once we have wrapped
      //    around the pipeline.
      if (sendProtocolStreamEnd > layout.pipelineBytes) {
        send_local_ch.slot_free.wait_until(
            group,
            CmpOp::CMP_GE,
            sendProtocolStreamEnd - layout.pipelineBytes,
            timeout);
      }

      // 3. Dual-dst copy: predecessor staging → local user buf +
      //    successor remote staging
      const std::size_t validBytes =
          valid_payload_bytes(dataOff, copyBytes, nbytes);
      if (validBytes > 0) {
        CopyOp::forward(
            dstPtr ? dstPtr + dataOff : nullptr,
            layout.staging_ptr(sendBuf, sendPosition),
            layout.staging_ptr(recvBuf, recvPosition),
            validBytes,
            group,
            dataOff,
            args...);
      }

      group.sync();
      if (group.is_leader()) {
        // 4. Signal successor that data is ready (send semantic: every step).
        send_remote_ch.data_ready.signal(
            SignalOp::SIGNAL_SET, sendProtocolStreamEnd);

        // 5. ACK predecessor that buffer is free (recv semantic: only at
        //    slot boundaries).
        if (recvPosition.chunkOff + copyBytes == layout.perChannelSlot ||
            isFinalChunk) {
          recv_remote_ch.slot_free.signal(
              SignalOp::SIGNAL_SET, recvProtocolStreamEnd);
        }
      }
      dataOff += copyBytes;
    }

    if (group.is_leader()) {
      recv_local_ch.recv_cursor =
          static_cast<int64_t>(recvBaseByte + recvProtocolBytes);
      send_local_ch.send_cursor =
          static_cast<int64_t>(sendBaseByte + sendProtocolBytes);
    }
    group.sync();
#endif
  }

  // Test-only accessors for poking at channel state.
  __host__ __device__ NvlChannelState& local_channel_at(int channel) {
    return local_channels_[channel];
  }
  __host__ __device__ NvlChannelState& remote_channel_at(int channel) {
    return remote_channels_[channel];
  }

  // Device accessors for 2D tile kernel (inlined pipeline)
  __host__ __device__ const P2pNvlTransportOptions& options() const {
    return options_;
  }
  __device__ LocalState& local_state() {
    return localState_;
  }
  __device__ RemoteState& remote_state() {
    return remoteState_;
  }

  // ===========================================================================
  // LL Protocol Operations
  // ===========================================================================

  /**
   * ll_send — Send data to peer's LL buffer via NVLink.
   *
   * Packs user data into LL lines and volatile-stores them to the
   * peer's LL buffer with inline flag signaling.
   *
   * PRECONDITION: llBufferSize > 0 in transport config.
   *
   * @param group         ThreadGroup (auto-converted to warp scope)
   * @param src           Local source buffer (8-byte aligned)
   * @param nbytes        Total bytes (must be a multiple of 8)
   * @param active_groups Number of groups calling concurrently.
   *   0 = default to max groups the LL buffer can support.
   *   >0 = explicit group count; buffer partitioned per group.group_id.
   * @param timeout       Timeout for flag polling
   */
  __device__ __forceinline__ void ll_send(
      const ThreadGroup& group,
      const char* src,
      size_t nbytes,
      int active_groups = 0,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__ // NVIDIA-only: depends on ll_send/ll_recv/ll128_* not yet
                     // ported to AMD
    PIPES_DEVICE_CHECK(remoteState_.llBuffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll(src, nbytes, options_.llBufferNumLines));

    const int maxGroups =
        (options_.llBufferNumLines >= static_cast<size_t>(kLlLinesPerWarp))
        ? static_cast<int>(options_.llBufferNumLines / kLlLinesPerWarp)
        : 1;
    const int effActive = active_groups > 0 ? active_groups : maxGroups;

    PIPES_DEVICE_CHECK(static_cast<uint32_t>(effActive) <= group.total_groups);
    PIPES_DEVICE_CHECK(group.group_id < static_cast<uint32_t>(effActive));

    const size_t perGroupLines = options_.llBufferNumLines / effActive;
    if (effActive > 1 && options_.llBufferNumLines > 0) {
      PIPES_DEVICE_CHECK(perGroupLines >= kLlLinesPerWarp);
    }
    const size_t bufferOffset = group.group_id * perGroupLines;

    comms::prims::ll_send(
        group,
        src,
        nbytes,
        remoteState_.llBuffer + bufferOffset,
        timeout,
        perGroupLines);
#else
    (void)group;
    (void)src;
    (void)nbytes;
    (void)active_groups;
    (void)timeout;
#endif
  }

  /**
   * ll_recv — Receive data from local LL buffer.
   *
   * Polls the local LL buffer (written remotely by peer), reads
   * payload to output buffer, and ACKs with kLlReadyToWrite.
   *
   * PRECONDITION: llBufferSize > 0 in transport config.
   *
   * @param group         ThreadGroup (auto-converted to warp scope)
   * @param dst           Local output buffer (8-byte aligned)
   * @param nbytes        Total bytes (must be a multiple of 8)
   * @param active_groups Number of groups calling concurrently.
   *   0 = default to max groups the LL buffer can support.
   *   >0 = explicit group count; buffer partitioned per group.group_id.
   * @param timeout       Timeout for flag polling
   */
  __device__ __forceinline__ void ll_recv(
      const ThreadGroup& group,
      char* dst,
      size_t nbytes,
      int active_groups = 0,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__ // NVIDIA-only: depends on ll_send/ll_recv/ll128_* not yet
                     // ported to AMD
    PIPES_DEVICE_CHECK(localState_.llBuffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll(dst, nbytes, options_.llBufferNumLines));

    const int maxGroups =
        (options_.llBufferNumLines >= static_cast<size_t>(kLlLinesPerWarp))
        ? static_cast<int>(options_.llBufferNumLines / kLlLinesPerWarp)
        : 1;
    const int effActive = active_groups > 0 ? active_groups : maxGroups;

    PIPES_DEVICE_CHECK(static_cast<uint32_t>(effActive) <= group.total_groups);
    PIPES_DEVICE_CHECK(group.group_id < static_cast<uint32_t>(effActive));

    const size_t perGroupLines = options_.llBufferNumLines / effActive;
    if (effActive > 1 && options_.llBufferNumLines > 0) {
      PIPES_DEVICE_CHECK(perGroupLines >= kLlLinesPerWarp);
    }
    const size_t bufferOffset = group.group_id * perGroupLines;

    comms::prims::ll_recv(
        group,
        dst,
        nbytes,
        localState_.llBuffer + bufferOffset,
        timeout,
        perGroupLines);
#else
    (void)group;
    (void)dst;
    (void)nbytes;
    (void)active_groups;
    (void)timeout;
#endif
  }

  /**
   * ll_forward — Receive from predecessor and forward to successor.
   *
   * Reads from this transport's local LL buffer (predecessor wrote here),
   * forwards to successor_transport's remote LL buffer, copies payload
   * to local output, and ACKs predecessor.
   *
   * PRECONDITION: llBufferSize > 0 in both this and successor transport.
   *
   * @param group                ThreadGroup (auto-converted to warp scope)
   * @param dst                  Local output buffer (8-byte aligned)
   * @param nbytes               Total bytes (must be a multiple of 8)
   * @param successor_transport  Transport for the successor peer
   * @param active_groups        Number of groups calling concurrently.
   *   0 = default to max groups the LL buffer can support.
   *   >0 = explicit group count; buffer partitioned per group.group_id.
   * @param timeout              Timeout for flag polling
   */
  __device__ __forceinline__ void ll_forward(
      const ThreadGroup& group,
      char* dst,
      size_t nbytes,
      const P2pNvlTransportDevice& successor_transport,
      int active_groups = 0,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__ // NVIDIA-only: depends on ll_send/ll_recv/ll128_* not yet
                     // ported to AMD
    PIPES_DEVICE_CHECK(localState_.llBuffer != nullptr);
    PIPES_DEVICE_CHECK(successor_transport.remoteState_.llBuffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll(dst, nbytes, options_.llBufferNumLines));

    const int myMax =
        (options_.llBufferNumLines >= static_cast<size_t>(kLlLinesPerWarp))
        ? static_cast<int>(options_.llBufferNumLines / kLlLinesPerWarp)
        : 1;
    const int succMax = (successor_transport.options_.llBufferNumLines >=
                         static_cast<size_t>(kLlLinesPerWarp))
        ? static_cast<int>(
              successor_transport.options_.llBufferNumLines / kLlLinesPerWarp)
        : 1;
    const int maxGroups = myMax < succMax ? myMax : succMax;
    const int effActive = active_groups > 0 ? active_groups : maxGroups;

    PIPES_DEVICE_CHECK(static_cast<uint32_t>(effActive) <= group.total_groups);
    PIPES_DEVICE_CHECK(group.group_id < static_cast<uint32_t>(effActive));

    const size_t myPerGroup = options_.llBufferNumLines / effActive;
    const size_t succPerGroup =
        successor_transport.options_.llBufferNumLines / effActive;
    if (effActive > 1) {
      if (options_.llBufferNumLines > 0) {
        PIPES_DEVICE_CHECK(myPerGroup >= kLlLinesPerWarp);
      }
      if (successor_transport.options_.llBufferNumLines > 0) {
        PIPES_DEVICE_CHECK(succPerGroup >= kLlLinesPerWarp);
      }
    }
    const size_t myOffset = group.group_id * myPerGroup;
    const size_t succOffset = group.group_id * succPerGroup;

    // Asymmetric buffer sizing: 0 means "pre-sized to fit message."
    // Use the non-zero value when only one side is chunked.
    size_t effectiveLines;
    if (myPerGroup > 0 && succPerGroup > 0) {
      effectiveLines = myPerGroup < succPerGroup ? myPerGroup : succPerGroup;
    } else if (myPerGroup > 0) {
      effectiveLines = myPerGroup;
    } else {
      effectiveLines = succPerGroup;
    }

    comms::prims::ll_forward(
        group,
        dst,
        nbytes,
        localState_.llBuffer + myOffset,
        successor_transport.remoteState_.llBuffer + succOffset,
        timeout,
        effectiveLines);
#else
    (void)group;
    (void)dst;
    (void)nbytes;
    (void)successor_transport;
    (void)active_groups;
    (void)timeout;
#endif
  }

  /**
   * get_ll_buffer_num_lines — Get the number of LL lines in the buffer.
   */
  __device__ __forceinline__ size_t get_ll_buffer_num_lines() const {
    return options_.llBufferNumLines;
  }

 private:
  struct NvlPipelinePosition {
    std::size_t slotOff;
    std::size_t chunkOff;
    std::size_t slotRemaining;
  };

  struct NvlChannelLayout {
    std::size_t slotSize;
    std::size_t perChannelSlot;
    std::size_t stagingOff;
    std::size_t pipelineBytes;
    std::size_t effectiveChunk;

    __device__ __forceinline__ NvlPipelinePosition
    position(uint64_t streamByte) const {
      const std::size_t pipelineOff =
          static_cast<std::size_t>(streamByte % pipelineBytes);
      const std::size_t slot = pipelineOff / perChannelSlot;
      const std::size_t chunkOff = pipelineOff - slot * perChannelSlot;
      return {
          slot * slotSize,
          chunkOff,
          perChannelSlot - chunkOff,
      };
    }

    __device__ __forceinline__ char* staging_ptr(
        char* __restrict__ base,
        const NvlPipelinePosition& position) const {
      return base + position.slotOff + stagingOff + position.chunkOff;
    }
  };

  __device__ __forceinline__ NvlChannelLayout make_channel_layout(
      const ThreadGroup& group,
      std::size_t maxSignalBytes,
      const char* op) const {
    const int maxChannels = options_.max_num_channels;
    if (group.total_groups > static_cast<uint32_t>(maxChannels)) {
      printf(
          "%s: group.total_groups=%u > max_num_channels=%d. "
          "Channel arrays would be accessed out of bounds.\n",
          op,
          group.total_groups,
          maxChannels);
      PIPES_DEVICE_TRAP();
    }

    const std::size_t slotSize = options_.dataBufferSize;
    const std::size_t perChannelSlot = options_.per_channel_slot;
    if (perChannelSlot == 0) {
      printf(
          "%s: per_channel_slot is 0 (dataBufferSize=%llu, "
          "max_num_channels=%d). Set perChannelSize when channels are "
          "enabled.\n",
          op,
          (unsigned long long)slotSize,
          maxChannels);
      PIPES_DEVICE_TRAP();
    }

    const std::size_t chunkSize =
        maxSignalBytes > 0 && maxSignalBytes < perChannelSlot
        ? (maxSignalBytes & ~15ULL)
        : perChannelSlot;
    return {
        slotSize,
        perChannelSlot,
        group.group_id * perChannelSlot,
        perChannelSlot * options_.pipelineDepth,
        chunkSize > 0 ? chunkSize : perChannelSlot,
    };
  }

  __device__ __forceinline__ static std::size_t align_tile_protocol_bytes(
      std::size_t nbytes) {
    return (nbytes + 15ULL) & ~15ULL;
  }

  __device__ __forceinline__ static uint64_t round_up_to_multiple(
      uint64_t value,
      std::size_t alignment) {
    if (alignment == 0) {
      return value;
    }
    const uint64_t alignment64 = static_cast<uint64_t>(alignment);
    return ((value + alignment64 - 1) / alignment64) * alignment64;
  }

  __device__ __forceinline__ static std::size_t signal_alignment(
      std::size_t maxSignalBytes,
      std::size_t perChannelSlot) {
    const bool usesPartialSlot =
        maxSignalBytes > 0 && maxSignalBytes < perChannelSlot;
    std::size_t alignment =
        usesPartialSlot ? (maxSignalBytes & ~15ULL) : perChannelSlot;
    return alignment == 0 ? perChannelSlot : alignment;
  }

  __device__ __forceinline__ static std::size_t
  tail_padding_for_signal_granularity(
      uint64_t baseByte,
      std::size_t maxSignalBytes,
      std::size_t perChannelSlot,
      std::size_t payloadBytes) {
    const std::size_t alignment =
        signal_alignment(maxSignalBytes, perChannelSlot);
    if (alignment == 0) {
      return 0;
    }
    const uint64_t payloadEnd =
        baseByte + align_tile_protocol_bytes(payloadBytes);
    return static_cast<std::size_t>(
        round_up_to_multiple(payloadEnd, alignment) - payloadEnd);
  }

  __device__ __forceinline__ static std::size_t valid_payload_bytes(
      std::size_t byteOffset,
      std::size_t chunkBytes,
      std::size_t payloadBytes) {
    if (byteOffset >= payloadBytes) {
      return 0;
    }
    const std::size_t remaining = payloadBytes - byteOffset;
    return chunkBytes < remaining ? chunkBytes : remaining;
  }

  const int myRank_{-1};
  const int peerRank_{-1};
  const P2pNvlTransportOptions options_{};
  LocalState localState_;
  RemoteState remoteState_;
  // Per-channel protocol state. Length = options_.max_num_channels.
  // local_channels_: this rank's endpoint; remote sender / recv write
  //   into it via NVLink (data_ready / slot_free fields).
  // remote_channels_: IPC-mapped pointer to the remote rank's local_channels_
  //   array; this rank's send / recv write into it to signal the remote rank.
  NvlChannelState* local_channels_{nullptr};
  NvlChannelState* remote_channels_{nullptr};
};

} // namespace comms::prims
