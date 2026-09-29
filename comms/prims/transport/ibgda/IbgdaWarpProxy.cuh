// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::prims {

inline constexpr uint32_t kIbgdaWarpProxyMaxPipelineDepth = 16;

} // namespace comms::prims

#if defined(__CUDACC__) && !defined(__HIP_PLATFORM_AMD__)

#include <cuda/atomic>

#include <cstddef>

#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/core/MemcpyCopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/P2pIbTransportProgressImpl.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims {

/**
 * Runs exact-length IBGDA data transfers over full-slot control streams with
 * one trailing IB proxy warp.
 *
 * A lazy-bound stream is keyed by (transport, logical channel), with an
 * independent SPSC stream in each direction. Workers publish monotonically
 * increasing absolute slot ordinals; the IB proxy warp derives the physical
 * pipeline slot from each ordinal and owns remote readiness polling, WQE
 * posting, receive credits, and completion retirement. A run exclusively owns
 * every stream passed through Ops until run() returns; transport objects must
 * have shared or global lifetime. Ops calls may return after work is published.
 * workerFn must use Ops::group() for synchronization and issue Ops calls
 * collectively from that single producer group; block-wide barriers and
 * concurrent subgroup issuers are unsupported.
 *
 * Full-slot is a control-plane invariant: maxSignalBytes must be zero and each
 * helper callback must describe one complete protocol slot. The worker may
 * publish fewer data bytes for the final slot, and the proxy transfers only
 * those bytes. Signals, credits, offsets, and ordinals remain full-slot.
 *
 * Completion has two shapes, and they differ in what they promise:
 *
 * - **Normal completion**: run() returns after every produced send is posted
 *   and every consumed receive is credited. Send completions are retired
 *   opportunistically to advance the slot-reuse frontier; the transport-owned
 *   tail may remain outstanding across runs.
 * - **Abort completion**: run() may return with a stream's posted frontier
 *   behind produced and/or credited behind requested. Pending publications are
 *   deliberately abandoned and their credits never issued. Drain is therefore
 *   not a postcondition of run() after abort; termination is. Recovery is
 *   `reconfigure()`, as everywhere else in the abort contract.
 */
template <uint32_t WorkerThreads, uint32_t MaxStreams = 8>
class IbgdaWarpProxy {
 private:
  // Barrier 0 is reserved for full-block synchronization in this kernel.
  static constexpr uint32_t kWorkerNamedBarrierId = 1;

  static_assert(WorkerThreads > 0);
  static_assert(MaxStreams > 0);
  static_assert(WorkerThreads % comms::device::kWarpSize == 0);

  static constexpr uint32_t kProxyThreads = comms::device::kWarpSize;
  static_assert(
      WorkerThreads + kProxyThreads <= 1024,
      "IbgdaWarpProxy exceeds the CUDA threads-per-block limit");

  static constexpr uint64_t kInvalidSequence = ~uint64_t{0};
  static constexpr uint32_t kInvalidStream = ~uint32_t{0};

  struct alignas(16) TxStream {
    std::size_t publishedBytes[kIbgdaWarpProxyMaxPipelineDepth];
    // Worker publishes produced. Proxy publishes posted after recording the
    // transport completion ticket, and advances retired only to make a
    // physical staging slot reusable.
    alignas(8) uint64_t produced;
    alignas(8) uint64_t posted;
    alignas(8) uint64_t retired;
    alignas(4) uint32_t bound;
  };

  struct alignas(16) RxStream {
    // Worker: requested/consumed/hasExpectedEnd. Proxy: arrived/credited.
    // In planned mode, requested is the exact exclusive bound rather than
    // speculative lookahead.
    alignas(8) uint64_t requested;
    alignas(8) uint64_t arrived;
    alignas(8) uint64_t consumed;
    alignas(8) uint64_t credited;
    alignas(4) uint32_t bound;
    uint32_t hasExpectedEnd;
  };

  struct alignas(16) DuplexStream {
    P2pIbgdaTransportDevice* transport;
    std::size_t slotBytes;
    uint32_t channel;
    uint32_t pipelineDepth;
    TxStream tx;
    RxStream rx;
  };

 public:
  static constexpr uint32_t kBlockThreads = WorkerThreads + kProxyThreads;

  struct alignas(16) SharedState {
    DuplexStream streams[MaxStreams];
    alignas(4) uint32_t streamCount;
    alignas(4) uint32_t producerDone;
  };

  class Ops {
   public:
    static constexpr uint32_t kWorkerThreads = WorkerThreads;
    static constexpr bool kWarpProxy = true;
    // Every ops policy names its wire format so callers can size transfers by
    // it (see BlockingIbOps). This proxy has no LL counterpart, so it is fixed.
    using WireProto = protocol::Simple;

    __device__ __forceinline__ ThreadGroup& group() {
      return workers_;
    }

    __device__ __forceinline__ void declare_expected_recvs(
        P2pIbgdaTransportDevice& transport,
        uint64_t totalRecvs) {
      IbgdaWarpProxy::validate_pipeline_depth(transport, workers_);
      activeRecvStream_ = IbgdaWarpProxy::declare_expected_recvs(
          storage_, transport, workers_, totalRecvs, activeRecvStream_);
    }

    template <typename CopyOp = Memcpy, typename... Args>
    __device__ __forceinline__ void send(
        P2pIbgdaTransportDevice& transport,
        const void* src,
        std::size_t nbytes,
        std::size_t maxSignalBytes = 0,
        Args... args) {
      static_assert(
          !detail::copyop_variable_size_v<CopyOp>,
          "IbgdaWarpProxy supports fixed-size CopyOps only");
      IbgdaWarpProxy::validate_pipeline_depth(transport, workers_);
      IbgdaWarpProxy::validate_full_slot_control(maxSignalBytes, workers_);
      detail::send_impl<P2pIbgdaTransportDevice, CopyOp>(
          transport,
          workers_,
          this,
          src,
          nbytes,
          maxSignalBytes,
          abortDevice_,
          nullptr,
          args...);
    }

    template <typename CopyOp = Memcpy, typename... Args>
    __device__ __forceinline__ void recv(
        P2pIbgdaTransportDevice& transport,
        void* dst,
        std::size_t nbytes,
        std::size_t maxSignalBytes = 0,
        Args... args) {
      static_assert(
          !detail::copyop_variable_size_v<CopyOp>,
          "IbgdaWarpProxy supports fixed-size CopyOps only");
      IbgdaWarpProxy::validate_pipeline_depth(transport, workers_);
      IbgdaWarpProxy::validate_full_slot_control(maxSignalBytes, workers_);
      detail::recv_impl<P2pIbgdaTransportDevice, CopyOp>(
          transport,
          workers_,
          this,
          dst,
          nbytes,
          maxSignalBytes,
          abortDevice_,
          nullptr,
          args...);
    }

    template <typename CopyOp = Memcpy, typename... Args>
    __device__ __forceinline__ void forward(
        P2pIbgdaTransportDevice& prev,
        void* dst,
        P2pIbgdaTransportDevice& next,
        std::size_t nbytes,
        std::size_t maxSignalBytes = 0,
        Args... args) {
      static_assert(
          !detail::copyop_variable_size_v<CopyOp>,
          "IbgdaWarpProxy supports fixed-size CopyOps only");
      IbgdaWarpProxy::validate_pipeline_depth(prev, workers_);
      IbgdaWarpProxy::validate_pipeline_depth(next, workers_);
      IbgdaWarpProxy::validate_full_slot_control(maxSignalBytes, workers_);
      detail::forward_impl<CopyOp, P2pIbgdaTransportDevice>(
          prev,
          workers_,
          this,
          dst,
          next,
          nbytes,
          maxSignalBytes,
          abortDevice_,
          nullptr,
          nullptr,
          args...);
    }

    // Waits until the proxy retires this physical slot's prior use. Returns
    // true only if abort stops that wait before the slot becomes reusable.
    [[nodiscard]] __device__ __forceinline__ bool prepare_send_slot(
        P2pIbgdaTransportDevice& transport,
        ThreadGroup& workers,
        uint32_t slot,
        uint64_t generation,
        const AbortDevice& abortDevice) {
      const uint32_t stream = IbgdaWarpProxy::bind_send_stream(
          storage_, transport, workers, slot, generation, activeSendStream_);
      activeSendStream_ = stream;
      return IbgdaWarpProxy::wait_send_slot_reusable(
          storage_, stream, workers, slot, generation, abortDevice);
    }

    __device__ __forceinline__ void submit_send(
        P2pIbgdaTransportDevice& transport,
        ThreadGroup& workers,
        const IbgdaLocalBuffer& source,
        std::size_t remoteOffset,
        std::size_t bytes,
        std::size_t protocolBytes,
        uint64_t slotFreeExpected,
        uint32_t slot,
        uint64_t generation) {
      const uint32_t stream = activeSendStream_;
      IbgdaWarpProxy::validate_stream_key(storage_, stream, transport, workers);
      // Once staging is complete, publish the local stream ordinal even if an
      // abort raced with this call. Keeping the SPSC counter gap-free lets the
      // worker unwind at its next wait; the proxy independently gates every
      // peer-visible WQE and may abandon this pending publication on abort.
      IbgdaWarpProxy::publish_send(
          storage_,
          stream,
          transport,
          workers,
          source,
          remoteOffset,
          bytes,
          protocolBytes,
          slotFreeExpected,
          slot,
          generation);
    }

    __device__ __forceinline__ uint64_t wait_recv(
        P2pIbgdaTransportDevice& transport,
        ThreadGroup& workers,
        std::size_t protocolBytes,
        const AbortDevice& abortDevice) {
      const uint32_t stream = IbgdaWarpProxy::bind_recv_stream(
          storage_, transport, workers, protocolBytes, activeRecvStream_);
      activeRecvStream_ = stream;
      return IbgdaWarpProxy::request_and_wait_recv(
          storage_, stream, workers, abortDevice);
    }

    [[nodiscard]] __device__ __forceinline__ bool recv_wait_succeeded(
        uint64_t sequence) const {
      return sequence != kInvalidSequence;
    }

    __device__ __forceinline__ void publish_recv(
        P2pIbgdaTransportDevice& transport,
        ThreadGroup& workers,
        std::size_t protocolBytes,
        uint64_t sequence) {
      // Advancing `recv.consumed` is what lets the proxy warp emit SLOT_FREE
      // for this chunk, so it must not happen for a receive that never landed.
      //
      // The sentinel carries that decision from `wait_recv()`, which already
      // made it group-uniformly through its own barrier. Re-asking here would
      // cost another block-wide barrier per chunk to learn something already
      // known, so this is a register compare.
      if (sequence == kInvalidSequence) {
        return;
      }
      const uint32_t stream = activeRecvStream_;
      IbgdaWarpProxy::validate_stream_key(storage_, stream, transport, workers);
      IbgdaWarpProxy::validate_recv_slot_bytes(
          storage_, stream, workers, protocolBytes);
      IbgdaWarpProxy::publish_recv_consumed(
          storage_, stream, workers, sequence);
    }

   private:
    friend class IbgdaWarpProxy<WorkerThreads, MaxStreams>;

    __device__ Ops(
        SharedState& storage,
        ThreadGroup workers,
        const AbortDevice& abortDevice)
        : storage_(storage), workers_(workers), abortDevice_(abortDevice) {}

    SharedState& storage_;
    ThreadGroup workers_;
    AbortDevice abortDevice_;
    uint32_t activeSendStream_{kInvalidStream};
    uint32_t activeRecvStream_{kInvalidStream};
  };

  template <typename WorkerFn>
  __device__ __forceinline__ static void run(
      SharedState& storage,
      ThreadGroup fullBlock,
      const AbortDevice& abortDevice,
      WorkerFn&& workerFn) {
    validate_block(fullBlock);
    initialize(storage, fullBlock);

    if (fullBlock.thread_id_in_group < WorkerThreads) {
      ThreadGroup workers = make_worker_group(fullBlock);
      Ops ops(storage, workers, abortDevice);
      workerFn(ops);
      finish_workers(storage, workers);
    } else {
      ThreadGroup proxy = make_proxy_group(fullBlock);
      run_proxy(storage, proxy, fullBlock, abortDevice);
    }

    fullBlock.sync();
  }

 private:
  __device__ __forceinline__ static ThreadGroup make_worker_group(
      const ThreadGroup& fullBlock) {
    return ThreadGroup{
        .thread_id_in_group = fullBlock.thread_id_in_group,
        .group_size = WorkerThreads,
        .group_id = fullBlock.group_id,
        .block_id = fullBlock.block_id,
        .total_groups = fullBlock.total_groups,
        .scope = SyncScope::MULTIWARP,
        .barrier_id = kWorkerNamedBarrierId,
    };
  }

  using BlockAtomicU32 = cuda::atomic_ref<uint32_t, cuda::thread_scope_block>;
  using BlockAtomicU64 = cuda::atomic_ref<uint64_t, cuda::thread_scope_block>;

  __device__ __forceinline__ static void validate_block(
      const ThreadGroup& fullBlock) {
    const uint32_t expected = WorkerThreads + kProxyThreads;
    const bool valid = blockDim.y == 1 && blockDim.z == 1 &&
        fullBlock.group_size == expected && blockDim.x == expected &&
        fullBlock.thread_id_in_group == threadIdx.x &&
        fullBlock.scope == SyncScope::BLOCK;
    if (!valid) {
      if (fullBlock.is_leader()) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy requires exactly %u workers and "
            "one trailing proxy warp; block=(%u,%u,%u) group=%u\n",
            WorkerThreads,
            blockDim.x,
            blockDim.y,
            blockDim.z,
            fullBlock.group_size);
      }
      PIPES_DEVICE_TRAP();
    }
  }

  __device__ __forceinline__ static ThreadGroup make_proxy_group(
      const ThreadGroup& fullBlock) {
    return ThreadGroup{
        .thread_id_in_group = fullBlock.thread_id_in_group - WorkerThreads,
        .group_size = comms::device::kWarpSize,
        .group_id = fullBlock.group_id,
        .block_id = fullBlock.block_id,
        .total_groups = fullBlock.total_groups,
        .scope = SyncScope::WARP,
    };
  }

  __device__ __forceinline__ static ThreadGroup make_solo_group(
      uint32_t channel,
      const ThreadGroup& fullBlock) {
    return ThreadGroup{
        .thread_id_in_group = 0,
        .group_size = 1,
        .group_id = channel,
        .block_id = fullBlock.block_id,
        .total_groups = fullBlock.total_groups,
        .scope = SyncScope::THREAD,
    };
  }

  __device__ __forceinline__ static void initialize(
      SharedState& storage,
      ThreadGroup& fullBlock) {
    if (fullBlock.is_leader()) {
      storage.streamCount = 0;
      storage.producerDone = 0;
    }
    fullBlock.sync();
  }

  __device__ __forceinline__ static void finish_workers(
      SharedState& storage,
      ThreadGroup& workers) {
    workers.sync();
    if (workers.is_leader()) {
      BlockAtomicU32 done(storage.producerDone);
      done.store(1, cuda::memory_order_release);
    }
  }

  __device__ __forceinline__ static void validate_pipeline_depth(
      P2pIbgdaTransportDevice& transport,
      ThreadGroup& workers) {
    const int depth = transport.channel_layout().pipelineDepth;
    if (depth <= 0 ||
        depth > static_cast<int>(kIbgdaWarpProxyMaxPipelineDepth)) {
      if (workers.is_leader()) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy pipeline depth=%d outside "
            "[1, %u]\n",
            depth,
            kIbgdaWarpProxyMaxPipelineDepth);
      }
      PIPES_DEVICE_TRAP();
    }
  }

  __device__ __forceinline__ static void validate_full_slot_control(
      std::size_t maxSignalBytes,
      ThreadGroup& workers) {
    if (maxSignalBytes != 0) {
      if (workers.is_leader()) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy full-slot control requires "
            "maxSignalBytes=0, got %llu\n",
            static_cast<unsigned long long>(maxSignalBytes));
      }
      PIPES_DEVICE_TRAP();
    }
  }

  __device__ __forceinline__ static uint64_t
  stream_ordinal(uint32_t pipelineDepth, uint32_t slot, uint64_t generation) {
    return generation * static_cast<uint64_t>(pipelineDepth) + slot;
  }

  __device__ __forceinline__ static uint64_t slot_free_expected(
      uint64_t ordinal,
      uint32_t pipelineDepth,
      std::size_t slotBytes) {
    const uint64_t streamEnd = (ordinal + 1) * slotBytes;
    const uint64_t pipelineBytes =
        static_cast<uint64_t>(pipelineDepth) * slotBytes;
    return streamEnd > pipelineBytes ? streamEnd - pipelineBytes : 0;
  }

  __device__ __forceinline__ static std::size_t send_staging_base(
      const DuplexStream& stream) {
    const auto& layout = stream.transport->channel_layout();
    const int slotIndex = layout.protoChannelSlot(
        static_cast<int>(stream.channel), protocol::Simple::kProtoSlot);
    return static_cast<std::size_t>(slotIndex) *
        detail::pipeline_window(layout);
  }

  __device__ __forceinline__ static void initialize_stream(
      DuplexStream& stream,
      P2pIbgdaTransportDevice& transport,
      uint32_t channel,
      uint32_t pipelineDepth,
      std::size_t slotBytes) {
    stream.transport = &transport;
    stream.slotBytes = slotBytes;
    stream.channel = channel;
    stream.pipelineDepth = pipelineDepth;
    stream.tx.produced = 0;
    stream.tx.posted = 0;
    stream.tx.retired = 0;
    stream.tx.bound = 0;
    stream.rx.requested = 0;
    stream.rx.arrived = 0;
    stream.rx.consumed = 0;
    stream.rx.credited = 0;
    stream.rx.bound = 0;
    stream.rx.hasExpectedEnd = 0;
  }

  __device__ __forceinline__ static uint32_t find_or_bind_stream(
      SharedState& storage,
      P2pIbgdaTransportDevice& transport,
      ThreadGroup& workers,
      uint32_t cachedStream = kInvalidStream) {
    uint32_t streamIndex = kInvalidStream;
    const uint32_t channel = static_cast<uint32_t>(workers.group_id);
    if (cachedStream < MaxStreams) {
      const auto& cached = storage.streams[cachedStream];
      if (cached.transport == &transport && cached.channel == channel) {
        return cachedStream;
      }
    }
    const uint32_t pipelineDepth =
        static_cast<uint32_t>(transport.channel_layout().pipelineDepth);
    const std::size_t slotBytes =
        detail::pipeline_chunk(transport.channel_layout());
    if (workers.is_leader()) {
      BlockAtomicU32 streamCount(storage.streamCount);
      const uint32_t count = streamCount.load(cuda::memory_order_acquire);
      for (uint32_t i = 0; i < count; ++i) {
        const auto& stream = storage.streams[i];
        if (stream.transport == &transport && stream.channel == channel) {
          streamIndex = i;
          break;
        }
      }
      if (streamIndex == kInvalidStream) {
        if (count >= MaxStreams) {
          printf(
              "[PIPES] FATAL: IbgdaWarpProxy needs more than %u streams\n",
              MaxStreams);
          PIPES_DEVICE_TRAP();
        }
        streamIndex = count;
        initialize_stream(
            storage.streams[streamIndex],
            transport,
            channel,
            pipelineDepth,
            slotBytes);
        streamCount.store(count + 1, cuda::memory_order_release);
      }
      const auto& stream = storage.streams[streamIndex];
      if (stream.pipelineDepth != pipelineDepth ||
          stream.slotBytes != slotBytes) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy stream geometry changed "
            "channel=%u\n",
            channel);
        PIPES_DEVICE_TRAP();
      }
    }
    return workers.broadcast(streamIndex);
  }

  __device__ __forceinline__ static void validate_stream_key(
      SharedState& storage,
      uint32_t streamIndex,
      P2pIbgdaTransportDevice& transport,
      ThreadGroup& workers) {
    if (workers.is_leader()) {
      BlockAtomicU32 streamCount(storage.streamCount);
      const uint32_t count = streamCount.load(cuda::memory_order_acquire);
      if (streamIndex >= count ||
          storage.streams[streamIndex].transport != &transport ||
          storage.streams[streamIndex].channel != workers.group_id) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy active stream mismatch "
            "channel=%u\n",
            workers.group_id);
        PIPES_DEVICE_TRAP();
      }
    }
  }

  __device__ __forceinline__ static uint32_t bind_send_stream(
      SharedState& storage,
      P2pIbgdaTransportDevice& transport,
      ThreadGroup& workers,
      uint32_t slot,
      uint64_t generation,
      uint32_t cachedStream = kInvalidStream) {
    const uint32_t streamIndex =
        find_or_bind_stream(storage, transport, workers, cachedStream);
    if (workers.is_leader()) {
      auto& stream = storage.streams[streamIndex];
      auto& tx = stream.tx;
      const uint64_t ordinal =
          stream_ordinal(stream.pipelineDepth, slot, generation);
      BlockAtomicU32 bound(tx.bound);
      if (bound.load(cuda::memory_order_acquire) == 0) {
        tx.produced = ordinal;
        tx.posted = ordinal;
        tx.retired =
            ordinal > stream.pipelineDepth ? ordinal - stream.pipelineDepth : 0;
        bound.store(1, cuda::memory_order_release);
      }
    }
    return streamIndex;
  }

  __device__ __forceinline__ static void validate_recv_slot_bytes(
      SharedState& storage,
      uint32_t streamIndex,
      ThreadGroup& workers,
      std::size_t protocolBytes) {
    const auto& stream = storage.streams[streamIndex];
    if (workers.is_leader() && protocolBytes != stream.slotBytes) {
      printf(
          "[PIPES] FATAL: IbgdaWarpProxy recv publication is not one full "
          "slot channel=%u bytes=%llu slotBytes=%llu\n",
          stream.channel,
          static_cast<unsigned long long>(protocolBytes),
          static_cast<unsigned long long>(stream.slotBytes));
      PIPES_DEVICE_TRAP();
    }
  }

  __device__ __forceinline__ static uint64_t recv_stream_ordinal(
      const DuplexStream& stream,
      int64_t baseByte) {
    if (baseByte < 0 ||
        static_cast<uint64_t>(baseByte) % stream.slotBytes != 0) {
      printf(
          "[PIPES] FATAL: IbgdaWarpProxy recv stream is not slot aligned "
          "channel=%u baseByte=%lld slotBytes=%llu\n",
          stream.channel,
          static_cast<long long>(baseByte),
          static_cast<unsigned long long>(stream.slotBytes));
      PIPES_DEVICE_TRAP();
    }
    return static_cast<uint64_t>(baseByte) / stream.slotBytes;
  }

  __device__ __forceinline__ static void initialize_recv_frontiers(
      RxStream& rx,
      uint64_t ordinal) {
    rx.requested = ordinal;
    rx.arrived = ordinal;
    rx.consumed = ordinal;
    rx.credited = ordinal;
    rx.hasExpectedEnd = 0;
  }

  __device__ __forceinline__ static uint32_t bind_recv_stream(
      SharedState& storage,
      P2pIbgdaTransportDevice& transport,
      ThreadGroup& workers,
      std::size_t protocolBytes,
      uint32_t cachedStream = kInvalidStream) {
    const uint32_t streamIndex =
        find_or_bind_stream(storage, transport, workers, cachedStream);
    validate_recv_slot_bytes(storage, streamIndex, workers, protocolBytes);
    if (workers.is_leader()) {
      auto& stream = storage.streams[streamIndex];
      auto& rx = stream.rx;
      BlockAtomicU32 bound(rx.bound);
      if (bound.load(cuda::memory_order_acquire) == 0) {
        const int64_t baseByte =
            transport
                .template local_channel_slot<protocol::Simple>(stream.channel)
                .recvProgress.activeBaseStep;
        const uint64_t ordinal = recv_stream_ordinal(stream, baseByte);
        initialize_recv_frontiers(rx, ordinal);
        bound.store(1, cuda::memory_order_release);
      }
    }
    return streamIndex;
  }

  __device__ __forceinline__ static uint32_t declare_expected_recvs(
      SharedState& storage,
      P2pIbgdaTransportDevice& transport,
      ThreadGroup& workers,
      uint64_t totalRecvs,
      uint32_t cachedStream = kInvalidStream) {
    const uint32_t streamIndex =
        find_or_bind_stream(storage, transport, workers, cachedStream);
    if (workers.is_leader()) {
      auto& stream = storage.streams[streamIndex];
      auto& rx = stream.rx;
      const auto& progress =
          transport
              .template local_channel_slot<protocol::Simple>(stream.channel)
              .recvProgress;
      BlockAtomicU32 bound(rx.bound);
      if (totalRecvs == 0 ||
          progress.activeStage != detail::IbSendRecvProgressStage::Done ||
          bound.load(cuda::memory_order_acquire) != 0) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy expected receives require an "
            "idle, unbound stream channel=%u count=%llu stage=%d\n",
            stream.channel,
            static_cast<unsigned long long>(totalRecvs),
            static_cast<int>(progress.activeStage));
        PIPES_DEVICE_TRAP();
      }
      const uint64_t ordinal = recv_stream_ordinal(stream, progress.nextStep);
      if (totalRecvs > kInvalidSequence - ordinal) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy expected recv ordinal overflow "
            "channel=%u\n",
            stream.channel);
        PIPES_DEVICE_TRAP();
      }
      initialize_recv_frontiers(rx, ordinal);
      rx.requested = ordinal + totalRecvs;
      rx.hasExpectedEnd = 1;
      bound.store(1, cuda::memory_order_release);
    }
    return streamIndex;
  }

  __device__ __forceinline__ static bool wait_send_slot_reusable(
      SharedState& storage,
      uint32_t streamIndex,
      ThreadGroup& workers,
      uint32_t slot,
      uint64_t generation,
      const AbortDevice& abortDevice) {
    uint32_t aborted = 0;
    if (workers.is_leader()) {
      auto& stream = storage.streams[streamIndex];
      const uint64_t ordinal =
          stream_ordinal(stream.pipelineDepth, slot, generation);
      const uint64_t requiredRetired = ordinal >= stream.pipelineDepth
          ? ordinal - stream.pipelineDepth + 1
          : 0;
      if (FT_ABORT_CHECK(
              abortDevice,
              "IbgdaWarpProxy preparing a send on an aborted communicator")) {
        aborted = 1U;
      }
      BlockAtomicU64 produced(stream.tx.produced);
      if (aborted == 0 &&
          produced.load(cuda::memory_order_relaxed) != ordinal) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy send stream is not contiguous "
            "channel=%u\n",
            stream.channel);
        PIPES_DEVICE_TRAP();
      }
      BlockAtomicU64 retired(stream.tx.retired);
      uint64_t current = retired.load(cuda::memory_order_acquire);
      while (aborted == 0 && current < requiredRetired) {
        if (FT_ABORT_CHECK(
                abortDevice,
                "IbgdaWarpProxy waiting for proxy CQ retirement "
                "channel=%u retired=%llu required=%llu",
                stream.channel,
                static_cast<unsigned long long>(current),
                static_cast<unsigned long long>(requiredRetired))) {
          aborted = 1U;
          break;
        }
        current = retired.load(cuda::memory_order_acquire);
      }
    }
    return workers.broadcast<uint32_t>(aborted) != 0U;
  }

  __device__ __forceinline__ static void publish_send(
      SharedState& storage,
      uint32_t streamIndex,
      P2pIbgdaTransportDevice& transport,
      ThreadGroup& workers,
      const IbgdaLocalBuffer& source,
      std::size_t remoteOffset,
      std::size_t bytes,
      std::size_t protocolBytes,
      uint64_t slotFreeExpected,
      uint32_t slot,
      uint64_t generation) {
    if (workers.is_leader()) {
      auto& stream = storage.streams[streamIndex];
      auto& tx = stream.tx;
      BlockAtomicU64 produced(tx.produced);
      const uint64_t ordinal = produced.load(cuda::memory_order_relaxed);
      if (stream.transport != &transport || slot >= stream.pipelineDepth ||
          stream_ordinal(stream.pipelineDepth, slot, generation) != ordinal) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy send publication violates "
            "full-slot stream geometry channel=%u ordinal=%llu\n",
            stream.channel,
            static_cast<unsigned long long>(ordinal));
        PIPES_DEVICE_TRAP();
      }
      const std::size_t expectedOffset = send_staging_base(stream) +
          static_cast<std::size_t>(slot) * stream.slotBytes;
      const void* expectedSource = transport.channel_layout()
                                       .sendStagingBuf.subBuffer(expectedOffset)
                                       .ptr;
      const uint64_t expectedSlotFree =
          slot_free_expected(ordinal, stream.pipelineDepth, stream.slotBytes);
      if (remoteOffset != expectedOffset || source.ptr != expectedSource ||
          bytes > stream.slotBytes || protocolBytes != stream.slotBytes ||
          slotFreeExpected != expectedSlotFree) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy send publication violates "
            "full-slot stream geometry channel=%u ordinal=%llu\n",
            stream.channel,
            static_cast<unsigned long long>(ordinal));
        PIPES_DEVICE_TRAP();
      }
      if (bytes == 0) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy send length is zero "
            "channel=%u ordinal=%llu\n",
            stream.channel,
            static_cast<unsigned long long>(ordinal));
        PIPES_DEVICE_TRAP();
      }
      tx.publishedBytes[slot] = bytes;
      produced.store(ordinal + 1, cuda::memory_order_release);
    }
  }

  __device__ __forceinline__ static uint64_t request_and_wait_recv(
      SharedState& storage,
      uint32_t streamIndex,
      ThreadGroup& workers,
      const AbortDevice& abortDevice) {
    uint64_t result = kInvalidSequence;
    if (workers.is_leader()) {
      auto& stream = storage.streams[streamIndex];
      auto& rx = stream.rx;
      bool aborted = FT_ABORT_CHECK(
          abortDevice,
          "IbgdaWarpProxy preparing a recv on an aborted communicator");
      BlockAtomicU64 consumed(rx.consumed);
      BlockAtomicU64 requested(rx.requested);
      const uint64_t sequence = consumed.load(cuda::memory_order_relaxed);
      const uint64_t requestedValue =
          requested.load(cuda::memory_order_acquire);
      if (!aborted) {
        if (rx.hasExpectedEnd != 0) {
          if (sequence >= requestedValue) {
            printf(
                "[PIPES] FATAL: IbgdaWarpProxy expected recv frontier mismatch "
                "channel=%u sequence=%llu requested=%llu\n",
                storage.streams[streamIndex].channel,
                static_cast<unsigned long long>(sequence),
                static_cast<unsigned long long>(requestedValue));
            PIPES_DEVICE_TRAP();
          }
        } else {
          if (requestedValue != sequence) {
            printf(
                "[PIPES] FATAL: IbgdaWarpProxy recv request is not contiguous\n");
            PIPES_DEVICE_TRAP();
          }
          requested.store(sequence + 1, cuda::memory_order_release);
        }
      }
      BlockAtomicU64 arrived(stream.rx.arrived);
      uint64_t current = arrived.load(cuda::memory_order_acquire);
      while (!aborted && current <= sequence) {
        if (FT_ABORT_CHECK(
                abortDevice,
                "IbgdaWarpProxy waiting for DATA_READY channel=%u "
                "arrived=%llu "
                "required=%llu",
                stream.channel,
                static_cast<unsigned long long>(current),
                static_cast<unsigned long long>(sequence + 1))) {
          aborted = true;
          break;
        }
        current = arrived.load(cuda::memory_order_acquire);
      }
      if (!aborted) {
        result = sequence;
      }
    }
    return workers.broadcast(result);
  }

  __device__ __forceinline__ static void publish_recv_consumed(
      SharedState& storage,
      uint32_t streamIndex,
      ThreadGroup& workers,
      uint64_t sequence) {
    if (workers.is_leader()) {
      auto& stream = storage.streams[streamIndex];
      BlockAtomicU64 consumed(stream.rx.consumed);
      if (consumed.load(cuda::memory_order_relaxed) != sequence) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy recv stream is not contiguous "
            "channel=%u\n",
            stream.channel);
        PIPES_DEVICE_TRAP();
      }
      consumed.store(sequence + 1, cuda::memory_order_release);
    }
  }

  __device__ __forceinline__ static void post_recv_credit_once(
      DuplexStream& stream,
      const ThreadGroup& fullBlock,
      const AbortDevice& abortDevice) {
    auto& rx = stream.rx;
    BlockAtomicU64 consumed(rx.consumed);
    BlockAtomicU64 credited(rx.credited);
    const uint64_t head = credited.load(cuda::memory_order_relaxed);
    const uint64_t consumedTail = consumed.load(cuda::memory_order_acquire);
    if (head >= consumedTail || abortDevice.isAborted()) {
      return;
    }
    const IbRemoteChannel remote = makeIbRemoteChannel(
        stream.transport->channel_layout(), static_cast<int>(stream.channel));
    ThreadGroup solo = make_solo_group(stream.channel, fullBlock);
    if (!stream.transport->try_signal(
            solo,
            remote.slotFree,
            stream.slotBytes,
            IbDirection::Recv,
            abortDevice)) {
      return;
    }
    credited.store(head + 1, cuda::memory_order_release);
  }

  __device__ __forceinline__ static void poll_data_ready_once(
      DuplexStream& stream,
      const AbortDevice& abortDevice) {
    auto& rx = stream.rx;
    BlockAtomicU64 requested(rx.requested);
    BlockAtomicU64 arrived(rx.arrived);
    const uint64_t currentArrived = arrived.load(cuda::memory_order_relaxed);
    const uint64_t currentRequested =
        requested.load(cuda::memory_order_acquire);
    if (currentArrived >= currentRequested) {
      return;
    }
    IbLocalChannel& local = stream.transport->local_channel(stream.channel);
    IbChannelProtoSlot& localSlot =
        stream.transport->template local_channel_slot<protocol::Simple>(
            stream.channel);
    unsigned long long current = 0;
    unsigned long long expected = 0;
    if (!detail::poll_recv_data_ready(
            *stream.transport,
            local,
            localSlot.dataReady,
            stream.slotBytes,
            current,
            expected)) {
      (void)FT_ABORT_CHECK(
          abortDevice,
          "IbgdaWarpProxy waiting for DATA_READY channel=%u "
          "expected=%llu current=%llu",
          stream.channel,
          expected,
          current);
      return;
    }
    arrived.store(currentArrived + 1, cuda::memory_order_release);
  }

  __device__ __forceinline__ static uint64_t tx_retire_high_watermark(
      uint32_t pipelineDepth) {
    // Start one slot before wrap so CQ progress can overlap the producer's
    // final free slot instead of beginning only after the producer stalls.
    return pipelineDepth == 1 ? 1 : static_cast<uint64_t>(pipelineDepth - 1);
  }

  __device__ __forceinline__ static void post_send_once(
      DuplexStream& stream,
      const ThreadGroup& fullBlock,
      const AbortDevice& abortDevice) {
    auto& tx = stream.tx;
    BlockAtomicU64 produced(tx.produced);
    BlockAtomicU64 posted(tx.posted);
    const uint64_t head = posted.load(cuda::memory_order_relaxed);
    const uint64_t producedTail = produced.load(cuda::memory_order_acquire);
    if (head >= producedTail) {
      return;
    }

    if (stream.transport->send_completion_lane_count() == 0) {
      printf("[PIPES] FATAL: IbgdaWarpProxy has no send QP lanes\n");
      PIPES_DEVICE_TRAP();
    }

    const uint32_t slot = static_cast<uint32_t>(head % stream.pipelineDepth);
    const uint64_t generation = head / stream.pipelineDepth;
    const uint64_t slotFreeExpected =
        slot_free_expected(head, stream.pipelineDepth, stream.slotBytes);
    if (slotFreeExpected != 0) {
      const IbChannelProtoSlot& localSlot =
          stream.transport->template local_channel_slot<protocol::Simple>(
              stream.channel);
      const uint64_t current =
          stream.transport->read_signal(localSlot.slotFree);
      if (current < slotFreeExpected) {
        (void)FT_ABORT_CHECK(
            abortDevice,
            "IbgdaWarpProxy waiting for SLOT_FREE channel=%u "
            "expected=%llu current=%llu",
            stream.channel,
            static_cast<unsigned long long>(slotFreeExpected),
            static_cast<unsigned long long>(current));
        return;
      }
    }

    const IbRemoteChannel remote = makeIbRemoteChannel(
        stream.transport->channel_layout(), static_cast<int>(stream.channel));
    ThreadGroup solo = make_solo_group(stream.channel, fullBlock);
    const std::size_t offset = send_staging_base(stream) +
        static_cast<std::size_t>(slot) * stream.slotBytes;
    if (FT_ABORT_CHECK(
            abortDevice,
            "IbgdaWarpProxy abandoning send before WQE publication channel=%u",
            stream.channel)) {
      return;
    }
    const IbLocalCompletionTicket ticket = stream.transport->put(
        solo,
        stream.transport->channel_layout().sendStagingBuf.subBuffer(offset),
        remote.recvStaging.subBuffer(offset),
        tx.publishedBytes[slot],
        remote.dataReady,
        stream.slotBytes,
        /*counterBuf=*/{},
        /*counterVal=*/0,
        /*signalPerLane=*/true,
        abortDevice);
    if (!ticket.posted) {
      return;
    }
    detail::record_send_completion(
        *stream.transport, stream.channel, slot, generation, ticket);
    posted.store(head + 1, cuda::memory_order_release);
  }

  __device__ __forceinline__ static void retire_send_at_high_watermark_once(
      DuplexStream& stream,
      const ThreadGroup& fullBlock,
      const AbortDevice& abortDevice) {
    BlockAtomicU64 posted(stream.tx.posted);
    BlockAtomicU64 retired(stream.tx.retired);
    const uint64_t head = retired.load(cuda::memory_order_relaxed);
    const uint64_t postedHead = posted.load(cuda::memory_order_acquire);
    if (head >= postedHead) {
      return;
    }
    const uint64_t highWatermark =
        tx_retire_high_watermark(stream.pipelineDepth);
    if (postedHead - head < highWatermark) {
      return;
    }
    const uint32_t slot = static_cast<uint32_t>(head % stream.pipelineDepth);
    const uint64_t nextGeneration = head / stream.pipelineDepth + 1;
    ThreadGroup solo = make_solo_group(stream.channel, fullBlock);
    const uint32_t status = detail::try_prepare_send_slot<protocol::Simple>(
        *stream.transport, solo, slot, nextGeneration, abortDevice);
    if (status == detail::kProgressReady) {
      retired.store(head + 1, cuda::memory_order_release);
    }
  }

  __device__ __forceinline__ static void run_proxy(
      SharedState& storage,
      ThreadGroup& proxy,
      const ThreadGroup& fullBlock,
      const AbortDevice& abortDevice) {
    BlockAtomicU32 producerDone(storage.producerDone);
    while (true) {
      uint32_t stop = 0;
      // Once a progress step observes abort, do not enter another peer-visible
      // posting step: a false DATA_READY or SLOT_FREE can strand peer recovery.
      if (proxy.is_leader()) {
        bool aborted = FT_ABORT_CHECK(
            abortDevice, "IbgdaWarpProxy::run_proxy abandoning drain");
        BlockAtomicU32 streamCount(storage.streamCount);
        const uint32_t count = streamCount.load(cuda::memory_order_acquire);
        for (uint32_t streamIndex = 0; streamIndex < count && !aborted;
             ++streamIndex) {
          auto& stream = storage.streams[streamIndex];
          BlockAtomicU32 txBound(stream.tx.bound);
          if (txBound.load(cuda::memory_order_acquire) != 0) {
            post_send_once(stream, fullBlock, abortDevice);
            aborted = abortDevice.isAborted();
            if (!aborted) {
              retire_send_at_high_watermark_once(
                  stream, fullBlock, abortDevice);
              aborted = abortDevice.isAborted();
            }
          }
          if (!aborted) {
            BlockAtomicU32 rxBound(stream.rx.bound);
            if (rxBound.load(cuda::memory_order_acquire) != 0) {
              post_recv_credit_once(stream, fullBlock, abortDevice);
              aborted = abortDevice.isAborted();
              if (!aborted) {
                poll_data_ready_once(stream, abortDevice);
                aborted = abortDevice.isAborted();
              }
            }
          }
        }

        if (aborted) {
          stop = 1U;
        } else if (producerDone.load(cuda::memory_order_acquire) != 0) {
          const uint32_t finalCount =
              streamCount.load(cuda::memory_order_acquire);
          bool drained = finalCount == count;
          for (uint32_t i = 0; i < count && drained; ++i) {
            auto& stream = storage.streams[i];
            BlockAtomicU32 txBound(stream.tx.bound);
            if (txBound.load(cuda::memory_order_acquire) != 0) {
              BlockAtomicU64 sendProduced(stream.tx.produced);
              BlockAtomicU64 sendPosted(stream.tx.posted);
              const uint64_t produced =
                  sendProduced.load(cuda::memory_order_acquire);
              const uint64_t posted =
                  sendPosted.load(cuda::memory_order_acquire);
              drained = posted == produced;
            }
            BlockAtomicU32 rxBound(stream.rx.bound);
            if (drained && rxBound.load(cuda::memory_order_acquire) != 0) {
              BlockAtomicU64 recvRequested(stream.rx.requested);
              BlockAtomicU64 recvCredited(stream.rx.credited);
              drained = recvCredited.load(cuda::memory_order_acquire) ==
                  recvRequested.load(cuda::memory_order_acquire);
            }
          }
          stop = drained ? 1U : 0U;
        }
      }
      stop = proxy.broadcast(stop);
      if (stop != 0) {
        break;
      }
    }
  }
};

} // namespace comms::prims

#endif // defined(__CUDACC__) && !defined(__HIP_PLATFORM_AMD__)
