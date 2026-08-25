// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::prims {

inline constexpr uint32_t kIbgdaWarpProxyMaxPipelineDepth = 16;
inline constexpr uint32_t kIbgdaWarpProxyQueueCapacity = 16;

inline constexpr bool isIbgdaWarpProxyPipelineDepthSupported(int depth) {
  return depth > 0 &&
      depth <= static_cast<int>(kIbgdaWarpProxyMaxPipelineDepth);
}

} // namespace comms::prims

#if defined(__CUDACC__) && !defined(__HIP_PLATFORM_AMD__)

#include <cuda/atomic>

#include <cstddef>

#include "comms/prims/core/MemcpyCopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/P2pIbTransportProgressImpl.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims {

/**
 * Runs fixed-size IBGDA send/recv work with one trailing service warp.
 *
 * Workers own staging copies and staging-slot completion waits. The service
 * warp owns remote readiness polling, WQE posting, ticket publication, and
 * receive credits. A run exclusively owns every (transport, channel) passed
 * through Ops until run() returns; transport objects must have shared or global
 * lifetime. Ops calls may return after work is queued. workerFn must use
 * Ops::group() for synchronization and issue Ops calls collectively from that
 * single producer group; block-wide barriers and concurrent subgroup issuers
 * are unsupported. Fused forwarding releases each upstream receive credit
 * before its dependent downstream send becomes eligible for posting.
 *
 * Completion has two shapes, and they differ in what they promise:
 *
 * - **Normal completion**: run() returns after staged sends are posted and
 *   receive credits are issued, so the queues are drained.
 * - **Abort completion**: run() returns with `send.posted < send.tail` and/or
 *   `recv.credited < recv.tail`. Pending commands are deliberately abandoned
 *   and their credits never issued -- publishing them would signal a peer for
 *   work this rank gave up on, which is what releases a correctly blocked peer
 *   and stops it reaching its own deadline. Queue drain is therefore NOT a
 *   postcondition of run(); termination is. Recovery is `reconfigure()`, as
 *   everywhere else in the abort contract.
 */
template <
    uint32_t WorkerThreads,
    uint32_t MaxPipelineDepth = kIbgdaWarpProxyMaxPipelineDepth>
class IbgdaWarpProxy {
 private:
  static constexpr uint32_t kPipelineSlotCapacity = MaxPipelineDepth;
  static constexpr uint32_t kQueueCapacity = kIbgdaWarpProxyQueueCapacity;
  // Barrier 0 is reserved for full-block synchronization in this kernel.
  static constexpr uint32_t kWorkerNamedBarrierId = 1;

  static_assert(WorkerThreads > 0);
  static_assert(MaxPipelineDepth > 0);
  static_assert(WorkerThreads % comms::device::kWarpSize == 0);
  static_assert(
      WorkerThreads + comms::device::kWarpSize <= 1024,
      "IbgdaWarpProxy exceeds the CUDA threads-per-block limit");

  static constexpr uint32_t kServiceThreads = comms::device::kWarpSize;
  static constexpr uint64_t kInvalidSequence = ~uint64_t{0};

  struct SendCommand {
    P2pIbgdaTransportDevice* transport;
    IbgdaLocalBuffer source;
    uint64_t remoteOffset;
    uint64_t bytes;
    uint64_t protocolBytes;
    uint64_t slotFreeExpected;
    uint64_t generation;
    uint64_t requiredRecvCredit;
    uint32_t channel;
    uint32_t slot;
  };

  struct RecvCommand {
    P2pIbgdaTransportDevice* transport;
    uint64_t protocolBytes;
    uint32_t channel;
  };

  struct SendSlotState {
    uint64_t lastCommand;
  };

  struct alignas(16) SendQueue {
    // Worker publishes tail. Service publishes posted after ticket recording.
    alignas(8) uint64_t tail;
    alignas(8) uint64_t posted;
    SendSlotState slots[kPipelineSlotCapacity];
    SendCommand commands[kQueueCapacity];
  };

  struct alignas(16) RecvQueue {
    // Worker: tail/copied. Service: ready/credited.
    alignas(8) uint64_t tail;
    alignas(8) uint64_t ready;
    alignas(8) uint64_t copied;
    alignas(8) uint64_t credited;
    RecvCommand commands[kQueueCapacity];
  };

 public:
  static constexpr uint32_t kBlockThreads =
      WorkerThreads + comms::device::kWarpSize;

  struct Config {
    uint32_t queueDepth{kQueueCapacity};
    uint64_t* queueFullCount{nullptr};
  };

  struct alignas(16) SharedState {
    SendQueue send;
    RecvQueue recv;
    alignas(8) uint64_t* queueFullCount;
    alignas(4) uint32_t queueDepth;
    alignas(4) uint32_t producerDone;
  };

  class Ops {
   public:
    static constexpr uint32_t kWorkerThreads = WorkerThreads;

    __device__ __forceinline__ ThreadGroup& group() {
      return workers_;
    }

    __device__ __forceinline__ void sync() {
      workers_.sync();
    }

    // Drains staged sends and issues outstanding receive credits -- unless the
    // operation aborts, in which case it returns with commands still queued and
    // their credits deliberately unissued. See the abort-completion note on the
    // class comment: "drained" is not a postcondition once an abort is latched,
    // termination is.
    __device__ __forceinline__ void drain() {
      IbgdaWarpProxy::drain_queues(storage_, workers_, timeout_);
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
      detail::send_impl<P2pIbgdaTransportDevice, CopyOp>(
          transport,
          workers_,
          this,
          src,
          nbytes,
          maxSignalBytes,
          timeout_,
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
      detail::recv_impl<P2pIbgdaTransportDevice, CopyOp>(
          transport,
          workers_,
          this,
          dst,
          nbytes,
          maxSignalBytes,
          timeout_,
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
      detail::forward_impl<CopyOp, P2pIbgdaTransportDevice>(
          prev,
          workers_,
          this,
          dst,
          next,
          nbytes,
          maxSignalBytes,
          timeout_,
          nullptr,
          nullptr,
          args...);
    }

    // Returns true when the slot could not be retired -- see
    // detail::prepare_send_slot. The caller must not stage or put on a slot the
    // NIC may still be reading.
    [[nodiscard]] __device__ __forceinline__ bool prepare_send_slot(
        P2pIbgdaTransportDevice& transport,
        ThreadGroup& workers,
        uint32_t slot,
        uint64_t generation,
        const Timeout& timeout) {
      IbgdaWarpProxy::wait_prior_send_posted(storage_, workers, slot, timeout);
      return detail::prepare_send_slot(
          transport, workers, slot, generation, timeout);
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
        uint64_t generation,
        uint64_t requiredRecvCredit,
        const Timeout& timeout) {
      // The wait reports its own verdict through the barrier it already had,
      // so declining to enqueue costs no extra synchronization. Enqueuing here
      // would hand the service warp a command it turns into a peer-visible put
      // with a fused DATA_READY.
      if (IbgdaWarpProxy::template wait_queue_space<true>(
              storage_, workers, timeout)) {
        return;
      }
      IbgdaWarpProxy::enqueue_send(
          storage_,
          workers,
          SendCommand{
              .transport = &transport,
              .source = source,
              .remoteOffset = remoteOffset,
              .bytes = bytes,
              .protocolBytes = protocolBytes,
              .slotFreeExpected = slotFreeExpected,
              .generation = generation,
              .requiredRecvCredit = requiredRecvCredit,
              .channel = static_cast<uint32_t>(workers.group_id),
              .slot = slot,
          });
    }

    __device__ __forceinline__ uint64_t wait_recv(
        P2pIbgdaTransportDevice& transport,
        ThreadGroup& workers,
        std::size_t protocolBytes,
        const Timeout& timeout) {
      // Same shape as `submit_send()`. `kInvalidSequence` is what makes
      // `publish_recv()` free: it can decline on a register compare instead of
      // taking another barrier to re-ask.
      if (IbgdaWarpProxy::template wait_queue_space<false>(
              storage_, workers, timeout)) {
        return kInvalidSequence;
      }
      const uint64_t sequence = IbgdaWarpProxy::enqueue_recv(
          storage_,
          workers,
          RecvCommand{
              .transport = &transport,
              .protocolBytes = protocolBytes,
              .channel = static_cast<uint32_t>(workers.group_id),
          });
      if (IbgdaWarpProxy::wait_recv_ready(
              storage_, workers, sequence, timeout)) {
        return kInvalidSequence;
      }
      return sequence;
    }

    __device__ __forceinline__ void publish_recv(
        P2pIbgdaTransportDevice& transport,
        ThreadGroup& workers,
        std::size_t protocolBytes,
        uint64_t sequence) {
      (void)transport;
      (void)protocolBytes;
      // Advancing `recv.copied` is what lets the service warp emit SLOT_FREE
      // for this chunk, so it must not happen for a receive that never landed.
      //
      // The sentinel carries that decision from `wait_recv()`, which already
      // made it group-uniformly through its own barrier. Re-asking here would
      // cost another block-wide barrier per chunk to learn something already
      // known, so this is a register compare.
      if (sequence == kInvalidSequence) {
        return;
      }
      IbgdaWarpProxy::publish_recv_copied(storage_, workers, sequence);
    }

   private:
    friend class IbgdaWarpProxy<WorkerThreads, MaxPipelineDepth>;

    __device__
    Ops(SharedState& storage, ThreadGroup workers, const Timeout& timeout)
        : storage_(storage), workers_(workers), timeout_(timeout) {}

    SharedState& storage_;
    ThreadGroup workers_;
    Timeout timeout_;
  };

  template <typename WorkerFn>
  __device__ __forceinline__ static void run(
      SharedState& storage,
      ThreadGroup fullBlock,
      const Timeout& timeout,
      WorkerFn&& workerFn) {
    run(storage,
        fullBlock,
        Config{},
        timeout,
        static_cast<WorkerFn&&>(workerFn));
  }

  template <typename WorkerFn>
  __device__ __forceinline__ static void run(
      SharedState& storage,
      ThreadGroup fullBlock,
      const Config& config,
      const Timeout& timeout,
      WorkerFn&& workerFn) {
    validate_block(fullBlock);
    validate_config(config, fullBlock);
    initialize(storage, fullBlock, config);

    if (fullBlock.thread_id_in_group < WorkerThreads) {
      ThreadGroup workers = make_worker_group(fullBlock);
      Ops ops(storage, workers, timeout);
      workerFn(ops);
      finish_workers(storage, workers);
    } else {
      ThreadGroup service = make_service_group(fullBlock);
      run_service(storage, service, fullBlock, timeout);
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
  using DeviceAtomicU64 = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>;

  __device__ __forceinline__ static void validate_block(
      const ThreadGroup& fullBlock) {
    const uint32_t expected = WorkerThreads + kServiceThreads;
    const bool valid = blockDim.y == 1 && blockDim.z == 1 &&
        fullBlock.group_size == expected && blockDim.x == expected &&
        fullBlock.thread_id_in_group == threadIdx.x &&
        fullBlock.scope == SyncScope::BLOCK;
    if (!valid) {
      if (fullBlock.is_leader()) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy requires exactly %u workers and "
            "one trailing service warp; block=(%u,%u,%u) group=%u\n",
            WorkerThreads,
            blockDim.x,
            blockDim.y,
            blockDim.z,
            fullBlock.group_size);
      }
      PIPES_DEVICE_TRAP();
    }
  }

  __device__ __forceinline__ static void validate_config(
      const Config& config,
      const ThreadGroup& fullBlock) {
    const bool valid = config.queueDepth > 0 &&
        config.queueDepth <= static_cast<uint32_t>(kQueueCapacity);
    if (!valid) {
      if (fullBlock.is_leader()) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy queue depth=%u outside [1, %u]\n",
            config.queueDepth,
            kQueueCapacity);
      }
      PIPES_DEVICE_TRAP();
    }
  }

  __device__ __forceinline__ static ThreadGroup make_service_group(
      const ThreadGroup& fullBlock) {
    return ThreadGroup{
        .thread_id_in_group = fullBlock.thread_id_in_group - WorkerThreads,
        .group_size = kServiceThreads,
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
      ThreadGroup& fullBlock,
      const Config& config) {
    if (fullBlock.is_leader()) {
      storage.send.tail = 0;
      storage.send.posted = 0;
      storage.recv.tail = 0;
      storage.recv.ready = 0;
      storage.recv.copied = 0;
      storage.recv.credited = 0;
      storage.queueFullCount = config.queueFullCount;
      storage.queueDepth = config.queueDepth;
      storage.producerDone = 0;
      for (uint32_t slot = 0; slot < kPipelineSlotCapacity; ++slot) {
        auto& sendSlot = storage.send.slots[slot];
        sendSlot.lastCommand = kInvalidSequence;
      }
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

  __device__ __forceinline__ static void drain_queues(
      SharedState& storage,
      ThreadGroup& workers,
      const Timeout& timeout) {
    workers.sync();
    if (workers.is_leader()) {
      BlockAtomicU64 sendTail(storage.send.tail);
      BlockAtomicU64 sendPosted(storage.send.posted);
      BlockAtomicU64 recvTail(storage.recv.tail);
      BlockAtomicU64 recvCredited(storage.recv.credited);
      const uint64_t targetSend = sendTail.load(cuda::memory_order_acquire);
      const uint64_t targetRecv = recvTail.load(cuda::memory_order_acquire);
      uint64_t currentSend = sendPosted.load(cuda::memory_order_acquire);
      uint64_t currentRecv = recvCredited.load(cuda::memory_order_acquire);
      while (currentSend < targetSend || currentRecv < targetRecv) {
        FT_ABORT_BREAK(
            timeout,
            "IbgdaWarpProxy drain waiting for service progress "
            "send=%llu/%llu recv=%llu/%llu",
            static_cast<unsigned long long>(currentSend),
            static_cast<unsigned long long>(targetSend),
            static_cast<unsigned long long>(currentRecv),
            static_cast<unsigned long long>(targetRecv));
        currentSend = sendPosted.load(cuda::memory_order_acquire);
        currentRecv = recvCredited.load(cuda::memory_order_acquire);
      }
    }
    workers.sync();
  }

  __device__ __forceinline__ static void validate_pipeline_depth(
      P2pIbgdaTransportDevice& transport,
      ThreadGroup& workers) {
    const int depth = transport.channel_layout().pipelineDepth;
    if (depth <= 0 || depth > static_cast<int>(kPipelineSlotCapacity)) {
      if (workers.is_leader()) {
        printf(
            "[PIPES] FATAL: IbgdaWarpProxy pipeline depth=%d outside "
            "[1, %u]\n",
            depth,
            kPipelineSlotCapacity);
      }
      PIPES_DEVICE_TRAP();
    }
  }

  template <bool IsSend>
  __device__ __forceinline__ static bool wait_queue_space(
      SharedState& storage,
      ThreadGroup& workers,
      const Timeout& timeout) {
    uint32_t aborted = 0;
    if (workers.is_leader()) {
      uint64_t& tailValue = IsSend ? storage.send.tail : storage.recv.tail;
      uint64_t& headValue =
          IsSend ? storage.send.posted : storage.recv.credited;
      BlockAtomicU64 tail(tailValue);
      BlockAtomicU64 head(headValue);
      const uint64_t currentTail = tail.load(cuda::memory_order_relaxed);
      uint64_t currentHead = head.load(cuda::memory_order_acquire);
      if (currentTail - currentHead >= storage.queueDepth &&
          storage.queueFullCount != nullptr) {
        DeviceAtomicU64 queueFullCount(*storage.queueFullCount);
        queueFullCount.fetch_add(1, cuda::memory_order_relaxed);
      }
      while (currentTail - currentHead >= storage.queueDepth) {
        if (FT_ABORT_CHECK(
                timeout,
                "IbgdaWarpProxy %s queue full channel=%u head=%llu tail=%llu",
                IsSend ? "send" : "recv",
                workers.group_id,
                static_cast<unsigned long long>(currentHead),
                static_cast<unsigned long long>(currentTail))) {
          aborted = 1U;
          break;
        }
        currentHead = head.load(cuda::memory_order_acquire);
      }
    }
    return workers.broadcast<uint32_t>(aborted) != 0U;
  }

  __device__ __forceinline__ static void wait_prior_send_posted(
      SharedState& storage,
      ThreadGroup& workers,
      uint32_t slot,
      const Timeout& timeout) {
    auto& slotState = storage.send.slots[slot];
    if (workers.is_leader()) {
      const uint64_t requiredPosted = slotState.lastCommand == kInvalidSequence
          ? 0
          : slotState.lastCommand + 1;
      BlockAtomicU64 posted(storage.send.posted);
      uint64_t currentPosted = posted.load(cuda::memory_order_acquire);
      while (currentPosted < requiredPosted) {
        FT_ABORT_BREAK(
            timeout,
            "IbgdaWarpProxy waiting for send WQE post channel=%u slot=%u "
            "posted=%llu required=%llu",
            workers.group_id,
            slot,
            static_cast<unsigned long long>(currentPosted),
            static_cast<unsigned long long>(requiredPosted));
        currentPosted = posted.load(cuda::memory_order_acquire);
      }
    }
    workers.sync();
  }

  __device__ __forceinline__ static void enqueue_send(
      SharedState& storage,
      ThreadGroup& workers,
      const SendCommand& command) {
    if (workers.is_leader()) {
      BlockAtomicU64 tail(storage.send.tail);
      const uint64_t sequence = tail.load(cuda::memory_order_relaxed);
      storage.send.commands[sequence % kQueueCapacity] = command;
      storage.send.slots[command.slot].lastCommand = sequence;
      tail.store(sequence + 1, cuda::memory_order_release);
    }
  }

  __device__ __forceinline__ static uint64_t enqueue_recv(
      SharedState& storage,
      ThreadGroup& workers,
      const RecvCommand& command) {
    uint64_t sequence = 0;
    if (workers.is_leader()) {
      BlockAtomicU64 tail(storage.recv.tail);
      sequence = tail.load(cuda::memory_order_relaxed);
      storage.recv.commands[sequence % kQueueCapacity] = command;
      tail.store(sequence + 1, cuda::memory_order_release);
    }
    return workers.broadcast(sequence);
  }

  // Same shape as `wait_queue_space`: the verdict leaves through the barrier
  // this already had, and the abort is read only while actually stalled.
  __device__ __forceinline__ static bool wait_recv_ready(
      SharedState& storage,
      ThreadGroup& workers,
      uint64_t sequence,
      const Timeout& timeout) {
    uint32_t aborted = 0;
    if (workers.is_leader()) {
      BlockAtomicU64 ready(storage.recv.ready);
      uint64_t current = ready.load(cuda::memory_order_acquire);
      while (current <= sequence) {
        if (FT_ABORT_CHECK(
                timeout,
                "IbgdaWarpProxy waiting for DATA_READY channel=%u ready=%llu "
                "required=%llu",
                workers.group_id,
                static_cast<unsigned long long>(current),
                static_cast<unsigned long long>(sequence + 1))) {
          aborted = 1U;
          break;
        }
        current = ready.load(cuda::memory_order_acquire);
      }
    }
    return workers.broadcast<uint32_t>(aborted) != 0U;
  }

  __device__ __forceinline__ static void publish_recv_copied(
      SharedState& storage,
      ThreadGroup& workers,
      uint64_t sequence) {
    if (workers.is_leader()) {
      BlockAtomicU64 copied(storage.recv.copied);
      copied.store(sequence + 1, cuda::memory_order_release);
    }
  }

  __device__ __forceinline__ static void post_recv_credits(
      SharedState& storage,
      const ThreadGroup& fullBlock) {
    BlockAtomicU64 copied(storage.recv.copied);
    BlockAtomicU64 credited(storage.recv.credited);
    uint64_t head = credited.load(cuda::memory_order_relaxed);
    const uint64_t copiedTail = copied.load(cuda::memory_order_acquire);
    while (head < copiedTail) {
      const RecvCommand command = storage.recv.commands[head % kQueueCapacity];
      const IbRemoteChannel remote = makeIbRemoteChannel(
          command.transport->channel_layout(),
          static_cast<int>(command.channel));
      ThreadGroup solo = make_solo_group(command.channel, fullBlock);
      command.transport->signal(
          solo, remote.slotFree, command.protocolBytes, IbDirection::Recv);
      credited.store(++head, cuda::memory_order_release);
    }
  }

  __device__ __forceinline__ static void publish_recv_readiness(
      SharedState& storage,
      const Timeout& timeout) {
    BlockAtomicU64 tail(storage.recv.tail);
    BlockAtomicU64 ready(storage.recv.ready);
    uint64_t currentReady = ready.load(cuda::memory_order_relaxed);
    const uint64_t currentTail = tail.load(cuda::memory_order_acquire);
    while (currentReady < currentTail) {
      const RecvCommand command =
          storage.recv.commands[currentReady % kQueueCapacity];
      IbLocalChannel& local = command.transport->local_channel(command.channel);
      IbChannelProtoSlot& localSlot =
          command.transport->template local_channel_slot<protocol::Simple>(
              command.channel);
      unsigned long long current = 0;
      unsigned long long expected = 0;
      if (!detail::poll_recv_data_ready(
              *command.transport,
              local,
              localSlot.dataReady,
              command.protocolBytes,
              current,
              expected)) {
        // CHECK rather than BREAK: the `break` below is unconditional, so this
        // call is only here for the log-and-trap side effect on abort.
        (void)FT_ABORT_CHECK(
            timeout,
            "IbgdaWarpProxy waiting for DATA_READY channel=%u "
            "expected=%llu current=%llu",
            command.channel,
            expected,
            current);
        break;
      }
      ready.store(++currentReady, cuda::memory_order_release);
    }
  }

  __device__ __forceinline__ static void post_send_once(
      SharedState& storage,
      const ThreadGroup& fullBlock,
      const Timeout& timeout) {
    BlockAtomicU64 tail(storage.send.tail);
    BlockAtomicU64 posted(storage.send.posted);
    const uint64_t head = posted.load(cuda::memory_order_relaxed);
    const uint64_t currentTail = tail.load(cuda::memory_order_acquire);
    if (head == currentTail) {
      return;
    }

    const SendCommand command = storage.send.commands[head % kQueueCapacity];
    if (command.transport->send_completion_lane_count() == 0) {
      printf("[PIPES] FATAL: IbgdaWarpProxy has no send QP lanes\n");
      PIPES_DEVICE_TRAP();
    }

    BlockAtomicU64 credited(storage.recv.credited);
    const uint64_t creditedTail = credited.load(cuda::memory_order_acquire);
    if (creditedTail < command.requiredRecvCredit) {
      // Not a loop: `post_send_once` returns to its caller's polling loop, so
      // this only needs the log-and-trap side effect on abort.
      (void)FT_ABORT_CHECK(
          timeout,
          "IbgdaWarpProxy waiting for receive credit channel=%u "
          "credited=%llu required=%llu",
          command.channel,
          static_cast<unsigned long long>(creditedTail),
          static_cast<unsigned long long>(command.requiredRecvCredit));
      return;
    }
    if (command.slotFreeExpected != 0) {
      const IbChannelProtoSlot& localSlot =
          command.transport->template local_channel_slot<protocol::Simple>(
              command.channel);
      const uint64_t current =
          command.transport->read_signal(localSlot.slotFree);
      if (current < command.slotFreeExpected) {
        // As above: the return is unconditional; this is the log-and-trap.
        (void)FT_ABORT_CHECK(
            timeout,
            "IbgdaWarpProxy waiting for SLOT_FREE channel=%u "
            "expected=%llu current=%llu",
            command.channel,
            static_cast<unsigned long long>(command.slotFreeExpected),
            static_cast<unsigned long long>(current));
        return;
      }
    }

    const IbRemoteChannel remote = makeIbRemoteChannel(
        command.transport->channel_layout(), static_cast<int>(command.channel));
    ThreadGroup solo = make_solo_group(command.channel, fullBlock);
    const IbLocalCompletionTicket ticket = command.transport->put(
        solo,
        command.source,
        remote.recvStaging.subBuffer(command.remoteOffset),
        command.bytes,
        remote.dataReady,
        command.protocolBytes,
        /*counterBuf=*/{},
        /*counterVal=*/0,
        /*signalPerLane=*/true);
    detail::record_send_completion(
        *command.transport,
        command.channel,
        command.slot,
        command.generation,
        ticket);
    posted.store(head + 1, cuda::memory_order_release);
  }

  __device__ __forceinline__ static void run_service(
      SharedState& storage,
      ThreadGroup& service,
      const ThreadGroup& fullBlock,
      const Timeout& timeout) {
    BlockAtomicU32 producerDone(storage.producerDone);
    while (true) {
      uint32_t stop = 0;
      if (service.is_leader()) {
        // An abort has to end this loop on its own. Its only other exit is a
        // fully drained queue, and an abort is precisely what makes that
        // unreachable: a worker that gave up mid-flight leaves posted < tail
        // (or credited < tail) forever, so the drain condition below never
        // becomes true and the service warp spins until the launch is killed.
        //
        // Exiting here does not strand the workers. Their credit and slot waits
        // are FT_ABORT_BREAK-guarded, so the same abort releases both sides --
        // which is the property that matters, since worker and service warp
        // only coordinate through these release/acquire counters and each is
        // otherwise waiting for the other to move them.
        //
        // Folded into the existing `stop` broadcast rather than given its own,
        // so the check is warp-uniform at no extra barrier. Leader-only keeps
        // it to one poll per iteration instead of one per lane.
        //
        // The check runs BEFORE the iteration's peer-visible work, and again
        // between each abortable step, rather than once at the bottom. Both are
        // needed, and for different reasons:
        //
        //   - `post_recv_credits()` emits `signal(slotFree)` and has no abort
        //     check of its own; `post_send_once()` issues a `put` with a fused
        //     `DATA_READY`. With the check below them, the iteration on which
        //     the abort first becomes visible has already sent one more round.
        //   - Hoisting alone does not close it: `publish_recv_readiness()` is
        //     itself abortable, so an abort first observed *inside* it would
        //     still be followed by `post_send_once()` in the same iteration.
        //
        // That is FT principle 4 -- never signal a peer for work you abandoned.
        // A false credit releases a peer that is correctly blocked and stops it
        // ever reaching its own deadline, so one rank's abort silently
        // suppresses fault detection on the rest.
        //
        // The between-step checks are plain `isAborted()` reads rather than new
        // return values: once an abortable step gives up it has already latched
        // the reason in shared state, so a subsequent read sees it. Cheap, and
        // it keeps the change to control flow instead of threading a status
        // through `publish_recv_readiness()`.
        bool aborted = FT_ABORT_CHECK(
            timeout, "IbgdaWarpProxy::run_service abandoning drain");
        // Each step runs only if nothing has aborted yet, and re-reads the flag
        // afterwards because the step itself may have given up inside.
        const auto step = [&](auto&& emit) {
          if (aborted) {
            return;
          }
          emit();
          aborted = timeout.isAborted();
        };
        step([&] { post_recv_credits(storage, fullBlock); });
        step([&] { publish_recv_readiness(storage, timeout); });
        step([&] { post_send_once(storage, fullBlock, timeout); });

        if (aborted) {
          stop = 1U;
        } else if (producerDone.load(cuda::memory_order_acquire) != 0) {
          BlockAtomicU64 sendTail(storage.send.tail);
          BlockAtomicU64 sendPosted(storage.send.posted);
          BlockAtomicU64 recvTail(storage.recv.tail);
          BlockAtomicU64 recvCredited(storage.recv.credited);
          stop = sendPosted.load(cuda::memory_order_acquire) ==
                      sendTail.load(cuda::memory_order_acquire) &&
                  recvCredited.load(cuda::memory_order_acquire) ==
                      recvTail.load(cuda::memory_order_acquire)
              ? 1U
              : 0U;
        }
      }
      stop = service.broadcast(stop);
      if (stop != 0) {
        break;
      }
    }
    service.sync();
  }
};

} // namespace comms::prims

#endif // defined(__CUDACC__) && !defined(__HIP_PLATFORM_AMD__)
