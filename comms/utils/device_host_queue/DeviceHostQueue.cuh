// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>

#include "comms/common/AtomicUtils.cuh"

// =============================================================================
// DeviceHostQueue<T, Policy> — bounded multi-producer ring in shared
// host-pinned memory for passing fixed-size proxy commands one way from GPU to
// CPU (the GPU delegates work to a CPU/engine proxy that executes it). GPU
// threads are the producers; the CPU is the consumer.
//
// Two consumer policies (ConsumerPolicy), selected at compile time:
//   Single  one CPU consumer drains in strict sequence order via a private
//           cursor (MPSC).
//   Multi   several CPU consumer threads drain concurrently (MPMC); each
//           published command is delivered to exactly one consumer. Consumers
//           claim sequences in FIFO order through a shared host-side head and
//           reclaim slots as a contiguous prefix. Processing across consumers
//           is not globally ordered, but there is no loss or duplication.
// The GPU producer protocol is identical for both policies.
//
// Coordinates (plain integers in shared memory; atomicity applied at the access
// site — see "Memory model"):
//   pi         producer reservation sequence (fetch_add / CAS)
//   readySeq  per-slot publish marker (== seq + 1 once published)
//   ci         reuse credit: a producer may overwrite a slot only after ci
//              passes it. The consumer advances ci as it copies commands out,
//              so copy-out frees the slot.
// All consumer claim/completion bookkeeping (the Multi head and done[] markers)
// lives on the host side, never in the cross-domain ring.
//
// Memory model. The device producer uses system-scope atomics (comms::device);
// the release store that publishes readySeq carries the barrier into the CPU
// domain (it orders the payload writes before the marker). The host consumer
// uses std::atomic_ref
// release/acquire (inner-shareable on Grace); that suffices because the ring is
// coherent host-pinned memory and the device peer issues the system-scope half.
// This split is the GB200-validated convention in comms (cf. ctran MemFence.h /
// GpeKernelSync). Multi-consumer coordination (head, done[]) is
// host-only and uses ordinary std::atomic, since all consumers are CPU threads.
//
// Minimal by design: no status/abort/timeout. The blocking variants spin with
// no escape; bounded waits and teardown belong to the caller's progress loop.
//
// Portable across CUDA and HIP/ROCm: device-side atomics go through
// comms::device (AtomicUtils.cuh), and the host-pinned mapped allocation uses
// the CUDA runtime API (hipified 1:1 on AMD, with the coherent-mapped flag).
// =============================================================================

namespace meta::comms {

// The ring's cross-domain words (readySeq, ci) are accessed atomically from
// the GPU (system-scope atomics via comms::device) and the CPU
// (std::atomic_ref). The torn-publish protection assumes those host accesses
// are single instructions, not lock-backed fallbacks; a non-lock-free path
// would not serialize against the GPU's atomic store to the same bytes. This
// always holds on the host ABIs this targets (x86-64, aarch64/Grace), but
// assert it explicitly.
//
// The device half needs no equivalent assert: comms::device uses raw PTX/HIP
// system-scope atomics (atom/ld/st.*.sys.b64), which the ISA guarantees atomic.
static_assert(
    std::atomic_ref<uint64_t>::is_always_lock_free &&
        std::atomic<uint64_t>::is_always_lock_free,
    "DeviceHostQueue requires lock-free 64-bit host atomics");

/// Outcome of a (non-blocking) queue operation.
enum class QueueOpStatus : uint32_t {
  Ok,
  Full, // write(): no credit
  Empty, // read(): ordered head not yet published
};

/// Consumer cardinality, selected at compile time.
enum class ConsumerPolicy : uint32_t {
  Single, // one CPU consumer (MPSC)
  // Many concurrent CPU consumers (MPMC). NOTE: on a coherent fabric (GB200)
  // Multi has LOWER dequeue throughput than Single (head-CAS contention) --
  // pick it to distribute per-command work across threads, not for throughput.
  Multi,
};

/// Data-flow direction. Only D2H (device produces, host consumes) is
/// implemented. H2D (host produces, device consumes) would need a different
/// host fence strategy -- a host system-scope release (DMB SY on aarch64), not
/// the inner-shareable release std::atomic emits -- so it is rejected at
/// compile time (see the static_assert in DeviceHostQueue) rather than silently
/// mis- synchronizing across the GPU domain.
enum class Direction : uint32_t {
  D2H, // device producer -> host consumer
  H2D, // host producer -> device consumer (not yet supported)
};

template <class T, Direction Dir, ConsumerPolicy Policy>
class DeviceHostQueue;

/// One ring slot: payload plus its publish marker. Consumer bookkeeping lives
/// on the host side (see HostConsumerState), never in this cross-domain ring.
template <class T>
struct alignas(64) DeviceHostQueueSlot {
  alignas(alignof(T)) std::byte payload[sizeof(T)];
  // Publish marker (system-scope atomic at the access site). MUST stay LAST: it
  // is written strictly after the payload, via a system-scope release store.
  uint64_t readySeq;
};

/// Shared control header. slots[] follows immediately in the same allocation
/// (see slotsOf()); capacity/mask are carried by value in the device handle,
/// not read from here.
template <class T>
struct alignas(64) DeviceHostQueueControl {
  alignas(64) uint64_t pi; // producer reservation
  alignas(64) uint64_t ci; // reuse credit
};

/// Pointer to the first slot, which lives immediately after the control header.
template <class T>
__host__ __device__ inline DeviceHostQueueSlot<T>* slotsOf(
    DeviceHostQueueControl<T>* ctrl) {
  return reinterpret_cast<DeviceHostQueueSlot<T>*>(
      reinterpret_cast<std::byte*>(ctrl) + sizeof(DeviceHostQueueControl<T>));
}

/// Host-side relax hint for spin-wait loops: the YIELD instruction on
/// aarch64/Grace, PAUSE on x86. It lets an SMT sibling make progress and draws
/// less power while polling, but does NOT deschedule, so an always-busy
/// consumer keeps its latency. A consumer that may idle or shares a core should
/// instead loop on the non-blocking read() with its own backoff (e.g.
/// std::this_thread::yield()).
inline void cpuRelax() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  // host-only helper; no-op if pulled into a device compile
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
  __builtin_ia32_pause();
#endif
}

// =============================================================================
// DeviceHostQueueProducer<T, Policy> — trivially-copyable, non-owning
// device producer handle. Returned by
// DeviceHostQueue::producerHandle() and passed BY VALUE into kernels.
// It holds no cursor, so it is safe to broadcast to every producer thread;
// write()/blockingWrite() may be called concurrently by any number of threads.
// The producer protocol does not depend on the consumer policy.
// =============================================================================
template <class T, ConsumerPolicy Policy = ConsumerPolicy::Single>
class DeviceHostQueueProducer {
 public:
  static_assert(std::is_trivially_copyable_v<T>);
  static_assert(std::is_trivially_destructible_v<T>);

  DeviceHostQueueProducer() = default;

  __host__ __device__ uint32_t capacity() const {
    return capacity_;
  }

  /// Approximate number of in-flight (reserved-but-not-credited) slots. Clamped
  /// to [0, capacity]; not exact under concurrency.
  __device__ uint32_t sizeApprox() const {
    uint64_t p = ::comms::device::ld_relaxed_sys_global(&ctrl_->pi);
    uint64_t c = ::comms::device::ld_relaxed_sys_global(&ctrl_->ci);
    uint64_t d = p > c ? p - c : 0;
    return d > capacity_ ? capacity_ : static_cast<uint32_t>(d);
  }

  /// Non-blocking producer. Reserves a slot only if credit is available, so a
  /// Full return never leaves an unpublished hole. Returns Ok or Full.
  __device__ QueueOpStatus write(const T& value) {
    uint64_t seq = ::comms::device::ld_relaxed_sys_global(&ctrl_->pi);
    for (;;) {
      uint64_t c = ::comms::device::ld_acquire_sys_global(&ctrl_->ci);
      // `seq` may be a stale snapshot of pi here — other producers can advance
      // pi and consumers can advance ci past it before this guard runs — so
      // guard against unsigned underflow: only report Full when seq >= ci. When
      // seq < ci, the CAS below fails and refreshes seq to the current pi.
      if (seq >= c && seq - c >= capacity_) {
        return QueueOpStatus::Full;
      }
      // Claim seq only if pi is unchanged; atomic_cas returns the prior value,
      // so a mismatch refreshes seq and we retry. Nothing is claimed on Full.
      uint64_t prev = ::comms::device::atomic_cas_relaxed_sys_global(
          &ctrl_->pi, seq, seq + 1);
      if (prev == seq) {
        break;
      }
      seq = prev;
    }
    publish(seq, value);
    return QueueOpStatus::Ok;
  }

  /// Blocking producer: reserve a slot, spin for reuse credit, then publish. No
  /// escape — assumes the consumer keeps draining. Callers needing to bail out
  /// (timeout/teardown) should loop on write() instead.
  ///
  /// Reserve-then-publish obligation: unlike write() (which CAS-claims pi only
  /// when it is about to publish), blockingWrite reserves its sequence with an
  /// unconditional fetch_add and is therefore committed to publishing it. A
  /// thread that reserves a sequence and then never publishes (aborts mid-call)
  /// leaves a permanent hole that wedges the in-order consumer at that
  /// sequence. Do not abort between the reservation and publish().
  __device__ void blockingWrite(const T& value) {
    uint64_t seq = ::comms::device::atomic_fetch_add_relaxed_sys_global(
        &ctrl_->pi, /*val=*/1);
    // seq is exclusively owned and unpublished until publish() below, so it
    // cannot be consumed while we spin; ci <= seq always holds here and the
    // subtraction never underflows (unlike write(), which guards a stale seq).
    while (seq - ::comms::device::ld_acquire_sys_global(&ctrl_->ci) >=
           capacity_) {
      // spin for reuse credit
    }
    publish(seq, value);
  }

 private:
  friend class DeviceHostQueue<T, Direction::D2H, Policy>;

  /// Publish the reserved slot `seq`: copy the payload in, then make it visible
  /// with a single system-scope *release* store of the marker (readySeq ==
  /// seq + 1). That release store orders the payload writes before the marker
  /// becomes visible, so a consumer that acquire-loads readySeq == seq + 1
  /// always sees the full payload. The release store alone carries this
  /// ordering — a separate release fence would be redundant (one would be
  /// needed only if the marker store were relaxed).
  __device__ void publish(uint64_t seq, const T& value) {
    DeviceHostQueueSlot<T>& slot = slots_[seq & mask_];
    memcpy(slot.payload, &value, sizeof(T));
    ::comms::device::st_release_sys_global(&slot.readySeq, seq + 1);
  }

  DeviceHostQueueControl<T>* ctrl_ = nullptr; // device pointer
  DeviceHostQueueSlot<T>* slots_ = nullptr; // device pointer
  uint32_t capacity_ = 0;
  uint32_t mask_ =
      0; // capacity - 1; slot index = seq & mask_ (capacity is a power of two)
};

// =============================================================================
// Host-side consumer state, specialized per policy. Kept off the shared ring;
// for Multi it is shared only among the (CPU) consumer threads.
// =============================================================================
template <ConsumerPolicy Policy>
struct HostConsumerState;

template <>
struct HostConsumerState<ConsumerPolicy::Single> {
  uint64_t nextToRead = 0; // private cursor of the one consumer
  explicit HostConsumerState(uint32_t /*capacity*/) {}
};

template <>
struct HostConsumerState<ConsumerPolicy::Multi> {
  // Shared FIFO claim counter: each consumer CAS-claims the next sequence.
  std::atomic<uint64_t> head{0};
  // Per-slot completion markers (seq & mask): done[s] == seq + 1 once seq is
  // copied out (absolute seqs -> ABA-safe). alignas(64) puts each on its own
  // line so consumers completing nearby seqs don't false-share. Costs cap *
  // 64B.
  struct alignas(64) DoneMarker : std::atomic<uint64_t> {};
  std::unique_ptr<DoneMarker[]> done;
  explicit HostConsumerState(uint32_t capacity)
      : done(std::make_unique<DoneMarker[]>(capacity)) {}
};

// =============================================================================
// DeviceHostQueue<T, Policy> — host RAII owner of the cudaHostAlloc
// allocation, and the CPU-side consumer. Non-copyable, non-movable. The host
// consumer methods (read/blockingRead) mirror the device producer over the
// same plain words via std::atomic_ref. Hand producerHandle() to the GPU.
//
// Single: read()/blockingRead() must be called by one consumer only. Multi:
// they are safe to call from many threads concurrently.
//
// Teardown: the caller must quiesce all producers (abort flag + stream sync)
// before destruction — freeing a live shared ring is undefined behavior.
// =============================================================================
template <
    class T,
    Direction Dir = Direction::D2H,
    ConsumerPolicy Policy = ConsumerPolicy::Single>
class DeviceHostQueue {
 public:
  static_assert(std::is_trivially_copyable_v<T>);
  static_assert(std::is_trivially_destructible_v<T>);
  static_assert(
      Dir == Direction::D2H,
      "DeviceHostQueue: only Direction::D2H (device producer -> host consumer) "
      "is supported; H2D needs a host system-scope fence and is not yet "
      "implemented");

  using Producer = DeviceHostQueueProducer<T, Policy>;

  struct ReadResult {
    T value;
    uint64_t seq;
  };

  /// Allocate and zero-initialize a queue. `capacity` must be a power of two.
  /// `requirePowerOfTwo` runs first (init-list), so a bad capacity throws
  /// before any allocation.
  explicit DeviceHostQueue(uint32_t capacity)
      : consumer_(requirePowerOfTwo(capacity)) {
    capacity_ = capacity;
    mask_ = capacity - 1;
    static_assert(
        sizeof(DeviceHostQueueControl<T>) % alignof(DeviceHostQueueSlot<T>) ==
            0,
        "slots[] must start on its natural alignment after the control header");
    static_assert(
        offsetof(DeviceHostQueueSlot<T>, readySeq) >
            offsetof(DeviceHostQueueSlot<T>, payload),
        "readySeq must remain the last field, after the payload");

    const size_t bytes = sizeof(DeviceHostQueueControl<T>) +
        static_cast<size_t>(capacity) * sizeof(DeviceHostQueueSlot<T>);

    void* host = nullptr;
#if defined(__HIP_PLATFORM_AMD__)
    // Fine-grained (coherent) + mapped so a GPU system-scope release store is
    // visible to a concurrently-polling CPU consumer mid-kernel, with no manual
    // cache flush. cudaHostAlloc below hipifies to hipHostMalloc.
    const unsigned kHostAllocFlags =
        hipHostMallocMapped | hipHostMallocCoherent;
#else
    const unsigned kHostAllocFlags = cudaHostAllocMapped;
#endif
    if (cudaError_t e = cudaHostAlloc(&host, bytes, kHostAllocFlags);
        e != cudaSuccess) {
      throw std::runtime_error(
          std::string("DeviceHostQueue: cudaHostAlloc: ") +
          cudaGetErrorString(e));
    }
    // cudaHostAlloc does not zero; this establishes pi=ci=0 and readySeq=0 so
    // the readySeq==seq+1 scheme is unambiguous from seq 0.
    std::memset(host, 0, bytes);
    void* dev = nullptr;
    if (cudaError_t e = cudaHostGetDevicePointer(&dev, host, 0);
        e != cudaSuccess) {
      // ctrlHost_ is still null here, so the destructor would not free `host`;
      // release it on this error path to avoid leaking the pinned allocation.
      cudaFreeHost(host);
      throw std::runtime_error(
          std::string("DeviceHostQueue: cudaHostGetDevicePointer: ") +
          cudaGetErrorString(e));
    }
    ctrlHost_ = reinterpret_cast<DeviceHostQueueControl<T>*>(host);
    ctrlDev_ = reinterpret_cast<DeviceHostQueueControl<T>*>(dev);
    slotsHost_ = slotsOf(ctrlHost_);
    slotsDev_ = slotsOf(ctrlDev_);
  }

  ~DeviceHostQueue() {
    // Precondition: producers already quiesced; freeing a live ring is
    // undefined behavior.
    if (ctrlHost_ != nullptr) {
      cudaFreeHost(ctrlHost_);
    }
  }

  DeviceHostQueue(const DeviceHostQueue&) = delete;
  DeviceHostQueue& operator=(const DeviceHostQueue&) = delete;
  DeviceHostQueue(DeviceHostQueue&&) = delete;
  DeviceHostQueue& operator=(DeviceHostQueue&&) = delete;

  uint32_t capacity() const {
    return capacity_;
  }

  /// Approximate backlog (published-but-not-yet-consumed slots). Clamped to
  /// [0, capacity]; not exact under concurrency.
  size_t sizeApprox() const {
    uint64_t p = std::atomic_ref<uint64_t>(ctrlHost_->pi)
                     .load(std::memory_order_relaxed);
    uint64_t c = std::atomic_ref<uint64_t>(ctrlHost_->ci)
                     .load(std::memory_order_relaxed);
    uint64_t d = p > c ? p - c : 0;
    return d > capacity_ ? capacity_ : static_cast<size_t>(d);
  }

  /// Consumer dequeue: copies the next published command out and releases its
  /// slot (advances ci) for producer reuse. Returns the payload and its
  /// sequence via `out`. Returns Ok or Empty.
  ///
  /// Single: call from one consumer only. Multi: safe from many threads; each
  /// command is returned to exactly one caller, claimed in FIFO order.
  QueueOpStatus read(ReadResult& out) {
    if constexpr (Policy == ConsumerPolicy::Single) {
      uint64_t seq = consumer_.nextToRead;
      DeviceHostQueueSlot<T>& slot = slotsHost_[seq & mask_];
      uint64_t ready = std::atomic_ref<uint64_t>(slot.readySeq)
                           .load(std::memory_order_acquire);
      if (ready != seq + 1) {
        return QueueOpStatus::Empty;
      }
      memcpy(&out.value, slot.payload, sizeof(T)); // copy out: slot now free
      out.seq = seq;
      consumer_.nextToRead = seq + 1;
      // Publish reuse credit; release pairs with the producer's acquire of ci.
      std::atomic_ref<uint64_t>(ctrlHost_->ci)
          .store(seq + 1, std::memory_order_release);
      return QueueOpStatus::Ok;
    } else { // ConsumerPolicy::Multi
      std::atomic<uint64_t>& head = consumer_.head;
      for (;;) {
        uint64_t h = head.load(std::memory_order_acquire);
        DeviceHostQueueSlot<T>& slot = slotsHost_[h & mask_];
        uint64_t ready = std::atomic_ref<uint64_t>(slot.readySeq)
                             .load(std::memory_order_acquire);
        if (ready != h + 1) {
          return QueueOpStatus::Empty; // head not yet published
        }
        if (!head.compare_exchange_weak(
                h,
                h + 1,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
          continue; // another consumer took h (or head moved); retry
        }
        // Exclusive owner of seq h. The slot stays valid while we copy: a
        // producer cannot reuse it until ci passes h, and ci only passes h
        // after the done-marker we publish just below.
        memcpy(&out.value, slot.payload, sizeof(T));
        out.seq = h;
        // Mark h consumed; release orders our copy-out before any consumer that
        // observes this marker advances ci past h.
        consumer_.done[h & mask_].store(h + 1, std::memory_order_release);
        advanceReuseCreditMulti();
        return QueueOpStatus::Ok;
      }
    }
  }

  /// Batched dequeue: copies up to `max` contiguous published commands into
  /// out[0..n) in sequence order and returns n (0 if the head is unpublished).
  /// Prefer this over read() in a drain loop -- it amortizes the contended
  /// cross-domain op over the batch.
  ///
  /// Single: scans from the private cursor and frees the whole batch with a
  /// SINGLE ci release store (vs one per command in read()).
  /// Multi: claims a contiguous published run [h, h+n) in ONE head CAS (vs one
  /// CAS per command in read()), cutting head-CAS contention ~n x. Safe to call
  /// concurrently; each command is delivered to exactly one caller.
  size_t readMulti(ReadResult* out, size_t max) {
    if constexpr (Policy == ConsumerPolicy::Single) {
      size_t n = 0;
      uint64_t seq = consumer_.nextToRead;
      while (n < max) {
        DeviceHostQueueSlot<T>& slot = slotsHost_[seq & mask_];
        // Per-slot acquire pairs with the producer's release of readySeq, so
        // the payload copied just below is fully visible.
        uint64_t ready = std::atomic_ref<uint64_t>(slot.readySeq)
                             .load(std::memory_order_acquire);
        if (ready != seq + 1) {
          break; // first unpublished slot ends the contiguous batch
        }
        memcpy(&out[n].value, slot.payload, sizeof(T));
        out[n].seq = seq;
        ++n;
        ++seq;
      }
      if (n > 0) {
        consumer_.nextToRead = seq;
        // One release store frees the whole batch and orders the n copy-outs
        // before the credit; pairs with the producer's acquire of ci.
        std::atomic_ref<uint64_t>(ctrlHost_->ci)
            .store(seq, std::memory_order_release);
      }
      return n;
    } else { // ConsumerPolicy::Multi
      std::atomic<uint64_t>& head = consumer_.head;
      for (;;) {
        uint64_t h = head.load(std::memory_order_acquire);
        // Count the contiguous published run [h, h+n). The acquire on each
        // readySeq pairs with the producer's release, so the payloads copied
        // after the claim are visible. A scanned-published slot can't be
        // reclaimed before we claim it (its done marker is unset and head has
        // not passed it), so the run stays valid through the CAS below.
        size_t n = 0;
        while (n < max) {
          DeviceHostQueueSlot<T>& slot = slotsHost_[(h + n) & mask_];
          uint64_t ready = std::atomic_ref<uint64_t>(slot.readySeq)
                               .load(std::memory_order_acquire);
          if (ready != h + n + 1) {
            break;
          }
          ++n;
        }
        if (n == 0) {
          return 0; // head not yet published
        }
        // Claim the whole run in one CAS; if head moved, re-peek and retry.
        if (!head.compare_exchange_weak(
                h,
                h + n,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
          continue;
        }
        for (size_t i = 0; i < n; ++i) {
          DeviceHostQueueSlot<T>& slot = slotsHost_[(h + i) & mask_];
          memcpy(&out[i].value, slot.payload, sizeof(T));
          out[i].seq = h + i;
          // Release: orders our copy-out before any consumer that observes this
          // marker advances ci past h+i.
          consumer_.done[(h + i) & mask_].store(
              h + i + 1, std::memory_order_release);
        }
        advanceReuseCreditMulti();
        return n;
      }
    }
  }

  /// Blocking consumer dequeue: spins until a command is available, then copies
  /// it out and releases its slot. Each empty poll issues a cpuRelax()
  /// hint (YIELD on aarch64/Grace, PAUSE on x86) so it does not burn the core
  /// hot. No escape (assumes a producer makes progress); a consumer that may
  /// idle, shares a core, or must also poll other things should loop on the
  /// non-blocking read() with its own backoff instead.
  void blockingRead(ReadResult& out) {
    while (read(out) == QueueOpStatus::Empty) {
      cpuRelax();
    }
  }

  /// Device producer handle for kernels — cursor-free and trivially copyable,
  /// so it is safe to pass BY VALUE to every producer thread. Idempotent.
  Producer producerHandle() const {
    Producer h;
    h.ctrl_ = ctrlDev_;
    h.slots_ = slotsDev_;
    h.capacity_ = capacity_;
    h.mask_ = mask_;
    return h;
  }

 private:
  static uint32_t requirePowerOfTwo(uint32_t capacity) {
    if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
      throw std::runtime_error(
          "DeviceHostQueue: capacity must be a power of two");
    }
    return capacity;
  }

  // Advance the shared reuse credit (ci) over the contiguous prefix of consumed
  // slots (those whose absolute done-marker is published) so producers reclaim
  // them. Opportunistic: whoever wins the ci CAS walks the prefix; a loser
  // reloads and bails once the winner has drained to the last visible marker.
  // So one consumer drains at a time -- not a livelock (each pass advances ci
  // or breaks; ci is monotonic, bounded by markers). Reclaim is thus
  // eventually- consistent and never deadlocks a producer (which blocks only on
  // a full ring, i.e. commands are waiting, so consumers keep draining). A
  // consumer that claims but never completes a seq stalls reclaim, like a dead
  // Single consumer
  // -- the caller's concern (no abort/timeout). Multi only.
  void advanceReuseCreditMulti() {
    std::atomic_ref<uint64_t> ci(ctrlHost_->ci);
    for (;;) {
      uint64_t c = ci.load(std::memory_order_relaxed);
      if (consumer_.done[c & mask_].load(std::memory_order_acquire) != c + 1) {
        break; // seq c not consumed yet -> end of contiguous freed prefix
      }
      // release: a producer acquiring ci sees the copy-out of every seq < c + 1
      // (transitively, via the done acquire above).
      if (ci.compare_exchange_weak(
              c, c + 1, std::memory_order_release, std::memory_order_relaxed)) {
        continue; // freed c; try to extend the prefix
      }
      // lost the race; reload -- the winner may already have drained the prefix
    }
  }

  DeviceHostQueueControl<T>* ctrlHost_ = nullptr;
  DeviceHostQueueControl<T>* ctrlDev_ = nullptr;
  DeviceHostQueueSlot<T>* slotsHost_ = nullptr;
  DeviceHostQueueSlot<T>* slotsDev_ = nullptr;
  uint32_t capacity_ = 0;
  uint32_t mask_ =
      0; // capacity - 1; slot index = seq & mask_ (capacity is a power of two)
  HostConsumerState<Policy> consumer_; // off-ring consumer bookkeeping
};

} // namespace meta::comms
