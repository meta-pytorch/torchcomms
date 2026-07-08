// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>

#include "comms/common/AtomicUtils.cuh"

// =============================================================================
// DeviceHostQueue<T, Policy> — bounded multi-producer ring in shared
// host-pinned memory for passing fixed-size proxy commands from GPU to CPU (the
// GPU delegates work to a CPU/engine proxy that runs it).
//
// Data flows one way: GPU threads are the producers, the CPU (the host owner)
// is the single consumer. Only ConsumerPolicy::Single is implemented: many
// producer threads reserve and publish slots, and the one host consumer drains
// them in sequence order. A future ConsumerPolicy::Multi would add multiple
// consumers, keeping per-consumer state off this shared ring.
//
// Coordinates (plain integers in shared memory; atomicity applied at the access
// site — see "Memory model"):
//   pi         producer reservation sequence (fetch_add / CAS)
//   readySeq  per-slot publish marker (== seq + 1 once published)
//   ci         reuse credit: a producer may overwrite a slot only after ci
//              passes it. read() advances ci as it copies each command out, so
//              copy-out frees the slot.
//
// Memory model. The device producer uses system-scope atomics (comms::device);
// the release store that publishes readySeq carries the barrier into the CPU
// domain (it orders the payload writes before the marker). The host consumer
// uses std::atomic_ref
// release/acquire (inner-shareable on Grace); that suffices because the ring is
// coherent host-pinned memory and the device peer issues the system-scope half.
// This split is the GB200-validated convention in comms (cf. ctran MemFence.h /
// GpeKernelSync).
//
// Minimal by design: no status/abort/timeout. The blocking variants spin with
// no escape; bounded waits and teardown belong to the caller's progress loop.
//
// Portable across CUDA and HIP/ROCm: device-side atomics go through
// comms::device (AtomicUtils.cuh), and the host-pinned mapped allocation uses
// the CUDA runtime API (hipified 1:1 on AMD, with the coherent-mapped flag).
// =============================================================================

namespace meta::comms {

/// Outcome of a (non-blocking) queue operation.
enum class QueueOpStatus : uint32_t {
  Ok,
  Full, // write(): no credit
  Empty, // read(): ordered head not yet published
};

/// Consumer cardinality. Only Single is implemented in the initial cut; Multi
/// is reserved for a future specialization.
enum class ConsumerPolicy : uint32_t {
  Single,
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

/// One ring slot: payload plus its publish marker. Per-consumer bookkeeping
/// (for a future Multi policy) lives on the consumer side, never in this
/// cross-domain ring.
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
// =============================================================================
template <class T, ConsumerPolicy Policy = ConsumerPolicy::Single>
class DeviceHostQueueProducer {
 public:
  static_assert(std::is_trivially_copyable_v<T>);
  static_assert(std::is_trivially_destructible_v<T>);
  static_assert(
      Policy == ConsumerPolicy::Single,
      "DeviceHostQueue: only ConsumerPolicy::Single is implemented");

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
// DeviceHostQueue<T, Policy> — host RAII owner of the cudaHostAlloc
// allocation, and the single (CPU) consumer. Non-copyable, non-movable. The
// host consumer methods (read/blockingRead) mirror the device producer over
// the same plain words via std::atomic_ref. Hand producerHandle() to the GPU.
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
  static_assert(
      Policy == ConsumerPolicy::Single,
      "DeviceHostQueue: only ConsumerPolicy::Single is implemented");

  using Producer = DeviceHostQueueProducer<T, Policy>;

  struct ReadResult {
    T value;
    uint64_t seq;
  };

  /// Allocate and zero-initialize a queue. `capacity` must be a power of two.
  explicit DeviceHostQueue(uint32_t capacity) {
    if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
      throw std::runtime_error(
          "DeviceHostQueue: capacity must be a power of two");
    }
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

  /// Single-owner host consumer dequeue: copies the next published command out
  /// and releases its slot (advances ci) for producer reuse. Returns the
  /// payload and its sequence via `out`. Returns Ok or Empty.
  QueueOpStatus read(ReadResult& out) {
    DeviceHostQueueSlot<T>& slot = slotsHost_[nextToRead_ & mask_];
    uint64_t ready = std::atomic_ref<uint64_t>(slot.readySeq)
                         .load(std::memory_order_acquire);
    if (ready != nextToRead_ + 1) {
      return QueueOpStatus::Empty;
    }
    memcpy(
        &out.value, slot.payload, sizeof(T)); // copy out: slot no longer needed
    out.seq = nextToRead_;
    ++nextToRead_;
    // Publish reuse credit; release pairs with the producer's acquire of ci.
    std::atomic_ref<uint64_t>(ctrlHost_->ci)
        .store(nextToRead_, std::memory_order_release);
    return QueueOpStatus::Ok;
  }

  /// Batched MPSC dequeue: copies up to `max` contiguous published commands
  /// into out[0..n) in sequence order, then releases all n slots with a SINGLE
  /// ci release store (vs one per command in read()). That store is the
  /// cross-domain write that bounds drain throughput, so batching it is the
  /// main consumer-side perf lever; prefer this over read() in a drain loop.
  /// Returns n; n == 0 means the head is not yet published. MPSC only.
  size_t readMulti(ReadResult* out, size_t max) {
    static_assert(
        Policy == ConsumerPolicy::Single,
        "readMulti is MPSC-only; with ConsumerPolicy::Multi each consumer "
        "claims via read()");
    size_t n = 0;
    uint64_t seq = nextToRead_;
    while (n < max) {
      DeviceHostQueueSlot<T>& slot = slotsHost_[seq & mask_];
      // Per-slot acquire pairs with the producer's release of readySeq, so the
      // payload copied just below is fully visible.
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
      nextToRead_ = seq;
      // One release store frees the whole batch and orders the n copy-outs
      // before the credit; pairs with the producer's acquire of ci.
      std::atomic_ref<uint64_t>(ctrlHost_->ci)
          .store(seq, std::memory_order_release);
    }
    return n;
  }

  /// Blocking host consumer dequeue: spins until the head is published, then
  /// copies it out and releases its slot. Each empty poll issues a cpuRelax()
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
  DeviceHostQueueControl<T>* ctrlHost_ = nullptr;
  DeviceHostQueueControl<T>* ctrlDev_ = nullptr;
  DeviceHostQueueSlot<T>* slotsHost_ = nullptr;
  DeviceHostQueueSlot<T>* slotsDev_ = nullptr;
  uint32_t capacity_ = 0;
  uint32_t mask_ =
      0; // capacity - 1; slot index = seq & mask_ (capacity is a power of two)
  uint64_t nextToRead_ = 0; // host consumer cursor (single consumer)
};

} // namespace meta::comms
