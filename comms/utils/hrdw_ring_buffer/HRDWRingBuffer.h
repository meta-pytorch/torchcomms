// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h> // @manual

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include <cuda/atomic> // @manual
#endif

// needed for US_TICK_TIMESTAMP_SHIFT
#include "comms/utils/hrdw_ring_buffer/GpuClockCalibration.h"

#include <algorithm>
#include <atomic>
#if __cplusplus >= 202002L
#include <bit>
#endif
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

// EMPTY epoch sentinel: a freshly initialized slot has epoch 0. Real
// writes encode epoch = uint32_t(slot >> shift) + 1, so 0 is unused.
#define HRDW_RINGBUFFER_SLOT_EMPTY 0u

namespace hrdw_ring_buffer {

// Memory coherence scope controls the allocation strategy, atomic
// operations, and read/write paths of the ring buffer.
//
//   System — pinned mapped memory, system-scope atomics, 128-bit atomic
//            slot exchanges. Host can poll concurrently via mapped
//            pointers. Requires sm_90+ for atom.exch.b128.
//
//   Device — device-global memory, device-scope atomics, 128-bit atomic
//            slot exchanges. Host reads via cudaMemcpy after
//            cudaStreamSynchronize (no concurrent reader). Requires
//            sm_90+ for atom.exch.b128.
enum class MemoryCoherenceScope { System, Device };

// Write policy — a structural property of the ring, like MemoryCoherenceScope.
//
//   Overwrite — the writer never blocks; a lapping write silently overwrites
//               the oldest unconsumed slot (the reader detects loss via its
//               writeIndex bound). The default; correct only when the consumer
//               is guaranteed to keep up.
//   Blocking  — lossless. The writer spins until the reader has consumed the
//               slot it is about to reuse (backpressure against a published
//               consumer cursor), so a write can never overwrite an unconsumed
//               entry. A Blocking ring allocates the consumer cursor AND its
//               reader publishes it — the two halves that make "the reader will
//               see every write" a structural guarantee, inseparable by
//               construction. Requires System scope (host-published cursor).
enum class WritePolicy { Overwrite, Blocking };

namespace detail {

// An empty stand-in for an absent member. Templated on T to produce distinct
// types so that multiple [[no_unique_address]] absent members each cost 0 bytes
// (two subobjects of the same type would need distinct addresses).
// Constructible from anything so the absent member silently ignores any
// initializer.
template <typename T>
struct Nothing {
  constexpr Nothing() noexcept = default;
  template <typename... Args>
  constexpr explicit Nothing(Args&&...) noexcept {}
};

// define_if<Cond, T> is T when Cond, otherwise an empty Nothing<T>.
// Pair with [[no_unique_address]] so the absent member costs zero bytes.
template <bool Cond, typename T>
using define_if = std::conditional_t<Cond, T, Nothing<T>>;

} // namespace detail

// ==========================================================================
// HRDWRingBuffer — Host-Read Device-Write ring buffer for GPU-to-CPU
// event transfer with lock-free, torn-read-safe polling.
//
// OVERVIEW
//   GPU kernels claim a slot via atomicAdd on a mapped writeIndex, then
//   emit a single 128-bit system-scope atomic exchange of {timestamp,
//   epoch, data}. The CPU polls each slot with a single 16B atomic load,
//   which the writer's slot-atomic publication guarantees observes either
//   the fully-old or fully-new state — no torn reads, no retries.
//
// USAGE (MemoryCoherenceScope::System — default)
//   1. Create a ring buffer. DataT must fit in 8 bytes (the entry packs
//      into exactly 16 bytes for atomic 128b stores). For non-trivially-
//      destructible DataT, the destructor IS invoked on the displaced
//      occupant when a slot is overwritten — the writer reclaims the
//      previous bytes via atom.exch.b128 and runs ~DataT() on them, so
//      refcounted DataT types stay balanced across overwrites.
//
//        HRDWRingBuffer<MyEvent> buf(4096);
//
//   2. Write events. Each write claims a slot, packs timestamp + epoch +
//      data into 16 bytes, and emits a single 128-bit store:
//
//        buf.write(stream, myEvent);
//
//   3. On the CPU, create a reader and poll periodically. Convert the
//      packed timestamp via GlobaltimerCalibration::toWallClock(uint32_t):
//
//        HRDWRingBufferReader<MyEvent> reader(buf);
//        reader.poll([&](const auto& entry, uint64_t slot) {
//          const MyEvent& evt = entry.data;
//          auto wall = cal.toWallClock(entry.timestamp);
//        });
//
// USAGE (MemoryCoherenceScope::Device)
//   1. Create a ring buffer. DataT must be ≤ 8 bytes so the entry
//      packs into 16 bytes for atomic 128b stores. Non-trivially-
//      destructible DataT will silently leak per-occupant resources
//      on ring teardown — the host can't reach into device memory to
//      run ~DataT(). Drain + destruct host-side if cleanup matters.
//
//        HRDWRingBuffer<MyEvent, MemoryCoherenceScope::Device> buf(4096);
//
//   2. Write events from GPU kernels via the device handle:
//
//        auto handle = buf.deviceHandle();
//        myKernel<<<...>>>(handle, ...);
//        // inside kernel: handle.write(myEvent);
//
//   3. Create a reader and poll from the host between steps:
//
//        HRDWRingBufferReader<MyEvent, MemoryCoherenceScope::Device>
//            reader(buf);
//        reader.poll(stream, [&](const auto& entry, uint64_t slot) {
//          const MyEvent& evt = entry.data;
//        });
//        // Poll again whenever you want fresh entries — poll() auto-
//        // advances its internal read cursor, so successive calls only
//        // return entries written since the previous one.
//
// LAYOUT (16 bytes, 16-byte aligned)
//   - timestamp: uint32_t — globaltimer_ns >> 10 (~1024ns ticks). Wraps
//                every 2^42 ns ≈ 73 min; the reader reconstructs full
//                precision against system_clock::now() via
//                GlobaltimerCalibration::toWallClock(uint32_t), so callers
//                must poll often enough that no slot lingers past the
//                ~73 min wrap window. Older entries alias to a more
//                recent time and cannot be recovered from 32 bits alone.
//   - epoch:     uint32_t — uint32_t(slot >> log2(size_)) + 1.
//                Zero is the EMPTY sentinel; the +1 keeps slot 0's epoch
//                non-zero so a never-written slot is always distinguishable
//                from any valid write.
//   - data:      DataT, ≤ 8 bytes (padded to 16 if smaller).
//
// MEMORY ORDERING
//   System scope:
//   - The writer issues a single 128b atomic exchange at system scope via
//     cuda::atomic_ref<unsigned __int128, thread_scope_system>::exchange
//     (libcu++). Lowers to atom.exch.b128 / atom.cas.b128 on sm_90+,
//     which is uncontended in practice because the preceding writeIndex
//     atomicAdd already serialized us against other writers. The
//     exchange is slot-atomic in hardware — the reader never sees a torn
//     entry from a concurrent writer.
//   - On unsupported architectures (HIP, pre-Hopper CUDA), the System
//     writer compiles to a trap so fat binaries link cleanly; only
//     sm_90+ cubins carry the real trace path.
//   - The displaced bytes are reinterpreted as HRDWEntry<DataT> and
//     ~DataT() is invoked on the previous occupant when the slot was
//     non-empty, so refcounted DataT types stay balanced across slot
//     overwrites.
//   - Release ordering w.r.t. the writeIndex atomicAdd is unnecessary:
//     the reader's per-slot epoch check + writeIndex bound is
//     self-correcting regardless of the order in which the slot write
//     and writeIndex bump become visible to the host.
//   - Reader pattern: a single 16B atomic relaxed-load of ring[idx]
//     (lowers to LDP on aarch64 / LSE2, cmpxchg16b on x86_64). Because
//     the writer's atom.exch.b128 publishes the slot atomically, the
//     host load sees either the fully-old or fully-new state. The reader
//     then compares the loaded epoch against `(slot >> log2(size_)) + 1`
//     to classify the entry as kSuccess / kOverwritten / kNotReady.
//   Device scope:
//   - Same writer pattern as System but uses atom.exch.relaxed.gpu.b128
//     (device scope rather than system scope). The b128 atomic store
//     prevents torn writes when two writers' slot claims collide on a
//     lapping ring; the reader uses the per-slot epoch to discard
//     stale-generation slots whose atomic stores happened to complete
//     out of claim order.
//   - Host reader uses cudaMemcpy after cudaStreamSynchronize rather
//     than concurrent mapped-memory polling, so there is no host-vs-
//     device race to worry about.
//
// RING SIZING
//   The size is rounded up to a power of 2. Larger rings reduce loss
//   from lapping; overwritten entries are reported as entriesLost.
// ==========================================================================

// 16-byte ring buffer entry. Layout is fixed for atomic 128b stores.
template <typename DataT, MemoryCoherenceScope C = MemoryCoherenceScope::System>
struct alignas(16) HRDWEntry {
  uint32_t timestamp;
  uint32_t epoch;
  DataT data;
};

// Device-side inline write into a ring buffer. Claims a slot via a
// scope-appropriate atomicAdd, then publishes {timestamp, epoch, data}
// as a single 128b atomic exchange (`.sys` for System, `.gpu` for
// Device). The b128 atomic publication is what prevents torn writes
// when two writers' slot claims collide on a lapping ring.
//
// Compiled to a trap on unsupported architectures (HIP, pre-Hopper
// CUDA) so fat binaries build cleanly — only sm_90+ cubins carry the
// real trace path.
template <typename DataT, MemoryCoherenceScope C = MemoryCoherenceScope::System>
struct HrdwRingBufferWriter {
  using EntryT = HRDWEntry<DataT, C>;
  static_assert(
      sizeof(EntryT) == 16,
      "HRDWEntry must be exactly 16 bytes (DataT must be <= 8 bytes)");
  static_assert(alignof(EntryT) == 16, "HRDWEntry must be 16-byte aligned");

#if defined(__CUDACC__) && !defined(__HIPCC__)
  // Atomic scope shared by every device access below: System for cross-host
  // coherence, Device for intra-GPU. Guarded to mirror the <cuda/atomic>
  // include above — cuda::thread_scope_* is unavailable under HIP.
  static constexpr auto kScope = (C == MemoryCoherenceScope::System)
      ? cuda::thread_scope_system
      : cuda::thread_scope_device;
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ __forceinline__ static void write(
      EntryT* ring,
      uint64_t* writeIndex,
      uint32_t mask,
      [[maybe_unused]] uint32_t shift,
      DataT data) {
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900)
    (void)ring;
    (void)writeIndex;
    (void)mask;
    (void)data;
    printf(
        "[HRDWRingBuffer] write unsupported on this GPU (requires sm_90+)\n");
#if defined(__HIPCC__)
    abort();
#else
    __trap();
#endif
#else
    // Relaxed fetch-add: the only invariant we need from this is "every
    // writer gets a unique slot number"; ordering with the subsequent
    // slot publication is unnecessary (the reader's epoch check is
    // self-correcting if the writeIndex bump and the slot bytes become
    // host-visible in either order).
    uint64_t slot = cuda::atomic_ref<uint64_t, kScope>(*writeIndex)
                        .fetch_add(1ULL, cuda::memory_order_relaxed);
    publishSlot(ring, slot, mask, shift, data);
#endif
  }

  // Blocking variant of write(): after claiming a slot, spin until the reader
  // has consumed the slot's prior occupant (i.e. the writer stays at most
  // `size` = mask + 1 ahead of the host-published `readIndex`), so a publish
  // can never lap an unconsumed entry. This makes the ring lossless regardless
  // of consumer timing — the non-blocking write() silently overwrites under a
  // slow/stalled consumer. `abortFlag` is the ring's device-visible abort flag
  // (raised by requestAbort() on teardown): a non-zero value breaks the wait so
  // a blocked writer can never hang the GPU once the consumer is gone. On abort
  // we still publish rather than abandon the claimed slot, which would stall
  // the in-order reader forever. Requires a valid host-published `readIndex`
  // and `abortFlag` (System scope).
  __device__ __forceinline__ static void write_blocking(
      EntryT* ring,
      uint64_t* writeIndex,
      uint32_t mask,
      [[maybe_unused]] uint32_t shift,
      DataT data,
      const uint64_t* readIndex,
      const uint32_t* abortFlag,
      uint32_t backoffNanos) {
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900)
    (void)ring;
    (void)writeIndex;
    (void)mask;
    (void)data;
    (void)readIndex;
    (void)abortFlag;
    (void)backoffNanos;
    printf(
        "[HRDWRingBuffer] write_blocking unsupported on this GPU (requires sm_90+)\n");
#if defined(__HIPCC__)
    abort();
#else
    __trap();
#endif
#else
    uint64_t slot = cuda::atomic_ref<uint64_t, kScope>(*writeIndex)
                        .fetch_add(1ULL, cuda::memory_order_relaxed);
    // Acquire-load pairs with the reader's release-store of its consumed
    // cursor. slot - readIndex > mask means slot's prior occupant
    // (slot - size) has not been consumed yet.
    auto readRef =
        cuda::atomic_ref<uint64_t, kScope>(*const_cast<uint64_t*>(readIndex));
    auto abortRef =
        cuda::atomic_ref<uint32_t, kScope>(*const_cast<uint32_t*>(abortFlag));
    while ((slot - readRef.load(cuda::memory_order_acquire)) > mask) {
      if (abortRef.load(cuda::memory_order_acquire) != 0) {
        break;
      }
      __nanosleep(backoffNanos);
    }
    publishSlot(ring, slot, mask, shift, data);
#endif
  }

  // Publish `data` at an already-claimed monotonic `slot`. Factored out so
  // write() and write_blocking() share the single hand-tuned 128b atomic
  // publication below.
  __device__ __forceinline__ static void publishSlot(
      EntryT* ring,
      uint64_t slot,
      uint32_t mask,
      uint32_t shift,
      DataT data) {
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900)
    (void)ring;
    (void)slot;
    (void)mask;
    (void)shift;
    (void)data;
    printf(
        "[HRDWRingBuffer] publishSlot unsupported on this GPU (requires sm_90+)\n");
#if defined(__HIPCC__)
    abort();
#else
    __trap();
#endif
#else
    uint64_t idx = slot & mask;

    EntryT packed{};
    packed.timestamp =
        static_cast<uint32_t>(readGlobaltimer() >> US_TICK_TIMESTAMP_SHIFT);
    packed.epoch = static_cast<uint32_t>(slot >> shift) + 1u;
    packed.data = data;

    // Single 128b atomic exchange via raw PTX. We use atom.exch.relaxed
    // (not st.relaxed) for ALL DataT — even trivially-destructible types
    // — because the RMW form is the only one hardware guarantees as
    // atomic across the GPU↔host boundary on systems without negotiated
    // PCIe AtomicOp support: st.b128 can decompose into two 64b
    // transactions, and the host's 16B atomic load would observe a torn
    // write whose halves landed in a stale order. (Going through
    // libcu++'s cuda::atomic_ref<T> for 16B T also miscompiled on
    // sm_100a in practice, so we emit the instruction directly.)
    //
    // For non-trivially-destructible DataT we run ~DataT() on the
    // displaced occupant; for trivially-destructible DataT the compiler
    // DCEs the prev_lo/prev_hi outputs.
    uint64_t packed_lo, packed_hi;
    __builtin_memcpy(&packed_lo, &packed, sizeof(packed_lo));
    __builtin_memcpy(
        &packed_hi,
        reinterpret_cast<const char*>(&packed) + sizeof(packed_lo),
        sizeof(packed_hi));

    // The two asm blocks are identical except for the PTX scope qualifier
    // on the atom.exch:
    //   .sys — coherent with all observers, including the host CPU and
    //          peer GPUs. Required for System scope because the host
    //          polls the pinned mapped ring concurrently with the GPU
    //          writer; the store must be flushed past L2 with a
    //          system-level fence so the host sees it.
    //   .gpu — coherent only within this GPU's L2 domain. Sufficient
    //          for Device scope because the only host read happens after
    //          cudaStreamSynchronize, which inserts its own full barrier.
    //          Cheaper per write (no cross-device fence). Would be a
    //          correctness bug for System.
    [[maybe_unused]] uint64_t prev_lo, prev_hi;
    if constexpr (C == MemoryCoherenceScope::System) {
      asm volatile(
          "{ .reg .b128 _src, _dst;\n\t"
          "  mov.b128 _src, {%2, %3};\n\t"
          "  atom.exch.relaxed.sys.b128 _dst, [%4], _src;\n\t"
          "  mov.b128 {%0, %1}, _dst; }"
          : "=l"(prev_lo), "=l"(prev_hi)
          : "l"(packed_lo), "l"(packed_hi), "l"(&ring[idx])
          : "memory");
    } else {
      asm volatile(
          "{ .reg .b128 _src, _dst;\n\t"
          "  mov.b128 _src, {%2, %3};\n\t"
          "  atom.exch.relaxed.gpu.b128 _dst, [%4], _src;\n\t"
          "  mov.b128 {%0, %1}, _dst; }"
          : "=l"(prev_lo), "=l"(prev_hi)
          : "l"(packed_lo), "l"(packed_hi), "l"(&ring[idx])
          : "memory");
    }

    if constexpr (!std::is_trivially_destructible_v<DataT>) {
      // Hold the displaced bytes in a raw char buffer (not an EntryT) so
      // the compiler never auto-destructs them at scope exit — that would
      // double-call ~DataT() after our explicit destruction below.
      alignas(EntryT) char prev_buf[sizeof(EntryT)];
      __builtin_memcpy(prev_buf, &prev_lo, sizeof(prev_lo));
      __builtin_memcpy(prev_buf + sizeof(prev_lo), &prev_hi, sizeof(prev_hi));
      auto* prev = reinterpret_cast<EntryT*>(prev_buf);
      if (prev->epoch != HRDW_RINGBUFFER_SLOT_EMPTY) {
        prev->data.~DataT();
      }
    }
#endif
  }
#endif
};

// Lightweight, trivially-copyable handle for writing into an HRDWRingBuffer
// from device code. Pass by value to GPU kernels. See
// HRDWRingBufferDeviceHandle.cuh for full documentation and usage examples.
template <
    typename DataT,
    MemoryCoherenceScope C = MemoryCoherenceScope::System,
    WritePolicy W = WritePolicy::Overwrite>
struct HRDWRingBufferDeviceHandle {
  HRDWEntry<DataT, C>* ring{};
  uint64_t* writeIndex{};
  uint32_t mask{};
  uint32_t shift{};
  [[no_unique_address]] detail::
      define_if<W == WritePolicy::Blocking, const uint64_t*> readIndex{};
  [[no_unique_address]] detail::
      define_if<W == WritePolicy::Blocking, const uint32_t*> abort{};
  [[no_unique_address]] detail::define_if<W == WritePolicy::Blocking, uint32_t>
      spinBackoffNanos{};

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ __forceinline__ void write(DataT data) {
    if constexpr (W == WritePolicy::Blocking) {
      HrdwRingBufferWriter<DataT, C>::write_blocking(
          ring,
          writeIndex,
          mask,
          shift,
          data,
          readIndex,
          abort,
          spinBackoffNanos);
    } else {
      HrdwRingBufferWriter<DataT, C>::write(
          ring, writeIndex, mask, shift, data);
    }
  }
#endif
};

namespace detail {
using OverwriteHandleProbe = HRDWRingBufferDeviceHandle<
    uint64_t,
    MemoryCoherenceScope::System,
    WritePolicy::Overwrite>;
using BlockingHandleProbe = HRDWRingBufferDeviceHandle<
    uint64_t,
    MemoryCoherenceScope::System,
    WritePolicy::Blocking>;
static_assert(
    std::is_trivially_copyable_v<OverwriteHandleProbe> &&
        std::is_trivially_copyable_v<BlockingHandleProbe>,
    "device handle must stay trivially copyable - it is passed by value to kernels");
static_assert(
    sizeof(OverwriteHandleProbe) == 24,
    "Overwrite device handle regressed: expected 2 pointers + 2 uint32s = 24 bytes");
static_assert(
    sizeof(BlockingHandleProbe) == 48,
    "Blocking device handle regressed: expected Overwrite + readIndex + abort + spin = 48 bytes");
} // namespace detail

// Templated kernel launch for host-side write(). The template definition
// is available in .cu compilation units; .cc files link against explicit
// instantiations provided by consumers (see
// comms/utils/colltrace/HRDWRingBufferInstantiations.cu for examples).
#if defined(__CUDACC__) || defined(__HIPCC__)
namespace detail {
template <typename DataT, MemoryCoherenceScope C = MemoryCoherenceScope::System>
__global__ void ringBufferWriteKernel(
    HRDWEntry<DataT, C>* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    uint32_t shift,
    DataT data) {
  HrdwRingBufferWriter<DataT, C>::write(ring, writeIdx, mask, shift, data);
}
} // namespace detail

template <typename DataT, MemoryCoherenceScope C = MemoryCoherenceScope::System>
cudaError_t launchRingBufferWrite(
    cudaStream_t stream,
    HRDWEntry<DataT, C>* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    uint32_t shift,
    DataT data) {
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  detail::ringBufferWriteKernel<DataT, C>
      <<<1, 1, 0, stream>>>(ring, writeIdx, mask, shift, data);
  return cudaGetLastError();
}
#else
// Declaration only — .cc files link against explicit instantiations.
template <typename DataT, MemoryCoherenceScope C = MemoryCoherenceScope::System>
cudaError_t launchRingBufferWrite(
    cudaStream_t stream,
    HRDWEntry<DataT, C>* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    uint32_t shift,
    DataT data);
#endif

namespace detail {

// Per-scope storage policy. Encapsulates allocation, deallocation, and
// teardown so HRDWRingBuffer's lifecycle helpers stay scope-agnostic.
//
//   System: pinned mapped host memory (cudaHostAlloc). Slots are
//           initialized to HRDW_RINGBUFFER_SLOT_EMPTY so the reader's
//           epoch check classifies them as kNotReady until the writer
//           publishes. ~DataT() is invoked on live entries at teardown
//           because the System writer reclaims displaced bytes on
//           overwrite — only the first `min(written, size)` slots are
//           still live.
//
//   Device: device-global memory (cudaMalloc). Slots are zeroed at
//           allocation so every slot's epoch starts at
//           HRDW_RINGBUFFER_SLOT_EMPTY, mirroring the System path —
//           drain bounds reads to `head - tail` so unwritten slots
//           are normally never observed, but the per-slot epoch check
//           still validates against stale-generation slots from
//           lapping writers. destructLiveEntries is a no-op because
//           the host can't dereference device pointers to run ~DataT()
//           on occupants — non-trivial DataT will silently leak its
//           occupant-held resources on teardown; callers needing
//           per-occupant cleanup must drain and destruct on the host
//           themselves.
template <typename DataT, MemoryCoherenceScope C>
struct HrdwRingBufferStorage;

template <typename DataT>
struct HrdwRingBufferStorage<DataT, MemoryCoherenceScope::System> {
  using EntryT = HRDWEntry<DataT, MemoryCoherenceScope::System>;

  static EntryT* allocateRing(uint32_t size) {
    void* p = nullptr;
    auto err = cudaHostAlloc(&p, sizeof(EntryT) * size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
      fprintf(
          stderr,
          "HRDWRingBuffer: Failed to allocate ring buffer: %s\n",
          cudaGetErrorString(err));
      return nullptr;
    }
    auto* ring = static_cast<EntryT*>(p);
    for (uint32_t i = 0; i < size; ++i) {
      ring[i].epoch = HRDW_RINGBUFFER_SLOT_EMPTY;
    }
    return ring;
  }

  template <typename T>
  static T* allocateDeviceScalar() {
    void* p = nullptr;
    auto err = cudaHostAlloc(&p, sizeof(T), cudaHostAllocDefault);
    if (err != cudaSuccess) {
      fprintf(
          stderr,
          "HRDWRingBuffer: Failed to allocate %zu-byte device-visible scalar: %s\n",
          sizeof(T),
          cudaGetErrorString(err));
      return nullptr;
    }
    auto* v = static_cast<T*>(p);
    *v = 0;
    return v;
  }

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  static void freeRing(EntryT* ring) {
    (void)cudaFreeHost(ring);
  }
  template <typename T>
  static void freeDeviceScalar(T* p) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    (void)cudaFreeHost(p);
  }

  static void
  destructLiveEntries(EntryT* ring, const uint64_t* writeIndex, uint32_t size) {
    if constexpr (!std::is_trivially_destructible_v<DataT>) {
      if (ring && writeIndex) {
        uint64_t written = *writeIndex;
        uint64_t count = std::min(written, static_cast<uint64_t>(size));
        for (uint64_t i = 0; i < count; ++i) {
          ring[i].data.~DataT();
        }
      }
    }
  }
};

template <typename DataT>
struct HrdwRingBufferStorage<DataT, MemoryCoherenceScope::Device> {
  using EntryT = HRDWEntry<DataT, MemoryCoherenceScope::Device>;

  static EntryT* allocateRing(uint32_t size) {
    void* p = nullptr;
    auto err = cudaMalloc(&p, sizeof(EntryT) * size);
    if (err != cudaSuccess) {
      fprintf(
          stderr,
          "HRDWRingBuffer: Failed to allocate ring buffer: %s\n",
          cudaGetErrorString(err));
      return nullptr;
    }
    auto* ring = static_cast<EntryT*>(p);
    // Zero the ring so every slot's epoch starts at HRDW_RINGBUFFER_SLOT_EMPTY.
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    auto memsetErr = cudaMemset(ring, 0, sizeof(EntryT) * size);
    if (memsetErr != cudaSuccess) {
      fprintf(
          stderr,
          "HRDWRingBuffer: Failed to zero ring buffer: %s\n",
          cudaGetErrorString(memsetErr));
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      (void)cudaFree(ring);
      return nullptr;
    }
    return ring;
  }

  // Device scope backs each device-visible scalar with device global memory.
  template <typename T>
  static T* allocateDeviceScalar() {
    void* p = nullptr;
    auto err = cudaMalloc(&p, sizeof(T));
    if (err != cudaSuccess) {
      fprintf(
          stderr,
          "HRDWRingBuffer: Failed to allocate %zu-byte device-visible scalar: %s\n",
          sizeof(T),
          cudaGetErrorString(err));
      return nullptr;
    }
    auto* v = static_cast<T*>(p);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    auto memsetErr = cudaMemset(v, 0, sizeof(T));
    if (memsetErr != cudaSuccess) {
      fprintf(
          stderr,
          "HRDWRingBuffer: Failed to zero device-visible scalar: %s\n",
          cudaGetErrorString(memsetErr));
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      (void)cudaFree(v);
      return nullptr;
    }
    return v;
  }

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  static void freeRing(EntryT* ring) {
    (void)cudaFree(ring);
  }
  template <typename T>
  static void freeDeviceScalar(T* p) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    (void)cudaFree(p);
  }

  // Teardown is a no-op: the ring lives in cudaMalloc'd device memory,
  // and the host can't dereference device pointers to invoke ~DataT()
  // on individual occupants. Callers passing a non-trivially-destructible
  // DataT are on the hook for their own cleanup — whatever resources
  // those occupants hold will not be released by ring destruction.
  static void destructLiveEntries(EntryT*, const uint64_t*, uint32_t) {}
};

} // namespace detail

// Host-Read Device-Write (HRDW) ring buffer. Device kernels atomically claim
// slots via writeIndex, then write entries whose format and atomicity depend
// on the MemoryCoherenceScope:
//
//   System: 128b atomic exchange on pinned mapped memory. The host reader
//           polls concurrently via mapped pointers and validates each entry
//           via a per-entry epoch field. DataT must be ≤ 8 bytes. For
//           non-trivially-destructible DataT, the writer runs ~DataT() on
//           the displaced occupant when a slot is overwritten.
//
//   Device: 128b atomic exchange (atom.exch.relaxed.gpu.b128, device
//           scope) on cudaMalloc'd device memory. The host reads via
//           cudaMemcpy after cudaStreamSynchronize (drain). DataT must
//           be ≤ 8 bytes (entry packs into 16 bytes for atomic 128b
//           stores) and is best kept trivially destructible — the host
//           can't run ~DataT() on device-resident occupants, so
//           non-trivial DataT will leak its per-occupant resources on
//           ring destruction. Per-slot epoch lets the reader discard
//           stale-generation slots whose atomic stores completed out
//           of claim order. Requires sm_90+ for atom.exch.b128.
//
// Move-only — CUDA resources are not copyable.
template <
    typename DataT,
    MemoryCoherenceScope C = MemoryCoherenceScope::System,
    WritePolicy W = WritePolicy::Overwrite>
class HRDWRingBuffer {
  static_assert(
      W == WritePolicy::Overwrite || C == MemoryCoherenceScope::System,
      "WritePolicy::Blocking requires MemoryCoherenceScope::System (the reader "
      "publishes the consumer cursor into host-visible memory)");
  using Storage = detail::HrdwRingBufferStorage<DataT, C>;
  static constexpr bool isSystem = (C == MemoryCoherenceScope::System);

 public:
  using Entry = HRDWEntry<DataT, C>;
  explicit HRDWRingBuffer(uint32_t size) {
    // Round up to next power of 2 if needed.
    if (size == 0) {
      size = 1;
    }
    if (size > (1u << 31)) {
      fprintf(
          stderr,
          "HRDWRingBuffer: size %u exceeds maximum power-of-2 size\n",
          size);
      return; // valid() will return false
    }
    if ((size & (size - 1)) != 0) {
#if __cplusplus >= 202002L
      uint32_t nextPowerOf2 = 1U << (std::bit_width(size - 1));
#else
      // C++17 fallback for std::bit_width
      uint32_t nextPowerOf2 = 1;
      while (nextPowerOf2 < size) {
        nextPowerOf2 <<= 1;
      }
#endif
      fprintf(
          stderr,
          "HRDWRingBuffer: size %u is not a power of 2, rounding up to %u\n",
          size,
          nextPowerOf2);
      size = nextPowerOf2;
    }

    // x & mask == x % size
    // but is just a single bitwise op
    size_ = size;
    mask_ = size - 1;
#if __cplusplus >= 202002L
    shift_ = static_cast<uint32_t>(std::countr_zero(size));
#else
    shift_ = 0;
    for (uint32_t s = size; s > 1; s >>= 1) {
      ++shift_;
    }
#endif

    ring_ = Storage::allocateRing(size_);
    if (!ring_) {
      return;
    }
    writeIndex_ = Storage::template allocateDeviceScalar<uint64_t>();
    if (!writeIndex_) {
      return; // already-failed ring; skip the Blocking allocs (avoids noise)
    }
    if constexpr (W == WritePolicy::Blocking) {
      readIndex_ = Storage::template allocateDeviceScalar<uint64_t>();
      abortFlag_ = Storage::template allocateDeviceScalar<uint32_t>();
    }
  }

  template <
      WritePolicy WP = W,
      std::enable_if_t<WP == WritePolicy::Blocking, int> = 0>
  HRDWRingBuffer(uint32_t size, uint32_t spinBackoffNanos)
      : HRDWRingBuffer(size) {
    spinBackoffNanos_ = spinBackoffNanos;
  }

  ~HRDWRingBuffer() {
    Storage::destructLiveEntries(ring_, writeIndex_, size_);
    if (ring_) {
      Storage::freeRing(ring_);
    }
    if (writeIndex_) {
      Storage::freeDeviceScalar(writeIndex_);
    }
    if constexpr (W == WritePolicy::Blocking) {
      if (readIndex_) {
        Storage::freeDeviceScalar(readIndex_);
      }
      if (abortFlag_) {
        Storage::freeDeviceScalar(abortFlag_);
      }
    }
  }

  // Move-only.
  HRDWRingBuffer(const HRDWRingBuffer&) = delete;
  HRDWRingBuffer& operator=(const HRDWRingBuffer&) = delete;

  HRDWRingBuffer(HRDWRingBuffer&& other) noexcept
      : ring_(std::exchange(other.ring_, nullptr)),
        writeIndex_(std::exchange(other.writeIndex_, nullptr)),
        size_(other.size_),
        mask_(other.mask_),
        shift_(other.shift_) {
    if constexpr (W == WritePolicy::Blocking) {
      readIndex_ = std::exchange(other.readIndex_, nullptr);
      abortFlag_ = std::exchange(other.abortFlag_, nullptr);
      spinBackoffNanos_ = other.spinBackoffNanos_;
    }
  }

  HRDWRingBuffer& operator=(HRDWRingBuffer&& other) noexcept {
    if (this != &other) {
      Storage::destructLiveEntries(ring_, writeIndex_, size_);
      if (ring_) {
        Storage::freeRing(ring_);
      }
      if (writeIndex_) {
        Storage::freeDeviceScalar(writeIndex_);
      }
      if constexpr (W == WritePolicy::Blocking) {
        if (readIndex_) {
          Storage::freeDeviceScalar(readIndex_);
        }
        if (abortFlag_) {
          Storage::freeDeviceScalar(abortFlag_);
        }
      }
      ring_ = std::exchange(other.ring_, nullptr);
      writeIndex_ = std::exchange(other.writeIndex_, nullptr);
      size_ = other.size_;
      mask_ = other.mask_;
      shift_ = other.shift_;
      if constexpr (W == WritePolicy::Blocking) {
        readIndex_ = std::exchange(other.readIndex_, nullptr);
        abortFlag_ = std::exchange(other.abortFlag_, nullptr);
        spinBackoffNanos_ = other.spinBackoffNanos_;
      }
    }
    return *this;
  }

  uint32_t size() const {
    return size_;
  }

  // Returns true if all allocations succeeded.
  bool valid() const {
    if (ring_ == nullptr || writeIndex_ == nullptr) {
      return false;
    }
    if constexpr (W == WritePolicy::Blocking) {
      return readIndex_ != nullptr && abortFlag_ != nullptr;
    }
    return true;
  }

  // Release every writer currently blocked in write_blocking() — call on
  // teardown/abort, when the consumer will no longer drain, so a backpressured
  // kernel can't hang the GPU forever. One-way: once aborted the ring stays
  // aborted (writers publish without waiting). Blocking rings only.
  void requestAbort() {
    static_assert(
        W == WritePolicy::Blocking,
        "requestAbort() is only meaningful for a WritePolicy::Blocking ring");
#if defined(__cpp_lib_atomic_ref)
    std::atomic_ref<uint32_t>(*abortFlag_).store(1, std::memory_order_release);
#else
    __atomic_store_n(abortFlag_, uint32_t{1}, __ATOMIC_RELEASE);
#endif
  }

  // Launch a single-thread kernel that atomically claims a slot and
  // writes an entry.
  cudaError_t write(cudaStream_t stream, DataT data) {
    assert(valid());
    return launchRingBufferWrite<DataT, C>(
        stream, ring_, writeIndex_, mask_, shift_, data);
  }

  // Return a lightweight, trivially-copyable handle that can be passed by
  // value to GPU kernels for inline device-side writes. See
  // HRDWRingBufferDeviceHandle.cuh for usage.
  HRDWRingBufferDeviceHandle<DataT, C, W> deviceHandle() const {
    assert(valid());
    if constexpr (W == WritePolicy::Blocking) {
      return {
          ring_,
          writeIndex_,
          mask_,
          shift_,
          readIndex_,
          abortFlag_,
          spinBackoffNanos_};
    } else {
      // Blocking-only handle fields are absent (zero-size) here; omit them.
      return {ring_, writeIndex_, mask_, shift_};
    }
  }

 private:
  template <typename, MemoryCoherenceScope, WritePolicy>
  friend class HRDWRingBufferReader;
  template <typename, MemoryCoherenceScope, WritePolicy>
  friend class HRDWRingBufferReaderBase;
  template <typename, MemoryCoherenceScope>
  friend class HRDWRingBufferTestAccessor;

  Entry* ring_{nullptr};
  uint64_t* writeIndex_{nullptr};
  uint32_t size_{0};
  // size_ - 1. Cached so `slot & mask_` is one bitwise op per write
  // instead of `slot % size_`.
  uint32_t mask_{0};
  // log2(size_). Cached so the per-slot epoch is computed as
  // `(slot >> shift_) + 1` — a shift instead of a divide.
  uint32_t shift_{0};
  // Blocking-only members, grouped last so the [[no_unique_address]] fields
  // that vanish on Overwrite rings don't split the always-present layout.
  //
  // Consumer cursor for WritePolicy::Blocking backpressure (host-published by
  // the reader, acquire-loaded by write_blocking).
  [[no_unique_address]] detail::define_if<W == WritePolicy::Blocking, uint64_t*>
      readIndex_{nullptr};
  // Device-visible abort flag; requestAbort() raises it to release blocked
  // writers on teardown. A 0/1 latch, so a uint32 (vs the uint64 cursors).
  [[no_unique_address]] detail::define_if<W == WritePolicy::Blocking, uint32_t*>
      abortFlag_{nullptr};
  // Per-iteration __nanosleep() backoff for the write_blocking() spin.
  // Configurable via the ctor; default 64 ns.
  [[no_unique_address]] detail::define_if<W == WritePolicy::Blocking, uint32_t>
      spinBackoffNanos_{64};
};

} // namespace hrdw_ring_buffer
