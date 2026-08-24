// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/common/AtomicUtils.cuh"
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/DeviceMacros.cuh"
#include "comms/prims/core/LlxPacket.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"

namespace comms::prims {

// =============================================================================
// LLImpl<P> — low-latency pack / unpack over LlxPacket geometry P
// =============================================================================
//
// Cooperative, ThreadGroup-driven encode/decode of a byte range into/out of the
// packet-formatted staging region:
//
//   pack:   user src   -> staging packets {data, flag=flagVal}
//   unpack: staging packets (flag==flagVal) -> user dst
//
// Parallelism (v1): one thread owns one whole packet, grid-strided across the
// group. `pack` writes each packet's trailing flag last; on the wire the whole
// staging region is fenced and RDMA-put as one unit, so the flag is never
// observed before its data. `unpack` spins only on the packets a thread owns
// (no group-wide readiness barrier), then decodes.
template <typename P>
struct LLImpl {
  using FlagType = typename P::FlagType;

  // Spins between timeout clock reads. A power-of-two minus one so the check
  // is a mask, not a modulo.
  static constexpr uint32_t kTimeoutPollMask = 1023;

  // Replicate the (32-bit) flagVal across a 64-bit flag word, so an 8 B flag
  // can be written/compared as one wide transfer.
  __device__ __forceinline__ static uint64_t replicated(FlagType flagVal) {
    return (static_cast<uint64_t>(flagVal) << 32) |
        static_cast<uint64_t>(flagVal);
  }

  // ---------------------------------------------------------------------------
  // Flag I/O — vectorized volatile access to the packet's trailing flag.
  //
  // The flag occupies the last `P::kFlag` bytes of the packet, entirely inside
  // `P::kFlagLane`'s 16 B slot, so it is read/written as ONE wide volatile
  // transfer (a u32 for the 4 B flag) instead of the old per-word
  // (`P::kFlagWords`) scalar loop. `flagVal` is
  // replicated across the full flag width, so a torn transfer where only part
  // of the flag carries the new flagVal fails `is_flag_set` — unpack never
  // accepts half-arrived data.
  //
  // These are single-lane primitives: only `P::kFlagLane` touches the flag.
  // Sharing that lane's readiness across the packet's `P::kThreadsPerPacket`
  // lanes (the warp reduction) is the ops layer's job and lives in unpack.
  // ---------------------------------------------------------------------------

  /// Volatile-store the trailing flag as one wide transfer, replicating
  /// `flagVal` across the full flag width.
  __device__ __forceinline__ static void store_flag(
      void* pkt,
      FlagType flagVal) {
#ifdef __CUDA_ARCH__
    void* fp = P::flag_ptr(pkt);
    if constexpr (P::kFlag == static_cast<int>(sizeof(uint32_t))) {
      comms::device::st_volatile_global(
          reinterpret_cast<volatile uint32_t*>(fp), flagVal);
    } else if constexpr (P::kFlag == static_cast<int>(sizeof(uint64_t))) {
      comms::device::st_volatile_global(
          reinterpret_cast<volatile uint64_t*>(fp), replicated(flagVal));
    } else {
      auto* p = reinterpret_cast<volatile FlagType*>(fp);
#pragma unroll
      for (int i = 0; i < P::kFlagWords; ++i) {
        comms::device::st_volatile_global(p + i, flagVal);
      }
    }
#else
    (void)pkt;
    (void)flagVal;
#endif
  }

  /// Volatile-load the trailing flag (first word).
  __device__ __forceinline__ static FlagType load_flag(const void* pkt) {
#ifdef __CUDA_ARCH__
    const auto* p =
        reinterpret_cast<const volatile FlagType*>(P::flag_ptr(pkt));
    return comms::device::ld_volatile_global(p);
#else
    (void)pkt;
    return 0;
#endif
  }

  /// True if the whole flag equals `flagVal`, read as one wide volatile load.
  __device__ __forceinline__ static bool is_flag_set(
      const void* pkt,
      FlagType flagVal) {
#ifdef __CUDA_ARCH__
    const void* fp = P::flag_ptr(pkt);
    if constexpr (P::kFlag == static_cast<int>(sizeof(uint32_t))) {
      return comms::device::ld_volatile_global(
                 reinterpret_cast<const volatile uint32_t*>(fp)) == flagVal;
    } else if constexpr (P::kFlag == static_cast<int>(sizeof(uint64_t))) {
      return comms::device::ld_volatile_global(
                 reinterpret_cast<const volatile uint64_t*>(fp)) ==
          replicated(flagVal);
    } else {
      const auto* p = reinterpret_cast<const volatile FlagType*>(fp);
#pragma unroll
      for (int i = 0; i < P::kFlagWords; ++i) {
        if (comms::device::ld_volatile_global(p + i) != flagVal) {
          return false;
        }
      }
      return true;
    }
#else
    (void)pkt;
    (void)flagVal;
    return true;
#endif
  }

  /// True once EVERY packet covering `nbytes` of payload in `staging` carries
  /// `flagVal`. Cooperative across `group`: each thread checks the packets it
  /// owns (the same grid-stride mapping `unpack` uses) with early exit, then
  /// the group AND-reduces so all threads reach the same verdict.
  ///
  /// The non-spinning counterpart to `unpack`'s per-packet wait, for callers
  /// that must not block inside the codec -- the resumable recv path returns
  /// `Waiting` on false rather than spinning.
  ///
  /// Lives here rather than in the transport because deciding which packets
  /// cover `nbytes`, and where each one sits, is packet-format knowledge --
  /// the same knowledge `pack` and `unpack` below encode. A copy of this walk
  /// outside this class is a second place for that layout to drift.
  ///
  /// Takes PAYLOAD bytes, like every other entry point here; callers holding a
  /// wire length must convert with `P::max_payload()`.
  template <typename Group>
  __device__ __forceinline__ static bool all_flags_set(
      Group& group,
      const void* staging,
      std::size_t nbytes,
      FlagType flagVal) {
#ifdef __CUDA_ARCH__
    const std::size_t nPackets = P::packet_count(nbytes);
    const auto* base = reinterpret_cast<const char*>(staging);
    bool myReady = true;
    for (std::size_t i = group.thread_id_in_group; i < nPackets;
         i += group.group_size) {
      if (!is_flag_set(
              base + i * static_cast<std::size_t>(P::kPacketBytes), flagVal)) {
        myReady = false;
        break;
      }
    }
    // group.all() rather than a hand-rolled reduction: it keeps the scratch
    // index tied to the group's own barrier.
    return group.all(myReady);
#else
    (void)group;
    (void)staging;
    (void)nbytes;
    (void)flagVal;
    return true;
#endif
  }

  /// Encode `nbytes` of `src` into consecutive packets in `staging`, stamping
  /// every packet's trailing flag with `flagVal`. Cooperative across `group`.
  template <typename Group>
  __device__ __forceinline__ static void pack(
      Group& group,
      void* staging,
      const void* src,
      std::size_t nbytes,
      FlagType flagVal) {
#ifdef __CUDA_ARCH__
    const std::size_t nPackets = P::packet_count(nbytes);
    auto* base = reinterpret_cast<char*>(staging);
    const auto* s = reinterpret_cast<const char*>(src);
    for (std::size_t i = group.thread_id_in_group; i < nPackets;
         i += group.group_size) {
      char* pkt = base + i * static_cast<std::size_t>(P::kPacketBytes);
      const std::size_t valid = P::valid_payload(i, nbytes);
      const std::size_t off = i * static_cast<std::size_t>(P::kData);
      // Payload bytes, zero-padding the packet's data-region tail.
      for (int b = 0; b < P::kData; ++b) {
        pkt[b] = static_cast<std::size_t>(b) < valid ? s[off + b] : char(0);
      }
      // Flag last (see class note on wire-side ordering).
      store_flag(pkt, flagVal);
    }
    group.sync();
#else
    (void)group;
    (void)staging;
    (void)src;
    (void)nbytes;
    (void)flagVal;
#endif
  }

  /// Wait until every packet covering `nbytes` carries flag == `flagVal`, then
  /// decode its payload into `dst`. Cooperative across `group`; each thread
  /// spins only on the packets it owns.
  // `timeout` bounds the readiness spin. Unlike Simple, LL's readiness lives in
  // the payload, so the wait happens HERE rather than in wait_signal -- without
  // a deadline a lost WQE, a dead peer, or a flagVal desync spins forever and
  // then holds the whole group at the group.sync() below. Each thread polls
  // only the packets it owns, so the check is per-thread
  // (FT_ABORT_BREAK) and not the leader-only group form. The
  // clock is read once per kTimeoutPollMask+1 spins: LL is the latency path,
  // and a clock64() on every poll is a measurable cost on a hot loop.
  //
  // Stays `void` -- the macro terminates the loops itself and every thread
  // still reaches the group.sync() below, which is all the abort contract
  // requires. `dst` is undefined after an abort. This mirrors
  // MemcpyCopyOp::recvLL; keep the two in step when either changes.
  template <typename Group>
  __device__ __forceinline__ static void unpack(
      Group& group,
      void* dst,
      const void* staging,
      std::size_t nbytes,
      FlagType flagVal,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    const std::size_t nPackets = P::packet_count(nbytes);
    const auto* base = reinterpret_cast<const char*>(staging);
    auto* d = reinterpret_cast<char*>(dst);
    for (std::size_t i = group.thread_id_in_group; i < nPackets;
         i += group.group_size) {
      const char* pkt = base + i * static_cast<std::size_t>(P::kPacketBytes);
      const std::size_t valid = P::valid_payload(i, nbytes);
      const std::size_t off = i * static_cast<std::size_t>(P::kData);

      if constexpr (P::kPacketBytes == static_cast<int>(sizeof(uint64_t))) {
        // 8 B packet: {data:4, flag:4} in one 8 B atomic word -- the poll and
        // the data read are one load. Shared with unpack_reduce so there is a
        // single abort-aware spin loop in this file.
        bool abandoned = false;
        const uint32_t data =
            load_ready_payload(pkt, flagVal, timeout, abandoned);
        // Leaves the packet loop as well: load_ready_payload only exits its own
        // spin. `dst` is undefined from here, which the abort contract permits.
        if (abandoned) {
          break;
        }
        const auto* db = reinterpret_cast<const char*>(&data);
#pragma unroll
        for (int b = 0; b < P::kData; ++b) {
          if (static_cast<std::size_t>(b) < valid) {
            d[off + b] = db[b];
          }
        }
      } else {
        // Large packet: poll the trailing flag, then read the payload with
        // vectorized volatile (L1-bypassing) loads.
        uint32_t spins = 0;
        while (!is_flag_set(pkt, flagVal)) {
          // Spin until this packet's flag reaches the current flagVal.
          if ((++spins & kTimeoutPollMask) == 0) {
            FT_ABORT_BREAK(
                timeout,
                "LLImpl::unpack waiting for LL flag %u on packet %llu",
                (unsigned)flagVal,
                (unsigned long long)i);
          }
        }
        if (spins >= kTimeoutPollMask) {
          FT_ABORT_BREAK(
              timeout,
              "LLImpl::unpack abandoning decode at packet %llu",
              (unsigned long long)i);
        }
        constexpr int kDataWords =
            P::kData / static_cast<int>(sizeof(uint64_t));
        const auto* sp = reinterpret_cast<const volatile uint64_t*>(pkt);
#pragma unroll
        for (int w = 0; w < kDataWords; ++w) {
          const uint64_t word = comms::device::ld_volatile_global(sp + w);
          const auto* wb = reinterpret_cast<const char*>(&word);
#pragma unroll
          for (int b = 0; b < static_cast<int>(sizeof(uint64_t)); ++b) {
            const std::size_t gb =
                static_cast<std::size_t>(w) * sizeof(uint64_t) +
                static_cast<std::size_t>(b);
            if (gb < valid) {
              d[off + gb] = wb[b];
            }
          }
        }
      }
    }
    group.sync();
#else
    (void)group;
    (void)dst;
    (void)staging;
    (void)nbytes;
    (void)flagVal;
#endif
  }

  /// Spin until `pkt` carries flag == `flagVal`, then return its data half.
  /// One wide volatile load yields both halves, so the readiness poll and the
  /// payload read are the same instruction -- the fused 8 B path unpack() uses.
  ///
  /// Sets `abandoned` when the spin gave up on an abort rather than seeing the
  /// flag; the return value is then whatever was last on the wire, which the
  /// abort contract permits. Callers MUST propagate it out of their own
  /// packet/element loop: this only leaves its own spin, and re-entering it for
  /// every remaining packet would burn a full poll interval each time before
  /// giving up again.
  __device__ __forceinline__ static uint32_t load_ready_payload(
      const char* pkt,
      FlagType flagVal,
      const Timeout& timeout,
      bool& abandoned) {
#ifdef __CUDA_ARCH__
    const auto* p = reinterpret_cast<const volatile uint64_t*>(pkt);
    uint64_t v;
    uint32_t spins = 0;
    do {
      v = comms::device::ld_volatile_global(p);
      if ((++spins & kTimeoutPollMask) == 0) {
        // Honor behavior(): under the default SKIP a deliberate abort has to
        // unwind, not kill the CUDA context from inside LL decode. CHECK rather
        // than BREAK so the flag can be raised before leaving the spin.
        if (FT_ABORT_CHECK(
                timeout,
                "LLImpl::load_ready_payload waiting for LL flag %u",
                (unsigned)flagVal)) {
          abandoned = true;
          break;
        }
      }
    } while (static_cast<FlagType>(v >> 32) !=
             flagVal); // high half is the flag
    return static_cast<uint32_t>(v); // low half is the payload
#else
    (void)pkt;
    (void)flagVal;
    (void)timeout;
    (void)abandoned;
    return 0;
#endif
  }

  /// Reduce the packet stream into `accum` under `Combine`: the accumulating
  /// counterpart of unpack(), which assigns. A reduce CopyOp cannot call
  /// unpack() -- that would overwrite the partial sum -- and staging into a
  /// scratch buffer first would cost an extra chunk-sized write and read on
  /// what is the latency path.
  ///
  /// Lives here rather than in the CopyOp so the packet walk, the valid-payload
  /// masking and the wide-T reassembly stay in the one place that owns the
  /// packet format, and so the readiness spin is not something each caller has
  /// to remember. A reduce op that re-implemented this walk without the spin
  /// would consume staging before the data landed.
  ///
  /// Two element/packet tilings, both reachable from the fused AllReduce's
  /// instantiation set. When an element fits inside a packet's data region
  /// (float, __half, __nv_bfloat16 against kData = 4) one thread owns one
  /// packet. When an element is wider (double, int64_t) it spans
  /// sizeof(T)/kData consecutive packets, so one thread instead owns one
  /// element and reassembles it before reducing -- a per-packet reduce would
  /// corrupt a value split across two packets.
  ///
  /// NOTE: scalar-granular by construction (one float / two halves per 4 B
  /// packet); the packet layout rules out the 16 B vectorized tile reduce the
  /// contiguous path uses.
  /// `Combine` is a default-constructible functor with
  /// `operator()(T& accum, const T& value)`. Passing the combiner in keeps the
  /// reduce vocabulary out of the codec: LLImpl owns the packet format, the
  /// caller owns what "reduce" means. It also keeps this header host-includable
  /// -- pulling in VecOps would drag device-only math (fmaxf/fminf) into every
  /// host TU that reaches LLImpl through MemcpyCopyOp.cuh.
  template <typename T, typename Combine, typename Group>
  __device__ __forceinline__ static void unpack_reduce(
      Group& group,
      T* accum,
      const void* staging,
      std::size_t nbytes,
      FlagType flagVal,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    constexpr std::size_t kData = static_cast<std::size_t>(P::kData);
    constexpr std::size_t kPacket = static_cast<std::size_t>(P::kPacketBytes);
    const auto* base = reinterpret_cast<const char*>(staging);

    if constexpr (kData % sizeof(T) == 0) {
      constexpr std::size_t kElemsPerPacket = kData / sizeof(T);
      const std::size_t nPackets = P::packet_count(nbytes);
      for (std::size_t i = group.thread_id_in_group; i < nPackets;
           i += group.group_size) {
        bool abandoned = false;
        const uint32_t data =
            load_ready_payload(base + i * kPacket, flagVal, timeout, abandoned);
        // `accum` is undefined from here, which the abort contract permits.
        if (abandoned) {
          break;
        }
        const T* payload = reinterpret_cast<const T*>(&data);
        const std::size_t nElem = P::valid_payload(i, nbytes) / sizeof(T);
        const std::size_t baseElem = i * kElemsPerPacket;
#pragma unroll
        for (std::size_t e = 0; e < kElemsPerPacket; ++e) {
          if (e < nElem) {
            Combine{}(accum[baseElem + e], payload[e]);
          }
        }
      }
    } else {
      // Entered when kData % sizeof(T) != 0, but the reassembly below needs the
      // stronger property: an element must be a whole number of packets.
      // Without it kPacketsPerElem truncates and each element is silently
      // under-filled -- at sizeof(T) == 3 it would be zero packets and `val`
      // would be reduced wholly uninitialized. Every currently instantiated
      // type (double, int64_t against kData == 4) satisfies this.
      static_assert(
          sizeof(T) % kData == 0,
          "LL wide-element reduce needs sizeof(T) to be a whole number of "
          "packet payloads");
      constexpr std::size_t kPacketsPerElem = sizeof(T) / kData;
      // Whole elements only. A chunk carrying a partial element would drop it
      // here (integer division) and leave the next chunk's `accum` misaligned.
      // Chunk sizing guarantees this: calcGeometry/make_progress_geometry align
      // chunkPayload to lcm(kData, 8), a multiple of sizeof(T). This check
      // is the backstop for a caller that bypasses that sizing.
      PIPES_DEVICE_CHECK_MSG(
          nbytes % sizeof(T) == 0, "LL reduce chunk must hold whole elements");
      const std::size_t nElems = nbytes / sizeof(T);
      for (std::size_t e = group.thread_id_in_group; e < nElems;
           e += group.group_size) {
        T val;
        auto* valBytes = reinterpret_cast<char*>(&val);
        bool abandoned = false;
#pragma unroll
        for (std::size_t k = 0; k < kPacketsPerElem; ++k) {
          const uint32_t word = load_ready_payload(
              base + (e * kPacketsPerElem + k) * kPacket,
              flagVal,
              timeout,
              abandoned);
          if (abandoned) {
            break;
          }
          const auto* wordBytes = reinterpret_cast<const char*>(&word);
#pragma unroll
          for (std::size_t b = 0; b < kData; ++b) {
            valBytes[k * kData + b] = wordBytes[b];
          }
        }
        // Skip the combine and leave the element loop: `val` is only partly
        // reassembled, so reducing it would corrupt `accum` with wire garbage
        // rather than merely leaving it undefined.
        if (abandoned) {
          break;
        }
        Combine{}(accum[e], val);
      }
    }
    group.sync();
#else
    (void)group;
    (void)accum;
    (void)staging;
    (void)nbytes;
    (void)flagVal;
    (void)timeout;
#endif
  }
};

} // namespace comms::prims
