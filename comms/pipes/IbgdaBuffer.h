// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include <endian.h>

#include "comms/pipes/IbgdaNicConfig.h"

// Allow compilation in both host (C++) and device (CUDA) contexts
#ifdef __CUDACC__
#define IBGDA_HOST_DEVICE __host__ __device__
#else
#define IBGDA_HOST_DEVICE
#endif

namespace comms::pipes {

// =============================================================================
// Strong Types for RDMA Memory Keys
// =============================================================================
//
// DOCA GPUNetIO expects memory registration keys (lkey/rkey) in network byte
// order (big-endian) for RDMA WQE construction. These strong types prevent
// accidental mixing of host and network byte order keys.
//
// Use case:
//   - ibv_reg_mr() returns keys in host byte order
//   - DOCA device APIs expect keys in network byte order
//   - These types ensure correct conversion at compile time

/**
 * HostLKey - Local key in host byte order
 *
 * Represents an lkey as returned by ibv_reg_mr() or similar APIs.
 * Must be converted to NetworkLKey before use in RDMA operations.
 */
struct HostLKey {
  uint32_t value{0};

  HostLKey() = default;
  IBGDA_HOST_DEVICE explicit HostLKey(uint32_t v) : value(v) {}

  IBGDA_HOST_DEVICE bool operator==(const HostLKey& other) const {
    return value == other.value;
  }
  IBGDA_HOST_DEVICE bool operator!=(const HostLKey& other) const {
    return value != other.value;
  }
};

/**
 * NetworkLKey - Local key in network byte order (big-endian)
 *
 * Represents an lkey ready for use in RDMA WQE construction.
 * This is the format expected by DOCA GPUNetIO device APIs.
 *
 * Supports implicit conversion from HostLKey (performs byte order conversion).
 */
struct NetworkLKey {
  uint32_t value{0};

  NetworkLKey() = default;
  IBGDA_HOST_DEVICE explicit NetworkLKey(uint32_t v) : value(v) {}

  // Implicit conversion from HostLKey (performs htobe32)
  // NOLINTNEXTLINE(google-explicit-constructor)
  /* implicit */ NetworkLKey(HostLKey hostKey)
      : value(htobe32(hostKey.value)) {}

  IBGDA_HOST_DEVICE bool operator==(const NetworkLKey& other) const {
    return value == other.value;
  }
  IBGDA_HOST_DEVICE bool operator!=(const NetworkLKey& other) const {
    return value != other.value;
  }
};

/**
 * HostRKey - Remote key in host byte order
 *
 * Represents an rkey as received from a remote peer (after network transport).
 * Must be converted to NetworkRKey before use in RDMA operations.
 */
struct HostRKey {
  uint32_t value{0};

  HostRKey() = default;
  IBGDA_HOST_DEVICE explicit HostRKey(uint32_t v) : value(v) {}

  IBGDA_HOST_DEVICE bool operator==(const HostRKey& other) const {
    return value == other.value;
  }
  IBGDA_HOST_DEVICE bool operator!=(const HostRKey& other) const {
    return value != other.value;
  }
};

/**
 * NetworkRKey - Remote key in network byte order (big-endian)
 *
 * Represents an rkey ready for use in RDMA WQE construction.
 * This is the format expected by DOCA GPUNetIO device APIs.
 *
 * Supports implicit conversion from HostRKey (performs byte order conversion).
 */
struct NetworkRKey {
  uint32_t value{0};

  NetworkRKey() = default;
  IBGDA_HOST_DEVICE explicit NetworkRKey(uint32_t v) : value(v) {}

  // Implicit conversion from HostRKey (performs htobe32)
  // Note: This constructor is intentionally NOT explicit - implicit conversion
  // is the desired behavior. It is also intentionally host-only (no
  // IBGDA_HOST_DEVICE) because htobe32() is not available on GPU. The
  // conversion happens on the host before passing to device code.
  // NOLINTNEXTLINE(google-explicit-constructor)
  /* implicit */ NetworkRKey(HostRKey hostKey)
      : value(htobe32(hostKey.value)) {}

  IBGDA_HOST_DEVICE bool operator==(const NetworkRKey& other) const {
    return value == other.value;
  }
  IBGDA_HOST_DEVICE bool operator!=(const NetworkRKey& other) const {
    return value != other.value;
  }
};

// =============================================================================
// Buffer Descriptors
// =============================================================================

/**
 * IbgdaLocalBuffer - Local buffer descriptor for RDMA operations
 *
 * Represents a buffer in the local GPU's memory that can be used
 * as a source for RDMA writes or destination for RDMA reads.
 * Carries one local key per NIC (lkeys[kMaxIbgdaNics]) for multi-NIC
 * IBGDA support — each NIC has its own ibv_pd with a distinct lkey for
 * the same physical buffer. The kernel-side P2pIbgdaTransportDevice
 * selects lkeys[transportIdx_] based on the rail it dispatches to.
 *
 * Backward compatibility:
 *   - lkey aliases lkeys[0] via anonymous union — existing call sites
 *     using `.lkey.value` continue to work unchanged.
 *   - The single-key constructor is [[deprecated]] (sets lkeys[0]=key,
 *     lkeys[1..N-1]=zero) — safe at numNics=1 (transportIdx_ always 0)
 *     but unsafe at numNics>1. Migrate to multi-key constructor.
 *
 * This struct is usable from both host and device code.
 */
struct IbgdaLocalBuffer {
  void* ptr{nullptr};
  union {
    NetworkLKey lkey; // legacy alias for lkeys[0] (zero-init via lkeys{})
    NetworkLKey lkeys[kMaxIbgdaNics];
  };

  IBGDA_HOST_DEVICE IbgdaLocalBuffer() : ptr(nullptr), lkeys{} {}

  // Single-key constructor — DEPRECATED. Sets lkeys[0]=key; lkeys[1..N-1]=0.
  // Safe only when consumed by a P2p with transportIdx_=0 (single-NIC). For
  // multi-NIC transports, use the multi-key constructor below.
  [[deprecated("Use multi-key constructor (ptr, keys) for multi-NIC support")]]
  IBGDA_HOST_DEVICE IbgdaLocalBuffer(void* p, NetworkLKey key)
      : ptr(p), lkeys{key} {}

  // Multi-key constructor — preferred. Caller passes the per-NIC lkeys
  // array by reference (type-checked, fixed size = kMaxIbgdaNics).
  // Single-NIC callers may zero-initialize the unused slots (NetworkLKey{}):
  //   NetworkLKey keys[kMaxIbgdaNics]{lkey};  // {lkey, NetworkLKey{}}
  //   IbgdaLocalBuffer buf(ptr, keys);
  // since lkeys[1..N-1] are never read when numNics=1 (transportIdx_=0).
  IBGDA_HOST_DEVICE IbgdaLocalBuffer(
      void* p,
      const NetworkLKey (&keys)[kMaxIbgdaNics])
      : ptr(p) {
    for (int n = 0; n < kMaxIbgdaNics; n++) {
      lkeys[n] = keys[n];
    }
  }

  /**
   * Create a sub-buffer at the given byte offset.
   * Propagates all NICs' lkeys via the array-ref constructor.
   */
  IBGDA_HOST_DEVICE IbgdaLocalBuffer subBuffer(std::size_t offset) const {
    return IbgdaLocalBuffer(static_cast<char*>(ptr) + offset, lkeys);
  }
};

/**
 * IbgdaRemoteBuffer - Remote buffer descriptor for RDMA operations
 *
 * Represents a buffer in a remote GPU's memory that can be accessed
 * via RDMA operations. Carries one remote key per NIC
 * (rkeys[kMaxIbgdaNics]) for multi-NIC IBGDA support — each NIC has
 * its own ibv_pd with a distinct rkey for the same physical remote
 * buffer. The kernel-side P2pIbgdaTransportDevice selects
 * rkeys[transportIdx_] based on the rail it dispatches to.
 *
 * Backward compatibility: same pattern as IbgdaLocalBuffer (rkey aliases
 * rkeys[0] via union; single-key constructor is [[deprecated]]).
 *
 * This struct is usable from both host and device code.
 */
struct IbgdaRemoteBuffer {
  void* ptr{nullptr};
  union {
    NetworkRKey rkey; // legacy alias for rkeys[0]
    NetworkRKey rkeys[kMaxIbgdaNics];
  };

  IBGDA_HOST_DEVICE IbgdaRemoteBuffer() : ptr(nullptr), rkeys{} {}

  // Single-key constructor — DEPRECATED. Sets rkeys[0]=key; rkeys[1..N-1]=0.
  [[deprecated("Use multi-key constructor (ptr, keys) for multi-NIC support")]]
  IBGDA_HOST_DEVICE IbgdaRemoteBuffer(void* p, NetworkRKey key)
      : ptr(p), rkeys{key} {}

  // Multi-key constructor — preferred. Caller passes the per-NIC rkeys
  // array by reference (type-checked, fixed size = kMaxIbgdaNics).
  IBGDA_HOST_DEVICE IbgdaRemoteBuffer(
      void* p,
      const NetworkRKey (&keys)[kMaxIbgdaNics])
      : ptr(p) {
    for (int n = 0; n < kMaxIbgdaNics; n++) {
      rkeys[n] = keys[n];
    }
  }

  /**
   * Create a sub-buffer at the given byte offset.
   * Propagates all NICs' rkeys via the array-ref constructor.
   */
  IBGDA_HOST_DEVICE IbgdaRemoteBuffer subBuffer(std::size_t offset) const {
    return IbgdaRemoteBuffer(static_cast<char*>(ptr) + offset, rkeys);
  }
};

// =============================================================================
// Signal Operation Types
// =============================================================================

/**
 * IbgdaSignalOp - Signal operation types for IBGDA transport
 *
 * Defines the atomic operation to perform on the remote signal buffer.
 * Note: SET is not yet supported by DOCA GPUNetIO, but included for
 * API consistency and future compatibility with torchcomms.
 */
enum class IbgdaSignalOp {
  ADD, // Atomic fetch-add (supported)
  SET, // Atomic set (not yet supported by DOCA)
};

/**
 * IbgdaCmpOp - Comparison operations for wait_signal
 *
 * Defines the comparison operation used when waiting for a signal value.
 * API consistent with torchcomms::device::CmpOp.
 */
enum class IbgdaCmpOp {
  EQ, // ==
  NE, // !=
  LT, // <
  LE, // <=
  GT, // >
  GE, // >= (most common for wait operations)
};

// =============================================================================
// Buffer Exchange Info
// =============================================================================

/**
 * IbgdaBufferExchInfo - Buffer info for exchange between hosts
 *
 * Represents a buffer's address and remote key in host byte order,
 * suitable for serialization and exchange between peers. The rkey
 * is stored in host byte order and will be converted to network
 * byte order when creating an IbgdaRemoteBuffer for RDMA operations.
 *
 * Use case:
 *   - Exchange buffer registration info between peers via bootstrap
 *   - Convert to IbgdaRemoteBuffer after receiving from peer
 */
struct IbgdaBufferExchInfo {
  uint64_t addr{0};
  HostRKey rkey{};

  IbgdaBufferExchInfo() = default;
  IbgdaBufferExchInfo(uint64_t a, HostRKey r) : addr(a), rkey(r) {}

  /**
   * Convert to IbgdaRemoteBuffer for RDMA operations.
   * The HostRKey is implicitly converted to NetworkRKey.
   *
   * Single-NIC path: populates rkeys[0] with the exchanged rkey;
   * rkeys[1..N-1] left zero (never read at numNics=1 because
   * transportIdx_=0). Multi-NIC code uses IbgdaBufferExchInfoMultiNic
   * (added in Group 1) instead of this struct.
   */
  IbgdaRemoteBuffer toRemoteBuffer() const {
    NetworkRKey keys[kMaxIbgdaNics]{NetworkRKey(rkey)};
    return IbgdaRemoteBuffer(reinterpret_cast<void*>(addr), keys);
  }

  /**
   * Create a sub-buffer at the given byte offset.
   */
  IbgdaBufferExchInfo subBuffer(std::size_t offset) const {
    return IbgdaBufferExchInfo(addr + offset, rkey);
  }
};

} // namespace comms::pipes
