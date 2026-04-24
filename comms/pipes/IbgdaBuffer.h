// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <endian.h>

#include "comms/pipes/HipCompat.cuh"
#include "comms/pipes/rdma/NicConstants.h"

// Allow compilation in both host (C++) and device (CUDA/HIP) contexts
#if defined(__CUDACC__) || defined(__HIPCC__)
#define IBGDA_HOST_DEVICE __host__ __device__
#else
#define IBGDA_HOST_DEVICE
#endif

// Bounds-check trap for NetworkLKeys / NetworkRKeys operator[].
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define IBGDA_KEYS_OOB_TRAP(kind, n, sz)              \
  do {                                                \
    printf(                                           \
        "Network" kind                                \
        "Keys: index %d out of range [0, %d) at "     \
        "%s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n", \
        (int)(n),                                     \
        (int)(sz),                                    \
        __FILE__,                                     \
        __LINE__,                                     \
        blockIdx.x,                                   \
        blockIdx.y,                                   \
        blockIdx.z,                                   \
        threadIdx.x,                                  \
        threadIdx.y,                                  \
        threadIdx.z);                                 \
    __trap();                                         \
  } while (0)
#else
#define IBGDA_KEYS_OOB_TRAP(kind, n, sz) assert((n) >= 0 && (n) < (sz))
#endif

namespace comms::pipes {

// NIC-aware byte order conversion for RDMA memory keys.
// mlx5 requires big-endian keys; bnxt/ionic use native byte order.
namespace detail {
inline uint32_t ibgdaNetworkByteOrderKey(uint32_t hostValue) {
#if defined(NIC_BNXT) || defined(NIC_IONIC)
  return hostValue; // Native byte order for non-mlx5 NICs
#else
  return htobe32(hostValue); // mlx5: big-endian
#endif
}
} // namespace detail

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

  // Implicit conversion from HostLKey (performs byte order conversion)
  // NOLINTNEXTLINE(google-explicit-constructor)
  /* implicit */ NetworkLKey(HostLKey hostKey)
      : value(detail::ibgdaNetworkByteOrderKey(hostKey.value)) {}

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

  // Implicit conversion from HostRKey (performs byte order conversion)
  // Note: This constructor is intentionally NOT explicit - implicit conversion
  // is the desired behavior. It is also intentionally host-only (no
  // IBGDA_HOST_DEVICE) because the conversion happens on the host before
  // passing to device code.
  // NOLINTNEXTLINE(google-explicit-constructor)
  /* implicit */ NetworkRKey(HostRKey hostKey)
      : value(detail::ibgdaNetworkByteOrderKey(hostKey.value)) {}

  IBGDA_HOST_DEVICE bool operator==(const NetworkRKey& other) const {
    return value == other.value;
  }
  IBGDA_HOST_DEVICE bool operator!=(const NetworkRKey& other) const {
    return value != other.value;
  }
};

// =============================================================================
// Per-NIC Key Arrays
// =============================================================================
//
// Named wrappers around fixed-size arrays of per-NIC keys with explicit
// `size` (count of populated NIC slots) and bounds-checked indexing.
//
// Construction:
//   NetworkLKeys keys{};               // size=0, no NICs
//   NetworkLKeys keys(numNics);        // pre-size for loop fill via op[]
//
// Indexed access:
//   auto v = keys[nic].value;          // traps if nic >= keys.size

/**
 * NetworkLKeys - Fixed-capacity array of per-NIC local keys with explicit
 * `size` and bounds-checked indexing.
 */
struct NetworkLKeys {
  NetworkLKey values[kMaxNicsPerGpu]{};
  int size{0};

  NetworkLKeys() = default;

  // Pre-size for progressive fill via operator[]:
  //   NetworkLKeys k(numNics);
  //   for (int n = 0; n < numNics; ++n) k[n] = ...;
  IBGDA_HOST_DEVICE explicit NetworkLKeys(int n) : size(n) {
    assert(n >= 0 && n <= kMaxNicsPerGpu);
  }

  // Variadic per-NIC ctor: size = number of keys.
  //   Single-NIC: NetworkLKeys{lkey}            (size=1)
  //   Multi-NIC:  NetworkLKeys{lkey0, lkey1}    (size=2)
  // explicit: NetworkLKey does NOT implicitly convert to NetworkLKeys —
  // every single-NIC site must spell the wrap, making it visible in code
  // review when refactoring for multi-NIC.
  template <typename... Rest>
  IBGDA_HOST_DEVICE explicit NetworkLKeys(NetworkLKey k0, Rest... rest)
      : size(static_cast<int>(1 + sizeof...(Rest))) {
    static_assert(
        1 + sizeof...(Rest) <= kMaxNicsPerGpu,
        "NetworkLKeys: too many keys for kMaxNicsPerGpu");
    const NetworkLKey arr[] = {k0, rest...};
    for (int i = 0; i < size; ++i) {
      values[i] = arr[i];
    }
  }

  IBGDA_HOST_DEVICE NetworkLKey& operator[](int n) {
    if (n < 0 || n >= size) {
      IBGDA_KEYS_OOB_TRAP("L", n, size);
    }
    return values[n];
  }
  IBGDA_HOST_DEVICE const NetworkLKey& operator[](int n) const {
    if (n < 0 || n >= size) {
      IBGDA_KEYS_OOB_TRAP("L", n, size);
    }
    return values[n];
  }
};

/**
 * NetworkRKeys - Mirror of NetworkLKeys for remote keys.
 */
struct NetworkRKeys {
  NetworkRKey values[kMaxNicsPerGpu]{};
  int size{0};

  NetworkRKeys() = default;

  IBGDA_HOST_DEVICE explicit NetworkRKeys(int n) : size(n) {
    assert(n >= 0 && n <= kMaxNicsPerGpu);
  }

  // Variadic per-NIC ctor: size = number of keys.
  //   Single-NIC: NetworkRKeys{rkey}            (size=1)
  //   Multi-NIC:  NetworkRKeys{rkey0, rkey1}    (size=2)
  // explicit — see NetworkLKeys for rationale.
  template <typename... Rest>
  IBGDA_HOST_DEVICE explicit NetworkRKeys(NetworkRKey k0, Rest... rest)
      : size(static_cast<int>(1 + sizeof...(Rest))) {
    static_assert(
        1 + sizeof...(Rest) <= kMaxNicsPerGpu,
        "NetworkRKeys: too many keys for kMaxNicsPerGpu");
    const NetworkRKey arr[] = {k0, rest...};
    for (int i = 0; i < size; ++i) {
      values[i] = arr[i];
    }
  }

  IBGDA_HOST_DEVICE NetworkRKey& operator[](int n) {
    if (n < 0 || n >= size) {
      IBGDA_KEYS_OOB_TRAP("R", n, size);
    }
    return values[n];
  }
  IBGDA_HOST_DEVICE const NetworkRKey& operator[](int n) const {
    if (n < 0 || n >= size) {
      IBGDA_KEYS_OOB_TRAP("R", n, size);
    }
    return values[n];
  }
};

// =============================================================================
// Buffer Descriptors
// =============================================================================

/**
 * IbgdaLocalBuffer - Local buffer descriptor for RDMA operations
 *
 * Represents a buffer in the local GPU's memory that can be used as a
 * source for RDMA writes or destination for RDMA reads. Carries one local
 * key per NIC (`lkey_per_device`) for multi-NIC IBGDA support — each NIC
 * has its own ibv_pd with a distinct lkey for the same physical buffer.
 * The kernel-side P2pIbgdaTransportDevice selects
 * `lkey_per_device[nic]` based on the rail it dispatches to.
 *
 * Indexing `lkey_per_device[n]` traps if `n >= lkey_per_device.size`.
 *
 * This struct is usable from both host and device code.
 */
struct IbgdaLocalBuffer {
  void* ptr{nullptr};
  NetworkLKeys lkey_per_device{};

  IbgdaLocalBuffer() = default;

  IBGDA_HOST_DEVICE IbgdaLocalBuffer(void* p, const NetworkLKeys& keys)
      : ptr(p), lkey_per_device(keys) {}

  /**
   * Create a sub-buffer at the given byte offset.
   * Propagates all NICs' lkeys.
   */
  IBGDA_HOST_DEVICE IbgdaLocalBuffer subBuffer(std::size_t offset) const {
    return IbgdaLocalBuffer(static_cast<char*>(ptr) + offset, lkey_per_device);
  }
};

/**
 * IbgdaRemoteBuffer - Remote buffer descriptor for RDMA operations
 *
 * Mirror of IbgdaLocalBuffer for remote buffers. Carries one remote key
 * per NIC (`rkey_per_device`) for multi-NIC IBGDA support. Indexing
 * `rkey_per_device[n]` traps if `n >= rkey_per_device.size`.
 *
 * This struct is usable from both host and device code.
 */
struct IbgdaRemoteBuffer {
  void* ptr{nullptr};
  NetworkRKeys rkey_per_device{};

  IbgdaRemoteBuffer() = default;

  IBGDA_HOST_DEVICE IbgdaRemoteBuffer(void* p, const NetworkRKeys& keys)
      : ptr(p), rkey_per_device(keys) {}

  /**
   * Create a sub-buffer at the given byte offset.
   * Propagates all NICs' rkeys.
   */
  IBGDA_HOST_DEVICE IbgdaRemoteBuffer subBuffer(std::size_t offset) const {
    return IbgdaRemoteBuffer(static_cast<char*>(ptr) + offset, rkey_per_device);
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
   * Single-NIC: rkey_per_device.size = 1, values[0] = rkey.
   */
  IbgdaRemoteBuffer toRemoteBuffer() const {
    return IbgdaRemoteBuffer(reinterpret_cast<void*>(addr), NetworkRKeys{rkey});
  }

  /**
   * Create a sub-buffer at the given byte offset.
   */
  IbgdaBufferExchInfo subBuffer(std::size_t offset) const {
    return IbgdaBufferExchInfo(addr + offset, rkey);
  }
};

/**
 * IbSendRecvState — device-side state for pipelined RDMA send/recv.
 *
 * Holds all buffer handles and config needed by send/recv.
 * All physical memory is allocated by MultipeerIbgdaTransport on the host;
 * this struct contains only pointers/handles into those allocations.
 *
 * Buffer layout:
 *   sendStaging / recvStaging: pipelineDepth * dataBufferSize bytes each.
 *     Logically divided into pipelineDepth slots of dataBufferSize bytes.
 *     For one send()/recv() call, a caller chooses active_blocks
 *     (1 <= active_blocks <= maxGroups). Each slot is then partitioned into
 *     active_blocks per-block regions:
 *       perBlockSlot = (dataBufferSize / active_blocks) & ~15ULL
 *     If max_signal_bytes is smaller than perBlockSlot, each per-block region
 *     is further subdivided into signaled sub-chunks:
 *       chunkSize = floor16(min(perBlockSlot, max_signal_bytes))
 *       chunksPerSlot = perBlockSlot / chunkSize
 *     stepState counts these sub-chunks, not whole slots.
 *
 *   signalBuf: 2 * maxGroups * sizeof(uint64_t).
 *     [0, maxGroups)             — DATA_READY (sender -> receiver)
 *     [maxGroups, 2*maxGroups)   — SLOT_FREE (receiver -> sender)
 *
 *   counterBuf: maxGroups * sizeof(uint64_t).
 *     [0, maxGroups)             — NIC_DONE counters (loopback atomic)
 *
 *   stepState: 2 * maxGroups * sizeof(int64_t).
 *     [0, maxGroups)             — sender step counters
 *     [maxGroups, 2*maxGroups)   — receiver step counters
 */
struct IbSendRecvState {
  IbgdaLocalBuffer
      sendStagingBuf; ///< Registered sendStaging (lkey for put src)
  IbgdaRemoteBuffer recvStagingBuf; ///< Peer's recvStaging (rkey for put dst)
  char* sendStagingPtr{
      nullptr}; ///< Raw sendStaging pointer (memcpy addressing)
  char* recvStagingPtr{
      nullptr}; ///< Raw local recvStaging pointer (recv memcpy)
  IbgdaLocalBuffer localSignalBuf; ///< Signal inbox (DATA_READY + SLOT_FREE)
  IbgdaRemoteBuffer remoteSignalBuf; ///< Peer's signal inbox
  IbgdaLocalBuffer localCounterBuf; ///< NIC_DONE counter inbox
  int64_t* stepState{nullptr}; ///< Per-group step counters
  int maxGroups{0}; ///< Layout size for signals/step arrays
  int pipelineDepth{0}; ///< Number of pipeline slots in the ring
  std::size_t dataBufferSize{0}; ///< Size of one pipeline slot in bytes
};

} // namespace comms::pipes
