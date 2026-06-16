// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"
#include "comms/prims/transport/self/P2pSelfTransportDevice.cuh"

namespace comms::prims {
// Transport union members must be safe for cudaMemcpy to device.
// Requirements:
// - Standard layout: predictable byte representation, no hidden members
// - Trivially destructible: source destructor after memcpy is a no-op
// See MultiPeerNvlTransport::initializeTransportsArray().
static_assert(
    std::is_standard_layout_v<P2pSelfTransportDevice> &&
        std::is_trivially_destructible_v<P2pSelfTransportDevice>,
    "P2pSelfTransportDevice must be standard layout with trivial destructor");
static_assert(
    std::is_standard_layout_v<P2pNvlTransportDevice> &&
        std::is_trivially_destructible_v<P2pNvlTransportDevice>,
    "P2pNvlTransportDevice must be standard layout with trivial destructor");
static_assert(
    std::is_standard_layout_v<P2pIbTransportDevice> &&
        std::is_trivially_destructible_v<P2pIbTransportDevice>,
    "P2pIbTransportDevice must be standard layout with trivial destructor");

/**
 * Transport type tag for discriminated union.
 * Used to identify which transport type is active in the Transport union.
 */
enum class TransportType : uint8_t {
  SELF,
  P2P_NVL,
  P2P_IBGDA,
  P2P_IBRC,
};

/// Human-readable name for TransportType (host-only).
inline const char* transport_type_name(TransportType t) {
  switch (t) {
    case TransportType::SELF:
      return "SELF";
    case TransportType::P2P_NVL:
      return "P2P_NVL";
    case TransportType::P2P_IBGDA:
      return "P2P_IBGDA";
    case TransportType::P2P_IBRC:
      return "P2P_IBRC";
  }
  return "UNKNOWN";
}

/**
 * Polymorphic transport wrapper using tagged union.
 * Allows storing either self-transport (intra-GPU) or P2P NVL transport
 * (inter-GPU) or an IB transport pointer in a single type for heterogeneous
 * communication patterns.
 *
 * NOTE: Transport handle structs contain pointers to externally-managed GPU
 * resources. Transport owns a copy of the handle, NOT the underlying GPU
 * memory. The parent transport object must outlive any Transport instance.
 *
 * Memory layout: [type tag (1 byte)] + [union of transport objects]
 *
 * Usage:
 *   Transport t1(selfTransport);      // Create self-transport
 *   Transport t2(p2pNvlTransport);    // Create P2P transport
 *   Transport t3 = std::move(t1);     // Move is supported
 *   // Copy is deleted - transports contain device pointers/IPC handles
 */
struct Transport {
  TransportType type;
  union {
    P2pSelfTransportDevice self;
    P2pNvlTransportDevice p2p_nvl;
    P2pIbTransportDevice p2p_ib;
  };

  /** Constructor for SelfTransportDevice */
  __host__ __device__ explicit Transport(const P2pSelfTransportDevice& s)
      : type(TransportType::SELF), self(s) {}

  /** Constructor for P2pNvlTransportDevice */
  __host__ __device__ explicit Transport(const P2pNvlTransportDevice& p)
      : type(TransportType::P2P_NVL), p2p_nvl(p) {}

  /** Constructor for IBGDA device transport (non-owning pointer). */
  __host__ __device__ explicit Transport(P2pIbgdaTransportDevice* p)
      : type(TransportType::P2P_IBGDA), p2p_ib(p) {}

  /** Constructor for IBRC device transport (non-owning pointer). */
  __host__ __device__ explicit Transport(P2pIbrcTransportDevice* p)
      : type(TransportType::P2P_IBRC), p2p_ib(p) {}

  /**
   * Delete copy constructor and copy assignment.
   * Transport objects contain device pointers and IPC handles that should not
   * be shallow-copied.
   */
  Transport(const Transport&) = delete;
  Transport& operator=(const Transport&) = delete;

  /**
   * Move constructor.
   * Uses placement new to move-construct the active union member.
   */
  __host__ __device__ Transport(Transport&& other) noexcept : type(other.type) {
    if (type == TransportType::SELF) {
      new (&self) P2pSelfTransportDevice(std::move(other.self));
    } else if (type == TransportType::P2P_NVL) {
      new (&p2p_nvl) P2pNvlTransportDevice(std::move(other.p2p_nvl));
    } else if (
        type == TransportType::P2P_IBGDA || type == TransportType::P2P_IBRC) {
      new (&p2p_ib) P2pIbTransportDevice(other.p2p_ib);
    }
  }

  /**
   * Move assignment operator.
   * Destroys current union member, then move-constructs from other.
   */
  __host__ __device__ Transport& operator=(Transport&& other) noexcept {
    if (this != &other) {
      // Destroy current union member.
      // IB transports are non-owning pointers, no cleanup needed.
      if (type == TransportType::SELF) {
        self.~P2pSelfTransportDevice();
      } else if (type == TransportType::P2P_NVL) {
        p2p_nvl.~P2pNvlTransportDevice();
      }

      // Move from other
      type = other.type;
      if (type == TransportType::SELF) {
        new (&self) P2pSelfTransportDevice(std::move(other.self));
      } else if (type == TransportType::P2P_NVL) {
        new (&p2p_nvl) P2pNvlTransportDevice(std::move(other.p2p_nvl));
      } else if (
          type == TransportType::P2P_IBGDA || type == TransportType::P2P_IBRC) {
        new (&p2p_ib) P2pIbTransportDevice(other.p2p_ib);
      }
    }
    return *this;
  }

  /**
   * Destructor.
   * Explicitly destroys the active union member.
   */
  __host__ __device__ ~Transport() {
    // Union members with non-trivial destructors need explicit cleanup.
    // IB transports are non-owning pointers, no cleanup needed.
    if (type == TransportType::SELF) {
      self.~P2pSelfTransportDevice();
    } else if (type == TransportType::P2P_NVL) {
      p2p_nvl.~P2pNvlTransportDevice();
    }
  }
};

} // namespace comms::prims
