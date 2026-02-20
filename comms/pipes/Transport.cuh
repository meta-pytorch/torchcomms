// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <new>
#include <type_traits>

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/P2pSelfTransportDevice.cuh"

namespace comms::pipes {
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

// Forward declaration for IBGDA transport (full definition in .cuh, needs CUDA)
class P2pIbgdaTransportDevice;

/**
 * Transport type tag for discriminated union.
 * Used to identify which transport type is active in the Transport union.
 */
enum class TransportType : uint8_t { SELF, P2P_NVL, P2P_IBGDA };

/**
 * Polymorphic transport wrapper using tagged union.
 * Allows storing either self-transport (intra-GPU) or P2P NVL transport
 * (inter-GPU) in a single type for heterogeneous communication patterns.
 *
 * NOTE: All three transport types are lightweight handle structs containing
 * pointers to externally-managed GPU resources. Transport owns a copy of the
 * handle, NOT the underlying GPU memory — the parent transport object
 * (MultiPeerNvlTransport, MultipeerIbgdaTransport) must outlive any Transport
 * instance.
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
    // Stored as pointer (not by value) because P2pIbgdaTransportDevice.cuh
    // includes DOCA device headers with CUDA-only intrinsics (atomicCAS, __ldg,
    // etc.) that cannot compile in .cc translation units. A forward declaration
    // + non-owning pointer avoids pulling those headers into Transport.cuh.
    P2pIbgdaTransportDevice* p2p_ibgda;
  };

  /** Constructor for SelfTransportDevice */
  __host__ __device__ explicit Transport(const P2pSelfTransportDevice& s)
      : type(TransportType::SELF), self(s) {}

  /** Constructor for P2pNvlTransportDevice */
  __host__ __device__ explicit Transport(const P2pNvlTransportDevice& p)
      : type(TransportType::P2P_NVL), p2p_nvl(p) {}

  /** Constructor for P2pIbgdaTransportDevice (non-owning pointer) */
  __host__ __device__ explicit Transport(P2pIbgdaTransportDevice* p)
      : type(TransportType::P2P_IBGDA), p2p_ibgda(p) {}

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
    } else {
      p2p_ibgda = other.p2p_ibgda;
    }
  }

  /**
   * Move assignment operator.
   * Destroys current union member, then move-constructs from other.
   */
  __host__ __device__ Transport& operator=(Transport&& other) noexcept {
    if (this != &other) {
      // Destroy current union member.
      // P2P_IBGDA is a non-owning pointer, no cleanup needed.
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
      } else {
        p2p_ibgda = other.p2p_ibgda;
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
    // P2P_IBGDA is a non-owning pointer, no cleanup needed.
    if (type == TransportType::SELF) {
      self.~P2pSelfTransportDevice();
    } else if (type == TransportType::P2P_NVL) {
      p2p_nvl.~P2pNvlTransportDevice();
    }
  }
};

} // namespace comms::pipes
