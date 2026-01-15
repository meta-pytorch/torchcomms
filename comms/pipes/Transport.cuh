// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <new>

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/P2pSelfTransportDevice.cuh"

namespace comms::pipes {

/**
 * Transport type tag for discriminated union.
 * Used to identify which transport type is active in the Transport union.
 */
enum class TransportType : uint8_t { SELF, P2P_NVL };

/**
 * Polymorphic transport wrapper using tagged union.
 * Allows storing either self-transport (intra-GPU) or P2P NVL transport
 * (inter-GPU) in a single type for heterogeneous communication patterns.
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
  };

  /** Constructor for SelfTransportDevice */
  __host__ __device__ explicit Transport(const P2pSelfTransportDevice& s)
      : type(TransportType::SELF), self(s) {}

  /** Constructor for P2pNvlTransportDevice */
  __host__ __device__ explicit Transport(const P2pNvlTransportDevice& p)
      : type(TransportType::P2P_NVL), p2p_nvl(p) {}

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
    } else {
      new (&p2p_nvl) P2pNvlTransportDevice(std::move(other.p2p_nvl));
    }
  }

  /**
   * Move assignment operator.
   * Destroys current union member, then move-constructs from other.
   */
  __host__ __device__ Transport& operator=(Transport&& other) noexcept {
    if (this != &other) {
      // Destroy current union member
      if (type == TransportType::SELF) {
        self.~P2pSelfTransportDevice();
      } else if (type == TransportType::P2P_NVL) {
        p2p_nvl.~P2pNvlTransportDevice();
      }

      // Move from other
      type = other.type;
      if (type == TransportType::SELF) {
        new (&self) P2pSelfTransportDevice(std::move(other.self));
      } else {
        new (&p2p_nvl) P2pNvlTransportDevice(std::move(other.p2p_nvl));
      }
    }
    return *this;
  }

  /**
   * Destructor.
   * Explicitly destroys the active union member.
   */
  __host__ __device__ ~Transport() {
    // Union members with non-trivial destructors need explicit cleanup
    if (type == TransportType::SELF) {
      self.~P2pSelfTransportDevice();
    } else if (type == TransportType::P2P_NVL) {
      p2p_nvl.~P2pNvlTransportDevice();
    }
  }
};

} // namespace comms::pipes
