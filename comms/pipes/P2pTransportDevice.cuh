// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

/**
 * P2pTransportDevice - Abstract base class for P2P GPU-to-GPU transport
 * ======================================================================
 *
 * Defines the interface for point-to-point communication between GPUs.
 * Concrete implementations (e.g., P2pNvlTransportDevice) provide the actual
 * transport mechanism (NVLink, IB, etc.).
 *
 * VIRTUAL METHODS:
 * ================
 * - send(): Transfer data to peer GPU
 * - recv(): Receive data from peer GPU
 * - write(): Direct memory write from src to dst
 *
 * USAGE:
 * ======
 * Derived classes must implement the virtual methods to provide specific
 * transport implementations.
 *
 * Example:
 *   P2pTransportDevice* transport = ...; // points to P2pNvlTransportDevice
 *   auto group = make_warp_group();
 *   transport->send(group, data, size);
 */
class P2pTransportDevice {
 public:
  __host__ __device__ P2pTransportDevice() = default;
  __host__ __device__ virtual ~P2pTransportDevice() = default;

  /**
   * send - Transfer data to peer GPU
   *
   * Sends 'nbytes' bytes from srcbuff to the peer GPU using the specific
   * transport implementation. All threads in the group cooperate to transfer
   * the data in parallel.
   *
   * @param group ThreadGroup for cooperative processing
   * @param srcbuff Source data pointer (device memory)
   * @param nbytes Number of bytes to send
   */
  __device__ virtual void
  send(ThreadGroup& group, void* srcbuff, std::size_t nbytes) = 0;

  /**
   * recv - Receive data from peer GPU
   *
   * Receives 'nbytes' bytes into dstbuff from the peer GPU's send() call.
   * Must be called simultaneously with peer's send() for the same byte count.
   *
   * @param group ThreadGroup for cooperative processing
   * @param dstbuff Destination data pointer (device memory)
   * @param nbytes Number of bytes to receive (must match sender's count)
   */
  __device__ virtual void
  recv(ThreadGroup& group, void* dstbuff, std::size_t nbytes) = 0;

  /**
   * write - Direct memory write from source to destination
   *
   * Performs a direct memory write operation, copying 'nbytes' from src_d
   * to dst_d. This can be used for local copies or remote writes depending
   * on the implementation.
   *
   * @param group ThreadGroup for cooperative processing
   * @param dst_d Destination pointer (device memory)
   * @param src_d Source pointer (device memory)
   * @param nbytes Number of bytes to write
   */
  __device__ virtual void write(
      ThreadGroup& group,
      char* dst_d,
      const char* src_d,
      std::size_t nbytes) = 0;
};

} // namespace comms::pipes
