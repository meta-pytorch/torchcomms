// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/rdma/NicConstants.h"

// Forward declarations
struct doca_gpu_dev_verbs_qp;

namespace comms::pipes {

// Forward declaration for return type
class P2pIbgdaTransportDevice;

/**
 * Per-NIC build spec for one peer's transport.
 *
 * Each entry corresponds to a physical NIC. The build function packs
 * these into a GPU-resident DeviceSpan<NicDeviceIbgdaResources> for the device
 * transport. Host callers are responsible for ordering the vector with
 * peer-specific NIC rotation so device-side `nic_for_group(g) = g % numNics`
 * produces balanced thread-per-peer scatter.
 */
struct NicDeviceIbgdaResourcesBuildSpec {
  std::vector<doca_gpu_dev_verbs_qp*> qps; // primary QPs on this NIC
  std::vector<doca_gpu_dev_verbs_qp*> companionQps; // companion QPs on this NIC
  NetworkLKey sinkLkey{}; // sink lkey for this NIC's PD
  int deviceId{0}; // physical NIC device id (informational)
};

/**
 * Parameters for building a single P2pIbgdaTransportDevice.
 *
 * The build function reads `nicResources` (one entry per NIC), copies QP
 * pointers to GPU memory, and constructs a DeviceSpan<NicDeviceIbgdaResources>
 * for the device transport.
 *
 * Single-NIC callers populate `nicResources` with one element.
 * Multi-NIC callers populate `nicResources` with one element per NIC; qps and
 * companionQps within a NicDeviceIbgdaResourcesBuildSpec must be the same size.
 */
struct P2pIbgdaTransportBuildParams {
  std::vector<NicDeviceIbgdaResourcesBuildSpec> h_nicDeviceIbgdaResources;
  IbgdaRemoteBuffer remoteSignalBuf{};
  IbgdaLocalBuffer localSignalBuf{};
  IbgdaLocalBuffer counterBuf{};
  IbgdaRemoteBuffer discardSignalSlot{};
  int numSignalSlots{0};
  int numCounterSlots{0};
  IbSendRecvState sendRecvState{};
};

/**
 * Build P2pIbgdaTransportDevice array on GPU.
 *
 * For each peer, allocates GPU arrays for QP pointers, copies them,
 * then constructs P2pIbgdaTransportDevice objects in GPU memory.
 * All GPU allocations are pushed into outGpuAllocations for cleanup.
 * If sendRecvState is populated in the build params, it is passed through
 * the transport constructor before copying to GPU.
 *
 * @param params Build parameters (one per peer)
 * @param numPeers Number of peers
 * @param outGpuAllocations Output: all GPU allocations (caller frees)
 * @return Pointer to GPU array of transport objects (also in outGpuAllocations)
 */
P2pIbgdaTransportDevice* buildDeviceTransportsOnGpu(
    const std::vector<P2pIbgdaTransportBuildParams>& params,
    int numPeers,
    std::vector<void*>& outGpuAllocations);

/**
 * Get size of P2pIbgdaTransportDevice struct.
 */
std::size_t getP2pIbgdaTransportDeviceSize();

} // namespace comms::pipes
