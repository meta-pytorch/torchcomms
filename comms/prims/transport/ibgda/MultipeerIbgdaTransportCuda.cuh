// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/rdma/NicConstants.h"
#ifdef __HIP_PLATFORM_AMD__
// On AMD, `doca_gpu_dev_verbs_qp` is a type alias (declared in
// `comms/prims/transport/amd/DocaCompat.h`) for `pipes_gda_gpu_dev_verbs_qp`.
// This .cuh only uses `doca_gpu_dev_verbs_qp*` pointers, so a plain
// forward declaration of the underlying struct + a `using` alias is
// sufficient and avoids dragging the full device-side headers
// (`pipes_gda/PipesGdaOps.h`, `nic/Mlx5NicBackend.h`, `HipDeviceCompat.h`)
// into host-side translation units that just need the pointer type.
struct pipes_gda_gpu_dev_verbs_qp;
using doca_gpu_dev_verbs_qp = pipes_gda_gpu_dev_verbs_qp;
#else
// Forward declarations
struct doca_gpu_dev_verbs_qp;
#endif

namespace comms::prims {

// Forward declarations for device-table types.
struct NicDeviceIbgdaResources;
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
 * Host resources used to populate one peer's fixed device tables.
 *
 * Range population reads the full-capacity per-NIC QP vectors and the exact
 * channel descriptors supplied for that call. Slot publication installs the
 * stable spans after the first prefix has been populated.
 *
 * Single-NIC callers populate `nicResources` with one element.
 * Multi-NIC callers populate `nicResources` with one element per NIC. qps and
 * companionQps both contain maxChannels * qpDirectionCount * qpsPerConnection
 * QPs.
 */
struct P2pIbgdaTransportBuildParams {
  P2pIbgdaTransportBuildParams() = default;
  explicit P2pIbgdaTransportBuildParams(IbChannelLayout channelLayoutIn)
      : channelLayout(channelLayoutIn) {}

  std::vector<NicDeviceIbgdaResourcesBuildSpec> h_nicDeviceIbgdaResources;
  IbgdaRemoteBuffer remoteSignalBuf{};
  IbgdaLocalBuffer localSignalBuf{};
  IbgdaLocalBuffer counterBuf{};
  int numSignalSlots{0};
  int numCounterSlots{0};
  int maxChannels{0};
  int qpsPerConnection{1};
  int qpDirectionCount{1};
  IbChannelLayout channelLayout{};
};

/**
 * Communicator-lifetime device publication tables. All pointers remain stable
 * until transport teardown; range operations only update their contents.
 */
struct IbgdaFixedDeviceTables {
  P2pIbgdaTransportDevice* transports{nullptr};
  doca_gpu_dev_verbs_qp** qps{nullptr};
  NicDeviceIbgdaResources* nicResources{nullptr};
  IbChannel* channels{nullptr};
  int numPeers{0};
  int numNics{0};
  int maxChannels{0};
  int qpsPerConnection{0};
  int qpDirectionCount{0};
  int pipelineDepth{0};
};

/** Allocate and initialize every fixed-capacity device publication table. */
IbgdaFixedDeviceTables allocateIbgdaFixedDeviceTables(
    int numPeers,
    int numNics,
    int maxChannels,
    int qpsPerConnection,
    int qpDirectionCount,
    int pipelineDepth,
    std::vector<void*>& outGpuAllocations);

/** Return one slot from the stable outer transport table. */
P2pIbgdaTransportDevice* ibgdaDeviceSlot(
    P2pIbgdaTransportDevice* transports,
    int peerIndex);

/** Allocate zeroed physical completion state for a channel range. */
IbSendCompletionSlot* allocateIbgdaCompletionSlots(
    std::size_t numChannels,
    int pipelineDepth,
    std::vector<void*>& outGpuAllocations);

/**
 * Populate one peer's canonical channel range without allocating memory.
 * QP vectors retain full-capacity indexing; `rangeChannels` contains exactly
 * [beginChannel, endChannel). The outer slot is published separately once the
 * first prefix is complete.
 */
void populateIbgdaDeviceRange(
    const IbgdaFixedDeviceTables& tables,
    int peerIndex,
    int beginChannel,
    int endChannel,
    const P2pIbgdaTransportBuildParams& params,
    const std::vector<IbChannel>& rangeChannels);

/** Publish the stable outer and per-NIC spans after the first prefix exists. */
void publishIbgdaDeviceSlot(
    const IbgdaFixedDeviceTables& tables,
    int peerIndex,
    const P2pIbgdaTransportBuildParams& params);

/** Clear one peer's canonical channel range without freeing table storage. */
bool clearIbgdaDeviceRange(
    const IbgdaFixedDeviceTables& tables,
    int peerIndex,
    int beginChannel,
    int endChannel) noexcept;

/** Restore one outer slot to an unpublished shape while retaining table bases.
 */
bool resetIbgdaDeviceSlot(
    const IbgdaFixedDeviceTables& tables,
    int peerIndex) noexcept;

} // namespace comms::prims
