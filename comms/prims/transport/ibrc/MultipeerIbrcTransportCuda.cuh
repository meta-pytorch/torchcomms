// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibrc/IbrcTypes.h"

namespace comms::prims {

// Forward declaration: the full definition lives in P2pIbrcTransportDevice.cuh,
// which transitively pulls device-only headers (CopyOp.cuh -> Tile.cuh ->
// hip/hip_bf16.h) that a host `.cc` cannot compile on AMD. Host callers only
// ever name `P2pIbrcTransportDevice*` (a pointer) and use these builders to
// construct/size the slots inside the `.cu` (mirrors IBGDA's
// MultipeerIbgdaTransportCuda).
class P2pIbrcTransportDevice;

// Size of one P2pIbrcTransportDevice slot (for host-side pointer arithmetic
// into the host-pinned mapped array of device handles).
std::size_t ibrcDeviceSlotSize();

// Default-construct (placement-new) each P2pIbrcTransportDevice slot in a
// host-pinned mapped array.
void constructIbrcDeviceSlots(void* slotsHost, int numSlots);

// Placement-new a single populated P2pIbrcTransportDevice into the host-pinned
// mapped array. Args mirror the device-handle constructor; all are plain-data
// host-safe types.
void writeIbrcDeviceSlot(
    void* slotsHost,
    int peerIndex,
    DeviceSpan<IbrcCmdQueueDevice> queues,
    uint32_t numNics,
    uint32_t maxChannels,
    uint32_t qpsPerConnection,
    DeviceSpan<IbLocalChannel> localChannels,
    IbgdaRemoteBuffer remoteSignalBuf,
    IbgdaLocalBuffer localSignalBuf,
    IbgdaLocalBuffer counterDeviceBuf,
    IbgdaLocalBuffer counterHostBuf,
    int numSignalSlots,
    int numCounterSlots,
    IbChannelLayout channelLayout);

} // namespace comms::prims
