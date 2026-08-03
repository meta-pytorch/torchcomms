// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/ibrc/MultipeerIbrcTransportCuda.cuh"

#include <new>

#include "comms/prims/transport/ibrc/P2pIbrcTransportDevice.cuh"

namespace comms::prims {

std::size_t ibrcDeviceSlotSize() {
  return sizeof(P2pIbrcTransportDevice);
}

void writeIbrcDeviceSlot(
    void* slotsHost,
    int peerIndex,
    DeviceSpan<IbrcCmdQueueDevice> queues,
    uint32_t numNics,
    uint32_t maxChannels,
    uint32_t qpsPerConnection,
    uint32_t qpDirectionCount,
    DeviceSpan<IbChannel> channels,
    IbgdaRemoteBuffer remoteSignalBuf,
    IbgdaLocalBuffer localSignalBuf,
    IbgdaLocalBuffer counterDeviceBuf,
    IbgdaLocalBuffer counterHostBuf,
    int numSignalSlots,
    int numCounterSlots,
    IbChannelLayout channelLayout) {
  auto* slots = static_cast<P2pIbrcTransportDevice*>(slotsHost);
  new (&slots[peerIndex]) P2pIbrcTransportDevice(
      queues,
      numNics,
      maxChannels,
      qpsPerConnection,
      qpDirectionCount,
      channels,
      remoteSignalBuf,
      localSignalBuf,
      counterDeviceBuf,
      counterHostBuf,
      numSignalSlots,
      numCounterSlots,
      channelLayout);
}

} // namespace comms::prims
