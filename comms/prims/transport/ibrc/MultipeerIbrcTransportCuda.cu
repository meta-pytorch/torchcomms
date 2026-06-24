// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/ibrc/MultipeerIbrcTransportCuda.cuh"

#include <new>

#include "comms/prims/transport/ibrc/P2pIbrcTransportDevice.cuh"

namespace comms::prims {

std::size_t ibrcDeviceSlotSize() {
  return sizeof(P2pIbrcTransportDevice);
}

void constructIbrcDeviceSlots(void* slotsHost, int numSlots) {
  auto* slots = static_cast<P2pIbrcTransportDevice*>(slotsHost);
  for (int i = 0; i < numSlots; ++i) {
    new (&slots[i]) P2pIbrcTransportDevice();
  }
}

void writeIbrcDeviceSlot(
    void* slotsHost,
    int peerIndex,
    DeviceSpan<IbrcCmdQueueDevice> queues,
    uint32_t numNics,
    uint32_t maxGroups,
    uint32_t numQpsPerPeerPerNic,
    DeviceSpan<IbrcBlockQpState> blockQpState,
    IbgdaRemoteBuffer remoteSignalBuf,
    IbgdaLocalBuffer localSignalBuf,
    IbgdaLocalBuffer counterDeviceBuf,
    IbgdaLocalBuffer counterHostBuf,
    int numSignalSlots,
    int numCounterSlots,
    IbSendRecvState sendRecvState) {
  auto* slots = static_cast<P2pIbrcTransportDevice*>(slotsHost);
  new (&slots[peerIndex]) P2pIbrcTransportDevice(
      queues,
      numNics,
      maxGroups,
      numQpsPerPeerPerNic,
      blockQpState,
      remoteSignalBuf,
      localSignalBuf,
      counterDeviceBuf,
      counterHostBuf,
      numSignalSlots,
      numCounterSlots,
      sendRecvState);
}

} // namespace comms::prims
