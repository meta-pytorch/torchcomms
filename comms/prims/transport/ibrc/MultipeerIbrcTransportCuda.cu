// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/ibrc/MultipeerIbrcTransportCuda.cuh"

#include <new>

#include "comms/prims/transport/ibrc/P2pIbrcTransportDevice.cuh"

namespace comms::prims {

std::size_t ibrcDeviceSlotSize() {
  return sizeof(P2pIbrcTransportDevice);
}

void constructIbrcDeviceSlots(
    void* slotsHost,
    int numSlots,
    int myRank,
    int firstPeerIndex) {
  auto* slots = static_cast<P2pIbrcTransportDevice*>(slotsHost);
  for (int i = 0; i < numSlots; ++i) {
    // Seed the diagnostic identity here too. This is the placeholder path, and
    // today the slot is never logged before `writeIbrcDeviceSlot` fills it in
    // -- but leaving the sentinels means one of the two construction paths for
    // this class can print `rank=-1 peer=-1` the moment that ordering changes.
    //
    // Peer *index* to peer *rank* skips this rank, matching
    // `MultipeerIbrcTransport::peerIndexToRank`.
    const int peerIndex = firstPeerIndex + i;
    const int peerRank = myRank < 0 || firstPeerIndex < 0
        ? -1
        : (peerIndex < myRank ? peerIndex : peerIndex + 1);
    new (&slots[i]) P2pIbrcTransportDevice(
        /*queues=*/{},
        /*nics=*/0,
        /*maxChannels=*/0,
        /*qpsPerConnection=*/0,
        /*localChannels=*/{},
        /*ownedRemoteSignalBuf=*/{},
        /*ownedLocalSignalBuf=*/{},
        /*ownedCounterDeviceBuf=*/{},
        /*ownedCounterHostBuf=*/{},
        /*numSignalSlots=*/0,
        /*numCounterSlots=*/0,
        /*channelLayout=*/{},
        /*abort=*/{},
        /*myRank=*/myRank,
        /*peerRank=*/peerRank);
  }
}

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
    IbChannelLayout channelLayout,
    comms::fault_tolerance::AbortDevice abort,
    int myRank,
    int peerRank) {
  auto* slots = static_cast<P2pIbrcTransportDevice*>(slotsHost);
  new (&slots[peerIndex]) P2pIbrcTransportDevice(
      queues,
      numNics,
      maxChannels,
      qpsPerConnection,
      localChannels,
      remoteSignalBuf,
      localSignalBuf,
      counterDeviceBuf,
      counterHostBuf,
      numSignalSlots,
      numCounterSlots,
      channelLayout,
      abort,
      myRank,
      peerRank);
}

} // namespace comms::prims
