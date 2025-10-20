// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <cassert>
#include "comms/common/IpcGpuBarrier.cuh"
#include "comms/utils/checks.h"

namespace meta::comms {

__host__ DeviceMailbox::DeviceMailbox(int nRanks, int nBlocks, void* flagsBuf)
    : nBlocks_(nBlocks), flags_(static_cast<FlagType*>(flagsBuf)) {
  assert(nRanks == NRANKS);
}

/* static */ __host__ std::pair<std::unique_ptr<DeviceBuffer>, DeviceMailbox>
DeviceMailbox::mallocAndInit(int nRanks, int nBlocks) {
  assert(nRanks == NRANKS);
  auto flagBuf =
      std::make_unique<DeviceBuffer>(nRanks * nBlocks * sizeof(FlagType));
  CUDA_CHECK(
      cudaMemset(flagBuf->get(), 0, nRanks * nBlocks * sizeof(FlagType)));
  DeviceMailbox mailbox{
      nRanks, nBlocks, static_cast<FlagType*>(flagBuf->get())};
  return {std::move(flagBuf), mailbox};
}

__host__ IpcGpuBarrier::IpcGpuBarrier(
    int nRanks,
    int nBlocks,
    int selfRank,
    const std::array<DeviceMailbox, NRANKS>& allMailboxes)
    : nBlocks_(nBlocks), selfRank_(selfRank), allMailboxes_(allMailboxes) {
  assert(nRanks == NRANKS);
}

/* static */ __host__
    std::pair<std::unique_ptr<IpcGpuBarrierResources>, IpcGpuBarrier>
    IpcGpuBarrier::mallocAndInit(
        int nRanks,
        int nBlocks,
        int selfRank,
        std::shared_ptr<ctran::bootstrap::IBootstrap> commBootstrap) {
  assert(nRanks == NRANKS);
  auto [selfMboxBuf, selfMbox] = DeviceMailbox::mallocAndInit(nRanks, nBlocks);

  auto memHandler =
      std::make_unique<IpcMemHandler>(commBootstrap, selfRank, nRanks);
  memHandler->addSelfDeviceMemPtr(selfMboxBuf->get());
  memHandler->exchangeMemPtrs();

  std::array<DeviceMailbox, NRANKS> allMailboxes;
  for (int i = 0; i < nRanks; i++) {
    if (i == selfRank) {
      allMailboxes[i] = selfMbox;
    } else {
      allMailboxes[i] =
          DeviceMailbox(nRanks, nBlocks, memHandler->getPeerDeviceMemPtr(i));
    }
  }

  IpcGpuBarrier barrier(nRanks, nBlocks, selfRank, allMailboxes);

  IpcGpuBarrierResources resources{
      .ipcMemHandler = std::move(memHandler),
      .selfMailboxBuf = std::move(selfMboxBuf),
  };
  return {
      std::make_unique<IpcGpuBarrierResources>(std::move(resources)), barrier};
}

} // namespace meta::comms
