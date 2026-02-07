// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "comms/common/bootstrap/IBootstrap.h"

namespace meta::comms {

/*
 * Processes on the same system can use IpcMemHandler to exchange their device
 * memory pointer via IPC. This class is NOT thread-safe, so only one thread in
 * each process should use it.
 * Example usage, on each rank:
 *    IpcMemHandler handler(comm, selfRank, nRanks);
 *    handler.addSelfDeviceMemPtr(localPtr);
 *    handler.exchangeMemPtrs();
 *    // the returned peer ptr supports peer access
 *    peerPtr = handler.getPeerDeviceMemPtr(peerRank);
 */
class IpcMemHandler {
 public:
  IpcMemHandler(
      std::shared_ptr<IBootstrap> commBootstrap,
      int32_t selfRank,
      int32_t nRanks);
  IpcMemHandler(const IpcMemHandler&) = delete;
  IpcMemHandler& operator=(const IpcMemHandler&) = delete;
  IpcMemHandler(IpcMemHandler&&) = delete;
  IpcMemHandler& operator=(IpcMemHandler&&) = delete;
  ~IpcMemHandler();

  void addSelfDeviceMemPtr(void* deviceMemPtr);

  void exchangeMemPtrs();

  // Get the address usable in local process that's either mmapped from
  // other process or return-as-is (local process)
  void* getPeerDeviceMemPtr(int32_t rank);

 private:
  std::shared_ptr<IBootstrap> commBootstrap_;
  const int32_t selfRank_{-1};
  const int32_t nRanks_{-1};
  std::vector<void*> memPtrs_;
  bool exchanged_{false};
};

} // namespace meta::comms
