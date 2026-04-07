// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_NVL_H_
#define CTRAN_NVL_H_

#include <memory>

#include "comms/ctran/CtranComm.h"
#include "comms/utils/commSpecs.h"

/**
 * CtranNvl class to be used by algorithms and ctranMapper.
 */
class CtranNvl {
 public:
  // Creates local NVL resources for a given communicator.
  // Input arguments:
  //   - comm: the Ctran communicator
  CtranNvl(CtranComm* comm);

  ~CtranNvl();

  // Return if the perr can be communicated via NVL backend
  // Input arguments:
  //   - rank: the rank of the peer in the current communicator
  bool isSupported(int rank);

  // Returns true if NVL support for this rank uses NVLink fabric (cross-node
  // MNNVL). Fabric peers require FABRIC-capable handles (cuMem VMM buffers).
  // cudaMalloc buffers produce cudaIpc handles which are intra-host only and
  // will fail when imported on a different physical machine.
  // Input arguments:
  //   - rank: the rank of the peer in the current communicator
  bool isNvlFabric(int rank) const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

#endif
