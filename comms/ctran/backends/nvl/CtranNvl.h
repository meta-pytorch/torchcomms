// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_NVL_H_
#define CTRAN_NVL_H_

#include <memory>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/CtranCtrl.h"
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

  // Register memory to be used for NVL operations. If registration fails
  // because memory type is not supported, commSuccess shall be returned and the
  // nvlRegElem will be set to nullptr. Mapper is expected to still handle the
  // case with alternative path.
  // Input arguments:
  //   - buf: the local buffer to be registered to network for direct RDMA
  //   access
  //   - len: the length of the local buffer
  //   - cudaDev: the cuda device id of the local buffer
  // Output arguments:
  //   - nvlRegElem: the nvlRegElem of the local buffer that stores the
  //                registration handle.
  static commResult_t regMem(
      const void* buf,
      const size_t len,
      const int cudaDev,
      void** nvlRegElem,
      bool shouldSupportCudaMalloc = false);

  // Deregister memory to be used for NVL operations.
  // Input arguments:
  //   - nvlRegElem: the nvlRegElem of the local buffer that stores the
  //                registration handle.
  static commResult_t deregMem(void* nvlRegElem);

  // Release the exported memory on remote rank.
  // Input arguments:
  //   - nvlRegElem: local registration
  // Output arguments:
  //   - msg: the reference to the control message to be sent to remote rank.
  //          Contents filled at return.
  static commResult_t remReleaseMem(void* nvlRegElem, ControlMsg& msg);

  // Return if the perr can be communicated via NVL backend
  // Input arguments:
  //   - rank: the rank of the peer in the current communicator
  bool isSupported(int rank);

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

#endif
