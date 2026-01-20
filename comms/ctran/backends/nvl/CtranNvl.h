// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_NVL_H_
#define CTRAN_NVL_H_

#include <memory>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/nvl/CtranNvlBase.h"
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

  // Register control message callback function if have any.
  // Input arguments:
  //   - ctrlMgr: the ctranCtrlManager to manage the control message
  void regCtrlCb(std::unique_ptr<CtranCtrlManager>& ctrlMgr);

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

  // Import a remote memory registration
  // Input arguments:
  //   - rank: the remote rank
  //   - msg: received control message including remote memory details
  // Output arguments:
  //   - buf: the local buffer mapped to the imported remote memory
  //   - rkey: the remoteAccessKey (rkey) of the remote buffer
  //   registration[]
  commResult_t importMem(
      void** buf,
      struct CtranNvlRemoteAccessKey* rkey,
      int rank,
      const ControlMsg& msg);

  // Export a local memory registration for remote rank to import.
  // Input arguments:
  //   - buf: the local buffer to be exported. The offset to the base of
  //          nvlRegElem will be passed to remote rank.
  //   - nvlRegElem: local registration of the to-be-exported buffer
  // Output arguments:
  //   - msg: the reference to the control message to be sent to remote rank.
  //          Contents filled at return.
  static commResult_t
  exportMem(const void* buf, void* nvlRegElem, ControlMsg& msg);

  // Release the exported memory on remote rank.
  // Input arguments:
  //   - nvlRegElem: local registration
  // Output arguments:
  //   - msg: the reference to the control message to be sent to remote rank.
  //          Contents filled at return.
  static commResult_t remReleaseMem(void* nvlRegElem, ControlMsg& msg);

  // Callback (CB) function to handle incoming NVL_RELEASE_MEM CB_CTRL msg
  // Input arguments:
  //   - rank: the rank sent the CB_CTRL msg
  //   - msg: the pointer to the received CB_CTRL msg
  //   - ctx: the context of the CB_CTRL msg; it is the CtranNvl object passed
  //          in at cb registration
  static commResult_t releaseMemCb(int rank, void* msgPtr, void* ctx);

  commResult_t releaseMem(CtranNvlRemoteAccessKey* rkey);

  // Return if the perr can be communicated via NVL backend
  // Input arguments:
  //   - rank: the rank of the peer in the current communicator
  bool isSupported(int rank);

  size_t getNumRemMem(int rank) const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

#endif
