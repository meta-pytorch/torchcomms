// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "nccl.h"

namespace ctran {
/**
 * Wrapper class for ncclComm_t to provide custom communicator based
 * communication support via CTran backend
 */

// Forwar declaration to avoid dependency on CtranEx.h
class CtranExRequest;

// This class is an adapter to ctran when its compiled with nccl. It requires
// existing ncclComm input to create a communicator. It servce only one use case
// - ftar. We plan to deprecate it together with ftar and nccl2.25 in favor of
// mccl.
class __attribute__((visibility("default"))) CtranExComm {
 public:
  CtranExComm(const ncclComm_t comm, const std::string& commDesc);

  ~CtranExComm();

  // Return whether a valid communicator and CTran backend is initialized
  bool isInitialized() const;

  // Register a user memory buffer for the given communicator
  // Input arguments:
  //   - ptr: the user buffer to be registered
  //   - size: the size of the user buffer
  //   - forceRegister: whether to force backend registration no matter global
  //                  NCCL_CTRAN_REGISTER configuration
  // Output arguments:
  //   - segHdl: a handle of the registered segment
  ncclResult_t regMem(
      const void* ptr,
      const size_t size,
      void** segHdl,
      bool forceRegister = false);

  // Deregister a user memory buffer for the given communicator
  // Input arguments:
  //   - segHdl: a handle of the registered segment
  ncclResult_t deregMem(void* segHdl);

  // Post a broadcast communication for the given communicator from the root
  // rank's sendbuff to all ranks' recvbuff. Calling this function will post the
  // broadcast and return immediately. The completion of broadcast is tracked by
  // the returned request. Unlike ncclBcast, the broadcast does not maintain
  // stream semantics. The caller is responsible for ensuring that the request
  // is released after use.
  // Input arguments:
  //   - sendbuff: the buffer to be sent from the root rank
  //   - recvbuff: the buffer to be received by all ranks
  //   - count: number of elements in the sendbuff/recvbuff
  //   - datatype: the data type of the sendbuff/recvbuff
  //   - root: the root rank of the broadcast
  // Output arguments:
  //   - req: the request to track the progress of the broadcast. If error is
  //          returned when testing or waiting on the returned request, use
  //          getAsyncErrorString() to get a human-readable error message.
  ncclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      CtranExRequest** req);

  // Return a human-readable message for the last asyncError.
  // User may query it after detected error when waiting on a CtranExRequest
  // request returned by broadcast.
  std::string getAsyncErrorString() const;

  // Query whether the given communicator supports broadcast
  bool supportBroadcast() const;

  // Return (read-only) the underlying NCCL comm object. Should only be used for
  // testing.
  const struct ncclComm* unsafeGetNcclComm() const {
    return comm_;
  }

 private:
  ncclComm_t comm_{NCCL_COMM_NULL};
};
} // namespace ctran
