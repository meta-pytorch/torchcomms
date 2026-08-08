// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <gmock/gmock.h>
#include <nccl.h> // @manual
#include <string>
#include <unordered_map>

#include "comms/torchcomms/ncclx/CtranApi.hpp"

namespace torch::comms::test {

/// gmock impl of CtranApi for the phase-1 (ctran-only) collective surface.
class CtranMock : public CtranApi {
 public:
  ~CtranMock() override = default;

  // Alias avoids passing a comma-bearing type token through the
  // MOCK_METHOD macro.
  using HintsMap = std::unordered_map<std::string, std::string>;

  MOCK_METHOD(CtranComm*, getCtranComm, (ncclComm_t comm), (override));

  MOCK_METHOD(bool, allGatherPSupport, (CtranComm * ctranComm), (override));

  MOCK_METHOD(
      ncclResult_t,
      allGatherPInit,
      (void* recvbuff,
       size_t maxRecvCount,
       const HintsMap& hints,
       ncclDataType_t datatype,
       CtranComm* ctranComm,
       cudaStream_t stream,
       void** request),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      allGatherPExec,
      (const void* sendbuff,
       size_t count,
       ncclDataType_t datatype,
       void* request),
      (override));

  MOCK_METHOD(ncclResult_t, allGatherPDestroy, (void* request), (override));
};

} // namespace torch::comms::test
