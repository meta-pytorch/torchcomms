// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once
#include <vector>

#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/utils/commSpecs.h"

namespace ctran {

namespace alltoallp {
struct PersistArgs {
  void* recvbuff;
  void* recvHdl;
  size_t maxRecvCount;
  commDataType_t datatype;
  bool skipCtrlMsg;
  std::vector<void*> remoteRecvBuffs;
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;
};

class AlgoImpl;
} // namespace alltoallp

} // namespace ctran
