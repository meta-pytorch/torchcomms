// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <algorithm>
#include "comm.h"
#include "comms/ctran/commstate/CommStateX.h"
#include "meta/RankUtil.h"
#include "socket.h"

class CtranComm;

namespace ncclx {

// Create CommStateX from ncclComm. Initializes rank topology via bootstrap
// allgather and sets up NVL fabric topologies. Virtual topology overrides
// (noLocal, vnode, vClique) are applied internally.
ncclResult_t createCommStateXFromNcclComm(void* _comm, CtranComm* ctranComm);

ncclResult_t assignMnnvlCliqueIdBasedOnCliqueSize(int* cliqueId);
} // namespace ncclx
