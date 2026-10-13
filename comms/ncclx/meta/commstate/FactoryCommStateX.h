// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <algorithm>
#include "comm.h"
#include "meta/MnnvlCliqueId.h"
#include "meta/RankUtil.h"
#include "socket.h"

class CtranComm;

namespace ncclx {

// Create CommStateX from ncclComm. Initializes rank topology via bootstrap
// allgather and sets up NVL fabric topologies. Virtual topology overrides
// (noLocal, vnode, vClique) are applied internally.
ncclResult_t initCommStateXFromNcclComm(void* _comm, CtranComm* ctranComm);
} // namespace ncclx
