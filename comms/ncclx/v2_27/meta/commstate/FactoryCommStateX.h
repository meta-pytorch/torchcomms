// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <algorithm>
#include "comm.h"
#include "comms/ctran/commstate/CommStateX.h"
#include "meta/RankUtil.h"
#include "socket.h"

namespace ncclx {

// This Factory must be used ONLY in NCCLx code, not in CTRAN code
// Ctran has it's own StateX factory
std::unique_ptr<CommStateX> createCommStateXFromNcclComm(void* _comm);

// This init function must be used ONLY in NCCLx code, not in CTRAN code
// Ctran has it's own StateX factory
ncclResult_t initCtranCommStatexFromNcclComm(
    ncclComm* ncclComm,
    CtranComm* ctranComm);

ncclResult_t assignMnnvlCliqueIdBasedOnCliqueSize(int* cliqueId);
} // namespace ncclx
