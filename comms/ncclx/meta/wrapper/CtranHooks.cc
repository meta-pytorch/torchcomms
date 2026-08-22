// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/wrapper/CtranHooks.h"

#include "comm.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/wrapper/MetaFactory.h"

namespace ncclx {

ncclResult_t runCtranGroupEndHook() {
  return metaCommToNccl(ctranGroupEndHook());
}

void ctranUpdateAsyncError(ncclComm* comm, ncclResult_t* asyncError) {
  // Check Ctran asyncError if no error happens in the baseline path
  if (NCCL_CTRAN_ENABLE && ctranInitialized(comm->ctranComm_.get()) &&
      (*asyncError == ncclSuccess || *asyncError == ncclInProgress)) {
    auto ctranAsyncError = metaCommToNccl(comm->ctranComm_->getAsyncResult());
    // Overwrite if ctranAsyncError is inProgress or error
    if (ctranAsyncError != ncclSuccess) {
      *asyncError = ctranAsyncError;
    }
  }
}

} // namespace ncclx
