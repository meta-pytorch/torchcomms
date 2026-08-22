// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/wrapper/CtranRegister.h"

#include "comm.h"
#include "debug.h"
#include "nccl.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/wrapper/MetaFactory.h"

namespace ncclx {

bool ctranRegisterEnabled() {
  return NCCL_CTRAN_REGISTER != NCCL_CTRAN_REGISTER::none;
}

bool ctranOwnsRegister(ncclComm* comm) {
  return ctranInitialized(comm->ctranComm_.get()) && ctranRegisterEnabled();
}

ncclResult_t logCtranRegisterConflict() {
  ERR(ncclInvalidUsage,
      "Invalid usage to turn on NCCL_CTRAN_REGISTER and NCCL_LOCAL_REGISTER at the same time.");
  return metaCommToNccl(commInvalidUsage);
}

ncclResult_t
ctranCommRegister(ncclComm* comm, void* buff, size_t size, void** handle) {
  return metaCommToNccl(
      comm->ctranComm_->ctran_->commRegister(buff, size, handle));
}

ncclResult_t ctranCommDeregister(ncclComm* comm, void* handle) {
  return metaCommToNccl(comm->ctranComm_->ctran_->commDeregister(handle));
}

ncclResult_t ctranGlobalRegisterWithPtr(void* buff, size_t size) {
  return metaCommToNccl(ctran::globalRegisterWithPtr(buff, size));
}

ncclResult_t ctranGlobalDeregisterWithPtr(void* buff, size_t size) {
  return metaCommToNccl(ctran::globalDeregisterWithPtr(buff, size));
}

} // namespace ncclx
