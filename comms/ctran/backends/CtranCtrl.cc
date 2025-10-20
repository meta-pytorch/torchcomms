// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/logger/LogUtils.h"

const std::string CmsgIbExportMem::name = "IB_EXPORT_MEM";
const std::string CmsgNvlExportMem::name = "NVL_EXPORT_MEM";
const std::string CmsgNvlReleaseMem::name = "NVL_RELEASE_MEM";

commResult_t CtranCtrlManager::regCb(int type, ContrlMsgCbFn fn, void* ctx) {
  if (this->hasCb(type)) {
    CLOGF(
        ERR,
        "Overwriting callback for msg type {}. It indicates a COMM internal bug",
        type);
    return commInternalError;
  }

  this->ctrlMsgCbMap_[type] = {fn, ctx};
  return commSuccess;
}

commResult_t CtranCtrlManager::runCb(int rank, int type, void* msg) const {
  if (!this->hasCb(type)) {
    CLOGF(
        ERR,
        "No callback registered for msg type {}. It indicates a COMM internal bug",
        type);
    return commInternalError;
  }

  auto& cb = this->ctrlMsgCbMap_.at(type);
  FB_COMMCHECK(cb.fn(rank, msg, cb.ctx));

  return commSuccess;
}

bool CtranCtrlManager::hasCb(int type) const {
  return this->ctrlMsgCbMap_.find(type) != this->ctrlMsgCbMap_.end();
}
