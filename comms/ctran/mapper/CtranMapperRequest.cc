// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/Checks.h"

static std::unordered_map<CtranMapperRequest::ReqType, std::string> reqTypeStr =
    {{CtranMapperRequest::ReqType::SEND_CTRL, "SEND_CTRL"},
     {CtranMapperRequest::ReqType::RECV_CTRL, "RECV_CTRL"},
     {CtranMapperRequest::ReqType::SEND_SYNC_CTRL, "SEND_SYNC_CTRL"},
     {CtranMapperRequest::ReqType::RECV_SYNC_CTRL, "RECV_SYNC_CTRL"},
     {CtranMapperRequest::ReqType::IB_PUT, "IB_PUT"},
     {CtranMapperRequest::ReqType::NVL_PUT, "NVL_PUT"}};

const std::string getReqTypeStr(CtranMapperRequest::ReqType type) {
  return reqTypeStr[type];
}

CtranMapperRequest::CtranMapperRequest(
    CtranMapperRequest::ReqType type,
    int peer,
    CtranMapperBackend backendParam)
    : type(type), peer(peer), backend(backendParam) {
  if (type == CtranMapperRequest::ReqType::RECV_CTRL) {
    this->recvCtrl.buf = nullptr;
    this->recvCtrl.key = nullptr;
  } else if (type == CtranMapperRequest::ReqType::NVL_PUT) {
    this->config_.kernElem_ = nullptr;
  }
  if (backend != CtranMapperBackend::IB &&
      backend != CtranMapperBackend::SOCKET) {
    CLOGF(
        ERR,
        "CTRAN-MAPPER: Unsupported backend {} for CtranMapperRequest",
        backend);
    FB_COMMCHECKTHROW(commInternalError);
  }
}

CtranMapperRequest::~CtranMapperRequest() {}
