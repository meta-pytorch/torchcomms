// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/torchcomms/ncclx/CtranApi.hpp"

#include <stdexcept>

#include <nccl.h> // @manual=//comms/ncclx:nccl

#include "comms/ctran/Ctran.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/utils/commSpecs.h"

// Defined in MetaFactory.cc (per-NCCLX-version). Forward-declared so we
// can call it without depending on NCCLX's private headers.
extern CtranComm* getCtranCommFromNcclComm(ncclComm* ncclComm);

namespace torch::comms {

namespace {

// Type conversions duplicated from MetaFactory.h. The originals are inline
// (not linker-visible) and the header is internal to NCCLX.
commDataType_t toCtranDataType(ncclDataType_t dt) {
  switch (dt) {
    case ncclInt8:
      return commInt8;
    case ncclUint8:
      return commUint8;
    case ncclInt32:
      return commInt32;
    case ncclUint32:
      return commUint32;
    case ncclInt64:
      return commInt64;
    case ncclUint64:
      return commUint64;
    case ncclFloat16:
      return commFloat16;
    case ncclFloat32:
      return commFloat32;
    case ncclFloat64:
      return commFloat64;
    case ncclBfloat16:
      return commBfloat16;
    case ncclFloat8e4m3:
      return commFloat8e4m3;
    case ncclFloat8e5m2:
      return commFloat8e5m2;
    default:
      throw std::runtime_error("Unsupported NCCL data type for ctran");
  }
}

ncclResult_t fromCtranResult(commResult_t result) {
  switch (result) {
    case commSuccess:
      return ncclSuccess;
    case commUnhandledCudaError:
      return ncclUnhandledCudaError;
    case commSystemError:
      return ncclSystemError;
    case commInternalError:
      return ncclInternalError;
    case commInvalidArgument:
      return ncclInvalidArgument;
    case commInvalidUsage:
      return ncclInvalidUsage;
    case commRemoteError:
      return ncclRemoteError;
    case commInProgress:
      return ncclInProgress;
    default:
      throw std::runtime_error("Unknown ctran commResult_t value");
  }
}

meta::comms::Hints toMetaCommHints(
    const std::unordered_map<std::string, std::string>& hints) {
  meta::comms::Hints out;
  for (const auto& [key, val] : hints) {
    out.set(key, val);
  }
  return out;
}

} // namespace

CtranComm* DefaultCtranApi::getCtranComm(ncclComm_t comm) {
  return getCtranCommFromNcclComm(comm);
}

bool DefaultCtranApi::allGatherPSupport(CtranComm* ctranComm) {
  return ctran::allGatherPSupport(ctranComm);
}

ncclResult_t DefaultCtranApi::allGatherPInit(
    void* recvbuff,
    size_t maxRecvCount,
    const std::unordered_map<std::string, std::string>& hints,
    ncclDataType_t datatype,
    CtranComm* ctranComm,
    cudaStream_t stream,
    void** request) {
  CtranPersistentRequest* pReq = nullptr;
  const auto res = ctran::allGatherPInit(
      recvbuff,
      maxRecvCount,
      toMetaCommHints(hints),
      toCtranDataType(datatype),
      ctranComm,
      stream,
      pReq);
  if (res == commSuccess) {
    *request = reinterpret_cast<void*>(pReq);
  }
  return fromCtranResult(res);
}

ncclResult_t DefaultCtranApi::allGatherPExec(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    void* request) {
  return fromCtranResult(
      ctran::allGatherPExec(
          sendbuff,
          count,
          toCtranDataType(datatype),
          reinterpret_cast<CtranPersistentRequest*>(request)));
}

ncclResult_t DefaultCtranApi::allGatherPDestroy(void* request) {
  if (request == nullptr) {
    return ncclSuccess;
  }
  auto* pReq = reinterpret_cast<CtranPersistentRequest*>(request);
  const auto res = ctran::allGatherPDestroy(pReq);
  // only delete the request if the destroy was successful, following existing
  // functionality
  if (res == commSuccess) {
    delete pReq;
  }
  return fromCtranResult(res);
}

} // namespace torch::comms
