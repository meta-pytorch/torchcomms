// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/analyzer/NCCLXCommsTracingServiceHandler.h"

#include <unordered_map>

#include <nccl.h> // @manual

#include <fmt/core.h>
#include <folly/json/dynamic.h>
#include <folly/json/json.h>
#include <thrift/lib/cpp2/protocol/Serializer.h>

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/ProcessGlobalErrorsUtil.h"
#include "comms/utils/trainer/TrainerContext.h"
#include "meta/RankUtil.h"

namespace ncclx {

namespace {
std::chrono::nanoseconds nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch());
}
} // namespace

NCCLXCommsTracingServiceHandler::NCCLXCommsTracingServiceHandler()
    : jobStartTimeNs_(nowNs()) {}

folly::coro::Task<std::unique_ptr<comms::GetCommsResponse>>
NCCLXCommsTracingServiceHandler::co_getComms(
    std::unique_ptr<comms::GetCommsRequest> request) {
  if (!NCCL_COMMSMONITOR_ENABLE) {
    throw std::runtime_error("NCCL_COMMSMONITOR_ENABLE must be enabled");
  }

  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      commHashToKeyValueMap;
  auto result = ncclCommDumpAll(commHashToKeyValueMap);
  if (result != ncclSuccess) {
    throw std::runtime_error(
        fmt::format(
            "Failed to dump all NCCL communicators, error: {}",
            ncclGetErrorString(result)));
  }

  comms::GetCommsResponse response;
  response.globalRank() = RankUtil::getGlobalRank().value();
  response.currentTimeNs() = nowNs().count();
  response.jobStartTimeNs() = jobStartTimeNs_.count();
  response.step() = ncclxGetIteration();
  response.stepStartTimeNs() = stepInfo_.withWLock([&response](auto& stepInfo) {
    // Different step number, assume the step started now
    if (stepInfo.stepOnLastRequest != *response.step()) {
      stepInfo.stepOnLastRequest = *response.step();
      stepInfo.lastRequestTsNs =
          std::chrono::nanoseconds(*response.currentTimeNs());
    }
    return stepInfo.lastRequestTsNs.count();
  });

  for (const auto& [commHash, keyValueMap] : commHashToKeyValueMap) {
    auto& ncclParsedEntry =
        response.commsForRank()->ncclParsedEntryMap()[commHash];
    folly::dynamic obj = folly::dynamic::object();
    for (const auto& [key, value] : keyValueMap) {
      obj[key] = folly::parseJson(value);
    }
    auto s = folly::toJson(obj);
    apache::thrift::SimpleJSONSerializer::deserialize(s, ncclParsedEntry);
  }

  {
    auto globalState = ProcessGlobalErrorsUtil::getAllState();
    for (const auto& ibErr : globalState.ibCompletionErrors) {
      comms::IbCompletionError thriftErr;
      thriftErr.timestampMs() = ibErr.timestampMs.count();
      thriftErr.peer() = ibErr.peer;
      thriftErr.statusStr() = ibErr.statusStr;
      thriftErr.status() = ibErr.status;
      thriftErr.opcodeStr() = ibErr.opcodeStr;
      thriftErr.opcode() = ibErr.opcode;
      thriftErr.reqSize() = ibErr.reqSize;
      thriftErr.vendorErr() = static_cast<int64_t>(ibErr.vendorErr);
      thriftErr.reqType() = ibErr.reqType;
      thriftErr.localGid() = ibErr.localGid;
      thriftErr.remoteGid() = ibErr.remoteGid;
      thriftErr.hcaName() = ibErr.hcaName;
      thriftErr.scaleupDomain() = ibErr.scaleupDomain;
      thriftErr.localHostname() = ibErr.localHostname;
      response.ibErrors().ensure().push_back(std::move(thriftErr));
    }
    // Flush after reading so errors are only reported once
    ProcessGlobalErrorsUtil::clearIbCompletionErrors();
  }

  {
    auto globalState = ProcessGlobalErrorsUtil::getAllState();
    for (const auto& cudaErr : globalState.cudaErrors) {
      comms::CudaError thriftErr;
      thriftErr.timestampMs() = cudaErr.timestampMs.count();
      thriftErr.errorString() = cudaErr.errorString;
      thriftErr.errorCode() = cudaErr.errorCode;
      response.cudaErrors().ensure().push_back(std::move(thriftErr));
    }
    ProcessGlobalErrorsUtil::clearCudaErrors();
  }

  co_return std::make_unique<comms::GetCommsResponse>(std::move(response));
}

}; // namespace ncclx
