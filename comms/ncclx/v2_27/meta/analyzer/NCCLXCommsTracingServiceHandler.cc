// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/analyzer/NCCLXCommsTracingServiceHandler.h"

#include <unordered_map>

#include <nccl.h> // @manual

#include <fmt/core.h>
#include <folly/json/dynamic.h>
#include <folly/json/json.h>
#include <thrift/lib/cpp2/protocol/Serializer.h>

#include "comms/utils/cvars/nccl_cvars.h"
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
  co_return std::make_unique<comms::GetCommsResponse>(std::move(response));
}

}; // namespace ncclx
