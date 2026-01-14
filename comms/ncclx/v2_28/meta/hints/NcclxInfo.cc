// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>
#include <string>
#include <unordered_map>

#include "nccl.h"
#include "param.h"

#include "meta/colltrace/CollTraceWrapper.h"

namespace ncclx {

// Anonymous namespace
namespace {

std::shared_ptr<const std::unordered_map<std::string, std::string>> ncclxInfo{};

std::once_flag initInfoFlag;

std::unordered_map<std::string, std::string> initBasicInfo() {
  std::unordered_map<std::string, std::string> info;
  info["ncclx_version"] = fmt::format(
      "{}.{}.{}.{}", NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, NCCL_SUFFIX);
  return info;
}

std::unordered_map<std::string, std::string> gatherNcclxInfo() {
  std::unordered_map<std::string, std::string> tempInfoMap{};
  tempInfoMap.merge(initBasicInfo());
  tempInfoMap.merge(::meta::comms::ncclx::collTraceGetInfo());
  return tempInfoMap;
}

void initNcclxInfoImpl() {
  ncclxInfo =
      std::make_shared<decltype(ncclxInfo)::element_type>(gatherNcclxInfo());
}

void initNcclxInfo() {
  std::call_once(initInfoFlag, []() { initNcclxInfoImpl(); });
}
} // namespace

__attribute__((visibility("default")))
std::shared_ptr<const std::unordered_map<std::string, std::string>>
getNcclxInfo() {
  initEnv();
  initNcclxInfo();

  return ncclxInfo;
}

std::unordered_map<std::string, std::string> testOnlyGatherNcclxInfo() {
  return gatherNcclxInfo();
}

} // namespace ncclx
