// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/Singleton.h>
#include <folly/SocketAddress.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/utils/CudaUtils.h"

#include "comms/ctran/utils/Checks.h"

namespace ctran::utils {

/* static */ BusId BusId::makeFrom(const int cudaDev) {
  BusId val;
  FB_CUDACHECKTHROW(
      cudaDeviceGetPCIBusId(val.busId_.data(), val.busId_.size(), cudaDev));
  return val;
}

/* static */ BusId BusId::makeFrom(const std::string& busIdStr) {
  std::vector<std::string_view> parts;

  folly::split(':', busIdStr, parts);
  if (parts.size() != 3) {
    throw std::invalid_argument(fmt::format(
        "Can not parse busIdStr: {}, it must have a format \"0000:00:00.0\"",
        busIdStr));
  }
  std::vector<std::string_view> subparts;
  folly::split('.', parts[2], subparts);
  if (subparts.size() != 2) {
    throw std::invalid_argument(fmt::format(
        "Can not parse busIdStr: {}, it must have a format \"0000:00:00.0\"",
        busIdStr));
  }

  BusId val;
  val.busId_ = busIdStr;
  return val;
}

/* static */ BusId BusId::makeFrom(const int64_t& busId) {
  char buf[17]{"00000000:00:00.0"};
  snprintf(
      buf,
      sizeof(buf),
      "%04lx:%02lx:%02lx.%01lx",
      (busId) >> 20,
      (busId & 0xff000) >> 12,
      (busId & 0xff0) >> 4,
      (busId & 0xf));

  return makeFrom(std::string(buf));
}

std::string BusId::toStr() noexcept {
  return busId_;
}

int64_t BusId::toInt64() {
  std::string hexStr;
  for (const auto c : busId_) {
    if (std::isxdigit(c)) {
      hexStr += c;
    }
  }
  return static_cast<int64_t>(std::stoull(hexStr, nullptr, 16));
}

folly::Expected<int, std::string> getCudaArch(int cudaDev) {
  int archMajor = -1;
  int archMinor = -1;
  try {
    FB_CUDACHECKTHROW(cudaDeviceGetAttribute(
        &archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev));
    FB_CUDACHECKTHROW(cudaDeviceGetAttribute(
        &archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev));
  } catch (const std::exception& ex) {
    return folly::makeUnexpected<std::string>(
        fmt::format("Failed to get cuda arch: {}", ex.what()));
  }

  return 100 * archMajor + 10 * archMinor;
}

} // namespace ctran::utils
