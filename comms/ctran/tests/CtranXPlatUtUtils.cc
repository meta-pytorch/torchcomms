// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include <folly/logging/xlog.h>
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/Utils.h"
#include "comms/testinfra/TestXPlatUtils.h"

std::unique_ptr<TestCtranCommRAII> createDummyCtranComm(int devId) {
  CUDACHECK_TEST(cudaSetDevice(devId));

  CHECK_EQ(ctran::utils::commCudaLibraryInit(), commSuccess);

  const std::string uuid{"0"};
  uint64_t commHash =
      ctran::utils::getHash(uuid.data(), static_cast<int>(uuid.size()));
  std::string commDesc("DummyCtranTestComm-0");

  auto result = createCtranCommWithBootstrap(0, 1, 0, commHash, commDesc);

  // Create a TestCtranCommRAII that also holds the bootstrap
  auto raii = std::make_unique<TestCtranCommRAII>(std::move(result.ctranComm));
  raii->bootstrap_ = std::move(result.bootstrap);
  return raii;
}
