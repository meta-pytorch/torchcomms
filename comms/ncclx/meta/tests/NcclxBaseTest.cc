// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ncclx/meta/tests/NcclxBaseTest.h"

#include <folly/logging/xlog.h>
#include <cstdlib>

void NcclxBaseTestFixture::SetUp(const NcclxEnvs& envs) {
  distSetUp();

  setenv("RANK", std::to_string(globalRank).c_str(), 1);

  // Save old env values and apply overrides.
  for (const auto& [key, value] : envs) {
    const char* oldVal = getenv(key.c_str());
    oldEnvs_[key] = oldVal ? std::optional<std::string>(oldVal) : std::nullopt;
    setenv(key.c_str(), value.c_str(), 1);
  }

  CUDACHECKABORT(cudaSetDevice(localRank));

  if (initEnvAtSetup) {
    initEnv();
    ncclCvarInit();
  }
}

void NcclxBaseTestFixture::TearDown() {
  // Restore original env values.
  for (const auto& [key, value] : oldEnvs_) {
    if (value) {
      setenv(key.c_str(), value->c_str(), 1);
    } else {
      unsetenv(key.c_str());
    }
  }

  distTearDown();
}
