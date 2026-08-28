// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ncclx/meta/tests/NcclxBaseTest.h"

#include <cstdio>
#include <cstdlib>

#ifndef CUDACHECKABORT
// Fallback when CUDACHECKABORT is not provided by the NCCL headers included
// in this build. Preserves abort-on-failure behavior. Writes to stderr because
// abort() is not required to flush stdio, and test stdout is usually a pipe.
#define CUDACHECKABORT(cmd)               \
  do {                                    \
    cudaError_t err = cmd;                \
    if (err != cudaSuccess) {             \
      fprintf(                            \
          stderr,                         \
          "Cuda failure '%s' at %s:%d\n", \
          cudaGetErrorString(err),        \
          __FILE__,                       \
          __LINE__);                      \
      abort();                            \
    }                                     \
  } while (0)
#endif

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
