// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#if defined(USE_ROCM)
#include "rccl.h" // @manual
#else
#include "nccl.h" // @manual
#endif

/*
 * This class allows you to test Ctran/NCCLX/RCCLX integration via the
 * NCCLX/RCCLX API.
 */
class CtranCclxIntegrationTestUtils : public CtranDistTest {
 public:
  ncclComm_t ncclComm{nullptr};
  CtranComm* ctranComm{nullptr};

  void SetUp() override;

  void TearDown() override;

  void ncclOrCtranRegister(void* buff, size_t size, void** handle);
  void ncclOrCtranDeregister(void* handle);

  void setTestCclxAPI(bool testCclxAPI) {
    LOG(INFO) << "CtranCclxIntegrationTestUtils testCclxAPI = " << testCclxAPI;
    testCclxAPI_ = testCclxAPI;
  }

 private:
  ncclComm_t createNcclComm();
  bool testCclxAPI_{false};
};
