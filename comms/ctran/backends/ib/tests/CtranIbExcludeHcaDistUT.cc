// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <iostream>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ctran::ibvwrap;

class CtranIbHcaTest : public ctran::CtranDistTestFixture {
 public:
  CtranIbHcaTest() = default;
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    ctran::CtranDistTestFixture::SetUp();
    this->comm_ = makeCtranComm();
  }

  void TearDown() override {
    this->comm_.reset();
    ctran::CtranDistTestFixture::TearDown();
  }

  void printTestDesc(const std::string& testName, const std::string& testDesc) {
    if (this->globalRank == 0) {
      std::cout << testName << " numRanks " << this->numRanks << "."
                << std::endl
                << testDesc << std::endl;
    }
  }

 protected:
  std::unique_ptr<CtranComm> comm_{nullptr};
};

TEST_F(CtranIbHcaTest, IbHcaExcludeDev) {
  this->printTestDesc(
      "IbHcaExcludeDev",
      "Expect the excluded device lists specified by NCCL_IB_HCA are not used.");

  int nDevices;
  CUDACHECK_TEST(cudaGetDeviceCount(&nDevices));

#if !defined(USE_ROCM)
  std::string ibHcaStr = "^mlx5_10,mlx5_3";
  std::vector<std::string> ibHcaExcludeDevs{"mlx5_10", "mlx5_3"};
#else
  std::string ibHcaStr = "^bnxt_re1,bnxt_re2";
  std::vector<std::string> ibHcaExcludeDevs{"bnxt_re1", "bnxt_re2"};
#endif
  setenv("NCCL_IB_HCA", ibHcaStr.c_str(), 1);
  // Reinitialize CVAR after setenv
  ncclCvarInit();

  // Rank 0 creates comm with differen local GPU, to check whether all used
  // devices match the condition
  for (int devId = 0; devId < nDevices; devId++) {
    auto ctrlMgr = std::make_unique<CtranCtrlManager>();
    CtranComm* comm = this->comm_.get();

    EXPECT_EQ(NCCL_IB_HCA_PREFIX, "^");

    try {
      auto ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());
      for (auto& dev : ibHcaExcludeDevs) {
        EXPECT_NE(ctranIb->getIbDevName(), dev);
      }
      printf(
          "CtranIbTest.IbHcaExcludeDev: Rank %d devId %d uses devName %s devPort %d\n",
          this->globalRank,
          devId,
          ctranIb->getIbDevName().c_str(),
          ctranIb->getIbDevPort());
    } catch (const std::bad_alloc& e) {
      printf("CtranIbTest: IB backend not enabled. Skip test\n");
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
