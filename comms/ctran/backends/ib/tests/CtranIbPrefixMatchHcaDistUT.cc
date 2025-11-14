// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <iostream>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ctran::ibvwrap;

class CtranIbHcaTest : public CtranDistTest {
 public:
  CtranIbHcaTest() = default;
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    CtranDistTest::SetUp();
  }

  void printTestDesc(const std::string& testName, const std::string& testDesc) {
    if (this->globalRank == 0) {
      std::cout << testName << " numRanks " << this->numRanks << "."
                << std::endl
                << testDesc << std::endl;
    }
  }
};

TEST_F(CtranIbHcaTest, IbHcaPrefixMatchDev) {
  this->printTestDesc(
      "IbHcaPrefixMatchDev",
      "Expect the used device lists match prefix specified in NCCL_IB_HCA.");

  int nDevices;
  CUDACHECK_TEST(cudaGetDeviceCount(&nDevices));

#if !defined(USE_ROCM)
  std::string ibHcaStr = "mlx5_";
#else
  std::string ibHcaStr = "bnxt_re";
#endif
  setenv("NCCL_IB_HCA", ibHcaStr.c_str(), 1);
  // Reinitialize CVAR after setenv
  ncclCvarInit();

  // Rank 0 creates comm with differen local GPU, to check whether all used
  // devices match the condition
  for (int devId = 0; devId < nDevices; devId++) {
    auto ctrlMgr = std::make_unique<CtranCtrlManager>();
    CtranComm* comm = this->commRAII->ctranComm;

    EXPECT_EQ(NCCL_IB_HCA_PREFIX, "");

    try {
      auto ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());
      EXPECT_EQ(
          ctranIb->getIbDevName().compare(0, ibHcaStr.size(), ibHcaStr), 0);
      printf(
          "CtranIbTest.IbHcaPrefixMatchDev: Rank %d devId %d uses devName %s devPort %d\n",
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
  ::testing::AddGlobalTestEnvironment(new CtranDistTestEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
