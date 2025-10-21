// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <iostream>
#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ctran::ibvwrap;

class CtranIbHcaTest : public NcclxBaseTest {
 public:
  CtranIbHcaTest() = default;
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    NcclxBaseTest::SetUp();
  }

  void printTestDesc(const std::string& testName, const std::string& testDesc) {
    if (this->globalRank == 0) {
      std::cout << testName << " numRanks " << this->numRanks << "."
                << std::endl
                << testDesc << std::endl;
    }
  }
};

TEST_F(CtranIbHcaTest, IbHcaExactMatchDev) {
  this->printTestDesc(
      "IbHcaExactMatchDev",
      "Expect only exactly matched device lists specified by NCCL_IB_HCA are used.");

  int nDevices;
  CUDACHECK_TEST(cudaGetDeviceCount(&nDevices));

  std::string ibHcaStr =
      "=mlx5_10:1,mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_11:1";
  std::vector<std::string> ibHcaExactDevs{
      "mlx5_10",
      "mlx5_0",
      "mlx5_3",
      "mlx5_4",
      "mlx5_5",
      "mlx5_6",
      "mlx5_9",
      "mlx5_11"};
  setenv("NCCL_IB_HCA", ibHcaStr.c_str(), 1);
  // Reinitialize CVAR after setenv
  ncclCvarInit();

  // Rank 0 creates comm with differen local GPU, to check whether all used
  // devices match the condition
  nDevices = std::min(nDevices, (int)ibHcaExactDevs.size());
  for (int devId = 0; devId < nDevices; devId++) {
    int myDevId = this->globalRank == 0 ? devId : this->localRank;
    auto ctrlMgr = std::make_unique<CtranCtrlManager>();
    // TODO: remove this once CtranComm has proper constructor
    auto commDeprecated =
        createNcclComm(this->globalRank, this->numRanks, myDevId);
    CtranComm* comm = commDeprecated->ctranComm_.get();

    EXPECT_EQ(NCCL_IB_HCA_PREFIX, "=");

    try {
      auto ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());
      EXPECT_EQ(ctranIb->getIbDevName(), ibHcaExactDevs[devId]);
      printf(
          "CtranIbTest.IbHcaExactMatchDev: Rank %d devId %d uses devName %s devPort %d\n",
          this->globalRank,
          devId,
          ctranIb->getIbDevName().c_str(),
          ctranIb->getIbDevPort());
    } catch (const std::bad_alloc& e) {
      printf("CtranIbTest: IB backend not enabled. Skip test\n");
    }
    NCCLCHECK_TEST(ncclCommDestroy(commDeprecated));
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
