#include <gtest/gtest.h>

#include "comms/testinfra/TestsDistUtils.h"

namespace {

size_t getGpuMemorySnapshot() {
  cudaDeviceSynchronize();

  size_t free, total;
  CUDACHECK_TEST(cudaMemGetInfo(&free, &total));

  return total - free;
}

} // namespace

class NcclCommMemoryBench : public NcclxBaseTest {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "vnode", 0);
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
    setenv("NCCL_LAZY_SETUP_CHANNELS", "0", 0);
    NcclxBaseTest::SetUp();

    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    NcclxBaseTest::TearDown();
  }
};

TEST_F(NcclCommMemoryBench, MeasureCommMemory) {
  auto before = getGpuMemorySnapshot();
  NcclCommRAII comm(globalRank, numRanks, localRank);
  auto after = getGpuMemorySnapshot();

  size_t memoryUsedBytes = after - before;
  double memoryUsedMB = static_cast<double>(memoryUsedBytes) / (1 << 20);

  if (globalRank == 0) {
    printf("NCCL Comm Memory: %.2f MB\n", memoryUsedMB);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  return RUN_ALL_TESTS();
}
