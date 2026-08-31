// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/*
 * Multi-peer NVL progress coverage, in its own binary because it needs more
 * ranks than the rest of the suite.
 *
 * `p2p_nvl_transport_test` is configured for exactly 2 ranks, and at two ranks
 * a rank's local peer index is always zero -- so `progressDirectionStride_` and
 * the per-peer slicing in MultiPeerNvlTransport are never exercised there. This
 * target runs at 4 ranks so that arithmetic has a nonzero offset under test in
 * CI, rather than only under a manual launch.
 */

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/tests/NvlProgressPayload.h"
#include "comms/prims/tests/P2pNvlTransportTest.cuh"
#include "comms/prims/transport/nvl/MultiPeerNvlTransport.h"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"
#include "comms/testinfra/DistEnvironmentBase.h"
#include "comms/testinfra/DistTestBase.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::prims::tests {

namespace {

int g_bootstrapSeq = 0;

/*
 * Per-test out-of-band bootstrap. The prefix must be identical on every rank or
 * they rendezvous on different stores and hang, and distinct between bootstraps
 * or a later one picks up an earlier one's keys. Deriving it from the running
 * test name gives both, since every rank runs the same test.
 */
std::shared_ptr<meta::comms::IBootstrap> makeTestBootstrap() {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string prefix = std::string(info->test_suite_name()) + "." +
      info->name() + "#" + std::to_string(g_bootstrapSeq++);
  return std::shared_ptr<meta::comms::IBootstrap>(
      meta::comms::createBootstrap(prefix));
}

MultiPeerNvlTransportConfig makeNvlConfig(
    std::size_t dataBufferSize,
    std::size_t pipelineDepth,
    int maxNumChannels = 1) {
  const auto channels = static_cast<std::size_t>(maxNumChannels);
  const std::size_t chunkAlign = 16 * std::max<std::size_t>(pipelineDepth, 1);
  const std::size_t perChannelSize =
      std::max<std::size_t>(16, ((dataBufferSize + channels - 1) / channels));
  return MultiPeerNvlTransportConfig{
      .pipelineDepth = pipelineDepth,
      .maxNumChannels = maxNumChannels,
      .perChannelSize =
          ((perChannelSize + chunkAlign - 1) / chunkAlign) * chunkAlign,
  };
}

} // namespace

class NvlProgressMultiPeerTestFixture : public ::testing::Test,
                                        public meta::comms::DistBaseTest {
 protected:
  void SetUp() override {
    distSetUp();
    g_bootstrapSeq = 0;
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    distTearDown();
  }
};

TEST_F(NvlProgressMultiPeerTestFixture, ProgressTwoPeersConcurrently) {
  // This target is configured for exactly 4 ranks. Anything else means the
  // launcher misconfigured the job, which must fail rather than skip -- a skip
  // here would restore exactly the silent no-coverage this target exists to
  // remove. The ring only needs 3 ranks to give the per-peer progress slice a
  // nonzero offset, but the guard matches the target so a miscount cannot pass.
  ASSERT_EQ(numRanks, 4) << "target is configured for 4 ranks, got "
                         << numRanks;

  // Ring topology: every rank exchanges with both its predecessor and its
  // successor concurrently, which keeps each rank's launch symmetric and
  // guarantees both peers are reciprocating rather than waiting on a rank that
  // never engages.
  const int predRank = (globalRank + numRanks - 1) % numRanks;
  const int succRank = (globalRank + 1) % numRanks;
  const size_t nbytes = 256 * 1024;

  auto config = makeNvlConfig(
      /*dataBufferSize=*/1024 * 1024,
      /*pipelineDepth=*/2,
      /*maxNumChannels=*/1);
  auto bootstrap = makeTestBootstrap();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  auto p2pPred = transport.buildP2pTransportDevice(predRank);
  auto p2pSucc = transport.buildP2pTransportDevice(succRank);

  DeviceBuffer predSrc(nbytes);
  DeviceBuffer predDst(nbytes);
  DeviceBuffer succSrc(nbytes);
  DeviceBuffer succDst(nbytes);

  // Keyed on the destination peer, so a chunk delivered to or from the wrong
  // peer mismatches even at the right offset -- which is exactly what a wrong
  // per-peer slice offset would produce.
  const std::vector<char> predHost =
      comms::prims::test::makeProgressPayload(globalRank, predRank, nbytes);
  const std::vector<char> succHost =
      comms::prims::test::makeProgressPayload(globalRank, succRank, nbytes);
  CUDACHECK_TEST(cudaMemcpy(
      predSrc.get(), predHost.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      succSrc.get(), succHost.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(predDst.get(), 0, nbytes));
  CUDACHECK_TEST(cudaMemset(succDst.get(), 0, nbytes));

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
  comms::prims::test::testProgressTwoPeerSendRecv(
      p2pPred,
      p2pSucc,
      predSrc.get(),
      predDst.get(),
      succSrc.get(),
      succDst.get(),
      nbytes,
      /*maxSignalBytes=*/0,
      AbortDevice(),
      /*blockSize=*/256);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);

  // This rank is its predecessor's successor and vice versa, so each peer
  // filled its payload keyed on this rank as the destination.
  std::vector<char> hostBuf(nbytes);
  CUDACHECK_TEST(cudaMemcpy(
      hostBuf.data(), predDst.get(), nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; i++) {
    ASSERT_EQ(
        static_cast<unsigned char>(hostBuf[i]),
        comms::prims::test::progressPayloadByte(predRank, globalRank, i))
        << "predecessor payload mismatch at byte " << i;
  }
  CUDACHECK_TEST(cudaMemcpy(
      hostBuf.data(), succDst.get(), nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; i++) {
    ASSERT_EQ(
        static_cast<unsigned char>(hostBuf[i]),
        comms::prims::test::progressPayloadByte(succRank, globalRank, i))
        << "successor payload mismatch at byte " << i;
  }
}

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  // InitGoogleTest first: it strips the --gtest_* flags from argv. gflags,
  // which folly::Init sets up, hard-errors on flags it does not recognise, so
  // the reverse order breaks every filtered invocation, including running this
  // suite under compute-sanitizer with a --gtest_filter.
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::DistEnvironmentBase());
  return RUN_ALL_TESTS();
}
