// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cstring>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes::tests {

// Serialize a transport state into zeroed memory field-by-field using raw
// memcpy, so that struct padding bytes stay zero.  Any form of construction
// (copy-ctor, two-arg ctor) may be widened by the compiler at -O2 to a
// full-struct store that carries over padding noise from the source.
template <typename State>
void canonicalizeState(const State& src, void* dst) {
  std::memset(dst, 0, sizeof(State));
  char* out = static_cast<char*>(dst);

  // dataBuffer (plain pointer, no padding)
  auto db = src.dataBuffer;
  std::memcpy(out + offsetof(State, dataBuffer), &db, sizeof(db));

  // Each DeviceSpan has {T* data_, uint32_t size_, [padding]}.
  // Write only data() and size(), leaving trailing padding zeroed.
  auto writeSpan = [out](std::size_t offset, auto& span) {
    auto ptr = span.data();
    auto sz = span.size();
    std::memcpy(out + offset, &ptr, sizeof(ptr));
    std::memcpy(out + offset + sizeof(ptr), &sz, sizeof(sz));
  };
  writeSpan(offsetof(State, stateBuffer), src.stateBuffer);
  writeSpan(offsetof(State, signalBuffer), src.signalBuffer);
  writeSpan(offsetof(State, barrierBuffer), src.barrierBuffer);
}

class MultiPeerNvlTransportFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  std::unique_ptr<MultiPeerNvlTransport> createTransport() {
    MultiPeerNvlTransportConfig config{
        .dataBufferSize = 256 * 1024,
        .chunkSize = 512,
        .pipelineDepth = 4,
        .signalCount = 4,
    };
    auto bootstrap = std::make_shared<MpiBootstrap>();
    auto transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, numRanks, bootstrap, config);
    transport->exchange();
    return transport;
  }
};

TEST_F(MultiPeerNvlTransportFixture, NumPeers) {
  auto transport = createTransport();

  EXPECT_EQ(transport->numPeers(), numRanks - 1);
}

TEST_F(MultiPeerNvlTransportFixture, DeviceTransportPtrNonNull) {
  auto transport = createTransport();

  P2pNvlTransportDevice* gpuPtr = transport->getDeviceTransportPtr();
  ASSERT_NE(gpuPtr, nullptr);
}

TEST_F(MultiPeerNvlTransportFixture, DeviceTransportPtrMatchesByValue) {
  auto transport = createTransport();

  const int nPeers = transport->numPeers();
  const std::size_t totalSize = nPeers * sizeof(P2pNvlTransportDevice);
  std::vector<char> rawBuf(totalSize);
  CUDACHECK_TEST(cudaMemcpy(
      rawBuf.data(),
      transport->getDeviceTransportPtr(),
      totalSize,
      cudaMemcpyDeviceToHost));

  auto* fromGpu = reinterpret_cast<const P2pNvlTransportDevice*>(rawBuf.data());

  int idx = 0;
  for (int peerRank = 0; peerRank < numRanks; ++peerRank) {
    if (peerRank == globalRank) {
      continue;
    }
    auto byValue = transport->getP2pTransportDevice(peerRank);

    // Compare LocalState: canonicalize both into zeroed buffers so
    // struct padding bytes are zero, then memcmp.
    alignas(LocalState) char gpuLocal[sizeof(LocalState)]{};
    alignas(LocalState) char valLocal[sizeof(LocalState)]{};
    canonicalizeState(fromGpu[idx].getLocalState(), gpuLocal);
    canonicalizeState(byValue.getLocalState(), valLocal);
    EXPECT_EQ(std::memcmp(gpuLocal, valLocal, sizeof(LocalState)), 0)
        << "LocalState mismatch at peer index " << idx
        << " (peerRank=" << peerRank << ")";

    // Compare RemoteState
    alignas(RemoteState) char gpuRemote[sizeof(RemoteState)]{};
    alignas(RemoteState) char valRemote[sizeof(RemoteState)]{};
    canonicalizeState(fromGpu[idx].getRemoteState(), gpuRemote);
    canonicalizeState(byValue.getRemoteState(), valRemote);
    EXPECT_EQ(std::memcmp(gpuRemote, valRemote, sizeof(RemoteState)), 0)
        << "RemoteState mismatch at peer index " << idx
        << " (peerRank=" << peerRank << ")";

    ++idx;
  }
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto mpi_env = std::make_unique<MPIEnvironmentBase>();
  ::testing::AddGlobalTestEnvironment(mpi_env.get());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
