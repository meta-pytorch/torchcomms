// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/futures/Future.h>
#include <folly/init/Init.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comms/common/bootstrap/tests/MockBootstrap.h"
#include "comms/prims/core/SignalState.cuh"
#include "comms/prims/memory/MultimemHandler.h"
#include "comms/prims/tests/MultimemNvlTransportTest.cuh"
#include "comms/prims/transport/nvl/MultimemNvlTransport.h"
#include "comms/testinfra/DistEnvironmentBase.h"
#include "comms/testinfra/DistTestBase.h"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::prims::tests {

namespace {

std::shared_ptr<meta::comms::IBootstrap> makeBootstrap(
    const std::string& prefix) {
  return std::shared_ptr<meta::comms::IBootstrap>(
      meta::comms::createBootstrap(prefix));
}

std::vector<int> identityRankMap(int size) {
  std::vector<int> rankMap(static_cast<std::size_t>(size));
  for (int rank = 0; rank < size; ++rank) {
    rankMap[static_cast<std::size_t>(rank)] = rank;
  }
  return rankMap;
}

// Collective check: returns true iff every rank reports that multimem is
// eligible on the given CUDA device. Tests use this to skip cleanly on
// non-NVLS hosts without leaving stragglers blocked in downstream collectives.
bool allRanksMultimemEligible(
    const std::shared_ptr<meta::comms::IBootstrap>& bootstrap,
    int rank,
    int nRanks,
    int cudaDevice) {
  std::vector<int> eligible(static_cast<std::size_t>(nRanks));
  eligible[static_cast<std::size_t>(rank)] =
      MultimemNvlTransport::isEligible(nRanks, cudaDevice) ? 1 : 0;
  auto rc =
      bootstrap->allGather(eligible.data(), sizeof(int), rank, nRanks).get();
  EXPECT_EQ(rc, 0);
  if (rc != 0) {
    return false;
  }
  for (const int v : eligible) {
    if (v == 0) {
      return false;
    }
  }
  return true;
}

MultimemNvlTransportConfig makeConfig(
    std::size_t dataBufferSize,
    uint32_t userSignalCount = 1,
    uint32_t internalSignalCount = 0) {
  MultimemNvlTransportConfig config{};
  config.dataBufferSize = dataBufferSize;
  config.userSignalCount = userSignalCount;
  config.internalSignalCount = internalSignalCount;
  return config;
}

using meta::comms::testing::MockBootstrap;
using StrictMockBootstrap = ::testing::StrictMock<MockBootstrap>;

// Builds a StrictMockBootstrap that, by default, forwards every IBootstrap
// API to `real`. Individual tests override specific methods via EXPECT_CALL
// to inject failures. StrictMock ensures that if MultimemNvlTransport (or
// its GpuMemHandler/MultimemHandler) ever grows a new bootstrap dependency
// we didn't account for, the test fails immediately with an "uninteresting
// mock function call" error -- locking down the bootstrap API surface.
std::shared_ptr<StrictMockBootstrap> makeDelegatingMock(
    const std::shared_ptr<meta::comms::IBootstrap>& real) {
  using ::testing::_;

  auto mock = std::make_shared<StrictMockBootstrap>();

  ON_CALL(*mock, allGather(_, _, _, _))
      .WillByDefault([real](void* buf, int len, int rank, int nRanks) {
        return real->allGather(buf, len, rank, nRanks);
      });
  ON_CALL(*mock, barrier(_, _)).WillByDefault([real](int rank, int nRanks) {
    return real->barrier(rank, nRanks);
  });
  ON_CALL(*mock, allGatherNvlDomain(_, _, _, _, _))
      .WillByDefault([real](
                         void* buf,
                         int len,
                         int r,
                         int n,
                         const std::vector<int>& rankMap) {
        return real->allGatherNvlDomain(buf, len, r, n, rankMap);
      });
  ON_CALL(*mock, barrierNvlDomain(_, _, _))
      .WillByDefault([real](int r, int n, const std::vector<int>& rankMap) {
        return real->barrierNvlDomain(r, n, rankMap);
      });
  ON_CALL(*mock, send(_, _, _, _))
      .WillByDefault([real](void* buf, int len, int peer, int tag) {
        return real->send(buf, len, peer, tag);
      });
  ON_CALL(*mock, recv(_, _, _, _))
      .WillByDefault([real](void* buf, int len, int peer, int tag) {
        return real->recv(buf, len, peer, tag);
      });

  return mock;
}

} // namespace

class MultimemNvlTransportTestFixture : public ::testing::Test,
                                        public meta::comms::DistBaseTest {
 protected:
  void SetUp() override {
    distSetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    distTearDown();
  }
};

TEST_F(
    MultimemNvlTransportTestFixture,
    EligibilityRequiresSupportAndThreeRanks) {
  EXPECT_FALSE(MultimemNvlTransport::isEligible(1, localRank));
  EXPECT_FALSE(MultimemNvlTransport::isEligible(2, localRank));
  EXPECT_EQ(
      MultimemNvlTransport::isEligible(3, localRank),
      MultimemHandler::isMultimemSupported(localRank));
}

// getDeviceTransport() must refuse to vend a handle before exchange() has
// installed the multicast overlay. Tested per-rank (no bootstrap needed for
// the throw path, so this runs even on hosts without NVLS).
TEST_F(
    MultimemNvlTransportTestFixture,
    GetDeviceTransportThrowsBeforeExchange) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_get_device_before_exchange");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(/*dataBufferSize=*/4096));
  EXPECT_THROW((void)transport.getDeviceTransport(), std::runtime_error);
}

// Happy-path exchange() from the primary (global-bootstrap +
// nvlRankToCommRank) constructor. Locks down the device handle shape:
// non-null local/multimem base pointers, distinct pointers, dataBufferSize
// echoed through, and user + internal signal spans sized to the requested
// slot counts. Also verifies the multimem pointer is stable across
// exchange() calls (idempotency) and that the two getAllocated* accessors
// report the configured / SignalState-aligned sizes.
TEST_F(MultimemNvlTransportTestFixture, ExchangeSetsUpDeviceHandle) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_exchange_happy_path");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr std::size_t kDataBytes = 8192;
  constexpr uint32_t kUserSignals = 2;
  constexpr uint32_t kInternalSignals = 3;

  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(kDataBytes, kUserSignals, kInternalSignals));

  transport.exchange();
  auto handle = transport.getDeviceTransport();

  EXPECT_NE(handle.localData, nullptr);
  EXPECT_NE(handle.multimemData, nullptr);
  EXPECT_NE(handle.localData, handle.multimemData)
      << "unicast and multicast VAs should be distinct";
  EXPECT_EQ(handle.dataBufferSize, kDataBytes);
  EXPECT_EQ(handle.userLocalSignals.size(), kUserSignals);
  EXPECT_EQ(handle.userMultimemSignals.size(), kUserSignals);
  EXPECT_EQ(handle.internalLocalSignals.size(), kInternalSignals);
  EXPECT_EQ(handle.internalMultimemSignals.size(), kInternalSignals);

  EXPECT_EQ(transport.getAllocatedDataBufferSize(), kDataBytes);
  EXPECT_EQ(
      transport.getAllocatedSignalBufferSize(),
      getSignalBufferSize(static_cast<int>(kUserSignals + kInternalSignals)));

  // Idempotency: a second exchange() must be a no-op.
  auto* firstMultimemBase = handle.multimemData;
  transport.exchange();
  auto handle2 = transport.getDeviceTransport();
  EXPECT_EQ(handle2.multimemData, firstMultimemBase);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// The user and internal signal spans must be disjoint contiguous regions,
// with the internal region starting immediately after the user region on
// both the local and the multimem sides. This is what lets the transport
// reserve internal slots for its own protocols without leaking into the
// user-visible SignalState indices.
TEST_F(MultimemNvlTransportTestFixture, UserAndInternalSignalSpansAreDisjoint) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_signal_span_layout");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr uint32_t kUserSignals = 4;
  constexpr uint32_t kInternalSignals = 2;

  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(/*dataBufferSize=*/4096, kUserSignals, kInternalSignals));
  transport.exchange();
  auto handle = transport.getDeviceTransport();

  // Internal region starts at userSignalCount elements past the user base,
  // on both mirrors.
  EXPECT_EQ(
      handle.internalLocalSignals.data(),
      handle.userLocalSignals.data() + kUserSignals);
  EXPECT_EQ(
      handle.internalMultimemSignals.data(),
      handle.userMultimemSignals.data() + kUserSignals);

  // User and internal regions do not overlap on either mirror.
  EXPECT_LE(
      handle.userLocalSignals.data() + handle.userLocalSignals.size(),
      handle.internalLocalSignals.data());
  EXPECT_LE(
      handle.userMultimemSignals.data() + handle.userMultimemSignals.size(),
      handle.internalMultimemSignals.data());

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// The compat constructor must also drive a real exchange to completion.
// Uses NVL-local (rank, size) coordinates against the same underlying NVL
// team.
TEST_F(MultimemNvlTransportTestFixture, CompatCtorExchangeSucceeds) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_compat_ctor_exchange");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  MultimemNvlTransport transport(
      /*nvlRank=*/globalRank,
      /*nvlRanks=*/numRanks,
      bootstrap,
      makeConfig(/*dataBufferSize=*/4096));
  ASSERT_NO_THROW(transport.exchange());
  auto handle = transport.getDeviceTransport();
  EXPECT_NE(handle.localData, nullptr);
  EXPECT_NE(handle.multimemData, nullptr);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// Poison-on-failure: inject a bootstrap failure symmetrically on every rank
// via a delegating StrictMock, verify exchange() throws, and verify a
// second exchange() on the same object also throws (the poisoned-object
// contract). The recovery path -- constructing a fresh MultimemNvlTransport
// and exchange()-ing successfully -- is proven in ExchangeSetsUpDeviceHandle
// (fresh transport per test) and by the underlying MultimemHandler failure
// tests, so we intentionally do not repeat the fresh-object recovery here.
TEST_F(MultimemNvlTransportTestFixture, ExchangePoisonsAfterFailure) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto real = makeBootstrap("mmnvl_exchange_poison");
  if (!allRanksMultimemEligible(real, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  using ::testing::_;
  using ::testing::AnyNumber;
  auto mock = makeDelegatingMock(real);
  // Inject failure on the very first NVL-domain allGather (agreeOnHandleType
  // inside MultimemHandler::exchange). Every rank throws symmetrically so
  // there's no need to shorten the store timeout.
  EXPECT_CALL(*mock, allGatherNvlDomain(_, _, _, _, _))
      .WillOnce([](void*, int, int, int, const std::vector<int>&) {
        return folly::makeSemiFuture(-1);
      });
  EXPECT_CALL(*mock, barrierNvlDomain(_, _, _)).Times(AnyNumber());

  MultimemNvlTransport transport(
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      globalRank,
      identityRankMap(numRanks),
      makeConfig(/*dataBufferSize=*/4096));

  EXPECT_THROW(transport.exchange(), std::runtime_error);

  // Second call must throw the poisoned-object error without touching the
  // bootstrap. The StrictMock has no further EXPECT_CALL, so any recovered
  // bootstrap call would fail the test.
  try {
    transport.exchange();
    FAIL() << "expected poisoned transport to throw on second exchange()";
  } catch (const std::runtime_error& ex) {
    EXPECT_NE(
        std::string(ex.what()).find("previous exchange() failed"),
        std::string::npos)
        << ex.what();
  }

  // getDeviceTransport must also refuse: the transport was never marked
  // exchanged.
  EXPECT_THROW((void)transport.getDeviceTransport(), std::runtime_error);

  ASSERT_EQ(real->barrier(globalRank, numRanks).get(), 0);
}

// -----------------------------------------------------------------------------
// Device signal API tests
// -----------------------------------------------------------------------------
// These tests launch the kernels declared in MultimemNvlTransportTest.cuh
// against a fully-exchanged transport. Each verifies one behavior of the
// device signal API end-to-end: multimem PTX store propagation, multimem
// atomic-add accumulation, user/internal span isolation, wait_until, and
// read_signal / read_internal_signal.

namespace {

// Device buffer for a single uint64_t output slot. Reset to a distinctive
// sentinel between tests so a missing kernel write is obvious.
struct DeviceUint64Slot {
  DeviceUint64Slot() {
    CUDACHECK_TEST(cudaMalloc(&ptr_, sizeof(uint64_t)));
    reset(kSentinel);
  }
  ~DeviceUint64Slot() {
    if (ptr_) {
      (void)cudaFree(ptr_);
    }
  }
  DeviceUint64Slot(const DeviceUint64Slot&) = delete;
  DeviceUint64Slot& operator=(const DeviceUint64Slot&) = delete;
  DeviceUint64Slot(DeviceUint64Slot&&) = delete;
  DeviceUint64Slot& operator=(DeviceUint64Slot&&) = delete;

  void reset(uint64_t v) {
    CUDACHECK_TEST(
        cudaMemcpy(ptr_, &v, sizeof(uint64_t), cudaMemcpyHostToDevice));
  }
  uint64_t read() const {
    uint64_t v = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&v, ptr_, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return v;
  }
  uint64_t* device_ptr() {
    return ptr_;
  }

  static constexpr uint64_t kSentinel = 0xDEADBEEFCAFEBABEULL;

 private:
  uint64_t* ptr_{nullptr};
};

// Convenience: construct + exchange a transport with the given signal
// counts. Skips the caller test if the NVL team isn't multimem-eligible.
std::unique_ptr<MultimemNvlTransport> makeExchangedTransport(
    const std::shared_ptr<meta::comms::IBootstrap>& bootstrap,
    int globalRank,
    int numRanks,
    int localRank,
    uint32_t userSignalCount,
    uint32_t internalSignalCount) {
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    return nullptr;
  }
  auto config = makeConfig(
      /*dataBufferSize=*/4096, userSignalCount, internalSignalCount);
  auto transport = std::make_unique<MultimemNvlTransport>(
      bootstrap, globalRank, identityRankMap(numRanks), config);
  transport->exchange();
  return transport;
}

} // namespace

// signal(SET) from rank 0 must broadcast a value to every rank's local
// signal state; every rank observes it through wait_signal_until +
// read_signal.
TEST_F(MultimemNvlTransportTestFixture, DeviceUserSignalSetBroadcasts) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_device_user_signal_set");
  auto transport = makeExchangedTransport(
      bootstrap,
      globalRank,
      numRanks,
      localRank,
      /*userSignalCount=*/1,
      /*internalSignalCount=*/0);
  if (!transport) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr uint64_t kSignalValue = 0xA5A50000ULL + 42;
  auto handle = transport->getDeviceTransport();

  if (globalRank == 0) {
    test::launchSetUserSignal(handle, /*signalId=*/0, kSignalValue);
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);

  DeviceUint64Slot out;
  test::launchWaitAndReadUserSignal(
      handle, /*signalId=*/0, CmpOp::CMP_EQ, kSignalValue, out.device_ptr());
  CUDACHECK_TEST(cudaDeviceSynchronize());
  EXPECT_EQ(out.read(), kSignalValue);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// signal(ADD) from every rank must accumulate atomically through the
// multimem VA. Every rank waits until the sum arrives, then reads it.
TEST_F(MultimemNvlTransportTestFixture, DeviceUserSignalAddAccumulates) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_device_user_signal_add");
  auto transport = makeExchangedTransport(
      bootstrap,
      globalRank,
      numRanks,
      localRank,
      /*userSignalCount=*/1,
      /*internalSignalCount=*/0);
  if (!transport) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  const uint64_t expected = static_cast<uint64_t>(numRanks);
  auto handle = transport->getDeviceTransport();

  test::launchAddUserSignal(handle, /*signalId=*/0, /*value=*/1);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  DeviceUint64Slot out;
  test::launchWaitAndReadUserSignal(
      handle, /*signalId=*/0, CmpOp::CMP_GE, expected, out.device_ptr());
  CUDACHECK_TEST(cudaDeviceSynchronize());
  EXPECT_EQ(out.read(), expected);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// signal_internal(SET) from rank 0 must reach every rank's internal span
// via wait_internal_signal_until + read_internal_signal, exercising the
// internal-signal path independently of the user path.
TEST_F(MultimemNvlTransportTestFixture, DeviceInternalSignalSetBroadcasts) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_device_internal_signal_set");
  auto transport = makeExchangedTransport(
      bootstrap,
      globalRank,
      numRanks,
      localRank,
      /*userSignalCount=*/1,
      /*internalSignalCount=*/1);
  if (!transport) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr uint64_t kValue = 0xC0DECAFE00000010ULL;
  auto handle = transport->getDeviceTransport();

  if (globalRank == 0) {
    test::launchSetInternalSignal(handle, /*signalId=*/0, kValue);
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);

  DeviceUint64Slot out;
  test::launchWaitAndReadInternalSignal(
      handle, /*signalId=*/0, CmpOp::CMP_EQ, kValue, out.device_ptr());
  CUDACHECK_TEST(cudaDeviceSynchronize());
  EXPECT_EQ(out.read(), kValue);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// signal_internal(ADD) accumulates on the internal span, disjoint from
// whatever the user span may be doing.
TEST_F(MultimemNvlTransportTestFixture, DeviceInternalSignalAddAccumulates) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_device_internal_signal_add");
  auto transport = makeExchangedTransport(
      bootstrap,
      globalRank,
      numRanks,
      localRank,
      /*userSignalCount=*/1,
      /*internalSignalCount=*/1);
  if (!transport) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  const uint64_t expected = static_cast<uint64_t>(numRanks);
  auto handle = transport->getDeviceTransport();

  test::launchAddInternalSignal(handle, /*signalId=*/0, /*value=*/1);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  DeviceUint64Slot out;
  test::launchWaitAndReadInternalSignal(
      handle, /*signalId=*/0, CmpOp::CMP_GE, expected, out.device_ptr());
  CUDACHECK_TEST(cudaDeviceSynchronize());
  EXPECT_EQ(out.read(), expected);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// User vs. internal signal spans are isolated in device memory: writing
// only the user slot must NOT be observable through the internal read
// path, and vice versa. Uses the same signalId (0) in both spans on
// purpose, so a bug that merged the two spans would surface as one write
// leaking into the other reader.
TEST_F(MultimemNvlTransportTestFixture, DeviceUserAndInternalSignalsIsolated) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_device_user_internal_isolation");
  auto transport = makeExchangedTransport(
      bootstrap,
      globalRank,
      numRanks,
      localRank,
      /*userSignalCount=*/1,
      /*internalSignalCount=*/1);
  if (!transport) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  // Distinct values so any aliasing between spans is obvious.
  constexpr uint64_t kUserValue = 0x11111111ULL;
  constexpr uint64_t kInternalValue = 0x22222222ULL;
  auto handle = transport->getDeviceTransport();

  if (globalRank == 0) {
    test::launchSetUserSignal(handle, /*signalId=*/0, kUserValue);
    test::launchSetInternalSignal(handle, /*signalId=*/0, kInternalValue);
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);

  // Each rank waits on both spans (broadcast from rank 0). Waiting on both
  // before reading ensures a stale sentinel doesn't slip through.
  DeviceUint64Slot userOut;
  test::launchWaitAndReadUserSignal(
      handle, /*signalId=*/0, CmpOp::CMP_EQ, kUserValue, userOut.device_ptr());
  CUDACHECK_TEST(cudaDeviceSynchronize());

  DeviceUint64Slot internalOut;
  test::launchWaitAndReadInternalSignal(
      handle,
      /*signalId=*/0,
      CmpOp::CMP_EQ,
      kInternalValue,
      internalOut.device_ptr());
  CUDACHECK_TEST(cudaDeviceSynchronize());

  EXPECT_EQ(userOut.read(), kUserValue);
  EXPECT_EQ(internalOut.read(), kInternalValue);

  // Cross-check via the no-wait reader that reads both spans in one shot.
  // Uses a 2-slot device buffer so the kernel writes out[0]=user and
  // out[1]=internal in a single launch.
  uint64_t* pairOut = nullptr;
  CUDACHECK_TEST(cudaMalloc(&pairOut, 2 * sizeof(uint64_t)));
  test::launchReadUserAndInternal(
      handle, /*userId=*/0, /*internalId=*/0, pairOut);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  uint64_t hostPair[2] = {0, 0};
  CUDACHECK_TEST(
      cudaMemcpy(hostPair, pairOut, sizeof(hostPair), cudaMemcpyDeviceToHost));
  EXPECT_EQ(hostPair[0], kUserValue);
  EXPECT_EQ(hostPair[1], kInternalValue);
  CUDACHECK_TEST(cudaFree(pairOut));

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  // folly::Init consumes glog/gflags argv before other initializers see them.
  folly::Init init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::DistEnvironmentBase());
  return RUN_ALL_TESTS();
}
