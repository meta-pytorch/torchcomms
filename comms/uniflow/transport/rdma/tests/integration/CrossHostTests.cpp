// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/// Cross-host integration test for RDMA transport.
/// Requires MPI with 2 ranks on 2 different hosts (nnodes=2, ppn=1).
/// Each rank creates an RdmaTransport on its local NIC(s) and connects
/// to the peer via MPI-exchanged connection info.
/// GPU tests require at least 1 CUDA device per host.

#include <dirent.h>
#include <mpi.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/rdma/RdmaTransport.h"

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace uniflow {

/// Friend-class wrapper to construct RegisteredSegment /
/// RemoteRegisteredSegment with handles for testing. The name must be exactly
/// "SegmentTest" to match the friend declaration in Segment.h.
class SegmentTest {
 public:
  static RegisteredSegment makeRegistered(
      Segment& segment,
      std::unique_ptr<RegistrationHandle> handle) {
    RegisteredSegment reg(segment);
    reg.handles_.push_back(std::move(handle));
    return reg;
  }

  static RemoteRegisteredSegment makeRemote(
      void* buf,
      size_t len,
      std::unique_ptr<RemoteRegistrationHandle> handle) {
    RemoteRegisteredSegment remote(buf, len);
    remote.handles_.push_back(std::move(handle));
    return remote;
  }
};

class CompletionHoldingIbvApi final : public IbvApi {
 public:
  Status postSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** badWr) override {
    postCount_.fetch_add(1, std::memory_order_relaxed);
    if (holdNextPost_.exchange(false, std::memory_order_acq_rel)) {
      auto* completionWr = wr;
      while (completionWr->next != nullptr) {
        completionWr = completionWr->next;
      }
      std::lock_guard lock(mutex_);
      heldWrId_ = completionWr->wr_id;
    }
    return IbvApi::postSend(qp, wr, badWr);
  }

  Result<int> pollCq(ibv_cq* cq, int numEntries, ibv_wc* wcs) override {
    pollCount_.fetch_add(1, std::memory_order_relaxed);
    auto result = IbvApi::pollCq(cq, numEntries, wcs);
    if (result.hasError()) {
      return result;
    }

    std::lock_guard lock(mutex_);
    int visible = 0;
    for (int i = 0; i < result.value(); ++i) {
      if (heldWrId_ && wcs[i].wr_id == *heldWrId_) {
        heldCompletion_ = wcs[i];
        heldWrId_.reset();
        completionCv_.notify_all();
      } else {
        wcs[visible++] = wcs[i];
      }
    }
    return visible;
  }

  void holdNextPostCompletion() {
    std::lock_guard lock(mutex_);
    heldCompletion_.reset();
    holdNextPost_.store(true, std::memory_order_release);
  }

  bool waitForCompletionHeld(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return completionCv_.wait_for(
        lock, timeout, [this] { return heldCompletion_.has_value(); });
  }

  bool heldCompletionSucceeded() {
    std::lock_guard lock(mutex_);
    return heldCompletion_ && heldCompletion_->status == IBV_WC_SUCCESS;
  }

  uint64_t pollCount() const {
    return pollCount_.load(std::memory_order_relaxed);
  }
  uint64_t postCount() const {
    return postCount_.load(std::memory_order_relaxed);
  }

 private:
  std::mutex mutex_;
  std::condition_variable completionCv_;
  std::atomic<bool> holdNextPost_{false};
  std::atomic<uint64_t> pollCount_{0};
  std::atomic<uint64_t> postCount_{0};
  std::optional<uint64_t> heldWrId_;
  std::optional<ibv_wc> heldCompletion_;
};

static void boundedMpiBarrier(std::chrono::seconds timeout) {
  MPI_Request request;
  if (MPI_Ibarrier(MPI_COMM_WORLD, &request) != MPI_SUCCESS) {
    ADD_FAILURE() << "MPI_Ibarrier failed";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return;
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  int complete = 0;
  while (!complete && std::chrono::steady_clock::now() < deadline) {
    if (MPI_Test(&request, &complete, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
      ADD_FAILURE() << "MPI_Test failed";
      MPI_Abort(MPI_COMM_WORLD, 1);
      return;
    }
    std::this_thread::yield();
  }
  if (!complete) {
    ADD_FAILURE() << "MPI barrier timed out";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

/// Exchange a variable-length byte vector between rank 0 and rank 1 via MPI.
/// Each rank sends its own data and receives the peer's data.
static std::vector<uint8_t> mpiExchange(
    const std::vector<uint8_t>& localData,
    int rank) {
  int peerRank = 1 - rank;

  // Exchange sizes first.
  int localSize = static_cast<int>(localData.size());
  int remoteSize = 0;
  MPI_Sendrecv(
      &localSize,
      1,
      MPI_INT,
      peerRank,
      0,
      &remoteSize,
      1,
      MPI_INT,
      peerRank,
      0,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  // Exchange payload.
  std::vector<uint8_t> remoteData(remoteSize);
  MPI_Sendrecv(
      localData.data(),
      localSize,
      MPI_BYTE,
      peerRank,
      1,
      remoteData.data(),
      remoteSize,
      MPI_BYTE,
      peerRank,
      1,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  return remoteData;
}

/// Exchange a uint64_t value between rank 0 and rank 1 via MPI.
static uint64_t mpiExchangeAddr(uint64_t localVal, int rank) {
  int peerRank = 1 - rank;
  uint64_t remoteVal = 0;
  MPI_Sendrecv(
      &localVal,
      1,
      MPI_UINT64_T,
      peerRank,
      2,
      &remoteVal,
      1,
      MPI_UINT64_T,
      peerRank,
      2,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);
  return remoteVal;
}

class CrossHostTest : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ASSERT_EQ(numRanks, 2) << "CrossHostTest requires exactly 2 MPI ranks";

    ibvApi_ = makeIbvApi();
    auto initStatus = ibvApi_->init();
    ASSERT_FALSE(initStatus.hasError())
        << "Failed to init IbvApi: " << initStatus.error().message();

    int nDev = 0;
    auto devResult = ibvApi_->getDeviceList(&nDev);
    ASSERT_TRUE(devResult.hasValue())
        << "Failed to get device list: " << devResult.error().message();
    deviceList_ = devResult.value();
    ASSERT_GE(nDev, 1) << "Need at least 1 RDMA device, found " << nDev;

    // Only use beth (backend ethernet) NICs — they are on the same backend
    // fabric and can reach each other across hosts.
    for (int i = 0; i < nDev; ++i) {
      auto nameResult = ibvApi_->getDeviceName(deviceList_[i]);
      ASSERT_TRUE(nameResult.hasValue());
      if (isBethNic(nameResult.value())) {
        deviceNames_.emplace_back(nameResult.value());
      }
    }
    numDevices_ = deviceNames_.size();
    ASSERT_GE(numDevices_, 1) << "Need at least 1 beth NIC, found 0";

    evbThread_ = std::make_unique<ScopedEventBaseThread>();
  }

  void TearDown() override {
    evbThread_.reset();
    if (deviceList_) {
      ibvApi_->freeDeviceList(deviceList_);
    }
    MpiBaseTestFixture::TearDown();
  }

  /// Check if an RDMA device's netdev name starts with "beth" (backend
  /// ethernet) by reading /sys/class/infiniband/<dev>/device/net/.
  static bool isBethNic(const std::string& devName) {
    std::string netdevDir = "/sys/class/infiniband/" + devName + "/device/net/";
    DIR* dir = opendir(netdevDir.c_str());
    if (!dir) {
      return false;
    }
    bool found = false;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
      std::string name = entry->d_name;
      if (name.rfind("beth", 0) == 0) {
        found = true;
        break;
      }
    }
    closedir(dir);
    return found;
  }

  struct ConnectedPair {
    std::unique_ptr<RdmaTransportFactory> factory;
    std::unique_ptr<Transport> transport;
  };

  struct SegmentRegistration {
    RegisteredSegment local;
    RemoteRegisteredSegment remote;
  };

  virtual std::shared_ptr<IbvApi> makeIbvApi() {
    return std::make_shared<IbvApi>();
  }

  /// Register a local segment, exchange registration payloads via MPI,
  /// import the remote segment, and return both registered segments.
  std::optional<SegmentRegistration> registerAndExchangeSegments(
      RdmaTransportFactory& factory,
      void* buf,
      size_t totalSize,
      MemoryType memType,
      int deviceId = -1) {
    Segment seg(buf, totalSize, memType, deviceId);
    auto regResult = factory.registerSegment(seg);
    EXPECT_TRUE(regResult.hasValue()) << regResult.error().message();
    if (regResult.hasError()) {
      return std::nullopt;
    }

    auto localPayload = regResult.value()->serialize();
    auto remotePayload = mpiExchange(localPayload, globalRank);

    auto remoteHandle = factory.importSegment(totalSize, remotePayload);
    EXPECT_TRUE(remoteHandle.hasValue()) << remoteHandle.error().message();
    if (remoteHandle.hasError()) {
      return std::nullopt;
    }

    auto localReg =
        SegmentTest::makeRegistered(seg, std::move(regResult.value()));

    uint64_t localAddr = reinterpret_cast<uint64_t>(buf);
    uint64_t remoteAddr = mpiExchangeAddr(localAddr, globalRank);

    auto remoteReg = SegmentTest::makeRemote(
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<void*>(remoteAddr),
        totalSize,
        std::move(remoteHandle.value()));

    return SegmentRegistration{std::move(localReg), std::move(remoteReg)};
  }

  /// Build a vector of TransferRequests, one per chunk of bufSize.
  static std::vector<TransferRequest> buildTransferRequests(
      RegisteredSegment& local,
      RemoteRegisteredSegment& remote,
      size_t bufSize,
      size_t numRequests) {
    std::vector<TransferRequest> reqs;
    reqs.reserve(numRequests);
    for (size_t r = 0; r < numRequests; ++r) {
      reqs.push_back(
          TransferRequest{
              .local = local.span(r * bufSize, bufSize),
              .remote = remote.span(r * bufSize, bufSize),
          });
    }
    return reqs;
  }

  /// Create a connected transport pair across hosts using MPI to exchange
  /// topology and connection info.
  /// @param numNics  Number of beth NICs to use for this rank's factory.
  /// @param config   RdmaTransportConfig (numQps, etc.).
  ConnectedPair connectCrossHost(
      size_t numNics = 1,
      RdmaTransportConfig config = {}) {
    EXPECT_GE(deviceNames_.size(), numNics)
        << "Need at least " << numNics << " beth NICs, found "
        << deviceNames_.size();

    std::vector<std::string> nicVec(
        deviceNames_.begin(), deviceNames_.begin() + numNics);

    ConnectedPair pair;
    auto* evb = evbThread_->getEventBase();

    pair.factory =
        std::make_unique<RdmaTransportFactory>(nicVec, evb, config, ibvApi_);
    pair.transport = connectTransport(*pair.factory);
    return pair;
  }

  /// Create a transport from `factory` and connect it to the peer's, using MPI
  /// to exchange topology and connection info.
  std::unique_ptr<Transport> connectTransport(RdmaTransportFactory& factory) {
    auto localTopo = factory.getTopology();
    auto remoteTopo = mpiExchange(localTopo, globalRank);

    auto transportResult = factory.createTransport(remoteTopo);
    EXPECT_TRUE(transportResult.hasValue())
        << "createTransport failed: " << transportResult.error().message();
    auto transport = std::move(transportResult.value());

    auto localInfo = transport->bind();
    auto remoteInfo = mpiExchange(localInfo, globalRank);
    auto connectStatus = transport->connect(remoteInfo);
    EXPECT_FALSE(connectStatus.hasError())
        << "connect failed: " << connectStatus.error().message();
    return transport;
  }

  std::shared_ptr<IbvApi> ibvApi_;
  ibv_device** deviceList_{nullptr};
  size_t numDevices_{0};
  std::vector<std::string> deviceNames_;
  std::unique_ptr<ScopedEventBaseThread> evbThread_;
};

// --- Connection test ---

TEST_F(CrossHostTest, TransportsConnectAcrossHosts) {
  auto pair = connectCrossHost();
  EXPECT_EQ(pair.transport->state(), TransportState::Connected);

  pair.transport->shutdown();
  EXPECT_EQ(pair.transport->state(), TransportState::Disconnected);
}

enum class HeldCompletionOp { Put, Get };

class TerminalTimeoutCrossHostTest
    : public CrossHostTest,
      public ::testing::WithParamInterface<HeldCompletionOp> {
 protected:
  std::shared_ptr<IbvApi> makeIbvApi() override {
    completionApi_ = std::make_shared<CompletionHoldingIbvApi>();
    return completionApi_;
  }

  std::shared_ptr<CompletionHoldingIbvApi> completionApi_;
};

// Verifies that withholding a real PUT or GET CQE fires the configured timeout,
// stops verbs work on that transport, and a new transport from the same
// factory then completes a PUT.
TEST_P(
    TerminalTimeoutCrossHostTest,
    HeldCompletionTimesOutThenNewTransportWorks) {
  using namespace std::chrono_literals;
  constexpr size_t kTransferSize = 4096;
  constexpr auto kWaitTimeout = 5s;

  RdmaTransportConfig config;
  config.requestTimeout = 1s;
  auto pair = connectCrossHost(/*numNics=*/1, config);
  std::vector<char> localBuf(kTransferSize, 0);
  auto segments = registerAndExchangeSegments(
      *pair.factory, localBuf.data(), localBuf.size(), MemoryType::DRAM);
  if (!segments) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return;
  }
  const std::vector<TransferRequest> requests = {{
      .local = segments->local.span(size_t{0}, kTransferSize),
      .remote = segments->remote.span(size_t{0}, kTransferSize),
  }};
  boundedMpiBarrier(15s);

  uint64_t terminalPollCount = 0;
  uint64_t terminalPostCount = 0;
  if (globalRank == 0) {
    completionApi_->holdNextPostCompletion();
    auto future = GetParam() == HeldCompletionOp::Put
        ? pair.transport->put(requests)
        : pair.transport->get(requests);

    if (!completionApi_->waitForCompletionHeld(kWaitTimeout)) {
      ADD_FAILURE() << "failed to retain the selected CQE";
      MPI_Abort(MPI_COMM_WORLD, 1);
      return;
    }
    EXPECT_TRUE(completionApi_->heldCompletionSucceeded());
    if (future.wait_for(kWaitTimeout) != std::future_status::ready) {
      ADD_FAILURE() << "operation did not reach its terminal deadline";
      MPI_Abort(MPI_COMM_WORLD, 1);
      return;
    }

    const auto status = future.get();
    if (!status.hasError()) {
      ADD_FAILURE() << "operation completed without a terminal error";
      MPI_Abort(MPI_COMM_WORLD, 1);
      return;
    }
    EXPECT_EQ(status.error().code(), ErrCode::Timeout);
    terminalPollCount = completionApi_->pollCount();
    terminalPostCount = completionApi_->postCount();

    auto rejected = GetParam() == HeldCompletionOp::Put
        ? pair.transport->put(requests)
        : pair.transport->get(requests);
    if (rejected.wait_for(kWaitTimeout) != std::future_status::ready) {
      ADD_FAILURE() << "post-timeout operation did not complete";
      MPI_Abort(MPI_COMM_WORLD, 1);
      return;
    }
    const auto rejectedStatus = rejected.get();
    if (!rejectedStatus.hasError()) {
      ADD_FAILURE() << "post-timeout operation was accepted";
      MPI_Abort(MPI_COMM_WORLD, 1);
      return;
    }
    EXPECT_EQ(rejectedStatus.error().code(), ErrCode::Aborted);
  }

  boundedMpiBarrier(15s);
  pair.transport->shutdown();
  EXPECT_EQ(pair.transport->state(), TransportState::Disconnected);
  pair.transport.reset();
  EXPECT_EQ(
      pair.factory->counters()->timeoutsFired.load(),
      uint64_t{globalRank == 0 ? 1u : 0u});
  if (globalRank == 0) {
    EXPECT_EQ(completionApi_->pollCount(), terminalPollCount);
    EXPECT_EQ(completionApi_->postCount(), terminalPostCount);
  }

  auto replacement = connectTransport(*pair.factory);
  boundedMpiBarrier(15s);
  if (globalRank == 0) {
    auto healthy = replacement->put(requests);
    if (healthy.wait_for(kWaitTimeout) != std::future_status::ready) {
      ADD_FAILURE() << "PUT on the new transport did not complete";
      MPI_Abort(MPI_COMM_WORLD, 1);
      return;
    }
    const auto healthyStatus = healthy.get();
    EXPECT_FALSE(healthyStatus.hasError()) << healthyStatus.error().message();
  }

  boundedMpiBarrier(15s);
  replacement->shutdown();
  EXPECT_EQ(replacement->state(), TransportState::Disconnected);
  evbThread_.reset();
  replacement.reset();
  segments.reset();
  pair.factory.reset();
}

INSTANTIATE_TEST_SUITE_P(
    PostedOperation,
    TerminalTimeoutCrossHostTest,
    ::testing::Values(HeldCompletionOp::Put, HeldCompletionOp::Get),
    [](const ::testing::TestParamInfo<HeldCompletionOp>& info) {
      return info.param == HeldCompletionOp::Put ? "Put" : "Get";
    });

// --- Parameterized transfer tests ---

struct CrossHostTransferParam {
  size_t bufSize;
  size_t numRequests;
  size_t numNicsRank0;
  size_t numNicsRank1;
  size_t numQps;
  std::string name;
};

std::string crossHostParamName(
    const ::testing::TestParamInfo<CrossHostTransferParam>& info) {
  return info.param.name;
}

// Returns true if any rank wants to skip (synchronized across all ranks).
bool anyRankWantsToSkip(bool localSkip) {
  int localVal = localSkip ? 1 : 0;
  int globalVal = 0;
  MPI_Allreduce(&localVal, &globalVal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return globalVal != 0;
}

// --- Parameterized DRAM put/get tests ---

class DramCrossHostTransferTest
    : public CrossHostTest,
      public ::testing::WithParamInterface<CrossHostTransferParam> {};

TEST_P(DramCrossHostTransferTest, Put) {
  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  const size_t numNics =
      (globalRank == 0) ? param.numNicsRank0 : param.numNicsRank1;

  if (anyRankWantsToSkip(numDevices_ < numNics)) {
    GTEST_SKIP() << "Some rank lacks sufficient NICs (local: " << numDevices_
                 << ", needed: " << numNics << ")";
  }

  RdmaTransportConfig config;
  config.numQps = static_cast<uint32_t>(param.numQps);
  auto pair = connectCrossHost(numNics, config);

  // Rank 0 is sender, rank 1 is receiver.
  std::vector<char> localBuf(totalSize);
  if (globalRank == 0) {
    for (size_t r = 0; r < numRequests; ++r) {
      std::memset(
          localBuf.data() + r * bufSize, static_cast<int>(0xA0 + r), bufSize);
    }
  } else {
    std::memset(localBuf.data(), 0, totalSize);
  }

  auto segments = registerAndExchangeSegments(
      *pair.factory, localBuf.data(), totalSize, MemoryType::DRAM);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    auto reqs = buildTransferRequests(
        segments->local, segments->remote, bufSize, numRequests);
    auto putStatus = pair.transport->put(reqs, {}).get();
    ASSERT_FALSE(putStatus.hasError())
        << "put failed: " << putStatus.error().message();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == 1) {
    for (size_t r = 0; r < numRequests; ++r) {
      uint8_t expected = static_cast<uint8_t>(0xA0 + r);
      for (size_t i = 0; i < bufSize; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(localBuf[r * bufSize + i]), expected)
            << "Data mismatch at request " << r << " byte " << i;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_P(DramCrossHostTransferTest, Get) {
  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  const size_t numNics =
      (globalRank == 0) ? param.numNicsRank0 : param.numNicsRank1;

  if (anyRankWantsToSkip(numDevices_ < numNics)) {
    GTEST_SKIP() << "Some rank lacks sufficient NICs (local: " << numDevices_
                 << ", needed: " << numNics << ")";
  }

  RdmaTransportConfig config;
  config.numQps = static_cast<uint32_t>(param.numQps);
  auto pair = connectCrossHost(numNics, config);

  // Rank 0 is reader (zeroed), rank 1 is source (filled).
  std::vector<char> localBuf(totalSize);
  if (globalRank == 0) {
    std::memset(localBuf.data(), 0, totalSize);
  } else {
    for (size_t r = 0; r < numRequests; ++r) {
      std::memset(
          localBuf.data() + r * bufSize, static_cast<int>(0xB0 + r), bufSize);
    }
  }

  auto segments = registerAndExchangeSegments(
      *pair.factory, localBuf.data(), totalSize, MemoryType::DRAM);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    auto reqs = buildTransferRequests(
        segments->local, segments->remote, bufSize, numRequests);
    auto getStatus = pair.transport->get(reqs, {}).get();
    ASSERT_FALSE(getStatus.hasError())
        << "get failed: " << getStatus.error().message();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == 0) {
    for (size_t r = 0; r < numRequests; ++r) {
      uint8_t expected = static_cast<uint8_t>(0xB0 + r);
      for (size_t i = 0; i < bufSize; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(localBuf[r * bufSize + i]), expected)
            << "Data mismatch at request " << r << " byte " << i;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

const size_t kLargeBufferSize = 12 * 1024 * 1024 + 12 * 1024; // 12MB + 12KB

INSTANTIATE_TEST_SUITE_P(
    DramCrossHostTransfer,
    DramCrossHostTransferTest,
    ::testing::Values(
        // Single NIC, single QP, varying buffer sizes and request counts.
        CrossHostTransferParam{4096, 1, 1, 1, 1, "4KB_single_req_1nic_1qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            1,
            1,
            1,
            "12MB12KB_single_req_1nic_1qp"},
        CrossHostTransferParam{
            1024 * 1024 * 1024,
            1,
            1,
            1,
            1,
            "1G_single_req_1nic_1qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            1,
            1,
            1,
            "12MB12KB_batch_req_1nic_1qp"},
        // Multi-NIC symmetric: both ranks use 2 NICs, 2 QPs.
        CrossHostTransferParam{4096, 1, 2, 2, 2, "4KB_single_req_2nic_2qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            2,
            2,
            2,
            "12MB12KB_single_req_2nic_2qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            2,
            2,
            2,
            "12MB12KB_batch_req_2nic_2qp"},
        // Asymmetric NICs: rank 0 uses 1 NIC, rank 1 uses 2 NICs, same QPs.
        CrossHostTransferParam{4096, 1, 1, 2, 2, "4KB_single_req_1v2nic_2qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            3,
            4,
            20,
            "12MB12KB_single_req_3v4nic_20qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            3,
            4,
            20,
            "12MB12KB_batch_req_3v4nic_20qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            3,
            7,
            20,
            "12MB12KB_single_req_3v7nic_20qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            3,
            7,
            20,
            "12MB12KB_batch_req_3v7nic_20qp"}),
    crossHostParamName);

// --- Parameterized GPU put/get tests ---

struct CudaBuffer {
  void* ptr{nullptr};
  size_t size{0};

  explicit CudaBuffer(size_t n, int device = 0) : size(n) {
    (void)cudaSetDevice(device);
    if (cudaMalloc(&ptr, n) != cudaSuccess) {
      ptr = nullptr;
    }
  }

  ~CudaBuffer() {
    if (ptr) {
      (void)cudaFree(ptr);
    }
  }

  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
};

class GpuCrossHostTransferTest
    : public CrossHostTest,
      public ::testing::WithParamInterface<CrossHostTransferParam> {};

TEST_P(GpuCrossHostTransferTest, Put) {
  int deviceCount = 0;
  bool noCuda =
      cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 1;
  if (anyRankWantsToSkip(noCuda)) {
    GTEST_SKIP() << "Some rank lacks CUDA devices (local: " << deviceCount
                 << ")";
  }

  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  const size_t numNics =
      (globalRank == 0) ? param.numNicsRank0 : param.numNicsRank1;

  if (anyRankWantsToSkip(numDevices_ < numNics)) {
    GTEST_SKIP() << "Some rank lacks sufficient NICs (local: " << numDevices_
                 << ", needed: " << numNics << ")";
  }

  // Rank 0 uses cuda:0, rank 1 uses cuda:1.
  const int cudaDev = globalRank;
  if (anyRankWantsToSkip(cudaDev >= deviceCount)) {
    GTEST_SKIP() << "Some rank lacks required CUDA device (need device "
                 << cudaDev << ", have " << deviceCount << ")";
  }

  RdmaTransportConfig config;
  config.numQps = static_cast<uint32_t>(param.numQps);
  auto pair = connectCrossHost(numNics, config);

  CudaBuffer gpuBuf(totalSize, cudaDev);
  ASSERT_NE(gpuBuf.ptr, nullptr) << "cudaMalloc failed on device " << cudaDev;

  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess) << "cudaSetDevice failed";
  if (globalRank == 0) {
    std::vector<char> staging(totalSize);
    for (size_t r = 0; r < numRequests; ++r) {
      std::memset(
          staging.data() + r * bufSize, static_cast<int>(0xC0 + r), bufSize);
    }
    ASSERT_EQ(
        cudaMemcpy(
            gpuBuf.ptr, staging.data(), totalSize, cudaMemcpyHostToDevice),
        cudaSuccess)
        << "cudaMemcpy failed";
  } else {
    ASSERT_EQ(cudaMemset(gpuBuf.ptr, 0, totalSize), cudaSuccess)
        << "cudaMemset failed";
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
      << "cudaDeviceSynchronize failed";

  auto segments = registerAndExchangeSegments(
      *pair.factory, gpuBuf.ptr, totalSize, MemoryType::VRAM, cudaDev);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    auto reqs = buildTransferRequests(
        segments->local, segments->remote, bufSize, numRequests);
    auto putStatus = pair.transport->put(reqs, {}).get();
    ASSERT_FALSE(putStatus.hasError())
        << "GPU put failed: " << putStatus.error().message();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == 1) {
    std::vector<char> verify(totalSize, 0);
    ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess) << "cudaSetDevice failed";
    ASSERT_EQ(
        cudaMemcpy(
            verify.data(), gpuBuf.ptr, totalSize, cudaMemcpyDeviceToHost),
        cudaSuccess)
        << "cudaMemcpy failed";
    for (size_t r = 0; r < numRequests; ++r) {
      uint8_t expected = static_cast<uint8_t>(0xC0 + r);
      for (size_t i = 0; i < bufSize; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(verify[r * bufSize + i]), expected)
            << "GPU data mismatch at request " << r << " byte " << i;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_P(GpuCrossHostTransferTest, Get) {
  int deviceCount = 0;
  bool noCuda =
      cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 1;
  if (anyRankWantsToSkip(noCuda)) {
    GTEST_SKIP() << "Some rank lacks CUDA devices (local: " << deviceCount
                 << ")";
  }

  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  const size_t numNics =
      (globalRank == 0) ? param.numNicsRank0 : param.numNicsRank1;

  if (anyRankWantsToSkip(numDevices_ < numNics)) {
    GTEST_SKIP() << "Some rank lacks sufficient NICs (local: " << numDevices_
                 << ", needed: " << numNics << ")";
  }

  // Rank 0 uses cuda:0, rank 1 uses cuda:1.
  const int cudaDev = globalRank;
  if (anyRankWantsToSkip(cudaDev >= deviceCount)) {
    GTEST_SKIP() << "Some rank lacks required CUDA device (need device "
                 << cudaDev << ", have " << deviceCount << ")";
  }

  RdmaTransportConfig config;
  config.numQps = static_cast<uint32_t>(param.numQps);
  auto pair = connectCrossHost(numNics, config);

  CudaBuffer gpuBuf(totalSize, cudaDev);
  ASSERT_NE(gpuBuf.ptr, nullptr) << "cudaMalloc failed on device " << cudaDev;

  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess) << "cudaSetDevice failed";
  if (globalRank == 0) {
    ASSERT_EQ(cudaMemset(gpuBuf.ptr, 0, totalSize), cudaSuccess)
        << "cudaMemset failed";
  } else {
    std::vector<char> staging(totalSize);
    for (size_t r = 0; r < numRequests; ++r) {
      std::memset(
          staging.data() + r * bufSize, static_cast<int>(0xD0 + r), bufSize);
    }
    ASSERT_EQ(
        cudaMemcpy(
            gpuBuf.ptr, staging.data(), totalSize, cudaMemcpyHostToDevice),
        cudaSuccess)
        << "cudaMemcpy failed";
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
      << "cudaDeviceSynchronize failed";

  auto segments = registerAndExchangeSegments(
      *pair.factory, gpuBuf.ptr, totalSize, MemoryType::VRAM, cudaDev);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    auto reqs = buildTransferRequests(
        segments->local, segments->remote, bufSize, numRequests);
    auto getStatus = pair.transport->get(reqs, {}).get();
    ASSERT_FALSE(getStatus.hasError())
        << "GPU get failed: " << getStatus.error().message();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == 0) {
    std::vector<char> verify(totalSize, 0);
    ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess) << "cudaSetDevice failed";
    ASSERT_EQ(
        cudaMemcpy(
            verify.data(), gpuBuf.ptr, totalSize, cudaMemcpyDeviceToHost),
        cudaSuccess)
        << "cudaMemcpy failed";
    for (size_t r = 0; r < numRequests; ++r) {
      uint8_t expected = static_cast<uint8_t>(0xD0 + r);
      for (size_t i = 0; i < bufSize; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(verify[r * bufSize + i]), expected)
            << "GPU data mismatch at request " << r << " byte " << i;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

INSTANTIATE_TEST_SUITE_P(
    GpuCrossHostTransfer,
    GpuCrossHostTransferTest,
    ::testing::Values(
        // Single NIC, single QP.
        CrossHostTransferParam{4096, 1, 1, 1, 1, "4KB_single_req_1nic_1qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            1,
            1,
            1,
            "12MB12KB_single_req_1nic_1qp"},
        CrossHostTransferParam{
            1024 * 1024 * 1024,
            1,
            1,
            1,
            1,
            "1G_single_req_1nic_1qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            1,
            1,
            1,
            "12MB12KB_batch_req_1nic_1qp"},
        // Multi-NIC symmetric: both ranks use 2 NICs, 2 QPs.
        CrossHostTransferParam{4096, 1, 2, 2, 2, "4KB_single_req_2nic_2qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            2,
            2,
            2,
            "12MB12KB_single_req_2nic_2qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            2,
            2,
            2,
            "12MB12KB_batch_req_2nic_2qp"},
        // Asymmetric NICs: rank 0 uses 1 NIC, rank 1 uses 2 NICs, same QPs.
        CrossHostTransferParam{4096, 1, 1, 2, 2, "4KB_single_req_1v2nic_2qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            3,
            4,
            20,
            "12MB12KB_single_req_3v4nic_20qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            3,
            4,
            20,
            "12MB12KB_batch_req_3v4nic_20qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            3,
            7,
            20,
            "12MB12KB_single_req_3v7nic_20qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            3,
            7,
            20,
            "12MB12KB_batch_req_3v7nic_20qp"}),
    crossHostParamName);

} // namespace uniflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase());
  return RUN_ALL_TESTS();
}
