// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/transport/IP2pHostTransport.h"
#include "comms/ctran/transport/ib/HostTransportImpl.h"
#include "comms/ctran/transport/ib/HostZcTransport.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ctran::transport;
using ctran::transport::ib::HostZcTransport;
using ctran::transport::ib::impl::exportRecvBuf;
using ctran::transport::ib::impl::importRemoteInfo;

// Lightweight wrapper to simulate a RegElem so toIbRegElem() can hand
// the IB backend the underlying ibRegElem pointer in tests.
struct TestRegHdl {
  const void* buf{nullptr};
  const std::size_t len{0};
  void* ibRegElem{nullptr};
};

void* wrapIbRegElem(TestRegHdl& hdl, void* ibRegElem) {
  hdl.ibRegElem = ibRegElem;
  return &hdl;
}

// ═══════════════════════════════════════════════════════════
// Channel layout
// ═══════════════════════════════════════════════════════════

struct ChannelLayout {
  int numChannels{1};
  int slotsPerChannel{0}; // 0 ⇒ pure-ZC
};

inline int slotFor(const ChannelLayout& layout, int ch, int step) {
  return layout.slotsPerChannel == 0
      ? kNoStagingSlot
      : (ch * layout.slotsPerChannel + step % layout.slotsPerChannel);
}

inline int roundFor(const ChannelLayout& layout, int step) {
  return layout.slotsPerChannel == 0 ? 0 : (step / layout.slotsPerChannel);
}

// ═══════════════════════════════════════════════════════════
// Unified chunk-level driver — same body for ZC and CB.
// ═══════════════════════════════════════════════════════════

template <bool IsSend, typename Buf>
void runChunkLoop(
    IP2pHostTransport* transport,
    int numVcs,
    Buf* buf,
    size_t totalSize,
    const ChannelLayout& layout,
    void* localMrHdl,
    void* remoteMrHdl,
    const CtranIbRemoteAccessKey* remoteKey) {
  const bool isZc = layout.slotsPerChannel == 0;
  const size_t chunkSize = isZc ? totalSize : transport->chunkSize();

  auto sliceBegin = [&](int ch) {
    return (totalSize / layout.numChannels) * ch;
  };
  auto sliceEnd = [&](int ch) {
    return ch == layout.numChannels - 1 ? totalSize : sliceBegin(ch + 1);
  };
  auto stepsForCh = [&](int ch) {
    if (isZc) {
      return 1;
    }
    const size_t sz = sliceEnd(ch) - sliceBegin(ch);
    return static_cast<int>((sz + chunkSize - 1) / chunkSize);
  };

  std::vector<int> issued(layout.numChannels, 0);
  std::vector<int> drained(layout.numChannels, 0);
  std::vector<std::vector<ChunkRequest>> reqs(layout.numChannels);
  std::vector<int> totalSteps(layout.numChannels);
  for (int ch = 0; ch < layout.numChannels; ++ch) {
    totalSteps[ch] = stepsForCh(ch);
    reqs[ch].resize(totalSteps[ch]);
  }

  bool done = false;
  while (!done) {
    COMMCHECK_TEST(transport->progress());
    done = true;
    for (int ch = 0; ch < layout.numChannels; ++ch) {
      const int vcIdx = numVcs > 0 ? (ch % numVcs) : 0;
      const int slot = slotFor(layout, ch, issued[ch]);

      if (issued[ch] < totalSteps[ch]) {
        const bool ready = IsSend ? transport->isReadyForSend(vcIdx, slot)
                                  : transport->isReadyForRecv(vcIdx, slot);
        if (ready) {
          const size_t off =
              isZc ? sliceBegin(ch) : sliceBegin(ch) + issued[ch] * chunkSize;
          const size_t end =
              isZc ? sliceEnd(ch) : std::min(off + chunkSize, sliceEnd(ch));
          const size_t len = end - off;
          if constexpr (IsSend) {
            COMMCHECK_TEST(transport->iSendChunk({
                .userBuf = buf,
                .offset = off,
                .len = len,
                .vcIdx = vcIdx,
                .req = &reqs[ch][issued[ch]],
                .stagingSlot = slot,
                .round = roundFor(layout, issued[ch]),
                .localMrHdl = isZc ? localMrHdl : nullptr,
                .remoteMrHdl = isZc ? remoteMrHdl : nullptr,
                .remoteKey = isZc ? remoteKey : nullptr,
            }));
          } else {
            COMMCHECK_TEST(transport->iRecvChunk({
                .userBuf = buf,
                .offset = off,
                .len = len,
                .vcIdx = vcIdx,
                .req = &reqs[ch][issued[ch]],
                .stagingSlot = slot,
                .round = roundFor(layout, issued[ch]),
            }));
          }
          ++issued[ch];
        }
        done = false;
      }

      if (drained[ch] < issued[ch]) {
        bool chunkDone = false;
        COMMCHECK_TEST(
            transport->testChunkDone(reqs[ch][drained[ch]], &chunkDone));
        if (chunkDone) {
          ++drained[ch];
        }
      }
      if (drained[ch] < totalSteps[ch]) {
        done = false;
      }
    }
  }
}

// Bidirectional driver — send + recv concurrent in the same outer pass.
template <typename Buf>
void runChunkLoopBidir(
    IP2pHostTransport* transport,
    int numVcs,
    Buf* sendBuf,
    Buf* recvBuf,
    size_t totalSize,
    const ChannelLayout& layout,
    void* sendLocalMrHdl,
    void* sendRemoteMrHdl,
    const CtranIbRemoteAccessKey* sendRemoteKey) {
  const bool isZc = layout.slotsPerChannel == 0;
  const size_t chunkSize = isZc ? totalSize : transport->chunkSize();

  auto sliceBegin = [&](int ch) {
    return (totalSize / layout.numChannels) * ch;
  };
  auto sliceEnd = [&](int ch) {
    return ch == layout.numChannels - 1 ? totalSize : sliceBegin(ch + 1);
  };
  auto stepsForCh = [&](int ch) {
    if (isZc) {
      return 1;
    }
    const size_t sz = sliceEnd(ch) - sliceBegin(ch);
    return static_cast<int>((sz + chunkSize - 1) / chunkSize);
  };

  std::vector<int> sIssued(layout.numChannels, 0);
  std::vector<int> sDrained(layout.numChannels, 0);
  std::vector<int> rIssued(layout.numChannels, 0);
  std::vector<int> rDrained(layout.numChannels, 0);
  std::vector<std::vector<ChunkRequest>> sReqs(layout.numChannels);
  std::vector<std::vector<ChunkRequest>> rReqs(layout.numChannels);
  std::vector<int> totalSteps(layout.numChannels);
  for (int ch = 0; ch < layout.numChannels; ++ch) {
    totalSteps[ch] = stepsForCh(ch);
    sReqs[ch].resize(totalSteps[ch]);
    rReqs[ch].resize(totalSteps[ch]);
  }

  bool done = false;
  while (!done) {
    COMMCHECK_TEST(transport->progress());
    done = true;
    for (int ch = 0; ch < layout.numChannels; ++ch) {
      const int vcIdx = ch % numVcs;

      // send side
      if (sIssued[ch] < totalSteps[ch]) {
        const int slot = slotFor(layout, ch, sIssued[ch]);
        if (transport->isReadyForSend(vcIdx, slot)) {
          const size_t off =
              isZc ? sliceBegin(ch) : sliceBegin(ch) + sIssued[ch] * chunkSize;
          const size_t end =
              isZc ? sliceEnd(ch) : std::min(off + chunkSize, sliceEnd(ch));
          COMMCHECK_TEST(transport->iSendChunk({
              .userBuf = sendBuf,
              .offset = off,
              .len = end - off,
              .vcIdx = vcIdx,
              .req = &sReqs[ch][sIssued[ch]],
              .stagingSlot = slot,
              .round = roundFor(layout, sIssued[ch]),
              .localMrHdl = isZc ? sendLocalMrHdl : nullptr,
              .remoteMrHdl = isZc ? sendRemoteMrHdl : nullptr,
              .remoteKey = isZc ? sendRemoteKey : nullptr,
          }));
          ++sIssued[ch];
        }
        done = false;
      }
      if (sDrained[ch] < sIssued[ch]) {
        bool chunkDone = false;
        COMMCHECK_TEST(
            transport->testChunkDone(sReqs[ch][sDrained[ch]], &chunkDone));
        if (chunkDone) {
          ++sDrained[ch];
        }
      }
      if (sDrained[ch] < totalSteps[ch]) {
        done = false;
      }

      // recv side
      if (rIssued[ch] < totalSteps[ch]) {
        const int slot = slotFor(layout, ch, rIssued[ch]);
        if (transport->isReadyForRecv(vcIdx, slot)) {
          const size_t off =
              isZc ? sliceBegin(ch) : sliceBegin(ch) + rIssued[ch] * chunkSize;
          const size_t end =
              isZc ? sliceEnd(ch) : std::min(off + chunkSize, sliceEnd(ch));
          COMMCHECK_TEST(transport->iRecvChunk({
              .userBuf = recvBuf,
              .offset = off,
              .len = end - off,
              .vcIdx = vcIdx,
              .req = &rReqs[ch][rIssued[ch]],
              .stagingSlot = slot,
              .round = roundFor(layout, rIssued[ch]),
          }));
          ++rIssued[ch];
        }
        done = false;
      }
      if (rDrained[ch] < rIssued[ch]) {
        bool chunkDone = false;
        COMMCHECK_TEST(
            transport->testChunkDone(rReqs[ch][rDrained[ch]], &chunkDone));
        if (chunkDone) {
          ++rDrained[ch];
        }
      }
      if (rDrained[ch] < totalSteps[ch]) {
        done = false;
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════
// Parameterized fixture: (HostTransportMode, numChannels)
// ═══════════════════════════════════════════════════════════

using TestParam = std::tuple<HostTransportMode, int>;

class HostTransportParamTest : public ctran::CtranDistTestFixture,
                               public ::testing::WithParamInterface<TestParam> {
 public:
  void SetUp() override {
    CtranDistTestFixture::SetUp();
    setenv("NCCL_CTRAN_IB_NUM_VCS_PER_RANK", "2", 1);
    ncclCvarInit();
    comm_ = makeCtranComm();
    comm = comm_.get();
  }
  void TearDown() override {
    comm_.reset();
    CtranDistTestFixture::TearDown();
    unsetenv("NCCL_CTRAN_IB_NUM_VCS_PER_RANK");
  }

 protected:
  HostTransportMode mode() const {
    return std::get<0>(GetParam());
  }
  int numChannels() const {
    return std::get<1>(GetParam());
  }
  bool isZc() const {
    return mode() == HostTransportMode::kZeroCopy;
  }

  std::unique_ptr<IP2pHostTransport> makeTransport(
      int peerRank,
      CtranIb* ctranIb,
      int /*pipelineDepth*/,
      size_t /*chunkSize*/) {
    // Only ZC is wired up in this diff; CB lands in a follow-up.
    return std::make_unique<HostZcTransport>(
        peerRank, ctranIb, globalRank, localRank, &comm->logMetaData_);
  }

  int numVcs(IP2pHostTransport* transport) const {
    return static_cast<HostZcTransport*>(transport)->numVcs();
  }

  void printTestDesc(const std::string& testName, const std::string& testDesc) {
    XLOG_IF(WARN, this->globalRank == 0)
        << testName << "/" << (isZc() ? "Zc" : "Cb")
        << "/Channels=" << numChannels() << " numRanks " << this->numRanks
        << ". Description: " << testDesc << std::endl;
  }

  void verifyData(
      const std::vector<int>& hostBuf,
      std::function<int(size_t)> expected,
      const std::string& label = "") {
    constexpr int kMaxMismatches = 5;
    int mismatches = 0;
    for (size_t i = 0; i < hostBuf.size(); ++i) {
      if (hostBuf[i] != expected(i)) {
        if (mismatches < kMaxMismatches) {
          ADD_FAILURE() << label << (label.empty() ? "" : " ")
                        << "mismatch at index " << i << ": got " << hostBuf[i]
                        << ", expected " << expected(i);
        }
        mismatches++;
      }
    }
    if (mismatches > kMaxMismatches) {
      ADD_FAILURE() << label << (label.empty() ? "" : " ") << "... and "
                    << (mismatches - kMaxMismatches) << " more mismatches ("
                    << mismatches << " total out of " << hostBuf.size() << ")";
    }
    EXPECT_EQ(mismatches, 0);
  }

  std::unique_ptr<CtranComm> comm_{nullptr};
  CtranComm* comm{nullptr};
  static constexpr int kSendRank = 0;
  static constexpr int kRecvRank = 1;
};

INSTANTIATE_TEST_SUITE_P(
    Modes,
    HostTransportParamTest,
    ::testing::Combine(
        // CB mode lands in a follow-up diff.
        ::testing::Values(HostTransportMode::kZeroCopy),
        // Single-channel exercises the trivial dispatch case; multi-channel
        // exercises the per-channel round-robin VC/slot mapping in the
        // unified runChunkLoop driver.
        ::testing::Values(1, 2, 4)),
    [](const auto& info) {
      const auto m = std::get<0>(info.param);
      const auto ch = std::get<1>(info.param);
      const char* modeName = (m == HostTransportMode::kZeroCopy) ? "Zc" : "Cb";
      return std::string(modeName) + "_Ch" + std::to_string(ch);
    });

// ═══════════════════════════════════════════════════════════
// RoundTrip — same body for every (mode, numChannels) tuple
// ═══════════════════════════════════════════════════════════

TEST_P(HostTransportParamTest, RoundTrip) {
  printTestDesc(
      "RoundTrip",
      "Sender → receiver via the unified driver across the parameterized "
      "number of channels.");

  std::unique_ptr<CtranIb> ctranIb;
  try {
    ctranIb = std::make_unique<CtranIb>(comm);
  } catch (const std::bad_alloc&) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
  CtranIbEpochRAII epochRAII(ctranIb.get());

  constexpr size_t kBufSize = 1 * 1024 * 1024;
  constexpr int kPipelineDepth = 8;
  constexpr size_t kChunkSize = 64 * 1024;

  int* buf = nullptr;
  const size_t numElems = kBufSize / sizeof(int);
  ASSERT_EQ(cudaMalloc(&buf, kBufSize), cudaSuccess);
  std::vector<int> hostBuf(numElems);
  for (size_t i = 0; i < numElems; ++i) {
    hostBuf[i] = (globalRank == kSendRank) ? static_cast<int>(i + 1) : -1;
  }
  ASSERT_EQ(
      cudaMemcpy(buf, hostBuf.data(), kBufSize, cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  void* ibRegElem = nullptr;
  if (isZc()) {
    COMMCHECK_TEST(CtranIb::regMem(buf, kBufSize, localRank, &ibRegElem));
  }
  TestRegHdl reg;
  void* localMrHdl = isZc() ? wrapIbRegElem(reg, ibRegElem) : nullptr;

  auto transport = makeTransport(
      globalRank == kSendRank ? kRecvRank : kSendRank,
      ctranIb.get(),
      kPipelineDepth,
      kChunkSize);
  // Per-transport caller-lock contract: every hot-path call below
  // (iSendCtrlMsg / iRecvCtrlMsg / waitCtrlMsgDone / iSendChunk /
  // iRecvChunk / testChunkDone / progress) requires the calling
  // thread to hold this transport's mutex.
  P2pTransportLockGuard transportGuard(transport.get());

  cudaStream_t copyStream;
  ASSERT_EQ(cudaStreamCreate(&copyStream), cudaSuccess);

  const ChannelLayout layout{
      .numChannels = numChannels(),
      .slotsPerChannel =
          isZc() ? 0 : std::max(1, kPipelineDepth / numChannels()),
  };

  if (globalRank == kRecvRank) {
    CtrlRequest ctrlReq;
    // Receiver builds the ctrl msg explicitly in its own buffer:
    // EXPORT_MEM for ZC, SYNC for CB. The buffer must outlive the
    // request.
    ControlMsg ctrlMsg;
    if (isZc()) {
      COMMCHECK_TEST(exportRecvBuf(localMrHdl, buf, ctrlMsg));
    } else {
      ctrlMsg.setType(ControlMsgType::SYNC);
    }
    COMMCHECK_TEST(transport->iSendCtrlMsg(
        static_cast<ControlMsgType>(ctrlMsg.type),
        &ctrlMsg,
        sizeof(ctrlMsg),
        &ctrlReq));
    COMMCHECK_TEST(transport->waitCtrlMsgDone(ctrlReq));
    runChunkLoop</*IsSend=*/false>(
        transport.get(),
        numVcs(transport.get()),
        buf,
        kBufSize,
        layout,
        nullptr,
        nullptr,
        nullptr);
  } else {
    CtrlRequest ctrlReq;
    // Sender provides its own buffer for iRecvCtrlMsg to land bytes in,
    // then resolves them into RemotePeerInfo via importRemoteInfo.
    ControlMsg recvMsg;
    COMMCHECK_TEST(
        transport->iRecvCtrlMsg(&recvMsg, sizeof(recvMsg), &ctrlReq));
    COMMCHECK_TEST(transport->waitCtrlMsgDone(ctrlReq));
    RemotePeerInfo remote{};
    COMMCHECK_TEST(importRemoteInfo(recvMsg, &remote));
    EXPECT_EQ(remote.isZeroCopy, isZc());
    runChunkLoop</*IsSend=*/true>(
        transport.get(),
        numVcs(transport.get()),
        buf,
        kBufSize,
        layout,
        localMrHdl,
        remote.memHdl,
        &remote.remoteKey);
  }

  ASSERT_EQ(cudaStreamSynchronize(copyStream), cudaSuccess);
  if (globalRank == kRecvRank) {
    ASSERT_EQ(
        cudaMemcpy(hostBuf.data(), buf, kBufSize, cudaMemcpyDeviceToHost),
        cudaSuccess);
    verifyData(hostBuf, [](size_t i) { return static_cast<int>(i + 1); });
  }
  ASSERT_EQ(cudaStreamDestroy(copyStream), cudaSuccess);
  if (ibRegElem) {
    COMMCHECK_TEST(CtranIb::deregMem(ibRegElem));
  }
  ASSERT_EQ(cudaFree(buf), cudaSuccess);
}

// ═══════════════════════════════════════════════════════════
// BidirectionalRoundTrip — same body for every (mode, numChannels) tuple
// ═══════════════════════════════════════════════════════════

TEST_P(HostTransportParamTest, BidirectionalRoundTrip) {
  printTestDesc(
      "BidirectionalRoundTrip",
      "Both ranks simultaneously send and receive across the parameterized "
      "number of channels.");

  std::unique_ptr<CtranIb> ctranIb;
  try {
    ctranIb = std::make_unique<CtranIb>(comm);
  } catch (const std::bad_alloc&) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
  CtranIbEpochRAII epochRAII(ctranIb.get());

  constexpr size_t kBufSize = 1 * 1024 * 1024;
  constexpr int kPipelineDepth = 8;
  constexpr size_t kChunkSize = 64 * 1024;

  int* sendBuf = nullptr;
  int* recvBuf = nullptr;
  const size_t numElems = kBufSize / sizeof(int);
  ASSERT_EQ(cudaMalloc(&sendBuf, kBufSize), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&recvBuf, kBufSize), cudaSuccess);
  std::vector<int> hostBuf(numElems);
  for (size_t i = 0; i < numElems; ++i) {
    hostBuf[i] = static_cast<int>((globalRank + 1) * 1000 + i);
  }
  ASSERT_EQ(
      cudaMemcpy(sendBuf, hostBuf.data(), kBufSize, cudaMemcpyHostToDevice),
      cudaSuccess);
  std::fill(hostBuf.begin(), hostBuf.end(), -1);
  ASSERT_EQ(
      cudaMemcpy(recvBuf, hostBuf.data(), kBufSize, cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  void* sendRegElem = nullptr;
  void* recvRegElem = nullptr;
  if (isZc()) {
    COMMCHECK_TEST(CtranIb::regMem(sendBuf, kBufSize, localRank, &sendRegElem));
    COMMCHECK_TEST(CtranIb::regMem(recvBuf, kBufSize, localRank, &recvRegElem));
  }
  TestRegHdl sendReg, recvReg;
  void* sendLocalMrHdl = isZc() ? wrapIbRegElem(sendReg, sendRegElem) : nullptr;
  void* recvLocalMrHdl = isZc() ? wrapIbRegElem(recvReg, recvRegElem) : nullptr;

  int peerRank = (globalRank == kSendRank) ? kRecvRank : kSendRank;
  auto transport =
      makeTransport(peerRank, ctranIb.get(), kPipelineDepth, kChunkSize);
  // Per-transport caller-lock contract: every hot-path call below
  // requires the calling thread to hold this transport's mutex.
  P2pTransportLockGuard transportGuard(transport.get());

  CtrlRequest sendCtrlReq;
  CtrlRequest recvCtrlReq;
  ControlMsg outMsg;
  if (isZc()) {
    COMMCHECK_TEST(exportRecvBuf(recvLocalMrHdl, recvBuf, outMsg));
  } else {
    outMsg.setType(ControlMsgType::SYNC);
  }
  ControlMsg recvMsg;
  COMMCHECK_TEST(transport->iSendCtrlMsg(
      static_cast<ControlMsgType>(outMsg.type),
      &outMsg,
      sizeof(outMsg),
      &sendCtrlReq));
  COMMCHECK_TEST(
      transport->iRecvCtrlMsg(&recvMsg, sizeof(recvMsg), &recvCtrlReq));
  COMMCHECK_TEST(transport->waitCtrlMsgDone(sendCtrlReq));
  COMMCHECK_TEST(transport->waitCtrlMsgDone(recvCtrlReq));
  RemotePeerInfo remote{};
  COMMCHECK_TEST(importRemoteInfo(recvMsg, &remote));
  EXPECT_EQ(remote.isZeroCopy, isZc());

  cudaStream_t copyStream;
  ASSERT_EQ(cudaStreamCreate(&copyStream), cudaSuccess);

  const ChannelLayout layout{
      .numChannels = numChannels(),
      .slotsPerChannel =
          isZc() ? 0 : std::max(1, kPipelineDepth / numChannels()),
  };
  runChunkLoopBidir(
      transport.get(),
      numVcs(transport.get()),
      sendBuf,
      recvBuf,
      kBufSize,
      layout,
      sendLocalMrHdl,
      remote.memHdl,
      &remote.remoteKey);

  ASSERT_EQ(cudaStreamSynchronize(copyStream), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(hostBuf.data(), recvBuf, kBufSize, cudaMemcpyDeviceToHost),
      cudaSuccess);
  verifyData(hostBuf, [peerRank](size_t i) {
    return static_cast<int>((peerRank + 1) * 1000 + i);
  });

  ASSERT_EQ(cudaStreamDestroy(copyStream), cudaSuccess);
  if (sendRegElem) {
    COMMCHECK_TEST(CtranIb::deregMem(sendRegElem));
  }
  if (recvRegElem) {
    COMMCHECK_TEST(CtranIb::deregMem(recvRegElem));
  }
  ASSERT_EQ(cudaFree(sendBuf), cudaSuccess);
  ASSERT_EQ(cudaFree(recvBuf), cudaSuccess);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
