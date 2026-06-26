// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/transport/IP2pHostTransport.h"
#include "comms/ctran/transport/ib/HostCbTransport.h"
#include "comms/ctran/transport/ib/HostTransportImpl.h"
#include "comms/ctran/transport/ib/HostZcTransport.h"
#include "comms/ctran/transport/ib/tests/HostTestKernels.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ctran::transport;
using ctran::transport::ib::HostCbTransport;
using ctran::transport::ib::HostZcTransport;
using ctran::transport::ib::IbRecvPerSlotKernelArgs;
using ctran::transport::ib::IbStagingCopyTestKernelArgs;
using ctran::transport::ib::launchIbRecvPerSlotKernel;
using ctran::transport::ib::launchIbStagingCopyTestKernel;
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
      int pipelineDepth,
      size_t chunkSize) {
    if (isZc()) {
      return std::make_unique<HostZcTransport>(
          peerRank, ctranIb, globalRank, localRank, &comm->logMetaData_);
    }
    return std::make_unique<HostCbTransport>(
        peerRank,
        ctranIb,
        comm->ctran_->gpe->gpeKernelSyncPool(),
        pipelineDepth,
        chunkSize,
        globalRank,
        localRank,
        &comm->logMetaData_);
  }

  // Configure CB transport's per-slot GpeKernelSync.nworkers and launch
  // the unified staging-copy kernel. ZC: no-op. Any of (sendBuf,
  // recvBuf) may be nullptr for unidirectional cases.
  void maybeSetupCbKernel(
      IP2pHostTransport* transport,
      char* sendBuf,
      size_t sendTotalSize,
      int sendBlocks,
      char* recvBuf,
      size_t recvTotalSize,
      int recvBlocks,
      cudaStream_t stream) {
    if (isZc()) {
      return;
    }
    auto* cb = static_cast<HostCbTransport*>(transport);
    cb->setKernelNumBlocks(sendBlocks, recvBlocks);
    IbStagingCopyTestKernelArgs args{};
    args.devTransport = cb->getDeviceTransport();
    args.sendBuf = sendBuf;
    args.sendTotalSize = sendTotalSize;
    args.sendBlocks = sendBlocks;
    args.recvBuf = recvBuf;
    args.recvTotalSize = recvTotalSize;
    args.recvBlocks = recvBlocks;
    launchIbStagingCopyTestKernel(args, stream);
  }

  int numVcs(IP2pHostTransport* transport) const {
    return isZc() ? static_cast<HostZcTransport*>(transport)->numVcs()
                  : static_cast<HostCbTransport*>(transport)->numVcs();
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
        ::testing::Values(
            HostTransportMode::kZeroCopy,
            HostTransportMode::kCopyBased),
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
  constexpr int kNumBlocks = 4;

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
    maybeSetupCbKernel(
        transport.get(),
        /*sendBuf=*/nullptr,
        /*sendTotalSize=*/0,
        /*sendBlocks=*/0,
        /*recvBuf=*/reinterpret_cast<char*>(buf),
        /*recvTotalSize=*/kBufSize,
        /*recvBlocks=*/kNumBlocks,
        copyStream);
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
    maybeSetupCbKernel(
        transport.get(),
        /*sendBuf=*/reinterpret_cast<char*>(buf),
        /*sendTotalSize=*/kBufSize,
        /*sendBlocks=*/kNumBlocks,
        /*recvBuf=*/nullptr,
        /*recvTotalSize=*/0,
        /*recvBlocks=*/0,
        copyStream);
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
  constexpr int kNumBlocks = 4;

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
  maybeSetupCbKernel(
      transport.get(),
      /*sendBuf=*/reinterpret_cast<char*>(sendBuf),
      /*sendTotalSize=*/kBufSize,
      /*sendBlocks=*/kNumBlocks / 2,
      /*recvBuf=*/reinterpret_cast<char*>(recvBuf),
      /*recvTotalSize=*/kBufSize,
      /*recvBlocks=*/kNumBlocks / 2,
      copyStream);

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

// ═══════════════════════════════════════════════════════════
// RecvWrapAround — regression test for pollRecvNotifications
// CQE mis-attribution when recv slot indices wrap past
// pipelineDepth. The race:
//
//   1. Receiver issues recvs on slots 0..7 round 0 (all WAIT_RECV).
//   2. Sender sends chunks for slots 0..3 round 0 ONLY, then pauses.
//   3. Receiver consumes those CQEs, the kernel processes them, the
//      slots return to IDLE, and the receiver-side driver immediately
//      reissues round 1 on slots 0..3 (WAIT_RECV round 1).
//   4. Sender resumes and sends chunks for slots 4..7 round 0. Their
//      CQEs arrive at the receiver AFTER slots 0..3 are already
//      WAIT_RECV round 1.
//
// The old `pollRecvNotifications` picked the lowest-s WAIT_RECV slot
// and would attribute the late round-0 CQEs to slot 0/1/2/3 round 1,
// stranding slots 4/5/6/7 round 0 forever. The fixed code picks the
// slot with the oldest round, which keeps FIFO across rounds and
// routes those CQEs to slots 4..7 round 0.
//
// Two ranks: rank 0 sends, rank 1 receives.
// ═══════════════════════════════════════════════════════════

class HostTransportRecvWrapTest : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    CtranDistTestFixture::SetUp();
    // 2 VCs: VC 0 carries the ctrl-msg (kCtrlMsgVc) and ResourceExchange
    // traffic; we pin all data chunks to VC 1. pollRecvNotifications
    // scans per-VC, so isolating data chunks on a single VC keeps the
    // bug's wrap-around race deterministic.
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
  std::unique_ptr<CtranComm> comm_{nullptr};
  CtranComm* comm{nullptr};
  static constexpr int kSendRank = 0;
  static constexpr int kRecvRank = 1;
};

// Watchdog that aborts the test process if the test body doesn't
// signal completion within `timeout`. Without a watchdog a regression
// in pollRecvNotifications would hang the entire buck test target
// until the (much longer) outer test timeout fires.
class TestWatchdog {
 public:
  TestWatchdog(std::chrono::seconds timeout, const std::string& label)
      : timeout_(timeout), label_(label) {
    thread_ = std::thread([this] { run(); });
  }
  ~TestWatchdog() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      done_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  void run() {
    std::unique_lock<std::mutex> lk(mu_);
    if (!cv_.wait_for(lk, timeout_, [this] { return done_; })) {
      fprintf(
          stderr,
          "TestWatchdog[%s]: TIMED OUT after %lld s — likely a hang. "
          "Aborting test process.\n",
          label_.c_str(),
          static_cast<long long>(timeout_.count()));
      fflush(stderr);
      std::abort();
    }
  }

  std::chrono::seconds timeout_;
  std::string label_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool done_{false};
  std::thread thread_;
};

TEST_F(HostTransportRecvWrapTest, RecvWrapAroundNoStranding) {
  if (this->numRanks < 2) {
    GTEST_SKIP() << "Requires 2 ranks";
  }

  std::unique_ptr<CtranIb> ctranIb;
  try {
    ctranIb = std::make_unique<CtranIb>(comm);
  } catch (const std::bad_alloc&) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
  CtranIbEpochRAII epochRAII(ctranIb.get());

  // pipelineDepth=8 is the device-side max (kDeviceMaxPipelineDepth).
  // chunkSize small + many chunks → many wraps → many race windows.
  // 1 send block per the existing single-block ibStagingCopyTestKernel;
  // the recv side uses the per-slot kernel below (1 block per slot,
  // configured via setKernelNumBlocks(send, recv=1)).
  constexpr int kPipelineDepth = 8;
  constexpr size_t kChunkSize = 4 * 1024;
  constexpr int kTotalChunks = 32; // 4 full rounds across 8 slots
  constexpr size_t kBufSize = kChunkSize * kTotalChunks;
  constexpr int kSendBlocks = 1;
  // VC 0 is the ctrl-msg VC; pin all data chunks to VC 1 so the
  // wrap-around race in pollRecvNotifications is exercised on a single
  // data VC's WAIT_RECV set.
  constexpr int kVcIdx = 1;

  // Watchdog: a buggy poll could either corrupt staging (caught by the
  // verify pass at the end) or strand a slot and hang the per-slot
  // recv kernel. The watchdog converts a hang into a hard test abort
  // within 30 s so a regression doesn't sit on the test infra timeout.
  TestWatchdog watchdog(
      std::chrono::seconds(30),
      "rank" + std::to_string(globalRank) + "/RecvWrapAround");

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

  auto transport = std::make_unique<HostCbTransport>(
      globalRank == kSendRank ? kRecvRank : kSendRank,
      ctranIb.get(),
      comm->ctran_->gpe->gpeKernelSyncPool(),
      kPipelineDepth,
      kChunkSize,
      globalRank,
      localRank,
      &comm->logMetaData_);
  P2pTransportLockGuard transportGuard(transport.get());

  cudaStream_t copyStream;
  ASSERT_EQ(cudaStreamCreate(&copyStream), cudaSuccess);

  // One-way SYNC ctrl-msg handshake: receiver → sender, matching the
  // CB path in RoundTrip. The CB ResourceExchangeMsg is posted lazily
  // inside the first iSend/iRecvChunk call via postResourceExchange().
  if (globalRank == kRecvRank) {
    CtrlRequest ctrlReq;
    ControlMsg outMsg;
    outMsg.setType(ControlMsgType::SYNC);
    COMMCHECK_TEST(transport->iSendCtrlMsg(
        static_cast<ControlMsgType>(outMsg.type),
        &outMsg,
        sizeof(outMsg),
        &ctrlReq));
    COMMCHECK_TEST(transport->waitCtrlMsgDone(ctrlReq));
  } else {
    CtrlRequest ctrlReq;
    ControlMsg recvMsg;
    COMMCHECK_TEST(
        transport->iRecvCtrlMsg(&recvMsg, sizeof(recvMsg), &ctrlReq));
    COMMCHECK_TEST(transport->waitCtrlMsgDone(ctrlReq));
  }

  if (globalRank == kRecvRank) {
    // Launch the per-slot recv kernel: 1 block per slot, each block
    // walks only its own slot's chunks across rounds. Slots are
    // INDEPENDENT — a stranded slot N does NOT block block M (m != n)
    // from advancing. This decoupling is what lets the bug actually
    // surface: a misattributed postFlag on slot 0 round 1 wakes block 0
    // immediately to copy stale slot-0 staging (before sender's chunk-8
    // iput has landed there), corrupting the user buffer for chunk 8.
    // With the sequential single-block kernel, block 0 can't reach c=8
    // until all of c=4..7 are processed, by which time sender's
    // round-1 iputs have already landed → corruption healed by timing.
    transport->setKernelNumBlocks(kSendBlocks, /*recvNumBlocks=*/1);
    IbRecvPerSlotKernelArgs kArgs{};
    kArgs.devTransport = transport->getDeviceTransport();
    kArgs.recvBuf = reinterpret_cast<char*>(buf);
    kArgs.recvTotalSize = kBufSize;
    kArgs.totalChunks = kTotalChunks;
    launchIbRecvPerSlotKernel(kArgs, copyStream);

    // Issue every recv chunk in chunk-natural order. We do NOT pace
    // these — the driver should reissue round 1 on slots 0..3 as soon
    // as those slots return to IDLE (which is exactly the race window
    // the sender's pause below opens).
    std::vector<ChunkRequest> reqs(kTotalChunks);
    int issued = 0;
    int drained = 0;
    while (drained < kTotalChunks) {
      COMMCHECK_TEST(transport->progress());
      if (issued < kTotalChunks) {
        const int slot = issued % kPipelineDepth;
        const int round = issued / kPipelineDepth;
        if (transport->isReadyForRecv(kVcIdx, slot)) {
          COMMCHECK_TEST(transport->iRecvChunk({
              .userBuf = buf,
              .offset = static_cast<size_t>(issued) * kChunkSize,
              .len = kChunkSize,
              .vcIdx = kVcIdx,
              .req = &reqs[issued],
              .stagingSlot = slot,
              .round = round,
          }));
          ++issued;
        }
      }
      if (drained < issued) {
        bool done = false;
        COMMCHECK_TEST(transport->testChunkDone(reqs[drained], &done));
        if (done) {
          ++drained;
        }
      }
    }
  } else {
    // Sender: launch the staging-copy kernel for send. The CB send
    // state machine relies on it to fill the staging slot before
    // ifetchAndAdd notifies the peer; without it, the WAIT_PREPARE
    // state never advances and the test hangs.
    // recvNumBlocks=0 leaves recv-side nworkers untouched (sender
    // never issues recvs on this transport).
    transport->setKernelNumBlocks(kSendBlocks, /*recvNumBlocks=*/0);
    IbStagingCopyTestKernelArgs kArgs{};
    kArgs.devTransport = transport->getDeviceTransport();
    kArgs.sendBuf = reinterpret_cast<char*>(buf);
    kArgs.sendTotalSize = kBufSize;
    kArgs.sendBlocks = kSendBlocks;
    launchIbStagingCopyTestKernel(kArgs, copyStream);

    // Drive the bug deterministically by pacing chunks into two
    // batches per round:
    //   - phase A: send 4 low-slot chunks (slots 0..3) of round r
    //   - drain them so receiver finishes round r on those slots and
    //     reissues round r+1 (WAIT_RECV) before any late CQE lands
    //   - sleep to widen the window in which slots 4..7 round r are
    //     still WAIT_RECV with no CQE yet, while slots 0..3 are
    //     WAIT_RECV round r+1
    //   - phase B: send 4 high-slot chunks (slots 4..7) of round r.
    //     With the bug, these CQEs get stolen by slot 0..3 round r+1,
    //     stranding 4..7 round r and hanging the receiver kernel.
    //
    // (For the last round, phase B may not have a corresponding round
    // r+1 yet, so the bug needs to fire on earlier rounds. With
    // kTotalChunks=32 we have rounds 0..3, plenty of windows.)
    constexpr int kRounds = kTotalChunks / kPipelineDepth;
    static_assert(
        kRounds >= 2, "Need at least 2 rounds to wrap and fire the race");

    std::vector<ChunkRequest> reqs(kTotalChunks);
    constexpr int kHalf = kPipelineDepth / 2; // 4

    auto sendOne = [&](int chunkIdx) {
      const int slot = chunkIdx % kPipelineDepth;
      const int round = chunkIdx / kPipelineDepth;
      while (!transport->isReadyForSend(kVcIdx, slot)) {
        COMMCHECK_TEST(transport->progress());
      }
      COMMCHECK_TEST(transport->iSendChunk({
          .userBuf = buf,
          .offset = static_cast<size_t>(chunkIdx) * kChunkSize,
          .len = kChunkSize,
          .vcIdx = kVcIdx,
          .req = &reqs[chunkIdx],
          .stagingSlot = slot,
          .round = round,
      }));
    };

    auto drainOne = [&](int chunkIdx) {
      bool done = false;
      while (!done) {
        COMMCHECK_TEST(transport->progress());
        COMMCHECK_TEST(transport->testChunkDone(reqs[chunkIdx], &done));
      }
    };

    for (int r = 0; r < kRounds; ++r) {
      const int base = r * kPipelineDepth;
      // Phase A: low half (slots 0..3) of round r.
      for (int s = 0; s < kHalf; ++s) {
        sendOne(base + s);
      }
      for (int s = 0; s < kHalf; ++s) {
        drainOne(base + s);
      }
      // Window: receiver consumes the low half, kernel completes them,
      // driver reissues round r+1 on slots 0..3 → WAIT_RECV round r+1.
      // Slots 4..7 are still WAIT_RECV round r with no CQE yet (we
      // haven't sent them). The sleep lets the receiver run through
      // all of those state transitions before we wake up.
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      // Phase B: high half (slots 4..7) of round r. Without the fix,
      // their notify CQEs get attributed to slot 0..3 round r+1,
      // leaving slots 4..7 round r stranded — the kernel either hangs
      // at chunk r*pipelineDepth+4 or processes corrupt staging.
      for (int s = kHalf; s < kPipelineDepth; ++s) {
        sendOne(base + s);
      }
      for (int s = kHalf; s < kPipelineDepth; ++s) {
        drainOne(base + s);
      }
    }
  }

  ASSERT_EQ(cudaStreamSynchronize(copyStream), cudaSuccess);
  if (globalRank == kRecvRank) {
    ASSERT_EQ(
        cudaMemcpy(hostBuf.data(), buf, kBufSize, cudaMemcpyDeviceToHost),
        cudaSuccess);
    // Total mismatches across the recv buffer. Per-chunk mismatch counts
    // are reported so a regression makes the failed slot/round obvious.
    constexpr int kMaxMismatches = 5;
    int totalMismatches = 0;
    for (int c = 0; c < kTotalChunks; ++c) {
      const size_t elemBase =
          static_cast<size_t>(c) * (kChunkSize / sizeof(int));
      const size_t elemEnd = elemBase + (kChunkSize / sizeof(int));
      int chunkMismatches = 0;
      for (size_t i = elemBase; i < elemEnd; ++i) {
        const int expected = static_cast<int>(i + 1);
        if (hostBuf[i] != expected) {
          if (totalMismatches < kMaxMismatches) {
            ADD_FAILURE() << "chunk " << c << " (slot=" << (c % kPipelineDepth)
                          << " round=" << (c / kPipelineDepth)
                          << ") mismatch at index " << i << ": got "
                          << hostBuf[i] << ", expected " << expected;
          }
          ++totalMismatches;
          ++chunkMismatches;
        }
      }
      if (chunkMismatches > 0) {
        fprintf(
            stderr,
            "[Recv] chunk %d (slot=%d round=%d): %d/%d mismatches\n",
            c,
            c % kPipelineDepth,
            c / kPipelineDepth,
            chunkMismatches,
            static_cast<int>(kChunkSize / sizeof(int)));
        fflush(stderr);
      }
    }
    EXPECT_EQ(totalMismatches, 0)
        << "Total mismatches: " << totalMismatches << " of " << numElems;
  }
  ASSERT_EQ(cudaStreamDestroy(copyStream), cudaSuccess);
  ASSERT_EQ(cudaFree(buf), cudaSuccess);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
