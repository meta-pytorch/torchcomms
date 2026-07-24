// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <folly/SocketAddress.h>
#include <folly/futures/Future.h>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/CtranIbImpl.h"
#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/ctran/bootstrap/AbortableSocket.h"
#include "comms/ctran/bootstrap/ISocketFactory.h"
#include "comms/utils/cvars/nccl_cvars.h"

using AbortPtr = std::shared_ptr<comms::fault_tolerance::Abort>;
using namespace std::literals::chrono_literals;

namespace {

// Configuration pinned by SetUp() so every assertion can compare
// against precomputed values:
//   NCCL_CTRAN_IB_DEVICES_PER_RANK   = 2
//   NCCL_CTRAN_IB_MAX_QPS            = 16
//   NCCL_CTRAN_IB_NUM_VCS_PER_RANK   = 4
// =>
//   kExpectedMaxVcsPerPeer = 4
//   kExpectedMaxVcsPerNic  = 4 / 2 = 2
//   kExpectedQpsPerVc      = 16 / 4 = 4 (no XRACK/XZONE/XDC override
//                                        in the test env, so the
//                                        connection-class configList
//                                        path is skipped and the slice
//                                        comes straight from the cvar).
constexpr int kExpectedNumNics = 2;
constexpr int kExpectedMaxQps = 16;
constexpr int kExpectedMaxVcsPerPeer = 4;
constexpr int kExpectedMaxVcsPerNic = kExpectedMaxVcsPerPeer / kExpectedNumNics;
constexpr int kExpectedQpsPerVc = kExpectedMaxQps / kExpectedMaxVcsPerPeer;

SocketServerAddr makeServerAddr() {
  SocketServerAddr addr;
  addr.port = 0; // OS picks
  addr.ipv4 = "127.0.0.1";
  addr.ifName = "lo";
  return addr;
}

// Helper to drain progress until the predicate is true or we timeout.
template <typename Pred>
bool progressUntil(
    CtranIb* ib,
    Pred pred,
    std::chrono::milliseconds timeout = 10s) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) {
      return true;
    }
    EXPECT_EQ(ib->progress(), commSuccess);
    std::this_thread::sleep_for(1ms);
  }
  return pred();
}

class CtranIbMultiVcPinnedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Pin the cvars that drive maxVcsPerPeer so this test is robust to the
    // host's NIC count. NCCL_CTRAN_IB_NUM_VCS_PER_RANK directly drives
    // the per-peer VC count; force numNics=2 and numVcsPerPeer=4 so we
    // get maxVcsPerNic = 4/2 = 2 and qpsPerVc = 16/4 = 4 regardless of
    // the host.
    setenv("NCCL_CTRAN_IB_DEVICES_PER_RANK", "2", 1);
    setenv("NCCL_CTRAN_IB_MAX_QPS", "16", 1);
    setenv("NCCL_CTRAN_IB_NUM_VCS_PER_RANK", "4", 1);
    ncclCvarInit();
    EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
    int deviceCount = 0;
    EXPECT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    ASSERT_GE(deviceCount, 2)
        << "Test requires at least 2 CUDA devices, found " << deviceCount;
  }

  void TearDown() override {
    EXPECT_EQ(cudaDeviceReset(), cudaSuccess);
    unsetenv("NCCL_CTRAN_IB_DEVICES_PER_RANK");
    unsetenv("NCCL_CTRAN_IB_MAX_QPS");
    unsetenv("NCCL_CTRAN_IB_NUM_VCS_PER_RANK");
    ncclCvarInit();
  }

  std::unique_ptr<CtranIb> makeIb(int rank, AbortPtr abortCtrl) {
    SocketServerAddr serverAddr = makeServerAddr();
    return std::make_unique<CtranIb>(
        rank,
        rank, // cudaDev
        /*commHash=*/0xc4ac4a1ull,
        std::string("channel-test"),
        /*enableLocalFlush=*/false,
        CtranIb::BootstrapMode::kSpecifiedServer,
        &serverAddr,
        abortCtrl,
        std::make_shared<ctran::bootstrap::AbortableSocketFactory>(),
        /*maxNumCqe=*/std::nullopt);
  }
};

// Two-rank harness: both ranks build a CtranIb, exchange listen addresses,
// then run their respective lambdas.
template <typename Action0, typename Action1>
void runTwoRanks(Action0 action0, Action1 action1) {
  auto [addr0Promise, addr0Future] =
      folly::makePromiseContract<folly::SocketAddress>();
  auto [addr1Promise, addr1Future] =
      folly::makePromiseContract<folly::SocketAddress>();

  std::thread t0([&]() {
    EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
    auto abortCtrl = comms::fault_tolerance::createAbort(/*enabled=*/true);
    SocketServerAddr serverAddr = makeServerAddr();
    auto ib = std::make_unique<CtranIb>(
        0,
        0,
        0xc4ac4a1ull,
        std::string("channel-test"),
        false,
        CtranIb::BootstrapMode::kSpecifiedServer,
        &serverAddr,
        abortCtrl,
        std::make_shared<ctran::bootstrap::AbortableSocketFactory>(),
        std::nullopt);
    auto maybeListen = ib->getListenSocketListenAddr();
    ASSERT_FALSE(maybeListen.hasError());
    addr0Promise.setValue(maybeListen.value());
    auto peerListen = std::move(addr1Future).get();
    SocketServerAddr peerAddr;
    peerAddr.port = peerListen.getPort();
    peerAddr.ipv4 = peerListen.getAddressStr();
    peerAddr.ifName = "lo";
    action0(ib.get(), &peerAddr);
  });
  std::thread t1([&]() {
    EXPECT_EQ(cudaSetDevice(1), cudaSuccess);
    auto abortCtrl = comms::fault_tolerance::createAbort(/*enabled=*/true);
    SocketServerAddr serverAddr = makeServerAddr();
    auto ib = std::make_unique<CtranIb>(
        1,
        1,
        0xc4ac4a1ull,
        std::string("channel-test"),
        false,
        CtranIb::BootstrapMode::kSpecifiedServer,
        &serverAddr,
        abortCtrl,
        std::make_shared<ctran::bootstrap::AbortableSocketFactory>(),
        std::nullopt);
    auto maybeListen = ib->getListenSocketListenAddr();
    ASSERT_FALSE(maybeListen.hasError());
    addr1Promise.setValue(maybeListen.value());
    auto peerListen = std::move(addr0Future).get();
    SocketServerAddr peerAddr;
    peerAddr.port = peerListen.getPort();
    peerAddr.ipv4 = peerListen.getAddressStr();
    peerAddr.ifName = "lo";
    action1(ib.get(), &peerAddr);
  });
  t0.join();
  t1.join();
}

// PR2 test: connectVcs lazily creates exactly maxVcsPerPeer VCs on both
// sides, and each VC is pinned to the expected NIC with exactly
// qpsPerVc data QPs (NCCL_CTRAN_IB_MAX_QPS / maxVcsPerPeer).
TEST_F(CtranIbMultiVcPinnedTest, GetVcsLazyCreatesAndPinsByNic) {
  runTwoRanks(
      // rank 0 (smaller -> initiator)
      [](CtranIb* ib, const SocketServerAddr* peerAddr) {
        const int peerRank = 1;
        EXPECT_EQ(ib->getMaxVcsPerPeer(), kExpectedMaxVcsPerPeer);
        EXPECT_EQ(ib->getMaxVcsPerNic(), kExpectedMaxVcsPerNic);

        const auto& vcs = ib->connectVcs(peerRank, peerAddr);
        ASSERT_EQ(static_cast<int>(vcs.size()), kExpectedMaxVcsPerPeer);
        for (int vcIdx = 0; vcIdx < kExpectedMaxVcsPerPeer; ++vcIdx) {
          int expectedNic = vcIdx / kExpectedMaxVcsPerNic;
          EXPECT_EQ(vcs[vcIdx]->getCtrlDevice(), expectedNic)
              << "vcIdx " << vcIdx;
          EXPECT_EQ(vcs[vcIdx]->getActiveDevices().size(), 1U)
              << "vcIdx " << vcIdx;
          EXPECT_EQ(vcs[vcIdx]->getMaxNumQp(), kExpectedQpsPerVc)
              << "vcIdx " << vcIdx;
          EXPECT_EQ(
              static_cast<int>(vcs[vcIdx]->getDataQpNums().size()),
              kExpectedQpsPerVc)
              << "vcIdx " << vcIdx;
        }

        // Cached: second call returns the same vector.
        const auto& vcs2 = ib->connectVcs(peerRank, peerAddr);
        EXPECT_EQ(vcs.data(), vcs2.data());
      },
      // rank 1 (larger -> waits for accept)
      [](CtranIb* ib, const SocketServerAddr* peerAddr) {
        const int peerRank = 0;
        EXPECT_EQ(ib->getMaxVcsPerPeer(), kExpectedMaxVcsPerPeer);
        EXPECT_EQ(ib->getMaxVcsPerNic(), kExpectedMaxVcsPerNic);

        const auto& vcs = ib->connectVcs(peerRank, peerAddr);
        ASSERT_EQ(static_cast<int>(vcs.size()), kExpectedMaxVcsPerPeer);
        for (int vcIdx = 0; vcIdx < kExpectedMaxVcsPerPeer; ++vcIdx) {
          int expectedNic = vcIdx / kExpectedMaxVcsPerNic;
          EXPECT_EQ(vcs[vcIdx]->getCtrlDevice(), expectedNic)
              << "vcIdx " << vcIdx;
          EXPECT_EQ(vcs[vcIdx]->getMaxNumQp(), kExpectedQpsPerVc)
              << "vcIdx " << vcIdx;
          EXPECT_EQ(
              static_cast<int>(vcs[vcIdx]->getDataQpNums().size()),
              kExpectedQpsPerVc)
              << "vcIdx " << vcIdx;
        }
      });
}

// PR3 test: vc->notify / vc->checkNotify round-trip on every per-peer VC.
TEST_F(CtranIbMultiVcPinnedTest, NotifyVcRoundTrip) {
  runTwoRanks(
      // rank 0: send notify on each VC
      [](CtranIb* ib, const SocketServerAddr* peerAddr) {
        const int peerRank = 1;
        const auto& vcs = ib->connectVcs(peerRank, peerAddr);
        ASSERT_EQ(static_cast<int>(vcs.size()), kExpectedMaxVcsPerPeer);

        for (int c = 0; c < kExpectedMaxVcsPerPeer; ++c) {
          CtranIbEpochRAII epochRAII(ib);
          CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[c]->mutex, {
            EXPECT_EQ(vcs[c]->notify(/*req=*/nullptr), commSuccess);
          });
        }
        // Drive progress so notifications get sent.
        for (int i = 0; i < 200; ++i) {
          EXPECT_EQ(ib->progress(), commSuccess);
          std::this_thread::sleep_for(2ms);
        }
      },
      // rank 1: poll for notify on each VC
      [](CtranIb* ib, const SocketServerAddr* peerAddr) {
        const int peerRank = 0;
        const auto& vcs = ib->connectVcs(peerRank, peerAddr);
        ASSERT_EQ(static_cast<int>(vcs.size()), kExpectedMaxVcsPerPeer);

        for (int c = 0; c < kExpectedMaxVcsPerPeer; ++c) {
          bool ok = progressUntil(ib, [&]() {
            CtranIbEpochRAII epochRAII(ib);
            bool notified = false;
            CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[c]->mutex, {
              EXPECT_EQ(vcs[c]->checkNotify(&notified), commSuccess);
            });
            return notified;
          });
          EXPECT_TRUE(ok) << "no notify on VC " << c;
        }
      });
}

// PR3 functional test: full put-notify round-trip through per-peer VCs.
// Models CtranIbDistUT.cc::runPutNotify but uses direct per-VC dispatch:
//   vc->isendCtrlMsg / vc->irecvCtrlMsg exchange the buffer info,
//   vc->iput performs the RDMA write with notify on the same VC,
//   vc->checkNotify polls for completion, then we verify data.
TEST_F(CtranIbMultiVcPinnedTest, IputVcRoundTrip) {
  constexpr size_t kBufCount = 1024;
  constexpr int kSendVal = 42;

  auto waitReq = [](CtranIb* ib, CtranIbRequest& req) {
    progressUntil(ib, [&]() { return req.isComplete(); });
  };

  runTwoRanks(
      // rank 0 = sender. Per VC: receive recv-side buffer info via
      // ctrl msg, vc->iput with notify, drive progress until done.
      [&](CtranIb* ib, const SocketServerAddr* peerAddr) {
        const int peerRank = 1;
        const auto& vcs = ib->connectVcs(peerRank, peerAddr);
        ASSERT_EQ(static_cast<int>(vcs.size()), kExpectedMaxVcsPerPeer);

        for (int c = 0; c < kExpectedMaxVcsPerPeer; ++c) {
          // Source buffer has VC-distinct content so the receiver can
          // verify each VC's payload independently.
          std::vector<int> sendBuf(kBufCount, kSendVal + c);
          void* sendHandle = nullptr;
          ASSERT_EQ(
              CtranIb::regMem(
                  sendBuf.data(),
                  kBufCount * sizeof(int),
                  /*cudaDev=*/0,
                  &sendHandle),
              commSuccess);

          // Receive the receiver's exported buffer info on this VC.
          ControlMsg recvMsg;
          CtranIbRequest ctrlReq;
          {
            CtranIbEpochRAII epoch(ib);
            CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[c]->mutex, {
              ASSERT_EQ(
                  vcs[c]->irecvCtrlMsg(&recvMsg, sizeof(recvMsg), ctrlReq),
                  commSuccess);
            });
          }
          waitReq(ib, ctrlReq);

          // vc->iput into the receiver's buffer with notify.
          void* remoteBuf = reinterpret_cast<void*>(recvMsg.ibDesc.remoteAddr);
          CtranIbRemoteAccessKey key{};
          for (int i = 0; i < recvMsg.ibDesc.nKeys; ++i) {
            key.rkeys[i] = recvMsg.ibDesc.rkeys[i];
          }
          CtranIbRequest putReq;
          {
            CtranIbEpochRAII epoch(ib);
            CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[c]->mutex, {
              ASSERT_EQ(
                  vcs[c]->iput(
                      sendBuf.data(),
                      remoteBuf,
                      kBufCount * sizeof(int),
                      sendHandle,
                      key,
                      /*notify=*/true,
                      /*config=*/nullptr,
                      &putReq,
                      /*fast=*/false),
                  commSuccess);
            });
          }
          waitReq(ib, putReq);
          ASSERT_EQ(CtranIb::deregMem(sendHandle), commSuccess);
        }
      },
      // rank 1 = receiver. Per VC: send buffer info, wait notify,
      // verify the VC-distinct payload.
      [&](CtranIb* ib, const SocketServerAddr* peerAddr) {
        const int peerRank = 0;
        const auto& vcs = ib->connectVcs(peerRank, peerAddr);
        ASSERT_EQ(static_cast<int>(vcs.size()), kExpectedMaxVcsPerPeer);

        for (int c = 0; c < kExpectedMaxVcsPerPeer; ++c) {
          std::vector<int> recvBuf(kBufCount, /*recvVal=*/-1);
          void* recvHandle = nullptr;
          ASSERT_EQ(
              CtranIb::regMem(
                  recvBuf.data(),
                  kBufCount * sizeof(int),
                  /*cudaDev=*/1,
                  &recvHandle),
              commSuccess);

          ControlMsg sendMsg;
          ASSERT_EQ(
              CtranIb::exportMem(recvBuf.data(), recvHandle, sendMsg),
              commSuccess);

          CtranIbRequest ctrlReq;
          {
            CtranIbEpochRAII epoch(ib);
            CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[c]->mutex, {
              ASSERT_EQ(
                  vcs[c]->isendCtrlMsg(
                      sendMsg.type, &sendMsg, sizeof(sendMsg), ctrlReq),
                  commSuccess);
            });
          }
          waitReq(ib, ctrlReq);

          // Wait for the put's notify on this VC.
          bool ok = progressUntil(ib, [&]() {
            CtranIbEpochRAII epoch(ib);
            bool notified = false;
            CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[c]->mutex, {
              EXPECT_EQ(vcs[c]->checkNotify(&notified), commSuccess);
            });
            return notified;
          });
          EXPECT_TRUE(ok) << "VC " << c << ": no notify received";

          // Verify the VC-distinct payload landed.
          for (size_t i = 0; i < kBufCount; ++i) {
            ASSERT_EQ(recvBuf[i], kSendVal + c) << "VC " << c << " idx " << i;
          }
          ASSERT_EQ(CtranIb::deregMem(recvHandle), commSuccess);
        }
      });
}

// PR3 functional test: vc->ifetchAndAdd produces correct results when
// every per-peer VC issues N atomic increments concurrently.
TEST_F(CtranIbMultiVcPinnedTest, FetchAndAddVcRoundTrip) {
  constexpr int kNumOpsPerVc = 50;

  auto waitReq = [](CtranIb* ib, CtranIbRequest& req) {
    progressUntil(ib, [&]() { return req.isComplete(); });
  };

  runTwoRanks(
      // rank 0 = adder. Per VC: receive remote buffer info, issue
      // kNumOpsPerVc vc->ifetchAndAdd ops with a signaled request
      // on the last one.
      [&](CtranIb* ib, const SocketServerAddr* peerAddr) {
        const int peerRank = 1;
        const auto& vcs = ib->connectVcs(peerRank, peerAddr);
        ASSERT_EQ(static_cast<int>(vcs.size()), kExpectedMaxVcsPerPeer);

        // Local scratch buffer used as fetched-value destination.
        std::vector<uint64_t> scratch(1, 0);
        void* scratchHandle = nullptr;
        ASSERT_EQ(
            CtranIb::regMem(
                scratch.data(),
                sizeof(uint64_t),
                /*cudaDev=*/0,
                &scratchHandle),
            commSuccess);

        for (int c = 0; c < kExpectedMaxVcsPerPeer; ++c) {
          // Receive the receiver's exported counter location.
          ControlMsg recvMsg;
          CtranIbRequest ctrlReq;
          {
            CtranIbEpochRAII epoch(ib);
            CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[c]->mutex, {
              ASSERT_EQ(
                  vcs[c]->irecvCtrlMsg(&recvMsg, sizeof(recvMsg), ctrlReq),
                  commSuccess);
            });
          }
          waitReq(ib, ctrlReq);

          void* remoteBuf = reinterpret_cast<void*>(recvMsg.ibDesc.remoteAddr);
          CtranIbRemoteAccessKey key{};
          for (int i = 0; i < recvMsg.ibDesc.nKeys; ++i) {
            key.rkeys[i] = recvMsg.ibDesc.rkeys[i];
          }

          // Issue kNumOpsPerVc atomic increments; signal only on the
          // last so we can wait for all of them in one shot.
          for (int i = 0; i < kNumOpsPerVc; ++i) {
            CtranIbRequest atomicReq;
            CtranIbRequest* reqPtr =
                (i == kNumOpsPerVc - 1) ? &atomicReq : nullptr;
            {
              CtranIbEpochRAII epoch(ib);
              CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[c]->mutex, {
                ASSERT_EQ(
                    vcs[c]->ifetchAndAdd(
                        scratch.data(),
                        remoteBuf,
                        /*addVal=*/1,
                        scratchHandle,
                        key,
                        reqPtr),
                    commSuccess);
              });
            }
            if (reqPtr) {
              waitReq(ib, atomicReq);
            }
          }

          // Tell receiver via a notify that all atomics on this VC are
          // posted (and thus visible after the atomicQp's CQEs are drained).
          {
            CtranIbEpochRAII epoch(ib);
            CtranIbRequest doneReq;
            CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[c]->mutex, {
              ASSERT_EQ(vcs[c]->notify(&doneReq), commSuccess);
            });
            waitReq(ib, doneReq);
          }
        }

        ASSERT_EQ(CtranIb::deregMem(scratchHandle), commSuccess);
      },
      // rank 1 = target. Per VC: register a fresh counter, send its
      // info, wait notify-of-done, verify the counter advanced exactly
      // kNumOpsPerVc times.
      [&](CtranIb* ib, const SocketServerAddr* peerAddr) {
        const int peerRank = 0;
        const auto& vcs = ib->connectVcs(peerRank, peerAddr);
        ASSERT_EQ(static_cast<int>(vcs.size()), kExpectedMaxVcsPerPeer);

        for (int c = 0; c < kExpectedMaxVcsPerPeer; ++c) {
          std::vector<uint64_t> counter(1, 0);
          void* counterHandle = nullptr;
          ASSERT_EQ(
              CtranIb::regMem(
                  counter.data(),
                  sizeof(uint64_t),
                  /*cudaDev=*/1,
                  &counterHandle),
              commSuccess);

          ControlMsg sendMsg;
          ASSERT_EQ(
              CtranIb::exportMem(counter.data(), counterHandle, sendMsg),
              commSuccess);
          CtranIbRequest ctrlReq;
          {
            CtranIbEpochRAII epoch(ib);
            CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[c]->mutex, {
              ASSERT_EQ(
                  vcs[c]->isendCtrlMsg(
                      sendMsg.type, &sendMsg, sizeof(sendMsg), ctrlReq),
                  commSuccess);
            });
          }
          waitReq(ib, ctrlReq);

          // Wait for the "all atomics done" notify.
          bool ok = progressUntil(ib, [&]() {
            CtranIbEpochRAII epoch(ib);
            bool notified = false;
            CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[c]->mutex, {
              EXPECT_EQ(vcs[c]->checkNotify(&notified), commSuccess);
            });
            return notified;
          });
          EXPECT_TRUE(ok) << "VC " << c << ": no notify received";

          EXPECT_EQ(counter[0], static_cast<uint64_t>(kNumOpsPerVc))
              << "VC " << c;
          ASSERT_EQ(CtranIb::deregMem(counterHandle), commSuccess);
        }
      });
}

// Guard: progress(device) must reject out-of-range device indices with
// commInternalError instead of polling an out-of-bounds CQ.
TEST_F(CtranIbMultiVcPinnedTest, ProgressInvalidDeviceReturnsError) {
  auto abortCtrl = comms::fault_tolerance::createAbort(/*enabled=*/true);
  auto ib = makeIb(/*rank=*/0, abortCtrl);
  ASSERT_NE(ib, nullptr);
  ASSERT_EQ(ib->getNumIbDevices(), kExpectedNumNics);

  CtranIbEpochRAII epoch(ib.get());
  // Negative device index is invalid.
  EXPECT_EQ(ib->progress(-1), commInternalError);
  // device == numIbDevices and beyond are out of range.
  EXPECT_EQ(ib->progress(kExpectedNumNics), commInternalError);
  EXPECT_EQ(ib->progress(kExpectedNumNics + 5), commInternalError);

  // Sanity: valid device indices still succeed.
  for (int d = 0; d < kExpectedNumNics; ++d) {
    EXPECT_EQ(ib->progress(d), commSuccess);
  }
}

// N-rank harness: each rank builds a CtranIb on its own thread, publishes
// its listen address into a shared table, waits until every rank has
// published, then runs `action(rank, ib, peerAddrs)` with peerAddrs[i]
// being rank i's listen address. peerAddrs[rank] (self) is unused.
template <typename Action>
void runNRanks(int nRanks, Action action) {
  struct Shared {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::optional<SocketServerAddr>> addrs;
  };
  auto shared = std::make_shared<Shared>();
  shared->addrs.resize(nRanks);

  std::vector<std::thread> threads;
  threads.reserve(nRanks);
  for (int rank = 0; rank < nRanks; ++rank) {
    threads.emplace_back([rank, nRanks, shared, &action]() {
      EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);
      auto abortCtrl = comms::fault_tolerance::createAbort(/*enabled=*/true);
      SocketServerAddr serverAddr = makeServerAddr();
      auto ib = std::make_unique<CtranIb>(
          rank,
          rank,
          0xc4ac4a1ull,
          std::string("channel-test"),
          /*enableLocalFlush=*/false,
          CtranIb::BootstrapMode::kSpecifiedServer,
          &serverAddr,
          abortCtrl,
          std::make_shared<ctran::bootstrap::AbortableSocketFactory>(),
          /*maxNumCqe=*/std::nullopt);
      auto maybeListen = ib->getListenSocketListenAddr();
      ASSERT_FALSE(maybeListen.hasError());
      auto listen = maybeListen.value();
      SocketServerAddr myPubAddr;
      myPubAddr.port = listen.getPort();
      myPubAddr.ipv4 = listen.getAddressStr();
      myPubAddr.ifName = "lo";

      std::vector<SocketServerAddr> peerAddrs;
      {
        std::unique_lock<std::mutex> lk(shared->mu);
        shared->addrs[rank] = myPubAddr;
        shared->cv.notify_all();
        shared->cv.wait(lk, [&]() {
          for (int i = 0; i < nRanks; ++i) {
            if (!shared->addrs[i].has_value()) {
              return false;
            }
          }
          return true;
        });
        peerAddrs.reserve(nRanks);
        for (int i = 0; i < nRanks; ++i) {
          peerAddrs.push_back(shared->addrs[i].value());
        }
      }
      action(rank, ib.get(), peerAddrs);
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}

// Multi-rank (4 ranks) test: each rank calls connectVcs() against its peers
// in a different order so that the larger-rank-waits side may be
// reached before, after, or concurrently with the smaller-rank-initiates
// side. The bootstrap must not deadlock no matter the relative order.
TEST_F(CtranIbMultiVcPinnedTest, MultiRank4OutOfOrderGetVcsNoDeadlock) {
  int deviceCount = 0;
  EXPECT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
  if (deviceCount < 4) {
    GTEST_SKIP() << "Test requires at least 4 CUDA devices, found "
                 << deviceCount;
  }

  constexpr int kNumRanks = 4;
  runNRanks(
      kNumRanks,
      [](int rank,
         CtranIb* ib,
         const std::vector<SocketServerAddr>& peerAddrs) {
        // Rotate the per-rank peer order so each rank issues connectVcs in
        // a different order:
        //   rank 0: peers (1, 2, 3)  -- always smaller-rank initiator
        //   rank 1: peers (2, 3, 0)  -- initiator for 2,3; waiter for 0
        //   rank 2: peers (3, 0, 1)  -- initiator for 3; waiter for 0,1
        //   rank 3: peers (0, 1, 2)  -- waiter for all
        // Together these exercise the case where a larger-rank waiter
        // is reached before, after, and concurrently with the
        // smaller-rank initiator across multiple peer pairs.
        std::vector<int> peerOrder;
        peerOrder.reserve(kNumRanks - 1);
        for (int i = 1; i < kNumRanks; ++i) {
          peerOrder.push_back((rank + i) % kNumRanks);
        }

        EXPECT_EQ(ib->getMaxVcsPerPeer(), kExpectedMaxVcsPerPeer);

        for (int peerRank : peerOrder) {
          const auto& vcs = ib->connectVcs(peerRank, &peerAddrs[peerRank]);
          ASSERT_EQ(static_cast<int>(vcs.size()), kExpectedMaxVcsPerPeer)
              << "rank " << rank << " peer " << peerRank;
        }

        // A second pass in any order must hit the cache (same vector).
        for (int peerRank : peerOrder) {
          const auto& vcs = ib->connectVcs(peerRank, &peerAddrs[peerRank]);
          const auto& vcs2 = ib->connectVcs(peerRank, &peerAddrs[peerRank]);
          EXPECT_EQ(vcs.data(), vcs2.data())
              << "rank " << rank << " peer " << peerRank;
        }
      });
}

} // namespace
