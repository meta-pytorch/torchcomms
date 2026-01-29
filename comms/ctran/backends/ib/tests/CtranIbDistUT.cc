// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdlib.h>
#include <array>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <gmock/gmock.h>
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/bootstrap/Socket.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ctran::ibvwrap;
using ctran::algos::GpeKernelSync;

extern __global__ void
waitValTestKernel(GpeKernelSync* sync, uint64_t* data, int cmpVal);

commResult_t waitIbReq(CtranIbRequest& req, std::unique_ptr<CtranIb>& ctranIb) {
  do {
    COMMCHECK_TEST(ctranIb->progress());
  } while (!req.isComplete());
  return commSuccess;
}

class CtranIbTest : public ctran::CtranDistTestFixture {
 public:
  CtranIbTest() = default;
  void SetUp() override {
    CtranDistTestFixture::SetUp();
    this->comm_ = makeCtranComm();
    this->comm = this->comm_.get();
    this->ctrlMgr = std::make_unique<CtranCtrlManager>();
    this->commIbRegCount = getIbRegCount();
  }

  void TearDown() override {
    this->ctrlMgr.reset();
    this->comm_.reset();
    CtranDistTestFixture::TearDown();
    ASSERT_EQ(getIbRegCount(), 0);
  }

  void printTestDesc(const std::string& testName, const std::string& testDesc) {
    // NOTE: Printing it as WARN to make this log visible as our default setting
    // is to only print WARN and above logs.
    XLOG_IF(WARN, this->globalRank == 0)
        << testName << " numRanks " << this->numRanks
        << ". Description: " << testDesc << std::endl;
  }

  size_t getIbRegCount() {
    CtranIbSingleton& s = CtranIbSingleton::getInstance();
    return s.getActiveRegCount();
  }

  constexpr static int kSockSyncLen = 16;
  void sockSend(int peerRank) {
    char buf[kSockSyncLen] = "ping";
    auto res = comm->bootstrap_->send(buf, sizeof(buf), peerRank, 0);
    ASSERT_EQ(static_cast<commResult_t>(std::move(res).get()), commSuccess);
  }

  void sockRecv(int peerRank) {
    char buf[kSockSyncLen];
    auto res = comm->bootstrap_->recv(buf, sizeof(buf), peerRank, 0);
    ASSERT_EQ(static_cast<commResult_t>(std::move(res).get()), commSuccess);
  }

  enum class NotifyMode {
    notifyAll, // notify on every put
    notifyLast, // notify only on the last put
    notifyNone // no notify, use socket to sync
  };

  void runPutNotify(
      const size_t bufCount,
      const int numPuts,
      bool localSignal,
      NotifyMode notifyMode,
      bool isGpuMem,
      bool preConnect = false,
      bool issueFastPut = false) {
    if (!ctranIb) {
      try {
        ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());
      } catch (const std::bad_alloc& e) {
        GTEST_SKIP() << "IB backend not enabled. Skip test";
      }
    }

    CtranIbEpochRAII epochRAII(ctranIb.get());

    const int recvVal = -1;

    int* buf = nullptr;
    std::vector<int> hostBuf(bufCount * numPuts, 0);
    void* handle = nullptr;
    CtranIbRequest ctrlReq, putReq;
    ControlMsg msg;

    // fill the buffer with different values and copy to GPU
    for (int i = 0; i < bufCount * numPuts; i++) {
      hostBuf[i] =
          this->globalRank == sendRank ? ((i / bufCount) + 1) : recvVal;
    }

    if (isGpuMem) {
      CUDACHECK_TEST(cudaMalloc(&buf, bufCount * sizeof(int) * numPuts));
      CUDACHECK_TEST(cudaMemcpy(
          buf,
          hostBuf.data(),
          bufCount * sizeof(int) * numPuts,
          cudaMemcpyHostToDevice));
      // Pageable host memory to device memory copy may return before DMA
      // complete. Thus, we need device sync to ensure DMA completion before
      // RDMA. see
      // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync
      CUDACHECK_TEST(cudaDeviceSynchronize());
    } else {
      buf = hostBuf.data();
    }

    // Register and export to a control msg
    COMMCHECK_TEST(
        CtranIb::regMem(
            buf, bufCount * sizeof(int) * numPuts, this->localRank, &handle));
    COMMCHECK_TEST(CtranIb::exportMem(buf, handle, msg));
    ASSERT_EQ(getIbRegCount(), commIbRegCount + 1);

    if (preConnect) {
      std::unordered_set<int> peerRanks;
      if (this->globalRank == recvRank) {
        peerRanks.insert(sendRank);
        COMMCHECK_TEST(ctranIb->preConnect(peerRanks));
      } else if (this->globalRank == sendRank) {
        peerRanks.insert(recvRank);
        COMMCHECK_TEST(ctranIb->preConnect(peerRanks));
      }
    }

    // Receiver sends the remoteAddr and rkey to sender
    if (this->globalRank == recvRank) {
      COMMCHECK_TEST(ctranIb->isendCtrlMsg(
          msg.type, &msg, sizeof(msg), sendRank, ctrlReq));
    } else if (this->globalRank == sendRank) {
      COMMCHECK_TEST(
          ctranIb->irecvCtrlMsg(&msg, sizeof(msg), recvRank, ctrlReq));
    } else {
      // no-op for non-communicating ranks
      COMMCHECK_TEST(ctrlReq.complete());
    }

    // Waits control message to be received
    waitIbReq(ctrlReq, ctranIb);

    std::chrono::system_clock::time_point start =
        std::chrono::system_clock::now();

    // Sender puts the data to rank 1 using remote addr and rkey received in
    // previous irecvCtrlMsg
    if (this->globalRank == sendRank) {
      void* remoteBuf = reinterpret_cast<void*>(msg.ibExp.remoteAddr);
      CtranIbRemoteAccessKey key{};
      for (int i = 0; i < msg.ibExp.nKeys; i++) {
        key.rkeys[i] = msg.ibExp.rkeys[i];
      }

      for (int i = 0; i < numPuts; i++) {
        putReq = CtranIbRequest(); // reset the request
        if (issueFastPut) {
          COMMCHECK_TEST(ctranIb->iput(
              buf + i * bufCount,
              (int*)remoteBuf + i * bufCount,
              bufCount * sizeof(int),
              recvRank,
              handle,
              key,
              notifyMode == NotifyMode::notifyAll ||
                  (i == numPuts - 1 && notifyMode == NotifyMode::notifyLast),
              nullptr,
              localSignal || i == numPuts - 1 ? &putReq : nullptr,
              /* fast */ true));
        } else {
          COMMCHECK_TEST(ctranIb->iput(
              buf + i * bufCount,
              (int*)remoteBuf + i * bufCount,
              bufCount * sizeof(int),
              recvRank,
              handle,
              key,
              // Completely skip notify if notifyMode is notifyNone, to test w/o
              // receiver side progress
              notifyMode == NotifyMode::notifyAll ||
                  (i == numPuts - 1 && notifyMode == NotifyMode::notifyLast),
              nullptr,
              localSignal || i == numPuts - 1 ? &putReq : nullptr));
        }

        if (localSignal || i == numPuts - 1) {
          // waits for put to finish
          COMMCHECK_TEST(waitIbReq(putReq, ctranIb));
        }
      }

      // Notify receiver the completion of all puts via socket if notifyMode
      // is notifyNone. All IB transport modes should support
      // truly one-sided put w/o notify, which doesn't require receiver side to
      // make progress in IB backend.
      if (notifyMode == NotifyMode::notifyNone) {
        sockSend(recvRank);
      }

    } else if (this->globalRank == recvRank) {
      // Receiver waits notify and check data
      // if notifyMode is notifyAll, wait notify for each put; otherwise handle
      // based on mode
      if (notifyMode == NotifyMode::notifyAll) {
        COMMCHECK_TEST(ctranIb->waitNotify(sendRank, numPuts));
        // PCI-e flush to ensure data is immediately visible to GPU
        auto flushReq = CtranIbRequest();
        COMMCHECK_TEST(ctranIb->iflush(buf, handle, &flushReq));
        COMMCHECK_TEST(waitIbReq(flushReq, ctranIb));
      } else if (notifyMode == NotifyMode::notifyNone) {
        // wait on socket to avoid progress in receiver side IB backend
        sockRecv(sendRank);
      } else {
        // notifyMode == NotifyMode::notifyLast
        COMMCHECK_TEST(ctranIb->waitNotify(sendRank));
      }

      // PCI-e flush to ensure data is immediately visible to GPU
      auto flushReq = CtranIbRequest();
      COMMCHECK_TEST(ctranIb->iflush(buf, handle, &flushReq));
      COMMCHECK_TEST(waitIbReq(flushReq, ctranIb));
    }

    std::chrono::system_clock::time_point end =
        std::chrono::system_clock::now();
    if (this->globalRank < 2) {
      printf(
          "Rank %d %s latency: %ld ns\n",
          this->globalRank,
          this->globalRank == 0 ? "Put" : "WaitNotify",
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count() /
              numPuts);
    }

    if (this->globalRank == recvRank) {
      if (isGpuMem) {
        CUDACHECK_TEST(cudaMemcpy(
            hostBuf.data(),
            buf,
            bufCount * sizeof(int) * numPuts,
            cudaMemcpyDeviceToHost));
      }
      for (int i = 0; i < bufCount * numPuts; i++) {
        EXPECT_EQ(hostBuf[i], (i / bufCount) + 1);
      }
    }
    COMMCHECK_TEST(CtranIb::deregMem(handle));
    ASSERT_EQ(getIbRegCount(), commIbRegCount);
    if (isGpuMem) {
      CUDACHECK_TEST(cudaFree(buf));
    }
  }

  void runGet(
      const size_t bufCount,
      const int numGets,
      bool localSignal,
      bool isGpuMem,
      bool issueFastGet = false) {
    if (!ctranIb) {
      try {
        ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());
      } catch (const std::bad_alloc& e) {
        GTEST_SKIP() << "IB backend not enabled. Skip test";
      }
    }

    CtranIbEpochRAII epochRAII(ctranIb.get());

    // In the Get case, dstRank is the rank that initiates the RDMA read.
    // To avoid confusion, since srcRank is the rank on the passive sending
    // side, we use srcRank and dstRank consistently in this test.
    auto srcRank = 1;
    auto dstRank = 0;

    const int srcVal = 99, dstVal = -1;

    int* buf = nullptr;
    std::vector<int> hostBuf(bufCount, 0);
    void* handle = nullptr;
    CtranIbRequest ctrlReq, getReq;
    ControlMsg msg;

    // fill the buffer with different values and copy to GPU
    for (int i = 0; i < bufCount; i++) {
      hostBuf[i] = this->globalRank == srcRank ? srcVal : dstVal;
    }

    if (isGpuMem) {
      CUDACHECK_TEST(cudaMalloc(&buf, bufCount * sizeof(int)));
      CUDACHECK_TEST(cudaMemcpy(
          buf, hostBuf.data(), bufCount * sizeof(int), cudaMemcpyHostToDevice));
      // Pageable host memory to device memory copy may return before DMA
      // complete. Thus, we need device sync to ensure DMA completion before
      // RDMA. see
      // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync
      CUDACHECK_TEST(cudaDeviceSynchronize());
    } else {
      buf = hostBuf.data();
    }

    // Register and export to a control msg
    COMMCHECK_TEST(
        CtranIb::regMem(buf, bufCount * sizeof(int), this->localRank, &handle));
    COMMCHECK_TEST(CtranIb::exportMem(buf, handle, msg));
    ASSERT_EQ(getIbRegCount(), commIbRegCount + 1);

    // Rank whose data will be read sends the remoteAddr and rkey to sender
    if (this->globalRank == srcRank) {
      COMMCHECK_TEST(
          ctranIb->isendCtrlMsg(msg.type, &msg, sizeof(msg), dstRank, ctrlReq));
    } else if (this->globalRank == dstRank) {
      COMMCHECK_TEST(
          ctranIb->irecvCtrlMsg(&msg, sizeof(msg), srcRank, ctrlReq));
    } else {
      // no-op for non-communicating ranks
      COMMCHECK_TEST(ctrlReq.complete());
    }

    // Waits control message to be received
    waitIbReq(ctrlReq, ctranIb);

    std::chrono::system_clock::time_point start =
        std::chrono::system_clock::now();

    // Receiver gets the data to rank 1 using remote addr and rkey received in
    // previous irecvCtrlMsg
    if (this->globalRank == dstRank) {
      void* remoteBuf = reinterpret_cast<void*>(msg.ibExp.remoteAddr);
      CtranIbRemoteAccessKey key{};
      for (int i = 0; i < msg.ibExp.nKeys; i++) {
        key.rkeys[i] = msg.ibExp.rkeys[i];
      }

      for (int i = 0; i < numGets; i++) {
        getReq = CtranIbRequest(); // reset the request
        COMMCHECK_TEST(ctranIb->iget(
            remoteBuf, // remote src buf
            buf, // local dst buf
            bufCount * sizeof(int),
            srcRank,
            handle,
            key,
            nullptr,
            localSignal || i == numGets - 1 ? &getReq : nullptr));

        if (localSignal || i == numGets - 1) {
          // waits for put to finish
          COMMCHECK_TEST(waitIbReq(getReq, ctranIb));
        }
      }

      // Notify receiver the completion of all Gets via socket since we don't
      // have remote notify in Get. All IB transport modes should support truly
      // one-sided get w/o notify, which doesn't require receiver side to make
      // progress in IB backend.
      sockSend(srcRank);

    } else if (this->globalRank == srcRank) {
      // wait on socket to avoid progress in receiver side IB backend
      sockRecv(dstRank);
    }

    std::chrono::system_clock::time_point end =
        std::chrono::system_clock::now();
    if (this->globalRank == dstRank) {
      printf(
          "Rank %d %s latency: %ld ns\n",
          this->globalRank,
          "Get",
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count() /
              numGets);
    }

    if (this->globalRank == dstRank) {
      if (isGpuMem) {
        CUDACHECK_TEST(cudaMemcpy(
            hostBuf.data(),
            buf,
            bufCount * sizeof(int),
            cudaMemcpyDeviceToHost));
      }
      for (int i = 0; i < bufCount; i++) {
        EXPECT_EQ(hostBuf[i], srcVal);
      }
    }
    COMMCHECK_TEST(CtranIb::deregMem(handle));
    ASSERT_EQ(getIbRegCount(), commIbRegCount);
    if (isGpuMem) {
      CUDACHECK_TEST(cudaFree(buf));
    }
  }

  void runNotify(const int numNotifies, bool localSignal) {
    try {
      ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    } catch (const std::bad_alloc& e) {
      GTEST_SKIP() << "IB backend not enabled. Skip test";
    }

    CtranIbEpochRAII epochRAII(ctranIb.get());

    // A dummy control message exchange to ensure QP connection
    ControlMsg msg;
    CtranIbRequest ctrlReq;
    if (this->globalRank == 0) {
      COMMCHECK_TEST(
          ctranIb->isendCtrlMsg(msg.type, &msg, sizeof(msg), 1, ctrlReq));
    } else if (this->globalRank == 1) {
      COMMCHECK_TEST(ctranIb->irecvCtrlMsg(&msg, sizeof(msg), 0, ctrlReq));
    } else {
      // no-op for non-communicating ranks
      COMMCHECK_TEST(ctrlReq.complete());
    }

    waitIbReq(ctrlReq, ctranIb);

    auto start = std::chrono::steady_clock::now();
    if (this->globalRank == 0) {
      for (int i = 0; i < numNotifies; i++) {
        CtranIbRequest req = CtranIbRequest(); // reset before each notify
        //  Issue notify each with signal, or just signal the last one
        COMMCHECK_TEST(ctranIb->notify(
            1, localSignal || i == numNotifies - 1 ? &req : nullptr));

        if (localSignal || i == numNotifies - 1) {
          // waits for any signaled notify to finish
          COMMCHECK_TEST(waitIbReq(req, ctranIb));
        }
      }
    } else if (this->globalRank == 1) {
      COMMCHECK_TEST(ctranIb->waitNotify(0, numNotifies));
    }

    auto end = std::chrono::steady_clock::now();
    if (this->globalRank < 2) {
      printf(
          "Rank %d %s latency: %ld ns\n",
          this->globalRank,
          this->globalRank == 0 ? "Notify" : "WaitNotify",
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count() /
              numNotifies);
    }
  }

  void runPutBatch(
      const size_t bufCount,
      const int numBatches,
      const int batchSize,
      bool withNotify,
      bool mixedNotify,
      bool isGpuMem,
      bool interleavePut = false,
      bool interleaveFastPut = false,
      bool fallbackPut = false,
      bool ibFastPath = true,
      bool verifySeqNum = false) {
    if (!ctranIb) {
      try {
        ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());
      } catch (const std::bad_alloc& e) {
        GTEST_SKIP() << "IB backend not enabled. Skip test";
      }
    }

    CtranIbEpochRAII epochRAII(ctranIb.get());

    const int sendVal = 99, recvVal = -1;

    int* buf = nullptr;
    std::vector<int> hostBuf(bufCount, 0);
    void* handle = nullptr;
    CtranIbRequest ctrlReq;
    ControlMsg msg;

    // fill the buffer with different values and copy to GPU
    for (int i = 0; i < bufCount; i++) {
      hostBuf[i] = this->globalRank == sendRank ? sendVal : recvVal;
    }

    if (isGpuMem) {
      CUDACHECK_TEST(cudaMalloc(&buf, bufCount * sizeof(int)));
      CUDACHECK_TEST(cudaMemcpy(
          buf, hostBuf.data(), bufCount * sizeof(int), cudaMemcpyHostToDevice));
      CUDACHECK_TEST(cudaDeviceSynchronize());
    } else {
      buf = hostBuf.data();
    }

    // Register and export to a control msg
    COMMCHECK_TEST(
        CtranIb::regMem(buf, bufCount * sizeof(int), this->localRank, &handle));
    COMMCHECK_TEST(CtranIb::exportMem(buf, handle, msg));
    ASSERT_EQ(getIbRegCount(), commIbRegCount + 1);

    // Receiver sends the remoteAddr and rkey to sender
    if (this->globalRank == recvRank) {
      COMMCHECK_TEST(ctranIb->isendCtrlMsg(
          msg.type, &msg, sizeof(msg), sendRank, ctrlReq));
    } else if (this->globalRank == sendRank) {
      COMMCHECK_TEST(
          ctranIb->irecvCtrlMsg(&msg, sizeof(msg), recvRank, ctrlReq));
    } else {
      // no-op for non-communicating ranks
      COMMCHECK_TEST(ctrlReq.complete());
    }

    // Waits control message to be received
    waitIbReq(ctrlReq, ctranIb);

    uint32_t startSndNxt = 0;
    uint32_t startRcvNxt = 0;
    if (verifySeqNum) {
      if (this->globalRank == sendRank) {
        auto vc = ctranIb->getVc(recvRank);
        if (vc) {
          startSndNxt = vc->getSndNxt();
        }
      } else if (this->globalRank == recvRank) {
        auto vc = ctranIb->getVc(sendRank);
        if (vc) {
          startRcvNxt = vc->getRcvNxt();
        }
      }
    }

    std::chrono::system_clock::time_point start =
        std::chrono::system_clock::now();

    // Calculate expected notifications (same logic for both sender and
    // receiver)
    size_t expectedNotifications = 0;
    if (mixedNotify) {
      // For mixed notify, every other message has notify set (starting with
      // first)
      expectedNotifications = (batchSize + 1) / 2 * numBatches;
    } else if (withNotify) {
      expectedNotifications = batchSize * numBatches;
    }
    if (interleavePut) {
      expectedNotifications += (numBatches - 1);
    }
    if (interleaveFastPut) {
      expectedNotifications += (numBatches - 1);
    }

    size_t totalNotifications = 0;
    // Sender puts the data using putBatch and optionally interleaved regular
    // puts
    std::vector<std::unique_ptr<CtranIbRequest>> requests;
    if (this->globalRank == sendRank) {
      void* remoteBuf = reinterpret_cast<void*>(msg.ibExp.remoteAddr);
      CtranIbRemoteAccessKey key{};
      for (int i = 0; i < msg.ibExp.nKeys; i++) {
        key.rkeys[i] = msg.ibExp.rkeys[i];
      }

      CtranIbRequest fallbackPutReq;
      if (fallbackPut) {
        // create a outstanding put to make the fast puts invalid
        COMMCHECK_TEST(ctranIb->iput(
            buf,
            remoteBuf,
            bufCount * sizeof(int),
            recvRank,
            handle,
            key,
            false, // notify
            nullptr,
            &fallbackPutReq));
      }

      for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        // Create batch of puts
        std::vector<PutIbMsg> putBatch;
        putBatch.reserve(batchSize);

        for (int i = 0; i < batchSize; i++) {
          PutIbMsg putMsg;
          putMsg.sbuf = buf;
          putMsg.dbuf = remoteBuf;
          putMsg.len = bufCount * sizeof(int);
          putMsg.ibRegElem = handle;
          putMsg.remoteAccessKey = key;

          // Test mixed notify behavior
          if (mixedNotify) {
            putMsg.notify = (i % 2 == 0); // alternate notify/no-notify
          } else {
            putMsg.notify = withNotify;
          }
          if (putMsg.notify) {
            totalNotifications++;
          }

          putMsg.config = nullptr;
          if (i == batchSize - 1) {
            requests.emplace_back(std::make_unique<CtranIbRequest>());
            putMsg.req = requests.back().get();
          } else {
            putMsg.req = nullptr; // No local signal for batch operations
          }
          putMsg.ibFastPath = ibFastPath;

          putBatch.push_back(putMsg);
        }

        // Issue putBatch
        COMMCHECK_TEST(ctranIb->iputBatch(putBatch, recvRank));

        // Optionally interleave with regular put
        if (interleavePut && batchIdx < numBatches - 1) {
          CtranIbRequest putReq;
          COMMCHECK_TEST(ctranIb->iput(
              buf,
              remoteBuf,
              bufCount * sizeof(int),
              recvRank,
              handle,
              key,
              true, // notify
              nullptr,
              &putReq));
          totalNotifications++;
          COMMCHECK_TEST(waitIbReq(putReq, ctranIb));
        }

        // Optionally interleave with fast put
        if (interleaveFastPut && batchIdx < numBatches - 1) {
          CtranIbRequest fastPutReq;
          COMMCHECK_TEST(ctranIb->iput(
              buf,
              remoteBuf,
              bufCount * sizeof(int),
              recvRank,
              handle,
              key,
              true, // notify (required for fast put)
              nullptr,
              &fastPutReq,
              true)); // fast
          totalNotifications++;
          COMMCHECK_TEST(waitIbReq(fastPutReq, ctranIb));
        }
      }

      CHECK(expectedNotifications == totalNotifications)
          << "This is a bug in the test";
      // wait for all batches to finish
      if (fallbackPut) {
        waitIbReq(fallbackPutReq, ctranIb);
      }
      while (!requests.empty()) {
        waitIbReq(*requests.back(), ctranIb);
        requests.pop_back();
      }

      if (verifySeqNum) {
        auto vc = ctranIb->getVc(recvRank);
        if (vc) {
          uint32_t endSndNxt = vc->getSndNxt();
          uint32_t diff = (endSndNxt - startSndNxt) & 0xFFFFFF;
          size_t expectedDiff = 0;
          if (ibFastPath) {
            expectedDiff += numBatches * batchSize;
          }
          if (interleaveFastPut) {
            expectedDiff += (numBatches - 1);
          }
          EXPECT_EQ(diff, expectedDiff) << "Sender SeqNum mismatch";
        }
      }

      // add a barrier
      sockRecv(recvRank);
    } else if (this->globalRank == recvRank) {
      // Wait for sender to finish all putBatch operations

      // Receiver waits for notifications
      if (expectedNotifications > 0) {
        COMMCHECK_TEST(ctranIb->waitNotify(sendRank, expectedNotifications));

        // PCI-e flush to ensure data is immediately visible to GPU
        auto flushReq = CtranIbRequest();
        COMMCHECK_TEST(ctranIb->iflush(buf, handle, &flushReq));
        COMMCHECK_TEST(waitIbReq(flushReq, ctranIb));
      }

      if (verifySeqNum) {
        auto vc = ctranIb->getVc(sendRank);
        if (vc) {
          uint32_t endRcvNxt = vc->getRcvNxt();
          uint32_t diff = (endRcvNxt - startRcvNxt) & 0xFFFFFF;
          size_t expectedDiff = 0;
          if (ibFastPath) {
            expectedDiff += numBatches * batchSize;
          }
          if (interleaveFastPut) {
            expectedDiff += (numBatches - 1);
          }
          EXPECT_EQ(diff, expectedDiff) << "Receiver SeqNum mismatch";
        }
      }

      // add a barrier
      sockSend(sendRank);
    }

    std::chrono::system_clock::time_point end =
        std::chrono::system_clock::now();
    if (this->globalRank < 2) {
      printf(
          "Rank %d %s latency: %ld ns\n",
          this->globalRank,
          this->globalRank == 0 ? "PutBatch" : "WaitNotify",
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count() /
              (numBatches * batchSize));
    }

    if (this->globalRank == recvRank) {
      if (isGpuMem) {
        CUDACHECK_TEST(cudaMemcpy(
            hostBuf.data(),
            buf,
            bufCount * sizeof(int),
            cudaMemcpyDeviceToHost));
      }
      for (int i = 0; i < bufCount; i++) {
        EXPECT_EQ(hostBuf[i], sendVal);
      }
    }

    COMMCHECK_TEST(CtranIb::deregMem(handle));
    ASSERT_EQ(getIbRegCount(), commIbRegCount);
    if (isGpuMem) {
      CUDACHECK_TEST(cudaFree(buf));
    }
  }

  void runAtomicOps(
      bool localSignal = false,
      bool isGpuMem = false,
      bool isFetchAdd = true) {
    try {
      ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    } catch (const std::bad_alloc& e) {
      GTEST_SKIP() << "IB backend not enabled. Skip test";
    }

    CtranIbEpochRAII epochRAII(ctranIb.get());

    const int dstRank = numRanks - 1;
    int numSenders = numRanks - 1;
    const int bufCount = 4096;
    uint64_t* buf = nullptr;
    std::vector<uint64_t> hostBuf(bufCount, 0);
    for (int i = 0; i < bufCount; i++) {
      hostBuf[i] = 0;
    }

    if (isGpuMem) {
      CUDACHECK_TEST(cudaMalloc(&buf, bufCount * sizeof(uint64_t)));
      CUDACHECK_TEST(cudaMemcpy(
          (void*)buf,
          (void*)hostBuf.data(),
          bufCount * sizeof(uint64_t),
          cudaMemcpyHostToDevice));
    } else {
      buf = (uint64_t*)hostBuf.data();
    }

    void* syncPtr = nullptr;
    ASSERT_EQ(
        cudaHostAlloc(&syncPtr, sizeof(GpeKernelSync), cudaHostAllocDefault),
        cudaSuccess);
    GpeKernelSync* sync = reinterpret_cast<GpeKernelSync*>(syncPtr);
    new (sync) GpeKernelSync(1);
    uint64_t* expValues = nullptr;
    ASSERT_EQ(
        cudaHostAlloc(
            &expValues, sizeof(uint64_t) * numRanks, cudaHostAllocDefault),
        cudaSuccess);

    void* handle = nullptr;
    CtranIbRequest ctrlReq, atomicReq;
    ControlMsg msg;

    // Register and export to a control msg
    COMMCHECK_TEST(
        CtranIb::regMem(
            buf, bufCount * sizeof(uint64_t), this->localRank, &handle));
    COMMCHECK_TEST(CtranIb::exportMem(buf, handle, msg));
    ASSERT_EQ(getIbRegCount(), commIbRegCount + 1);

    // Receiver sends the remoteAddr and rkey to sender
    if (this->globalRank == dstRank) {
      for (auto rank = 0; rank < numSenders; rank++) {
        ctrlReq = CtranIbRequest();
        COMMCHECK_TEST(
            ctranIb->isendCtrlMsg(msg.type, &msg, sizeof(msg), rank, ctrlReq));
        waitIbReq(ctrlReq, ctranIb);
      }
    } else {
      COMMCHECK_TEST(
          ctranIb->irecvCtrlMsg(&msg, sizeof(msg), dstRank, ctrlReq));
      waitIbReq(ctrlReq, ctranIb);
    }

    std::chrono::system_clock::time_point start =
        std::chrono::system_clock::now();
    const int numOps = 100;
    if (isFetchAdd) {
      // Atomic FAdd testing
      int finalSum = numOps * (numRanks - 1);
      if (this->globalRank != dstRank) {
        void* remoteBuf = reinterpret_cast<void*>(msg.ibExp.remoteAddr);
        CtranIbRemoteAccessKey key{};
        for (int i = 0; i < msg.ibExp.nKeys; i++) {
          key.rkeys[i] = msg.ibExp.rkeys[i];
        }
        for (int i = 0; i < numOps; i++) {
          atomicReq = CtranIbRequest();
          COMMCHECK_TEST(ctranIb->ifetchAndAdd(
              buf,
              remoteBuf,
              1,
              dstRank,
              handle,
              key,
              localSignal || (i == numOps - 1) ? &atomicReq : nullptr));
          if (localSignal || (i == numOps - 1)) {
            COMMCHECK_TEST(waitIbReq(atomicReq, ctranIb));
          }
        }
      } else {
        if (isGpuMem) {
          std::array<void*, 3> kernArgs;
          dim3 grid = {1, 1, 1};
          dim3 blocks = {1, 1, 1};
          kernArgs.at(0) = &sync;
          kernArgs.at(1) = &buf;
          kernArgs.at(2) = &finalSum;
          ASSERT_EQ(
              cudaLaunchKernel(
                  (const void*)waitValTestKernel,
                  grid,
                  blocks,
                  kernArgs.data(),
                  0,
                  0),
              cudaSuccess);
          while (!sync->isComplete(0)) {
            std::this_thread::yield();
          }
        } else {
          std::atomic<uint64_t>* addr =
              reinterpret_cast<std::atomic<uint64_t>*>(buf);
          while (std::atomic_load(addr) != finalSum) {
            std::this_thread::yield();
          }
        }
      }
    } else {
      // Atomic Set testing
      void* remoteBuf = reinterpret_cast<void*>(msg.ibExp.remoteAddr);
      CtranIbRemoteAccessKey key{};
      for (int i = 0; i < msg.ibExp.nKeys; i++) {
        key.rkeys[i] = msg.ibExp.rkeys[i];
      }
      for (int i = 0; i < numOps; i++) {
        // In every iteration, all senders atomic-set the same address in the
        // last rank and notify it. The last rank will check whether the final
        // value is from any of the senders
        if (this->globalRank != dstRank) {
          ctrlReq = CtranIbRequest();
          COMMCHECK_TEST(ctranIb->iatomicSet(
              remoteBuf, globalRank * 1000 + i, dstRank, key, nullptr));
          COMMCHECK_TEST(ctranIb->notify(dstRank, &ctrlReq));
          COMMCHECK_TEST(waitIbReq(ctrlReq, ctranIb));
          COMMCHECK_TEST(ctranIb->waitNotify(dstRank));
        } else {
          for (auto rank = 0; rank < numSenders; rank++) {
            COMMCHECK_TEST(ctranIb->waitNotify(rank));
            expValues[rank] = rank * 1000 + i;
          }
          // Confirm that the loaded value is from any sender rank
          uint64_t val;
          if (isGpuMem) {
            CUDACHECK_TEST(cudaMemcpy(
                (void*)hostBuf.data(),
                (void*)buf,
                bufCount * sizeof(uint64_t),
                cudaMemcpyDeviceToHost));
            val = hostBuf[0];
          } else {
            val = buf[0];
          }
          bool passCheck = false;
          for (int rank = 0; rank < numSenders; rank++) {
            if (val == expValues[rank]) {
              passCheck = true;
              break;
            }
          }
          ASSERT_EQ(passCheck, true);
          std::vector<CtranIbRequest> reqs;
          reqs.resize(numSenders, CtranIbRequest());
          for (auto rank = 0; rank < numSenders; rank++) {
            COMMCHECK_TEST(ctranIb->notify(rank, &reqs[rank]));
          }
          for (auto& req : reqs) {
            COMMCHECK_TEST(waitIbReq(req, ctranIb));
          }
        }
      }
    }
    std::chrono::system_clock::time_point end =
        std::chrono::system_clock::now();
    printf(
        "Rank %d %s %s latency: %ld ns\n",
        globalRank,
        isFetchAdd ? "FAdd" : "Set",
        globalRank != dstRank ? "writer" : "reader",
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count() /
            numOps);

    ASSERT_EQ(cudaFreeHost(syncPtr), cudaSuccess);
    ASSERT_EQ(cudaFreeHost(expValues), cudaSuccess);
    COMMCHECK_TEST(CtranIb::deregMem(handle));
  }

 protected:
  std::unique_ptr<CtranComm> comm_{nullptr};
  CtranComm* comm{nullptr};
  std::unique_ptr<CtranIb> ctranIb{nullptr};
  std::unique_ptr<CtranCtrlManager> ctrlMgr{nullptr};
  size_t commIbRegCount{0};
  const int sendRank{0}, recvRank{1};
};

TEST_F(CtranIbTest, NormalInitialize) {
  this->printTestDesc(
      "NormalInitialize",
      "Expect CtranIb to be initialized without internal error.");

  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

TEST_F(CtranIbTest, InitializeWithoutComm) {
  const std::string eth = "eth0";
  EnvRAII env1(NCCL_SOCKET_IFNAME, eth);
  this->printTestDesc(
      "InitializeWithoutComm",
      "Expect CtranIb to be initialized without internal error.");

  const auto& rank = this->comm->statex_->rank();
  const auto& cudaDev = this->comm->statex_->cudaDev();
  const auto& commHash = this->comm->statex_->commHash();
  const auto& commDesc = this->comm->config_.commDesc;

  auto maybeAddr = ctran::bootstrap::getInterfaceAddress(
      NCCL_SOCKET_IFNAME, NCCL_SOCKET_IPADDR_PREFIX);
  ASSERT_FALSE(maybeAddr.hasError());

  // Create server socket, and bind it to reserve the port. The subsequent
  // test can use that port. The socket object will be destroyed (& port
  // released) when it goes out of scope.
  ctran::bootstrap::ServerSocket serverSocket(1);
  serverSocket.bind(
      folly::SocketAddress(*maybeAddr, 0), NCCL_SOCKET_IFNAME, true);
  int port = serverSocket.getListenAddress()->getPort();
  SocketServerAddr qpServerAddr{
      .port = port,
      .ipv6 = maybeAddr->str(),
      .ifName = NCCL_SOCKET_IFNAME // use same ifname as server
  };

  std::unique_ptr<CtranIb> ctranIb{nullptr};
  try {
    ctranIb = std::make_unique<CtranIb>(
        rank,
        cudaDev,
        commHash,
        commDesc,
        this->ctrlMgr.get(),
        true /*enableLocalFlush*/,
        CtranIb::BootstrapMode::kSpecifiedServer,
        &qpServerAddr);
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }

  // test send/recv control message
  // i.e. bootstrap connect/accept with provided ip/port
  struct SocketServerAddrTmp {
    int port{-1};
    char ipv6[1024];
    char ifName[64];
  };
  SocketServerAddrTmp qpServerAddrTmp{
      .port = qpServerAddr.port,
  };
  strcpy(qpServerAddrTmp.ipv6, qpServerAddr.ipv6.c_str());
  strcpy(qpServerAddrTmp.ifName, qpServerAddr.ifName.c_str());
  std::vector<SocketServerAddrTmp> qpServerAddrs(this->numRanks);
  qpServerAddrs[this->globalRank] = qpServerAddrTmp;
  auto resFuture = comm->bootstrap_->allGather(
      qpServerAddrs.data(),
      sizeof(SocketServerAddrTmp),
      comm->statex_->rank(),
      comm->statex_->nRanks());
  COMMCHECK_TEST(static_cast<commResult_t>(std::move(resFuture).get()));

  CtranIbRequest req;
  ControlMsg smsg(ControlMsgType::IB_EXPORT_MEM);
  ControlMsg rmsg(ControlMsgType::IB_EXPORT_MEM);

  CtranIbEpochRAII epochRAII(ctranIb.get());

  smsg.ibExp.remoteAddr = 99;
  smsg.ibExp.rkeys[0] = 1;
  smsg.ibExp.nKeys = 1;
  SocketServerAddr remoteAddr;
  if (this->globalRank == 0) {
    remoteAddr.port = qpServerAddrs[1].port;
    remoteAddr.ipv6.assign(qpServerAddrs[1].ipv6);
    remoteAddr.ifName.assign(qpServerAddrs[1].ifName);
    COMMCHECK_TEST(ctranIb->isendCtrlMsg(
        smsg.type, &smsg, sizeof(smsg), 1, req, &remoteAddr));
  } else if (this->globalRank == 1) {
    remoteAddr.port = qpServerAddrs[0].port;
    remoteAddr.ipv6.assign(qpServerAddrs[0].ipv6);
    remoteAddr.ifName.assign(qpServerAddrs[0].ifName);
    COMMCHECK_TEST(
        ctranIb->irecvCtrlMsg(&rmsg, sizeof(rmsg), 0, req, &remoteAddr));
  } else {
    // no-op for non-communicating ranks
    COMMCHECK_TEST(req.complete());
  }

  waitIbReq(req, ctranIb);

  if (this->globalRank == 1) {
    EXPECT_EQ(rmsg.ibExp.rkeys[0], smsg.ibExp.rkeys[0]);
    EXPECT_EQ(rmsg.ibExp.remoteAddr, smsg.ibExp.remoteAddr);
  }
}

TEST_F(CtranIbTest, InitializeWithoutCommAndExternalBootstrap) {
  const std::string eth = "eth0";
  EnvRAII env1(NCCL_SOCKET_IFNAME, eth);
  this->printTestDesc(
      "InitializeWithoutCommAndExternalBootstrap",
      "Expect CtranIb to be initialized without internal error.");

  const auto& rank = this->comm->statex_->rank();
  const auto& cudaDev = this->comm->statex_->cudaDev();
  const auto& commHash = this->comm->statex_->commHash();
  const auto& commDesc = this->comm->config_.commDesc;

  std::unique_ptr<CtranIb> ctranIb{nullptr};
  try {
    ctranIb = std::make_unique<CtranIb>(
        rank,
        cudaDev,
        commHash,
        commDesc,
        this->ctrlMgr.get(),
        false /*enableLocalFlush*/,
        CtranIb::BootstrapMode::kExternal);
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }

  ctranIb.reset();
}

TEST_F(CtranIbTest, RegMem) {
  this->printTestDesc(
      "RegMem",
      "Expect RegMem and deregMem can be finished without internal error.");

  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    size_t len = 2048576;
    constexpr int numThreads = 10;
    std::vector<void*> bufs(numThreads, nullptr);

    for (int i = 0; i < numThreads; i++) {
      CUDACHECK_TEST(cudaMalloc(&bufs[i], len));
    }

    // Stress regMem by multiple threads
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; i++) {
      std::thread t(
          [&](int tid) {
            void* handle = nullptr;
            COMMCHECK_TEST(
                CtranIb::regMem(bufs[tid], len, this->localRank, &handle));
            EXPECT_NE(handle, nullptr);

            COMMCHECK_TEST(CtranIb::deregMem(handle));
          },
          i);
      threads.push_back(std::move(t));
    }

    for (auto& t : threads) {
      t.join();
    }

    for (int i = 0; i < numThreads; i++) {
      CUDACHECK_TEST(cudaFree(bufs[i]));
    }
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

TEST_F(CtranIbTest, ExportMem) {
  this->printTestDesc(
      "ExportMem",
      "Expect ExportMem generates control message with the correct content.");

  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    void* buf = nullptr;
    size_t len = 2048576;
    void* handle = nullptr;
    ControlMsg msg;

    CUDACHECK_TEST(cudaMalloc(&buf, len));
    COMMCHECK_TEST(CtranIb::regMem(buf, len, this->localRank, &handle));
    EXPECT_NE(handle, nullptr);
    ASSERT_EQ(getIbRegCount(), commIbRegCount + 1);

    COMMCHECK_TEST(CtranIb::exportMem(buf, handle, msg));
    EXPECT_EQ(msg.type, ControlMsgType::IB_EXPORT_MEM);
    EXPECT_EQ(msg.ibExp.remoteAddr, reinterpret_cast<uint64_t>(buf));
    auto mrs = reinterpret_cast<std::vector<ibverbx::ibv_mr*>*>(handle);
    EXPECT_EQ(msg.ibExp.rkeys[0], (*mrs).at(0)->rkey);

    auto ibRemoteAccesskeys = CtranIb::getRemoteAccessKey(handle);
    EXPECT_EQ(ibRemoteAccesskeys.nKeys, NCCL_CTRAN_IB_DEVICES_PER_RANK);
    for (auto i = 0; i < NCCL_CTRAN_IB_DEVICES_PER_RANK; i++) {
      EXPECT_EQ(ibRemoteAccesskeys.rkeys[i], (*mrs).at(i)->rkey);
    }

    COMMCHECK_TEST(CtranIb::deregMem(handle));
    ASSERT_EQ(getIbRegCount(), commIbRegCount);
    CUDACHECK_TEST(cudaFree(buf));
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

TEST_F(CtranIbTest, SmallRegMem) {
  this->printTestDesc("SmallRegMem", "Expect RegMem succeeds with small size");

  // Expect registration fails due to small size <= pageSize
  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());

    for (size_t len : {4096, 512}) {
      void* handle = nullptr;
      void* buf = nullptr;
      commResult_t res;

      CUDACHECK_TEST(cudaMalloc(&buf, len));
      res = CtranIb::regMem(buf, len, this->localRank, &handle);
      EXPECT_EQ(res, commSuccess);
      EXPECT_NE(handle, nullptr);
      ASSERT_EQ(getIbRegCount(), commIbRegCount + 1);
      COMMCHECK_TEST(CtranIb::deregMem(handle));
      CUDACHECK_TEST(cudaFree(buf));
    }
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

TEST_F(CtranIbTest, CtrlMsg) {
  this->printTestDesc(
      "SendRecvCtrlMsg",
      "Expect rank 2 can issue multiple send control msgs to ranks 0 and 1, and match to the corresponding recvs");

  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    std::vector<CtranIbRequest> reqs;
    std::vector<ControlMsg> smsgs;
    ControlMsg rmsg0(ControlMsgType::IB_EXPORT_MEM);
    ControlMsg rmsg1(ControlMsgType::IB_EXPORT_MEM);

    if (this->numRanks < 3) {
      GTEST_SKIP() << "Need at least 3 ranks to run this test";
    }

    CtranIbEpochRAII epochRAII(ctranIb.get());
    // Choose largest rank as sender to test bootstrap + pendingOps logic;
    // The larger one will be connected via ListenThread and has to put
    // ctrlMsg into pendingOps
    const int sendRank = 2, recvRank0 = 0, recvRank1 = 1;

    if (this->globalRank == sendRank) {
      reqs.resize(3, CtranIbRequest());
      smsgs.resize(3, ControlMsg(ControlMsgType::IB_EXPORT_MEM));
      // send two msgs to rank 1
      smsgs[0].ibExp.remoteAddr = 99;
      smsgs[0].ibExp.rkeys[0] = recvRank0;
      smsgs[0].ibExp.rkeys[1] = recvRank0;
      smsgs[0].ibExp.nKeys = 2;
      COMMCHECK_TEST(ctranIb->isendCtrlMsg(
          smsgs[0].type, &smsgs[0], sizeof(smsgs[0]), recvRank0, reqs[0]));

      // let recvRank0 connected via ListenThread first; thus the next
      // isendCtrlMsg shall be directly posted. Expect the two msgs are
      // arrived in order
      sleep(2);

      smsgs[1].ibExp.remoteAddr = 100;
      smsgs[1].ibExp.rkeys[0] = recvRank0;
      smsgs[1].ibExp.rkeys[1] = recvRank0;
      smsgs[1].ibExp.nKeys = 2;

      COMMCHECK_TEST(ctranIb->isendCtrlMsg(
          smsgs[1].type, &smsgs[1], sizeof(smsgs[1]), recvRank0, reqs[1]));

      // send one msg to rank 2
      smsgs[2].ibExp.remoteAddr = 101;
      smsgs[2].ibExp.rkeys[0] = recvRank1;
      smsgs[2].ibExp.rkeys[1] = recvRank1;
      smsgs[2].ibExp.nKeys = 2;
      COMMCHECK_TEST(ctranIb->isendCtrlMsg(
          smsgs[2].type, &smsgs[2], sizeof(smsgs[2]), recvRank1, reqs[2]));
    } else if (this->globalRank == recvRank0) {
      reqs.resize(2, CtranIbRequest());
      sleep(1); // let sendRank put msgs into pendingOps first

      // receive two msgs from rank 0; assuming receive in order
      COMMCHECK_TEST(
          ctranIb->irecvCtrlMsg(&rmsg0, sizeof(rmsg0), sendRank, reqs[0]));
      COMMCHECK_TEST(
          ctranIb->irecvCtrlMsg(&rmsg1, sizeof(rmsg1), sendRank, reqs[1]));
    } else if (this->globalRank == recvRank1) {
      reqs.resize(1, CtranIbRequest());

      // receive one msg from rank 0
      COMMCHECK_TEST(
          ctranIb->irecvCtrlMsg(&rmsg0, sizeof(rmsg0), sendRank, reqs[0]));
    }

    for (auto& req : reqs) {
      waitIbReq(req, ctranIb);
    }

    if (this->globalRank == recvRank0) {
      EXPECT_EQ(rmsg0.ibExp.rkeys[0], recvRank0);
      EXPECT_EQ(rmsg0.ibExp.rkeys[1], recvRank0);
      EXPECT_EQ(rmsg0.ibExp.nKeys, 2);
      EXPECT_EQ(rmsg0.ibExp.remoteAddr, 99);
      EXPECT_EQ(rmsg1.ibExp.rkeys[0], recvRank0);
      EXPECT_EQ(rmsg1.ibExp.rkeys[1], recvRank0);
      EXPECT_EQ(rmsg1.ibExp.nKeys, 2);
      EXPECT_EQ(rmsg1.ibExp.remoteAddr, 100);
    } else if (this->globalRank == recvRank1) {
      EXPECT_EQ(rmsg0.ibExp.rkeys[0], recvRank1);
      EXPECT_EQ(rmsg0.ibExp.rkeys[1], recvRank1);
      EXPECT_EQ(rmsg0.ibExp.nKeys, 2);
      EXPECT_EQ(rmsg0.ibExp.remoteAddr, 101);
    }
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

TEST_F(CtranIbTest, MatchAnyCtrlMsg) {
  this->printTestDesc(
      "MatchAnyCtrlMsg",
      "Expect rank 0 can issue a send control msg to rank 1 and matches to the UNSPECIFIED recv on rank1");

  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    const int nCtrl = 300; // exceed MAX_CONTROL_MSGS
    std::vector<CtranIbRequest> reqs(nCtrl);
    std::vector<ControlMsg> smsgs(nCtrl);
    std::vector<ControlMsg> rmsgs(nCtrl);

    CtranIbEpochRAII epochRAII(ctranIb.get());

    for (int i = 0; i < nCtrl; i++) {
      auto& smsg = smsgs[i];
      auto& rmsg = rmsgs[i];
      auto& req = reqs[i];
      smsg.setType(ControlMsgType::IB_EXPORT_MEM);
      smsg.ibExp.remoteAddr = 99;
      smsg.ibExp.rkeys[0] = i + 1;
      smsg.ibExp.nKeys = 1;
      rmsg.setType(ControlMsgType::UNSPECIFIED);
      rmsg.ibExp.remoteAddr = 0;
      rmsg.ibExp.rkeys[0] = 0;
      rmsg.ibExp.nKeys = 1;

      if (this->globalRank == 0) {
        COMMCHECK_TEST(
            ctranIb->isendCtrlMsg(smsg.type, &smsg, sizeof(smsg), 1, req));
      } else if (this->globalRank == 1) {
        COMMCHECK_TEST(ctranIb->irecvCtrlMsg(&rmsg, sizeof(rmsg), 0, req));
      } else {
        // no-op for non-communicating ranks
        COMMCHECK_TEST(req.complete());
      }
    }

    for (int i = 0; i < nCtrl; i++) {
      waitIbReq(reqs[i], ctranIb);

      if (this->globalRank == 1) {
        auto& rmsg = rmsgs[i];
        EXPECT_EQ(rmsg.type, ControlMsgType::IB_EXPORT_MEM);
        EXPECT_EQ(rmsg.ibExp.rkeys[0], i + 1);
        EXPECT_EQ(rmsg.ibExp.remoteAddr, 99);
        EXPECT_EQ(rmsg.ibExp.nKeys, 1);
      }
    }
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

namespace {
constexpr int testRkey = 9;
constexpr uint64_t testRemoteAddr = 100;
bool testCbFlag = false;
commResult_t testCtrlMsgCb(int peer, void* msgPtr, void* ctx) {
  bool* testCbFlagPtr = reinterpret_cast<bool*>(ctx);
  *testCbFlagPtr = true;
  EXPECT_EQ(peer, 0);

  auto msg = reinterpret_cast<ControlMsg*>(msgPtr);
  EXPECT_EQ(msg->type, ControlMsgType::IB_EXPORT_MEM);
  EXPECT_EQ(msg->ibExp.rkeys[0], testRkey);
  EXPECT_EQ(msg->ibExp.remoteAddr, testRemoteAddr);
  return commSuccess;
}
} // namespace

TEST_F(CtranIbTest, CbCtrlMsg) {
  this->printTestDesc(
      "CbCtrlMsg",
      "Expect rank 0 can issue a send control msg that triggers corresponding callback on rank 1");

  try {
    // Register callback
    this->ctrlMgr->regCb(
        ControlMsgType::IB_EXPORT_MEM, testCtrlMsgCb, &testCbFlag);

    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    CtranIbRequest req;
    ControlMsg smsg(ControlMsgType::IB_EXPORT_MEM);

    CtranIbEpochRAII epochRAII(ctranIb.get());

    smsg.ibExp.remoteAddr = testRemoteAddr;
    smsg.ibExp.rkeys[0] = testRkey;
    smsg.ibExp.nKeys = 1;
    if (this->globalRank == 0) {
      COMMCHECK_TEST(
          ctranIb->isendCtrlMsg(smsg.type, &smsg, sizeof(smsg), 1, req));

      // Wait until send finishes
      waitIbReq(req, ctranIb);
    } else if (this->globalRank == 1) {
      // Wait until callback is triggered
      do {
        COMMCHECK_TEST(ctranIb->progress());
      } while (!testCbFlag);
    } else {
      // no-op for non-communicating ranks
      COMMCHECK_TEST(req.complete());
    }

  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

TEST_F(CtranIbTest, LocalFlush) {
  this->printTestDesc("LocalFlush", "Expect rank 0 can issue a local flush");

  try {
    auto ctranIb = std::make_unique<CtranIb>(
        globalRank,
        localRank,
        0,
        "ib_dist_test",
        this->ctrlMgr.get(),
        true /*enableLocalFlush*/,
        CtranIb::BootstrapMode::kDefaultServer);

    CtranIbRequest req;

    CtranIbEpochRAII epochRAII(ctranIb.get());
    constexpr int kBufSize = 8192;
    void* buf = nullptr;
    void* handle = nullptr;
    CUDACHECK_TEST(cudaMalloc(&buf, kBufSize));
    COMMCHECK_TEST(CtranIb::regMem(buf, kBufSize, this->localRank, &handle));

    COMMCHECK_TEST(ctranIb->iflush(buf, handle, &req));
    waitIbReq(req, ctranIb);

    COMMCHECK_TEST(CtranIb::deregMem(handle));
    CUDACHECK_TEST(cudaFree(buf));
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

class CtranIbTestParam
    : public CtranIbTest,
      public ::testing::WithParamInterface<enum NCCL_CTRAN_IB_VC_MODE> {};

TEST_P(CtranIbTestParam, CpuMemPutNotify) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "CpuMemPutNotify",
      "Expect rank 0 can put data from its local CPU data to rank 1 who waits on notify. "
      "The received data should be equal to send data on rank 0.");

  runPutNotify(
      /* bufCount */ 8192,
      /* numPuts*/ 1000,
      /*localSignal*/ true,
      NotifyMode::notifyAll,
      /*isGpuMem*/ false);
}

TEST_P(CtranIbTestParam, GpuMemPutNotify) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemPutNotify",
      "Expect rank 0 can issue multiple puts from its local GPU data to rank 1 who waits on notify. "
      "The received data should be equal to send data on rank 0.");
  runPutNotify(
      /* bufCount */ 8192,
      /* numPuts*/ 1000,
      /*localSignal*/ true,
      NotifyMode::notifyAll,
      /*isGpuMem*/ true);
}

TEST_P(CtranIbTestParam, CpuMemGet) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "CpuMemGet",
      "Expect rank 0 can get data from rank 1 to its local CPU data. "
      "The received data should be equal to send data on rank 1.");

  runGet(
      /* bufCount */ 8192,
      /* numGets*/ 1000,
      /*localSignal*/ true,
      /*isGpuMem*/ false);
}

TEST_P(CtranIbTestParam, GpuMemGet) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemGet",
      "Expect rank 0 can issue multiple gets from rank 1 to its local GPU data. "
      "The received data should be equal to send data on rank 1.");
  runGet(
      /* bufCount */ 8192,
      /* numGets*/ 1000,
      /*localSignal*/ true,
      /*isGpuMem*/ true);
}

TEST_P(CtranIbTestParam, GpuMemGetFast) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemGetFast",
      "Expect rank 0 can issue multiple fast get from rank 1 to its local GPU data."
      "The received data should be equal to send data on rank 0.");
  runGet(
      /* bufCount */ NCCL_CTRAN_IB_QP_SCALING_THRESHOLD / sizeof(int),
      /* numGets*/ NCCL_CTRAN_IB_QP_MAX_MSGS,
      /*localSignal*/ true,
      /*isGpuMem*/ true,
      /*issueFastGet*/ true);
}

TEST_P(CtranIbTestParam, GpuMemPutNotifyFast) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemPutNotifyFast",
      "Expect rank 0 can issue multiple fast puts from its local GPU data to rank 1 who waits on notify. "
      "The received data should be equal to send data on rank 0.");
  runPutNotify(
      /* bufCount */ NCCL_CTRAN_IB_QP_SCALING_THRESHOLD / sizeof(int),
      /* numPuts*/ NCCL_CTRAN_IB_QP_MAX_MSGS,
      /*localSignal*/ true,
      NotifyMode::notifyAll,
      /*isGpuMem*/ true,
      /*preConnect*/ false,
      /*issueFastPut*/ true);
}

TEST_P(CtranIbTestParam, GpuMemPutNotifyLastFast) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemPutNotifyLastFast",
      "Expect rank 0 can issue multiple fast puts from its local GPU data to rank 1 who waits on 1 notify. "
      "The received data should be equal to send data on rank 0.");
  runPutNotify(
      /* bufCount */ NCCL_CTRAN_IB_QP_SCALING_THRESHOLD / sizeof(int),
      /* numPuts*/ NCCL_CTRAN_IB_QP_MAX_MSGS,
      /*localSignal*/ true,
      NotifyMode::notifyLast,
      /*isGpuMem*/ true,
      /*preConnect*/ false,
      /*issueFastPut*/ true);
}

TEST_P(CtranIbTestParam, GpuMemPutNotifyPreConnect) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemPutNotifyPreConnect",
      "Ranks are pre-connected before exchanging control messages. "
      "Expect rank 0 can put data from its local GPU data to rank 1 who waits on notify. "
      "The received data should be equal to send data on rank 0.");

  runPutNotify(
      /* bufCount */ 8192,
      /* numPuts*/ 1000,
      /*localSignal*/ true,
      NotifyMode::notifyAll,
      /*isGpuMem*/ true,
      /*preConnect*/ true);
}

TEST_P(CtranIbTestParam, GpuMemPutNotifyFastPreConnect) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemPutNotifyFastPreConnect",
      "Ranks are pre-connected before exchanging control messages. "
      "Expect rank 0 can issue multiple fast puts from its local GPU data to rank 1 who waits on notify. "
      "The received data should be equal to send data on rank 0.");
  runPutNotify(
      /* bufCount */ NCCL_CTRAN_IB_QP_SCALING_THRESHOLD / sizeof(int),
      /* numPuts*/ NCCL_CTRAN_IB_QP_MAX_MSGS,
      /*localSignal*/ true,
      NotifyMode::notifyAll,
      /*isGpuMem*/ true,
      /*preConnect*/ true,
      /*issueFastPut*/ true);
}

TEST_P(CtranIbTestParam, GpuMemPutNotifyNoSignal) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemPutNotifyNoSignal",
      "Expect rank 0 can issue multiple puts without local signal from its local GPU data to rank 1 who waits on notify. "
      "Expect ctranIb can handle local flush automatically."
      "The received data should be equal to send data on rank 0.");
  runPutNotify(
      /* bufCount */ 8192,
      /* numPuts*/ 1000,
      /*localSignal*/ false,
      NotifyMode::notifyAll,
      /*isGpuMem*/ true);
}

TEST_P(CtranIbTestParam, GpuMemPutNotifyNoSignalFast) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemPutNotifyNoSignalFast",
      "Expect rank 0 can issue multiple fast puts without local signal from its local GPU data to rank 1 who waits on notify. "
      "Expect ctranIb can handle local flush automatically."
      "The received data should be equal to send data on rank 0.");
  runPutNotify(
      /* bufCount */ NCCL_CTRAN_IB_QP_SCALING_THRESHOLD / sizeof(int),
      /* numPuts*/ NCCL_CTRAN_IB_QP_MAX_MSGS,
      /*localSignal*/ false,
      NotifyMode::notifyAll,
      /*isGpuMem*/ true,
      /*preConnect*/ false,
      /*issueFastPut*/ true);
}

TEST_P(CtranIbTestParam, GpuMemPutNotifyNoSignalMultiQp) {
  this->printTestDesc(
      "GpuMemPutNotifyNoSignalMultiQp",
      "Expect rank 0 can issue multiple puts to multiple QPs without local signal from its local GPU data to rank 1 who waits on notify. "
      "Expect ctranIb can handle local flush automatically for all data QPs."
      "The received data should be equal to send data on rank 0.");

  EnvRAII env1(NCCL_CTRAN_IB_MAX_QPS, 4);
  EnvRAII env2(NCCL_CTRAN_IB_QP_SCALING_THRESHOLD, 1024UL);
  EnvRAII env3(NCCL_CTRAN_IB_VC_MODE, GetParam());

  runPutNotify(
      /* bufCount */ 8192,
      /* numPuts*/ 1000,
      /*localSignal*/ false,
      NotifyMode::notifyAll,
      /*isGpuMem*/ true);
}

TEST_P(CtranIbTestParam, GpuMemPutNoNotify) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemPutNotify",
      "Expect rank 0 can issue multiple unsignaled and no-notify put to rank 1 and notify only on the last put. "
      "The received data should be equal to send data on rank 0.");
  runPutNotify(
      /* bufCount */ 8192,
      /* numPuts*/ 1000,
      /*localSignal*/ false,
      NotifyMode::notifyNone,
      /*isGpuMem*/ true);
}

TEST_P(CtranIbTestParam, GpuMemPutNotifyLast) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemPutNotify",
      "Expect rank 0 can issue multiple unsignaled and no-notify put to rank 1 and notify only on the last put. "
      "The received data should be equal to send data on rank 0.");
  runPutNotify(
      /* bufCount */ 8192,
      /* numPuts*/ 1000,
      /*localSignal*/ false,
      NotifyMode::notifyLast,
      /*isGpuMem*/ true);
}

TEST_P(CtranIbTestParam, Notify) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "Notify",
      "Expect rank 0 can issue a notify to rank 1 and rank 1 can wait on the notify via waitNotify.");

  runNotify(/* numNotifies*/ 1000, /*localSignal*/ true);
}

TEST_P(CtranIbTestParam, NotifyNoReq) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "Notify",
      "Expect rank 0 can issue multiple notify without local signal to rank 1 and rank 1 can wait on each of the notifications via waitNotify.");
  runNotify(/* numNotifies*/ 1000, /*localSignal*/ false);
}

TEST_F(CtranIbTest, CpuMemFetchAndAdd) {
  this->printTestDesc(
      "CpuMemFetchAndAdd",
      "Expect all ranks(except last rank) to do atomic increment to the same value in the last rank. "
      "The final value in the last rank should equal the number of fetch_and_add operations from all other ranks");
  runAtomicOps(false, false);
}

TEST_F(CtranIbTest, GpuMemFetchAndAdd) {
  this->printTestDesc(
      "GpuMemFetchAndAdd",
      "Expect all ranks(except last rank) to do atomic increment to the same value(in GPU memory) in the last rank. "
      "The final value in the last rank should equal the number of fetch_and_add operations from all other ranks");
  runAtomicOps(false, true);
}

TEST_F(CtranIbTest, GpuMemFetchAndAddSignal) {
  this->printTestDesc(
      "GpuMemFetchAndAddSignal",
      "Expect all ranks(except last rank) to do atomic increment to the same value(in GPU memory) in the last rank. "
      "The final value in the last rank should equal the number of fetch_and_add operations from all other ranks");
  runAtomicOps(true, true);
}

TEST_F(CtranIbTest, CpuMemAtomicSet) {
  this->printTestDesc(
      "CpuMemAtomicSet",
      "Expect every rank(except last rank) to atomic set its unique value in the last rank. "
      "The loaded value in the last rank could be any of the unique values from all other ranks");
  runAtomicOps(false, false, false);
}
TEST_F(CtranIbTest, GpuMemAtomicSet) {
  this->printTestDesc(
      "GpuMemAtomicSet",
      "Expect every rank(except last rank) to atomic set its unique value in the last rank(in GPU mem). "
      "The loaded value in the last rank could be any of the unique values from all other ranks");
  runAtomicOps(false, true, false);
}

TEST_F(CtranIbTest, MultiPutTrafficProfiler) {
  this->printTestDesc(
      "MultiPutTrafficProfiler",
      "Expect rank 0 puts data from its local GPU data to other ranks and "
      "the traffic profiling can catch exact bytes as expected per device and per QP.");

  setenv("NCCL_CTRAN_TRANSPORT_PROFILER", "true", 1);
  ncclCvarInit();

#undef BUF_COUNT
#define BUF_COUNT 8192
  try {
    auto ctranIb = std::make_unique<CtranIb>(
        this->comm, this->ctrlMgr.get(), true /* enableLocalFlush */);
    int* buf;
    void* handle = nullptr;
    ControlMsg sendMsg;
    std::unordered_map<int, ControlMsg> recvMsgs;
    CtranIbRequest ctrlSReq;
    std::unordered_map<int, CtranIbRequest> ctrlRReqs;
    std::unordered_map<int, CtranIbRequest> putReqs;
    const int rootRank = 0;

    CUDACHECK_TEST(cudaSetDevice(this->localRank));

    // Allocate and register buffer
    CUDACHECK_TEST(cudaMalloc(&buf, BUF_COUNT * sizeof(int)));
    COMMCHECK_TEST(
        CtranIb::regMem(
            buf, BUF_COUNT * sizeof(int), this->localRank, &handle));
    ASSERT_EQ(getIbRegCount(), commIbRegCount + 1);

    CtranIbEpochRAII epochRAII(ctranIb.get());

    // rootRank receives remoteAddr from all ranks
    if (this->globalRank == rootRank) {
      for (int i = 0; i < this->numRanks; i++) {
        // skip rootRank itself
        if (i == rootRank) {
          continue;
        }

        recvMsgs[i] = ControlMsg(ControlMsgType::IB_EXPORT_MEM);
        ctrlRReqs[i] = CtranIbRequest();
        COMMCHECK_TEST(ctranIb->irecvCtrlMsg(
            &recvMsgs[i], sizeof(recvMsgs[i]), i, ctrlRReqs[i]));
      }
    }

    // All rank sends the remoteAddr and rkey to rootRank
    COMMCHECK_TEST(CtranIb::exportMem(buf, handle, sendMsg));
    COMMCHECK_TEST(ctranIb->isendCtrlMsg(
        sendMsg.type, &sendMsg, sizeof(sendMsg), rootRank, ctrlSReq));

    // rootRank puts data to N-1 other ranks
    if (this->globalRank == rootRank) {
      while (putReqs.size() < this->numRanks - 1) {
        for (int i = 0; i < this->numRanks; i++) {
          // skip rootRank itself
          if (i == rootRank) {
            continue;
          }

          // wait control messages to be received from this sender
          // or skip if has already put
          if (!ctrlRReqs[i].isComplete() || putReqs.count(i)) {
            COMMCHECK_TEST(ctranIb->progress());
            continue;
          }

          void* remoteBuf =
              reinterpret_cast<void*>(recvMsgs[i].ibExp.remoteAddr);
          CtranIbRemoteAccessKey remoteKey{};
          for (int j = 0; j < recvMsgs[i].ibExp.nKeys; j++) {
            remoteKey.rkeys[j] = recvMsgs[i].ibExp.rkeys[j];
          }

          putReqs[i] = CtranIbRequest();
          ctranIb->iput(
              buf,
              remoteBuf,
              BUF_COUNT * sizeof(int),
              i,
              handle,
              remoteKey,
              true,
              nullptr,
              &putReqs[i]);
        }
      }

      // waits for all put to finish
      while (!putReqs.empty()) {
        COMMCHECK_TEST(ctranIb->progress());

        auto it = putReqs.begin();
        int rank = it->first;
        auto& req = it->second;
        if (req.isComplete()) {
          putReqs.erase(rank);
        }
      }
    } else {
      // Other rank ensures send control messages has completed
      waitIbReq(ctrlSReq, ctranIb);

      // Other rank waits notify to safely free buffer
      COMMCHECK_TEST(ctranIb->waitNotify(rootRank));

      // PCI-e flush to ensure data is immediately visible to GPU
      auto flushReq = CtranIbRequest();
      COMMCHECK_TEST(ctranIb->iflush(buf, handle, &flushReq));
      waitIbReq(flushReq, ctranIb);
    }

    // Free resources
    COMMCHECK_TEST(ctranIb->deregMem(handle));
    ASSERT_EQ(getIbRegCount(), commIbRegCount);
    CUDACHECK_TEST(cudaFree(buf));

  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }

  unsetenv("NCCL_CTRAN_TRANSPORT_PROFILER");
}

TEST_F(CtranIbTest, InvalidPeer) {
  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    auto invalidPeer = this->comm->statex_->nRanks();

    CtranIbEpochRAII epochRAII(ctranIb.get());

    auto msg = ControlMsg(ControlMsgType::IB_EXPORT_MEM);
    CtranIbRequest req;
    EXPECT_EQ(
        ctranIb->isendCtrlMsg(msg.type, &msg, sizeof(msg), invalidPeer, req),
        commInternalError);

    EXPECT_EQ(
        ctranIb->irecvCtrlMsg(&msg, sizeof(msg), invalidPeer, req),
        commInternalError);

    CtranIbRemoteAccessKey key{};
    EXPECT_EQ(
        ctranIb->iput(
            nullptr,
            nullptr,
            1024,
            invalidPeer,
            nullptr,
            key,
            true,
            nullptr,
            &req),
        commInternalError);

    EXPECT_EQ(ctranIb->notify(invalidPeer, &req), commInternalError);
    EXPECT_EQ(ctranIb->waitNotify(invalidPeer), commInternalError);
    bool notify;
    EXPECT_EQ(ctranIb->checkNotify(invalidPeer, &notify), commInternalError);

  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

TEST_F(CtranIbTest, NotReadyPeer) {
  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    constexpr int peerRank = 0;

    CtranIbEpochRAII epochRAII(ctranIb.get());

    // Valid peer but would require a control message to be exchanged to
    // establish the connection
    CtranIbRemoteAccessKey key{};
    EXPECT_EQ(
        ctranIb->iput(
            nullptr,
            nullptr,
            1024,
            peerRank,
            nullptr,
            key,
            true,
            nullptr,
            nullptr /*req*/),
        commInternalError);

    EXPECT_EQ(ctranIb->notify(peerRank, nullptr), commInternalError);

  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

TEST_F(CtranIbTest, InvalidMemoryWaitNotify) {
  this->printTestDesc(
      "InvalidMemoryWaitNotify",
      "Expect waitNotify to return error when called after put with invalid remote memory (wrong rkey or remote address)");
  ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());
  CtranIbEpochRAII epochRAII(ctranIb.get());

  if (this->globalRank == recvRank) {
    COMMCHECK_TEST(ctranIb->preConnect({sendRank}));
  } else if (this->globalRank == sendRank) {
    COMMCHECK_TEST(ctranIb->preConnect({recvRank}));
  }
  void* buf = nullptr;
  void* handle = nullptr;
  ControlMsg msg;
  CtranIbRequest ctrlReq;
  const size_t bufSize = 8192;

  // Allocate and register memory
  CUDACHECK_TEST(cudaMalloc(&buf, bufSize));
  COMMCHECK_TEST(CtranIb::regMem(buf, bufSize, this->localRank, &handle));
  COMMCHECK_TEST(CtranIb::exportMem(buf, handle, msg));

  if (this->globalRank == sendRank) {
    // Use invalid remote address (corrupted)
    void* invalidRemoteBuf = reinterpret_cast<void*>(0xdeadbeef);

    // Use invalid rkey (corrupted)
    CtranIbRemoteAccessKey invalidKey{};
    for (int i = 0; i < msg.ibExp.nKeys; i++) {
      invalidKey.rkeys[i] = 0xbadbeef; // Invalid rkey
    }

    // Try to put with invalid remote memory - this should succeed locally
    // but fail on the remote side when trying to access invalid memory
    CtranIbRequest putReq;
    auto putResult = ctranIb->iput(
        buf,
        invalidRemoteBuf, // Invalid remote address
        bufSize,
        recvRank,
        handle,
        invalidKey, // Invalid rkey
        true, // notify
        nullptr,
        &putReq);
    EXPECT_EQ(putResult, commSuccess);

    commResult_t res = commSuccess;
    do {
      res = ctranIb->progress();
    } while (res == commSuccess && !putReq.isComplete());
    EXPECT_EQ(res, commRemoteError);
    EXPECT_FALSE(putReq.isComplete());
  }

  // Clean up
  COMMCHECK_TEST(CtranIb::deregMem(handle));
  CUDACHECK_TEST(cudaFree(buf));
}

TEST_F(CtranIbTest, envQpConfig) {
  this->printTestDesc(
      "envQpConfig",
      "Test the CVARs of topology-aware QP scaling config are set correctly in CtranIb.");

  constexpr auto dc1 = "nha1", dc2 = "nha2", zone1 = "nha1.c084",
                 zone2 = "nha1.c085", zone3 = "nha2.c080";
  // set a fake topology to test QP changes for each IB VC
  std::unordered_map<int, std::vector<std::string>> expectedTopology = {
      {0, {"rtsw098.c084.f00.nha1", dc1, zone1}}, // rank-0
      {1, {"rtsw099.c084.f00.nha1", dc1, zone1}}, // rank-1: x-rack with rank-0
      {2, {"rtsw100.c085.f00.nha1", dc1, zone2}}, // rank-2: x-zone with rank-0
      {3, {"rtsw101.c080.f00.nha2", dc2, zone3}} // rank-3: x-dc with rank-0
  };
  // dummy values for testing
  // {<QP Scaling Threshold>, <number of QPs>, <VC mode>}
  std::vector<std::vector<std::string>> kPeerTestQpConfig = {
      {"128", "0", "spray", "2"},
      {"256", "4", "dqplb", "4"},
      {"512", "8", "spray", "8"},
      {"1024", "16", "dqplb", "12"},
  };
  EnvRAII xRackQps(NCCL_CTRAN_IB_QP_CONFIG_XRACK, kPeerTestQpConfig[1]);
  EnvRAII xZoneQps(NCCL_CTRAN_IB_QP_CONFIG_XZONE, kPeerTestQpConfig[2]);
  EnvRAII xDcQps(NCCL_CTRAN_IB_QP_CONFIG_XDC, kPeerTestQpConfig[3]);

  // statex should be created by now
  ASSERT_NE(comm->statex_, nullptr);
  // generate a fake topology for testing if CVARs taking effect
  std::vector<ncclx::RankTopology> testRankTopologies{};
  for (int rank = 0; rank < comm->statex_->nRanks(); rank++) {
    ncclx::RankTopology topo;
    topo.rank = rank;
    auto fakeHostName = "fakehost.0" + std::to_string(rank);
    std::strcpy(topo.host, fakeHostName.c_str());
    std::strcpy(topo.rtsw, expectedTopology.at(rank)[0].c_str());
    std::strcpy(topo.dc, expectedTopology.at(rank)[1].c_str());
    std::strcpy(topo.zone, expectedTopology.at(rank)[2].c_str());
    testRankTopologies.emplace_back(topo);
  }
  // create a new statex with fake topology for testing
  auto testStatex = std::make_unique<ncclx::CommStateX>(
      this->globalRank,
      this->numRanks,
      this->localRank,
      comm->statex_->cudaArch(),
      comm->statex_->busId(),
      comm->statex_->commHash(),
      testRankTopologies,
      comm->statex_->commRanksToWorldRanksRef(),
      "envQpConfigTestComm");
  // save old statex to be restored after the test
  auto oldStatex = std::move(comm->statex_);
  // overwrite statex
  comm->statex_ = std::move(testStatex);
  auto restoreOldStatexGuard =
      folly::makeGuard([&]() mutable { comm->statex_ = std::move(oldStatex); });

  std::unique_ptr<CtranIb> ctranIb = nullptr;
  try {
    ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }

  // check QP scaling configs
  if (this->globalRank == 0) {
    for (int peer = 1; peer < comm->statex_->nRanks(); peer++) {
      // send control message to establish connection before checking QP
      // config; otherwise, the internal VC would not be set
      ControlMsg msg(ControlMsgType::SYNC);
      CtranIbRequest ctrlReq = CtranIbRequest();
      COMMCHECK_TEST(
          ctranIb->isendCtrlMsg(msg.type, &msg, sizeof(msg), peer, ctrlReq));
      waitIbReq(ctrlReq, ctranIb);

      CtranIb::CtranIbVcConfig_t config;
      EXPECT_EQ(ctranIb->getVcConfig(peer, config), commSuccess);
      // Check QP Scaling Threshold
      EXPECT_EQ(std::get<0>(config), std::stoul(kPeerTestQpConfig[peer].at(0)));
      // Check # of QPs
      EXPECT_EQ(std::get<1>(config), std::stoi(kPeerTestQpConfig[peer].at(1)));
      // Check VC Mode
      if (kPeerTestQpConfig[peer].at(2) == "spray") {
        EXPECT_EQ(std::get<2>(config), NCCL_CTRAN_IB_VC_MODE::spray);
      } else {
        EXPECT_EQ(kPeerTestQpConfig[peer].at(2), "dqplb"); // Only other option
        EXPECT_EQ(std::get<2>(config), NCCL_CTRAN_IB_VC_MODE::dqplb);
      }
      // Check # of QPs per VC
      EXPECT_EQ(std::get<3>(config), std::stoi(kPeerTestQpConfig[peer].at(3)));
    }
  } else {
    ControlMsg msg(ControlMsgType::SYNC);
    CtranIbRequest ctrlReq = CtranIbRequest();
    COMMCHECK_TEST(ctranIb->irecvCtrlMsg(&msg, sizeof(msg), 0, ctrlReq));
    waitIbReq(ctrlReq, ctranIb);
  }
}

TEST_F(CtranIbTest, ValidBeTopology) {
  std::string kSuDomain1 = "nha1.c084.u001";
  constexpr auto dc1 = "nha1", dc2 = "nha2", zone1 = "nha1.c084",
                 zone2 = "nha1.c085", zone3 = "nha2.c080";
  // set a fake topology to test QP changes for each IB VC
  std::unordered_map<int, std::vector<std::string>> expectedTopology = {
      {0, {"rtsw098.c084.f00.nha1", kSuDomain1, dc1, zone1}}, // rank-0
      {1, {"rtsw099.c084.f00.nha1", kSuDomain1, dc1, zone1}}, // rank-1: x-rack
                                                              // with rank-0
      {2, {"rtsw100.c085.f00.nha1", kSuDomain1, dc1, zone2}}, // rank-2: x-zone
                                                              // with rank-0
      {3, {"rtsw101.c080.f00.nha2", kSuDomain1, dc2, zone3}} // rank-3: x-dc
                                                             // with rank-0
  };

  std::vector<ncclx::RankTopology> testRankTopologies{};
  for (int rank = 0; rank < comm->statex_->nRanks(); rank++) {
    ncclx::RankTopology topo;
    topo.rank = rank;
    auto fakeHostName = "fakehost.0" + std::to_string(rank);
    std::strcpy(topo.host, fakeHostName.c_str());
    std::strcpy(topo.rtsw, expectedTopology.at(rank)[0].c_str());
    std::strcpy(topo.su, expectedTopology.at(rank)[1].c_str());
    std::strcpy(topo.dc, expectedTopology.at(rank)[2].c_str());
    std::strcpy(topo.zone, expectedTopology.at(rank)[3].c_str());
    testRankTopologies.emplace_back(topo);
  }

  // create a new statex with fake topology for testing
  auto testStatex = std::make_unique<ncclx::CommStateX>(
      this->globalRank,
      this->numRanks,
      this->localRank,
      comm->statex_->cudaArch(),
      comm->statex_->busId(),
      comm->statex_->commHash(),
      testRankTopologies,
      comm->statex_->commRanksToWorldRanksRef(),
      "ValidBeTopologyTestComm");

  // save old statex to be restored after the test
  auto oldStatex = std::move(comm->statex_);
  // overwrite statex
  comm->statex_ = std::move(testStatex);

  std::unique_ptr<CtranIb> ctranIb = nullptr;
  try {
    ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

TEST_F(CtranIbTest, InvalidBeTopology) {
  std::string kSuDomain1 = "";
  constexpr auto dc1 = "nha1", dc2 = "nha2", zone1 = "nha1.c084",
                 zone2 = "nha1.c085", zone3 = "nha2.c080";
  // set a fake topology to test QP changes for each IB VC. No rtsw or SU info
  // is present for the ranks
  std::unordered_map<int, std::vector<std::string>> expectedTopology = {
      {0, {dc1, zone1}}, // rank-0
      {1, {dc1, zone1}}, // rank-1: x-rack with rank-0
      {2, {dc1, zone2}}, // rank-2: x-zone with rank-0
      {3, {dc2, zone3}} // rank-3: x-dc with rank-0
  };

  std::vector<ncclx::RankTopology> testRankTopologies{};
  for (int rank = 0; rank < comm->statex_->nRanks(); rank++) {
    ncclx::RankTopology topo;
    topo.rank = rank;
    auto fakeHostName = "fakehost.0" + std::to_string(rank);
    std::strcpy(topo.host, fakeHostName.c_str());
    std::strcpy(topo.dc, expectedTopology.at(rank)[0].c_str());
    std::strcpy(topo.zone, expectedTopology.at(rank)[1].c_str());
    std::strcpy(topo.su, kSuDomain1.c_str());
    testRankTopologies.emplace_back(topo);
  }

  // create a new statex with fake topology for testing
  auto testStatex = std::make_unique<ncclx::CommStateX>(
      this->globalRank,
      this->numRanks,
      this->localRank,
      comm->statex_->cudaArch(),
      comm->statex_->busId(),
      comm->statex_->commHash(),
      testRankTopologies,
      comm->statex_->commRanksToWorldRanksRef(),
      "InvalidBeTopologyTestComm");

  // save old statex to be restored after the test
  auto oldStatex = std::move(comm->statex_);
  // overwrite statex
  comm->statex_ = std::move(testStatex);

  std::unique_ptr<CtranIb> ctranIb = nullptr;
  try {
    ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  } catch (const ctran::utils::Exception& e) {
    EXPECT_THAT(
        e.what(), testing::HasSubstr("COMM internal failure: internal error"));
    ASSERT_EQ(ctranIb, nullptr);
  }
}

TEST_F(CtranIbTest, pgTrafficClassConfig) {
  std::vector<std::string> pgTrafficClass = {"PP_P2P_0:200", "PP_P2P_1:208"};
  EnvRAII env1(NCCL_CTRAN_IB_PG_TRAFFIC_CLASS, pgTrafficClass);
  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    constexpr int peerRank = 0;

    CtranIbEpochRAII epochRAII(ctranIb.get());

    // Valid peer but would require a control message to be exchanged to
    // establish the connection
    CtranIbRemoteAccessKey key{};
    EXPECT_EQ(
        ctranIb->iput(
            nullptr,
            nullptr,
            1024,
            peerRank,
            nullptr,
            key,
            true,
            nullptr,
            nullptr /*req*/),
        commInternalError);

    EXPECT_EQ(ctranIb->notify(peerRank, nullptr), commInternalError);

  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

TEST_F(CtranIbTest, pgTrafficClassConfigWithoutComm) {
  const std::string eth = "eth0";
  EnvRAII env1(NCCL_SOCKET_IFNAME, eth);
  std::vector<std::string> pgTrafficClass = {"PP_P2P_0:200", "PP_P2P_1:208"};
  EnvRAII env2(NCCL_CTRAN_IB_PG_TRAFFIC_CLASS, pgTrafficClass);
  const auto& rank = this->comm->statex_->rank();
  const auto& cudaDev = this->comm->statex_->cudaDev();
  const auto& commHash = this->comm->statex_->commHash();
  const auto& commDesc = this->comm->config_.commDesc;

  auto maybeAddr = ctran::bootstrap::getInterfaceAddress(
      NCCL_SOCKET_IFNAME, NCCL_SOCKET_IPADDR_PREFIX);
  ASSERT_FALSE(maybeAddr.hasError());

  // Create server socket, and bind it to reserve the port. The subsequent
  // test can use that port. The socket object will be destroyed (& port
  // released) when it goes out of scope.
  ctran::bootstrap::ServerSocket serverSocket(1);
  serverSocket.bind(
      folly::SocketAddress(*maybeAddr, 0), NCCL_SOCKET_IFNAME, true);
  int port = serverSocket.getListenAddress()->getPort();
  SocketServerAddr qpServerAddr{.port = port, .ipv6 = maybeAddr->str()};

  try {
    auto ctranIb = std::make_unique<CtranIb>(
        rank,
        cudaDev,
        commHash,
        commDesc,
        this->ctrlMgr.get(),
        true /*enableLocalFlush*/,
        CtranIb::BootstrapMode::kSpecifiedServer,
        &qpServerAddr);
    constexpr int peerRank = 0;
    CtranIbRemoteAccessKey key{};

    CtranIbEpochRAII epochRAII(ctranIb.get());

    EXPECT_EQ(
        ctranIb->iput(
            nullptr,
            nullptr,
            1024,
            peerRank,
            nullptr,
            key,
            true,
            nullptr,
            nullptr /*req*/),
        commInternalError);

    EXPECT_EQ(ctranIb->notify(peerRank, nullptr), commInternalError);

  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

TEST_F(CtranIbTest, AccessWithoutEpochLock) {
  try {
    EnvRAII env1(NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK, true);

    auto ctranIb = std::make_unique<CtranIb>(comm, this->ctrlMgr.get());
    int peerRank = (globalRank + 1) % numRanks;
    CtranIbRequest ctrlReq;
    ControlMsg msg;

    EXPECT_EQ(
        ctranIb->isendCtrlMsg(msg.type, &msg, sizeof(msg), peerRank, ctrlReq),
        commInternalError);

  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled (" << e.what() << "). Skip test";
  }
}

TEST_F(CtranIbTest, EpochUnlockWithoutLock) {
  try {
    auto ctranIb = std::make_unique<CtranIb>(comm, this->ctrlMgr.get());
    EXPECT_EQ(ctranIb->epochUnlock(), commInternalError);

  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled (" << e.what() << "). Skip test";
  }
}

TEST_F(CtranIbTest, DoubleEpochLock) {
  try {
    auto ctranIb = std::make_unique<CtranIb>(comm, this->ctrlMgr.get());
    EXPECT_EQ(ctranIb->epochLock(), commSuccess);

    // Expect inProgress is returned if epochLock is called by another thread
    std::thread t(
        [&]() { EXPECT_EQ(ctranIb->epochTryLock(), commInProgress); });
    t.join();

    // Expect error if called by the same thread
    EXPECT_EQ(ctranIb->epochLock(), commInternalError);

  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled (" << e.what() << "). Skip test";
  }
}

TEST_F(CtranIbTest, CtrlMsgAndPreConnect) {
  this->printTestDesc(
      "CtrlMsgAndPreConnect",
      "Expect rank 0 can issue a send control msg, followed by preConnect"
      "the preConnect is expected to be a no-op");

  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    CtranIbRequest req;
    ControlMsg smsg(ControlMsgType::IB_EXPORT_MEM);
    ControlMsg rmsg(ControlMsgType::IB_EXPORT_MEM);
    constexpr int sendRank = 0, recvRank = 1;

    CtranIbEpochRAII epochRAII(ctranIb.get());

    smsg.ibExp.remoteAddr = 99;
    smsg.ibExp.rkeys[0] = 1;
    smsg.ibExp.nKeys = 1;
    if (this->globalRank == sendRank) {
      COMMCHECK_TEST(
          ctranIb->isendCtrlMsg(smsg.type, &smsg, sizeof(smsg), recvRank, req));
    } else if (this->globalRank == recvRank) {
      COMMCHECK_TEST(ctranIb->irecvCtrlMsg(&rmsg, sizeof(rmsg), sendRank, req));
    } else {
      // no-op for non-communicating ranks
      COMMCHECK_TEST(req.complete());
    }

    waitIbReq(req, ctranIb);

    if (this->globalRank == recvRank) {
      EXPECT_EQ(rmsg.ibExp.remoteAddr, smsg.ibExp.remoteAddr);
      EXPECT_EQ(rmsg.ibExp.nKeys, smsg.ibExp.nKeys);
      EXPECT_EQ(rmsg.ibExp.rkeys[0], smsg.ibExp.rkeys[0]);
    }

    // pre-connect the peer
    std::unordered_set<int> peerRanks;
    if (this->globalRank == recvRank) {
      peerRanks.insert(sendRank);
      COMMCHECK_TEST(ctranIb->preConnect(peerRanks));
    } else if (this->globalRank == sendRank) {
      peerRanks.insert(recvRank);
      COMMCHECK_TEST(ctranIb->preConnect(peerRanks));
    }
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

TEST_P(CtranIbTestParam, InvalidIputFastNotify) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "InvalidIputFastNotify",
      "Three failure cases of calling fast iput: "
      "1. the bufsize is larger than maxWqeSize;"
      "2. the number of outstanding fast iput is equal to NCCL_CTRAN_IB_QP_MAX_MSGS;"
      "3. issuing fast iput without waiting on regular put completion;"
      "In any of the case, expect fast iput to return systemError");
  ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());

  auto bufCount = NCCL_CTRAN_IB_QP_SCALING_THRESHOLD / sizeof(int);
  // NCCL_CTRAN_IB_QP_SCALING_THRESHOLD can be overridden internally in Ctran.
  // So we get the actual threshold from vc config
  // send control message to establish connection before checking QP
  // config; otherwise, the internal VC would not be set
  {
    ControlMsg msg(ControlMsgType::SYNC);
    CtranIbRequest ctrlReq;
    CtranIb::CtranIbVcConfig_t config;
    if (this->globalRank == recvRank) {
      COMMCHECK_TEST(ctranIb->isendCtrlMsg(
          msg.type, &msg, sizeof(msg), sendRank, ctrlReq));
      waitIbReq(ctrlReq, ctranIb);
      COMMCHECK_TEST(ctranIb->getVcConfig(sendRank, config));
      bufCount = std::get<0>(config) / sizeof(int);
    } else if (this->globalRank == sendRank) {
      COMMCHECK_TEST(
          ctranIb->irecvCtrlMsg(&msg, sizeof(msg), recvRank, ctrlReq));
      waitIbReq(ctrlReq, ctranIb);
      COMMCHECK_TEST(ctranIb->getVcConfig(recvRank, config));
      bufCount = std::get<0>(config) / sizeof(int);
    }
  }

  CtranIbEpochRAII epochRAII(ctranIb.get());
  const int sendVal = 99, recvVal = -1;

  int* buf = nullptr;
  std::vector<int> hostBuf(bufCount, 0);
  void* handle = nullptr;
  CtranIbRequest ctrlReq, putReq, putReqFast;
  ControlMsg msg;

  // fill the buffer with different values and copy to GPU
  for (int i = 0; i < bufCount; i++) {
    hostBuf[i] = this->globalRank == sendRank ? sendVal : recvVal;
  }

  CUDACHECK_TEST(cudaMalloc(&buf, bufCount * sizeof(int)));
  CUDACHECK_TEST(cudaMemcpy(
      buf, hostBuf.data(), bufCount * sizeof(int), cudaMemcpyHostToDevice));

  // Register and export to a control msg
  COMMCHECK_TEST(
      CtranIb::regMem(buf, bufCount * sizeof(int), this->localRank, &handle));
  COMMCHECK_TEST(CtranIb::exportMem(buf, handle, msg));
  ASSERT_EQ(getIbRegCount(), commIbRegCount + 1);

  // Receiver sends the remoteAddr and rkey to sender
  if (this->globalRank == recvRank) {
    COMMCHECK_TEST(
        ctranIb->isendCtrlMsg(msg.type, &msg, sizeof(msg), sendRank, ctrlReq));
  } else if (this->globalRank == sendRank) {
    COMMCHECK_TEST(ctranIb->irecvCtrlMsg(&msg, sizeof(msg), recvRank, ctrlReq));
  } else {
    // no-op for non-communicating ranks
    COMMCHECK_TEST(ctrlReq.complete());
  }

  // Waits control message to be received
  waitIbReq(ctrlReq, ctranIb);

  if (this->globalRank == sendRank) {
    void* remoteBuf = reinterpret_cast<void*>(msg.ibExp.remoteAddr);
    CtranIbRemoteAccessKey key{};
    for (int i = 0; i < msg.ibExp.nKeys; i++) {
      key.rkeys[i] = msg.ibExp.rkeys[i];
    }

    // Failure case 1: the bufsize is larger than maxWqeSize
    auto res = ctranIb->iput(
        buf,
        remoteBuf,
        bufCount * 2 * sizeof(int),
        recvRank,
        handle,
        key,
        /* notify */ true,
        nullptr,
        &putReqFast,
        /* fast */ true);
    EXPECT_EQ(res, commSystemError);

    putReqFast = CtranIbRequest();
    for (int i = 0; i < NCCL_CTRAN_IB_QP_MAX_MSGS; i++) {
      res = ctranIb->iput(
          buf,
          remoteBuf,
          bufCount * sizeof(int),
          recvRank,
          handle,
          key,
          /* notify */ true,
          nullptr,
          (i == NCCL_CTRAN_IB_QP_MAX_MSGS - 1) ? &putReqFast : nullptr,
          /* fast */ true);
      EXPECT_EQ(res, commSuccess);
    }
    // Failure case 2: the number of outstanding fast iput is equal to
    // NCCL_CTRAN_IB_QP_MAX_MSGS
    res = ctranIb->iput(
        buf,
        remoteBuf,
        bufCount * sizeof(int),
        recvRank,
        handle,
        key,
        /* notify */ true,
        nullptr,
        &putReqFast,
        /* fast */ true);
    EXPECT_EQ(res, commSystemError);
    COMMCHECK_TEST(waitIbReq(putReqFast, ctranIb));

    // Sender puts the data to recvRank, first with iput(), second with
    // expect systemError on iputFast
    putReq = CtranIbRequest();
    COMMCHECK_TEST(ctranIb->iput(
        buf,
        remoteBuf,
        bufCount * sizeof(int),
        recvRank,
        handle,
        key,
        /* notify */ true,
        nullptr,
        &putReq));
    // Failure case 3: issuing fast iput without waiting on regular put
    // completion
    res = ctranIb->iput(
        buf,
        remoteBuf,
        bufCount * sizeof(int),
        recvRank,
        handle,
        key,
        /* notify */ true,
        nullptr,
        &putReqFast,
        /* fast */ true);
    EXPECT_EQ(res, commSystemError);

    // After waiting on first iput() to complete, expect second
    // iputFastNotify() to succeed
    COMMCHECK_TEST(waitIbReq(putReq, ctranIb));
    putReqFast = CtranIbRequest();
    COMMCHECK_TEST(ctranIb->iput(
        buf,
        remoteBuf,
        bufCount * sizeof(int),
        recvRank,
        handle,
        key,
        /* notify */ true,
        nullptr,
        &putReqFast,
        /* fast */ true));
    COMMCHECK_TEST(waitIbReq(putReqFast, ctranIb));
  } else if (this->globalRank == recvRank) {
    // The sender issued NCCL_CTRAN_IB_QP_MAX_MSGS + 1 fast puts and 1 regular
    // put.
    COMMCHECK_TEST(
        ctranIb->waitNotify(sendRank, NCCL_CTRAN_IB_QP_MAX_MSGS + 2));
    // PCI-e flush to ensure data is immediately visible to GPU
    auto flushReq = CtranIbRequest();
    COMMCHECK_TEST(ctranIb->iflush(buf, handle, &flushReq));
    COMMCHECK_TEST(waitIbReq(flushReq, ctranIb));

    CUDACHECK_TEST(cudaMemcpy(
        hostBuf.data(), buf, bufCount * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < bufCount; i++) {
      EXPECT_EQ(hostBuf[i], sendVal);
    }
  }

  COMMCHECK_TEST(CtranIb::deregMem(handle));
  ASSERT_EQ(getIbRegCount(), commIbRegCount);
  CUDACHECK_TEST(cudaFree(buf));
}

TEST_P(CtranIbTestParam, GpuMemPutNoSignalMixedFastRegular) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemPutNoSignalMixedFastRegular",
      "Expect rank 0 issue iputFast (w/ notify w/o signal) followed by iput (w/o notify w/o signal).");

  ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());

  auto bufCount = NCCL_CTRAN_IB_QP_SCALING_THRESHOLD / sizeof(int);

  CtranIbEpochRAII epochRAII(ctranIb.get());
  const int sendVal = 99, recvVal = -1;

  int* buf = nullptr;
  std::vector<int> hostBuf(bufCount, 0);
  void* handle = nullptr;
  CtranIbRequest ctrlReq, putReq;
  ControlMsg msg;

  // fill the buffer with different values and copy to GPU
  for (int i = 0; i < bufCount; i++) {
    hostBuf[i] = this->globalRank == sendRank ? sendVal : recvVal;
  }

  CUDACHECK_TEST(cudaMalloc(&buf, bufCount * sizeof(int)));
  CUDACHECK_TEST(cudaMemcpy(
      buf, hostBuf.data(), bufCount * sizeof(int), cudaMemcpyHostToDevice));

  // Register and export to a control msg
  COMMCHECK_TEST(
      CtranIb::regMem(buf, bufCount * sizeof(int), this->localRank, &handle));
  COMMCHECK_TEST(CtranIb::exportMem(buf, handle, msg));
  ASSERT_EQ(getIbRegCount(), commIbRegCount + 1);

  // Receiver sends the remoteAddr and rkey to sender
  if (this->globalRank == recvRank) {
    COMMCHECK_TEST(
        ctranIb->isendCtrlMsg(msg.type, &msg, sizeof(msg), sendRank, ctrlReq));
  } else if (this->globalRank == sendRank) {
    COMMCHECK_TEST(ctranIb->irecvCtrlMsg(&msg, sizeof(msg), recvRank, ctrlReq));
  } else {
    // no-op for non-communicating ranks
    COMMCHECK_TEST(ctrlReq.complete());
  }

  // Waits control message to be received
  waitIbReq(ctrlReq, ctranIb);

  const int numFastPuts = NCCL_CTRAN_IB_QP_MAX_MSGS, numPuts = 1000;
  if (this->globalRank == sendRank) {
    void* remoteBuf = reinterpret_cast<void*>(msg.ibExp.remoteAddr);
    CtranIbRemoteAccessKey key{};
    for (int i = 0; i < msg.ibExp.nKeys; i++) {
      key.rkeys[i] = msg.ibExp.rkeys[i];
    }

    for (int i = 0; i < numFastPuts; i++) {
      COMMCHECK_TEST(ctranIb->iput(
          buf,
          remoteBuf,
          bufCount * sizeof(int),
          recvRank,
          handle,
          key,
          /* notify */ true,
          /* config */ nullptr,
          /* req */ nullptr,
          /* fast */ true)); // NoSignal
    }
    putReq = CtranIbRequest();
    for (int i = 0; i < numPuts; i++) {
      COMMCHECK_TEST(ctranIb->iput(
          buf,
          remoteBuf,
          bufCount * sizeof(int),
          recvRank,
          handle,
          key,
          /* notify */ (i == numPuts - 1) ? true : false, // NoNotify
          nullptr,
          (i == numPuts - 1) ? &putReq : nullptr)); // NoSignal
    }
    COMMCHECK_TEST(waitIbReq(putReq, ctranIb));
  } else if (this->globalRank == recvRank) {
    // numFastPuts notifies from fast path and 1 notify from regular path.
    COMMCHECK_TEST(ctranIb->waitNotify(sendRank, numFastPuts + 1));
    // PCI-e flush to ensure data is immediately visible to GPU
    auto flushReq = CtranIbRequest();
    COMMCHECK_TEST(ctranIb->iflush(buf, handle, &flushReq));
    COMMCHECK_TEST(waitIbReq(flushReq, ctranIb));
    CUDACHECK_TEST(cudaMemcpy(
        hostBuf.data(), buf, bufCount * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < bufCount; i++) {
      EXPECT_EQ(hostBuf[i], sendVal);
    }
  }
  COMMCHECK_TEST(CtranIb::deregMem(handle));
  ASSERT_EQ(getIbRegCount(), commIbRegCount);
  CUDACHECK_TEST(cudaFree(buf));
}

TEST_P(CtranIbTestParam, GpuMemPutNotifyLastMixedFastRegular) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "GpuMemPutNotifyLastMixedFastRegular",
      "Expect rank 0 issue iputFast (w/o notify w/o signal) followed by iput (notifyLast w/o signal).");

  ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());

  auto bufCount = NCCL_CTRAN_IB_QP_SCALING_THRESHOLD / sizeof(int);

  CtranIbEpochRAII epochRAII(ctranIb.get());
  const int sendVal = 99, recvVal = -1;

  int* buf = nullptr;
  std::vector<int> hostBuf(bufCount, 0);
  void* handle = nullptr;
  CtranIbRequest ctrlReq, putReq;
  ControlMsg msg;

  // fill the buffer with different values and copy to GPU
  for (int i = 0; i < bufCount; i++) {
    hostBuf[i] = this->globalRank == sendRank ? sendVal : recvVal;
  }

  CUDACHECK_TEST(cudaMalloc(&buf, bufCount * sizeof(int)));
  CUDACHECK_TEST(cudaMemcpy(
      buf, hostBuf.data(), bufCount * sizeof(int), cudaMemcpyHostToDevice));

  // Register and export to a control msg
  COMMCHECK_TEST(
      CtranIb::regMem(buf, bufCount * sizeof(int), this->localRank, &handle));
  COMMCHECK_TEST(CtranIb::exportMem(buf, handle, msg));
  ASSERT_EQ(getIbRegCount(), commIbRegCount + 1);

  // Receiver sends the remoteAddr and rkey to sender
  if (this->globalRank == recvRank) {
    COMMCHECK_TEST(
        ctranIb->isendCtrlMsg(msg.type, &msg, sizeof(msg), sendRank, ctrlReq));
  } else if (this->globalRank == sendRank) {
    COMMCHECK_TEST(ctranIb->irecvCtrlMsg(&msg, sizeof(msg), recvRank, ctrlReq));
  } else {
    // no-op for non-communicating ranks
    COMMCHECK_TEST(ctrlReq.complete());
  }

  // Waits control message to be received
  waitIbReq(ctrlReq, ctranIb);

  const int numFastPuts = NCCL_CTRAN_IB_QP_MAX_MSGS, numPuts = 1000;
  if (this->globalRank == sendRank) {
    void* remoteBuf = reinterpret_cast<void*>(msg.ibExp.remoteAddr);
    CtranIbRemoteAccessKey key{};
    for (int i = 0; i < msg.ibExp.nKeys; i++) {
      key.rkeys[i] = msg.ibExp.rkeys[i];
    }

    for (int i = 0; i < numFastPuts; i++) {
      COMMCHECK_TEST(ctranIb->iput(
          buf,
          remoteBuf,
          bufCount * sizeof(int),
          recvRank,
          handle,
          key,
          /* notify */ false, // NoNotify
          /* config */ nullptr,
          /* req */ nullptr, // NoSignal
          /* fast */ true));
    }
    putReq = CtranIbRequest();
    for (int i = 0; i < numPuts; i++) {
      COMMCHECK_TEST(ctranIb->iput(
          buf,
          remoteBuf,
          bufCount * sizeof(int),
          recvRank,
          handle,
          key,
          /* notify */ (i == numPuts - 1) ? true : false, // notifyLast
          nullptr,
          (i == numPuts - 1) ? &putReq : nullptr)); // NoSignal
    }
    COMMCHECK_TEST(waitIbReq(putReq, ctranIb));
  } else if (this->globalRank == recvRank) {
    // 1 notify from slow path.
    COMMCHECK_TEST(ctranIb->waitNotify(sendRank, 1));
    // PCI-e flush to ensure data is immediately visible to GPU
    auto flushReq = CtranIbRequest();
    COMMCHECK_TEST(ctranIb->iflush(buf, handle, &flushReq));
    COMMCHECK_TEST(waitIbReq(flushReq, ctranIb));
    CUDACHECK_TEST(cudaMemcpy(
        hostBuf.data(), buf, bufCount * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < bufCount; i++) {
      EXPECT_EQ(hostBuf[i], sendVal);
    }
  }
  COMMCHECK_TEST(CtranIb::deregMem(handle));
  ASSERT_EQ(getIbRegCount(), commIbRegCount);
  CUDACHECK_TEST(cudaFree(buf));
}

class CtranIbTestWithQueuePairProfiler : public CtranIbTest {
 public:
  CtranIbTestWithQueuePairProfiler() = default;
  void SetUp() override {
    setenv("NCCL_CTRAN_TRANSPORT_PROFILER", "true", 1); // enable profiler
    setenv("NCCL_CTRAN_IB_MAX_QPS", "1", 1); // for tracking puts easier
    setenv("NCCL_CTRAN_DEVICE_TRAFFIC_SAMPLING_WEIGHT", "1", 1);
    setenv("NCCL_CTRAN_QP_PROFILING_ENABLE", "true", 1); // enable QP profiling
    CtranIbTest::SetUp();
  }
};

TEST_F(CtranIbTestWithQueuePairProfiler, GpuMemPutNotifyMixedFastRegular) {
  uint64_t qpScalingThreshold = 16384UL;

  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, NCCL_CTRAN_IB_VC_MODE::spray);
  EnvRAII env2(NCCL_CTRAN_IB_QP_SCALING_THRESHOLD, qpScalingThreshold);
  EnvRAII env3(NCCL_CTRAN_IB_QP_MAX_MSGS, 256UL);

  this->printTestDesc(
      "GpuMemPutNotifyMixedFastRegular",
      "Expect rank 0 issue iputFast followed by iput: verify iputFast uses IBV_WR_RDMA_WRITE_WITH_IMM and iput uses IBV_WR_RDMA_WRITE");

  const std::string algo = "test";
  CtranMapperContext context(algo, 0, 0);
  comm->ctran_->mapper->setContext(std::move(context));

  // create ib + register profiler
  ctranIb = std::make_unique<CtranIb>(comm, ctrlMgr.get());

  // Sanity check default VC Mode == spray
  // A dummy control message exchange to ensure QP connection
  // NCCL_CTRAN_IB_QP_SCALING_THRESHOLD and NCCL_CTRAN_IB_QP_MAX_MSGS can be
  // overridden internally in Ctran. So we get the actual threshold from vc
  // config
  ControlMsg msg;
  CtranIbRequest ctrlReq;
  CtranIb::CtranIbVcConfig_t config;
  if (this->globalRank == recvRank) {
    COMMCHECK_TEST(
        ctranIb->isendCtrlMsg(msg.type, &msg, sizeof(msg), sendRank, ctrlReq));
    waitIbReq(ctrlReq, ctranIb);
    COMMCHECK_TEST(ctranIb->getVcConfig(sendRank, config));
    qpScalingThreshold = NCCL_CTRAN_IB_QP_SCALING_THRESHOLD =
        std::get<0>(config);
    NCCL_CTRAN_IB_QP_MAX_MSGS = std::get<3>(config);
  } else if (this->globalRank == sendRank) {
    COMMCHECK_TEST(ctranIb->irecvCtrlMsg(&msg, sizeof(msg), recvRank, ctrlReq));
    waitIbReq(ctrlReq, ctranIb);
    COMMCHECK_TEST(ctranIb->getVcConfig(recvRank, config));
    qpScalingThreshold = NCCL_CTRAN_IB_QP_SCALING_THRESHOLD =
        std::get<0>(config);
    NCCL_CTRAN_IB_QP_MAX_MSGS = std::get<3>(config);
  }

  if (this->globalRank == sendRank) {
    // send control message to establish connection before checking QP config;
    // otherwise, the internal VC would not be set
    CtranIb::CtranIbVcConfig_t config;
    EXPECT_EQ(ctranIb->getVcConfig(this->recvRank, config), commSuccess);
    EXPECT_EQ(std::get<2>(config), NCCL_CTRAN_IB_VC_MODE::spray);
  }

  // First send iputFast: whose size is equal to
  // maxWqeSize(qpScalingThreshold).
  auto bufCount = qpScalingThreshold / sizeof(int);
  const int iputFastNumPuts = NCCL_CTRAN_IB_QP_MAX_MSGS;
  runPutNotify(
      /* bufCount */ bufCount,
      /* numPuts */ iputFastNumPuts,
      /* localSignal */ false,
      NotifyMode::notifyAll,
      /* isGpuMem */ true,
      /* preConnect */ false,
      /* issueFastPut */ true);

  // Then send regular iput: whose size is equal to
  // maxWqeSize(qpScalingThreshold) so it can fit in one WQE.
  const int iputNumPuts = 3;
  runPutNotify(
      /* bufCount */ bufCount,
      /* numPuts */ iputNumPuts,
      /* localSignal */ false,
      NotifyMode::notifyLast,
      /* isGpuMem */ true,
      /* preConnect */ false);
}

// Tests for different vc_modes
inline std::string getTestName(
    const testing::TestParamInfo<CtranIbTestParam::ParamType>& info) {
  if (info.param == NCCL_CTRAN_IB_VC_MODE::dqplb) {
    return "dqplb";
  } else {
    return "spray";
  }
}

// Tests for putBatch functionality
TEST_P(CtranIbTestParam, PutBatchBasic) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "PutBatchBasic",
      "Test basic putBatch functionality - sends data in a write batch to the qp in a write chained way");

  runPutBatch(
      /* bufCount */ 8192,
      /* numBatches */ 1,
      /* batchSize */ 5,
      /* withNotify */ true,
      /* mixedNotify */ false,
      /* isGpuMem */ true);
}

TEST_P(CtranIbTestParam, PutBatchMultiple) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "PutBatchMultiple", "Test multiple putBatches work correctly");

  runPutBatch(
      /* bufCount */ 4096,
      /* numBatches */ 3,
      /* batchSize */ 4,
      /* withNotify */ true,
      /* mixedNotify */ false,
      /* isGpuMem */ true);
}

TEST_P(CtranIbTestParam, PutBatchMixedNotify) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "PutBatchMixedNotify",
      "Test putBatch with both notify and not notify set work and notifications are delivered correctly");

  runPutBatch(
      /* bufCount */ 4096,
      /* numBatches */ 1,
      /* batchSize */ 6,
      /* withNotify */ false,
      /* mixedNotify */ true,
      /* isGpuMem */ true);
}

TEST_P(CtranIbTestParam, PutBatchWithNotify) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "PutBatchWithNotify",
      "Test putBatch with all requests having notify set");

  runPutBatch(
      /* bufCount */ 4096,
      /* numBatches */ 2,
      /* batchSize */ 4,
      /* withNotify */ true,
      /* mixedNotify */ false,
      /* isGpuMem */ true);
}

TEST_P(CtranIbTestParam, PutBatchWithoutNotify) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "PutBatchWithoutNotify",
      "Test putBatch with no requests having notify set");

  runPutBatch(
      /* bufCount */ 4096,
      /* numBatches */ 2,
      /* batchSize */ 4,
      /* withNotify */ false,
      /* mixedNotify */ false,
      /* isGpuMem */ true);
}

TEST_P(CtranIbTestParam, PutBatchInterleavedPut) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "PutBatchInterleavedPut",
      "Test interleaving putBatch and regular put works and notifications for both are delivered correctly");

  runPutBatch(
      /* bufCount */ 4096,
      /* numBatches */ 3,
      /* batchSize */ 3,
      /* withNotify */ true,
      /* mixedNotify */ false,
      /* isGpuMem */ true,
      /* interleavePut */ true);
}

TEST_P(CtranIbTestParam, PutBatchInterleavedFastPut) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "PutBatchInterleavedFastPut",
      "Test interleaving putBatch and fast put works and notifications for both are delivered correctly");

  runPutBatch(
      /* bufCount */ NCCL_CTRAN_IB_QP_SCALING_THRESHOLD / sizeof(int),
      /* numBatches */ 3,
      /* batchSize */ 2,
      /* withNotify */ true,
      /* mixedNotify */ false,
      /* isGpuMem */ true,
      /* interleavePut */ false,
      /* interleaveFastPut */ true);
}

TEST_P(CtranIbTestParam, PutBatchInterleavedBoth) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "PutBatchInterleavedBoth",
      "Test interleaving putBatch with both regular put and fast put works and notifications are delivered correctly");

  runPutBatch(
      /* bufCount */ NCCL_CTRAN_IB_QP_SCALING_THRESHOLD / sizeof(int),
      /* numBatches */ 4,
      /* batchSize */ 2,
      /* withNotify */ true,
      /* mixedNotify */ false,
      /* isGpuMem */ true,
      /* interleavePut */ true,
      /* interleaveFastPut */ true);
}

TEST_P(CtranIbTestParam, PutBatchCpuMem) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc("PutBatchCpuMem", "Test putBatch with CPU memory");

  runPutBatch(
      /* bufCount */ 4096,
      /* numBatches */ 2,
      /* batchSize */ 3,
      /* withNotify */ true,
      /* mixedNotify */ false,
      /* isGpuMem */ false);
}

TEST_P(CtranIbTestParam, PutBatchFallback) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, GetParam());
  this->printTestDesc(
      "PutBatchFallback",
      "Test fallback putBatch functionality - tests that batch put will fallback to indiviual puts");

  runPutBatch(
      /* bufCount */ 8192,
      /* numBatches */ 1,
      /* batchSize */ 5,
      /* withNotify */ true,
      /* mixedNotify */ false,
      /* isGpuMem */ true,
      /*interleavePut*/ false,
      /*interleaveFastPut*/ false,
      /*fallbackPut=*/true);
}

// Verify that iputBatch respects ibFastPath field by checking internal VC
// state.
// 1. If fast path is true, the sender sends packets with RDMA_WRITE_WITH_IMM,
// which increments sndNxt.
// 2. If fast path is false (in spray mode), the sender sends packets with
// RDMA_WRITE, which does not increment sndNxt. We verify the fix by checking if
// sndNxt/rcvNxt are updated accordingly.
TEST_F(CtranIbTest, PutBatchFastPathVerification) {
  EnvRAII env1(NCCL_CTRAN_IB_VC_MODE, NCCL_CTRAN_IB_VC_MODE::spray);

  this->printTestDesc(
      "PutBatchFastPathVerification",
      "Verify that iputBatch respects ibFastPath field by checking internal VC state (sndNxt). "
      "In spray mode, fast path updates sndNxt, while slow path does not.");

  // 1. Test Fast Path
  runPutBatch(
      /* bufCount */ 1024,
      /* numBatches */ 1,
      /* batchSize */ 5,
      /* withNotify */ true,
      /* mixedNotify */ false,
      /* isGpuMem */ true,
      /* interleavePut */ false,
      /* interleaveFastPut */ false,
      /* fallbackPut */ false,
      /* ibFastPath */ true,
      /* verifySeqNum */ true);

  // 2. Test Slow Path (FastPath=false)
  runPutBatch(
      /* bufCount */ 1024,
      /* numBatches */ 1,
      /* batchSize */ 5,
      /* withNotify */ true,
      /* mixedNotify */ false,
      /* isGpuMem */ true,
      /* interleavePut */ false,
      /* interleaveFastPut */ false,
      /* fallbackPut */ false,
      /* ibFastPath */ false,
      /* verifySeqNum */ true);
}

INSTANTIATE_TEST_SUITE_P(
    CtranIbTest,
    CtranIbTestParam,
    ::testing::Values(
        NCCL_CTRAN_IB_VC_MODE::dqplb,
        NCCL_CTRAN_IB_VC_MODE::spray),
    getTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
