// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <nccl.h>
#include <stdlib.h>
#include "comms/ctran/CtranEx.h"
#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

using namespace ctran;

class CtranExFailureTest : public CtranExBaseTest {
 protected:
  const std::string defaultDesc_{"CtranExFailureTest"};
  int* buf_{nullptr};
  size_t bufSize_{8192};
  void* regHdl_{nullptr};
  void* peerRemoteBuf_{nullptr};
  uint32_t peerRemoteKey_{0};

  void prepareBuf(const size_t bufSize) {
    CUDACHECK_TEST(cudaMalloc(&buf_, bufSize));

    // Use first half as send buffer, and second half as recv buffer
    size_t dataCnt = bufSize / sizeof(int) / 2;
    assignChunkValue(buf_, dataCnt, globalRank, 1);
    assignChunkValue(buf_ + dataCnt, dataCnt, 0, 0);
  }

  void checkDataRecv(const size_t bufSize, const int leftPeer) {
    size_t dataCnt = bufSize / sizeof(int) / 2;
    auto errs = checkChunkValue(buf_ + dataCnt, dataCnt, leftPeer, 1);
    EXPECT_EQ(errs, 0) << "Found received data from leftPeer " << leftPeer
                       << " is not correct at globalRank " << globalRank << ", "
                       << errs << " errors";
  }

  void injectFailure(
      const int rank,
      const std::string& ibverb,
      const int seq,
      FailureType type = FailureType::API_ERROR) {
    setFailureInjection(ibverb.c_str(), seq, rank, type);
  }
};

// Test case description doc:
// https://docs.google.com/document/d/1Xu3mtjDPROSvWfv5VCeyFB_TGVqyqRcZPWoiVAuPMew

// General test scenerio:
// Rank 0, 1, 2, 3 form a ring and each rank sends data to its right peer, i.e.
//                       Rank 0 --> Rank 1
//                         ^          |
//                         |          v
//                       Rank 3 <-- Rank 2

TEST_F(CtranExFailureTest, qpExchangeFailure) {
  // QP exchange test case 1: bootstrapConnect before bootstrapAccept
  // Scenario: Rank 3 doesn't start it's QP server
  // Expected behavior:
  //    - Rank 0 and Rank 2 fail with ncclRemoteError
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  std::unique_ptr<CtranEx> ctranEx = nullptr;
  if (globalRank != 3) {
    createCtranEx(hostInfo, ctranEx);
    ASSERT_NE(ctranEx, nullptr);
  }

  barrier();

  // Exchange ctrl msg as a Ring where local rank sends data to right peer,
  // and receives data from left peer.
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;
  CtranExRequest *sreq = nullptr, *rreq = nullptr;

  // use ctrl msg to trigger qp exchange
  switch (globalRank) {
    case 0:
      // Rank 0 isendCtrl to Rank 3 fail with ncclRemote Error
      // Rank 0 irecvCtrl to Rank 1 succeed
      ASSERT_NE(
          ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq),
          ncclSuccess);
      ASSERT_EQ(
          ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
          ncclSuccess);
      ASSERT_EQ(sreq, nullptr);
      ASSERT_NE(rreq, nullptr);
      ASSERT_EQ(rreq->wait(), ncclSuccess);
      break;
    case 2:
      // Rank 2 irecvCtrl to Rank 3 fail with ncclRemote Error
      // Rank 2 isendCtrl to Rank 1 succeed
      ASSERT_EQ(
          ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq),
          ncclSuccess);
      ASSERT_NE(
          ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
          ncclSuccess);
      ASSERT_NE(sreq, nullptr);
      ASSERT_EQ(rreq, nullptr);
      ASSERT_EQ(sreq->wait(), ncclSuccess);
      break;
    case 3:
      // Rank 3: didn't start QP server
      break;
    default:
      // Rank 1 (w/o communication with Rank 3): isendCtrl & irecvCtrl succeed
      ASSERT_EQ(
          ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq),
          ncclSuccess);
      ASSERT_EQ(
          ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
          ncclSuccess);
      ASSERT_NE(sreq, nullptr);
      ASSERT_NE(rreq, nullptr);
      ASSERT_EQ(rreq->wait(), ncclSuccess);
      ASSERT_EQ(sreq->wait(), ncclSuccess);
      break;
  }
}

TEST_F(CtranExFailureTest, iputPostSendFailure) {
  // Data transfer w/ iput() test case 1: Sender ibv_post_send failed
  // Scenario: Rank 3 send data to Rank 0 and Rank 3's ibv_post_send failed
  // during iput
  // Expected behavior:
  //    - Sender (Rank 3): fail with unhandled system error
  //    - Receiver (Rank 0): keep waiting at waitNotify until timeout since it
  //    won’t receive a notification from sender
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer.
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;
  CtranExRequest* req = nullptr;
  std::unique_ptr<CtranExRequest> sreq, rreq, preq;

  prepareBuf(bufSize_);
  ASSERT_EQ(ctranEx->regMem(buf_, bufSize_, &regHdl_), ncclSuccess);
  ASSERT_NE(regHdl_, nullptr);

  // Send local buffer info to left peer to enable remote access from it
  barrier();
  ASSERT_EQ(
      ctranEx->isendCtrl(
          buf_, bufSize_, regHdl_, leftPeer, allHostInfos[leftPeer], &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  sreq = std::unique_ptr<CtranExRequest>(req);

  // Receive remote buffer info from right peer to enable remote access to it
  ASSERT_EQ(
      ctranEx->irecvCtrl(
          rightPeer,
          allHostInfos[rightPeer],
          &peerRemoteBuf_,
          &peerRemoteKey_,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  rreq = std::unique_ptr<CtranExRequest>(req);

  // Wait receive ctrl msg to finish and check the received buffer info
  ASSERT_EQ(rreq->wait(), ncclSuccess);
  EXPECT_NE(peerRemoteBuf_, nullptr);
  EXPECT_NE(peerRemoteKey_, 0);
  ASSERT_EQ(sreq->wait(), ncclSuccess);

  // do a barrier to make sure all ranks are ready for failure injection
  barrier();

  // inject the 1st ibv_post_send failure to rank 3
  injectFailure(3, "ibv_post_send", 0);
  // Send data to right peer
  if (globalRank == 3) {
    ASSERT_EQ(
        ctranEx->iput(
            buf_,
            bufSize_ / 2,
            regHdl_,
            rightPeer,
            reinterpret_cast<int*>(peerRemoteBuf_) + bufSize_ / 2 / sizeof(int),
            peerRemoteKey_,
            true /* notify */,
            &req),
        ncclSystemError);
    ASSERT_EQ(ctranEx->deregMem(regHdl_), ncclSuccess);
    CUDACHECK_TEST(cudaFree(buf_));
    return;
  }

  ASSERT_EQ(
      ctranEx->iput(
          buf_,
          bufSize_ / 2,
          regHdl_,
          rightPeer,
          reinterpret_cast<int*>(peerRemoteBuf_) + bufSize_ / 2 / sizeof(int),
          peerRemoteKey_,
          true /* notify */,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  preq = std::unique_ptr<CtranExRequest>(req);
  if (globalRank == 0) {
    auto timerBegin = std::chrono::steady_clock::now();
    bool done = false;
    while (true) {
      auto now = std::chrono::steady_clock::now();
      ASSERT_EQ(ctranEx->checkNotify(leftPeer, done), ncclSuccess);
      if (done ||
          std::chrono::duration_cast<std::chrono::milliseconds>(
              now - timerBegin)
                  .count() > 10 * 1000) {
        break;
      }
    }
    ASSERT_FALSE(done);
  } else if (globalRank != 3) {
    ASSERT_EQ(ctranEx->waitNotify(leftPeer), ncclSuccess);
    checkDataRecv(bufSize_, leftPeer);
  }

  ASSERT_EQ(preq->wait(), ncclSuccess);

  ASSERT_EQ(ctranEx->deregMem(regHdl_), ncclSuccess);
  CUDACHECK_TEST(cudaFree(buf_));
}

TEST_F(CtranExFailureTest, iputPostRecvFailure) {
  // Data transfer w/ iput() test case 2: Receiver ibv_post_recv failed
  // Scenario: Rank 3 receives data from Rank 2 and Rank 3's ibv_post_recv
  // failed
  // Expected behavior:
  //    - Sender (Rank 2): succeed since the receiver has prefilled sendCtrl
  //    messages
  //    - Receiver (Rank 3): fail with unhandled system error
  //    won’t receive a notification from sender
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer.
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;
  CtranExRequest* req = nullptr;
  std::unique_ptr<CtranExRequest> sreq, rreq, preq;

  prepareBuf(bufSize_);
  ASSERT_EQ(ctranEx->regMem(buf_, bufSize_, &regHdl_), ncclSuccess);
  ASSERT_NE(regHdl_, nullptr);
  barrier();
  // Send local buffer info to left peer to enable remote access from it
  ASSERT_EQ(
      ctranEx->isendCtrl(
          buf_, bufSize_, regHdl_, leftPeer, allHostInfos[leftPeer], &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  sreq = std::unique_ptr<CtranExRequest>(req);

  // Receive remote buffer info from right peer to enable remote access to it
  ASSERT_EQ(
      ctranEx->irecvCtrl(
          rightPeer,
          allHostInfos[rightPeer],
          &peerRemoteBuf_,
          &peerRemoteKey_,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  rreq = std::unique_ptr<CtranExRequest>(req);

  // Wait receive ctrl msg to finish and check the received buffer info
  ASSERT_EQ(rreq->wait(), ncclSuccess);
  EXPECT_NE(peerRemoteBuf_, nullptr);
  EXPECT_NE(peerRemoteKey_, 0);
  ASSERT_EQ(sreq->wait(), ncclSuccess);

  // do a barrier to make sure all ranks are ready for failure injection
  barrier();

  // inject the 1st ibv_post_recv failure to rank 3
  injectFailure(3, "ibv_post_recv", 0);
  // Send data to right peer
  ASSERT_EQ(
      ctranEx->iput(
          buf_,
          bufSize_ / 2,
          regHdl_,
          rightPeer,
          reinterpret_cast<int*>(peerRemoteBuf_) + bufSize_ / 2 / sizeof(int),
          peerRemoteKey_,
          true /* notify */,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  preq = std::unique_ptr<CtranExRequest>(req);

  // Wait left peer to transfer data to my buf, and check the received data
  if (globalRank == 3) {
    ASSERT_EQ(ctranEx->waitNotify(leftPeer), ncclSystemError);
  } else {
    ASSERT_EQ(ctranEx->waitNotify(leftPeer), ncclSuccess);
    checkDataRecv(bufSize_, leftPeer);
  }
  ASSERT_EQ(preq->wait(), ncclSuccess);

  ASSERT_EQ(ctranEx->deregMem(regHdl_), ncclSuccess);
  CUDACHECK_TEST(cudaFree(buf_));
}

TEST_F(CtranExFailureTest, ctrlMsgPostSendFailure) {
  // ctrl Msg test case 1: Sender ibv_post_send failed
  // Scenario: Rank 3 is sending a ctrlSync Msg to Rank 2 and Rank 3's
  // ibv_post_send failed
  // Expected behavior:
  //    - Sender (Rank 3): fail with unhandled system error
  //    - Receiver (Rank 2): keep waiting at waitNotify until timeout since it
  //        won’t receive a notification
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer.
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;
  CtranExRequest *sreq = nullptr, *rreq = nullptr;
  barrier();
  // use ctrl msg to trigger qp exchange
  ASSERT_EQ(
      ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq), ncclSuccess);
  ASSERT_EQ(
      ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
      ncclSuccess);
  ASSERT_NE(sreq, nullptr);
  ASSERT_NE(rreq, nullptr);
  ASSERT_EQ(sreq->wait(), ncclSuccess);
  ASSERT_EQ(rreq->wait(), ncclSuccess);

  // do a barrier to make sure all ranks are ready for failure injection
  barrier();

  // inject the 1st ibv_post_send failure to rank 3
  injectFailure(3, "ibv_post_send", 0);
  // Send ctrlSync to left peer
  if (globalRank == 3) {
    ASSERT_EQ(
        ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq),
        ncclSystemError);
  } else {
    ASSERT_EQ(
        ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq),
        ncclSuccess);
  }

  // Receive ctrlSync from right peer
  ASSERT_EQ(
      ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
      ncclSuccess);
  ASSERT_NE(rreq, nullptr);
  ASSERT_NE(sreq, nullptr);
  ASSERT_EQ(sreq->wait(), ncclSuccess);

  if (globalRank == 2) {
    bool done = false;
    auto timerBegin = std::chrono::steady_clock::now();
    while (true) {
      auto now = std::chrono::steady_clock::now();
      ASSERT_EQ(rreq->test(done), ncclSuccess);
      if (done ||
          std::chrono::duration_cast<std::chrono::milliseconds>(
              now - timerBegin)
                  .count() > 5 * 1000) {
        break;
      }
    }
    ASSERT_FALSE(done);
  } else if (globalRank != 3) {
    ASSERT_EQ(rreq->wait(), ncclSuccess);
  }
}

TEST_F(CtranExFailureTest, ctrlMsgPostRecvFailure) {
  // ctrl Msg test case 2: Receiver ibv_post_recv failed
  // Scenario: Rank 3 is receiving a ctrlSync Msg from Rank 0 and Rank 3's
  // ibv_post_recv failed
  // Expected behavior:
  //    - Sender (Rank 0): succeed since sender has prefilled recvCtrl
  //    - Receiver (Rank 3): fail with unhandled system error
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer.
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;
  CtranExRequest *sreq = nullptr, *rreq = nullptr;

  barrier();
  // use ctrl msg to trigger qp exchange
  ASSERT_EQ(
      ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq), ncclSuccess);
  ASSERT_EQ(
      ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
      ncclSuccess);
  ASSERT_NE(sreq, nullptr);
  ASSERT_NE(rreq, nullptr);
  ASSERT_EQ(sreq->wait(), ncclSuccess);
  ASSERT_EQ(rreq->wait(), ncclSuccess);

  // do a barrier to make sure all ranks are ready for failure injection
  barrier();

  // inject the 1st ibv_post_send failure to rank 3
  injectFailure(3, "ibv_post_recv", 0);
  // Send ctrlSync to left peer
  ASSERT_EQ(
      ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq), ncclSuccess);
  ASSERT_NE(sreq, nullptr);

  // Receive ctrlSync from right peer
  ASSERT_EQ(
      ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
      ncclSuccess);

  ASSERT_NE(rreq, nullptr);
  if (globalRank == 3) {
    ASSERT_EQ(
        rreq->wait(),
        ncclSystemError); // Rank 3 will fail when do ibv_post_recv
  } else {
    ASSERT_EQ(rreq->wait(), ncclSuccess); // Other ranks will succeed
  }
  ASSERT_EQ(sreq->wait(), ncclSuccess);
}

TEST_F(CtranExFailureTest, iputSenderPollCqFailureSlowPath) {
  // Data transfer w/ iput() test case 3: Sender ibv_poll_cq failed
  // Scenario: Rank 3 is sending data with iput() to Rank 0 and Rank 3's
  // ibv_poll_cq failed
  // Expected behavior:
  //    - Sender (Rank 3): fail with unhandled system error
  //    - Receiver (Rank 0): keep waiting at waitNotify until timeout since it
  //    won’t receive a notification from sender
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);
  bufSize_ =
      2ULL * CTRAN_IB_FAST_PATH_MSG_MAX_SIZE; // keep it larger than
                                              // maxBufSize in CtranIbVc.h
  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  // Overwrite after CtranEx creation before QP connection, to be always spray
  // mode so that notification would be hanlded after put local completion on
  // rank 3.
  EnvRAII env(NCCL_CTRAN_IB_VC_MODE, NCCL_CTRAN_IB_VC_MODE::spray);

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer.
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;
  CtranExRequest* req = nullptr;
  std::unique_ptr<CtranExRequest> sreq, rreq, preq;

  prepareBuf(bufSize_);
  ASSERT_EQ(ctranEx->regMem(buf_, bufSize_, &regHdl_), ncclSuccess);
  ASSERT_NE(regHdl_, nullptr);

  barrier();
  // Send local buffer info to left peer to enable remote access from it
  ASSERT_EQ(
      ctranEx->isendCtrl(
          buf_, bufSize_, regHdl_, leftPeer, allHostInfos[leftPeer], &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  sreq = std::unique_ptr<CtranExRequest>(req);

  // Receive remote buffer info from right peer to enable remote access to it
  ASSERT_EQ(
      ctranEx->irecvCtrl(
          rightPeer,
          allHostInfos[rightPeer],
          &peerRemoteBuf_,
          &peerRemoteKey_,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  rreq = std::unique_ptr<CtranExRequest>(req);

  // Wait receive ctrl msg to finish and check the received buffer info
  ASSERT_EQ(rreq->wait(), ncclSuccess);
  EXPECT_NE(peerRemoteBuf_, nullptr);
  EXPECT_NE(peerRemoteKey_, 0);
  ASSERT_EQ(sreq->wait(), ncclSuccess);

  // do a barrier to make sure all ranks are ready for failure injection
  barrier();

  // inject the 1st ibv_poll_cq failure to rank 3
  injectFailure(3, "ibv_poll_cq", 0);

  // Send data to right peer
  ASSERT_EQ(
      ctranEx->iput(
          buf_,
          bufSize_ / 2,
          regHdl_,
          rightPeer,
          reinterpret_cast<int*>(peerRemoteBuf_) + bufSize_ / 2 / sizeof(int),
          peerRemoteKey_,
          true /* notify */,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  preq = std::unique_ptr<CtranExRequest>(req);
  if (globalRank == 0) {
    auto timerBegin = std::chrono::steady_clock::now();
    bool done = false;
    while (true) {
      auto now = std::chrono::steady_clock::now();
      ASSERT_EQ(ctranEx->checkNotify(leftPeer, done), ncclSuccess);
      if (done ||
          std::chrono::duration_cast<std::chrono::milliseconds>(
              now - timerBegin)
                  .count() > 10 * 1000) {
        break;
      }
    }
    ASSERT_FALSE(done); // Rank 0 will timeout since rank 3 fail to send
  } else if (globalRank != 3) {
    ASSERT_EQ(ctranEx->waitNotify(leftPeer), ncclSuccess);
    checkDataRecv(bufSize_, leftPeer);
  }
  if (globalRank == 3) {
    ASSERT_EQ(preq->wait(), ncclSystemError); // Rank 3 will fail when poll cq
  } else {
    ASSERT_EQ(preq->wait(), ncclSuccess);
  }
  barrier();
  ASSERT_EQ(ctranEx->deregMem(regHdl_), ncclSuccess);
  CUDACHECK_TEST(cudaFree(buf_));
}

TEST_F(CtranExFailureTest, iputSenderPollCqFailureFastPath) {
  // Data transfer w/ iput() test case 3: Sender ibv_poll_cq failed
  // Scenario: Rank 3 is sending data with iput() to Rank 0 and Rank 3's
  // ibv_poll_cq failed
  // Expected behavior:
  //    - Sender (Rank 3): fail with unhandled system error
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer.
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;
  CtranExRequest* req = nullptr;
  std::unique_ptr<CtranExRequest> sreq, rreq, preq;

  prepareBuf(bufSize_);
  ASSERT_EQ(ctranEx->regMem(buf_, bufSize_, &regHdl_), ncclSuccess);
  ASSERT_NE(regHdl_, nullptr);

  barrier();
  // Send local buffer info to left peer to enable remote access from it
  ASSERT_EQ(
      ctranEx->isendCtrl(
          buf_, bufSize_, regHdl_, leftPeer, allHostInfos[leftPeer], &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  sreq = std::unique_ptr<CtranExRequest>(req);

  // Receive remote buffer info from right peer to enable remote access to it
  ASSERT_EQ(
      ctranEx->irecvCtrl(
          rightPeer,
          allHostInfos[rightPeer],
          &peerRemoteBuf_,
          &peerRemoteKey_,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  rreq = std::unique_ptr<CtranExRequest>(req);

  // Wait receive ctrl msg to finish and check the received buffer info
  ASSERT_EQ(rreq->wait(), ncclSuccess);
  EXPECT_NE(peerRemoteBuf_, nullptr);
  EXPECT_NE(peerRemoteKey_, 0);
  ASSERT_EQ(sreq->wait(), ncclSuccess);

  // do a barrier to make sure all ranks are ready for failure injection
  barrier();

  // inject the 1st ibv_poll_cq failure to rank 3
  injectFailure(3, "ibv_poll_cq", 0);

  // Send data to right peer
  ASSERT_EQ(
      ctranEx->iput(
          buf_,
          bufSize_ / 2,
          regHdl_,
          rightPeer,
          reinterpret_cast<int*>(peerRemoteBuf_) + bufSize_ / 2 / sizeof(int),
          peerRemoteKey_,
          true /* notify */,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  preq = std::unique_ptr<CtranExRequest>(req);
  if (globalRank != 3) {
    ASSERT_EQ(ctranEx->waitNotify(leftPeer), ncclSuccess);
    checkDataRecv(bufSize_, leftPeer);
  }
  if (globalRank == 3) {
    ASSERT_EQ(preq->wait(), ncclSystemError); // Rank 3 will fail when poll cq
  } else {
    ASSERT_EQ(preq->wait(), ncclSuccess);
  }
  barrier();
  ASSERT_EQ(ctranEx->deregMem(regHdl_), ncclSuccess);
  CUDACHECK_TEST(cudaFree(buf_));
}

TEST_F(CtranExFailureTest, iputReceiverPollCqFailure) {
  // Data transfer w/ iput() test case 4: Receiver ibv_poll_cq failed
  // Scenario: Rank 3 is receiving data from Rank 2 and Rank 3's
  // ibv_poll_cq failed when Rank 2 does IBV_WR_RDMA_WRITE
  // Expected behavior:
  //    - Sender (Rank 2): succeed since receiver has prefilled sendCtrl
  //    messages
  //    - Receiver (Rank 3): fail with unhandled system error
  // Note: In real case: the error caused ibv_poll_cq failed will actually cause
  // receiver side failed as well

  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer.
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;
  CtranExRequest* req = nullptr;
  std::unique_ptr<CtranExRequest> sreq, rreq, preq;

  prepareBuf(bufSize_);
  ASSERT_EQ(ctranEx->regMem(buf_, bufSize_, &regHdl_), ncclSuccess);
  ASSERT_NE(regHdl_, nullptr);

  barrier();
  // Send local buffer info to left peer to enable remote access from it
  ASSERT_EQ(
      ctranEx->isendCtrl(
          buf_, bufSize_, regHdl_, leftPeer, allHostInfos[leftPeer], &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  sreq = std::unique_ptr<CtranExRequest>(req);

  // Receive remote buffer info from right peer to enable remote access to it
  ASSERT_EQ(
      ctranEx->irecvCtrl(
          rightPeer,
          allHostInfos[rightPeer],
          &peerRemoteBuf_,
          &peerRemoteKey_,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  rreq = std::unique_ptr<CtranExRequest>(req);

  // Wait receive ctrl msg to finish and check the received buffer info
  ASSERT_EQ(rreq->wait(), ncclSuccess);
  EXPECT_NE(peerRemoteBuf_, nullptr);
  EXPECT_NE(peerRemoteKey_, 0);
  ASSERT_EQ(sreq->wait(), ncclSuccess);

  // do a barrier to make sure all ranks are ready for failure injection
  barrier();

  // Send data to right peer
  ASSERT_EQ(
      ctranEx->iput(
          buf_,
          bufSize_ / 2,
          regHdl_,
          rightPeer,
          reinterpret_cast<int*>(peerRemoteBuf_) + bufSize_ / 2 / sizeof(int),
          peerRemoteKey_,
          true /* notify */,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  preq = std::unique_ptr<CtranExRequest>(req);

  ASSERT_EQ(preq->wait(), ncclSuccess);

  // inject the 1st ibv_poll_cq failure to rank 3
  injectFailure(3, "ibv_poll_cq", 0);

  // Wait left peer to transfer data to my buf, and check the received data
  if (globalRank == 3) {
    ASSERT_EQ(ctranEx->waitNotify(leftPeer), ncclSystemError);
  } else {
    ASSERT_EQ(ctranEx->waitNotify(leftPeer), ncclSuccess);
    checkDataRecv(bufSize_, leftPeer);
  }

  ASSERT_EQ(ctranEx->deregMem(regHdl_), ncclSuccess);
  CUDACHECK_TEST(cudaFree(buf_));
}

TEST_F(CtranExFailureTest, ctrlMsgSenderPollCqFailure) {
  // ctrl Msg  test case 3: Sender ibv_poll_cq failed
  // Scenario: Rank 3 is sending a ctrlMsg to Rank 2 and Rank 3's
  // ibv_poll_cq failed
  // Expected behavior:
  //    - Sender (Rank 3): fail with unhandled system error
  //    - Receiver (Rank 2): succeed since the right peer has sent ctrlMsg to it
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer.
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;
  CtranExRequest *sreq = nullptr, *rreq = nullptr;

  barrier();
  // use ctrl msg to trigger qp exchange
  ASSERT_EQ(
      ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq), ncclSuccess);
  ASSERT_EQ(
      ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
      ncclSuccess);
  ASSERT_NE(sreq, nullptr);
  ASSERT_NE(rreq, nullptr);
  ASSERT_EQ(sreq->wait(), ncclSuccess);
  ASSERT_EQ(rreq->wait(), ncclSuccess);

  // do a barrier to make sure all ranks are ready for failure injection
  barrier();

  // inject the 1st ibv_poll_cq failure to rank 3
  injectFailure(3, "ibv_poll_cq", 0);

  // Send ctrlSync to left peer
  ASSERT_EQ(
      ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq), ncclSuccess);
  ASSERT_NE(sreq, nullptr);
  if (globalRank == 3) {
    ASSERT_EQ(sreq->wait(), ncclSystemError);
  } else {
    ASSERT_EQ(sreq->wait(), ncclSuccess);
  }

  // Receive ctrlSync from right peer
  ASSERT_EQ(
      ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
      ncclSuccess);
  ASSERT_NE(rreq, nullptr);
}

TEST_F(CtranExFailureTest, ctrlMsgReceiverPollCqFailure) {
  // ctrl Msg  test case 4: Receiver ibv_poll_cq failed
  // Scenario: Rank 3 is receiving a ctrlMsg from Rank 0 and Rank 3's
  // ibv_poll_cq failed
  // Expected behavior:
  //    - Sender (Rank 0): succeed since the recvCtrl queue has been prefilled
  //    - Receiver (Rank 3): fail with unhandled system error
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer.
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;
  CtranExRequest *sreq = nullptr, *rreq = nullptr;
  barrier();
  // use ctrl msg to trigger qp exchange
  ASSERT_EQ(
      ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq), ncclSuccess);
  ASSERT_EQ(
      ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
      ncclSuccess);
  ASSERT_NE(sreq, nullptr);
  ASSERT_NE(rreq, nullptr);
  ASSERT_EQ(sreq->wait(), ncclSuccess);
  ASSERT_EQ(rreq->wait(), ncclSuccess);

  // do a barrier to make sure all ranks are ready for failure injection
  barrier();

  // Send ctrlSync to left peer
  ASSERT_EQ(
      ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq), ncclSuccess);
  ASSERT_NE(sreq, nullptr);

  if (globalRank == 3) {
    // make sure Rank 3 has sent so the following poll_cq will fail on irecvCtrl
    ASSERT_EQ(sreq->wait(), ncclSuccess);
  }

  // inject the 1st ibv_poll_cq failure to rank 3
  injectFailure(3, "ibv_poll_cq", 0);

  // Receive ctrlSync from right peer
  ASSERT_EQ(
      ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
      ncclSuccess);
  ASSERT_NE(rreq, nullptr);

  if (globalRank == 3) {
    bool done = false;
    ASSERT_EQ(
        rreq->test(done), ncclSystemError); // Rank 3 will fail when do poll_cq
  } else {
    ASSERT_EQ(rreq->wait(), ncclSuccess); // Other ranks will succeed
    ASSERT_EQ(sreq->wait(), ncclSuccess);
  }
}

TEST_F(CtranExFailureTest, iputReceiverPollCqWcTimeout) {
  // Data transfer w/ iput() test case 5: Receiver ibv_poll_cq WC timeout
  // Scenario: Rank 3 is receiving data from Rank 2 and Rank 3's
  // ibv_poll_cq indicated WC timeout when Rank 2 does
  // IBV_WR_RDMA_WRITE_WITH_IMM Expected behavior:
  //    - Sender (Rank 2): succeed since receiver has prefilled sendCtrl
  //    messages
  //    - Receiver (Rank 3): fail with remote error
  // Note: In real case: the error caused ibv_poll_cq failed will actually cause
  // sender side failure as well

  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer.
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;
  CtranExRequest* req = nullptr;
  std::unique_ptr<CtranExRequest> sreq, rreq, preq;

  prepareBuf(bufSize_);
  ASSERT_EQ(ctranEx->regMem(buf_, bufSize_, &regHdl_), ncclSuccess);
  ASSERT_NE(regHdl_, nullptr);
  barrier();
  // Send local buffer info to left peer to enable remote access from it
  ASSERT_EQ(
      ctranEx->isendCtrl(
          buf_, bufSize_, regHdl_, leftPeer, allHostInfos[leftPeer], &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  sreq = std::unique_ptr<CtranExRequest>(req);

  // Receive remote buffer info from right peer to enable remote access to it
  ASSERT_EQ(
      ctranEx->irecvCtrl(
          rightPeer,
          allHostInfos[rightPeer],
          &peerRemoteBuf_,
          &peerRemoteKey_,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  rreq = std::unique_ptr<CtranExRequest>(req);

  // Wait receive ctrl msg to finish and check the received buffer info
  ASSERT_EQ(rreq->wait(), ncclSuccess);
  EXPECT_NE(peerRemoteBuf_, nullptr);
  EXPECT_NE(peerRemoteKey_, 0);
  ASSERT_EQ(sreq->wait(), ncclSuccess);

  // do a barrier to make sure all ranks are ready for failure injection
  barrier();

  // Send data to right peer
  ASSERT_EQ(
      ctranEx->iput(
          buf_,
          bufSize_ / 2,
          regHdl_,
          rightPeer,
          reinterpret_cast<int*>(peerRemoteBuf_) + bufSize_ / 2 / sizeof(int),
          peerRemoteKey_,
          true /* notify */,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  preq = std::unique_ptr<CtranExRequest>(req);

  ASSERT_EQ(preq->wait(), ncclSuccess);

  // inject work completion timeout failure to the 1st ibv_poll_cq call on rank
  // 3
  injectFailure(3, "ibv_poll_cq", 0, FailureType::WC_TIMEOUT);

  // Wait left peer to transfer data to my buf, and check the received data
  if (globalRank == 3) {
    ASSERT_EQ(ctranEx->waitNotify(leftPeer), ncclRemoteError);
  } else {
    ASSERT_EQ(ctranEx->waitNotify(leftPeer), ncclSuccess);
    checkDataRecv(bufSize_, leftPeer);
  }

  ASSERT_EQ(ctranEx->deregMem(regHdl_), ncclSuccess);
  CUDACHECK_TEST(cudaFree(buf_));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
