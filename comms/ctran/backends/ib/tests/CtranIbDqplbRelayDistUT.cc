// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ctran::ibvwrap;

namespace {

commResult_t waitIbReq(CtranIbRequest& req, std::unique_ptr<CtranIb>& ctranIb) {
  do {
    COMMCHECK_TEST(ctranIb->progress());
  } while (!req.isComplete());
  return commSuccess;
}

class CtranIbDqplbRelayTest : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    CtranDistTestFixture::SetUp();
    comm_ = makeCtranComm();
    comm = comm_.get();
    commIbRegCount = getIbRegCount();
  }

  void TearDown() override {
    comm_.reset();
    CtranDistTestFixture::TearDown();
    ASSERT_EQ(getIbRegCount(), 0);
  }

  void printTestDesc(const std::string& testName, const std::string& testDesc) {
    XLOG_IF(WARN, globalRank == 0) << testName << " numRanks " << numRanks
                                   << ". Description: " << testDesc;
  }

  size_t getIbRegCount() {
    auto s = CtranIbSingleton::getInstance();
    CHECK_VALID_IB_SINGLETON(s);
    return s->getActiveRegCount();
  }

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

  struct Observation {
    int64_t firstUnexpectedOffset;
    int iteration;
    unsigned char firstUnexpectedValue;
    size_t rank0ByteCount;
    size_t rank2InitByteCount;
  };

  struct DqplbEnvRAII {
    // The relay case depends on multi-NIC DQPLB over multiple QPs. It does not
    // issue iflush, so NCCL_CTRAN_NET_FORCE_FLUSH is not part of the local test
    // setup.
    std::vector<std::string> qpConfig{"524288", "2", "dqplb", "128"};
    EnvRAII<decltype(NCCL_CTRAN_IB_DEVICES_PER_RANK)> devices{
        NCCL_CTRAN_IB_DEVICES_PER_RANK,
        2};
    // SAME_ZONE uses global defaults, so set them to dqplb mode
    EnvRAII<decltype(NCCL_CTRAN_IB_QP_SCALING_THRESHOLD)> scalingTh{
        NCCL_CTRAN_IB_QP_SCALING_THRESHOLD,
        524288};
    EnvRAII<decltype(NCCL_CTRAN_IB_MAX_QPS)> maxQps{NCCL_CTRAN_IB_MAX_QPS, 2};
    EnvRAII<decltype(NCCL_CTRAN_IB_VC_MODE)> vcMode{
        NCCL_CTRAN_IB_VC_MODE,
        NCCL_CTRAN_IB_VC_MODE::dqplb};
    EnvRAII<decltype(NCCL_CTRAN_IB_QP_MAX_MSGS)> maxMsgs{
        NCCL_CTRAN_IB_QP_MAX_MSGS,
        128};
    EnvRAII<decltype(NCCL_CTRAN_IB_QP_CONFIG_XZONE)> xZoneQps{
        NCCL_CTRAN_IB_QP_CONFIG_XZONE,
        qpConfig};
    EnvRAII<decltype(NCCL_CTRAN_IB_QP_CONFIG_XDC)> xDcQps{
        NCCL_CTRAN_IB_QP_CONFIG_XDC,
        qpConfig};
  };

  void runRelayOrderingCase(bool notifyAll, Observation* observation) {
    ASSERT_NE(observation, nullptr);
    *observation = Observation{-1, -1, 0, 0, 0};

    // Rank 0 writes two adjacent regions into rank 2's GPU buffer. Rank 2
    // waits for the notifications required by the mode and forwards that same
    // GPU buffer to rank 1. Rank 1 should receive only rank-0 bytes. Seeing
    // rank-2 initialization at rank 1 means rank 2 forwarded from its GPU
    // buffer before rank 0's earlier write was visible there.
    if (numRanks <= kRelayRank) {
      GTEST_SKIP()
          << "DQPLB relay ordering repro needs at least 3 ranks; run with "
          << "the 2-node/4-rank target on rtptest GB200 hosts.";
    }

    if (!ctranIb) {
      try {
        ctranIb = std::make_unique<CtranIb>(comm);
      } catch (const std::bad_alloc&) {
        GTEST_SKIP() << "IB backend not enabled. Skip test";
      }
    }

    CtranIbEpochRAII epochRAII(ctranIb.get());

    unsigned char* buf = nullptr;
    void* handle = nullptr;
    ControlMsg msg;

    CUDACHECK_TEST(cudaMalloc(&buf, kTotalBytes));
    auto cleanup = folly::makeGuard([&]() {
      if (handle != nullptr) {
        FB_COMMCHECKIGNORE(CtranIb::deregMem(handle));
      }
      if (buf != nullptr) {
        FB_CUDACHECKIGNORE(cudaFree(buf));
      }
    });

    COMMCHECK_TEST(CtranIb::regMem(buf, kTotalBytes, localRank, &handle));
    COMMCHECK_TEST(CtranIb::exportMem(buf, handle, msg));
    ASSERT_EQ(getIbRegCount(), commIbRegCount + 1);

    ControlMsg relayMsg;
    CtranIbRequest srcCtrlReq;
    if (globalRank == kRelayRank) {
      COMMCHECK_TEST(ctranIb->isendCtrlMsg(
          msg.type, &msg, sizeof(msg), kSrcRank, srcCtrlReq));
    } else if (globalRank == kSrcRank) {
      COMMCHECK_TEST(ctranIb->irecvCtrlMsg(
          &relayMsg, sizeof(relayMsg), kRelayRank, srcCtrlReq));
    } else {
      COMMCHECK_TEST(srcCtrlReq.complete());
    }

    ControlMsg sinkMsg;
    CtranIbRequest relayCtrlReq;
    if (globalRank == kSinkRank) {
      COMMCHECK_TEST(ctranIb->isendCtrlMsg(
          msg.type, &msg, sizeof(msg), kRelayRank, relayCtrlReq));
    } else if (globalRank == kRelayRank) {
      COMMCHECK_TEST(ctranIb->irecvCtrlMsg(
          &sinkMsg, sizeof(sinkMsg), kSinkRank, relayCtrlReq));
    } else {
      COMMCHECK_TEST(relayCtrlReq.complete());
    }

    COMMCHECK_TEST(waitIbReq(srcCtrlReq, ctranIb));
    COMMCHECK_TEST(waitIbReq(relayCtrlReq, ctranIb));

    std::vector<unsigned char> hostStep;
    if (globalRank == kSinkRank) {
      hostStep.resize(kStepBytes);
    }

    for (int iteration = 0; iteration < kIterations; ++iteration) {
      unsigned char fill = 0;
      if (globalRank == kSrcRank) {
        fill = kExpected;
      } else if (globalRank == kRelayRank) {
        fill = kRelayInit;
      } else if (globalRank == kSinkRank) {
        fill = kSinkInit;
      }
      CUDACHECK_TEST(cudaMemset(buf, fill, kTotalBytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      // Keep each round independent. The ordering contract under test starts
      // after every rank has reset its local GPU buffer.
      if (globalRank == 0) {
        for (int rank = 1; rank < numRanks; ++rank) {
          sockRecv(rank);
        }
        for (int rank = 1; rank < numRanks; ++rank) {
          sockSend(rank);
        }
      } else {
        sockSend(0);
        sockRecv(0);
      }

      if (globalRank == kSrcRank) {
        issueSourcePuts(buf, handle, relayMsg, notifyAll);
      } else if (globalRank == kRelayRank) {
        relayPuts(buf, handle, sinkMsg, notifyAll);
      } else if (globalRank == kSinkRank) {
        validateSink(buf, hostStep, iteration, observation);
      }
    }

    XLOG_IF(WARN, globalRank == kSinkRank)
        << "dqplb relay " << (notifyAll ? "notify-all" : "notify-last")
        << " rank1SinkFirstUnexpectedOffset="
        << observation->firstUnexpectedOffset
        << " iteration=" << observation->iteration << " firstUnexpectedValue="
        << static_cast<int>(observation->firstUnexpectedValue)
        << " rank0ByteCount=" << observation->rank0ByteCount
        << " rank2InitByteCount=" << observation->rank2InitByteCount
        << " totalBytes=" << kStepBytes << " iterations=" << kIterations;
  }

 protected:
  void issueSourcePuts(
      unsigned char* buf,
      void* handle,
      const ControlMsg& relayMsg,
      bool notifyAll) {
    void* remoteBuf = reinterpret_cast<void*>(relayMsg.ibDesc.remoteAddr);
    CtranIbRemoteAccessKey key{};
    for (int i = 0; i < relayMsg.ibDesc.nKeys; i++) {
      key.rkeys[i] = relayMsg.ibDesc.rkeys[i];
    }

    std::vector<CtranIbRequest> putReqs(kPutOffsets.size());
    for (size_t put = 0; put < kPutOffsets.size(); ++put) {
      COMMCHECK_TEST(ctranIb->iput(
          buf + kPutOffsets[put],
          static_cast<unsigned char*>(remoteBuf) + kPutOffsets[put],
          kPutBytes[put],
          kRelayRank,
          handle,
          key,
          notifyAll || put == kPutOffsets.size() - 1,
          nullptr,
          &putReqs[put]));
    }

    // Keep the source registration alive until rank 2 has forwarded and rank 1
    // has validated. Only then wait for local completion of every put.
    sockRecv(kRelayRank);
    for (auto& putReq : putReqs) {
      COMMCHECK_TEST(waitIbReq(putReq, ctranIb));
    }
    sockSend(kRelayRank);
  }

  void relayPuts(
      unsigned char* buf,
      void* handle,
      const ControlMsg& sinkMsg,
      bool notifyAll) {
    COMMCHECK_TEST(ctranIb->waitNotify(kSrcRank, notifyAll ? 2 : 1));

    void* remoteBuf = reinterpret_cast<void*>(sinkMsg.ibDesc.remoteAddr);
    CtranIbRemoteAccessKey key{};
    for (int i = 0; i < sinkMsg.ibDesc.nKeys; i++) {
      key.rkeys[i] = sinkMsg.ibDesc.rkeys[i];
    }

    std::vector<CtranIbRequest> putReqs(kPutOffsets.size());
    for (size_t put = 0; put < kPutOffsets.size(); ++put) {
      COMMCHECK_TEST(ctranIb->iput(
          buf + kPutOffsets[put],
          static_cast<unsigned char*>(remoteBuf) + kPutOffsets[put],
          kPutBytes[put],
          kSinkRank,
          handle,
          key,
          /*notify*/ true,
          nullptr,
          &putReqs[put]));
    }

    sockRecv(kSinkRank);
    for (auto& putReq : putReqs) {
      COMMCHECK_TEST(waitIbReq(putReq, ctranIb));
    }
    sockSend(kSinkRank);

    sockSend(kSrcRank);
    sockRecv(kSrcRank);
  }

  void validateSink(
      unsigned char* buf,
      std::vector<unsigned char>& hostStep,
      int iteration,
      Observation* observation) {
    COMMCHECK_TEST(ctranIb->waitNotify(kRelayRank, 2));

    CUDACHECK_TEST(
        cudaMemcpy(hostStep.data(), buf, kStepBytes, cudaMemcpyDeviceToHost));

    for (size_t byte = 0; byte < hostStep.size(); ++byte) {
      const unsigned char value = hostStep[byte];
      if (value == kExpected) {
        ++observation->rank0ByteCount;
      } else {
        if (value == kRelayInit) {
          ++observation->rank2InitByteCount;
        }
        if (observation->firstUnexpectedOffset < 0) {
          observation->firstUnexpectedOffset = static_cast<int64_t>(byte);
          observation->iteration = iteration;
          observation->firstUnexpectedValue = value;
        }
      }
    }

    sockSend(kRelayRank);
    sockRecv(kRelayRank);
  }

  static constexpr int kSockSyncLen = 16;
  static constexpr int kSrcRank = 0;
  static constexpr int kRelayRank = 2;
  static constexpr int kSinkRank = 1;
  static constexpr int kIterations = 500;
  static constexpr size_t kPriorBytes = 512 * 1024;
  static constexpr size_t kNotifyBytes = 4 * 1024;
  static constexpr size_t kStepBytes = kPriorBytes + kNotifyBytes;
  static constexpr size_t kTotalBytes = 2 * 1024 * 1024;
  static constexpr unsigned char kExpected = 0x51;
  static constexpr unsigned char kRelayInit = 0xCD;
  static constexpr unsigned char kSinkInit = 0xEF;
  static constexpr std::array<size_t, 2> kPutOffsets = {0, kPriorBytes};
  static constexpr std::array<size_t, 2> kPutBytes = {
      kPriorBytes,
      kNotifyBytes};

  std::unique_ptr<CtranComm> comm_{nullptr};
  CtranComm* comm{nullptr};
  std::unique_ptr<CtranIb> ctranIb{nullptr};
  size_t commIbRegCount{0};
};

TEST_F(CtranIbDqplbRelayTest, DqplbGpuRelayNotifyAllOrdering) {
  DqplbEnvRAII env;
  printTestDesc(
      "DqplbGpuRelayNotifyAllOrdering",
      "Expect rank 1 to receive only rank 0 bytes when rank 2 forwards after waiting for every rank 0 notification.");

  Observation observation;
  runRelayOrderingCase(/*notifyAll*/ true, &observation);
  if (globalRank == kSinkRank) {
    EXPECT_EQ(observation.firstUnexpectedOffset, -1)
        << "rank 1 received bytes other than rank 0 data after notify-all "
           "relay: firstUnexpectedOffset="
        << observation.firstUnexpectedOffset
        << " iteration=" << observation.iteration << " firstUnexpectedValue="
        << static_cast<int>(observation.firstUnexpectedValue)
        << " rank0ByteCount=" << observation.rank0ByteCount
        << " rank2InitByteCount=" << observation.rank2InitByteCount;
  }
}

// Diagnostic observation for the notify-last relay pattern. Rank 2 waits for
// only the final write notification before forwarding. With DQPLB over multiple
// QPs/NICs, we have observed rank 1 receive rank 2's initialized bytes because
// that wait does not establish receiver-side completion for the earlier
// no-notify write.
TEST_F(CtranIbDqplbRelayTest, DqplbGpuRelayNotifyLastOrderingObservation) {
  DqplbEnvRAII env;
  printTestDesc(
      "DqplbGpuRelayNotifyLastOrderingObservation",
      "Diagnostic repro where rank 1 can receive rank 2 initialization bytes instead of rank 0 bytes when rank 2 forwards after only the final rank 0 notification.");

  Observation observation;
  runRelayOrderingCase(/*notifyAll*/ false, &observation);
  if (globalRank == kSinkRank) {
    XLOG_IF(WARN, observation.firstUnexpectedOffset >= 0)
        << "rank 1 received bytes other than rank 0 data after notify-last "
           "relay: firstUnexpectedOffset="
        << observation.firstUnexpectedOffset
        << " iteration=" << observation.iteration << " firstUnexpectedValue="
        << static_cast<int>(observation.firstUnexpectedValue)
        << " rank0ByteCount=" << observation.rank0ByteCount
        << " rank2InitByteCount=" << observation.rank2InitByteCount;
    XLOG_IF(WARN, observation.firstUnexpectedOffset < 0)
        << "notify-last relay did not reproduce stale rank 2 bytes in this "
           "run: rank0ByteCount="
        << observation.rank0ByteCount
        << " rank2InitByteCount=" << observation.rank2InitByteCount;
  }
}

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
