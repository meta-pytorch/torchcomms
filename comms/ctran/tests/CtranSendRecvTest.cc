// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <future>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/tests/CtranTestUtils.h"

namespace ctran::testing {

static const std::chrono::milliseconds kDefaultTimeoutTwoSeconds =
    std::chrono::milliseconds(2000);

class CtranSendRecvTest : public CtranIntraProcessFixture {
 public:
  using PeerConfig = std::pair<int, size_t>; // {rank, nElems}

 protected:
  static constexpr int kNRanks = 2;
  static_assert(kNRanks % 2 == 0);
  static constexpr commDataType_t kType = commFloat32;
  static constexpr size_t kTypeSize = sizeof(float);
  static constexpr size_t kBufferNElem = kBufferSize / kTypeSize;

  void SetUp() override {
    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal", 1);
    setenv("NCCL_IGNORE_TOPO_LOAD_FAILURE", "1", 1);

    CtranIntraProcessFixture::SetUp();
  }
  void startWorkers(bool abortEnabled) {
    std::vector<std::shared_ptr<::ctran::utils::Abort>> aborts;
    if (abortEnabled) {
      for (int i = 0; i < kNRanks; ++i) {
        aborts.push_back(ctran::utils::createAbort(/*enabled=*/true));
      }
    }
    CtranIntraProcessFixture::startWorkers(kNRanks, /*aborts=*/aborts);
  }

  void runSend(
      int peer,
      size_t nElem,
      PerRankState& state,
      bool expectError = false,
      std::optional<std::chrono::milliseconds> timeout = std::nullopt) {
    runSendRecv(
        /*sendPeers=*/{{peer, nElem}},
        /*recvPeers=*/{},
        state,
        expectError,
        timeout);
  }
  void runRecv(
      int peer,
      size_t nElem,
      PerRankState& state,
      bool expectError = false,
      std::optional<std::chrono::milliseconds> timeout = std::nullopt) {
    runSendRecv(
        /*sendPeers=*/{},
        /*recvPeers=*/{{peer, nElem}},
        state,
        expectError,
        timeout);
  };

  void validateConfigs(std::vector<PeerConfig> peers) {
    for (const auto& [peer, nElem] : peers) {
      ASSERT_TRUE(nElem <= kBufferNElem);
    }
  }

  void runSendRecv(
      std::vector<PeerConfig> sendPeers,
      std::vector<PeerConfig> recvPeers,
      PerRankState& state,
      bool expectError = false,
      std::optional<std::chrono::milliseconds> timeout = std::nullopt) {
    validateConfigs(sendPeers);
    validateConfigs(recvPeers);

    void* srcHandle;
    void* dstHandle;
    ASSERT_EQ(
        commSuccess,
        state.ctranComm->ctran_->commRegister(
            state.srcBuffer, kBufferSize, &srcHandle));
    ASSERT_EQ(
        commSuccess,
        state.ctranComm->ctran_->commRegister(
            state.dstBuffer, kBufferSize, &dstHandle));
    SCOPE_EXIT {
      // deregistering will happen after streamSync below
      state.ctranComm->ctran_->commDeregister(dstHandle);
      state.ctranComm->ctran_->commDeregister(srcHandle);
    };

    for (const auto& [peer, nElem] : sendPeers) {
      EXPECT_EQ(
          commSuccess,
          ctranSend(
              state.srcBuffer,
              nElem,
              kType,
              peer,
              state.ctranComm.get(),
              state.stream));
    }
    for (const auto& [peer, nElem] : recvPeers) {
      EXPECT_EQ(
          commSuccess,
          ctranRecv(
              state.dstBuffer,
              nElem,
              kType,
              peer,
              state.ctranComm.get(),
              state.stream));
    }
    EXPECT_EQ(commSuccess, ctranGroupEndHook(timeout));

    // ensure async execution completion and no error
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(state.stream));
    if (!expectError) {
      EXPECT_EQ(commSuccess, state.ctranComm->getAsyncResult());
    } else {
      // TODO(T238821628): update error code for now we strictly check for
      // remote error, since this is the error type returned by network error
      // and temporarily for abort as well.
      EXPECT_EQ(commRemoteError, state.ctranComm->getAsyncResult());
    }
  }
};

TEST_F(CtranSendRecvTest, SendRecvUnidirectional_AbortFeatureDisabled) {
  startWorkers(/*abortEnabled=*/false);
  run(/*rank=*/0,
      [this](PerRankState& state) { runSend(1, kBufferNElem, state); });
  run(/*rank=*/1,
      [this](PerRankState& state) { runRecv(0, kBufferNElem, state); });
}

TEST_F(CtranSendRecvTest, SendRecvBidirectional_AbortFeatureDisabled) {
  startWorkers(/*abortEnabled=*/false);
  for (int rank = 0; rank < kNRanks; rank++) {
    int peer = rank ^ 1;
    // for both send & recv
    std::vector<PeerConfig> cfg = {{peer, kBufferNElem}};
    run(/*rank=*/rank,
        [this, cfg](PerRankState& state) { runSendRecv(cfg, cfg, state); });
  }
}

TEST_F(CtranSendRecvTest, SendRecvUnidirectional) {
  startWorkers(/*abortEnabled=*/true);
  run(/*rank=*/0,
      [this](PerRankState& state) { runSend(1, kBufferNElem, state); });
  run(/*rank=*/1,
      [this](PerRankState& state) { runRecv(0, kBufferNElem, state); });
}

TEST_F(CtranSendRecvTest, SendRecvBidirectional) {
  startWorkers(/*abortEnabled=*/true);
  for (int rank = 0; rank < kNRanks; rank++) {
    int peer = rank ^ 1;
    // for both send & recv
    std::vector<PeerConfig> cfg = {{peer, kBufferNElem}};
    run(/*rank=*/rank,
        [this, cfg](PerRankState& state) { runSendRecv(cfg, cfg, state); });
  }
}

std::future<void> launchWatcher(CtranIntraProcessFixture::PerRankState& state) {
  return std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(kDefaultTimeoutTwoSeconds);
    EXPECT_EQ(cudaErrorNotReady, cudaStreamQuery(state.stream));
    state.ctranComm->setAbort();
  });
}

TEST_F(CtranSendRecvTest, TwoReceiversActiveAbort) {
  startWorkers(/*abortEnabled=*/true);
  for (const int rank : {0, 1}) {
    run(rank, [this, rank](PerRankState& state) {
      const int peer = rank ^ 1;
      auto timer = launchWatcher(state);
      runRecv(peer, kBufferNElem, state, /*expectError=*/true);
    });
  }
}

TEST_F(CtranSendRecvTest, TwoSendersActiveAbort) {
  startWorkers(/*abortEnabled=*/true);
  for (const int rank : {0, 1}) {
    run(rank, [this, rank](PerRankState& state) {
      const int peer = rank ^ 1;
      auto timer = launchWatcher(state);
      runSend(peer, kBufferNElem, state, /*expectError=*/true);
    });
  }
}

TEST_F(CtranSendRecvTest, TwoReceiversTimeout) {
  startWorkers(/*abortEnabled=*/true);
  for (const int rank : {0, 1}) {
    run(rank, [this, rank](PerRankState& state) {
      const int peer = rank ^ 1;
      runRecv(
          peer,
          kBufferNElem,
          state,
          /*expectError=*/true,
          kDefaultTimeoutTwoSeconds);
    });
  }
}

TEST_F(CtranSendRecvTest, TwoSendersTimeout) {
  startWorkers(/*abortEnabled=*/true);
  for (const int rank : {0, 1}) {
    run(rank, [this, rank](PerRankState& state) {
      const int peer = rank ^ 1;
      runSend(
          peer,
          kBufferNElem,
          state,
          /*expectError=*/true,
          kDefaultTimeoutTwoSeconds);
    });
  }
}

class CtranSendRecvIbTest : public CtranSendRecvTest {
 public:
  CtranSendRecvIbTest() = default;

  void SetUp() override {
    setenv(
        "NCCL_CTRAN_IB_QP_CONFIG_ALGO",
        "alltoall:1048576,16,dqplb,128,224|sendrecv:1048576,32,dqplb,128,224",
        1);
    ncclCvarInit();
    CtranSendRecvTest::SetUp();
  }
};

TEST_F(CtranSendRecvIbTest, SendRecvIbconfig) {
  startWorkers(/*abortEnabled=*/true);

  // Only rank 0 needs to validate the config since it's a global setting
  run(/*rank=*/0, [this](PerRankState& state) {
    auto* ctranComm = state.ctranComm.get();
    ASSERT_NE(ctranComm, nullptr);

    // Skip test if ctran algo is not available
    if (ctranComm->ctran_->algo == nullptr) {
      GTEST_SKIP() << "No ctran algo found, skip test";
    }

    // Get the IB config for SENDRECV collective
    CtranIbConfig* ctranIbConfigPtr =
        ctranComm->ctran_->algo->getCollToVcConfig(CollType::SENDRECV);

    ASSERT_NE(ctranIbConfigPtr, nullptr)
        << "SendRecv IB config should not be null";

    // Verify the configuration matches the environment variable set in SetUp:
    EXPECT_EQ(ctranIbConfigPtr->qpScalingTh, 1048576);
    EXPECT_EQ(ctranIbConfigPtr->numQps, 32);
    EXPECT_EQ(ctranIbConfigPtr->vcMode, NCCL_CTRAN_IB_VC_MODE::dqplb);
    EXPECT_EQ(ctranIbConfigPtr->qpMsgs, 128);
  });

  run(/*rank=*/1, [](PerRankState& state) {
    // No operation needed, just exit
  });
}

} // namespace ctran::testing
