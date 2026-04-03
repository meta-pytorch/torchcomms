// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <string>

#include "proxy.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/ProxyMock.h"

class ProxyMockTest : public ::testing::Test {
 public:
  ProxyMockTest() = default;

  void SetUp() override {
    // Ensure mock config env var is initialized
    ncclCvarInit();
  }

  void TearDown() override {}

  struct MockConfig {
    int opCount{100};
    int rank{1};
    int remoteRank{2};
    int step{10};
    int numMatch{1};
    int delaySec{0};

    void reset() {
      opCount = 100;
      rank = 1;
      remoteRank = 2;
      step = 10;
      numMatch = 1;
      delaySec = 0;
    }
  };

  void setMatchingSubArg(MockConfig& config, struct ncclProxySubArgs& subArgs) {
    subArgs.traceArgs.collInfo.commHash = commHash_;
    subArgs.traceArgs.collInfo.opCount = config.opCount;
    subArgs.traceArgs.proxyOpId = proxyOpId_;
    subArgs.traceArgs.rank = config.rank;
    subArgs.traceArgs.remoteRank = config.remoteRank;
  }

  void setMockConfig(MockConfig& config) {
    NCCL_PROXYMOCK_NET_SEND_FAILURE.clear();
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(config.opCount));
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(config.rank));
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(
        std::to_string(config.remoteRank));
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(config.step));
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(config.numMatch));
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(config.delaySec));

    // Manually re-initialze state of the mock instance
    auto& instance = ProxyMockNetSendFailure::getInstance();
    instance.initialize();
  }

 protected:
  const uint64_t commHash_{0x9999};
  const int proxyOpId_{2};
};

TEST_F(ProxyMockTest, ExactMatch) {
  struct MockConfig mockConfig;
  struct ncclProxySubArgs subArgs;
  void* request = NULL;

  setMockConfig(mockConfig);
  setMatchingSubArg(mockConfig, subArgs);

  // mock is performed for a subArgs with the same signature
  for (int i = 0; i < 5; i++) {
    bool mocked =
        ProxyMockNetSendFailure::mock(&subArgs, mockConfig.step, &request);
    EXPECT_TRUE(mocked);
  }
}

TEST_F(ProxyMockTest, MultiExactMatch) {
  struct MockConfig mockConfig;
  struct ncclProxySubArgs subArgs;
  void* request = NULL;

  mockConfig.numMatch = 10;
  setMockConfig(mockConfig);
  setMatchingSubArg(mockConfig, subArgs);

  // mock is performed at exact numMatch times
  for (int i = 0; i < 20; i++) {
    // change proxyOpId so mock considers it as a different signature
    subArgs.traceArgs.proxyOpId = i;
    bool mocked =
        ProxyMockNetSendFailure::mock(&subArgs, mockConfig.step, &request);
    if (i < mockConfig.numMatch) {
      EXPECT_TRUE(mocked);
    } else {
      EXPECT_FALSE(mocked);
    }
  }
}

TEST_F(ProxyMockTest, Disable) {
  struct MockConfig mockConfig;
  struct ncclProxySubArgs subArgs;
  void* request = NULL;

  // do not set mock config to disable mock
  NCCL_PROXYMOCK_NET_SEND_FAILURE.clear();
  auto& instance = ProxyMockNetSendFailure::getInstance();
  instance.initialize();
  setMatchingSubArg(mockConfig, subArgs);
  bool mocked =
      ProxyMockNetSendFailure::mock(&subArgs, mockConfig.step, &request);
  EXPECT_FALSE(mocked);

  // set num_match to less than 1 to disable mock
  mockConfig.numMatch = 0;
  setMockConfig(mockConfig);
  mocked = ProxyMockNetSendFailure::mock(&subArgs, mockConfig.step, &request);
  EXPECT_FALSE(mocked);
}

TEST_F(ProxyMockTest, NotMatch) {
  struct MockConfig mockConfig;
  struct ncclProxySubArgs subArgs;
  void* request = NULL;
  bool mocked = true;

  setMockConfig(mockConfig);

  // opCount doesn't match
  setMatchingSubArg(mockConfig, subArgs);
  subArgs.traceArgs.collInfo.opCount = mockConfig.opCount + 1;
  mocked = ProxyMockNetSendFailure::mock(&subArgs, mockConfig.step, &request);
  EXPECT_FALSE(mocked);

  // rank doesn't match
  setMatchingSubArg(mockConfig, subArgs);
  subArgs.traceArgs.rank = mockConfig.rank + 1;
  mocked = ProxyMockNetSendFailure::mock(&subArgs, mockConfig.step, &request);
  EXPECT_FALSE(mocked);

  // remoteRank doesn't match
  setMatchingSubArg(mockConfig, subArgs);
  subArgs.traceArgs.remoteRank = mockConfig.remoteRank + 1;
  mocked = ProxyMockNetSendFailure::mock(&subArgs, mockConfig.step, &request);
  EXPECT_FALSE(mocked);

  // do not match any step smaller than specified
  setMatchingSubArg(mockConfig, subArgs);
  for (int i = 0; i < mockConfig.step; i++) {
    mocked = ProxyMockNetSendFailure::mock(&subArgs, i, &request);
    EXPECT_FALSE(mocked);
  }
}

TEST_F(ProxyMockTest, MatchAny) {
  struct MockConfig mockConfig;
  struct ncclProxySubArgs subArgs;
  void* request = NULL;

  // match any opCount
  setMatchingSubArg(mockConfig, subArgs);
  mockConfig.opCount = -1;
  setMockConfig(mockConfig);
  subArgs.traceArgs.collInfo.opCount = 99;

  bool mocked =
      ProxyMockNetSendFailure::mock(&subArgs, mockConfig.step, &request);
  EXPECT_TRUE(mocked);

  // match any rank
  mockConfig.reset();
  setMatchingSubArg(mockConfig, subArgs);
  mockConfig.rank = -1;
  setMockConfig(mockConfig);

  mocked = ProxyMockNetSendFailure::mock(&subArgs, mockConfig.step, &request);
  EXPECT_TRUE(mocked);

  // match any remoteRank
  mockConfig.reset();
  setMatchingSubArg(mockConfig, subArgs);
  mockConfig.remoteRank = -1;
  setMockConfig(mockConfig);

  mocked = ProxyMockNetSendFailure::mock(&subArgs, mockConfig.step, &request);
  EXPECT_TRUE(mocked);

  // match any step
  mockConfig.reset();
  setMatchingSubArg(mockConfig, subArgs);
  mockConfig.step = -1;
  mockConfig.numMatch = 10;
  setMockConfig(mockConfig);

  for (int i = 0; i < mockConfig.numMatch; i++) {
    mocked = ProxyMockNetSendFailure::mock(&subArgs, i, &request);
    EXPECT_TRUE(mocked);
  }
}

TEST_F(ProxyMockTest, Delay) {
  struct MockConfig mockConfig;
  struct ncclProxySubArgs subArgs;
  void* request = NULL;

  mockConfig.delaySec = 10;
  setMockConfig(mockConfig);
  setMatchingSubArg(mockConfig, subArgs);

  // Expect mock delays the sends until the delaySec is reached
  auto begin = std::chrono::high_resolution_clock::now();
  bool mocked = false;
  do {
    mocked = ProxyMockNetSendFailure::mock(&subArgs, mockConfig.step, &request);
    // wake up every 2 seconds to check if mock is passed to avoid burning CPU
    sleep(2);
  } while (mocked);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - begin);
  EXPECT_GE(duration.count(), mockConfig.delaySec);
}
