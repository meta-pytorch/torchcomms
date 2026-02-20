// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/Random.h>
#include <folly/init/Init.h>
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

#include "comm.h"
#include "nccl.h"

#include "comms/utils/cvars/nccl_cvars.h"

class NcclxLazyConnectTestFixture : public NcclxBaseTestFixture {
 public:
  ncclComm_t rootComm{nullptr};
  cudaStream_t stream{nullptr};
  void* sendBuf{nullptr};
  void* recvBuf{nullptr};
  ncclDataType_t dataType{ncclBfloat16};

 protected:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  void TearDown() override {
    if (sendBuf) {
      CUDACHECK_TEST(cudaFree(sendBuf));
      sendBuf = nullptr;
    }
    if (recvBuf) {
      CUDACHECK_TEST(cudaFree(recvBuf));
      recvBuf = nullptr;
    }
    if (stream) {
      CUDACHECK_TEST(cudaStreamDestroy(stream));
      stream = nullptr;
    }
    NcclxBaseTestFixture::TearDown();
  }

  void splitComm(ncclComm_t* newChildComm) {
    ncclComm_t childComm;
    ncclConfig_t childCommConfig = NCCL_CONFIG_INITIALIZER;
    childCommConfig.commDesc = "child_communicator";
    // split rootComm into two communicators, in round-robin fashion
    // e.g. 8-rank rootComm ->
    //        ranks 0, 2, 4, 6 form 1st childComm
    //        ranks 1, 3, 5, 7 form 2nd childComm
    // int groupSize = rootComm->statex->nRanks() / 2;
    int groupSize = rootComm->nRanks / 2;
    int* groupRanks = new int[groupSize];
    for (int i = 0; i < groupSize; ++i) {
      groupRanks[i] = 2 * i + globalRank % 2;
    }
    // childCommConfig.splitGroupRanks = groupRanks;
    // childCommConfig.splitGroupSize = groupSize;
    NCCLCHECK_TEST(ncclCommSplit(
        rootComm, globalRank % 2, globalRank, &childComm, &childCommConfig));
    ASSERT_NE(nullptr, childComm);

    *newChildComm = childComm;
  }

  void
  prepBuffers(size_t sendBytes, size_t recvBytes, ncclComm_t comm = nullptr) {
    if (comm == nullptr) {
      comm = rootComm;
    }
    ASSERT_NE(comm, nullptr);
    CUDACHECK_TEST(cudaMalloc(&sendBuf, sendBytes));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, recvBytes));
    CUDACHECK_TEST(cudaMemset(sendBuf, comm->rank, sendBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvBytes));
    // Ensure value has been set before colletive runs on nonblocking stream
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  void checkAlgoInitState(ncclComm_t comm, int expectedAlgo) {
    // NOTE: single-rank communicator won't connect algorithms, skip the check
    if (comm->nRanks == 1) {
      return;
    }
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      if (a == expectedAlgo) {
        EXPECT_TRUE(comm->initAlgoChannels[a])
            << "Algorithm " << expectedAlgo << " is expected to be connected";
      } else {
        EXPECT_FALSE(comm->initAlgoChannels[a]);
      }
    }
  }

  void checkAlgoChannelState(
      ncclComm_t comm,
      int algoShouldConnect,
      bool isLazyConnect) {
    // NOTE: single-rank communicator won't setup any channels and algorithms,
    // skip the check
    if (comm->nRanks == 1) {
      return;
    }
    if (algoShouldConnect != NCCL_NUM_ALGORITHMS) {
      // some channels should be initialized if we expect any algorithm to be
      // connected
      EXPECT_NE(comm->nChannelsReady, 0);
    }
    // FIXME: only check RING and TREE for now as COLLNET and NVLS are not
    // well supported for lazy connect and lazy channel features
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      if (a == algoShouldConnect) {
        EXPECT_NE(comm->algoConnectedChannels[a], 0);
      } else if (
          isLazyConnect && (a == NCCL_ALGO_RING || a == NCCL_ALGO_TREE)) {
        EXPECT_EQ(comm->algoConnectedChannels[a], 0);
      }
    }
  }
};

TEST_P(NcclxLazyConnectTestFixture, InitOnly) {
  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &rootComm, numRanks, ncclUid, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);
  // Nothing should be connected or initialized if no collective is called
  if (NCCL_RUNTIME_CONNECT) {
    // Algorithms should not be connected
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      EXPECT_FALSE(rootComm->initAlgoChannels[a]);
    }
  }
  if (NCCL_LAZY_SETUP_CHANNELS) {
    // channels should not be initialized
    EXPECT_EQ(rootComm->nChannelsReady, 0);
  }
  if (NCCL_RUNTIME_CONNECT) {
    // Algorithms should not be connected
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      EXPECT_EQ(rootComm->algoConnectedChannels[a], 0);
    }
  }
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_P(NcclxLazyConnectTestFixture, AllReduceRing) {
  EnvRAII algo(NCCL_ALGO, std::string("RING"));
  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &rootComm, numRanks, ncclUid, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);

  size_t count = 1 << 10; // 1K elements
  prepBuffers(count * ncclTypeSize(dataType), count * ncclTypeSize(dataType));

  // run baseline allreduce
  auto res = ncclAllReduce(
      sendBuf, recvBuf, count, dataType, ncclSum, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);

  if (NCCL_RUNTIME_CONNECT) {
    // RING should be connected
    checkAlgoInitState(rootComm, NCCL_ALGO_RING);
  }
  if (NCCL_LAZY_SETUP_CHANNELS) {
    // RING should be connected in rootComm
    // other algorithms should not be connected if NCCL_RUNTIME_CONNECT is
    // enabled
    checkAlgoChannelState(rootComm, NCCL_ALGO_RING, NCCL_RUNTIME_CONNECT);
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_P(NcclxLazyConnectTestFixture, AllReduceTree) {
  EnvRAII algo(NCCL_ALGO, std::string("TREE"));
  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &rootComm, numRanks, ncclUid, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);

  size_t count = 1 << 10; // 1K elements
  prepBuffers(count * ncclTypeSize(dataType), count * ncclTypeSize(dataType));

  // run baseline allreduce
  auto res = ncclAllReduce(
      sendBuf, recvBuf, count, dataType, ncclSum, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);

  if (NCCL_RUNTIME_CONNECT) {
    // TREE should be connected
    checkAlgoInitState(rootComm, NCCL_ALGO_TREE);
  }
  if (NCCL_LAZY_SETUP_CHANNELS) {
    // TREE should be connected in rootComm
    // other algorithms should not be connected if NCCL_RUNTIME_CONNECT is
    // enabled
    checkAlgoChannelState(rootComm, NCCL_ALGO_TREE, NCCL_RUNTIME_CONNECT);
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

// Test that AllReduce with NVLS algorithm works under lazy connect/setup.
TEST_P(NcclxLazyConnectTestFixture, AllReduceNvls) {
  EnvRAII algo(NCCL_ALGO, std::string("AllReduce:NVLS"));
  EnvRAII nvlsEnable(NCCL_NVLS_ENABLE, (int64_t)1);
  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &rootComm, numRanks, ncclUid, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);

  // With lazy setup channels, nvlsResources should not be initialized yet
  if (NCCL_LAZY_SETUP_CHANNELS) {
    EXPECT_EQ(rootComm->nvlsResources, nullptr);
  }

  size_t count = 1 << 10; // 1K elements
  prepBuffers(count * ncclTypeSize(dataType), count * ncclTypeSize(dataType));

  auto res = ncclAllReduce(
      sendBuf, recvBuf, count, dataType, ncclSum, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_P(NcclxLazyConnectTestFixture, AllReduceTreeIncreaseChannel) {
  EnvRAII algo(NCCL_ALGO, std::string("TREE"));
  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &rootComm, numRanks, ncclUid, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);

  size_t smallCount = 1 << 10; // 1K elements
  size_t count = 1 << 20; // 1M BF16 elements
  prepBuffers(count * ncclTypeSize(dataType), count * ncclTypeSize(dataType));

  // run baseline allreduce
  auto res = ncclAllReduce(
      sendBuf, recvBuf, smallCount, dataType, ncclSum, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);
  res = ncclAllReduce(
      sendBuf, recvBuf, count, dataType, ncclSum, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);

  if (NCCL_RUNTIME_CONNECT) {
    // TREE should be connected
    checkAlgoInitState(rootComm, NCCL_ALGO_TREE);
  }
  if (NCCL_LAZY_SETUP_CHANNELS) {
    // TREE should be connected in rootComm
    checkAlgoChannelState(rootComm, NCCL_ALGO_TREE, NCCL_RUNTIME_CONNECT);
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_P(NcclxLazyConnectTestFixture, Alltoall) {
  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &rootComm, numRanks, ncclUid, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);

  size_t count = 1 << 20; // 1M BF16 elements
  size_t bytesPerRank = count * ncclTypeSize(dataType);
  size_t sendBytes = bytesPerRank * numRanks;
  size_t recvBytes = sendBytes;

  prepBuffers(sendBytes, recvBytes, rootComm);

  // run small alltoall
  auto res = ncclAllToAll(sendBuf, recvBuf, 2, dataType, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  auto prevNchannelsReady = rootComm->nChannelsReady;
  if (NCCL_RUNTIME_CONNECT) {
    // Algorithms should not be connected
    checkAlgoInitState(rootComm, NCCL_NUM_ALGORITHMS);
  }
  if (NCCL_LAZY_SETUP_CHANNELS) {
    // Algorithms should not be connected
    checkAlgoChannelState(
        rootComm, NCCL_NUM_ALGORITHMS, NCCL_LAZY_SETUP_CHANNELS);
  }

  // run large alltoall
  res = ncclAllToAll(sendBuf, recvBuf, count, dataType, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // more channels should initialized for large Alltoall, in lazy channel mode
  if (NCCL_LAZY_SETUP_CHANNELS && rootComm->nRanks > 1) {
    EXPECT_GE(rootComm->nChannelsReady, prevNchannelsReady);
  }

  for (int i = 0; i < rootComm->nRanks; ++i) {
    std::vector<char> observedVals(bytesPerRank, folly::Random::rand32());
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(),
        (char*)recvBuf + bytesPerRank * i,
        bytesPerRank,
        cudaMemcpyDefault));
    EXPECT_THAT(observedVals, testing::Each(i));
  }

  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_P(NcclxLazyConnectTestFixture, AlltoallAndAllGather) {
  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &rootComm, numRanks, ncclUid, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);

  size_t count = 1 << 20; // 1M BF16 elements
  size_t bytesPerRank = count * ncclTypeSize(dataType);
  size_t sendBytes = bytesPerRank * numRanks;
  size_t recvBytes = sendBytes * numRanks;

  prepBuffers(sendBytes, recvBytes, rootComm);

  // run alltoall
  auto res = ncclAllToAll(sendBuf, recvBuf, count, dataType, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);
  // run allgather
  res = ncclAllGather(sendBuf, recvBuf, count, dataType, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  if (NCCL_RUNTIME_CONNECT) {
    // RING should be connected
    checkAlgoInitState(rootComm, NCCL_ALGO_RING);
  }
  if (NCCL_LAZY_SETUP_CHANNELS) {
    // RING should be connected in rootComm
    checkAlgoChannelState(rootComm, NCCL_ALGO_RING, NCCL_RUNTIME_CONNECT);
  }

  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

// test that p2p channels higher than collective channels
// expected behavior is that all channels should initialized
TEST_P(NcclxLazyConnectTestFixture, higherP2pChThanColl) {
  EnvRAII p2pMinCh(NCCL_MIN_P2P_NCHANNELS, (int64_t)MAXCHANNELS);
  EnvRAII p2pMaxCh(NCCL_MAX_P2P_NCHANNELS, (int64_t)MAXCHANNELS);

  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &rootComm, numRanks, ncclUid, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);
  // p2p channels should be higher than collective channels
  EXPECT_GE(rootComm->p2pnChannels, rootComm->collChannels);

  size_t count = 1 << 21; // 2M BF16 elements
  size_t allgatherCount = 1 << 10; // 1K elements for allgather
  size_t bytesPerRank = count * ncclTypeSize(dataType);
  size_t sendBytes = bytesPerRank * numRanks;
  size_t recvBytes = sendBytes * numRanks;

  prepBuffers(sendBytes, recvBytes, rootComm);

  // run alltoall
  auto res = ncclAllToAll(sendBuf, recvBuf, count, dataType, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);
  // run allgather
  res = ncclAllGather(
      sendBuf, recvBuf, allgatherCount, dataType, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  if (NCCL_RUNTIME_CONNECT) {
    // RING should be connected
    checkAlgoInitState(rootComm, NCCL_ALGO_RING);
  }
  if (NCCL_LAZY_SETUP_CHANNELS) {
    // RING should be connected in rootComm
    checkAlgoChannelState(rootComm, NCCL_ALGO_RING, NCCL_RUNTIME_CONNECT);
  }

  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_P(NcclxLazyConnectTestFixture, ChildCommAllGather) {
  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &rootComm, numRanks, ncclUid, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);

  ncclComm_t childComm;
  splitComm(&childComm);

  size_t count = 1 << 10; // 1K elements
  size_t sendBytes = count * ncclTypeSize(dataType);
  size_t recvBytes = sendBytes * numRanks;

  prepBuffers(sendBytes, recvBytes, childComm);

  // run baseline allgather
  auto res =
      ncclAllGather(sendBuf, recvBuf, count, dataType, childComm, stream);
  EXPECT_EQ(res, ncclSuccess);

  // Allgather should be using RING
  if (NCCL_RUNTIME_CONNECT) {
    // RING should be connected
    checkAlgoInitState(childComm, NCCL_ALGO_RING);
    // rootComm should not connect any algorithm
    checkAlgoInitState(rootComm, NCCL_NUM_ALGORITHMS);
  }
  if (NCCL_LAZY_SETUP_CHANNELS) {
    // RING should be connected in childComm
    checkAlgoChannelState(childComm, NCCL_ALGO_RING, NCCL_RUNTIME_CONNECT);
    // rootComm should not connect any algorithm
    checkAlgoChannelState(rootComm, NCCL_NUM_ALGORITHMS, NCCL_RUNTIME_CONNECT);
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  for (int i = 0; i < childComm->nRanks; ++i) {
    std::vector<char> observedVals(sendBytes, folly::Random::rand32());
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(),
        (char*)recvBuf + i * sendBytes,
        sendBytes,
        cudaMemcpyDefault));
    EXPECT_THAT(observedVals, testing::Each(i));
  }

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

// FIXME: debug unexpected NCCL error for v2.25
// TEST_P(NcclxLazyConnectTestFixture, AllReduceCollNet) {
//   // Test if CollNet works with lazy connect. If collnet is not available,
//   // fallback path will be tested.
//   EnvRAII algo(NCCL_ALGO, std::string("CollnetDirect"));
//   EnvRAII collnet(NCCL_COLLNET_ENABLE, true);
//   NCCLCHECK_TEST(ncclCommInitRankConfig(
//       &rootComm, numRanks, ncclUid, globalRank, nullptr));
//   ASSERT_NE(nullptr, rootComm);

//   size_t count = 1 << 10; // 1K elements
//   prepBuffers(count * ncclTypeSize(dataType), count *
//   ncclTypeSize(dataType));

//   // run baseline allreduce
//   auto res = ncclAllReduce(
//       sendBuf, recvBuf, count, dataType, ncclSum, rootComm, stream);
//   EXPECT_EQ(res, ncclSuccess);

//   if (NCCL_RUNTIME_CONNECT) {
//     // COLLNET would be connected, if available
//     for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
//       if (rootComm->collNetSupport == 1 && a == NCCL_ALGO_COLLNET_DIRECT) {
//         EXPECT_TRUE(rootComm->initAlgoChannels[a]);
//       }
//     }
//   }
//   if (NCCL_LAZY_SETUP_CHANNELS) {
//     // some channels should be initialized
//     EXPECT_NE(rootComm->nChannelsReady, 0);
//     // COLLNET should be connected in rootComm, if available
//     if (rootComm->collNetSupport == 1) {
//       checkAlgoChannelState(
//           rootComm, NCCL_ALGO_COLLNET_DIRECT, NCCL_RUNTIME_CONNECT);
//     }
//   }

//   CUDACHECK_TEST(cudaStreamSynchronize(stream));
//   NCCLCHECK_TEST(ncclCommDestroy(rootComm));
// }

// FIXME: check if we need to enable this test with tuner plugin, which needs
// more work due v2.25 API changes TEST_P(NcclxLazyConnectTestFixture,
// AllReduceTuner) {
//   // Test if tuner pluging works with lazy connect. If pluging is not
//   present,
//   // fallback path will be tested.
//   // TODO: use a mock tuner plugin to test the logic
//   EnvRAII algo(NCCL_ALGO, std::string("TREE"));
//   EnvRAII tuner(NCCL_TUNER_PLUGIN, std::string(""));
//   NCCLCHECK_TEST(
//       ncclCommInitRankConfig(&rootComm, numRanks, ncclUid, globalRank,
//       nullptr));
//   ASSERT_NE(nullptr, rootComm);

//   size_t count = 1 << 10; // 1K elements
//   prepBuffers(count * ncclTypeSize(dataType), count *
//   ncclTypeSize(dataType));

//   int expectedAlgo = NCCL_ALGO_TREE;
//   int expectedProtocol = NCCL_PROTO_LL;
//   int expectedNchannels = 1;
//   if (rootComm->tuner && rootComm->tunerContext) {
//     // tuner plugin is present
//     rootComm->tuner->getCollInfo(
//         rootComm->tunerContext,
//         ncclFuncAllReduce,
//         count * ncclTypeSize(dataType),
//         rootComm->collNetSupport,
//         rootComm->nvlsSupport,
//         0,
//         &expectedAlgo,
//         &expectedProtocol,
//         &expectedNchannels);
//   }

//   // run baseline allreduce
//   auto res = ncclAllReduce(
//       sendBuf, recvBuf, count, dataType, ncclSum, rootComm, stream);
//   EXPECT_EQ(res, ncclSuccess);

//   if (NCCL_RUNTIME_CONNECT) {
//     // Selected algo would be connected
//     for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
//       if (a == expectedAlgo) {
//         EXPECT_TRUE(rootComm->initAlgoChannels[a]);
//       } else {
//         EXPECT_FALSE(rootComm->initAlgoChannels[a]);
//       }
//     }
//   }
//   if (NCCL_LAZY_SETUP_CHANNELS) {
//     // some channels should be initialized
//     EXPECT_GE(rootComm->nChannelsReady, expectedNchannels);
//     // Selected algo should be connected in rootComm
//     checkAlgoChannelState(rootComm, expectedAlgo, NCCL_RUNTIME_CONNECT);
//   }

//   CUDACHECK_TEST(cudaStreamSynchronize(stream));
//   NCCLCHECK_TEST(ncclCommDestroy(rootComm));
// }

TEST_P(NcclxLazyConnectTestFixture, ChildCommLazyConfig) {
  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &rootComm, numRanks, ncclUid, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);
  // split/duplicate a communicator always enable lazy connect and setup
  // channels
  ncclComm_t childComm = nullptr;
  ncclConfig_t childCommConfig = NCCL_CONFIG_INITIALIZER;
  childCommConfig.lazyConnect = 1;
  childCommConfig.lazySetupChannels = 1;
  NCCLCHECK_TEST(
      ncclCommSplit(rootComm, 0, globalRank, &childComm, &childCommConfig));
  ASSERT_NE(nullptr, childComm);

  // child comm should always have lazy connect and setup channels enabled and
  // not allocate any channels
  EXPECT_EQ(childComm->config.lazyConnect, 1);
  EXPECT_EQ(childComm->config.lazySetupChannels, 1);
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    EXPECT_FALSE(childComm->initAlgoChannels[a]);
  }
  EXPECT_EQ(childComm->nChannelsReady, 0);

  // Nothing should be connected or initialized if no collective is called
  if (NCCL_RUNTIME_CONNECT) {
    // Algorithms should not be connected
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      EXPECT_FALSE(rootComm->initAlgoChannels[a]);
      EXPECT_EQ(rootComm->algoConnectedChannels[a], 0);
    }
  }
  if (NCCL_LAZY_SETUP_CHANNELS) {
    // channels should not be initialized
    EXPECT_EQ(rootComm->nChannelsReady, 0);
  }
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_P(NcclxLazyConnectTestFixture, coalescedAllReduce) {
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, ncclUid, globalRank, nullptr));
  ASSERT_NE(nullptr, comm);

  size_t count = 1 << 10; // 1K elements
  prepBuffers(
      count * ncclTypeSize(dataType), count * ncclTypeSize(dataType), comm);

  // posting multiple small allreduce in single group, e.g., coalesced
  constexpr int numAR = 10;
  ncclGroupStart();
  for (int i = 0; i < numAR; ++i) {
    EXPECT_EQ(
        ncclAllReduce(sendBuf, recvBuf, count, dataType, ncclSum, comm, stream),
        ncclSuccess);
  }
  ncclGroupEnd();

  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

INSTANTIATE_TEST_SUITE_P(
    MyTestSuite,
    NcclxLazyConnectTestFixture,
    testing::Values(
        NcclxEnvs({{"NCCL_RUNTIME_CONNECT", "1"}}),
        NcclxEnvs({
            {"NCCL_RUNTIME_CONNECT", "1"},
            {"NCCL_LAZY_SETUP_CHANNELS", "1"},
        }),
        NcclxEnvs({
            {"NCCL_RUNTIME_CONNECT", "0"},
            {"NCCL_LAZY_SETUP_CHANNELS", "1"},
        }),
        NcclxEnvs({
            {"NCCL_RUNTIME_CONNECT", "1"},
            {"NCCL_LAZY_SETUP_CHANNELS", "1"},
            {"NCCL_MEM_USE_SLAB_ALLOCATOR", "1"},
        }),
        NcclxEnvs({
            {"NCCL_RUNTIME_CONNECT", "1"},
            {"NCCL_LAZY_SETUP_CHANNELS", "1"},
            {"NCCL_MEM_USE_SLAB_ALLOCATOR", "1"},
            {"NCCL_USE_TRANSPORT_EXT", "1"},
        }),
        NcclxEnvs({
            {"NCCL_RUNTIME_CONNECT", "1"},
            {"NCCL_LAZY_SETUP_CHANNELS", "1"},
            {"NCCL_USE_TRANSPORT_PROXY", "shared"},
            {"NCCL_CHANNEL_METADATA_LOCATION", "host"},
        })),
    [](const testing::TestParamInfo<NcclxLazyConnectTestFixture::ParamType>&
           info) {
      // generate test-name for a given NcclxEnvs
      std::string name = "";
      for (const auto& [key, val] : info.param) {
        if (key == "NCCL_RUNTIME_CONNECT" && val == "1") {
          name += "lazy_connect_";
        }
        if (key == "NCCL_LAZY_SETUP_CHANNELS" && val == "1") {
          name += "setupChannels_";
        }
        if (key == "NCCL_MEM_USE_SLAB_ALLOCATOR" && val == "1") {
          name += "slab_allocator_";
        }
        if (key == "NCCL_USE_TRANSPORT_EXT" && val == "1") {
          name += "transExt_";
        }
        if (key == "NCCL_USE_TRANSPORT_PROXY" && val != "none") {
          name += "proxy_" + val + "_";
        }
      }
      return name;
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
