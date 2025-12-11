// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"
#include "comms/ctran/algos/AllToAllvDedup/ResourceImpl.h"
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/tests/CtranUtUtils.h"
#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/testinfra/TestUtils.h"

using ctran::alltoallvdedup::ExecArgs;
using ctran::alltoallvdedup::ExecKernArgs;
using ctran::alltoallvdedup::PersistArgs;
using ctran::alltoallvdedup::PersistConfig;
using ctran::alltoallvdedup::PrepareConfig;
using ctran::alltoallvdedup::ResourceImpl;
using ncclx::CommStateX;

namespace ctran::alltoallvdedup {
extern commResult_t launchPrepareForTest(
    const ExecKernArgs& args,
    const PrepareConfig& config,
    const ncclx::CommStateX* statex,
    const CtranAlgoDeviceState* dStateMock,
    cudaStream_t stream,
    const int role);
extern void setupPrepareKernelConfig(
    const ExecKernArgs& args,
    const ncclx::CommStateX* statex,
    PrepareConfig& config);
} // namespace ctran::alltoallvdedup

namespace {

using PrepareCombinedParam = std::tuple<int, int, int, int, int, int>;

class AllToAllvDedupPrepareBench : public ::testing::Test,
                                   public CudaBenchBase,
                                   public CtranBaseTest {
 public:
  AllToAllvDedupPrepareBench() = default;

 protected:
  void SetUp() override {
    ncclCvarInit();
    ctran::logging::initCtranLogging();
  }

  CtranAlgoDeviceState* dStateMock_;
  std::unique_ptr<CommStateX> statexMock_;
  std::unique_ptr<ResourceImpl> resource_;
  std::vector<void*> deviceArgs_;

  struct {
    int totalNumSendBlocks;
    int numBuckets;
    int numRecvBuckets;
    int nRanks;
    int nNodes;
    int nLocalRanks;
    int numThreads;
    int roles;
  } params_;

  void setPrepareTestParam(PrepareCombinedParam& params);
  void setResetTestParam(std::tuple<int, int, int>& params);

  void setStatexMock();

  ExecKernArgs createKernArgsMock(ExecArgs& execArgs);

  ExecArgs createExecArgsMock();
  void assignResourceBuf(ExecKernArgs& args);
};

void AllToAllvDedupPrepareBench::assignResourceBuf(ExecKernArgs& args) {
  resource_ = std::make_unique<ResourceImpl>(
      statexMock_.get(),
      nullptr, // mapper
      nullptr // logMetadata
  );
  constexpr bool kSkipRem = true;
  COMMCHECK_ASSERT(
      resource_->initialize(args.pArgs, args.config, stream, kSkipRem));
  resource_->assignToKernArgs(args, kSkipRem);
}

void AllToAllvDedupPrepareBench::setStatexMock() {
  CtranAlgoDeviceState dStateMockH;
  dStateMockH.enableTraceLog = NCCL_CTRAN_ENABLE_DEV_TRACE_LOG;
  dStateMockH.statex.rank_ = 0;
  dStateMockH.statex.pid_ = 0;
  dStateMockH.statex.localRank_ = 0;
  dStateMockH.statex.localRanks_ = params_.nLocalRanks;
  dStateMockH.statex.nRanks_ = params_.nRanks;
  dStateMockH.statex.nNodes_ = params_.nNodes;
  dStateMockH.statex.commHash_ = 999;
  allocDevArg({dStateMockH}, dStateMock_);

  statexMock_ = std::make_unique<CommStateX>(
      0,
      params_.nRanks,
      0,
      90, // H100
      25, // busId
      999,
      std::vector<ncclx::RankTopology>{},
      std::vector<int>{},
      "");

  statexMock_->initRankTopologyVnode(params_.nLocalRanks);

  ASSERT_EQ(statexMock_->localRank(), 0);
  ASSERT_EQ(statexMock_->node(), 0);
  ASSERT_EQ(statexMock_->nNodes(), params_.nNodes);
  ASSERT_EQ(statexMock_->nLocalRanks(), params_.nLocalRanks);
}

void AllToAllvDedupPrepareBench::setPrepareTestParam(
    PrepareCombinedParam& params) {
  auto& [prepareRole, totalNumSendBlocks, numBuckets, nLocalRanks, nNodes, numThreads] =
      params;
  const auto nRanks = nLocalRanks * nNodes;
  params_.totalNumSendBlocks = totalNumSendBlocks;
  params_.numBuckets = numBuckets;
  params_.numRecvBuckets = numBuckets / nRanks;
  params_.nRanks = nRanks;
  params_.nLocalRanks = nLocalRanks;
  params_.nNodes = nNodes;
  params_.numThreads = numThreads;
  params_.roles = prepareRole;
}

void AllToAllvDedupPrepareBench::setResetTestParam(
    std::tuple<int, int, int>& params) {
  auto& [nLocalRanks, nNodes, numThreads] = params;
  const auto nRanks = nLocalRanks * nNodes;
  params_.totalNumSendBlocks = 1;
  params_.numRecvBuckets = 1;
  params_.nRanks = nRanks;
  params_.nLocalRanks = nLocalRanks;
  params_.nNodes = nNodes;
  params_.numThreads = numThreads;
}

ExecKernArgs AllToAllvDedupPrepareBench::createKernArgsMock(
    ExecArgs& execArgs) {
  // mimic 4MB chunk size, each block is 8192 * sizeof(int);
  // it is OK to keep them constant as not impact on prepare perf nor
  // correctness check
  const int kChunkSize = 4 * 1024 * 1024;
  const int kNumChunks = 4;
  const int kBlockCount = 8192;
  const int typeSize = 4;
  const int kBlockNumRecvBuckets = 8;
  const int maxNumStepBlks = kChunkSize / (kBlockCount * typeSize);
  const int maxNumSteps =
      params_.totalNumSendBlocks * params_.nRanks / maxNumStepBlks + 1;
  const uint64_t opCount = 0;

  PersistArgs pArgs = {
      .totalNumSendBlocks = params_.totalNumSendBlocks,
      .blockCount = kBlockCount,
      .blockNumRecvBuckets = kBlockNumRecvBuckets,
      .numRecvBuckets = params_.numRecvBuckets,
      .datatype = commInt32,
      .maxNumSteps = maxNumSteps,
      .maxNumStepBlks = maxNumStepBlks,
  };

  PersistConfig config = {
      .numThreads = params_.numThreads,
      .numSendGroups = 1,
      .numSendWorkers = 1,
      .numFwdWorkers = 1,
      .numRecvGroups = 1,
      .numRecvWorkers = 1,
      .numIntraFwdWorkers = 1,
      .numIntraRecvWorkers = 1,
      .tmpChunkSize = kChunkSize,
      .tmpNumChunks = kNumChunks};

  return {
      .opCount = opCount,
      .execArgs = execArgs,
      .pArgs = pArgs,
      .config = config};
}

ExecArgs AllToAllvDedupPrepareBench::createExecArgsMock() {
  std::vector<int> sendIdxH(params_.nNodes * params_.totalNumSendBlocks);
  std::vector<int> fwdIdxH(
      params_.nLocalRanks * params_.nNodes * params_.totalNumSendBlocks);
  std::vector<int> recvIdxH(
      params_.numRecvBuckets * params_.nRanks * params_.totalNumSendBlocks);
  for (int n = 0; n < params_.nNodes; n++) {
    int idx = 0;
    const auto offset = n * params_.totalNumSendBlocks;
    for (int b = 0; b < params_.totalNumSendBlocks; b++) {
      sendIdxH[offset + b] = b % 2 ? idx++ : -1;
    }
  }
  for (int r = 0; r < params_.nLocalRanks; r++) {
    for (int n = 0; n < params_.nNodes; n++) {
      int idx = 0;
      const auto offset = (r * params_.nNodes + n) * params_.totalNumSendBlocks;
      for (int b = 0; b < params_.totalNumSendBlocks; b++) {
        fwdIdxH[offset + b] = b % 2 ? idx++ : -1;
      }
    }
  }
  for (int bkt = 0; bkt < params_.numRecvBuckets; bkt++) {
    for (int r = 0; r < params_.nRanks; r++) {
      int idx = 0;
      const auto offset =
          (bkt * params_.nRanks + r) * params_.totalNumSendBlocks;
      for (int b = 0; b < params_.totalNumSendBlocks; b++) {
        recvIdxH[offset + b] = b % 2 ? idx++ : -1;
      }
    }
  }

  int *sendIdx, *fwdIdx, *recvIdx;
  allocDevArg(sendIdxH, sendIdx);
  allocDevArg(fwdIdxH, fwdIdx);
  allocDevArg(recvIdxH, recvIdx);

  return ExecArgs{
      .sendBuff = nullptr,
      .sendIdx = sendIdx,
      .fwdIdx = fwdIdx,
      .recvIdx = recvIdx,
      .recvBuff = nullptr,
      .recvBlockIds = nullptr,
  };
}
} // namespace

using ctran::alltoallvdedup::PrepareRole;

std::string prepareRolesStr(const int roles) {
  if (roles == static_cast<int>(PrepareRole::kPrepareAll)) {
    return "prepareAll";
  } else {
    std::vector<std::string> roleNames;
    if (prepareRoleContains(roles, PrepareRole::kPrepTmpSendIdx)) {
      roleNames.push_back("prepareTmpSendIdx");
    }
    if (prepareRoleContains(roles, PrepareRole::kPrepTmpFwdIdx)) {
      roleNames.push_back("prepareTmpFwdIdx");
    }
    if (prepareRoleContains(roles, PrepareRole::kPrepTmpIntraFwdIdx)) {
      roleNames.push_back("prepareTmpIntraFwdIdx");
    }
    if (prepareRoleContains(roles, PrepareRole::kPrepTmpRecvIdx)) {
      roleNames.push_back("prepareTmpRecvIdx");
    }
    if (prepareRoleContains(roles, PrepareRole::kPrepTmpRecvOffsets)) {
      roleNames.push_back("prepareTmpRecvOffsets");
    }
    if (prepareRoleContains(roles, PrepareRole::kPrepTmpRecvRedIdx)) {
      roleNames.push_back("prepareTmpRecvRedIdx");
    }
    if (prepareRoleContains(roles, PrepareRole::kResetSync)) {
      roleNames.push_back("resetSync");
    }
    return folly::join(",", roleNames);
  }
}

class AllToAllvDedupPrepareBenchParamFixture
    : public AllToAllvDedupPrepareBench,
      public ::testing::WithParamInterface<PrepareCombinedParam> {};

TEST_P(AllToAllvDedupPrepareBenchParamFixture, PreparePerf) {
  auto param = GetParam();
  setPrepareTestParam(param);

  const int numIter = 1000;
  const int numWarmup = 50;

  setStatexMock();
  ExecArgs execArgs = createExecArgsMock();
  ExecKernArgs args = createKernArgsMock(execArgs);
  assignResourceBuf(args);

  PrepareConfig config;
  setupPrepareKernelConfig(args, statexMock_.get(), config);
  // override test-specified numThreads; used to launch kernel in
  // launchPrepareForTest
  config.numThreads = params_.numThreads;
  const auto numBlocks = config.numSendIdxWorkers +
      config.numIntraFwdIdxWorkers + config.numFwdIxWorkers +
      config.numRecvIdxWorkers + config.numRecvOffsetWorkers +
      config.numRecvRedIdxWorkers + config.numResetSyncWorkers;

  for (auto x = 0; x < numWarmup; x++) {
    COMMCHECK_ASSERT(launchPrepareForTest(
        args, config, statexMock_.get(), dStateMock_, stream, params_.roles));
    args.opCount++;
  }
  CUDACHECK_ASSERT(cudaDeviceSynchronize());

  // Actual run
  startTiming();
  for (auto x = 0; x < numIter; x++) {
    COMMCHECK_ASSERT(launchPrepareForTest(
        args, config, statexMock_.get(), dStateMock_, stream, params_.roles));
    args.opCount++;
  }
  stopTiming();
  CUDACHECK_ASSERT(cudaDeviceSynchronize());
  float gpuTimeMs = measureTime();
  std::cout << fmt::format(
      "{} totalNumSendBlocks {} numBuckets {} nLocalRanks {} nNodes {} (numRecvBuckets {}) numThreadBlocks {} ({},{},{},{},{},{},{}) numThreads {} latency {:.2f} us from {} iterations \n",
      prepareRolesStr(params_.roles),
      params_.totalNumSendBlocks,
      params_.numBuckets,
      params_.nLocalRanks,
      params_.nNodes,
      params_.numRecvBuckets,
      numBlocks,
      config.numSendIdxWorkers,
      config.numIntraFwdIdxWorkers,
      config.numFwdIxWorkers,
      config.numRecvIdxWorkers,
      config.numRecvOffsetWorkers,
      config.numRecvRedIdxWorkers,
      config.numResetSyncWorkers,
      params_.numThreads,
      gpuTimeMs * 1e3 / numIter,
      numIter);

  // release input arguments and statexMock; resource buffers released at
  // resource destructor.
  releaseDevArgs();
}

std::vector<PrepareCombinedParam> GenPrepareCombinParams() {
  std::vector<PrepareCombinedParam> allParams;

  std::vector<int> allBlocks = {1024, 2048, 4096, 8192};
  std::vector<int> allBuckets = {64, 128, 256};
  std::vector<int> allLocalRanks = {2, 4, 8};
  std::vector<int> allNodes = {2, 4, 8};
  std::vector<int> allThreads = {128, 256, 512};

  const auto kBuckets = 128;
  const auto kNLocalRanks = 8;
  const auto kNNodes = 4;
  const auto kBlocks = 8192;
  const auto kThreads = 256;

  auto v1 = (int)PrepareRole::kPrepareAll;
  for (const auto& v2 : allBlocks) {
    for (const auto& v3 : allBuckets) {
      for (const auto& v5 : allNodes) {
        for (const auto& v6 : allThreads) {
          if (1) {
            allParams.emplace_back(
                std::make_tuple(v1, v2, v3, kNLocalRanks, v5, v6));
          }
        }
      }
    }
  }

  std::vector<int> roles = {
      (int)PrepareRole::kPrepTmpSendIdx, (int)PrepareRole::kPrepTmpFwdIdx};
  for (const auto& v1 : roles) {
    for (const auto& v2 : allBlocks) {
      for (const auto& v5 : allNodes) {
        if (1) {
          allParams.emplace_back(
              std::make_tuple(v1, v2, kBuckets, kNLocalRanks, v5, kThreads));
        }
      }
    }
  }

  v1 = (int)PrepareRole::kPrepTmpIntraFwdIdx;
  for (const auto& v2 : allBlocks) {
    for (const auto& v4 : allLocalRanks) {
      if (1) {
        allParams.emplace_back(
            std::make_tuple(v1, v2, kBuckets, v4, kNNodes, kThreads));
      }
    }
  }

  roles = {
      (int)PrepareRole::kPrepTmpRecvIdx,
      (int)PrepareRole::kPrepTmpRecvOffsets,
      (int)PrepareRole::kPrepTmpRecvRedIdx};
  for (const auto& v1 : roles) {
    for (const auto& v2 : allBlocks) {
      for (const auto& v3 : allBuckets) {
        for (const auto& v5 : allNodes) {
          if (1) {
            allParams.emplace_back(
                std::make_tuple(v1, v2, v3, kNLocalRanks, v5, kThreads));
          }
        }
      }
    }
  }

  v1 = (int)PrepareRole::kResetSync;
  for (const auto& v4 : allLocalRanks) {
    for (const auto& v5 : allNodes) {
      if (1) {
        allParams.emplace_back(
            std::make_tuple(v1, kBlocks, kBuckets, v4, v5, kThreads));
      }
    }
  }

  return allParams;
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    AllToAllvDedupPrepareBenchParamFixture,
    testing::ValuesIn(GenPrepareCombinParams()),
    [&](const testing::TestParamInfo<
        AllToAllvDedupPrepareBenchParamFixture::ParamType>& info) {
      return prepareRolesStr(std::get<0>(info.param)) + "_" +
          std::to_string(std::get<1>(info.param)) + "blocks_" +
          std::to_string(std::get<2>(info.param)) + "buckets_" +
          std::to_string(std::get<3>(info.param)) + "nLocalRanks_" +
          std::to_string(std::get<4>(info.param)) + "nNodes_" +
          std::to_string(std::get<5>(info.param)) + "numThreads";
    });
