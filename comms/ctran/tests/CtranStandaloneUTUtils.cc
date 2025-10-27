// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/tests/CtranStandaloneUTUtils.h"

#include <functional>

#include <cuda_runtime.h>

#include <folly/futures/Future.h>
#include <folly/synchronization/CallOnce.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/tests/bootstrap/IntraProcessBootstrap.h"
#include "comms/ctran/tests/bootstrap/MockBootstrap.h"
#include "comms/ctran/utils/Abort.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/utils/InitFolly.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/Logger.h"

namespace ctran::testing {

namespace {

void initUtLogger(
    const std::string& contextName,
    const std::string& logPrefix) {
  NcclLogger::init(
      NcclLoggerInitConfig{
          .contextName = contextName,
          .logPrefix = logPrefix,
          .logFilePath =
              meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
          .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
              meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
          .threadContextFn = []() {
            int cudaDev = -1;
            cudaGetDevice(&cudaDev);
            return cudaDev;
          }});
}
folly::once_flag once;
void initOnce() {
  folly::call_once(once, [] {
    meta::comms::initFolly();
    ncclCvarInit();
    ctran::utils::commCudaLibraryInit();
    initUtLogger(
        /*contextName=*/"comms.ctran.tests", /*logPrefix=*/"CTran-UT");
  });
}

} // namespace

void CtranStandaloneBaseTest::setupBase() {
  setenv("NCCL_CTRAN_ENABLE", "INFO", 1);
  setenv("NCCL_DEBUG", "INFO", 1);

  rank = 0;
  cudaDev = 0;
  FB_CUDACHECKIGNORE(cudaSetDevice(cudaDev));

  // Ensure logger is initialized
  initOnce();
}

void CtranStandaloneBaseTest::initCtranComm(
    std::shared_ptr<::ctran::utils::Abort> abort) {
  ctranComm = std::make_unique<CtranComm>(abort);

  ctranComm->bootstrap_ = std::make_unique<testing::MockBootstrap>();
  ((testing::MockBootstrap*)ctranComm->bootstrap_.get())
      ->expectSuccessfulCtranInitCalls();

  ncclx::RankTopology topo;
  topo.rank = rank;
  std::strncpy(topo.dc, "ut_dc", ncclx::kMaxNameLen);
  std::strncpy(topo.zone, "ut_zone", ncclx::kMaxNameLen);
  std::strncpy(topo.host, "ut_host", ncclx::kMaxNameLen);
  // we can only set one of the two, rtsw or su.
  std::strncpy(topo.rtsw, "", ncclx::kMaxNameLen);
  std::strncpy(topo.su, "ut_su", ncclx::kMaxNameLen);

  std::vector<ncclx::RankTopology> rankTopologies = {topo};
  std::vector<int> commRanksToWorldRanks = {0};

  ctranComm->statex_ = std::make_unique<ncclx::CommStateX>(
      /*rank=*/0,
      /*nRanks=*/1,
      /*cudaDev=*/cudaDev,
      /*cudaArch=*/900, // H100
      /*busId=*/-1,
      /*commHash=*/1234,
      /*rankTopologies=*/std::move(rankTopologies),
      /*commRanksToWorldRanks=*/std::move(commRanksToWorldRanks),
      /*commDesc=*/kCommDesc);

  ASSERT_EQ(ctranInit(ctranComm.get()), commSuccess);

  CLOGF(INFO, "UT CTran initialized");
}

namespace {

void initRankStatesTopologyWrapper(
    ncclx::CommStateX* statex,
    ctran::bootstrap::IBootstrap* bootstrap,
    int nRanks) {
  // Fake topology with nLocalRanks=1
  if (NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::nolocal) {
    statex->initRankTopologyNolocal();
  } else if (NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::vnode) {
    ASSERT_GE(nRanks, NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS);
    statex->initRankTopologyVnode(NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS);
  } else {
    statex->initRankStatesTopology(std::move(bootstrap));
  }
}

using PerRankState = CtranStandaloneMultiRankBaseTest::PerRankState;
static void resetPerRankState(PerRankState& state) {
  if (state.dstBuffer != nullptr) {
    FB_COMMCHECKTHROW(ctran::utils::commCudaFree(state.dstBuffer));
  }
  if (state.srcBuffer != nullptr) {
    FB_COMMCHECKTHROW(ctran::utils::commCudaFree(state.srcBuffer));
  }
  if (state.stream != nullptr) {
    FB_CUDACHECKTHROW(cudaStreamDestroy(state.stream));
  }
  state.ctranComm.reset(nullptr);
}

constexpr uint64_t kCommId{21};
constexpr int kCommHash{-1};
constexpr std::string_view kCommDesc{"ut_multirank_comm_desc"};
void initCtranComm(
    std::shared_ptr<ctran::testing::IntraProcessBootstrap::State>
        sharedBootstrapState,
    CtranComm* ctranComm,
    int nRanks,
    int rank,
    int cudaDev) {
  FB_CUDACHECKTHROW(cudaSetDevice(cudaDev));

  ctranComm->bootstrap_ =
      std::make_unique<ctran::testing::IntraProcessBootstrap>(
          sharedBootstrapState);

  ctranComm->logMetaData_.commId = kCommId;
  ctranComm->logMetaData_.commHash = kCommHash;
  ctranComm->logMetaData_.commDesc = std::string(kCommDesc);
  ctranComm->logMetaData_.rank = rank;
  ctranComm->logMetaData_.nRanks = nRanks;

  const int cudaArch = ctran::utils::getCudaArch(cudaDev).value_or(-1);
  const int64_t busId = ctran::utils::BusId::makeFrom(cudaDev).toInt64();

  std::vector<ncclx::RankTopology> rankTopologies{};
  std::vector<int> commRanksToWorldRanks{};
  ctranComm->statex_ = std::make_unique<ncclx::CommStateX>(
      rank,
      nRanks,
      cudaDev,
      cudaArch,
      busId,
      kCommHash,
      std::move(rankTopologies),
      std::move(commRanksToWorldRanks),
      std::string{kCommDesc});
  initRankStatesTopologyWrapper(
      ctranComm->statex_.get(), ctranComm->bootstrap_.get(), nRanks);

  FB_COMMCHECKTHROW(ctranInit(ctranComm));

  CLOGF(INFO, "UT MultiRank CTran initialized");
}

void workerRoutine(PerRankState& state) {
  // set dev first for correct logging
  ASSERT_EQ(cudaSuccess, cudaSetDevice(state.cudaDev));

  int rank = state.rank;
  SCOPE_EXIT {
    resetPerRankState(state);
  };
  CLOGF(
      INFO,
      "rank [{}/{}] worker started, cudaDev {}",
      rank,
      state.nRanks,
      state.cudaDev);

  initCtranComm(
      state.sharedBootstrapState,
      state.ctranComm.get(),
      state.nRanks,
      state.rank,
      state.cudaDev);
  FB_CUDACHECKTHROW(cudaStreamCreate(&state.stream));
  FB_COMMCHECKTHROW_EX(
      ctran::utils::commCudaMalloc(
          reinterpret_cast<char**>(&state.srcBuffer),
          CtranStandaloneMultiRankBaseTest::kBufferSize,
          &state.ctranComm->logMetaData_,
          "UT_workerRoutine"),
      rank,
      kCommHash);
  FB_COMMCHECKTHROW_EX(
      ctran::utils::commCudaMalloc(
          reinterpret_cast<char**>(&state.dstBuffer),
          CtranStandaloneMultiRankBaseTest::kBufferSize,
          &state.ctranComm->logMetaData_,
          "UT_workerRoutine"),
      rank,
      kCommHash);

  CLOGF(INFO, "rank [{}/{}] worker waiting for work", rank, state.nRanks);

  auto& sf = state.workSemiFuture;
  sf.wait();

  CLOGF(INFO, "rank [{}/{}] worker received work", rank, state.nRanks);

  auto work = sf.value();
  work(state);

  CLOGF(INFO, "rank [{}/{}] worker completed work", rank, state.nRanks);
}

} // namespace

void CtranStandaloneMultiRankBaseTest::SetUp() {
  setenv("NCCL_CTRAN_ENABLE", "INFO", 1);
  setenv("NCCL_DEBUG", "INFO", 1);
  // Ensure logger is initialized
  initOnce();
}

void CtranStandaloneMultiRankBaseTest::startWorkers(
    int nRanks,
    const std::vector<std::shared_ptr<::ctran::utils::Abort>>& aborts) {
  ASSERT_TRUE(aborts.size() == 0 || aborts.size() == nRanks)
      << "must supply either 0 or nRanks number of abort controls";

  // Create shared bootstrap state for all workers
  auto sharedBootstrapState =
      std::make_shared<testing::IntraProcessBootstrap::State>();

  // Reserve space to prevent reallocation that would invalidate references
  perRankStates_.reserve(nRanks);

  for (int i = 0; i < nRanks; ++i) {
    perRankStates_.emplace_back();
    auto& state = perRankStates_.back();
    state.sharedBootstrapState = sharedBootstrapState;
    state.ctranComm = std::make_unique<CtranComm>(
        aborts.size() == 0 ? ::ctran::utils::createAbort(/*enabled=*/false)
                           : folly::copy(aborts[i]));
    state.nRanks = nRanks;
    state.rank = i;
    state.cudaDev = i;
    workers_.emplace_back(&workerRoutine, std::ref(state));
  }
}

void CtranStandaloneMultiRankBaseTest::TearDown() {
  for (auto& worker : workers_) {
    worker.join();
  }
}

}; // namespace ctran::testing
