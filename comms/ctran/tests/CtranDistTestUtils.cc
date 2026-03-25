// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CtranDistTestUtils.h"

#include <chrono>
#include <thread>

#include <folly/logging/xlog.h>

#include "comms/ctran/utils/CudaUtils.h"
#include "comms/ctran/utils/Utils.h"
#include "comms/mccl/bootstrap/Bootstrap.h"
#include "comms/mccl/bootstrap/CtranAdapter.h"
#include "comms/mccl/utils/Utils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

namespace ctran {

InitEnvType getInitEnvType() {
  if (checkTcpStoreEnv()) {
    return InitEnvType::TCP_STORE;
  }
  return InitEnvType::MPI;
}

// ============================================================================
// CtranDistEnvironment Implementation
// ============================================================================

void CtranDistEnvironment::SetUp() {
  meta::comms::DistEnvironmentBase::SetUp();

  // Ctran-specific env vars
  setenv("NCCL_CTRAN_PROFILING", "none", 1);
  setenv("NCCL_CTRAN_ENABLE", "1", 0);
  setenv("NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1", 0);

#ifdef NCCL_COMM_STATE_DEBUG_TOPO_NOLOCAL
  setenv("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal", 1);
#endif
#ifdef NCCL_COMM_STATE_DEBUG_TOPO_VNODE
  setenv("NCCL_COMM_STATE_DEBUG_TOPO", "vnode", 1);
#endif

#if defined(TEST_ENABLE_FASTINIT)
  setenv("NCCL_FASTINIT_MODE", "ring_hybrid", 1);
#else
  setenv("NCCL_FASTINIT_MODE", "none", 1);
#endif

#if defined(TEST_ENABLE_CTRAN)
  setenv("NCCL_CTRAN_ENABLE", "1", 1);
#endif

#if defined(TEST_ENABLE_LOCAL_REGISTER)
  setenv("NCCL_LOCAL_REGISTER", "1", 1);
#endif

#if defined(TEST_CUDA_GRAPH_MODE)
  setenv("NCCL_CTRAN_ALLOW_CUDA_GRAPH", "1", 1);
#endif
}

// ============================================================================
// CtranDistTestFixture Implementation
// ============================================================================

void CtranDistTestFixture::SetUp() {
  distSetUp();

  cudaDev = localRank;

  CtranTestFixtureBase::SetUp();

  setenv("RANK", std::to_string(globalRank).c_str(), 1);

#ifdef NCCL_COMM_STATE_DEBUG_TOPO_NOLOCAL
  enableNolocal = true;
#endif

  if (globalRank == 0) {
    XLOG(DBG) << "Testing with NCCL_COMM_STATE_DEBUG_TOPO="
              << (enableNolocal ? "nolocal" : "default");
  }

  stream.emplace(cudaStreamNonBlocking);
}

void CtranDistTestFixture::TearDown() {
  stream.reset();
  distTearDown();
}

std::vector<std::string> CtranDistTestFixture::exchangeInitUrls(
    const std::string& selfUrl,
    int numRanks,
    int selfRank) {
  constexpr size_t kMaxUrlLen = 256;
  std::vector<char> buf(numRanks * kMaxUrlLen, 0);

  CHECK(selfUrl.size() < kMaxUrlLen) << "URL too long for allGather buffer";
  std::memcpy(
      buf.data() + selfRank * kMaxUrlLen, selfUrl.data(), selfUrl.size());

  auto res = bootstrap_->allGather(buf.data(), kMaxUrlLen, selfRank, numRanks);
  CHECK_EQ(std::move(res).get(), 0) << "exchangeInitUrls allGather failed";

  std::vector<std::string> urls;
  urls.reserve(numRanks);
  for (int i = 0; i < numRanks; ++i) {
    urls.emplace_back(buf.data() + i * kMaxUrlLen);
  }
  return urls;
}

std::unique_ptr<CtranComm> CtranDistTestFixture::makeCtranComm() {
  const auto initType = getInitEnvType();
  const std::string uuid{"0"};
  uint64_t commHash =
      ctran::utils::getHash(uuid.data(), static_cast<int>(uuid.size()));
  std::string commDesc = fmt::format("CtranTestComm-{}", globalRank);

  auto comm =
      std::make_unique<CtranComm>(ctran::utils::createAbort(/*enabled=*/false));
  comm->logMetaData_.commId = 0;
  comm->logMetaData_.commHash = commHash;
  comm->logMetaData_.commDesc = commDesc;
  comm->logMetaData_.rank = globalRank;
  comm->logMetaData_.nRanks = numRanks;

  const auto useVirtualTopo =
      (NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::nolocal ||
       NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::vnode);

  int cudaDev;
  CUDACHECK_TEST(cudaGetDevice(&cudaDev));
  const int cudaArch = ctran::utils::getCudaArch(cudaDev).value_or(-1);
  const int64_t busId = ctran::utils::BusId::makeFrom(cudaDev).toInt64();

  std::vector<ncclx::RankTopology> rankTopologies{};
  std::vector<int> commRanksToWorldRanks{};
  comm->statex_ = std::make_unique<ncclx::CommStateX>(
      globalRank,
      numRanks,
      cudaDev,
      cudaArch,
      busId,
      commHash,
      rankTopologies,
      commRanksToWorldRanks,
      commDesc);

  if (initType == InitEnvType::MPI && useVirtualTopo) {
    mccl::utils::initRankTopologyNoSystem(comm->statex_.get());

    const auto localRank = comm->statex_->localRank();
    const auto node = comm->statex_->node();

    comm->bootstrap_ =
        std::make_unique<meta::comms::MpiBootstrap>(localRank, node);
  } else if (initType == InitEnvType::MPI) {
    comm->bootstrap_ = std::make_unique<meta::comms::MpiBootstrap>();
    mccl::utils::initRankTopology(comm->statex_.get(), comm->bootstrap_.get());
  } else {
    auto bootstrap = std::make_shared<mccl::bootstrap::Bootstrap>(
        NCCL_SOCKET_IFNAME,
        mccl::bootstrap::Options{
            .port = 0, .ifAddrPrefix = NCCL_SOCKET_IPADDR_PREFIX});

    std::string selfUrl = bootstrap->semi_getInitUrl().get();
    XLOG(DBG) << "Rank " << globalRank << " initURL: " << selfUrl;

    auto allUrls = exchangeInitUrls(selfUrl, numRanks, globalRank);

    std::vector<mccl::InitURL> urlVec(allUrls.begin(), allUrls.end());

    bootstrap->init(urlVec, static_cast<size_t>(globalRank), 0 /* uuid */);

    comm->bootstrap_ =
        std::make_unique<mccl::bootstrap::CtranAdapter>(bootstrap);
    mccl::utils::initRankTopology(comm->statex_.get(), comm->bootstrap_.get());
  }

  comm->config_.commDesc = comm->statex_->commDesc().c_str();

  COMMCHECK_TEST(ctranInit(comm.get()));
  CHECK(ctranInitialized(comm.get())) << "Ctran not initialized";
  return comm;
}

} // namespace ctran
