// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CtranDistTestUtils.h"

#include <folly/logging/xlog.h>

#include "comms/ctran/tests/bootstrap/CtranTestBootstrap.h"
#include "comms/ctran/utils/CudaUtils.h"
#include "comms/ctran/utils/Utils.h"
#include "comms/testinfra/DistEnvironmentBase.h"

namespace ctran {

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

std::unique_ptr<CtranComm> CtranDistTestFixture::makeCtranComm() {
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

  // Create global bootstrap (MPI or TcpStore depending on env)
  std::unique_ptr<meta::comms::IBootstrap> commBootstrap(
      meta::comms::createBootstrap("ctrancomm"));

  // Initialize topology
  if (NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::nolocal) {
    comm->statex_->initRankTopologyNolocal();
  } else if (NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::vnode) {
    comm->statex_->initRankTopologyVnode(
        NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS);
  } else {
    comm->statex_->initRankStatesTopology(commBootstrap.get());
  }

  comm->bootstrap_ = std::make_unique<ctran::testing::CtranTestBootstrap>(
      std::move(commBootstrap));

  comm->config_.commDesc = comm->statex_->commDesc().c_str();

  COMMCHECK_TEST(ctranInit(comm.get()));
  CHECK(ctranInitialized(comm.get())) << "Ctran not initialized";
  return comm;
}

} // namespace ctran
