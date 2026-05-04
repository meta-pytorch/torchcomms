// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CtranDistTestUtils.h"

#include <folly/logging/xlog.h>

#include "comms/ctran/tests/bootstrap/CtranTestBootstrap.h"
#include "comms/ctran/utils/CudaUtils.h"
#include "comms/ctran/utils/Utils.h"
#include "comms/testinfra/DistEnvironmentBase.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"

namespace ctran {

// ============================================================================
// CtranDistEnvironment Implementation
// ============================================================================

void CtranDistEnvironment::SetUp() {
  meta::comms::DistEnvironmentBase::SetUp();

  // Ctran-specific env vars
  setenv("NCCL_CTRAN_PROFILING", "none", 1);
  setenv("NCCL_CTRAN_ENABLE", "1", 0);
  setenv("NCCL_COLLTRACE", "trace", 0);

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
  SetUp(CtranEnvs{});
}

void CtranDistTestFixture::SetUp(const CtranEnvs& envs) {
  // Save old env values and apply overrides.
  for (const auto& [key, value] : envs) {
    const char* oldVal = getenv(key.c_str());
    savedEnvs_[key] =
        oldVal ? std::optional<std::string>(oldVal) : std::nullopt;
    setenv(key.c_str(), value.c_str(), 1);
  }

  ncclCvarInit();
  distSetUp();

  cudaDev = localRank;

  CtranTestFixtureBase::SetUp();

  setenv("RANK", std::to_string(globalRank).c_str(), 1);

#ifdef NCCL_COMM_STATE_DEBUG_TOPO_NOLOCAL
  enableNolocal = true;
#endif

  if (globalRank == 0) {
    XLOG(DBG) << "Testing with NCCL_COMM_STATE_DEBUG_TOPO="
              << (isNolocalTopo() ? "nolocal" : "default");
  }

  stream.emplace(cudaStreamNonBlocking);
}

void CtranDistTestFixture::TearDown() {
  stream.reset();

  // Restore original env values.
  for (const auto& [key, value] : savedEnvs_) {
    if (value) {
      setenv(key.c_str(), value->c_str(), 1);
    } else {
      unsetenv(key.c_str());
    }
  }
  savedEnvs_.clear();

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

  // Initialize standalone colltrace with CommDumpPlugin so tests can verify
  // colltrace records without requiring a full ncclComm
  {
    meta::comms::colltrace::CollTraceConfig collTraceConfig;
    std::vector<std::unique_ptr<meta::comms::colltrace::ICollTracePlugin>>
        plugins;
    plugins.push_back(
        std::make_unique<meta::comms::colltrace::CommDumpPlugin>(
            meta::comms::colltrace::CommDumpConfig{.pastCollSize = 1024}));
    comm->colltraceNew_ = std::shared_ptr<meta::comms::colltrace::ICollTrace>(
        new meta::comms::colltrace::CollTrace(
            collTraceConfig,
            comm->logMetaData_,
            []() -> meta::comms::CommsMaybeVoid { return folly::unit; },
            std::move(plugins)));
  }

  return comm;
}

std::unordered_map<std::string, std::string> dumpCollTrace(CtranComm* comm) {
  using namespace meta::comms::colltrace;
  if (comm->colltraceNew_ == nullptr) {
    return {};
  }
  auto* plugin = comm->colltraceNew_->getPluginByName(
      std::string{CommDumpPlugin::kCommDumpPluginName});
  auto* commDumpPlugin = dynamic_cast<CommDumpPlugin*>(plugin);
  if (commDumpPlugin == nullptr) {
    return {};
  }
  auto dump = commDumpPlugin->dump();
  if (dump.hasError()) {
    return {};
  }
  return commDumpToMap(dump.value());
}

std::unordered_map<std::string, std::string> waitForCollTraceDrain(
    CtranComm* comm,
    int timeoutMs) {
  if (comm->colltraceNew_ == nullptr) {
    return {};
  }
  constexpr int kPollIntervalMs = 50;
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  std::unordered_map<std::string, std::string> dumpMap;
  while (std::chrono::steady_clock::now() < deadline) {
    dumpMap = dumpCollTrace(comm);
    auto it = dumpMap.find("CT_currentColls");
    if (it != dumpMap.end() && it->second == "[]") {
      return dumpMap;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
  }
  // Return whatever we have after timeout
  return dumpMap.empty() ? dumpCollTrace(comm) : dumpMap;
}

void CtranDistTestFixture::barrierNvlDomain(CtranComm* comm) {
  auto resFuture = comm->bootstrap_->barrierNvlDomain(
      comm->statex_->localRank(),
      comm->statex_->nLocalRanks(),
      comm->statex_->localRankToRanks());
  COMMCHECK_TEST(static_cast<commResult_t>(std::move(resFuture).get()));
}

} // namespace ctran
