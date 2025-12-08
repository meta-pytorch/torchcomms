// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <optional>
#include "caffe2/torch/csrc/distributed/c10d/TCPStore.hpp"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/mccl/bootstrap/CtranAdapter.h"
#include "comms/mccl/utils/Utils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/commSpecs.h"

void logGpuMemoryStats(int gpu);

void commSetMyThreadLoggingName(std::string_view name);

commResult_t commMemAllocDisjoint(
    void** ptr,
    std::vector<size_t>& disjointSegmentSizes,
    std::vector<TestMemSegment>& segments,
    bool setRdmaSupport = true,
    std::optional<CUmemAllocationHandleType> handleType =
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);

commResult_t commMemFreeDisjoint(
    void* ptr,
    std::vector<size_t>& disjointSegmentSizes);

void* commMemAlloc(
    size_t bufSize,
    MemAllocType memType,
    std::vector<TestMemSegment>& segments);
void commMemFree(void* buf, size_t bufSize, MemAllocType memType);

class TestCtranCommRAII {
 public:
  TestCtranCommRAII(std::unique_ptr<CtranComm> ctranComm)
      : ctranComm(std::move(ctranComm)) {}
  std::unique_ptr<CtranComm> ctranComm{nullptr};
  std::shared_ptr<mccl::bootstrap::Bootstrap> bootstrap_{nullptr};

  ~TestCtranCommRAII() {
    if (ctranComm) {
      ctranComm.reset();
    }
  }
};

std::unique_ptr<TestCtranCommRAII> createDummyCtranComm(int devId = 0);

enum class InitEnvType { MPI, TCP_STORE };
inline InitEnvType getInitEnvType() {
  const char* localRankEnv = getenv("LOCAL_RANK");
  const char* globalRankEnv = getenv("GLOBAL_RANK");
  const char* worldSizeEnv = getenv("WORLD_SIZE");
  const char* localSizeEnv = getenv("LOCAL_SIZE");
  const char* masterAddrEnv = getenv("MASTER_ADDR");
  const char* masterPortEnv = getenv("MASTER_PORT");
  if (localRankEnv && globalRankEnv && worldSizeEnv && localSizeEnv &&
      masterAddrEnv && masterPortEnv) {
    return InitEnvType::TCP_STORE;
  } else {
    return InitEnvType::MPI;
  }
}

class CtranDistTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    if (getInitEnvType() == InitEnvType::MPI) {
      // initializing MPI
      MPICHECK_TEST(MPI_Init(nullptr, nullptr));
    }

    // Turn off NCCL debug logging by default, can be overridden by individual
    // tests or command line
    setenv("NCCL_DEBUG", "WARN", 0);
    // Disable FBWHOAMI Topology failure for tests
    setenv("NCCL_IGNORE_TOPO_LOAD_FAILURE", "0", 1);
    setenv("NCCL_CTRAN_PROFILING", "none", 1);
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1", 0);

#ifdef NCCL_COMM_STATE_DEBUG_TOPO_NOLOCAL
    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal", 1);
#endif
#ifdef NCCL_COMM_STATE_DEBUG_TOPO_VNODE
    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "vnode", 1);
#endif

// Allow each test to choose different fast init mode
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

  void TearDown() override {
    if (getInitEnvType() == InitEnvType::MPI) {
      MPICHECK_TEST(MPI_Finalize());
    }
  }
};

struct WorldInfo {
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int localSize{0};
};

inline WorldInfo getMpiWorldInfo() {
  int localRank, globalRank, numRanks, localSize = 0;

  MPICHECK_TEST(MPI_Comm_rank(MPI_COMM_WORLD, &globalRank));
  MPICHECK_TEST(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

  MPI_Comm localComm = MPI_COMM_NULL;
  MPI_Comm_split_type(
      MPI_COMM_WORLD,
      MPI_COMM_TYPE_SHARED,
      globalRank,
      MPI_INFO_NULL,
      &localComm);
  MPICHECK_TEST(MPI_Comm_rank(localComm, &localRank));
  MPICHECK_TEST(MPI_Comm_size(localComm, &localSize));
  MPICHECK_TEST(MPI_Comm_free(&localComm));

  return WorldInfo{
      .localRank = localRank,
      .globalRank = globalRank,
      .numRanks = numRanks,
      .localSize = localSize};
}

inline WorldInfo getTcpStoreWorldInfo() {
  const char* localRankEnv = getenv("LOCAL_RANK");
  const char* globalRankEnv = getenv("GLOBAL_RANK");
  const char* worldSizeEnv = getenv("WORLD_SIZE");
  const char* localSizeEnv = getenv("LOCAL_SIZE");
  return WorldInfo{
      .localRank = std::stoi(localRankEnv),
      .globalRank = std::stoi(globalRankEnv),
      .numRanks = std::stoi(worldSizeEnv),
      .localSize = std::stoi(localSizeEnv)};
}

static std::atomic<int> testCount = 0;
inline void incrTestCount() {
  testCount.fetch_add(1);
}

enum class TcpStorePhase { INIT, START, END };

inline std::string getTcpStoreKey(enum TcpStorePhase phase) {
  auto keyPrefix = std::string("commid_") + std::to_string(testCount);
  switch (phase) {
    case TcpStorePhase::INIT:
      return keyPrefix;
    case TcpStorePhase::START:
      return keyPrefix + "_start";
    case TcpStorePhase::END:
      return keyPrefix + "_end";
  }
}

inline std::unique_ptr<c10d::TCPStore> createTcpStore(bool isServer) {
  const char* masterAddrStr = getenv("MASTER_ADDR");
  const char* masterPortStr = getenv("MASTER_PORT");
  if (!masterAddrStr) {
    LOG(FATAL) << "MASTER_ADDR env variable is not set";
  }
  if (!masterPortStr) {
    LOG(FATAL) << "MASTER_PORT env variable is not set";
  }

  incrTestCount();
  auto key = fmt::format("test_tcpstore_init_{}", testCount);

  const std::string masterAddr(masterAddrStr);
  c10d::TCPStoreOptions opts;
  opts.port = std::stoi(masterPortStr);
  opts.waitWorkers = false;
  opts.useLibUV = true;
  opts.isServer = isServer;

  LOG(INFO) << "TCPStore "
            << (isServer ? "server starting on " : "client connecting to ")
            << masterAddr << ":" << opts.port << " ..." << " using key " << key;

  if (isServer) {
    auto server = std::make_unique<c10d::TCPStore>(masterAddr, opts);
    server->set(key, {1});
    LOG(INFO) << "TCPStore server started.";
    return server;
  }

  // TCPStore Client may start before fresh TCPStore Server has started
  // We need to retry until we connect to a fresh TCPStore Server
  while (true) {
    try {
      auto server = std::make_unique<c10d::TCPStore>(masterAddr, opts);
      if (server->check({key})) {
        LOG(INFO) << "TCPStore client started.";
        return server;
      }
    } catch (...) {
      LOG(INFO) << "Connected to stale TCPStore Server. "
                << "Waiting for fresh TCPStore Server to start.";
      std::this_thread::sleep_for(
          std::chrono::milliseconds{100}); // Sleep for 100ms
    }
  }
}

// Helper struct to hold bootstrap that needs to stay alive with the CtranComm
struct CtranCommWithBootstrap {
  std::shared_ptr<mccl::bootstrap::Bootstrap> bootstrap;
  std::unique_ptr<CtranComm> ctranComm;
};

inline CtranCommWithBootstrap createCtranCommWithBootstrap(
    int rank,
    int nRanks,
    uint64_t commId = 22,
    int commHash = -1,
    std::string_view commDesc = "ctran_comm_raii_comm_desc") {
  int cudaDev;
  CUDACHECK_TEST(cudaGetDevice(&cudaDev));

  COMMCHECK_TEST(ctran::utils::commCudaLibraryInit());

  std::unique_ptr<CtranComm> ctranComm = std::make_unique<CtranComm>(
      ::ctran::utils::createAbort(/*enabled=*/false));

  // Create and initialize bootstrap; needed for CTRAN backend initialization
  auto bootstrap = std::make_shared<mccl::bootstrap::Bootstrap>(
      NCCL_SOCKET_IFNAME,
      mccl::bootstrap::Options{
          .port = 0, .ifAddrPrefix = NCCL_SOCKET_IPADDR_PREFIX});

  const std::string selfUrl = bootstrap->semi_getInitUrl().get();
  std::vector<std::string> urls(nRanks);
  urls[rank] = selfUrl;

  // For single-rank case, just use our own URL
  // For multi-rank case, caller should use exchangeInitUrls to get all URLs
  if (nRanks == 1) {
    bootstrap->init(urls, rank, /*uuid=*/0);
  }

  ctranComm->bootstrap_ =
      std::make_unique<mccl::bootstrap::CtranAdapter>(bootstrap);

  ctranComm->logMetaData_.commId = commId;
  ctranComm->logMetaData_.commHash = commHash;
  ctranComm->logMetaData_.commDesc = std::string(commDesc);
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
      commHash,
      std::move(rankTopologies),
      std::move(commRanksToWorldRanks),
      std::string{commDesc});

  // For single-rank communicators (nRanks=1), use nolocal topology mode
  // which doesn't require bootstrap communication.
  if (nRanks == 1) {
    ctranComm->statex_->initRankTopologyNolocal();
  }

  FB_COMMCHECKTHROW(ctranInit(ctranComm.get()));

  return CtranCommWithBootstrap{
      .bootstrap = std::move(bootstrap),
      .ctranComm = std::move(ctranComm),
  };
}

class CtranDistTest : public ::testing::Test {
 public:
  std::unique_ptr<TestCtranCommRAII> commRAII;

  void SetUp() override {
    initWorldInfo();
    setenv("RANK", std::to_string(this->globalRank).c_str(), 1);
    CUDACHECK_TEST(cudaSetDevice(this->localRank));

    ncclCvarInit();
    ctran::logging::initCtranLogging(true /*alwaysInit*/);
    ctran::utils::commCudaLibraryInit();

#ifdef NCCL_COMM_STATE_DEBUG_TOPO_NOLOCAL
    enableNolocal = true;
#endif
    if (globalRank == 0) {
      LOG(INFO) << "Testing with NCCL_COMM_STATE_DEBUG_TOPO="
                << (enableNolocal ? "nolocal" : "default");
    }
    if (getInitEnvType() == InitEnvType::TCP_STORE) {
      tcpStore_ = createTcpStore(isTcpStoreServer());
    }

    LOG(INFO) << "Creating TestCtranCommRAII on rank " << globalRank;
    commRAII = createCtranCommRAII();
  }

  void TearDown() override {
    // trigger the destructor so that tests can verify the resources in
    // ctranComm is freed
    commRAII.reset();
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int localSize{0};
  bool enableNolocal{false};

 private:
  void initWorldInfo() {
    WorldInfo info;
    if (getInitEnvType() == InitEnvType::MPI) {
      info = getMpiWorldInfo();
    } else if (getInitEnvType() == InitEnvType::TCP_STORE) {
      info = getTcpStoreWorldInfo();
    } else {
      throw std::runtime_error("Unsupported init env type");
    }
    localRank = info.localRank;
    globalRank = info.globalRank;
    numRanks = info.numRanks;
    localSize = info.localSize;
  }

  bool isTcpStoreServer() {
    return globalRank == 0;
  }

  std::vector<std::string>
  exchangeInitUrls(const std::string& selfUrl, int numRanks, int selfRank) {
    std::vector<std::string> res(numRanks);
    if (getInitEnvType() == InitEnvType::TCP_STORE) {
      std::vector<std::string> rankKeys(numRanks);
      const auto keyUid = getTcpStoreKey(TcpStorePhase::INIT);
      for (int i = 0; i < numRanks; ++i) {
        rankKeys.at(i) = fmt::format("rank_{}_{}", i, keyUid);
      }
      const auto selfRankKey = fmt::format("rank_{}_{}", selfRank, keyUid);
      std::vector<uint8_t> urlBuf(selfUrl.begin(), selfUrl.end());
      tcpStore_->set(selfRankKey, urlBuf);

      // Wait for urls set by peer ranks
      tcpStore_->wait(rankKeys);
      if (tcpStore_->check(rankKeys)) {
        auto rankUrls = tcpStore_->multiGet(rankKeys);
        for (int i = 0; i < numRanks; ++i) {
          const auto& url = rankUrls.at(i);
          res[i] = std::string(url.begin(), url.end());
        }
      } else {
        LOG(FATAL) << "TCPStore key check returned false";
      }
    } else {
      // MPI all gather
      const size_t kMaxUrlLen = 512;
      std::vector<char> urls(kMaxUrlLen * numRanks);
      std::copy(
          selfUrl.begin(), selfUrl.end(), urls.begin() + kMaxUrlLen * selfRank);
      MPI_Allgather(
          urls.data() + kMaxUrlLen * selfRank,
          kMaxUrlLen,
          MPI_CHAR,
          urls.data(),
          kMaxUrlLen,
          MPI_CHAR,
          MPI_COMM_WORLD);
      for (int i = 0; i < numRanks; ++i) {
        const char* start = urls.data() + kMaxUrlLen * i;
        size_t len = strnlen(start, kMaxUrlLen);
        res[i] = std::string(start, len);
      }
    }
    return res;
  }

  static constexpr uint64_t kCommId{22};
  static constexpr int kCommHash{-1};
  static constexpr std::string_view kCommDesc{"ctran_comm_raii_comm_desc"};
  std::shared_ptr<mccl::bootstrap::Bootstrap> bootstrap_{nullptr};

  std::unique_ptr<TestCtranCommRAII> createCtranCommRAII() {
    int cudaDev;
    CUDACHECK_TEST(cudaGetDevice(&cudaDev));

    COMMCHECK_TEST(ctran::utils::commCudaLibraryInit());

    // For multi-rank cases, we need to set up bootstrap before creating the
    // CtranComm This is different from single-rank case where bootstrap can be
    // self-initialized
    if (numRanks > 1) {
      // Create bootstrap first
      bootstrap_ = std::make_shared<mccl::bootstrap::Bootstrap>(
          NCCL_SOCKET_IFNAME,
          mccl::bootstrap::Options{
              .port = 0, .ifAddrPrefix = NCCL_SOCKET_IPADDR_PREFIX});

      const std::string selfUrl = bootstrap_->semi_getInitUrl().get();
      LOG(INFO) << "Rank " << globalRank << " initURL: " << selfUrl;

      auto urls = exchangeInitUrls(selfUrl, numRanks, globalRank);
      bootstrap_->init(urls, globalRank, /*uuid=*/0);

      // Now create CtranComm with the initialized bootstrap
      std::unique_ptr<CtranComm> ctranComm = std::make_unique<CtranComm>(
          ::ctran::utils::createAbort(/*enabled=*/false));

      ctranComm->bootstrap_ =
          std::make_unique<mccl::bootstrap::CtranAdapter>(bootstrap_);

      ctranComm->logMetaData_.commId = kCommId;
      ctranComm->logMetaData_.commHash = kCommHash;
      ctranComm->logMetaData_.commDesc = std::string(kCommDesc);
      ctranComm->logMetaData_.rank = globalRank;
      ctranComm->logMetaData_.nRanks = numRanks;

      const int cudaArch = ctran::utils::getCudaArch(cudaDev).value_or(-1);
      const int64_t busId = ctran::utils::BusId::makeFrom(cudaDev).toInt64();

      std::vector<ncclx::RankTopology> rankTopologies{};
      std::vector<int> commRanksToWorldRanks{};
      ctranComm->statex_ = std::make_unique<ncclx::CommStateX>(
          globalRank,
          numRanks,
          cudaDev,
          cudaArch,
          busId,
          kCommHash,
          std::move(rankTopologies),
          std::move(commRanksToWorldRanks),
          std::string{kCommDesc});

      mccl::utils::initRankTopology(
          ctranComm->statex_.get(), ctranComm->bootstrap_.get());

      FB_COMMCHECKTHROW(ctranInit(ctranComm.get()));

      auto raii = std::make_unique<TestCtranCommRAII>(std::move(ctranComm));
      raii->bootstrap_ = bootstrap_;
      return raii;
    } else {
      // Single-rank case - use the helper function
      auto result = createCtranCommWithBootstrap(
          globalRank, numRanks, kCommId, kCommHash, kCommDesc);
      auto raii =
          std::make_unique<TestCtranCommRAII>(std::move(result.ctranComm));
      raii->bootstrap_ = std::move(result.bootstrap);
      return raii;
    }
  }

  std::unique_ptr<c10d::TCPStore> tcpStore_{nullptr};
};
