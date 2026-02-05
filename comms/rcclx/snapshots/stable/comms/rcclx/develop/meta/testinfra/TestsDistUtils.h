// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gtest/gtest.h>
#include <ifaddrs.h>
#include "TestUtils.h"
#include "caffe2/torch/csrc/distributed/c10d/TCPStore.hpp"

#include "mpi.h"
#include "nccl.h" // @manual

#include "TestXPlatUtils.h"

#define MPICHECK_TEST(cmd)                                             \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

inline void initializeMpi() {
  // initializing MPI
  MPICHECK_TEST(MPI_Init(nullptr, nullptr));
}

#define CUDACHECKABORT(cmd)                               \
  do {                                                    \
    cudaError_t err = cmd;                                \
    if (err != cudaSuccess) {                             \
      WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
      abort();                                            \
    }                                                     \
  } while (false)

inline std::tuple<int, int, int> getMpiInfo() {
  int localRank, globalRank, numRanks = 0;

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
  MPICHECK_TEST(MPI_Comm_free(&localComm));

  return std::make_tuple(localRank, globalRank, numRanks);
}

inline std::tuple<int, int, int, int> getMpiInfoFull() {
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

  return std::make_tuple(localRank, globalRank, numRanks, localSize);
}

inline std::vector<std::string> getMpiHostNames() {
  int globalRank = 0;
  int numRanks = 0;

  MPICHECK_TEST(MPI_Comm_rank(MPI_COMM_WORLD, &globalRank));
  MPICHECK_TEST(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

  // all gather host names
  const size_t kMaxHostNameLen = 256;
  std::unique_ptr<char> hostNames(new char(kMaxHostNameLen * numRanks));
  gethostname(hostNames.get() + kMaxHostNameLen * globalRank, kMaxHostNameLen);

  MPI_Allgather(
      hostNames.get() + kMaxHostNameLen * globalRank,
      kMaxHostNameLen,
      MPI_CHAR,
      hostNames.get(),
      kMaxHostNameLen,
      MPI_CHAR,
      MPI_COMM_WORLD);

  std::vector<std::string> hostNamesVec{};
  for (int i = 0; i < numRanks; ++i) {
    hostNamesVec.push_back(std::string(hostNames.get() + kMaxHostNameLen * i));
  }
  return hostNamesVec;
}

inline void finalizeMpi() {
  // finalizing MPI
  MPICHECK_TEST(MPI_Finalize());
}

inline bool checkTcpStoreEnv() {
  // Check if LOCAL_RANK, GLOBAL_RANK, WORLD_SIZE, MASTER_ADDR and MASTER_PORT
  // environment variable is set
  const char* localRankEnv = getenv("LOCAL_RANK");
  const char* globalRankEnv = getenv("GLOBAL_RANK");
  const char* worldSizeEnv = getenv("WORLD_SIZE");
  const char* localSizeEnv = getenv("LOCAL_SIZE");
  const char* masterAddrEnv = getenv("MASTER_ADDR");
  const char* masterPortEnv = getenv("MASTER_PORT");
  return (
      localRankEnv && globalRankEnv && worldSizeEnv && localSizeEnv &&
      masterAddrEnv && masterPortEnv);
}

inline void initializeTcpStoreOrMpi() {
  if (!checkTcpStoreEnv()) {
    initializeMpi();
  }
}

inline std::tuple<int, int, int, int> getTcpStoreOrMpiInfo() {
  if (!checkTcpStoreEnv()) {
    return getMpiInfoFull();
  }

  const char* localRankEnv = getenv("LOCAL_RANK");
  const char* globalRankEnv = getenv("GLOBAL_RANK");
  const char* worldSizeEnv = getenv("WORLD_SIZE");
  const char* localSizeEnv = getenv("LOCAL_SIZE");
  return std::make_tuple(
      std::stoi(localRankEnv),
      std::stoi(globalRankEnv),
      std::stoi(worldSizeEnv),
      std::stoi(localSizeEnv));
}

inline void finalizeTcpStoreOrMpi() {
  if (!checkTcpStoreEnv()) {
    finalizeMpi();
  }
}

static std::atomic<int> testCount = 0;
inline void incrTestCount() {
  testCount.fetch_add(1);
}
inline std::string getTcpStoreKey(bool start) {
  return (start) ? std::string("commid_") + std::to_string(testCount)
                 : std::string("commid_finish_") + std::to_string(testCount);
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
  auto key = getTcpStoreKey(/*start=*/true);

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

inline void finalizeNcclComm(int globalRank, c10d::TCPStore* server) {
  if (server == nullptr) {
    return;
  }

  auto startKey = getTcpStoreKey(/*start=*/true);
  auto endKey = getTcpStoreKey(/*start=*/false);

  if (globalRank == 0) {
    LOG(INFO) << "TCPStore deleting key " << startKey;
    server->deleteKey(startKey);
    // Server will exit after setting endKey
    LOG(INFO) << "TCPStore setting " << endKey;
    server->set(endKey, {1});
  } else {
    // We need to wait till the server sets endKey and exits
    // We ignore "failed to recv, got 0 bytes" from TCPStore client if the
    // server has exited already
    try {
      LOG(INFO) << "TCPStore waiting for " << endKey;
      server->wait({endKey});
    } catch (...) {
      LOG(INFO) << "TCPStore " << endKey << " wait failed";
    }
  }
}

class DistEnvironmentBase : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeTcpStoreOrMpi();
    // Turn off NCCL debug logging by default, can be overridden by individual
    // tests or command line
    setenv("NCCL_DEBUG", "WARN", 0);
    // Disable FBWHOAMI Topology failure for tests
    setenv("NCCL_IGNORE_TOPO_LOAD_FAILURE", "1", 1);
  }
  void TearDown() override {
    finalizeTcpStoreOrMpi();
  }
};

/**
 * Very thin RAII wrapper for ncclComm_t. The class is designed to be used in
 * the same way ncclComm_t to be used, but handle ncclCommDestroy automatically
 * in the destruction of the object. Sample usage:
 * NcclCommRAII comm{globalRank, numRanks, devId} // Init
 * func(comm)       // Pass to other functions as ncclComm_t
 * comm->proxyState // Dereference as if it is ncclComm_t
 */

// Convenience function to create a NCCL comm with all global ranks
// For creating comm with subset of ranks, would need call the internal steps
// directly.
inline ncclComm_t createNcclComm(
    int globalRank,
    int numRanks,
    int devId,
    bool isMock = false,
    ncclConfig_t* customConfig = nullptr,
    c10d::TCPStore* server = nullptr) {
  ncclComm_t comm;
  ncclConfig_t config;
  if (customConfig) {
    config = *customConfig;
  } else {
    config = NCCL_CONFIG_INITIALIZER;
  }

  ncclUniqueId id;
  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (globalRank == 0) {
    NCCLCHECK_TEST(ncclGetUniqueId(&id));
  }

  std::vector<uint8_t> idStorage;
  if (server != nullptr && checkTcpStoreEnv()) {
    auto startKey = getTcpStoreKey(/*start=*/true);
    auto endKey = getTcpStoreKey(/*start=*/false);

    // use TCPStore to broadcast NCCL unique ID
    if (globalRank == 0) {
      server->deleteKey(endKey);
      idStorage.resize(sizeof(id));
      ::memcpy(idStorage.data(), &id, sizeof(id));
      server->set(startKey, idStorage);
    } else {
      // Wait for commid to be set by rank 0
      server->wait({startKey});
      if (server->check({startKey})) {
        idStorage.resize(sizeof(id));
        idStorage = server->get(startKey);
        ::memcpy(&id, idStorage.data(), sizeof(id));
      } else {
        LOG(INFO) << startKey.c_str() << " check returned false";
      }
    }
  } else {
    // use MPI to broadcast NCCL unique ID
    MPICHECK_TEST(
        MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  }

  CUDACHECK_TEST(cudaSetDevice(devId));

  // initializing NCCL
  auto res = ncclCommInitRankConfig(&comm, numRanks, id, globalRank, &config);
  if (res != ncclInProgress) {
    if (isMock) {
      NCCLCHECKTHROW_TEST(res);
    } else {
      NCCLCHECK_TEST(res);
    }
  }

  return comm;
}

class NcclCommRAII {
 public:
  NcclCommRAII(
      int globalRank,
      int numRanks,
      int localRank,
      bool isMock = false,
      ncclConfig_t* config = nullptr,
      c10d::TCPStore* server = nullptr)
      : globalRank_(globalRank), server_(server) {
    comm_ = createNcclComm(
        globalRank, numRanks, localRank, isMock, config, server_);
  }

  ncclComm& operator*() {
    return *comm_;
  }

  ncclComm_t operator->() const {
    return comm_;
  }

  operator ncclComm_t() {
    return comm_;
  }

  ~NcclCommRAII() {
    finalizeNcclComm(globalRank_, server_);
    NCCLCHECK_TEST(ncclCommDestroy(comm_));
  }

 private:
  ncclComm_t comm_;
  int globalRank_{0};
  c10d::TCPStore* server_{nullptr};
  bool destroyed_ = false;
};

// Similar to RcclxBaseTestFixture but requires init mode envs to be passed by
// compiler flag following legacy base test path. We will intergrate with
// RcclxBaseTestFixture later.
class RcclxBaseTest : public ::testing::Test {
 private:
  bool skipServerReset_ = false;

 public:
  ncclComm_t comm;
  RcclxBaseTest() = default;
  RcclxBaseTest(bool skipServerReset) : skipServerReset_(skipServerReset) {};

 protected:
  void SetUp() override {
    std::tie(
        this->localRank, this->globalRank, this->numRanks, this->localSize) =
        getTcpStoreOrMpiInfo();

    setenv("RANK", std::to_string(this->globalRank).c_str(), 1);
#ifdef NCCL_COMM_STATE_DEBUG_TOPO_NOLOCAL
    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal", 1);
    enableNolocal = true;
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

    CUDACHECKABORT(cudaSetDevice(this->localRank));

    if (initEnvAtSetup) {
      initEnv();
    }

    if (globalRank == 0) {
      LOG(INFO) << "Testing with NCCL_COMM_STATE_DEBUG_TOPO="
                << (enableNolocal ? "nolocal" : "default");
    }

    isServer = globalRank == 0;
    if (checkTcpStoreEnv()) {
      // launch TCPStore server if rank0 else launch client
      server = createTcpStore(isServer);
      return;
    }

    if (isServer) {
      // Skip TCP server setup if rank 0
      server = createTcpStore(true);
    }
  }

  void TearDown() override {
    // Child class requests to handle server reset
    if (!skipServerReset_ && server) {
      LOG(INFO) << "TCPStore " << (isServer ? "server" : "client")
                << " stopping ...";
      server.reset();
      LOG(INFO) << "TCPStore " << (isServer ? "server" : "client")
                << " stopped.";
    }
  }

  // void intraNodeBarrier(ncclComm_t comm) {
  //   NCCLCHECK_TEST(bootstrapIntraNodeBarrier(
  //       comm->bootstrap,
  //       comm->localRankToRank,
  //       comm->ctranComm_->statex_->localRank(),
  //       comm->ctranComm_->statex_->nLocalRanks(),
  //       comm->ctranComm_->statex_->localRankToRank(0)));
  // }

  void setInitEnv(bool init) {
    initEnvAtSetup = init;
  }

  template <typename T>
  void intraNodeAllGather(ncclComm_t comm, std::vector<T>& data) {
    NCCLCHECK_TEST(bootstrapAllGather(comm->bootstrap, data.data(), sizeof(T)));
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int localSize{0};
  std::unique_ptr<c10d::TCPStore> server{nullptr};
  bool enableNolocal{false};
  bool initEnvAtSetup{true};
  bool isServer{false};
};
