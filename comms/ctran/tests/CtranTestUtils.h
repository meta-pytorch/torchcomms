// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h> // @manual
#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <optional>
#include <vector>
#include "caffe2/torch/csrc/distributed/c10d/TCPStore.hpp"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/mccl/bootstrap/Bootstrap.h"
#include "comms/mccl/bootstrap/CtranAdapter.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"
#include "comms/utils/commSpecs.h"

namespace ctran {

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

void logGpuMemoryStats(int gpu);

void commSetMyThreadLoggingName(std::string_view name);

// Template function to get commDataType_t from C++ type.
template <typename T>
inline consteval commDataType_t getCommDataType() {
  if constexpr (std::is_same_v<T, int8_t>) {
    return commInt8;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return commInt32;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return commInt64;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return commUint8;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return commUint32;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return commUint64;
  } else if constexpr (std::is_same_v<T, float>) {
    return commFloat32;
  } else if constexpr (std::is_same_v<T, double>) {
    return commFloat64;
  } else if constexpr (std::is_same_v<T, __half>) {
    return commFloat16;
#if defined(__CUDA_BF16_TYPES_EXIST__)
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return commBfloat16;
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
  } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    return commFloat8e4m3;
  } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
    return commFloat8e5m2;
#endif
  } else {
    return commFloat32;
  }
}

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

// Bootstrap initialization type
enum class InitEnvType { MPI, TCP_STORE };

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

std::unique_ptr<c10d::TCPStore> createTcpStore(bool isServer);

// Detect which initialization environment to use
InitEnvType getInitEnvType();

class CtranEnvironmentBase : public ::testing::Environment {
 public:
  void SetUp() override;
  void TearDown() override;
};

// CtranDistTestFixture is a fixture for testing Ctran with multiple
// processes/ranks that supports both MPI and TCPStore bootstrap methods.
class CtranDistTestFixture : public ::testing::Test {
 public:
 protected:
  void SetUp() override;
  void TearDown() override;

  std::unique_ptr<CtranComm> makeCtranComm();

  // Rank information
  int globalRank{-1};
  int numRanks{-1};
  int localRank{-1};
  int numLocalRanks_{-1};
  bool enableNolocal{false};

  // CUDA stream for tests (RAII managed)
  std::optional<meta::comms::CudaStream> stream{std::nullopt};

 private:
  void setUpMpi();
  void setUpTcpStore();

  // TCP Store support
  std::unique_ptr<c10d::TCPStore> tcpStore_{nullptr};
  bool isTcpStoreServer() const;
  std::vector<std::string>
  exchangeInitUrls(const std::string& selfUrl, int numRanks, int selfRank);

  // Test counter for TCP Store key generation
  static std::atomic<int> testCount_;
};

} // namespace ctran
