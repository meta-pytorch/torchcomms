// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h> // @manual
#include <gtest/gtest.h>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_set>
#include <vector>

#include <folly/futures/Future.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/commstate/CommStateX.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/tests/bootstrap/IntraProcessBootstrap.h"
#include "comms/ctran/tests/bootstrap/MockBootstrap.h"
#include "comms/ctran/utils/Abort.h"
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

  FB_COMMCHECKTHROW_EX(ctranInit(ctranComm.get()), ctranComm->logMetaData_);

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
enum class InitEnvType { MPI, TCP_STORE, STANDALONE };

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

// ============================================================================
// Base Test Fixture Hierarchy
// ============================================================================
//
// CtranTestFixtureBase
//       |
//       +-- CtranStandaloneFixture (single-rank with MockBootstrap)
//       |
//       +-- CtranDistTestFixture (multi-rank distributed tests) [in
//       CtranDistTestUtils.h] |       - MPI mode: real multi-process with MPI
//       bootstrap |       - TCPStore mode: real multi-process with TCPStore
//       bootstrap
//       |
//       +-- CtranIntraProcessFixture (multi-rank simulation in single process)
//               - Uses threads + IntraProcessBootstrap
//               - Orchestrated work dispatch via run(rank, work)
//
// ============================================================================

// Base class with common utilities for all Ctran test fixtures.
// Provides environment setup (logger, cvars) and CUDA initialization.
class CtranTestFixtureBase : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  // Initialize environment variables, logger, and CUDA library.
  // This is called automatically by SetUp().
  void setupEnvironment();

  // CUDA device index (defaults to 0 for standalone, localRank for distributed)
  int cudaDev{0};

  // CUDA stream for tests (RAII managed)
  std::optional<meta::comms::CudaStream> stream{std::nullopt};
};

// Standalone mode fixture for single-rank testing with MockBootstrap.
// Use this for testing GPU kernels, GPE, mapper, etc. without multi-process
// coordination or the overhead of mpirun/TCPStore.
class CtranStandaloneFixture : public CtranTestFixtureBase {
 protected:
  static constexpr std::string_view kCommDesc{"ut_comm_desc"};

  void SetUp() override;
  void TearDown() override;

  // Create a CtranComm with MockBootstrap for single-rank testing.
  // @param abort: Optional abort control for fault tolerance testing.
  //               Defaults to enabled abort.
  std::unique_ptr<CtranComm> makeCtranComm(
      std::shared_ptr<::ctran::utils::Abort> abort =
          ctran::utils::createAbort(/*enabled=*/true));

  int rank{0}; // Always 0 for standalone tests
};

// Intra-process multi-rank fixture for testing with IntraProcessBootstrap.
// This allows testing multi-rank scenarios within a single process using
// threads, without requiring mpirun or external coordination.
//
// Use this fixture for:
// - Unit tests that need multi-rank semantics without real networking
// - Tests where you need to orchestrate different work to different ranks
// - Fast multi-rank tests without mpirun overhead
//
// For true distributed testing (multiple processes), use CtranDistTestFixture.
class CtranIntraProcessFixture : public CtranTestFixtureBase {
 public:
  static constexpr size_t kBufferSize = 128 * 1024;

  struct PerRankState;
  using Work = std::function<void(PerRankState&)>;

  struct PerRankState {
    // Ideally we could use the IBootstrap interface, but it makes UT debugging
    // hard since the barriers are not named. We use the specific
    // IntraProcessBootstrap class for namedBarriers.
    ::ctran::testing::IntraProcessBootstrap* getBootstrap() {
      return reinterpret_cast<::ctran::testing::IntraProcessBootstrap*>(
          ctranComm->bootstrap_.get());
    }

    std::shared_ptr<::ctran::testing::IntraProcessBootstrap::State>
        sharedBootstrapState;
    std::unique_ptr<CtranComm> ctranComm{nullptr};
    int nRanks{1};
    int rank{0};
    int cudaDev{0};
    cudaStream_t stream{nullptr};

    // device buffer for collectives
    void* srcBuffer{nullptr};
    void* dstBuffer{nullptr};

    folly::Promise<Work> workPromise;
    folly::SemiFuture<Work> workSemiFuture{workPromise.getSemiFuture()};
  };

 protected:
  std::vector<PerRankState> perRankStates_;
  std::vector<std::thread> workers_;

  void SetUp() override;

  void startWorkers(
      int nRanks,
      const std::vector<std::shared_ptr<::ctran::utils::Abort>>& aborts);

  void run(int rank, const Work& work) {
    perRankStates_[rank].workPromise.setValue(work);
  }

  void TearDown() override;
};

// ============================================================================
// Test Helpers (Utility class, not a fixture)
// ============================================================================
//
// CtranTestHelpers provides common verification and memory utilities for
// NCCL-integrated CTRAN tests. This is a utility class (not a test fixture)
// that can be composed into test fixtures or inherited as a mixin.
//
// Use this for:
// - Backend usage verification (verifyBackendsUsed)
// - GPE leak checking (verifyGpeLeak)
// - Device memory argument helpers (allocDevArg, checkDevArg, etc.)
// - Buffer preparation with different memory types (prepareBuf, releaseBuf)
//
// ============================================================================

class CtranTestHelpers {
 public:
  CtranTestHelpers();

  // Check no GPE internal memory leak after finished collective kernel
  void verifyGpeLeak(ICtran* ctran);

  // Reset backend usage counters
  void resetBackendsUsed(ICtran* ctran);

  // Verify the traffic at each backend is expected based on the memType and
  // statex topo.
  void verifyBackendsUsed(
      ICtran* ctran,
      const ncclx::CommStateX* statex,
      MemAllocType memType);

  // Verify backends used with excluded backends list
  void verifyBackendsUsed(
      ICtran* ctran,
      const ncclx::CommStateX* statex,
      MemAllocType memType,
      const std::vector<CtranMapperBackend>& excludedBackends);

  // Allocate device memory without initialization
  void allocDevArg(size_t nbytes, void*& ptr);

  // Allocate device memory and copy from vector
  template <typename T>
  void allocDevArg(const std::vector<T>& argVec, T*& ptr) {
    void* ptr_ = nullptr;
    size_t nbytes = sizeof(T) * argVec.size();
    allocDevArg(nbytes, ptr_);
    CUDACHECK_ASSERT(cudaMemcpy(
        ptr_, argVec.data(), argVec.size() * sizeof(T), cudaMemcpyDefault));
    ptr = reinterpret_cast<T*>(ptr_);
  }

  // Release all device arguments allocated by allocDevArg
  void releaseDevArgs();

  // Release a single device argument
  void releaseDevArg(void* ptr);

  // Asynchronously assign value to device argument from vector
  template <typename T>
  void assignDevArg(T* buf, const std::vector<T>& vals, cudaStream_t stream) {
    CUDACHECK_ASSERT(cudaMemcpyAsync(
        buf, vals.data(), vals.size() * sizeof(T), cudaMemcpyDefault, stream));
  }

  // Asynchronously fill device argument with single value
  template <typename T>
  void assignDevArg(T* buf, int count, T val, cudaStream_t stream) {
    std::vector<T> expVals(count, val);
    return assignDevArg(buf, expVals, stream);
  }

  // Check device argument values against expected vector
  template <typename T>
  void checkDevArg(
      const T* buf,
      const std::vector<T> expVals,
      std::vector<std::string>& errs,
      int maxNumErrs = 10) {
    const auto count = expVals.size();
    std::vector<T> obsVals(count, static_cast<T>(-1));
    CUDACHECK_ASSERT(
        cudaMemcpy(obsVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    errs.reserve(maxNumErrs);
    for (size_t i = 0; i < count; ++i) {
      if (obsVals[i] != expVals[i] &&
          errs.size() < static_cast<size_t>(maxNumErrs)) {
        errs.push_back(
            fmt::format(
                "observed[{}] = {}, expected = {}", i, obsVals[i], expVals[i]));
      }
    }
  }

  // Check device argument values against single expected value
  template <typename T>
  void checkDevArg(
      const T* buf,
      int count,
      T expVal,
      std::vector<std::string>& errs,
      int maxNumErrs = 10) {
    std::vector<T> expVals(count, expVal);
    return checkDevArg(buf, expVals, errs, maxNumErrs);
  }

  // Prepare buffer with kMemCudaMalloc memory type.
  // For NCCL memory types (kMemNcclMemAlloc, kCuMemAllocDisjoint),
  // use CtranNcclTestHelpers instead.
  void* prepareBuf(
      size_t bufSize,
      MemAllocType memType,
      std::vector<TestMemSegment>& segments);

  // Release buffer allocated by prepareBuf.
  // For NCCL memory types, use CtranNcclTestHelpers instead.
  void releaseBuf(void* buf, size_t bufSize, MemAllocType memType);

  // Align size to page boundary
  size_t pageAligned(size_t nBytes) const {
    return ((nBytes + pageSize_ - 1) / pageSize_) * pageSize_;
  }

 private:
  bool isBackendValid(
      const std::vector<CtranMapperBackend>& excludedBackends,
      CtranMapperBackend backend);

  size_t pageSize_{0};
  std::unordered_set<void*> devArgs_;
};

} // namespace ctran
