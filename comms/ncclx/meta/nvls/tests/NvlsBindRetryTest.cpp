// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "comm.h"
#include "cudawrap.h"
#include "multicast.h"

namespace {

struct DriverState {
  std::vector<CUresult> mapResults{CUDA_SUCCESS};
  std::vector<std::vector<CUresult>> gatheredResults{
      {CUDA_SUCCESS, CUDA_SUCCESS}};
  ncclResult_t gatherFailure{ncclSuccess};
  std::string failOnce;
  std::vector<std::string> events;
  std::set<CUmemGenericAllocationHandle> handles;
  std::set<CUdeviceptr> reservations;
  std::set<CUdeviceptr> mappings;
  int creates{0};
  size_t maps{0};
  size_t gathers{0};

  CUresult cleanup(const char* operation) {
    events.emplace_back(operation);
    if (failOnce == operation) {
      failOnce.clear();
      return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
  }
};

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
thread_local DriverState* activeDriver = nullptr;

class NvlsBindRetryTest : public testing::Test {
 protected:
  void SetUp() override {
    activeDriver = &driver;
    comm.localRank = 0;
    comm.localRanks = 2;
    comm.localRankToRank = ranks;
    replace(
        pfn_cuGetErrorString, +[](CUresult, const char** text) {
          *text = "injected CUDA error";
          return CUDA_SUCCESS;
        });
    replace(
        pfn_cuMulticastGetGranularity,
        +[](size_t* granularity,
            const CUmulticastObjectProp*,
            CUmulticastGranularity_flags) {
          *granularity = 4096;
          return CUDA_SUCCESS;
        });
    replace(
        pfn_cuMulticastCreate,
        +[](CUmemGenericAllocationHandle* handle,
            const CUmulticastObjectProp*) {
          *handle = ++activeDriver->creates;
          activeDriver->handles.insert(*handle);
          activeDriver->events.emplace_back("create");
          return CUDA_SUCCESS;
        });
    replace(
        pfn_cuMulticastAddDevice,
        +[](CUmemGenericAllocationHandle, CUdevice) { return CUDA_SUCCESS; });
    replace(
        pfn_cuMemAddressReserve,
        +[](CUdeviceptr* base,
            size_t,
            size_t,
            CUdeviceptr,
            unsigned long long) {
          *base = 0x100000 * activeDriver->creates;
          activeDriver->reservations.insert(*base);
          return CUDA_SUCCESS;
        });
    replace(
        pfn_cuMemMap,
        +[](CUdeviceptr base,
            size_t,
            size_t,
            CUmemGenericAllocationHandle,
            unsigned long long) {
          auto& state = *activeDriver;
          state.events.emplace_back("map");
          const auto result = state.mapResults.at(state.maps++);
          if (result == CUDA_SUCCESS) {
            state.mappings.insert(base);
          }
          return result;
        });
    replace(
        pfn_cuMemUnmap, +[](CUdeviceptr base, size_t) {
          const auto result = activeDriver->cleanup("unmap");
          if (result == CUDA_SUCCESS) {
            EXPECT_EQ(activeDriver->mappings.erase(base), 1);
          }
          return result;
        });
    replace(
        pfn_cuMemAddressFree, +[](CUdeviceptr base, size_t) {
          const auto result = activeDriver->cleanup("free");
          if (result == CUDA_SUCCESS) {
            EXPECT_EQ(activeDriver->mappings.count(base), 0);
            EXPECT_EQ(activeDriver->reservations.erase(base), 1);
          }
          return result;
        });
    replace(
        pfn_cuMemRelease, +[](CUmemGenericAllocationHandle handle) {
          const auto result = activeDriver->cleanup("release");
          if (result == CUDA_SUCCESS) {
            EXPECT_EQ(activeDriver->handles.erase(handle), 1);
          }
          return result;
        });
    replace(
        pfn_cuMemSetAccess,
        +[](CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) {
          return CUDA_SUCCESS;
        });
    replace(ncclCuMemHandleType, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  }

  void TearDown() override {
    for (auto restore = restorers.rbegin(); restore != restorers.rend();
         ++restore) {
      (*restore)();
    }
    activeDriver = nullptr;
  }

  template <typename T>
  void replace(T& slot, T value) {
    const auto previous = slot;
    restorers.emplace_back([&slot, previous] { slot = previous; });
    slot = value;
  }

  ncclResult_t build() {
    const ncclMcRequest request{4096, 4096};
    return ncclMcGroupBuildPartitions(&comm, &request, 1, &group, &partition);
  }

  void expectReleased() {
    EXPECT_TRUE(driver.handles.empty());
    EXPECT_TRUE(driver.reservations.empty());
    EXPECT_TRUE(driver.mappings.empty());
  }

  DriverState driver;
  ncclComm comm{};
  int ranks[3]{0, 1, 2};
  ncclMcGroup* group{nullptr};
  ncclMcPartition partition{};
  std::vector<std::function<void()>> restorers;
};

} // namespace

// Wrap only bootstrap transport; the production allgather result selection,
// watchdog map, group rebuild and cleanup paths remain linked unchanged.
extern "C" ncclResult_t wrapAllGather(
    void*,
    int*,
    int rank,
    int nranks,
    void* data,
    int size) asm("__wrap__Z27bootstrapIntraNodeAllGatherPvPiiiS_i");
extern "C" ncclResult_t
wrapAllGather(void*, int*, int rank, int nranks, void* data, int size) {
  auto& state = *activeDriver;
  state.events.emplace_back("gather");
  if (state.gatherFailure != ncclSuccess) {
    return state.gatherFailure;
  }
  const auto& results = state.gatheredResults.at(state.gathers++);
  EXPECT_EQ(size, sizeof(CUresult));
  EXPECT_EQ(nranks, results.size());
  auto* allResults = static_cast<CUresult*>(data);
  EXPECT_EQ(allResults[rank], results.at(rank));
  std::copy(results.begin(), results.end(), allResults);
  return ncclSuccess;
}

extern "C" ncclResult_t wrapBarrier(void*, int*, int, int, int) asm(
    "__wrap__Z25bootstrapIntraNodeBarrierPvPiiii");
extern "C" ncclResult_t wrapBarrier(void*, int*, int, int, int) {
  activeDriver->events.emplace_back("barrier");
  return ncclSuccess;
}

extern "C" ncclResult_t
wrapBroadcast(void*, int*, int, int, int, void*, int) asm(
    "__wrap__Z27bootstrapIntraNodeBroadcastPvPiiiiS_i");
extern "C" ncclResult_t wrapBroadcast(void*, int*, int, int, int, void*, int) {
  return ncclSuccess;
}

namespace {

TEST_F(NvlsBindRetryTest, SuccessfulPeerUnmapsBeforeRebuildingGroup) {
  driver.mapResults = {CUDA_SUCCESS, CUDA_SUCCESS};
  driver.gatheredResults = {
      {CUDA_SUCCESS, CUDA_ERROR_SYSTEM_NOT_READY},
      {CUDA_SUCCESS, CUDA_SUCCESS},
      {CUDA_SUCCESS, CUDA_SUCCESS},
      {CUDA_SUCCESS, CUDA_SUCCESS}};

  ASSERT_EQ(build(), ncclSuccess);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(partition.mcHandle, 2);
  const std::vector<std::string> expected{
      "create",
      "barrier",
      "map",
      "gather",
      "unmap",
      "gather",
      "free",
      "release",
      "gather",
      "barrier",
      "create",
      "barrier",
      "map",
      "gather"};
  EXPECT_EQ(driver.events, expected);
  EXPECT_EQ(ncclMcGroupDestroy(&group), ncclSuccess);
  expectReleased();
}

TEST_F(NvlsBindRetryTest, FailedLocalMapRetriesWithoutUnmapping) {
  driver.mapResults = {CUDA_ERROR_SYSTEM_NOT_READY, CUDA_SUCCESS};
  driver.gatheredResults = {
      {CUDA_ERROR_SYSTEM_NOT_READY, CUDA_SUCCESS},
      {CUDA_SUCCESS, CUDA_SUCCESS},
      {CUDA_SUCCESS, CUDA_SUCCESS},
      {CUDA_SUCCESS, CUDA_SUCCESS}};

  ASSERT_EQ(build(), ncclSuccess);
  EXPECT_EQ(driver.creates, 2);
  EXPECT_EQ(std::count(driver.events.begin(), driver.events.end(), "unmap"), 0);
  EXPECT_EQ(ncclMcGroupDestroy(&group), ncclSuccess);
  expectReleased();
}

TEST_F(NvlsBindRetryTest, ExhaustedRetriesReleaseFinalGroup) {
  driver.mapResults = {
      CUDA_ERROR_SYSTEM_NOT_READY, CUDA_ERROR_SYSTEM_NOT_READY};
  driver.gatheredResults = {
      {CUDA_ERROR_SYSTEM_NOT_READY, CUDA_SUCCESS},
      {CUDA_SUCCESS, CUDA_SUCCESS},
      {CUDA_SUCCESS, CUDA_SUCCESS},
      {CUDA_ERROR_SYSTEM_NOT_READY, CUDA_SUCCESS},
      {CUDA_SUCCESS, CUDA_SUCCESS}};

  EXPECT_EQ(build(), ncclUnhandledCudaError);
  EXPECT_EQ(group, nullptr);
  EXPECT_EQ(driver.creates, 2);
  expectReleased();
}

TEST_F(NvlsBindRetryTest, FatalPeerErrorDominatesRetryableError) {
  comm.localRanks = 3;
  driver.gatheredResults = {
      {CUDA_SUCCESS, CUDA_ERROR_SYSTEM_NOT_READY, CUDA_ERROR_INVALID_VALUE},
      {CUDA_SUCCESS, CUDA_SUCCESS, CUDA_SUCCESS}};

  EXPECT_EQ(build(), ncclUnhandledCudaError);
  EXPECT_EQ(group, nullptr);
  EXPECT_EQ(driver.creates, 1);
  expectReleased();
}

TEST_F(NvlsBindRetryTest, AllGatherFailureUnmapsSuccessfulLocalMap) {
  driver.gatherFailure = ncclSystemError;

  EXPECT_EQ(build(), ncclSystemError);
  const std::vector<std::string> expected{
      "create", "barrier", "map", "gather", "unmap", "free", "release"};
  EXPECT_EQ(driver.events, expected);
  EXPECT_EQ(group, nullptr);
  expectReleased();
}

class NvlsRetryCleanupTest : public NvlsBindRetryTest,
                             public testing::WithParamInterface<const char*> {};

TEST_P(NvlsRetryCleanupTest, FailedTeardownStopsRebuildAndRetriesCleanup) {
  driver.failOnce = GetParam();
  driver.gatheredResults = {
      {CUDA_SUCCESS, CUDA_ERROR_SYSTEM_NOT_READY},
      {driver.failOnce == "unmap" ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS,
       CUDA_SUCCESS}};
  if (driver.failOnce != "unmap") {
    driver.gatheredResults.push_back({CUDA_ERROR_INVALID_VALUE, CUDA_SUCCESS});
  }

  EXPECT_EQ(build(), ncclUnhandledCudaError);
  EXPECT_EQ(group, nullptr);
  EXPECT_EQ(driver.creates, 1);
  EXPECT_EQ(
      std::count(driver.events.begin(), driver.events.end(), GetParam()), 2);
  EXPECT_EQ(
      std::count(driver.events.begin(), driver.events.end(), "barrier"), 1);
  EXPECT_EQ(driver.gathers, driver.gatheredResults.size());
  expectReleased();
}

INSTANTIATE_TEST_SUITE_P(
    CleanupOperations,
    NvlsRetryCleanupTest,
    testing::Values("unmap", "free", "release"));

TEST_F(NvlsBindRetryTest, PeerUnmapFailureStopsRetryBeforeGroupTeardown) {
  driver.mapResults = {CUDA_ERROR_SYSTEM_NOT_READY};
  driver.gatheredResults = {
      {CUDA_ERROR_SYSTEM_NOT_READY, CUDA_SUCCESS},
      {CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE}};

  EXPECT_EQ(build(), ncclUnhandledCudaError);
  const std::vector<std::string> expected{
      "create", "barrier", "map", "gather", "gather", "free", "release"};
  EXPECT_EQ(driver.events, expected);
  EXPECT_EQ(group, nullptr);
  EXPECT_EQ(driver.creates, 1);
  expectReleased();
}

TEST_F(NvlsBindRetryTest, PeerGroupTeardownFailureStopsRetryBeforeBarrier) {
  driver.gatheredResults = {
      {CUDA_SUCCESS, CUDA_ERROR_SYSTEM_NOT_READY},
      {CUDA_SUCCESS, CUDA_SUCCESS},
      {CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE}};

  EXPECT_EQ(build(), ncclUnhandledCudaError);
  const std::vector<std::string> expected{
      "create",
      "barrier",
      "map",
      "gather",
      "unmap",
      "gather",
      "free",
      "release",
      "gather"};
  EXPECT_EQ(driver.events, expected);
  EXPECT_EQ(group, nullptr);
  EXPECT_EQ(driver.creates, 1);
  expectReleased();
}

} // namespace
