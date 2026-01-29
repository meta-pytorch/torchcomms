// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CtranTestUtils.h"

#include <atomic>
#include <chrono>
#include <thread>

#include <folly/logging/xlog.h>

#include "comms/ctran/tests/bootstrap/MockBootstrap.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/ErrorStackTraceUtil.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/ctran/utils/Utils.h"
#include "comms/mccl/utils/Utils.h"
#include "comms/utils/InitFolly.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/Logger.h"

namespace ctran {

std::unique_ptr<TestCtranCommRAII> createDummyCtranComm(int devId) {
  CUDACHECK_TEST(cudaSetDevice(devId));

  CHECK_EQ(ctran::utils::commCudaLibraryInit(), commSuccess);

  const std::string uuid{"0"};
  uint64_t commHash =
      ctran::utils::getHash(uuid.data(), static_cast<int>(uuid.size()));
  std::string commDesc = fmt::format("DummyCtranTestComm-{}", 0);

  auto result = createCtranCommWithBootstrap(0, 1, 0, commHash, commDesc);

  // Create a TestCtranCommRAII that also holds the bootstrap
  auto raii = std::make_unique<TestCtranCommRAII>(std::move(result.ctranComm));
  raii->bootstrap_ = std::move(result.bootstrap);
  return raii;
}

namespace {
size_t getSegmentSize(const size_t bufSize, const size_t numSegments) {
  // commMemAllocDisjoint internally would align the size to 2MB per segment
  // (queried from cuMemGetAllocationGranularity)
  return ctran::utils::align(bufSize, numSegments) / numSegments;
}
} // namespace

void logGpuMemoryStats(int gpu) {
  size_t free, total;
  CUDACHECK_TEST(cudaMemGetInfo(&free, &total));
  auto mbFree = static_cast<double>(free) / (1024 * 1024);
  auto mbTotal = static_cast<double>(total) / (1024 * 1024);
  LOG(INFO) << "GPU " << gpu << " memory: " << "freeBytes=" << free << " ("
            << mbFree << "MB), " << "totalBytes=" << total << "(" << mbTotal
            << "MB)";
}

void commSetMyThreadLoggingName(std::string_view name) {
  meta::comms::logger::initThreadMetaData(name);
}

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

commResult_t commMemAllocDisjoint(
    void** ptr,
    std::vector<size_t>& disjointSegmentSizes,
    std::vector<TestMemSegment>& segments,
    bool setRdmaSupport,
    std::optional<CUmemAllocationHandleType> handleType,
    size_t reservedVASize) {
  commResult_t ret = commSuccess;

  size_t numSegments = disjointSegmentSizes.size();
  size_t size = 0;
  for (int i = 0; i < numSegments; ++i) {
    size += disjointSegmentSizes[i];
  }
  size_t vaSize = 0;
  size_t memGran = 0;
  CUdeviceptr curPtr;
  CUdevice currentDev;
  CUmemAllocationProp memprop = {};
  CUmemAccessDesc accessDesc = {};
  std::vector<CUmemGenericAllocationHandle> handles(numSegments);
  std::vector<CUmemGenericAllocationHandle> unusedHandles(numSegments);
  int cudaDev;

  if (ptr == NULL || size == 0) {
    return ErrorStackTraceUtil::log(commInvalidArgument);
  }

  if (ctran::utils::commCudaLibraryInit() != commSuccess) {
    return ErrorStackTraceUtil::log(commSystemError);
  }

  // Still allow cumem based allocation if cumem is supported.
  if (!ctran::utils::getCuMemSysSupported()) {
    return ErrorStackTraceUtil::log(commSystemError);
  }
  CUDACHECK_TEST(cudaGetDevice(&cudaDev));
  FB_CUCHECK(cuDeviceGet(&currentDev, cudaDev));

  memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  if (handleType) {
    ctran::utils::setCuMemHandleTypeForProp(memprop, handleType.value());
  }
  memprop.location.id = currentDev;
  if (setRdmaSupport) {
    // Query device to see if RDMA support is available
    if (ctran::utils::gpuDirectRdmaWithCudaVmmSupported(currentDev, cudaDev)) {
      memprop.allocFlags.gpuDirectRDMACapable = 1;
    }
  }
  FB_CUCHECK(cuMemGetAllocationGranularity(
      &memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  // Calculate mapped size (sum of aligned segment sizes)
  size_t mappedSize = 0;
  std::vector<size_t> alignedSizes(numSegments);
  for (int i = 0; i < numSegments; i++) {
    alignedSizes[i] = disjointSegmentSizes[i];
    ALIGN_SIZE(alignedSizes[i], memGran);
    mappedSize += alignedSizes[i];
  }

  // Use reservedVASize if specified, otherwise use mappedSize
  vaSize = (reservedVASize > 0) ? reservedVASize : mappedSize;
  ALIGN_SIZE(vaSize, memGran);

  if (vaSize < mappedSize) {
    LOG(ERROR) << "reservedVASize " << reservedVASize
               << " is smaller than mapped size " << mappedSize;
    return ErrorStackTraceUtil::log(commInvalidArgument);
  }

  for (int i = 0; i < numSegments; i++) {
    /* Allocate the physical memory on the device */
    FB_CUCHECK(cuMemCreate(&handles[i], alignedSizes[i], &memprop, 0));
    FB_CUCHECK(cuMemCreate(&unusedHandles[i], alignedSizes[i], &memprop, 0));
  }
  // Free unused handles
  for (int i = 0; i < unusedHandles.size(); i++) {
    FB_CUCHECK(cuMemRelease(unusedHandles[i]));
  }
  /* Reserve a virtual address range */
  FB_CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, vaSize, memGran, 0, 0));
  /* Map the virtual address range to the physical allocation */
  curPtr = (CUdeviceptr)*ptr;
  for (int i = 0; i < numSegments; i++) {
    FB_CUCHECK(cuMemMap(curPtr, alignedSizes[i], 0, handles[i], 0));
    segments.emplace_back(reinterpret_cast<void*>(curPtr), alignedSizes[i]);
    LOG(INFO) << "ncclMemAllocDisjoint maps segments[" << i << "] ptr "
              << reinterpret_cast<void*>(curPtr) << " size " << alignedSizes[i]
              << "/" << vaSize;

    curPtr = ctran::utils::addDevicePtr(curPtr, alignedSizes[i]);
  }
  // Now allow RW access to the mapped memory only (not the entire reserved VA).
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  FB_CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, mappedSize, &accessDesc, 1));

  return ret;
}

commResult_t commMemFreeDisjoint(
    void* ptr,
    std::vector<size_t>& disjointSegmentSizes,
    size_t reservedVASize) {
  commResult_t ret = commSuccess;
  int saveDevice;
  CUmemGenericAllocationHandle handle;

  CUDACHECK_TEST(cudaGetDevice(&saveDevice));
  CUdevice ptrDev = 0;

  if (ptr == NULL) {
    cudaSetDevice(saveDevice);
    return ErrorStackTraceUtil::log(commInvalidArgument);
  }

  if (ctran::utils::commCudaLibraryInit() != commSuccess) {
    cudaSetDevice(saveDevice);
    return ErrorStackTraceUtil::log(commSystemError);
  }

  if (!ctran::utils::getCuMemSysSupported()) {
    cudaSetDevice(saveDevice);
    return ErrorStackTraceUtil::log(commSystemError);
  }

  FB_CUCHECK(cuPointerGetAttribute(
      (void*)&ptrDev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr));
  CUDACHECK_TEST(cudaSetDevice((int)ptrDev));

  size_t memGran = 0;
  CUmemAllocationProp memprop = {};
  FB_CUCHECK(cuMemGetAllocationGranularity(
      &memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  size_t mappedSize = 0;
  size_t numSegments = disjointSegmentSizes.size();
  std::vector<size_t> alignedSizes(numSegments);
  for (int i = 0; i < numSegments; i++) {
    alignedSizes[i] = disjointSegmentSizes[i];
    ALIGN_SIZE(alignedSizes[i], memGran);
    mappedSize += alignedSizes[i];
  }

  // Use reservedVASize if specified, otherwise use mappedSize
  size_t vaSize = (reservedVASize > 0) ? reservedVASize : mappedSize;
  ALIGN_SIZE(vaSize, memGran);

  CUdeviceptr curPtr = (CUdeviceptr)ptr;
  for (int i = 0; i < alignedSizes.size(); i++) {
    FB_CUCHECK(cuMemRetainAllocationHandle(&handle, (void*)curPtr));
    LOG(INFO) << "ncclMemFreeDisjoint unmaps segments[" << i << "] ptr "
              << reinterpret_cast<void*>(curPtr) << " size " << alignedSizes[i]
              << "/" << vaSize;
    FB_CUCHECK(cuMemRelease(handle));
    FB_CUCHECK(cuMemUnmap(curPtr, alignedSizes[i]));
    // call to cuMemRetainAllocationHandle increments reference count, requires
    // double cuMemRelease
    FB_CUCHECK(cuMemRelease(handle));
    curPtr = ctran::utils::addDevicePtr(curPtr, alignedSizes[i]);
  }
  FB_CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, vaSize));
  cudaSetDevice(saveDevice);
  return ret;
}

commResult_t commMemAllocExpandable(
    ExpandableTestBuffer* buf,
    size_t reservedSize,
    size_t initialMappedSize,
    bool setRdmaSupport) {
  buf->reservedSize = reservedSize;

  // Create initial segment sizes vector
  std::vector<size_t> initialSegments = {initialMappedSize};

  // Call commMemAllocDisjoint with larger reservedVASize
  FB_COMMCHECK(commMemAllocDisjoint(
      &buf->base,
      initialSegments,
      buf->segments,
      setRdmaSupport,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      reservedSize));

  buf->mappedSize = buf->segments[0].size;
  buf->segmentSize = buf->mappedSize;

  CUDACHECK_TEST(cudaGetDevice(&buf->cudaDev));
  CUdevice currentDev;
  FB_CUCHECK(cuDeviceGet(&currentDev, buf->cudaDev));

  buf->memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  buf->memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  buf->memprop.location.id = currentDev;
  buf->memprop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  if (setRdmaSupport &&
      ctran::utils::gpuDirectRdmaWithCudaVmmSupported(
          currentDev, buf->cudaDev)) {
    buf->memprop.allocFlags.gpuDirectRDMACapable = 1;
  }

  buf->accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  buf->accessDesc.location.id = currentDev;
  buf->accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  // Store initial handle via cuMemRetainAllocationHandle
  CUmemGenericAllocationHandle initialHandle;
  FB_CUCHECK(cuMemRetainAllocationHandle(&initialHandle, buf->base));
  buf->handles.push_back(initialHandle);

  return commSuccess;
}

commResult_t commMemExpandBuffer(
    ExpandableTestBuffer* buf,
    size_t newMappedSize) {
  if (newMappedSize <= buf->mappedSize) {
    LOG(ERROR) << "newMappedSize " << newMappedSize
               << " must be greater than current mappedSize "
               << buf->mappedSize;
    return commInvalidArgument;
  }
  if (newMappedSize > buf->reservedSize) {
    LOG(ERROR) << "newMappedSize " << newMappedSize << " exceeds reservedSize "
               << buf->reservedSize;
    return commInvalidArgument;
  }

  // Calculate number of new segments needed
  size_t additionalSize = newMappedSize - buf->mappedSize;
  size_t numNewSegments =
      (additionalSize + buf->segmentSize - 1) / buf->segmentSize;

  CUdeviceptr curPtr =
      ctran::utils::addDevicePtr((CUdeviceptr)buf->base, buf->mappedSize);

  for (size_t i = 0; i < numNewSegments; i++) {
    CUmemGenericAllocationHandle handle;
    FB_CUCHECK(cuMemCreate(&handle, buf->segmentSize, &buf->memprop, 0));
    FB_CUCHECK(cuMemMap(curPtr, buf->segmentSize, 0, handle, 0));

    buf->handles.push_back(handle);
    buf->segments.emplace_back(
        reinterpret_cast<void*>(curPtr), buf->segmentSize);

    LOG(INFO) << "commMemExpandBuffer maps new segment ptr "
              << reinterpret_cast<void*>(curPtr) << " size "
              << buf->segmentSize;

    curPtr = ctran::utils::addDevicePtr(curPtr, buf->segmentSize);
  }

  // Set access for new region
  size_t newRegionSize = numNewSegments * buf->segmentSize;
  FB_CUCHECK(cuMemSetAccess(
      ctran::utils::addDevicePtr((CUdeviceptr)buf->base, buf->mappedSize),
      newRegionSize,
      &buf->accessDesc,
      1));

  buf->mappedSize += newRegionSize;

  return commSuccess;
}

commResult_t commMemFreeExpandable(ExpandableTestBuffer* buf) {
  if (buf->base == nullptr) {
    return commSuccess;
  }

  // Unmap and release all segments
  CUdeviceptr curPtr = (CUdeviceptr)buf->base;
  for (size_t i = 0; i < buf->segments.size(); i++) {
    FB_CUCHECK(cuMemUnmap(curPtr, buf->segments[i].size));
    curPtr = ctran::utils::addDevicePtr(curPtr, buf->segments[i].size);
  }

  // Release all handles
  for (auto& handle : buf->handles) {
    FB_CUCHECK(cuMemRelease(handle));
  }

  // Free the reserved VA
  FB_CUCHECK(cuMemAddressFree((CUdeviceptr)buf->base, buf->reservedSize));

  *buf = ExpandableTestBuffer{};
  return commSuccess;
}

// Wrapper for memory allocation in tests with different memory types
// - bufSize: size of the buffer to allocate
// - memType: memory type to allocate
// - segments: vector of underlying allocated segments. It can be two segments
//             with kCuMemAllocDisjoint type, which map to a single virtual
//             memory range. For other mem types, it should be 1 segment.
// - numSegments: optional number of segments for kCuMemAllocDisjoint type
//                (default: 2). Ignored for other mem types.
// - return: pointer to the allocated virtual memory range.
void* commMemAlloc(
    size_t bufSize,
    MemAllocType memType,
    std::vector<TestMemSegment>& segments,
    size_t numSegments) {
  void* buf = nullptr;
  switch (memType) {
    case kMemCudaMalloc:
      CUDACHECK_TEST(cudaMalloc(&buf, bufSize));
      segments.emplace_back(buf, bufSize);
      break;
    case kCuMemAllocDisjoint: {
      // Allocate disjoint segments mapping to a single virtual memory range;
      // it mimics the behavior of Pytorch CCA expandable segment mode where a
      // single tensor may be mapped by multiple disjoint segments.
      const auto segSize = getSegmentSize(bufSize, numSegments);
      std::vector<size_t> disjointSegSizes(numSegments, segSize);
      COMMCHECK_TEST(commMemAllocDisjoint(&buf, disjointSegSizes, segments));
      break;
    }
    case kMemHostManaged:
      CUDACHECK_TEST(cudaMallocHost(&buf, bufSize));
      segments.emplace_back(buf, bufSize);
      break;
    case kMemCuMemAlloc: {
      std::vector<size_t> segSize(1, bufSize);
      COMMCHECK_TEST(commMemAllocDisjoint(&buf, segSize, segments));
      break;
    }
    case kMemHostUnregistered:
      // Allocate a host buffer using malloc (not CUDA-registered)
      buf = malloc(bufSize);
      CHECK(buf != nullptr);
      segments.emplace_back(buf, bufSize);
      break;
    default:
      XLOG(FATAL) << "Unsupported memType: " << memType;
      break;
  }
  return buf;
}

void commMemFree(
    void* buf,
    size_t bufSize,
    MemAllocType memType,
    size_t numSegments) {
  switch (memType) {
    case kMemCudaMalloc:
      CUDACHECK_TEST(cudaFree(buf));
      break;
    case kCuMemAllocDisjoint: {
      const auto segSize = getSegmentSize(bufSize, numSegments);
      std::vector<size_t> disjointSegSizes(numSegments, segSize);
      commMemFreeDisjoint(buf, disjointSegSizes);
      break;
    }
    case kMemHostManaged:
      cudaFreeHost(buf);
      break;
    case kMemCuMemAlloc: {
      std::vector<size_t> segSize(1, bufSize);
      commMemFreeDisjoint(buf, segSize);
      break;
    }
    case kMemHostUnregistered:
      free(buf);
      break;
    default:
      XLOG(FATAL) << "Unsupported memType: " << memType;
      break;
  }
}

// ============================================================================
// CtranTestFixtureBase Implementation
// ============================================================================

void CtranTestFixtureBase::SetUp() {
  setupEnvironment();
}

void CtranTestFixtureBase::TearDown() {
  stream.reset();
}

void CtranTestFixtureBase::setupEnvironment() {
  setenv("NCCL_CTRAN_ENABLE", "1", 1);
  setenv("NCCL_DEBUG", "INFO", 1);

  FB_CUDACHECKIGNORE(cudaSetDevice(cudaDev));

  // Ensure logger and libraries are initialized (uses call_once internally)
  static folly::once_flag once;
  folly::call_once(once, [] {
    meta::comms::initFolly();
    ncclCvarInit();
    ctran::utils::commCudaLibraryInit();
    ctran::logging::initCtranLogging(true /*alwaysInit*/);
  });
}

// ============================================================================
// CtranStandaloneFixture Implementation
// ============================================================================

void CtranStandaloneFixture::SetUp() {
  rank = 0;
  cudaDev = 0;
  CtranTestFixtureBase::SetUp();
}

void CtranStandaloneFixture::TearDown() {
  CtranTestFixtureBase::TearDown();
}

std::unique_ptr<CtranComm> CtranStandaloneFixture::makeCtranComm(
    std::shared_ptr<::ctran::utils::Abort> abort) {
  auto ctranComm = std::make_unique<CtranComm>(abort);

  ctranComm->bootstrap_ = std::make_unique<testing::MockBootstrap>();
  static_cast<testing::MockBootstrap*>(ctranComm->bootstrap_.get())
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
      /*commDesc=*/std::string(kCommDesc));

  EXPECT_EQ(ctranInit(ctranComm.get()), commSuccess);

  CLOGF(INFO, "UT CTran initialized");

  return ctranComm;
}

// ============================================================================
// CtranIntraProcessFixture Implementation
// ============================================================================

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

using PerRankState = CtranIntraProcessFixture::PerRankState;
static void resetPerRankState(PerRankState& state) {
  if (state.dstBuffer != nullptr) {
    FB_COMMCHECKTHROW_EX_NOCOMM(ctran::utils::commCudaFree(state.dstBuffer));
  }
  if (state.srcBuffer != nullptr) {
    FB_COMMCHECKTHROW_EX_NOCOMM(ctran::utils::commCudaFree(state.srcBuffer));
  }
  if (state.stream != nullptr) {
    FB_CUDACHECKTHROW_EX(
        cudaStreamDestroy(state.stream), state.ctranComm->logMetaData_);
  }
  state.ctranComm.reset(nullptr);
}

constexpr uint64_t kMultiRankCommId{21};
constexpr int kMultiRankCommHash{-1};
constexpr std::string_view kMultiRankCommDesc{"ut_multirank_comm_desc"};

void initCtranCommMultiRank(
    std::shared_ptr<ctran::testing::IntraProcessBootstrap::State>
        sharedBootstrapState,
    CtranComm* ctranComm,
    int nRanks,
    int rank,
    int cudaDev) {
  FB_CUDACHECKTHROW_EX(
      cudaSetDevice(cudaDev),
      rank,
      kMultiRankCommHash,
      std::string(kMultiRankCommDesc));

  ctranComm->bootstrap_ =
      std::make_unique<ctran::testing::IntraProcessBootstrap>(
          sharedBootstrapState);

  ctranComm->logMetaData_.commId = kMultiRankCommId;
  ctranComm->logMetaData_.commHash = kMultiRankCommHash;
  ctranComm->logMetaData_.commDesc = std::string(kMultiRankCommDesc);
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
      kMultiRankCommHash,
      std::move(rankTopologies),
      std::move(commRanksToWorldRanks),
      std::string{kMultiRankCommDesc});
  initRankStatesTopologyWrapper(
      ctranComm->statex_.get(), ctranComm->bootstrap_.get(), nRanks);

  FB_COMMCHECKTHROW_EX_NOCOMM(ctranInit(ctranComm));

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

  initCtranCommMultiRank(
      state.sharedBootstrapState,
      state.ctranComm.get(),
      state.nRanks,
      state.rank,
      state.cudaDev);
  FB_CUDACHECKTHROW_EX(
      cudaStreamCreate(&state.stream), state.ctranComm->logMetaData_);
  FB_COMMCHECKTHROW_EX(
      ctran::utils::commCudaMalloc(
          reinterpret_cast<char**>(&state.srcBuffer),
          CtranIntraProcessFixture::kBufferSize,
          &state.ctranComm->logMetaData_,
          "UT_workerRoutine"),
      state.ctranComm->logMetaData_);
  FB_COMMCHECKTHROW_EX(
      ctran::utils::commCudaMalloc(
          reinterpret_cast<char**>(&state.dstBuffer),
          CtranIntraProcessFixture::kBufferSize,
          &state.ctranComm->logMetaData_,
          "UT_workerRoutine"),
      state.ctranComm->logMetaData_);

  CLOGF(INFO, "rank [{}/{}] worker waiting for work", rank, state.nRanks);

  auto& sf = state.workSemiFuture;
  sf.wait();

  CLOGF(INFO, "rank [{}/{}] worker received work", rank, state.nRanks);

  auto work = sf.value();
  work(state);

  CLOGF(INFO, "rank [{}/{}] worker completed work", rank, state.nRanks);
}

} // namespace

void CtranIntraProcessFixture::SetUp() {
  // Call base class setup which handles environment variables,
  // CUDA library init, and logging initialization
  CtranTestFixtureBase::SetUp();
}

void CtranIntraProcessFixture::startWorkers(
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

void CtranIntraProcessFixture::TearDown() {
  for (auto& worker : workers_) {
    worker.join();
  }
}

} // namespace ctran

// ============================================================================
// CtranTestHelpers Implementation
// ============================================================================

namespace ctran {

CtranTestHelpers::CtranTestHelpers() {
  pageSize_ = getpagesize();
}

bool CtranTestHelpers::isBackendValid(
    const std::vector<CtranMapperBackend>& excludedBackends,
    CtranMapperBackend backend) {
  return std::find(excludedBackends.begin(), excludedBackends.end(), backend) ==
      excludedBackends.end();
}

void CtranTestHelpers::verifyGpeLeak(ICtran* ctran) {
  ASSERT_EQ(ctran->gpe->numInUseKernelElems(), 0);
  ASSERT_EQ(ctran->gpe->numInUseKernelFlags(), 0);
}

void CtranTestHelpers::resetBackendsUsed(ICtran* ctran) {
  ctran->mapper->iPutCount[CtranMapperBackend::NVL] = 0;
  ctran->mapper->iPutCount[CtranMapperBackend::IB] = 0;
}

void CtranTestHelpers::verifyBackendsUsed(
    ICtran* ctran,
    const ncclx::CommStateX* statex,
    MemAllocType memType) {
  verifyBackendsUsed(ctran, statex, memType, {});
}

void CtranTestHelpers::verifyBackendsUsed(
    ICtran* ctran,
    const ncclx::CommStateX* statex,
    MemAllocType memType,
    const std::vector<CtranMapperBackend>& excludedBackends) {
  const int nRanks = statex->nRanks();
  const int nLocalRanks = statex->nLocalRanks();

  switch (memType) {
    case kMemNcclMemAlloc:
    case kCuMemAllocDisjoint:
      // Expect usage from NVL backend unless excluded by particular
      // collective
      if (nLocalRanks > 1 &&
          isBackendValid(excludedBackends, CtranMapperBackend::NVL)) {
        if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE) {
          ASSERT_GT(ctran->mapper->iCopyCount, 0);
        } else {
          ASSERT_GT(ctran->mapper->iPutCount[CtranMapperBackend::NVL], 0);
        }
      } else {
        ASSERT_EQ(ctran->mapper->iPutCount[CtranMapperBackend::NVL], 0);
      }

      // Expect usage from IB backend unless excluded by particular collective
      if (nRanks > nLocalRanks &&
          isBackendValid(excludedBackends, CtranMapperBackend::IB)) {
        ASSERT_GT(ctran->mapper->iPutCount[CtranMapperBackend::IB], 0);
      }
      // Do not assume no IB usage, because IB backend may be used also for
      // local ranks if NVL backend is not available
      break;

    case kMemCudaMalloc:
      // memType is kMemCudaMalloc
      // Expect usage from IB backend as long as nRanks > 1, unless excluded
      // by particular collective
      if (nRanks > 1 &&
          isBackendValid(excludedBackends, CtranMapperBackend::IB)) {
        ASSERT_GT(ctran->mapper->iPutCount[CtranMapperBackend::IB], 0);
      }
      // Do not assume no IB usage, because IB backend may be used also for
      // local ranks if NVL backend is not available
      break;

    default:
      ASSERT_TRUE(false) << "Unsupported memType " << memType;
  }
}

void CtranTestHelpers::allocDevArg(size_t nbytes, void*& ptr) {
  CUDACHECK_ASSERT(cudaMalloc(&ptr, nbytes));
  devArgs_.insert(ptr);
}

void CtranTestHelpers::releaseDevArgs() {
  for (auto ptr : devArgs_) {
    CUDACHECK_TEST(cudaFree(ptr));
  }
  devArgs_.clear();
}

void CtranTestHelpers::releaseDevArg(void* ptr) {
  cudaFree(ptr);
  devArgs_.erase(ptr);
}

void* CtranTestHelpers::prepareBuf(
    size_t bufSize,
    MemAllocType memType,
    std::vector<TestMemSegment>& segments) {
  void* buf = nullptr;
  if (memType == kMemCudaMalloc) {
    CUDACHECK_TEST(cudaMalloc(&buf, bufSize));
    segments.emplace_back(buf, bufSize);
  } else {
    XLOG(FATAL)
        << "CtranTestHelpers only supports kMemCudaMalloc. "
        << "Use CtranNcclTestHelpers for kMemNcclMemAlloc or kCuMemAllocDisjoint.";
  }
  return buf;
}

void CtranTestHelpers::releaseBuf(
    void* buf,
    size_t bufSize,
    MemAllocType memType) {
  if (memType == kMemCudaMalloc) {
    CUDACHECK_TEST(cudaFree(buf));
  } else {
    XLOG(FATAL)
        << "CtranTestHelpers only supports kMemCudaMalloc. "
        << "Use CtranNcclTestHelpers for kMemNcclMemAlloc or kCuMemAllocDisjoint.";
  }
}

} // namespace ctran
