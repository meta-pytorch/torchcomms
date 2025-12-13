// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CtranTestUtils.h"

#include <atomic>
#include <chrono>
#include <thread>

#include <folly/logging/xlog.h>

#include "comms/ctran/utils/ErrorStackTraceUtil.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/ctran/utils/Utils.h"
#include "comms/mccl/utils/Utils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

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

static std::atomic<int> testCount = 0;
inline void incrTestCount() {
  testCount.fetch_add(1);
}

std::unique_ptr<c10d::TCPStore> createTcpStore(bool isServer) {
  const char* masterAddrStr = getenv("MASTER_ADDR");
  const char* masterPortStr = getenv("MASTER_PORT");
  if (!masterAddrStr) {
    XLOG(FATAL) << "MASTER_ADDR env variable is not set";
  }
  if (!masterPortStr) {
    XLOG(FATAL) << "MASTER_PORT env variable is not set";
  }

  incrTestCount();
  auto key = fmt::format("test_tcpstore_init_{}", testCount.load());

  const std::string masterAddr(masterAddrStr);
  c10d::TCPStoreOptions opts{
      .port = static_cast<uint16_t>(std::stoi(masterPortStr)),
      .waitWorkers = false,
      .useLibUV = true,
      .isServer = isServer,
  };

  XLOG(INFO) << "TCPStore "
             << (isServer ? "server starting on " : "client connecting to ")
             << masterAddr << ":" << opts.port << " ..." << " using key "
             << key;

  if (isServer) {
    auto server = std::make_unique<c10d::TCPStore>(masterAddr, opts);
    server->set(key, {1});
    XLOG(INFO) << "TCPStore server started.";
    return server;
  }

  // TCPStore Client may start before fresh TCPStore Server has started
  // We need to retry until we connect to a fresh TCPStore Server
  while (true) {
    try {
      auto server = std::make_unique<c10d::TCPStore>(masterAddr, opts);
      if (server->check({key})) {
        XLOG(INFO) << "TCPStore client started.";
        return server;
      }
    } catch (...) {
      XLOG(INFO) << "Connected to stale TCPStore Server. "
                 << "Waiting for fresh TCPStore Server to start.";
      std::this_thread::sleep_for(
          std::chrono::milliseconds{100}); // Sleep for 100ms
    }
  }
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
    std::optional<CUmemAllocationHandleType> handleType) {
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

  vaSize = 0;
  std::vector<size_t> alignedSizes(numSegments);
  for (int i = 0; i < numSegments; i++) {
    alignedSizes[i] = disjointSegmentSizes[i];
    ALIGN_SIZE(alignedSizes[i], memGran);
    vaSize += alignedSizes[i];
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
  // Now allow RW access to the newly mapped memory.
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  FB_CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, vaSize, &accessDesc, 1));

  return ret;
}

commResult_t commMemFreeDisjoint(
    void* ptr,
    std::vector<size_t>& disjointSegmentSizes) {
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

  size_t vaSize = 0;
  size_t numSegments = disjointSegmentSizes.size();
  std::vector<size_t> alignedSizes(numSegments);
  for (int i = 0; i < numSegments; i++) {
    alignedSizes[i] = disjointSegmentSizes[i];
    ALIGN_SIZE(alignedSizes[i], memGran);
    vaSize += alignedSizes[i];
  }

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

// Wrapper for memory allocation in tests with different memory types
// - bufSize: size of the buffer to allocate
// - memType: memory type to allocate
// - segments: vector of underlying allocated segments. It can be two segments
//             with kCuMemAllocDisjoint type, which map to a single virtual
//             memory range. For other mem types, it should be 1 segment.
// - return: pointer to the allocated virtual memory range.
void* commMemAlloc(
    size_t bufSize,
    MemAllocType memType,
    std::vector<TestMemSegment>& segments) {
  void* buf = nullptr;
  switch (memType) {
    case kMemCudaMalloc:
      CUDACHECK_TEST(cudaMalloc(&buf, bufSize));
      segments.emplace_back(buf, bufSize);
      break;
    case kCuMemAllocDisjoint: {
      // Allocate disjoint segments mapping to a single virtual memory range;
      // it mimics the behavior of Pytorch CCA expandable segment mode where a
      // single tensor may be mapped by two disjoint segments.
      const auto segSize = getSegmentSize(bufSize, 2);
      std::vector<size_t> disjointSegSizes(2, segSize);
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

void commMemFree(void* buf, size_t bufSize, MemAllocType memType) {
  switch (memType) {
    case kMemCudaMalloc:
      CUDACHECK_TEST(cudaFree(buf));
      break;
    case kCuMemAllocDisjoint: {
      const auto segSize = getSegmentSize(bufSize, 2);
      std::vector<size_t> disjointSegSizes(2, segSize);
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

InitEnvType getInitEnvType() {
  if (checkTcpStoreEnv()) {
    return InitEnvType::TCP_STORE;
  }
  return InitEnvType::MPI;
}

void CtranEnvironmentBase::SetUp() {
  const auto initType = getInitEnvType();
  if (initType == InitEnvType::MPI) {
    MPI_CHECK(MPI_Init(nullptr, nullptr));
  }
  // TCPStore doesn't need global initialization

  // Set up default envs for CTRAN tests
  // Default logging level = INFO
  // Individual test can override the logging level
  setenv("NCCL_DEBUG", "INFO", 0);

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

void CtranEnvironmentBase::TearDown() {
  const auto initType = getInitEnvType();
  if (initType == InitEnvType::MPI) {
    MPI_CHECK(MPI_Finalize());
  }
  // TCPStore doesn't need global cleanup
}

void CtranDistTestFixture::SetUp() {
  const auto initType = getInitEnvType();

  // Get rank info based on initialization type
  if (initType == InitEnvType::MPI) {
    setUpMpi();
  } else if (initType == InitEnvType::TCP_STORE) {
    setUpTcpStore();
  }

  // Initialize ctran settings
  setenv("RANK", std::to_string(globalRank).c_str(), 1);
  CUDACHECK_TEST(cudaSetDevice(localRank));
  ncclCvarInit();
  ctran::logging::initCtranLogging(true /*alwaysInit*/);
  COMMCHECK_TEST(ctran::utils::commCudaLibraryInit());

#ifdef NCCL_COMM_STATE_DEBUG_TOPO_NOLOCAL
  enableNolocal = true;
#endif

  if (globalRank == 0) {
    XLOG(INFO) << "Testing with NCCL_COMM_STATE_DEBUG_TOPO="
               << (enableNolocal ? "nolocal" : "default");
  }

  stream.emplace(cudaStreamNonBlocking); // Create RAII non-blocking CUDA stream
}

void CtranDistTestFixture::TearDown() {
  stream.reset(); // Reset the CUDA stream (RAII handles destruction)
  tcpStore_.reset();
}

void CtranDistTestFixture::setUpMpi() {
  // Get rank info via MPI
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &globalRank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

  MPI_Comm localComm{MPI_COMM_NULL};
  MPI_CHECK(MPI_Comm_split_type(
      MPI_COMM_WORLD,
      OMPI_COMM_TYPE_HOST,
      globalRank,
      MPI_INFO_NULL,
      &localComm));
  MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
  MPI_CHECK(MPI_Comm_size(localComm, &numLocalRanks_));
  MPI_CHECK(MPI_Comm_free(&localComm));
}

void CtranDistTestFixture::setUpTcpStore() {
  // Get rank info from environment variables
  localRank = std::stoi(getenv("LOCAL_RANK"));
  globalRank = std::stoi(getenv("GLOBAL_RANK"));
  numRanks = std::stoi(getenv("WORLD_SIZE"));
  numLocalRanks_ = std::stoi(getenv("LOCAL_SIZE"));

  tcpStore_ = createTcpStore(isTcpStoreServer()); // Initialize TCP Store
}

bool CtranDistTestFixture::isTcpStoreServer() const {
  return globalRank == 0;
}

std::vector<std::string> CtranDistTestFixture::exchangeInitUrls(
    const std::string& selfUrl,
    int numRanks,
    int selfRank) {
  const auto initType = getInitEnvType();
  CHECK(initType == InitEnvType::TCP_STORE);

  std::vector<std::string> res(numRanks);
  std::vector<std::string> rankKeys(numRanks);

  const auto testNum = testCount.load();
  const auto keyUid = fmt::format("commid_{}", testNum);

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
    XLOG(FATAL) << "TCPStore key check returned false";
  }

  return res;
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

  // Initialize StateX before bootstrap, so bootstran can honor DEBUG_TOPO set
  // by StateX
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

  // Use appropriate bootstrap based on init type
  if (initType == InitEnvType::MPI && useVirtualTopo) {
    // Explicitly initialize virtual topology which doesn't need bootstrap
    mccl::utils::initRankTopologyNoSystem(comm->statex_.get());

    // statex can be queried after topo initialization
    const auto localRank = comm->statex_->localRank();
    const auto node = comm->statex_->node();

    // Create bootstrap with virtual localRank and node for internal localComm
    comm->bootstrap_ =
        std::make_unique<meta::comms::MpiBootstrap>(localRank, node);
  } else if (initType == InitEnvType::MPI) {
    comm->bootstrap_ = std::make_unique<meta::comms::MpiBootstrap>();
    // Initialize StateX with topology using helper function
    mccl::utils::initRankTopology(comm->statex_.get(), comm->bootstrap_.get());
  } else {
    // For TCP Store, create and initialize mccl::bootstrap::Bootstrap
    // then wrap with CtranAdapter
    auto bootstrap = std::make_shared<mccl::bootstrap::Bootstrap>(
        NCCL_SOCKET_IFNAME,
        mccl::bootstrap::Options{
            .port = 0, .ifAddrPrefix = NCCL_SOCKET_IPADDR_PREFIX});

    // Get our own URL and exchange with all ranks
    std::string selfUrl = bootstrap->semi_getInitUrl().get();
    XLOG(INFO) << "Rank " << globalRank << " initURL: " << selfUrl;

    auto allUrls = exchangeInitUrls(selfUrl, numRanks, globalRank);

    // Convert to vector of InitURL for init() call
    std::vector<mccl::InitURL> urlVec(allUrls.begin(), allUrls.end());

    // Initialize the bootstrap with all URLs
    // void init(urls, myRank, uuid, abort, timeout)
    bootstrap->init(urlVec, static_cast<size_t>(globalRank), 0 /* uuid */);

    comm->bootstrap_ =
        std::make_unique<mccl::bootstrap::CtranAdapter>(bootstrap);
    // Initialize StateX with topology using helper function
    mccl::utils::initRankTopology(comm->statex_.get(), comm->bootstrap_.get());
  }

  // TODO: add memCache if enabled

  // Initialize Ctran
  comm->config_.commDesc = comm->statex_->commDesc().c_str();

  COMMCHECK_TEST(ctranInit(comm.get()));
  CHECK(ctranInitialized(comm.get())) << "Ctran not initialized";
  return comm;
}

} // namespace ctran
