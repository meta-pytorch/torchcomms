// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/memory/memCacheAllocator.h"

#include <fmt/format.h>
#include <folly/Singleton.h>

#include "comms/ctran/memory/Utils.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/DevUtils.cuh"

#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ncclx::memory {

static folly::Singleton<memCacheAllocator> memCacheAllocatorSingleton;

std::shared_ptr<memCacheAllocator> memCacheAllocator::getInstance() {
  ctran::utils::commCudaLibraryInit();
  if (!ctran::utils::getCuMemSysSupported()) {
    CLOGF(
        ERR,
        "NCCLX memory cache allocator only works with low-level cuMem APIs. Make sure CUDA Toolkit is 11.3 or higher.");
    return nullptr;
  }
  auto obj = memCacheAllocatorSingleton.try_get();
  if (!obj) {
    throw std::runtime_error("Failed to get memCacheAllocator singleton");
  }
  obj->init();
  return obj;
}

commResult_t memCacheAllocator::init() {
  std::unique_lock lock(mutex_);
  if (!intialized_) {
    slabAllocator_ = std::make_unique<SlabAllocator>();
    freeRegionMaps_[BucketType::DEFAULT] = memRegionMap_t();
    freeRegionMaps_[BucketType::NVL] = memRegionMap_t();
    freeRegionMaps_[BucketType::RDMA] = memRegionMap_t();
    if (NCCL_MEM_POOL_SIZE > 0) {
      // preallocate the memory pool
      size_t newSlabSize = NCCL_MEM_POOL_SIZE;
      FB_COMMCHECKTHROW_EX_NOCOMM(slabAllocator_->cuMalloc(
          (void**)&poolPtr_,
          NCCL_MEM_POOL_SIZE,
          "memCacheAllocator::init",
          nullptr,
          &poolHandle_,
          &newSlabSize));
      CLOGF_SUBSYS(
          INFO,
          ALLOC,
          "ncclx::memory::memCacheAllocator::init size {} pointer {} handle {:x}",
          newSlabSize,
          poolPtr_,
          ctran::utils::toFormattableHandle(poolHandle_));
      poolRemainSize_ = newSlabSize;
    }
    intialized_ = true;
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "Initialized NCCLX intenal buffer pool, size {}",
        NCCL_MEM_POOL_SIZE);
  }

  return commSuccess;
}

commResult_t memCacheAllocator::reset() {
  std::unique_lock lock(mutex_);
  if (intialized_) {
    CLOGF_SUBSYS(INFO, INIT, "Reset NCCLX intenal buffer pool");
    printSnapshot();

    auto nOccupiedRegions = countOccupiedRegions();
    if (nOccupiedRegions > 0) {
      FB_ERRORRETURN(
          commInvalidUsage,
          "There are {} used regions not yet released, might indicate COMM internal bugs",
          nOccupiedRegions);
    }

    freeRegionMaps_.clear();
    allocatedRegions_.clear();
    cachedRegionMap_.clear();
    slabAllocator_.reset();

    intialized_ = false;
  }

  return commSuccess;
}

memCacheAllocator::~memCacheAllocator() {
  CLOGF_SUBSYS(INFO, INIT, "Shutting down NCCLX memory cache allocator");
  FB_COMMCHECKTHROW_EX_NOCOMM(reset());
}

std::shared_ptr<memRegion> memCacheAllocator::getFreeMemReg(
    size_t nBytes,
    BucketType bucket) {
  // check if any free region with the same size is available
  auto regionBuket = folly::get_ptr(freeRegionMaps_, bucket);
  auto freeRegions = folly::get_ptr(*regionBuket, nBytes);
  if (NCCL_USE_SHARED_BUFFER_POOL && freeRegions != nullptr &&
      !freeRegions->empty()) {
    // free region with same size is available, no need to allocate a new one
    // NOTE: get free region from the front of the queue, which is the least
    // recently used region. We may consider adding a option to change the
    // policy, e.g. MRU
    auto region = freeRegions->front();
    freeRegions->pop_front();
    if (region->refCnt > 0) {
      CLOGF(
          ERR,
          "Found a free region with size {} is already in use, region={}",
          nBytes,
          region->toString().c_str());
    }
    CLOGF_SUBSYS(
        INFO, ALLOC, "Reusing free region {}", region->toString().c_str());
    return region;
  }
  return nullptr;
}

std::shared_ptr<memRegion> memCacheAllocator::createNewMemReg(
    size_t nBytes,
    BucketType bucket,
    const CommLogData* logMetaData,
    const char* callsite) {
  auto region = std::make_shared<memRegion>();
  if (poolPtr_ != nullptr && nBytes <= poolRemainSize_) {
    // pre allocated pool has enough space for this allocation, create a new
    // memRegion from the pool without allocating new slabs
    region->ptr = reinterpret_cast<void*>((uintptr_t)poolPtr_);
    region->cuHandle = poolHandle_;
    poolPtr_ = (char*)poolPtr_ + nBytes;
    poolRemainSize_ -= nBytes;
    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "Consume memory from pre-allocated pool ({} + {})",
        poolPtr_,
        nBytes);
  } else {
    // no free region is available, request a new memory region from allocator
    CUmemGenericAllocationHandle handle{};

    FB_COMMCHECKTHROW_EX_NOCOMM(slabAllocator_->cuMalloc(
        &region->ptr, nBytes, callsite, logMetaData, &handle));
    region->cuHandle = handle;

    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "Memory Pool is full, pool size {}, allocated new memory {} with size {}",
        NCCL_MEM_POOL_SIZE,
        region->ptr,
        nBytes);
  }
  // track the newly allocated region to the the ptr list
  allocatedRegions_.push_back(region);
  region->size = nBytes;
  region->creatorTid = syscall(SYS_gettid);
  region->bucket = bucket;
  CLOGF_SUBSYS(
      INFO, ALLOC, "Created new region {}", region->toString().c_str());
  return region;
}

commResult_t memCacheAllocator::getCachedCuMemById(
    const std::string& key,
    void** ptr,
    CUmemGenericAllocationHandle* cuHandle,
    size_t nBytes,
    const CommLogData* logMetaData,
    const char* callsite,
    BucketType bucket) {
  // We align the allocation bytes to 16 before enterring allocation/caching
  // logic
  size_t alignedSize16 = nBytes;
  alignedSize16 = ctran::utils::roundUp(alignedSize16, kMinAlignSize);
  std::unique_lock lock(mutex_);
  auto hashKey = genHash(key);
  // check if the given key is cached
  auto region = folly::get_default(cachedRegionMap_, hashKey);
  // Grab a new region if
  // 1. given key is not cached before
  // 2. the cached region is not available (refCnt > 0 && region is used by
  // another key)
  if (!isRegionUsable(region, key)) {
    region = getFreeMemReg(alignedSize16, bucket);
    if (region == nullptr) {
      // Need to create a new region
      CLOGF_SUBSYS(
          INFO,
          ALLOC,
          "No free region with size {} is available, need to create a new region",
          nBytes);
      region = createNewMemReg(nBytes, bucket, logMetaData, callsite);
    }

    if (region->ptr == nullptr) {
      *ptr = nullptr;
      FB_ERRORRETURN(
          commInvalidArgument,
          "Failed to allocate memory for key={}, hash={}",
          key,
          hashKey);
    }
    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "Caching new region for key {} hashKey {:x}: {}",
        key.c_str(),
        hashKey,
        region->toString().c_str());
  } else if (region->refCnt == 0) {
    // need to take this region off the free list if no one is using it
    freeMemReg(region);
  }

  region->ownerKey = key;
  region->refCnt++;

  if (cuHandle) {
    *cuHandle = region->cuHandle;
  }

  // insert/update the cached region
  cachedRegionMap_[hashKey] = region;

  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "Return the cached (requested key={}, hashKey={:x}, size {}) region {}",
      key.c_str(),
      hashKey,
      alignedSize16,
      region->toString().c_str());

  *ptr = region->ptr;
  return commSuccess;
}

commResult_t memCacheAllocator::reserve(const std::string& key) {
  std::unique_lock lock(mutex_);
  auto hashKey = genHash(key);

  auto cachedRegion = folly::get_default(cachedRegionMap_, hashKey);
  if (cachedRegion == nullptr) {
    std::string errStr = fmt::format(
        "Provided Key {} hash {:x} is not cached or not managed by cache allocator",
        key,
        hashKey);
    FB_ERRORRETURN(commInvalidArgument, "{}", errStr.c_str());
  }
  if (!isRegionUsable(cachedRegion, key)) {
    std::string errStr = fmt::format(
        "Cached region for key {} hash {:x} is currently reserved by hashKey {}, cannot be shared",
        key,
        hashKey,
        cachedRegion->ownerKey);
    CLOGF_TRACE(ALLOC, "{}", errStr.c_str());
    return commInProgress;
  }

  // if a region is reserved for the first time, move it out of free list, so it
  // cannot be reserved by other keys during get()
  if (cachedRegion->refCnt == 0) {
    freeMemReg(cachedRegion);
  }

  cachedRegion->refCnt++;
  cachedRegion->ownerKey = key;
  CLOGF_TRACE(
      ALLOC,
      "reserved region {} using key {} hash {:x}",
      cachedRegion->toString().c_str(),
      key.c_str(),
      hashKey);

  return commSuccess;
}

commResult_t memCacheAllocator::release(const std::vector<std::string>& keys) {
  std::unique_lock lock(mutex_);
  auto res = commSuccess;
  for (const auto& key : keys) {
    auto cachedRegion = folly::get_default(cachedRegionMap_, genHash(key));
    if (cachedRegion == nullptr) {
      FB_ERRORRETURN(commInvalidUsage, "key {} is not cached", key);
    }
    if (cachedRegion->refCnt == 0) {
      FB_ERRORRETURN(
          commInvalidUsage,
          "ptr {}, key {} has a double free called, region={}",
          cachedRegion->ptr,
          key.c_str(),
          cachedRegion->toString().c_str());
    }
    cachedRegion->refCnt--;
    if (cachedRegion->refCnt == 0) {
      CLOGF_SUBSYS(
          INFO,
          ALLOC,
          "Releasing region {} back to free list",
          cachedRegion->toString().c_str());
      cachedRegion->ownerKey = std::string(kUnAssignedOwner);
      auto regionBuket = folly::get_ptr(freeRegionMaps_, cachedRegion->bucket);
      auto freeRegions = folly::get_ptr(*regionBuket, cachedRegion->size);
      if (freeRegions == nullptr) {
        freeRegionMaps_[cachedRegion->bucket][cachedRegion->size] = {
            cachedRegion};
      } else {
        freeRegions->emplace_back(cachedRegion);
      }
    }
  }
  return res;
}

bool memCacheAllocator::isRegionUsable(
    std::shared_ptr<memRegion> region,
    const std::string& key) const {
  return region != nullptr && (region->refCnt == 0 || region->ownerKey == key);
}

size_t memCacheAllocator::getNumUsedReg() const {
  std::unique_lock lock(mutex_);
  return countOccupiedRegions();
}

size_t memCacheAllocator::getNumAllocReg() const {
  std::unique_lock lock(mutex_);
  return allocatedRegions_.size();
}

size_t memCacheAllocator::getUsedMem() const {
  std::unique_lock lock(mutex_);
  return slabAllocator_->getUsedMem();
}

void memCacheAllocator::printSnapshot() const {
  size_t avai_bytes = 0, total_bytes = 0;
  FB_CUDACHECKTHROW(cudaMemGetInfo(&avai_bytes, &total_bytes));
  // convert to GiB
  double avai_mem = avai_bytes / (1024 * 1024 * 1024);
  double total_mem = total_bytes / (1024 * 1024 * 1024);
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "NCCLX memory allocator snapshot: NCCL used {} bytes, {} allocated regions ({} buckets) {} regions in-use, avai_mem: {:.2f} of {:.2f} GiB",
      slabAllocator_->getUsedMem(),
      allocatedRegions_.size(),
      freeRegionMaps_.size(),
      countOccupiedRegions(),
      avai_mem,
      total_mem);
  for (const auto& [bucket, regionMap] : freeRegionMaps_) {
    for (const auto& [size, vec] : regionMap) {
      CLOGF_SUBSYS(
          INFO,
          ALLOC,
          "\tFree region map size in bucket-{}: {}, count: {}",
          int(bucket),
          size,
          vec.size());
      for (const auto& region : vec) {
        CLOGF_SUBSYS(
            INFO,
            ALLOC,
            "\t\tFree region with size {}: {}",
            size,
            region->toString().c_str());
      }
    }
  }
  for (const auto& [key, region] : cachedRegionMap_) {
    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "\tCached region map key: {} -> {}",
        key,
        region->toString().c_str());
  }
}

} // namespace ncclx::memory
