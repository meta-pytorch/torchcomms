// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cuda.h>
#include <sched.h>
#include <cstdint>
#include <deque>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <folly/MapUtil.h>
#include <folly/SharedMutex.h>

#include "comms/ctran/memory/SlabAllocator.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/alloc.h"

// Allow test class to access getFreeMemReg() method
class memCacheAllocatorTest;
namespace ncclx::memory {

constexpr const char* kUnAssignedOwner = "kUnAssignedOwner";
enum class BucketType : uint8_t { DEFAULT, NVL, RDMA };
/* strcuture to contain key inforation of a memory region, could be pointing to
 * the same handle */
struct memRegion {
  void* ptr{nullptr};
  std::size_t size{0};
  CUmemGenericAllocationHandle cuHandle{0};
  std::string ownerKey{kUnAssignedOwner};
  pid_t creatorTid{0};
  int refCnt{0};
  BucketType bucket{BucketType::DEFAULT};

  memRegion() = default;
  memRegion(
      void* ptr,
      size_t size,
      CUmemGenericAllocationHandle cuHandle,
      const std::string& ownerKey,
      pid_t creatorTid,
      BucketType bucket)
      : ptr(ptr),
        size(size),
        cuHandle(cuHandle),
        ownerKey(ownerKey),
        creatorTid(creatorTid),
        bucket(bucket) {}

  std::string toString() const {
    std::stringstream ss;
    ss << "[ptr: " << ptr << ", size: " << size << ", bucket: " << int(bucket)
       << ", creatorTid: " << creatorTid << ", ownerKey: " << std::hex
       << ownerKey << std::dec << ", refCnt: " << refCnt << "]";
    return ss.str();
  }
};

using memRegionMap_t =
    std::unordered_map<size_t, std::deque<std::shared_ptr<memRegion>>>;

/* Singleton class of memory allocator */
class memCacheAllocator {
  // Allow test class to access getFreeMemReg() method
  friend class ::memCacheAllocatorTest;

 public:
  ~memCacheAllocator();
  static std::shared_ptr<memCacheAllocator> getInstance();

  /* Initialize the allocator with a pre-allocated memory buffer */
  commResult_t init();

  /* Retrieve a CUDA buffer from the pool using the given id/key. If cache miss,
   * new buffer may be retrivied by calling getFreeMemReg and cached.
   * @param key: unique string key of the buffer.
   * @param ptr: return a pointer of CUDA buffer.
   * @param cuHandle: return the CUMem handle of the buffer that can used for
   * sharing the buffer with other processes. Skip if nullptr is passed in.
   * @param nBytes: size of the requested buffer.
   * @param logMetaData: log metadata
   * @param callsite: callsite
   * @param bucket: bucket id to search/create for free regions
   */
  commResult_t getCachedCuMemById(
      const std::string& key,
      void** ptr,
      CUmemGenericAllocationHandle* cuHandle,
      size_t nBytes,
      const CommLogData* logMetaData,
      const char* callsite,
      BucketType bucket = BucketType::DEFAULT);

  /* Reserve a buffer from the pool to be used by a communicator until it is
   * released. The buffer must be cached in the pool before.
   * @param key: unique key to access cached memory region.
   */
  commResult_t reserve(const std::string& key);

  /* Release a buffer back to the pool to be reused later
   * @param keys or ptr: key used to cache the buffer or pointer obtained
   * previously
   */
  commResult_t release(const std::vector<std::string>& key);

  /* Query number of regions currently in use */
  size_t getNumUsedReg() const;

  /* Query number of regions allocated so far */
  size_t getNumAllocReg() const;

  /* Query the amont of memory in byte currently allocated by the allocator */
  size_t getUsedMem() const;

  /* Print snapshot of memory allocated */
  void printSnapshot() const;

  /* clean up the allocated resources */
  commResult_t reset();

 private:
  /* Return a available free memory region larger than nBytes. If no such
   * regions are available, return nullptr.
   * @param nBytes: size of the requested buffer.
   * @param bucket: bucket id to search for free regions
   */
  std::shared_ptr<memRegion> getFreeMemReg(size_t nBytes, BucketType bucket);

  /* Create a new memory region larger than nBytes. If first attempt to
   * get from pre-allocated pool, allocate a new one if no more regions are
   * available. nullptr will be returned if no memory can be allocated.
   * @param nBytes: size of the requested buffer.
   * @param logMetaData: log metadata
   * @param callsite: callsite
   * the newly created memory region cannot be used by other users until the
   * owner releases it.
   */
  std::shared_ptr<memRegion> createNewMemReg(
      size_t nBytes,
      BucketType bucket,
      const CommLogData* logMetaData,
      const char* callsite);

  void freeMemReg(std::shared_ptr<memRegion> reg) {
    // FIXME: more efficient way to erase region from the free list
    auto regionBucket = folly::get_ptr(freeRegionMaps_, reg->bucket);
    auto freeSegments = folly::get_ptr(*regionBucket, reg->size);
    for (auto it = freeSegments->begin(); it != freeSegments->end(); ++it) {
      if (it->get() == reg.get()) {
        freeSegments->erase(it);
        break;
      }
    }
  }

  /* Internal helper function to generate a hash from given string */
  inline uint64_t genHash(const std::string& key) const {
    return std::hash<std::string>{}(key);
  }
  /* internal method to check whether a key has a corresponding accessible
   * region*/
  bool isRegionUsable(std::shared_ptr<memRegion> region, const std::string& key)
      const;
  int countOccupiedRegions() const {
    std::unordered_set<uintptr_t> unique_addrs;
    for (const auto& [_, reg] : cachedRegionMap_) {
      if (reg->refCnt > 0) {
        unique_addrs.insert(reinterpret_cast<uintptr_t>(reg->ptr));
      }
    }
    return unique_addrs.size();
  }

  // Current pool ptr and remaining unused size; will allocate new segments if
  // no remaining size is usable
  void* poolPtr_{nullptr};
  size_t poolRemainSize_{0};
  CUmemGenericAllocationHandle poolHandle_{0};
  // regions (pointers) of all memory regions allocated at runtime
  std::vector<std::shared_ptr<memRegion>> allocatedRegions_;
  // map of free regions with size as key for quick search
  std::unordered_map<BucketType, memRegionMap_t> freeRegionMaps_;
  // map of cached regions with id/key as key for quick search
  std::unordered_map<uint64_t, std::shared_ptr<memRegion>> cachedRegionMap_;
  bool intialized_{false};
  mutable folly::SharedMutex mutex_;
  std::unique_ptr<SlabAllocator> slabAllocator_;
};

}; // namespace ncclx::memory
