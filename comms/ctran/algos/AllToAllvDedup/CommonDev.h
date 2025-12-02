// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/utils/DevAttribute.h"

namespace ctran::alltoallvdedup {
constexpr int MAX_NUM_RECV_BUCKETS = 16;
constexpr int MAX_NUM_NODES = 8;
constexpr int MAX_NUM_GROUPS_PER_ROLE = 8;

template <typename T>
DEVICE_ATTRIBUTE T* ptrElemOffset(void* ptr, size_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<T*>(ptr) + offset);
}

DEVICE_ATTRIBUTE int bucketToRank(const PersistArgs& pArgs, int bucket) {
  return bucket / pArgs.numRecvBuckets;
}

#define ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

DEVICE_ATTRIBUTE FwdChkHdr* getTmpChunkHdr(void* ptr) {
  return reinterpret_cast<FwdChkHdr*>(ptr);
}

DEVICE_ATTRIBUTE int* getTmpChunkBlockIds(void* ptr) {
  // chunk format: hdr, [blkId] * numBlocks, data
  return reinterpret_cast<int*>(
      reinterpret_cast<char*>(ptr) + sizeof(FwdChkHdr));
}
DEVICE_ATTRIBUTE int* getTmpChunkBlockIds(void* ptr, const size_t numBlocks) {
  return reinterpret_cast<int*>(
      reinterpret_cast<char*>(ptr) + sizeof(FwdChkHdr));
}

// Shift by dynamically defined hdr len based on actual number of blocks to be
// transferred in order to keep a single contig chunk for RDMA, used when SEND
// rank transfers to FWD rank.
template <typename T>
DEVICE_ATTRIBUTE T* getTmpChunkData(void* ptr, const size_t numBlocks) {
  auto dataPtr = reinterpret_cast<char*>(ptr) +
      ALIGN(sizeof(FwdChkHdr) + numBlocks * sizeof(int), 16);
  return reinterpret_cast<T*>(dataPtr);
}

DEVICE_ATTRIBUTE size_t
getTmpChunkActualLen(const PersistArgs& pArgs, const size_t actualNumBlocks) {
  return ALIGN(sizeof(FwdChkHdr) + pArgs.maxNumStepBlks * sizeof(int), 16) +
      actualNumBlocks * pArgs.blockCount * pArgs.typeSize;
}

DEVICE_ATTRIBUTE int getTmpChunkIdx(
    const PersistConfig& config,
    const int step) {
  return step % config.tmpNumChunks;
}

DEVICE_ATTRIBUTE void* getTmpChunkPtr(
    const PersistConfig& config,
    void* ptr,
    const int step,
    const int myId) {
  size_t myOffset = myId * config.tmpNumChunks * config.tmpChunkSize;
  size_t chunkOffset = getTmpChunkIdx(config, step) * config.tmpChunkSize;
  return reinterpret_cast<char*>(ptr) + myOffset + chunkOffset;
}

DEVICE_ATTRIBUTE size_t getOffset(void* ptr, void* base) {
  return reinterpret_cast<size_t>(ptr) - reinterpret_cast<size_t>(base);
}

DEVICE_ATTRIBUTE int getMaxNumBlocksPerChunk(
    const PersistConfig* config,
    const PersistArgs& pArgs) {
  // chunk format: hdr, [blkId] * numBlocks, data
  return (config->tmpChunkSize - sizeof(FwdChkHdr)) /
      (sizeof(int) + ALIGN(pArgs.blockCount * pArgs.typeSize, 16));
}

DEVICE_ATTRIBUTE size_t getChunkHeaderLen(const size_t numBlocks) {
  return ALIGN(sizeof(FwdChkHdr) + numBlocks * sizeof(int), 16);
}

} // namespace ctran::alltoallvdedup
