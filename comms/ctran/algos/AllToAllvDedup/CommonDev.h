// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/utils/DevAttribute.h"

namespace ctran::alltoallvdedup {
const int MAX_NUM_RECV_BUCKETS = 16;
const int MAX_NUM_BLOCKS_PER_CHUNK = 512;

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

DEVICE_ATTRIBUTE LocalBucketsBitMap* getTmpChunkBitMaps(void* ptr) {
  return reinterpret_cast<LocalBucketsBitMap*>(
      reinterpret_cast<char*>(ptr) + sizeof(FwdChkHdr));
}

DEVICE_ATTRIBUTE int* getTmpChunkBlockIds(void* ptr, const size_t numBlocks) {
  return reinterpret_cast<int*>(
      reinterpret_cast<char*>(ptr) + sizeof(FwdChkHdr) +
      numBlocks * sizeof(LocalBucketsBitMap));
}

// Shift by dynamically defined hdr len based on actual number of blocks to be
// transferred in order to keep a single contig chunk for RDMA, used when SEND
// rank transfers to FWD rank.
DEVICE_ATTRIBUTE void* getTmpChunkData(size_t hdrLen, void* ptr) {
  return reinterpret_cast<char*>(ptr) + hdrLen;
}

DEVICE_ATTRIBUTE int getTmpChunkIdx(
    const PersistConfig& config,
    const int step) {
  return step & (config.tmpNumChunks - 1);
}

DEVICE_ATTRIBUTE void* getTmpChunkPtr(
    const PersistConfig& config,
    void* ptr,
    const int step,
    const int nodeId) {
  size_t nodeOffset = nodeId * config.tmpNumChunks * config.tmpChunkSize;
  size_t chunkOffset = getTmpChunkIdx(config, step) * config.tmpChunkSize;
  return reinterpret_cast<char*>(ptr) + nodeOffset + chunkOffset;
}

DEVICE_ATTRIBUTE void* getTmpChunkPtrByIdx(
    const PersistConfig& config,
    void* ptr,
    const int chunkIdx,
    // myId is either nodeId or localRank
    const int myId) {
  size_t myOffset = myId * config.tmpNumChunks * config.tmpChunkSize;
  size_t chunkOffset = chunkIdx * config.tmpChunkSize;
  return reinterpret_cast<char*>(ptr) + myOffset + chunkOffset;
}

// GPE thread has already maintained tmpRemFwdBuffs per node; skip nodeId shift
DEVICE_ATTRIBUTE void*
getTmpChunkPtr(const PersistConfig& config, void* ptr, const int step) {
  size_t chunkOffset = getTmpChunkIdx(config, step) * config.tmpChunkSize;
  return reinterpret_cast<char*>(ptr) + chunkOffset;
}

DEVICE_ATTRIBUTE size_t getOffset(void* ptr, void* base) {
  return reinterpret_cast<size_t>(ptr) - reinterpret_cast<size_t>(base);
}

DEVICE_ATTRIBUTE algos::MPSCTbSync<>* getFwdRecvSync(
    const PersistConfig& config,
    const KernSync& kSync,
    int owner, // owner local rank of the sync
    int peer, // peer local rank that the sync is for
    int chunkIdx) {
  return ptrElemOffset<algos::MPSCTbSync<>>(
      kSync.remFwdRecvSyncs[owner], peer * config.tmpNumChunks + chunkIdx);
}

DEVICE_ATTRIBUTE size_t getMaxNumBlocksPerChunk(
    const PersistConfig* config,
    const PersistArgs& pArgs,
    const size_t typeSize) {
  return (config->tmpChunkSize - sizeof(FwdChkHdr)) /
      (sizeof(LocalBucketsBitMap) + sizeof(int) + pArgs.blockCount * typeSize);
}

template <typename T>
DEVICE_ATTRIBUTE size_t
getMaxNumBlocksPerChunk(const PersistConfig& config, const PersistArgs& pArgs) {
  return getMaxNumBlocksPerChunk(&config, pArgs, sizeof(T));
}

DEVICE_ATTRIBUTE size_t getChunkHeaderLen(const size_t numBlocks) {
  return ALIGN(
      sizeof(FwdChkHdr) +
          numBlocks * (sizeof(LocalBucketsBitMap) + sizeof(int)),
      16);
}
} // namespace ctran::alltoallvdedup
