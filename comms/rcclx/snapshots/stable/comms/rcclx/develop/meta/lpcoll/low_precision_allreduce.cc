/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "low_precision_allreduce.h"
#include "comm.h"

/**
 * Performs low precision allreduce operation using FP8 quantization for
 * bandwidth efficiency. Implements scatter-reduce-allgather pattern with local
 * reduction between scatter and allgather phases.
 */
HOT ncclResult_t ncclLowPrecisionAllReduce(
    const void* RESTRICT sendbuff,
    void* RESTRICT recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream) {
  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  if (count == 0) {
    return ncclSuccess;
  }

  // Track buffer pool size for debugging purposes
  size_t bufferPoolSizeBefore = 0;
  if (rank == 0) {
    bufferPoolSizeBefore = comm->lowPrecisionBufferPool.maxBufferSize;
  }

  // Calculate FP8 element count for direct bfloat16 data
  // Input: count bfloat16 elements
  // After quantization: count FP8 elements (1:1 mapping)
  size_t fp8Count = count;

  // Ensure buffer pool is initialized with FP8 count
  NCCLCHECK(ncclEnsureLowPrecisionBufferPool(comm, fp8Count, nRanks));

  // Get pre-allocated buffers from communicator's pool
  rccl_float8* fp8Phase1Buffer;
  rccl_float8* fp8Phase2Buffer;
  rccl_float8* fp8AllGatherBuffer;
  float* floatReductionBuffer;
  float* floatOutputBuffer; // not used for allreduce

  NCCLCHECK(ncclLowPrecisionBufferPoolGetBuffers(
      &comm->lowPrecisionBufferPool,
      fp8Count,
      nRanks,
      &fp8Phase1Buffer,
      &fp8Phase2Buffer,
      &fp8AllGatherBuffer,
      &floatReductionBuffer,
      nullptr));

  size_t uniformChunkSize = fp8Count / nRanks;
  // Align uniformChunkSize to cache line for optimal memory access.
  // This ensures efficient memory access patterns during scatter-reduce,
  // local reduction, and all-gather phases
  uniformChunkSize = (uniformChunkSize / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
  size_t myChunkStart = rank * uniformChunkSize;

  // Create buffer layouts (stack allocated, CUDA graph compatible)
  // Input layout uses original count and is contiguous
  // All FP8 layouts use fp8Count
  PhaseBufferLayout phase1Layout(
      PhaseBufferLayout::CONTIGUOUS, fp8Count, nRanks, rank, uniformChunkSize);

  size_t phase2TotalSize = nRanks * uniformChunkSize;
  PhaseBufferLayout phase2Layout(
      PhaseBufferLayout::INTERLEAVED,
      phase2TotalSize,
      nRanks,
      rank,
      uniformChunkSize);
  // Phase4 layout also uses fp8Count for FP8 operations
  PhaseBufferLayout phase4Layout(
      PhaseBufferLayout::CONTIGUOUS, fp8Count, nRanks, rank, uniformChunkSize);

  // Calculate kernel configuration
  struct ncclLowPrecisionKernelConfig kernelConfig;
  NCCLCHECK(ncclCalculateLowPrecisionKernelConfig(
      count, uniformChunkSize, &kernelConfig));

  const void* actualInput = sendbuff;

  // Phase 1: Quantize input data to FP8 format based on input data type
  if (datatype == ncclBfloat16) {
    // Input: count bfloat16 elements → Output: count FP8 elements (1:1 mapping)
    quantizeBF16ToFp8Kernel<<<
        kernelConfig.fullGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(
        (const uint16_t*)actualInput, fp8Phase1Buffer, fp8Count, 0, count);
  } else if (datatype == ncclFloat) {
    // Input: count float elements → Output: count FP8 elements (1:1 mapping)
    quantizeFloatToFp8Kernel<<<
        kernelConfig.fullGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(
        (const float*)actualInput, fp8Phase1Buffer, fp8Count, 0, count);
  }

  // Phase 2: Scatter-reduce exchange pattern
  // Use async memory operations for CUDA graph compatibility
  NCCLCHECK(ncclGroupStart());

  // Copy own contribution using async memcpy (CUDA graph compatible)
  size_t mySelfContribOffset = phase2Layout.getContributionOffset(rank);
  size_t myDataForMyChunk = phase1Layout.getChunkOffset(rank);

  // Use cudaMemcpyAsync instead of cudaMemcpy for CUDA graph compatibility
  CUDACHECK(cudaMemcpyAsync(
      ((char*)fp8Phase2Buffer) + mySelfContribOffset * sizeof(rccl_float8),
      ((char*)fp8Phase1Buffer) + myDataForMyChunk * sizeof(rccl_float8),
      uniformChunkSize * sizeof(rccl_float8),
      cudaMemcpyDeviceToDevice,
      stream));

  // Send/receive operations
  for (int otherRank = 0; otherRank < nRanks; otherRank++) {
    if (otherRank == rank)
      continue;

    size_t chunkForOtherRank = phase1Layout.getChunkOffset(otherRank);
    NCCLCHECK(ncclSend(
        ((rccl_float8*)fp8Phase1Buffer) + chunkForOtherRank,
        uniformChunkSize,
        ncclUint8,
        otherRank,
        comm,
        stream));

    size_t otherRankContribOffset =
        phase2Layout.getContributionOffset(otherRank);
    NCCLCHECK(ncclRecv(
        ((rccl_float8*)fp8Phase2Buffer) + otherRankContribOffset,
        uniformChunkSize,
        ncclUint8,
        otherRank,
        comm,
        stream));
  }
  NCCLCHECK(ncclGroupEnd());

  // Phase 3: Local reduction
  size_t warpsPerBlock = (kernelConfig.blockSize + 63) / 64;
  size_t sharedMemBytes =
      warpsPerBlock * sizeof(float) + kernelConfig.blockSize * sizeof(float);

  localReductionKernel<<<
      kernelConfig.chunkGridSize,
      kernelConfig.blockSize,
      sharedMemBytes,
      stream>>>(
      fp8Phase2Buffer,
      floatReductionBuffer + myChunkStart,
      phase2TotalSize,
      0,
      uniformChunkSize,
      nRanks,
      rank);

  // Phase 3b: Re-quantize reduced results
  quantizeFloatToFp8Kernel<<<
      kernelConfig.chunkGridSize,
      kernelConfig.blockSize,
      0,
      stream>>>(
      floatReductionBuffer + myChunkStart,
      ((rccl_float8*)fp8Phase1Buffer) + myChunkStart,
      uniformChunkSize, // Use uniformChunkSize instead of count
      0,
      uniformChunkSize);

  // Phase 4: All-gather
  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < nRanks; r++) {
    if (r == rank) {
      size_t myLayoutStart = phase4Layout.getChunkOffset(rank);
      CUDACHECK(cudaMemcpyAsync(
          ((char*)fp8AllGatherBuffer) + myLayoutStart * sizeof(rccl_float8),
          ((char*)fp8Phase1Buffer) + myLayoutStart * sizeof(rccl_float8),
          uniformChunkSize * sizeof(rccl_float8),
          cudaMemcpyDeviceToDevice,
          stream));
      continue;
    }

    size_t mySendOffset = phase4Layout.getChunkOffset(rank);
    size_t rRecvOffset = phase4Layout.getChunkOffset(r);

    NCCLCHECK(ncclSend(
        ((rccl_float8*)fp8Phase1Buffer) + mySendOffset,
        uniformChunkSize,
        ncclUint8,
        r,
        comm,
        stream));

    NCCLCHECK(ncclRecv(
        ((rccl_float8*)fp8AllGatherBuffer) + rRecvOffset,
        uniformChunkSize,
        ncclUint8,
        r,
        comm,
        stream));
  }
  NCCLCHECK(ncclGroupEnd());

  // Phase 5: Final dequantization based on output data type
  if (datatype == ncclBfloat16) {
    // Input: count FP8 elements → Output: count bfloat16 elements (1:1 mapping)
    dequantizeFp8ToBF16Kernel<<<
        kernelConfig.fullGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(
        fp8AllGatherBuffer, (uint16_t*)recvbuff, fp8Count, 0, fp8Count);
  } else if (datatype == ncclFloat) {
    // Input: count FP8 elements → Output: count float elements (1:1 mapping)
    dequantizeFp8ToFloatKernel<<<
        kernelConfig.fullGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(fp8AllGatherBuffer, (float*)recvbuff, fp8Count, 0, fp8Count);
  }

  return ncclSuccess;
}
