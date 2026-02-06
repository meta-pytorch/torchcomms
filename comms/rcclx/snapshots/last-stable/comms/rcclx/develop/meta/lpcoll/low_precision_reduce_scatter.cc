/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "low_precision_reduce_scatter.h"
#include "comm.h"

/**
 * Performs low precision reduce-scatter operation using FP8 quantization.
 * Combines input data with reduction operation and distributes unique chunks to
 * each rank.
 */
HOT ncclResult_t ncclLowPrecisionReduceScatter(
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

  // Early exit for empty data
  if (count == 0) {
    return ncclSuccess;
  }

  // Calculate FP8 element count for direct bfloat16 data
  // Input: count bfloat16 elements
  // After quantization: count FP8 elements (1:1 mapping)
  size_t fp8Count = count;

  // Calculate data partitioning: divide FP8 data evenly across ranks
  size_t uniformChunkSize = fp8Count / nRanks;
  // Align uniformChunkSize to cache line for optimal memory access.
  // This ensures efficient memory access patterns during scatter-reduce
  // and local reduction phases
  uniformChunkSize = (uniformChunkSize / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
  size_t myChunkStart = rank * uniformChunkSize;
  size_t myChunkSize = uniformChunkSize;

  // Ensure buffer pool is initialized with FP8 count
  NCCLCHECK(ncclEnsureLowPrecisionBufferPool(comm, fp8Count, nRanks));

  // Get pre-allocated buffers from communicator's pool
  rccl_float8* fp8Phase1Buffer;
  rccl_float8* fp8Phase2Buffer;
  rccl_float8* fp8AllGatherBuffer; // not used for reduce-scatter
  float* floatReductionBuffer;
  float* floatOutputBuffer; // not used for reduce-scatter

  NCCLCHECK(ncclLowPrecisionBufferPoolGetBuffers(
      &comm->lowPrecisionBufferPool,
      fp8Count,
      nRanks,
      &fp8Phase1Buffer,
      &fp8Phase2Buffer,
      nullptr, // not needed
      &floatReductionBuffer,
      nullptr)); // not needed

  // Setup buffer layouts for each algorithm phase
  // Input layout uses original count and is contiguous
  // FP8 layouts use fp8Count and uniformChunkSize
  PhaseBufferLayout phase1Layout(
      PhaseBufferLayout::CONTIGUOUS, fp8Count, nRanks, rank, uniformChunkSize);

  // Phase 2: all-to-all exchange buffer layout
  size_t phase2TotalSize = nRanks * uniformChunkSize;
  PhaseBufferLayout phase2Layout(
      PhaseBufferLayout::INTERLEAVED,
      phase2TotalSize,
      nRanks,
      rank,
      uniformChunkSize);

  // Calculate kernel configuration
  struct ncclLowPrecisionKernelConfig kernelConfig;
  NCCLCHECK(
      ncclCalculateLowPrecisionKernelConfig(count, myChunkSize, &kernelConfig));

  // Handle in-place vs out-of-place buffer addressing correctly
  const void* actualInput;
  void* actualOutput;

  if (sendbuff == recvbuff) {
    // In-place operation: For reduce-scatter, the framework provides
    // rank-specific buffers Each rank has its own buffer where sendbuff ==
    // recvbuff We read from and write to the same buffer (RCCL manages internal
    // communication)
    actualInput = sendbuff; // Read from this rank's buffer
    actualOutput = recvbuff; // Write to this rank's buffer (same as sendbuff)
  } else {
    // Out-of-place operation: separate input and output buffers
    actualInput = sendbuff;
    actualOutput = recvbuff;
  }

  TRACE(
      NCCL_COLL,
      "Quantized ARG ReduceScatter: count=%zu, using blockSize=%d, maxBlocks=%d",
      count,
      kernelConfig.blockSize,
      kernelConfig.maxBlocks);

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
  // Each rank sends its portion of ALL chunks to the rank responsible for
  // reducing that chunk
  NCCLCHECK(ncclGroupStart());

  // Copy own contribution for the chunk that I will reduce
  size_t mySelfContribOffset = phase2Layout.getContributionOffset(rank);
  size_t myDataForMyChunk = phase1Layout.getChunkOffset(rank);

  CUDACHECK(cudaMemcpyAsync(
      ((char*)fp8Phase2Buffer) + mySelfContribOffset * sizeof(rccl_float8),
      ((char*)fp8Phase1Buffer) + myDataForMyChunk * sizeof(rccl_float8),
      uniformChunkSize * sizeof(rccl_float8),
      cudaMemcpyDeviceToDevice,
      stream));

  // Send/receive operations for scatter-reduce pattern
  for (int otherRank = 0; otherRank < nRanks; otherRank++) {
    if (otherRank == rank)
      continue; // Already copied own data above

    // Send chunk 'otherRank' data to otherRank for their reduction
    size_t chunkForOtherRank = phase1Layout.getChunkOffset(otherRank);

    NCCLCHECK(ncclSend(
        ((rccl_float8*)fp8Phase1Buffer) + chunkForOtherRank,
        uniformChunkSize,
        ncclUint8,
        otherRank,
        comm,
        stream));

    // Receive chunk 'rank' data from otherRank for MY reduction
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

  // Phase 3: Local reduction of received FP8 contributions from all ranks
  // Calculate shared memory requirements for efficient reduction
  size_t warpsPerBlock = (kernelConfig.blockSize + 63) / 64;
  size_t sharedMemBytes =
      warpsPerBlock * sizeof(float) + kernelConfig.blockSize * sizeof(float);

  localReductionKernel<<<
      kernelConfig.chunkGridSize,
      kernelConfig.blockSize,
      sharedMemBytes,
      stream>>>(
      fp8Phase2Buffer,
      floatReductionBuffer,
      phase2TotalSize,
      0,
      uniformChunkSize,
      nRanks,
      rank);

  // Phase 4: Convert final float32 results to output buffer based on output
  // data type
  if (datatype == ncclBfloat16) {
    // Input: myChunkSize float elements → Output: uniformChunkSize bfloat16
    // elements (1:1 mapping)
    dequantizeFloatToBF16Kernel<<<
        kernelConfig.chunkGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(
        floatReductionBuffer,
        (uint16_t*)actualOutput,
        myChunkSize,
        0,
        myChunkSize);
  } else if (datatype == ncclFloat) {
    // For float output, no dequantize step needed - direct copy from float
    // buffer
    CUDACHECK(cudaMemcpyAsync(
        actualOutput,
        floatReductionBuffer,
        myChunkSize * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream));
  }

  return ncclSuccess;
}
