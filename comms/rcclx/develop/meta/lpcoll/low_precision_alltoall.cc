/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "low_precision_alltoall.h"
#include "comm.h"

/**
 * Performs low precision all-to-all communication using FP8 quantization.
 * Each rank sends unique data to every other rank with bandwidth efficiency.
 */
HOT ncclResult_t ncclLowPrecisionAllToAll(
    const void* RESTRICT sendbuff,
    void* RESTRICT recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  if (count == 0) {
    return ncclSuccess;
  }

  // Calculate element counts based on data type
  size_t chunkSize = count; // Elements to send/receive per rank pair
  size_t totalInputElements = count * nRanks; // Total input elements per rank
  size_t totalOutputElements = count * nRanks; // Total output elements per rank

  // Calculate FP8 element counts (1:1 mapping for bfloat16 and float)
  size_t fp8ChunkSize = chunkSize; // FP8 elements per rank pair (1:1 mapping)
  size_t totalFp8Elements = totalInputElements; // Total FP8 elements per rank

  // Ensure buffer pool is initialized with FP8 count
  NCCLCHECK(ncclEnsureLowPrecisionBufferPool(comm, totalFp8Elements, nRanks));

  // Get pre-allocated buffers from communicator's pool
  rccl_float8* fp8Phase1Buffer;
  rccl_float8* fp8Phase2Buffer;
  rccl_float8* fp8AllGatherBuffer; // not used for alltoall
  float* floatReductionBuffer; // not used for alltoall
  float* floatOutputBuffer; // not used for alltoall

  NCCLCHECK(ncclLowPrecisionBufferPoolGetBuffers(
      &comm->lowPrecisionBufferPool,
      totalFp8Elements,
      nRanks,
      &fp8Phase1Buffer,
      &fp8Phase2Buffer,
      nullptr, // not needed
      nullptr, // not needed
      nullptr)); // not needed

  // Setup buffer layouts for each algorithm phase
  // Input layout uses original counts and is contiguous
  // FP8 layouts use FP8 counts
  PhaseBufferLayout phase1Layout(
      PhaseBufferLayout::CONTIGUOUS,
      totalFp8Elements,
      nRanks,
      rank,
      fp8ChunkSize);

  // Phase 2: Output buffer will contain FP8 chunks received from each rank
  PhaseBufferLayout phase2Layout(
      PhaseBufferLayout::CONTIGUOUS,
      totalFp8Elements,
      nRanks,
      rank,
      fp8ChunkSize);

  // Calculate kernel configuration
  struct ncclLowPrecisionKernelConfig kernelConfig;
  NCCLCHECK(ncclCalculateLowPrecisionKernelConfig(
      totalFp8Elements, fp8ChunkSize, &kernelConfig));

  // Handle buffer addressing (AllToAll is always out-of-place)
  const void* actualInput = sendbuff;
  void* actualOutput = recvbuff;

  // Phase 1: Quantize input data to FP8 format based on input data type
  if (datatype == ncclBfloat16) {
    // Input: totalInputElements bfloat16 → Output: totalFp8Elements FP8 (1:1
    // mapping)
    quantizeBF16ToFp8Kernel<<<
        kernelConfig.fullGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(
        (const uint16_t*)actualInput,
        fp8Phase1Buffer,
        totalFp8Elements,
        0,
        totalInputElements);
  } else if (datatype == ncclFloat) {
    // Input: totalInputElements float → Output: totalFp8Elements FP8 (1:1
    // mapping)
    quantizeFloatToFp8Kernel<<<
        kernelConfig.fullGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(
        (const float*)actualInput,
        fp8Phase1Buffer,
        totalFp8Elements,
        0,
        totalInputElements);
  }

  // Phase 2: All-to-all exchange of quantized data chunks between ranks
  // Correct AllToAll semantics:
  // - rank i sends chunk j (elements j*chunkSize to (j+1)*chunkSize-1) to rank
  // j
  // - rank i receives chunk i from rank j and places it at position j in output
  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < nRanks; r++) {
    if (r == rank) {
      // Handle self-case: copy my own FP8 chunk to my own output position
      // My chunk: FP8 chunk rank from my input → FP8 position rank in my output
      size_t myInputChunkOffset =
          rank * fp8ChunkSize; // My FP8 chunk from my input
      size_t myOutputOffset =
          rank * fp8ChunkSize; // FP8 position rank in my output

      CUDACHECK(cudaMemcpyAsync(
          ((char*)fp8Phase2Buffer) + myOutputOffset * sizeof(rccl_float8),
          ((char*)fp8Phase1Buffer) + myInputChunkOffset * sizeof(rccl_float8),
          fp8ChunkSize * sizeof(rccl_float8),
          cudaMemcpyDeviceToDevice,
          stream));
      continue;
    }

    // Send chunk r (destined for rank r) from my input to rank r
    size_t sendChunkOffset = r * fp8ChunkSize; // FP8 chunk r from my input

    NCCLCHECK(ncclSend(
        ((rccl_float8*)fp8Phase1Buffer) + sendChunkOffset,
        fp8ChunkSize,
        ncclUint8,
        r,
        comm,
        stream));

    // Receive chunk rank (my chunk) from rank r and place it at position r in
    // my output
    size_t recvOffset = r * fp8ChunkSize; // FP8 position r in my output

    NCCLCHECK(ncclRecv(
        ((rccl_float8*)fp8Phase2Buffer) + recvOffset,
        fp8ChunkSize,
        ncclUint8,
        r,
        comm,
        stream));
  }
  NCCLCHECK(ncclGroupEnd());

  // Phase 3: Final dequantization from FP8 based on output data type
  if (datatype == ncclBfloat16) {
    // Input: totalFp8Elements FP8 → Output: totalOutputElements bfloat16 (1:1
    // mapping)
    dequantizeFp8ToBF16Kernel<<<
        kernelConfig.fullGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(
        fp8Phase2Buffer,
        (uint16_t*)actualOutput,
        totalFp8Elements,
        0,
        totalFp8Elements);
  } else if (datatype == ncclFloat) {
    // Input: totalFp8Elements FP8 → Output: totalOutputElements float (1:1
    // mapping)
    dequantizeFp8ToFloatKernel<<<
        kernelConfig.fullGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(
        fp8Phase2Buffer,
        (float*)actualOutput,
        totalFp8Elements,
        0,
        totalFp8Elements);
  }

  return ncclSuccess;
}
