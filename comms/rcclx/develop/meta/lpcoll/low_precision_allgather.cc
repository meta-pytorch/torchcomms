/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "low_precision_allgather.h"
#include "comm.h"

/**
 * Performs low precision allgather operation by quantizing input data to FP8
 * format, exchanging quantized chunks between ranks, and dequantizing back to
 * original format.
 */
HOT ncclResult_t ncclLowPrecisionAllGather(
    const void* RESTRICT sendbuff,
    void* RESTRICT recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  // Calculate FP8 element count for supported data types
  // Input per rank: count elements → Output per rank: count FP8 elements (1:1
  // mapping) Total output: count*nRanks FP8 elements
  size_t fp8CountPerRank = count;
  size_t totalFp8Count = fp8CountPerRank * nRanks;

  // Ensure buffer pool is initialized with total FP8 count
  NCCLCHECK(ncclEnsureLowPrecisionBufferPool(comm, totalFp8Count, nRanks));

  // Get pre-allocated buffers from communicator's pool
  rccl_float8* fp8Phase1Buffer;
  rccl_float8* fp8Phase2Buffer;
  rccl_float8* fp8AllGatherBuffer; // not used for allgather
  float* floatReductionBuffer; // not used for allgather
  float* floatOutputBuffer; // not used for allgather

  NCCLCHECK(ncclLowPrecisionBufferPoolGetBuffers(
      &comm->lowPrecisionBufferPool,
      totalFp8Count,
      nRanks,
      &fp8Phase1Buffer,
      &fp8Phase2Buffer,
      nullptr, // not needed
      nullptr, // not needed
      nullptr));

  // FP8 layouts use fp8CountPerRank (FP8 elements per rank)
  size_t uniformChunkSize = fp8CountPerRank;
  size_t outputCount = totalFp8Count;

  // Setup buffer layouts for each algorithm phase
  // Input layout uses original count and is contiguous
  // FP8 layouts use fp8CountPerRank (FP8 elements per rank)
  PhaseBufferLayout phase1Layout(
      PhaseBufferLayout::CONTIGUOUS,
      fp8CountPerRank,
      nRanks,
      rank,
      uniformChunkSize);

  // Phase 2: all-to-all exchange buffer layout - interleaved output
  size_t phase2TotalSize = nRanks * uniformChunkSize;
  PhaseBufferLayout phase2Layout(
      PhaseBufferLayout::INTERLEAVED,
      phase2TotalSize,
      nRanks,
      rank,
      uniformChunkSize);

  // Calculate kernel configuration
  struct ncclLowPrecisionKernelConfig kernelConfig;
  NCCLCHECK(ncclCalculateLowPrecisionKernelConfig(
      count, uniformChunkSize, &kernelConfig));

  const void* actualInput;
  void* actualOutput;

  if (datatype == ncclBfloat16) {
    // Standard NCCL allgather in-place condition: sendbuff == recvbuff + rank *
    // sendcount
    if (sendbuff == ((const uint16_t*)recvbuff) + rank * count) {
      // In-place allgather: input is at rank's offset within the receive buffer
      actualInput = ((const uint16_t*)recvbuff) + rank * count;
      actualOutput = recvbuff;
    } else {
      // Out-of-place operation: separate input and output buffers
      actualInput = sendbuff;
      actualOutput = recvbuff;
    }
  } else if (datatype == ncclFloat) {
    // Standard NCCL allgather in-place condition: sendbuff == recvbuff + rank *
    // sendcount
    if (sendbuff == ((const float*)recvbuff) + rank * count) {
      // In-place allgather: input is at rank's offset within the receive buffer
      actualInput = ((const float*)recvbuff) + rank * count;
      actualOutput = recvbuff;
    } else {
      // Out-of-place operation: separate input and output buffers
      actualInput = sendbuff;
      actualOutput = recvbuff;
    }
  }

  // Phase 1: Quantize input data to FP8 format based on input data type
  if (datatype == ncclBfloat16) {
    // Input: count bfloat16 elements → Output: count FP8 elements (1:1 mapping)
    quantizeBF16ToFp8Kernel<<<
        kernelConfig.chunkGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(
        (const uint16_t*)actualInput,
        fp8Phase1Buffer,
        fp8CountPerRank,
        0,
        count);
  } else if (datatype == ncclFloat) {
    // Input: count float elements → Output: count FP8 elements (1:1 mapping)
    quantizeFloatToFp8Kernel<<<
        kernelConfig.chunkGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(
        (const float*)actualInput, fp8Phase1Buffer, fp8CountPerRank, 0, count);
  }

  // Phase 2: All-to-all exchange of quantized data chunks between ranks
  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < nRanks; r++) {
    if (r == rank) {
      // Copy own data directly to output position
      CUDACHECK(cudaMemcpyAsync(
          ((rccl_float8*)fp8Phase2Buffer) +
              phase2Layout.getContributionOffset(r),
          fp8Phase1Buffer,
          uniformChunkSize * sizeof(rccl_float8),
          cudaMemcpyDeviceToDevice,
          stream));
    } else {
      // Send this rank's data to rank r
      NCCLCHECK(ncclSend(
          fp8Phase1Buffer, uniformChunkSize, ncclUint8, r, comm, stream));

      // Receive rank r's contribution for gathering
      size_t rContribOffset = phase2Layout.getContributionOffset(r);
      NCCLCHECK(ncclRecv(
          ((rccl_float8*)fp8Phase2Buffer) + rContribOffset,
          uniformChunkSize,
          ncclUint8,
          r,
          comm,
          stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // Phase 3: Final dequantization from FP8 based on output data type
  if (datatype == ncclBfloat16) {
    // Input: totalFp8Count FP8 elements → Output: count*nRanks bfloat16
    // elements (1:1 mapping)
    dequantizeFp8ToBF16Kernel<<<
        kernelConfig.fullGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(
        fp8Phase2Buffer,
        (uint16_t*)actualOutput,
        totalFp8Count,
        0,
        totalFp8Count);
  } else if (datatype == ncclFloat) {
    // Input: totalFp8Count FP8 elements → Output: count*nRanks float elements
    // (1:1 mapping)
    dequantizeFp8ToFloatKernel<<<
        kernelConfig.fullGridSize,
        kernelConfig.blockSize,
        0,
        stream>>>(
        fp8Phase2Buffer, (float*)actualOutput, totalFp8Count, 0, totalFp8Count);
  }

  return ncclSuccess;
}
