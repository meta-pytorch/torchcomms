/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "p2p_allgather.h"
#include "comm.h"

/**
 * P2P AllGather implementation optimized for single node 8 rank systems.
 * Uses direct broadcast approach with concurrent send/receive operations for
 * optimal bandwidth utilization.
 */
HOT ncclResult_t ncclP2PAllGather(
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

  // P2P operations are data type agnostic - they just copy data
  size_t elementSize = ncclTypeSize(datatype);

  // Handle in-place vs out-of-place buffer addressing correctly for allgather
  const void* actualInput;
  void* actualOutput;

  // Standard NCCL allgather in-place condition: sendbuff == recvbuff + rank *
  // sendcount * elementSize
  if (sendbuff == ((const char*)recvbuff) + rank * count * elementSize) {
    // In-place allgather: input is at rank's offset within the receive buffer
    actualInput = ((const char*)recvbuff) + rank * count * elementSize;
    actualOutput = recvbuff;
  } else {
    // Out-of-place operation: separate input and output buffers
    actualInput = sendbuff;
    actualOutput = recvbuff;
  }

  // Optimized P2P AllGather for single node systems
  // Direct broadcast approach is optimal for ≤8 ranks with high-bandwidth
  // interconnects
  NCCLCHECK(ncclGroupStart());

  // Handle self-copy first (most efficient as it's just a memory copy)
  size_t myOffsetBytes = rank * count * elementSize;
  CUDACHECK(cudaMemcpyAsync(
      (char*)actualOutput + myOffsetBytes,
      actualInput,
      count * elementSize,
      cudaMemcpyDeviceToDevice,
      stream));

  // Direct P2P broadcast/gather - optimal for single node with ≤8 ranks
  // Each rank broadcasts its data while simultaneously gathering others' data
  for (int r = 0; r < nRanks; r++) {
    if (r == rank)
      continue; // Already handled self-case above

    // Send my data to rank r (I'm broadcasting to everyone)
    NCCLCHECK(ncclSend(actualInput, count, datatype, r, comm, stream));

    // Receive rank r's data directly to final output position
    size_t rOffsetBytes = r * count * elementSize;
    NCCLCHECK(ncclRecv(
        (char*)actualOutput + rOffsetBytes, count, datatype, r, comm, stream));
  }

  NCCLCHECK(ncclGroupEnd());

  return ncclSuccess;
}
