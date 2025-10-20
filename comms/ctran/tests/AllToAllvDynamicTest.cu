// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <assert.h>
#include <stdio.h>
#include <cstddef>

/* numblocks = numRanks or numSendSplitLengths*/
__global__ void initializeDataBuffersKernel(
    size_t maxCount,
    int** sendbuffs,
    int** recvbuffs,
    size_t* sendcountsDev,
    int maxTotalExperts,
    int numRanks) {
  auto r = blockIdx.x;
  if (r < numRanks) {
    for (size_t i = threadIdx.x; i < maxCount * maxTotalExperts;
         i += blockDim.x) {
      recvbuffs[r][i] = -1;
    }
  }
  for (size_t i = threadIdx.x; i < sendcountsDev[r]; i += blockDim.x) {
    sendbuffs[r][i] = r + i;
  }
}

/* numblocks = numRanks * numExperts */
__global__ void initializeBufferPtrKernel(
    size_t maxCount,
    int* sendbuff,
    int** sendbuffs,
    size_t* sendSplitLengthsDev) {
  auto r = blockIdx.x;
  long cur_index = 0;
  for (size_t i = 0; i < r; i++) {
    cur_index += sendSplitLengthsDev[i];
  }
  sendbuffs[r] = sendbuff + cur_index;
}

/* numblocks = numRanks */
__global__ void checkDataBuffersKernel(
    size_t maxCount,
    size_t* counts,
    int globalRank,
    int** recvbuffs) {
  auto r = blockIdx.x;
  for (size_t i = threadIdx.x; i < maxCount; i += blockDim.x) {
    if (i < counts[r]) {
      if (recvbuffs[r][i] != globalRank + i) {
        printf(
            "recvbuffs[%d][%lu]=%d, expected=%lu\n",
            r,
            i,
            recvbuffs[r][i],
            globalRank + i);
        asm("trap;");
      }
    } else {
      if (recvbuffs[r][i] != -1) {
        printf("recvbuffs[%d][%lu]=%d, expected=-1\n", r, i, recvbuffs[r][i]);
        asm("trap;");
      }
    }
  }
}

/* numblocks = numRanks */
__global__ void checkDataBuffersNonContigKernel(
    size_t maxCount,
    int maxNumExperts,
    size_t* recvSplits,
    size_t* recvIndices,
    size_t* recvIndicesBlockLengths,
    size_t numSendSplitLengths,
    int** recvbuffs,
    int globalRank) {
  // Check the data received from rank r
  auto r = blockIdx.x;

  auto curRecvIndicesPos = 0, lastIndex = 0, curRecvbuffOffset = 0;
  for (int i = 0; i < r; i++) {
    curRecvIndicesPos += recvIndicesBlockLengths[i];
  }
  for (int i = 0; i < recvIndicesBlockLengths[r] + 1; i++) {
    if (i == recvIndicesBlockLengths[r]) {
      for (size_t k = curRecvbuffOffset; k < maxCount * maxNumExperts; k++) {
        if (recvbuffs[r][k] != -1) {
          printf("recvbuffs[%d][%lu]=%d, expected=-1\n", r, k, recvbuffs[r][k]);
          asm("trap;");
        }
      }
    } else {
      auto curIndex = recvIndices[curRecvIndicesPos + i];
      for (int j = lastIndex; j < curIndex; j++) {
        for (size_t k = curRecvbuffOffset;
             k < curRecvbuffOffset + recvSplits[r * numSendSplitLengths + j];
             k++) {
          if (recvbuffs[r][k] != -1) {
            printf(
                "recvbuffs[%d][%lu]=%d, expected=-1\n", r, k, recvbuffs[r][k]);
            asm("trap;");
          }
        }
        curRecvbuffOffset += recvSplits[r * numSendSplitLengths + j];
      }

      for (size_t k = curRecvbuffOffset; k <
           curRecvbuffOffset + recvSplits[r * numSendSplitLengths + curIndex];
           k++) {
        if (recvbuffs[r][k] != curIndex + k - curRecvbuffOffset) {
          printf(
              "recvbuffs[%d][%lu]=%d, expected=%lu\n",
              r,
              k,
              recvbuffs[r][k],
              curIndex + k - curRecvbuffOffset);
          asm("trap;");
        }
      }
      curRecvbuffOffset += recvSplits[r * numSendSplitLengths + curIndex];
      lastIndex = curIndex + 1;
    }
  }
}

/* numblocks = 1, threadsPerBlock = numRanks * numExperts */
__global__ void equalCountsKernel(size_t* sendCounts, size_t count) {
  sendCounts[threadIdx.x] = count;
}

/* numblocks = 1, threadsPerBlock = numRanks or numExperts (non-contig)  */
__global__ void checkEqualCountsKernel(size_t* recvCounts, size_t count) {
  if (recvCounts[threadIdx.x] != count) {
    printf(
        "recvCounts[%d]=%lu, expected=%lu\n",
        threadIdx.x,
        recvCounts[threadIdx.x],
        count);
    asm("trap;");
  }
}

/* numblocks = 1, threadsPerBlock = numRanks * numExperts */
__global__ void randomCountsKernel(
    size_t* sendCounts,
    size_t* randomCountsMatrix,
    int globalRank,
    int numRanks) {
  size_t x = globalRank * blockDim.x + threadIdx.x;
  sendCounts[threadIdx.x] = randomCountsMatrix[x];
}

/* numblocks = 1, threadsPerBlock = numRanks */
__global__ void checkRandomCountsKernel(
    size_t* recvCounts,
    size_t* randomCountsMatrix,
    int globalRank,
    int numRanks) {
  size_t x = threadIdx.x * numRanks + globalRank;
  if (recvCounts[threadIdx.x] != randomCountsMatrix[x]) {
    printf(
        "recvCounts[%d]=%lu, expected=%lu\n",
        threadIdx.x,
        recvCounts[threadIdx.x],
        randomCountsMatrix[x]);
    asm("trap;");
  }
}

/* numblocks = numRanks, threadsPerBlock = numSendSplitLengths */
__global__ void checkRandomCountsNonContigKernel(
    size_t* recvSplits,
    size_t* randomCountsMatrix,
    size_t numSendSplitLengths,
    int numRanks,
    int maxNumExperts) {
  auto r = blockIdx.x, i = threadIdx.x;
  if (recvSplits[r * numSendSplitLengths + i] !=
      randomCountsMatrix[r * numRanks * maxNumExperts + i]) {
    printf(
        "recvSplits[%d][%d]=%lu, expected=%lu, actual index: %lu\n",
        r,
        i,
        recvSplits[r * numSendSplitLengths + i],
        randomCountsMatrix[r * numRanks * maxNumExperts + i],
        r * numSendSplitLengths + i);
    asm("trap;");
  }
}

/* numblocks = numRanks, threadsPerBlock =
 * sendIndicesBlockLengthsHost[globalRank] */
__global__ void initRecvIndicesKernel(
    size_t* recvIndices,
    size_t* recvIndicesBlockLengths,
    size_t* sendIndices,
    size_t curSendIndicesPos) {
  auto r = blockIdx.x, i = threadIdx.x;
  size_t curRecvIndicesPos = 0;
  for (int j = 0; j < r; j++) {
    curRecvIndicesPos += recvIndicesBlockLengths[j];
  }
  recvIndices[curRecvIndicesPos + i] = sendIndices[curSendIndicesPos + i];
}

/* numblocks = numRanks, threadsPerBlock = 1 */
__global__ void initRecvIndicesBlockLengthKernel(
    size_t* recvIndicesBlockLengths,
    size_t myIndicesBlockLengths) {
  auto r = blockIdx.x;
  recvIndicesBlockLengths[r] = myIndicesBlockLengths;
}
