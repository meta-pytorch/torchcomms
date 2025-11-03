// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace ctran::alltoallvdedup::IndexMapDev {
/* Index map
 * Index map is used to describe mapping of block index inMap send buffer on
 * sender rank to the corresponding position inMap the receive buffer on the
 * receiver rank. The blocks sent to the receiver rank can be noncontiguous
 * in send buffer, and all sent blocks are contiguous in receiver buffer.
 *
 * Map is with format [sendIdx: recvIdx] and size of total
 * number of sendIndices. If a send index does not have a mapping position inMap
 * the receive buffer, use -1 as value. For the non -1 recvIdx in the map, they
 * are contiguous integers starting at 0.
 *
 * Example: `0:0 1:1 2:-1 3:2 4:-1` indicates send indices 0,1,3 are sent to the
 * given receiver rank, and stored as receive indices `0,1,2`. Send indices 2,4
 * are not sent.
 */

// A single thread block to transpose the index map and return the
// number of non -1 indices.
// Input:
// - inMap: Index map with format of [sendIdx: recvIdx].
// - size: size of inMap in number of key:val pairs, including values with -1.
// Output:
// - outMap: output index map with format of recvIdx : sendIdx.
// - return value: count of transposed non -1 indices.
//
// Example:
// inMap:  `0:0 1:1 2:-1 3:2 4:-1`, size: 5
// outMap: `0:0 1:1 2:3 3:-1 4:-1`, return 3
__device__ inline int
transpose(const int* inMap, const size_t size, int* outMap) {
  __shared__ int shmCount;
  for (auto i = threadIdx.x; i < size; i += blockDim.x) {
    outMap[i] = -1;
  }
  if (threadIdx.x == 0) {
    shmCount = 0;
  }
  __syncthreads();

  for (auto b = threadIdx.x; b < size; b += blockDim.x) {
    const auto idx = inMap[b];
    if (idx != -1) {
      outMap[idx] = b;
      atomicAdd(&shmCount, 1);
    }
  }
  __syncthreads();
  return shmCount;
}

// A single thread block to count the number of non -1 indices in the index map.
//
// Input:
// - inMap: Index map with format of [sendIdx: recvIdx], where sendIdx is
//          contignous integer from 0.
// - size: size of inMap in number of key:val pairs, including values with -1.
// Output:
// - return value: count of non -1 indices.
__device__ inline int count(const int* inMap, const size_t size) {
  __shared__ int maxIdx;
  if (threadIdx.x == 0) {
    maxIdx = -1;
  }
  __syncthreads();

  // Since all non -1 value indices are contignous, the last indices indicates
  // the total number of non -1 indices.
  for (auto i = threadIdx.x; i < size; i += blockDim.x) {
    auto b = size - i - 1;
    if (inMap[b] != -1) {
      atomicMax(&maxIdx, inMap[b]);
      break;
    }
  }
  __syncthreads();
  // - Worst case: all indices are -1, then we have to traverse till head.
  // maxIdx would be -1 and return 0.
  // - Average case: find the largest indices in a few
  // hops, then return maxVal + 1.
  return maxIdx + 1;
}

__device__ inline int
countMerge(const int* inMaps[], const size_t size, const int numMaps) {
  __shared__ int shmCount;
  if (threadIdx.x == 0) {
    shmCount = 0;
  }
  __syncthreads();

  for (auto b = threadIdx.x; b < size; b += blockDim.x) {
    // Find if a given index is present in any of the maps.
    for (auto mapId = 0; mapId < numMaps; mapId++) {
      // count only once for a given index
      if (inMaps[mapId][b] != -1) {
        atomicAdd(&shmCount, 1);
        break;
      }
    }
  }
  __syncthreads();
  return shmCount;
}

// A single thread block to transpose the subset of indices inMap and return the
// number of transposed non -1 indices.
//
// Input:
// - inMap: Index map with format of [sendIdx: recvIdx], where sendIdx is
//          contignous integer from 0.
// - subIndices: subset of send indices to be transposed. The specified indices
//               must exist in inMap.
// - subSize: size of subIndices. It must not exceed the size of inMap.
// Output:
// - outMap: output map with format of [relative_recvIdx: relative_sendIdx],
//           where relative_recvIdx is contignous integer from 0, and
//           relative_sendIdx is the relative position of the found sendIdx in
//           subIndices.
// - firstRecvIdx (optional): the first matching recvIdx
// - return value: count of transposed non -1 indices.
//
// Example:
// inMap:  `3:5 4:6 5:-1 6:7 7:-1 8:8`, subIndices: `4,5,6,8`, subSize: 4
// outMap: `0:0 1:2 2:3`, firstRecvIdx: 6, return 3 (found sendIndices are
//          `4,6,8`, with relative position `0,2,3`)
__device__ inline int transposeSubset(
    const int* inMap,
    const int* subIndices,
    const size_t subSize,
    int* outMap,
    int* firstRecvIdx = nullptr,
    int* lastRecvIdx = nullptr) {
  __shared__ int shmCount;
  if (threadIdx.x == 0) {
    shmCount = 0;
  }

  __shared__ int shmFirstRecvIdx;
  __shared__ int shmLastRecvIdx;
  if (threadIdx.x == 0) {
    shmFirstRecvIdx = INT_MAX;
    shmLastRecvIdx = -1;
  }
  __syncthreads();

  // Find the first matching recvIdx that is not -1,
  // so we can use it to compute the relative position of all found recvIndices.
  // E.g., `3:5 4:6 5:-1 6:7 7:-1 8:8`, subIndices: `4,5,6,8`
  // -> firstMatch: 6 (sendIdx 4)
  for (auto b = threadIdx.x; b < subSize; b += blockDim.x) {
    int idx = subIndices[b];
    if (inMap[idx] != -1) {
      atomicMin(&shmFirstRecvIdx, inMap[idx]);
      break;
    }
  }
  __syncthreads();

  int firstRecvIdx_ = shmFirstRecvIdx;

  // Loop subIndices, if match in inMap, transpose to recvIdx:sendIdx and store
  // the relative offset of both indices as in subIndices.
  // -> outMap:  0:0 1:2 2:3, count=3 ()
  for (auto b = threadIdx.x; b < subSize; b += blockDim.x) {
    int sIdx = subIndices[b];
    if (inMap[sIdx] != -1) {
      // convert to relative to first recvIdx; recvIdx are always contiguous
      const auto rIdx = inMap[sIdx] - firstRecvIdx_;
      outMap[rIdx] = b;

      if (lastRecvIdx != nullptr) {
        atomicMax(&shmLastRecvIdx, rIdx);
      }
      atomicAdd(&shmCount, 1);
    }
  }
  __syncthreads();

  // Optionally return the first matched recvIdx
  if (firstRecvIdx != nullptr) {
    *firstRecvIdx = firstRecvIdx_;
  }
  // Optionally return the last matched recvIdx
  if (lastRecvIdx != nullptr) {
    *lastRecvIdx = shmLastRecvIdx;
  }
  return shmCount;
}
} // namespace ctran::alltoallvdedup::IndexMapDev
