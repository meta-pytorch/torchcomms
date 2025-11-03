#include "comms/ctran/algos/AllToAllvDedup/IndexMapDev.cuh"

__global__ void testIndexMapTransposeKernel(
    const int* idxMap,
    const int size,
    int* outMap,
    const int iter) {
  for (int x = 0; x < iter; x++) {
    ctran::alltoallvdedup::IndexMapDev::transpose(
        idxMap, size, outMap + x * size);
  }
}

__global__ void testIndexMapTransposeSubsetKernel(
    const int* idxMap,
    const int* subIndices,
    const int subSize,
    const int size,
    int* outMap,
    const int iter) {
  for (int x = 0; x < iter; x++) {
    ctran::alltoallvdedup::IndexMapDev::transposeSubset(
        idxMap, subIndices, subSize, outMap + x * size);
  }
}

__global__ void
testIndexMapCountKernel(const int* idxMap, const int size, int* outCount) {
  *outCount = ctran::alltoallvdedup::IndexMapDev::count(idxMap, size);
}

__global__ void testIndexMapCountMergeKernel(
    const int** idxMaps,
    const int size,
    const int numMaps,
    int* outCount) {
  *outCount =
      ctran::alltoallvdedup::IndexMapDev::countMerge(idxMaps, size, numMaps);
}

__global__ void testIndexMapTransposeSubsetWithFirstLastKernel(
    const int* idxMap,
    const int* subIndices,
    const int subSize,
    const int size,
    int* outMap,
    int* firstRecvIdx,
    int* lastRecvIdx,
    int* count) {
  *count = ctran::alltoallvdedup::IndexMapDev::transposeSubset(
      idxMap, subIndices, subSize, outMap, firstRecvIdx, lastRecvIdx);
}
