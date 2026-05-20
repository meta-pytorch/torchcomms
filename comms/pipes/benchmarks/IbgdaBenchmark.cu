// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/IbgdaBenchmark.cuh"

#include <cuda_runtime.h>

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes::benchmark {

// Maximum number of peers supported by multi-peer kernels.
constexpr int kMaxPeers = 128;

// Single-shot kernel implementations for correctness verification.
// Each kernel does exactly one put + signal + counter, then waits on the
// local counter slot. No warmup, no loop.

__global__ void ibgdaPutSignalWaitLocalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    IbgdaLocalBuffer localCounterBuf,
    int counterId) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto resolvedSignalBuf =
        remoteSignalBuf.subBuffer(signalId * sizeof(uint64_t));
    auto resolvedCounterBuf =
        localCounterBuf.subBuffer(counterId * sizeof(uint64_t));
    transport->put(
        localBuf,
        remoteBuf,
        nbytes,
        resolvedSignalBuf,
        1,
        resolvedCounterBuf,
        1);
    transport->wait_counter(resolvedCounterBuf, 1);
  }
}

// Batched kernel implementations - these run multiple iterations in a single
// kernel launch to exclude kernel launch overhead and use GPU cycle counters
// for accurate timing.

__global__ void ibgdaPutWaitLocalBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto resolvedCounterBuf =
        localCounterBuf.subBuffer(counterId * sizeof(uint64_t));
    uint64_t expected = 1;

    // Counter-only put: signalBuf={} (ptr==nullptr disables signaling), the
    // companion QP loopback atomically increments the local counter when the
    // put completes at the NIC. GPU waits on the local counter slot.
    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      transport->put(
          localBuf,
          remoteBuf,
          nbytes,
          IbgdaRemoteBuffer{},
          0,
          resolvedCounterBuf,
          1);
      transport->wait_counter(resolvedCounterBuf, expected);
      expected++;
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      transport->put(
          localBuf,
          remoteBuf,
          nbytes,
          IbgdaRemoteBuffer{},
          0,
          resolvedCounterBuf,
          1);
      transport->wait_counter(resolvedCounterBuf, expected);
      expected++;
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaPutSignalWaitLocalBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto resolvedSignalBuf =
        remoteSignalBuf.subBuffer(signalId * sizeof(uint64_t));
    auto resolvedCounterBuf =
        localCounterBuf.subBuffer(counterId * sizeof(uint64_t));
    uint64_t expected = 1;

    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      transport->put(
          localBuf,
          remoteBuf,
          nbytes,
          resolvedSignalBuf,
          1,
          resolvedCounterBuf,
          1);
      transport->wait_counter(resolvedCounterBuf, expected);
      expected++;
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      transport->put(
          localBuf,
          remoteBuf,
          nbytes,
          resolvedSignalBuf,
          1,
          resolvedCounterBuf,
          1);
      transport->wait_counter(resolvedCounterBuf, expected);
      expected++;
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaSignalOnlyBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto resolvedSignalBuf =
        remoteSignalBuf.subBuffer(signalId * sizeof(uint64_t));

    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      transport->signal(resolvedSignalBuf, 1);
      transport->flush();
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      transport->signal(resolvedSignalBuf, 1);
      transport->flush();
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

// Multi-peer kernel implementations

__global__ void ibgdaMultiPeerSerialCounterFanOutBatchKernel(
    P2pIbgdaTransportDevice* transportsBase,
    std::size_t transportStride,
    int numPeers,
    IbgdaLocalBuffer localBuf,
    const IbgdaRemoteBuffer* remoteDataBufs,
    std::size_t nbytes,
    const IbgdaRemoteBuffer* remoteSignalBufs,
    int signalId,
    IbgdaLocalBuffer localCounterBuf,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // Helper to index into transport array by byte stride
    auto getTransport = [&](int peerIdx) -> P2pIbgdaTransportDevice* {
      return reinterpret_cast<P2pIbgdaTransportDevice*>(
          reinterpret_cast<char*>(transportsBase) + peerIdx * transportStride);
    };

    // Per-peer counter slot p; each companion QP increments its own slot.
    auto perPeerCounter = [&](int p) {
      return localCounterBuf.subBuffer(p * sizeof(uint64_t));
    };

    uint64_t expected = 1;

    // Warmup
    for (int i = 0; i < 10; i++) {
      for (int p = 0; p < numPeers; p++) {
        getTransport(p)->put(
            localBuf,
            remoteDataBufs[p],
            nbytes,
            remoteSignalBufs[p].subBuffer(signalId * sizeof(uint64_t)),
            1,
            perPeerCounter(p),
            1);
      }
      // O(N) waits — one wait_counter per peer
      for (int p = 0; p < numPeers; p++) {
        getTransport(p)->wait_counter(perPeerCounter(p), expected);
      }
      expected++;
    }

    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      // Fire put+signal+counter to all peers — each peer's companion QP
      // increments its OWN per-peer counter slot
      for (int p = 0; p < numPeers; p++) {
        getTransport(p)->put(
            localBuf,
            remoteDataBufs[p],
            nbytes,
            remoteSignalBufs[p].subBuffer(signalId * sizeof(uint64_t)),
            1,
            perPeerCounter(p),
            1);
      }
      // O(N) waits — one wait_counter per peer
      for (int p = 0; p < numPeers; p++) {
        getTransport(p)->wait_counter(perPeerCounter(p), expected);
      }
      expected++;
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaMultiPeerCounterFanOutBatchKernel(
    P2pIbgdaTransportDevice* transportsBase,
    std::size_t transportStride,
    int numPeers,
    IbgdaLocalBuffer localBuf,
    const IbgdaRemoteBuffer* remoteDataBufs,
    std::size_t nbytes,
    const IbgdaRemoteBuffer* remoteSignalBufs,
    int signalId,
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto getTransport = [&](int peerIdx) -> P2pIbgdaTransportDevice* {
      return reinterpret_cast<P2pIbgdaTransportDevice*>(
          reinterpret_cast<char*>(transportsBase) + peerIdx * transportStride);
    };

    auto resolvedCounterBuf =
        localCounterBuf.subBuffer(counterId * sizeof(uint64_t));
    uint64_t expected = static_cast<uint64_t>(numPeers);

    // Warmup
    for (int i = 0; i < 10; i++) {
      for (int p = 0; p < numPeers; p++) {
        getTransport(p)->put(
            localBuf,
            remoteDataBufs[p],
            nbytes,
            remoteSignalBufs[p].subBuffer(signalId * sizeof(uint64_t)),
            1,
            resolvedCounterBuf,
            1);
      }
      // Single wait — all numPeers companion QPs increment the same slot.
      // Any transport works (counter buf is local); use peer 0.
      getTransport(0)->wait_counter(resolvedCounterBuf, expected);
      expected += numPeers;
    }

    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      // Fire put+signal+counter to all peers — all companion QPs write to
      // the SAME counter slot via loopback atomic fetch-add
      for (int p = 0; p < numPeers; p++) {
        getTransport(p)->put(
            localBuf,
            remoteDataBufs[p],
            nbytes,
            remoteSignalBufs[p].subBuffer(signalId * sizeof(uint64_t)),
            1,
            resolvedCounterBuf,
            1);
      }
      // O(1) wait — single counter wait until it reaches expected
      getTransport(0)->wait_counter(resolvedCounterBuf, expected);
      expected += numPeers;
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

// Launch wrapper implementations

// Single-shot launchers for correctness verification (exactly 1
// put+signal+counter)

void launchIbgdaPutSignalSingle(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    const IbgdaLocalBuffer& localCounterBuf,
    int counterId,
    cudaStream_t stream) {
  ibgdaPutSignalWaitLocalKernel<<<1, 32, 0, stream>>>(
      transport,
      localBuf,
      remoteBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      localCounterBuf,
      counterId);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// Batched launchers for performance measurement

void launchIbgdaPutWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaLocalBuffer& localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaPutWaitLocalBatchKernel<<<1, 32, 0, stream>>>(
      transport,
      localBuf,
      remoteBuf,
      nbytes,
      localCounterBuf,
      counterId,
      numIters,
      totalCycles);
}

void launchIbgdaPutSignalWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    const IbgdaLocalBuffer& localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaPutSignalWaitLocalBatchKernel<<<1, 32, 0, stream>>>(
      transport,
      localBuf,
      remoteBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      localCounterBuf,
      counterId,
      numIters,
      totalCycles);
}

void launchIbgdaSignalOnlyBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaSignalOnlyBatchKernel<<<1, 32, 0, stream>>>(
      transport, remoteSignalBuf, signalId, numIters, totalCycles);
}

void launchMultiPeerSerialCounterFanOutBatch(
    P2pIbgdaTransportDevice* transportsBase,
    std::size_t transportStride,
    int numPeers,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer* remoteDataBufs,
    std::size_t nbytes,
    const IbgdaRemoteBuffer* remoteSignalBufs,
    int signalId,
    const IbgdaLocalBuffer& localCounterBuf,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaMultiPeerSerialCounterFanOutBatchKernel<<<1, 32, 0, stream>>>(
      transportsBase,
      transportStride,
      numPeers,
      localBuf,
      remoteDataBufs,
      nbytes,
      remoteSignalBufs,
      signalId,
      localCounterBuf,
      numIters,
      totalCycles);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

void launchMultiPeerCounterFanOutBatch(
    P2pIbgdaTransportDevice* transportsBase,
    std::size_t transportStride,
    int numPeers,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer* remoteDataBufs,
    std::size_t nbytes,
    const IbgdaRemoteBuffer* remoteSignalBufs,
    int signalId,
    const IbgdaLocalBuffer& localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaMultiPeerCounterFanOutBatchKernel<<<1, 32, 0, stream>>>(
      transportsBase,
      transportStride,
      numPeers,
      localBuf,
      remoteDataBufs,
      nbytes,
      remoteSignalBufs,
      signalId,
      localCounterBuf,
      counterId,
      numIters,
      totalCycles);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

} // namespace comms::pipes::benchmark
