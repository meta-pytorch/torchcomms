// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/benchmarks/IbgdaBenchmark.cuh"

#include <cuda_runtime.h>

#include "comms/prims/core/ThreadGroup.cuh"

namespace comms::prims::benchmark {

// On AMD MI200/MI300, `clock64()` returns the SM core clock (~1.5-2.0 GHz,
// frequency-scaled), while `wall_clock64()` is a fixed 100 MHz monotonic
// counter. The fixture's `cyclesToUs` divides by 0.1 GHz (matches
// wall_clock64), so using `clock64()` would inflate reported latencies
// by ~16x. NVIDIA's clock64() is the SM clock that matches the fixture's
// cudaDevAttrClockRate query.
#ifdef __HIP_PLATFORM_AMD__
#define BENCHMARK_CLOCK64() wall_clock64()
#else
#define BENCHMARK_CLOCK64() clock64()
#endif

// Maximum number of peers supported by multi-peer kernels.
constexpr int kMaxPeers = 128;

// Single-shot kernel implementations for correctness verification.
// Each kernel does exactly one put + signal + counter, then waits on the
// local counter slot. No warmup, no loop.

__global__ void ibgdaPutSignalWaitCounterKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    int counterId) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport.reset_counter(counterId);
    transport.put(localBuf, remoteBuf, nbytes, signalId, 1, counterId, 1);
    transport.wait_counter(counterId, 1);
  }
}

// Batched kernel implementations - these run multiple iterations in a single
// kernel launch to exclude kernel launch overhead and use GPU cycle counters
// for accurate timing.

__global__ void ibgdaPutWaitCounterBatchKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int counterId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport.reset_counter(counterId);
    uint64_t expected = 1;

    // Counter-only put: signalId=-1 disables signaling. The put completion
    // increments the transport-owned local counter slot.
    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      transport.put(localBuf, remoteBuf, nbytes, -1, 0, counterId, 1);
      transport.wait_counter(counterId, expected);
      expected++;
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = BENCHMARK_CLOCK64();

    for (int i = 0; i < numIters; i++) {
      transport.put(localBuf, remoteBuf, nbytes, -1, 0, counterId, 1);
      transport.wait_counter(counterId, expected);
      expected++;
    }

    unsigned long long endCycle = BENCHMARK_CLOCK64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaPutFlushBatchKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // Raw write (no signal, no counter), completion observed via flush().
    for (int i = 0; i < 10; i++) {
      transport.put(localBuf, remoteBuf, nbytes, IbgdaRemoteBuffer{}, 0);
      transport.flush();
    }

    unsigned long long startCycle = BENCHMARK_CLOCK64();

    for (int i = 0; i < numIters; i++) {
      transport.put(localBuf, remoteBuf, nbytes, IbgdaRemoteBuffer{}, 0);
      transport.flush();
    }

    unsigned long long endCycle = BENCHMARK_CLOCK64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaThreadScopeMultiBlockPutFlushBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytesPerBlock,
    int numIters,
    unsigned long long* blockCycles) {
  if (threadIdx.x == 0) {
    const std::size_t offset = blockIdx.x * nbytesPerBlock;
    IbgdaLocalBuffer blockLocalBuf = localBuf.subBuffer(offset);
    IbgdaRemoteBuffer blockRemoteBuf = remoteBuf.subBuffer(offset);

    for (int i = 0; i < 10; i++) {
      transport->put(blockLocalBuf, blockRemoteBuf, nbytesPerBlock);
      transport->flush();
    }

    unsigned long long startCycle = BENCHMARK_CLOCK64();

    for (int i = 0; i < numIters; i++) {
      transport->put(blockLocalBuf, blockRemoteBuf, nbytesPerBlock);
      transport->flush();
    }

    unsigned long long endCycle = BENCHMARK_CLOCK64();
    blockCycles[blockIdx.x] = endCycle - startCycle;
  }
}

__global__ void ibgdaPutSignalWaitCounterBatchKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    int counterId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport.reset_counter(counterId);
    uint64_t expected = 1;

    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      transport.put(localBuf, remoteBuf, nbytes, signalId, 1, counterId, 1);
      transport.wait_counter(counterId, expected);
      expected++;
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = BENCHMARK_CLOCK64();

    for (int i = 0; i < numIters; i++) {
      transport.put(localBuf, remoteBuf, nbytes, signalId, 1, counterId, 1);
      transport.wait_counter(counterId, expected);
      expected++;
    }

    unsigned long long endCycle = BENCHMARK_CLOCK64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaSignalOnlyBatchKernel(
    P2pIbTransportDevice transport,
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
      transport.signal(resolvedSignalBuf, 1);
      transport.flush();
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = BENCHMARK_CLOCK64();

    for (int i = 0; i < numIters; i++) {
      transport.signal(resolvedSignalBuf, 1);
      transport.flush();
    }

    unsigned long long endCycle = BENCHMARK_CLOCK64();
    *totalCycles = endCycle - startCycle;
  }
}

// Multi-peer kernel implementations. `transports[p]` is the handle for peer p;
// it is copied to a local (a small trivially-copyable tagged-union handle) so
// device calls dispatch on the backend tag regardless of method const-ness.

__global__ void ibgdaMultiPeerSerialCounterFanOutBatchKernel(
    const P2pIbTransportDevice* transports,
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
    // Per-peer handle accessor (returns a copy of the handle for peer p).
    auto T = [&](int peerIdx) -> P2pIbTransportDevice {
      return transports[peerIdx];
    };

    // Per-peer counter slot p; each peer's completion increments its own slot.
    auto perPeerCounter = [&](int p) {
      return localCounterBuf.subBuffer(p * sizeof(uint64_t));
    };

    uint64_t expected = 1;

    // Warmup
    for (int i = 0; i < 10; i++) {
      for (int p = 0; p < numPeers; p++) {
        T(p).put(
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
        T(p).wait_counter(perPeerCounter(p), expected);
      }
      expected++;
    }

    unsigned long long startCycle = BENCHMARK_CLOCK64();

    for (int i = 0; i < numIters; i++) {
      // Fire put+signal+counter to all peers — each peer's completion
      // increments its OWN per-peer counter slot
      for (int p = 0; p < numPeers; p++) {
        T(p).put(
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
        T(p).wait_counter(perPeerCounter(p), expected);
      }
      expected++;
    }

    unsigned long long endCycle = BENCHMARK_CLOCK64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaMultiPeerCounterFanOutBatchKernel(
    const P2pIbTransportDevice* transports,
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
    auto T = [&](int peerIdx) -> P2pIbTransportDevice {
      return transports[peerIdx];
    };

    auto resolvedCounterBuf =
        localCounterBuf.subBuffer(counterId * sizeof(uint64_t));
    uint64_t expected = static_cast<uint64_t>(numPeers);

    // Warmup
    for (int i = 0; i < 10; i++) {
      for (int p = 0; p < numPeers; p++) {
        T(p).put(
            localBuf,
            remoteDataBufs[p],
            nbytes,
            remoteSignalBufs[p].subBuffer(signalId * sizeof(uint64_t)),
            1,
            resolvedCounterBuf,
            1);
      }
      // Single wait — all numPeers completions increment the same slot.
      // Any handle works (counter buf is local); use peer 0.
      T(0).wait_counter(resolvedCounterBuf, expected);
      expected += numPeers;
    }

    unsigned long long startCycle = BENCHMARK_CLOCK64();

    for (int i = 0; i < numIters; i++) {
      // Fire put+signal+counter to all peers — all completions write to
      // the SAME counter slot via fetch-add
      for (int p = 0; p < numPeers; p++) {
        T(p).put(
            localBuf,
            remoteDataBufs[p],
            nbytes,
            remoteSignalBufs[p].subBuffer(signalId * sizeof(uint64_t)),
            1,
            resolvedCounterBuf,
            1);
      }
      // O(1) wait — single counter wait until it reaches expected
      T(0).wait_counter(resolvedCounterBuf, expected);
      expected += numPeers;
    }

    unsigned long long endCycle = BENCHMARK_CLOCK64();
    *totalCycles = endCycle - startCycle;
  }
}

// Launch wrapper implementations

// Single-shot launchers for correctness verification (exactly 1
// put+signal+counter)

void launchIbgdaPutSignalSingle(
    P2pIbTransportDevice transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    int counterId,
    cudaStream_t stream) {
  ibgdaPutSignalWaitCounterKernel<<<1, 32, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, signalId, counterId);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// Batched launchers for performance measurement

void launchIbgdaPutWaitCounterBatch(
    P2pIbTransportDevice transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int counterId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaPutWaitCounterBatchKernel<<<1, 32, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, counterId, numIters, totalCycles);
}

void launchIbgdaPutFlushBatch(
    P2pIbTransportDevice transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaPutFlushBatchKernel<<<1, 32, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, numIters, totalCycles);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

void launchIbgdaThreadScopeMultiBlockPutFlushBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytesPerBlock,
    int numBlocks,
    int numIters,
    unsigned long long* blockCycles,
    cudaStream_t stream) {
  ibgdaThreadScopeMultiBlockPutFlushBatchKernel<<<numBlocks, 32, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytesPerBlock, numIters, blockCycles);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

void launchIbgdaPutSignalWaitCounterBatch(
    P2pIbTransportDevice transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    int counterId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaPutSignalWaitCounterBatchKernel<<<1, 32, 0, stream>>>(
      transport,
      localBuf,
      remoteBuf,
      nbytes,
      signalId,
      counterId,
      numIters,
      totalCycles);
}

void launchIbgdaSignalOnlyBatch(
    P2pIbTransportDevice transport,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaSignalOnlyBatchKernel<<<1, 32, 0, stream>>>(
      transport, remoteSignalBuf, signalId, numIters, totalCycles);
}

void launchMultiPeerSerialCounterFanOutBatch(
    const P2pIbTransportDevice* transports,
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
      transports,
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
    const P2pIbTransportDevice* transports,
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
      transports,
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

} // namespace comms::prims::benchmark
