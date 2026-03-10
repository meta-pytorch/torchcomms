// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/IbgdaBenchmark.cuh"

#include <cuda_runtime.h>

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes::benchmark {

// Maximum number of peers supported by multi-peer kernels.
constexpr int kMaxPeers = 128;

// Single-shot kernel implementations for correctness verification.
// Each kernel does exactly one put + signal + wait_local, no warmup, no loop.

__global__ void ibgdaPutSignalWaitLocalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto work = transport->put_signal(
        localBuf, remoteBuf, nbytes, remoteSignalBuf, signalId, 1);
    transport->wait_local(work);
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
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      auto work = transport->put(localBuf, remoteBuf, nbytes);
      transport->wait_local(work);
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      auto work = transport->put(localBuf, remoteBuf, nbytes);
      transport->wait_local(work);
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
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      auto work = transport->put_signal(
          localBuf, remoteBuf, nbytes, remoteSignalBuf, signalId, 1);
      transport->wait_local(work);
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      auto work = transport->put_signal(
          localBuf, remoteBuf, nbytes, remoteSignalBuf, signalId, 1);
      transport->wait_local(work);
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
    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      auto work = transport->signal_remote(remoteSignalBuf, signalId, 1);
      transport->wait_local(work);
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      auto work = transport->signal_remote(remoteSignalBuf, signalId, 1);
      transport->wait_local(work);
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaPutCqPollWaitBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      auto work = transport->put(localBuf, remoteBuf, nbytes);
      transport->wait_local(work);
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      auto work = transport->put(localBuf, remoteBuf, nbytes);
      transport->wait_local(work);
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaPutSignalCounterBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    volatile uint64_t* ctr =
        static_cast<volatile uint64_t*>(localCounterBuf.ptr) + counterId;
    uint64_t expected = 1;

    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      transport->put_signal_counter_remote(
          localDataBuf,
          remoteDataBuf,
          nbytes,
          remoteSignalBuf,
          signalId,
          1,
          localCounterBuf,
          counterId,
          1);
      while (*ctr < expected) {
      }
      expected++;
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      transport->put_signal_counter_remote(
          localDataBuf,
          remoteDataBuf,
          nbytes,
          remoteSignalBuf,
          signalId,
          1,
          localCounterBuf,
          counterId,
          1);
      while (*ctr < expected) {
      }
      expected++;
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

// Multi-peer kernel implementations

__global__ void ibgdaMultiPeerCqPollFanOutBatchKernel(
    P2pIbgdaTransportDevice* transportsBase,
    std::size_t transportStride,
    int numPeers,
    IbgdaLocalBuffer localBuf,
    const IbgdaRemoteBuffer* remoteDataBufs,
    std::size_t nbytes,
    const IbgdaRemoteBuffer* remoteSignalBufs,
    int signalId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // Helper to index into transport array by byte stride
    auto getTransport = [&](int peerIdx) -> P2pIbgdaTransportDevice* {
      return reinterpret_cast<P2pIbgdaTransportDevice*>(
          reinterpret_cast<char*>(transportsBase) + peerIdx * transportStride);
    };

    // Warmup
    IbgdaWork works[kMaxPeers];
    for (int i = 0; i < 10; i++) {
      for (int p = 0; p < numPeers; p++) {
        getTransport(p)->put(localBuf, remoteDataBufs[p], nbytes);
        works[p] =
            getTransport(p)->signal_remote(remoteSignalBufs[p], signalId, 1);
      }
      for (int p = 0; p < numPeers; p++) {
        getTransport(p)->wait_local(works[p]);
      }
    }

    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      // Fire put + signal to all peers (same main QP work as counter path)
      for (int p = 0; p < numPeers; p++) {
        getTransport(p)->put(localBuf, remoteDataBufs[p], nbytes);
        works[p] =
            getTransport(p)->signal_remote(remoteSignalBufs[p], signalId, 1);
      }
      // Poll CQ at each signal's ticket — O(N) CQ polls, no extra WQEs
      for (int p = 0; p < numPeers; p++) {
        getTransport(p)->wait_local(works[p]);
      }
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

    volatile uint64_t* ctr =
        static_cast<volatile uint64_t*>(localCounterBuf.ptr) + counterId;
    uint64_t expected = static_cast<uint64_t>(numPeers);

    // Warmup
    for (int i = 0; i < 10; i++) {
      for (int p = 0; p < numPeers; p++) {
        getTransport(p)->put_signal_counter_remote(
            localBuf,
            remoteDataBufs[p],
            nbytes,
            remoteSignalBufs[p],
            signalId,
            1,
            localCounterBuf,
            counterId,
            1);
      }
      // Single poll — wait for all numPeers companion QPs to increment
      while (*ctr < expected) {
      }
      expected += numPeers;
    }

    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      // Fire put+signal+counter to all peers — all companion QPs write to
      // the SAME counter slot via loopback atomic fetch-add
      for (int p = 0; p < numPeers; p++) {
        getTransport(p)->put_signal_counter_remote(
            localBuf,
            remoteDataBufs[p],
            nbytes,
            remoteSignalBufs[p],
            signalId,
            1,
            localCounterBuf,
            counterId,
            1);
      }
      // O(1) poll — single volatile read until counter reaches expected
      while (*ctr < expected) {
      }
      expected += numPeers;
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

// Launch wrapper implementations

// Single-shot launchers for correctness verification (exactly 1 put+signal)

void launchIbgdaPutSignalSingle(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    cudaStream_t stream) {
  ibgdaPutSignalWaitLocalKernel<<<1, 32, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, remoteSignalBuf, signalId);
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
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaPutWaitLocalBatchKernel<<<1, 32, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, numIters, totalCycles);
}

void launchIbgdaPutSignalWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
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

void launchIbgdaPutCqPollWaitBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaPutCqPollWaitBatchKernel<<<1, 32, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, numIters, totalCycles);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

void launchIbgdaPutSignalCounterBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localDataBuf,
    const IbgdaRemoteBuffer& remoteDataBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    const IbgdaLocalBuffer& localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaPutSignalCounterBatchKernel<<<1, 32, 0, stream>>>(
      transport,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
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

void launchMultiPeerCqPollFanOutBatch(
    P2pIbgdaTransportDevice* transportsBase,
    std::size_t transportStride,
    int numPeers,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer* remoteDataBufs,
    std::size_t nbytes,
    const IbgdaRemoteBuffer* remoteSignalBufs,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaMultiPeerCqPollFanOutBatchKernel<<<1, 32, 0, stream>>>(
      transportsBase,
      transportStride,
      numPeers,
      localBuf,
      remoteDataBufs,
      nbytes,
      remoteSignalBufs,
      signalId,
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
