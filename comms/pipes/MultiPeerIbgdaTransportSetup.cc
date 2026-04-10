// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultiPeerIbgdaTransportSetup.h"

#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <glog/logging.h>

#include "comms/pipes/MultipeerIbgdaTransport.h"

namespace comms::pipes {

namespace {

// kP2pSignalCount is now in P2pIbgdaTransportState.h (shared with device code).

#define CUDA_CHECK(cmd)                                                    \
  do {                                                                     \
    cudaError_t err = (cmd);                                               \
    if (err != cudaSuccess) {                                              \
      throw std::runtime_error(                                            \
          std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + \
          __FILE__ + ":" + std::to_string(__LINE__));                      \
    }                                                                      \
  } while (0)

} // namespace

MultiPeerIbgdaTransportSetup::MultiPeerIbgdaTransportSetup(
    MultipeerIbgdaTransport& ibgdaTransport,
    int myRank,
    int nRanks,
    const MultiPeerIbgdaTransportSetupConfig& config,
    int maxChannelsPerPeer,
    cudaStream_t stream)
    : ibgdaTransport_(ibgdaTransport),
      myRank_(myRank),
      nRanks_(nRanks),
      config_(config),
      maxChannelsPerPeer_(maxChannelsPerPeer),
      stream_(stream) {
  CHECK_GT(maxChannelsPerPeer_, 0)
      << "maxChannelsPerPeer must be >= 1, got " << maxChannelsPerPeer_;
  CHECK(config_.dataBufferSize % maxChannelsPerPeer_ == 0)
      << "dataBufferSize (" << config_.dataBufferSize
      << ") must be divisible by maxChannelsPerPeer (" << maxChannelsPerPeer_
      << ") for clean staging subdivision";

  // Grouped counter layout: [send_ch0..send_chN-1 | recv_ch0..recv_chN-1]
  // per peer. Total: nRanks * maxChannelsPerPeer * 2 counters.
  const size_t counterCount =
      static_cast<size_t>(nRanks_) * maxChannelsPerPeer_ * 2;
  CUDA_CHECK(cudaMalloc(&d_iterationCounter_, counterCount * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemsetAsync(
      d_iterationCounter_, 0, counterCount * sizeof(uint64_t), stream_));
}

MultiPeerIbgdaTransportSetup::~MultiPeerIbgdaTransportSetup() {
  if (stagingBuffer_) {
    ibgdaTransport_.deregisterBuffer(stagingBuffer_);
    (void)cudaFree(stagingBuffer_);
  }
  if (signalBuffer_) {
    ibgdaTransport_.deregisterBuffer(signalBuffer_);
    (void)cudaFree(signalBuffer_);
  }
  if (d_iterationCounter_) {
    (void)cudaFree(d_iterationCounter_);
  }
}

void MultiPeerIbgdaTransportSetup::exchangeBuffers() {
  const int nPeers = nRanks_ - 1;
  const size_t perPeerStagingSize =
      static_cast<size_t>(config_.pipelineDepth) * config_.dataBufferSize;

  // 1. Allocate staging data buffers.
  // Each peer needs TWO staging regions: one for send, one for recv.
  // Layout: [send_peer0 | send_peer1 | ... | recv_peer0 | recv_peer1 | ...]
  // This avoids the data race where the sender's memcpy and the remote
  // RDMA put write to the same memory.
  const size_t totalStagingSize = perPeerStagingSize * nPeers * 2;
  CUDA_CHECK(cudaMalloc(&stagingBuffer_, totalStagingSize));
  CUDA_CHECK(cudaMemsetAsync(stagingBuffer_, 0, totalStagingSize, stream_));

  // 2. Register staging buffers with NIC
  localStagingBuf_ =
      ibgdaTransport_.registerBuffer(stagingBuffer_, totalStagingSize);

  // 3. Exchange staging buffers (COLLECTIVE) — get per-peer rkeys
  remoteStagingBufs_ = ibgdaTransport_.exchangeBuffer(localStagingBuf_);

  // 4. Allocate signal buffers (scaled by maxChannelsPerPeer)
  const size_t signalBufSize = static_cast<size_t>(nRanks_) *
      maxChannelsPerPeer_ * kP2pSignalCount * sizeof(uint64_t);
  CUDA_CHECK(cudaMalloc(&signalBuffer_, signalBufSize));
  CUDA_CHECK(cudaMemsetAsync(signalBuffer_, 0, signalBufSize, stream_));

  // 5. Register signal buffers with NIC
  localSignalBuf_ =
      ibgdaTransport_.registerBuffer(signalBuffer_, signalBufSize);

  // 6. Exchange signal buffers (COLLECTIVE)
  remoteSignalBufs_ = ibgdaTransport_.exchangeBuffer(localSignalBuf_);

  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // Pre-build host-side peer states for buildP2pTransportDevice() in
  // build_device_handle().
  h_peerStates_.resize(nRanks_);

  int peerIdx = 0;
  for (int peer = 0; peer < nRanks_; peer++) {
    if (peer == myRank_) {
      continue;
    }

    // Send staging: peerIdx-th region in the SEND half of our staging buffer.
    size_t sendStagingOffset = peerIdx * perPeerStagingSize;

    // Recv staging: peerIdx-th region in the RECV half of our staging buffer.
    // Recv half starts at nPeers * perPeerStagingSize.
    const size_t recvHalfOffset =
        static_cast<size_t>(nRanks_ - 1) * perPeerStagingSize;
    size_t recvStagingOffset = recvHalfOffset + peerIdx * perPeerStagingSize;

    // Remote staging: our RECV region within the peer's staging buffer.
    // The peer's recv half starts at nPeers * perPeerStagingSize in their
    // buffer. Within that half, our data lands at remotePeerIndex.
    int remotePeerIndex = (myRank_ < peer) ? myRank_ : (myRank_ - 1);
    size_t remoteStagingOffset =
        recvHalfOffset + remotePeerIndex * perPeerStagingSize;

    // exchangeBuffer() returns nRanks-1 entries indexed by peer index
    // (skip-self ordering), NOT by global rank. Convert rank → peer index.
    int exchIdx = (peer < myRank_) ? peer : (peer - 1);

    // Signal ID bases: channel 0's pair. Device adds channelId *
    // kP2pSignalCount.
    int localSignalId = peer * maxChannelsPerPeer_ * kP2pSignalCount;
    int remoteSignalId = myRank_ * maxChannelsPerPeer_ * kP2pSignalCount;

    CHECK_LT(exchIdx, static_cast<int>(remoteStagingBufs_.size()));
    CHECK_LT(exchIdx, static_cast<int>(remoteSignalBufs_.size()));
    const size_t channelDataBufSize =
        config_.dataBufferSize / maxChannelsPerPeer_;
    h_peerStates_[peer] = P2pIbgdaTransportState{
        .localStagingBuf = localStagingBuf_.subBuffer(sendStagingOffset),
        .remoteStagingBuf =
            remoteStagingBufs_.at(exchIdx).subBuffer(remoteStagingOffset),
        .recvStagingBuf = localStagingBuf_.subBuffer(recvStagingOffset),
        .localSignalBuf = localSignalBuf_,
        .remoteSignalBuf = remoteSignalBufs_.at(exchIdx),
        .localSignalId = localSignalId,
        .remoteSignalId = remoteSignalId,
        .dataBufferSize = config_.dataBufferSize,
        .chunkSize = config_.chunkSize,
        .pipelineDepth = config_.pipelineDepth,
        .maxChannelsPerPeer = maxChannelsPerPeer_,
        .channelDataBufferSize = channelDataBufSize,
        .channelStride =
            static_cast<size_t>(config_.pipelineDepth) * channelDataBufSize,
    };
    peerIdx++;
  }

  // Cross-rank consistency: maxChannelsPerPeer_ is derived from the same
  // compile-time autotune table (getMaxIbgdaChannelsPerPeer) on all ranks,
  // so it is guaranteed consistent. If a future change makes this value
  // rank-dependent, add a collective allReduce/allGather check here to
  // assert all ranks agree — a mismatch would cause silent signal/staging
  // offset corruption.
  LOG(INFO) << "MultiPeerIbgdaTransportSetup: rank " << myRank_
            << " exchanged buffers — staging=" << totalStagingSize
            << "B, signals=" << signalBufSize
            << ", pipelineDepth=" << config_.pipelineDepth
            << ", dataBufferSize=" << config_.dataBufferSize
            << ", maxChannelsPerPeer=" << maxChannelsPerPeer_;
}

uint64_t* MultiPeerIbgdaTransportSetup::getIterationCounter() const {
  return d_iterationCounter_;
}

const std::vector<P2pIbgdaTransportState>&
MultiPeerIbgdaTransportSetup::getHostPeerStates() const {
  return h_peerStates_;
}

} // namespace comms::pipes
