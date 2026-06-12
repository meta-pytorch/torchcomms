// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/MultiPeerIbTransport.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <fmt/core.h>
#include <glog/logging.h>

#include "comms/ctran/ibverbx/IbverbxSymbols.h"
// GPU DMA-BUF export for MR registration. Generic (no DOCA context): on NVIDIA
// it is cuMemGetHandleForAddressRange via DocaHostUtils (with the CUDA driver
// address-range lookup from CudaDriverLazy); on AMD it is the HSA path provided
// through DocaCompat.
#ifdef __HIP_PLATFORM_AMD__
#include "comms/prims/transport/amd/DocaCompat.h"
#else
#include "comms/prims/platform/CudaDriverLazy.h"
#include "comms/prims/platform/DocaHostUtils.h"
#endif

namespace comms::prims {

MultiPeerIbTransportBase::MultiPeerIbTransportBase(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    MultipeerIbTransportConfig config)
    : myRank_(myRank),
      nRanks_(nRanks),
      bootstrap_(std::move(bootstrap)),
      config_(std::move(config)) {
  if (myRank_ < 0 || myRank_ >= nRanks_) {
    throw std::invalid_argument("Invalid rank");
  }
  if (nRanks_ < 2) {
    throw std::invalid_argument("Need at least 2 ranks");
  }
}

IbgdaLocalBuffer MultiPeerIbTransportBase::registerBuffer(
    void* ptr,
    std::size_t size) {
  if (ptr == nullptr || size == 0) {
    throw std::invalid_argument("Invalid buffer pointer or size");
  }

  // Fast path: containment lookup — if [ptr, ptr+size) falls entirely within an
  // existing registration, return the cached per-NIC lkeys with no driver call.
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = registeredBuffers_.upper_bound(addr);
  if (it != registeredBuffers_.begin()) {
    --it;
    if (addr + size <= it->first + it->second.allocSize) {
      it->second.refs++;
      VLOG(1) << "MultiPeerIbTransport: cache hit for ptr=" << ptr
              << " allocBase=0x" << std::hex << it->first << std::dec
              << " refs=" << it->second.refs;
      NetworkLKeys keys(numNics_);
      for (int n = 0; n < numNics_; ++n) {
        keys[n] = NetworkLKey(HostLKey(it->second.mrs[n]->lkey));
      }
      return IbgdaLocalBuffer(ptr, keys);
    }
  }

  // Cache miss: resolve the GPU allocation base and register one MR per NIC
  // (DMABUF-first, ibv_reg_mr fallback). Generic — no DOCA, no backend hook.
#ifdef __HIP_PLATFORM_AMD__
  // HIP doesn't expose an exact cuMemGetAddressRange equivalent; for the common
  // case where the caller passes the allocation base, register the requested
  // range.
  uintptr_t allocBase = reinterpret_cast<uintptr_t>(ptr);
  std::size_t allocSize = size;
#else
  if (cuda_driver_lazy_init() != 0 || pfn_cuMemGetAddressRange == nullptr) {
    throw std::runtime_error(
        "registerBuffer: failed to initialize CUDA driver API");
  }
  CUdeviceptr allocBase = 0;
  std::size_t allocSize = 0;
  CUresult cuRes =
      pfn_cuMemGetAddressRange(&allocBase, &allocSize, (CUdeviceptr)ptr);
  if (cuRes != CUDA_SUCCESS || allocBase == 0) {
    throw std::runtime_error(
        "registerBuffer: cuMemGetAddressRange failed for ptr");
  }
#endif
  auto& symbols = ibverbx::ibvSymbols;
  int accessFlags = ibverbx::IBV_ACCESS_LOCAL_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_WRITE | ibverbx::IBV_ACCESS_REMOTE_READ |
      ibverbx::IBV_ACCESS_REMOTE_ATOMIC;

  CachedMr cached;
  cached.allocSize = allocSize;
  cached.refs = 1;

  // Try DMABUF first per NIC, fall back to plain reg_mr per NIC. If any NIC's
  // registration fails, deregister everything already registered and propagate.
  for (int n = 0; n < numNics_; ++n) {
    ibverbx::ibv_mr* mr = nullptr;
    auto dmabuf = export_gpu_dmabuf_aligned(
        reinterpret_cast<void*>(allocBase), allocSize);
    if (dmabuf) {
      if (symbols.ibv_internal_reg_dmabuf_mr != nullptr) {
        mr = symbols.ibv_internal_reg_dmabuf_mr(
            nics_[n].ibvPd,
            dmabuf->alignment.dmabufOffset,
            allocSize,
            static_cast<uint64_t>(allocBase),
            dmabuf->fd,
            accessFlags);
      }
      close(dmabuf->fd);
    }
    if (!mr) {
      errno = 0;
      mr = symbols.ibv_internal_reg_mr(
          nics_[n].ibvPd,
          reinterpret_cast<void*>(allocBase),
          allocSize,
          accessFlags);
      if (!mr) {
        const int savedErrno = errno;
        for (int j = 0; j < n; ++j) {
          symbols.ibv_internal_dereg_mr(cached.mrs[j]);
        }
        throw std::runtime_error(
            fmt::format(
                "Failed to register buffer with RDMA on NIC {} "
                "(allocBase=0x{:x} allocSize={} errno={} ({}))",
                n,
                allocBase,
                allocSize,
                savedErrno,
                std::strerror(savedErrno)));
      }
    }
    cached.mrs[n] = mr;
  }

  registeredBuffers_.emplace(static_cast<uintptr_t>(allocBase), cached);

  NetworkLKeys keys(numNics_);
  for (int n = 0; n < numNics_; ++n) {
    keys[n] = NetworkLKey(HostLKey(cached.mrs[n]->lkey));
  }
  return IbgdaLocalBuffer(ptr, keys);
}

void MultiPeerIbTransportBase::deregisterBuffer(void* ptr) {
  // Containment lookup on the ordered map avoids resolving the allocation range
  // again (which fails once the underlying memory is freed).
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = registeredBuffers_.upper_bound(addr);
  if (it != registeredBuffers_.begin()) {
    --it;
    if (addr < it->first + it->second.allocSize) {
      it->second.refs--;
      VLOG(1) << "MultiPeerIbTransport: deregister ptr=" << ptr
              << " allocBase=0x" << std::hex << it->first << std::dec
              << " refs=" << it->second.refs;
      if (it->second.refs <= 0) {
        // Deregistration is backend-agnostic (no PD/DOCA needed).
        for (int n = 0; n < numNics_; ++n) {
          ibverbx::ibvSymbols.ibv_internal_dereg_mr(it->second.mrs[n]);
        }
        registeredBuffers_.erase(it);
      }
      return;
    }
  }
  LOG(WARNING) << "MultiPeerIbTransport: buffer not registered: " << ptr;
}

std::vector<IbgdaRemoteBuffer> MultiPeerIbTransportBase::exchangeBuffer(
    const IbgdaLocalBuffer& localBuf) {
  const int numPeers = nRanks_ - 1;

  // Containment lookup (same as deregisterBuffer): find the registered
  // allocation covering localBuf.ptr. Avoids re-resolving the allocation base;
  // sub-buffers resolve correctly via the ordered map.
  const auto addr = reinterpret_cast<uintptr_t>(localBuf.ptr);
  auto it = registeredBuffers_.upper_bound(addr);
  if (it == registeredBuffers_.begin()) {
    throw std::runtime_error(
        "Buffer not registered - call registerBuffer() first");
  }
  --it;
  if (addr >= it->first + it->second.allocSize) {
    throw std::runtime_error(
        "Buffer not registered - call registerBuffer() first");
  }

  // allGather addr + per-NIC rkeys; one entry per rank.
  std::vector<IbgdaBufferExchInfo> allInfo(nRanks_);
  allInfo[myRank_].addr = reinterpret_cast<uint64_t>(localBuf.ptr);
  allInfo[myRank_].numNics = numNics_;
  for (int n = 0; n < numNics_; ++n) {
    allInfo[myRank_].rkey_per_device[n] = HostRKey(it->second.mrs[n]->rkey);
  }

  auto result =
      bootstrap_
          ->allGather(
              allInfo.data(), sizeof(IbgdaBufferExchInfo), myRank_, nRanks_)
          .get();
  if (result != 0) {
    throw std::runtime_error(
        "MultiPeerIbTransport::exchangeBuffer allGather failed");
  }

  std::vector<IbgdaRemoteBuffer> peerBuffers(numPeers);
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    const int peerRank = peerIndexToRank(peerIndex);
    peerBuffers[peerIndex] = allInfo[peerRank].toRemoteBuffer();
  }

  VLOG(1) << "MultiPeerIbTransport: exchanged buffer info with " << numPeers
          << " peers";
  return peerBuffers;
}

bool MultiPeerIbTransportBase::isPeerMaterialized(int peerRank) const {
  if (peerRank == myRank_ || peerRank < 0 || peerRank >= nRanks_) {
    throw std::invalid_argument(
        fmt::format(
            "isPeerMaterialized: invalid peerRank={} (myRank={}, nRanks={})",
            peerRank,
            myRank_,
            nRanks_));
  }
  if (!config_.ibLazyConnect) {
    return true;
  }
  return peerMaterialized_[rankToPeerIndex(peerRank)];
}

void MultiPeerIbTransportBase::queuePeerForMaterialization(int peerRank) {
  if (!config_.ibLazyConnect) {
    return;
  }
  if (materializationFailed_) {
    throw std::runtime_error(
        "MultiPeerIbTransport: lazy peer materialization previously failed; "
        "retry is not supported");
  }
  if (peerRank == myRank_ || peerRank < 0 || peerRank >= nRanks_) {
    throw std::invalid_argument(
        fmt::format(
            "queuePeerForMaterialization: invalid peerRank={} (myRank={}, "
            "nRanks={})",
            peerRank,
            myRank_,
            nRanks_));
  }
  if (isPeerMaterialized(peerRank)) {
    return;
  }
  for (int p : pendingPeers_) {
    if (p == peerRank) {
      return;
    }
  }
  pendingPeers_.push_back(peerRank);
}

std::vector<IbTransportExchInfoAll> MultiPeerIbTransportBase::allGatherExchInfo(
    const IbTransportExchInfoAll& localInfo) {
  if (nRanks_ > kMaxRanksForAllGather) {
    throw std::runtime_error(
        fmt::format(
            "Too many ranks ({}) for allGather-based exchange, max is {}",
            nRanks_,
            kMaxRanksForAllGather));
  }
  std::vector<IbTransportExchInfoAll> allInfo(nRanks_);
  allInfo[myRank_] = localInfo;
  auto result =
      bootstrap_
          ->allGather(
              allInfo.data(), sizeof(IbTransportExchInfoAll), myRank_, nRanks_)
          .get();
  if (result != 0) {
    throw std::runtime_error(
        "MultiPeerIbTransport::allGatherExchInfo allGather failed");
  }
  return allInfo;
}

void MultiPeerIbTransportBase::validatePeerTopology(
    const std::vector<IbTransportExchInfoAll>& allInfo) const {
  const int numPeers = nRanks_ - 1;
  for (int peerIndex = 0; peerIndex < numPeers; ++peerIndex) {
    const int peerRank = peerIndexToRank(peerIndex);
    const auto& peerInfo = allInfo[peerRank];
    // Same-rail pairing relies on the symmetric (myRank+peerRank) % numNics
    // offset, which only makes sense when both sides agree on numNics.
    if (peerInfo.numNics != numNics_) {
      throw std::runtime_error(
          fmt::format(
              "Peer rank {} reports numNics={} but my numNics={}; all ranks "
              "must agree on numNics for same-rail pairing",
              peerRank,
              peerInfo.numNics,
              numNics_));
    }
    if (peerInfo.numQpsPerPeerPerNic != config_.numQpsPerPeerPerNic) {
      throw std::runtime_error(
          fmt::format(
              "Peer rank {} reports numQpsPerPeerPerNic={} but mine is {}; all "
              "ranks must use the same numQpsPerPeerPerNic",
              peerRank,
              peerInfo.numQpsPerPeerPerNic,
              config_.numQpsPerPeerPerNic));
    }
  }
}

void MultiPeerIbTransportBase::exchangeRawWithPeer(
    int peerRank,
    const void* localPayload,
    void* remotePayload,
    std::size_t bytes,
    int tag) {
  auto timeoutUs = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::milliseconds(config_.materializePeerTimeoutMs));
  auto waitFuture = [&](auto&& future, const char* op) -> int {
    try {
      return std::move(future).get(timeoutUs);
    } catch (const std::exception&) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} {} with peer {} timed out ({}ms)",
              myRank_,
              op,
              peerRank,
              config_.materializePeerTimeoutMs));
    }
  };

  // Lower rank recvs first to avoid deadlock with blocking bootstrap
  // implementations (e.g. MpiBootstrap uses blocking MPI_Send/MPI_Recv).
  if (myRank_ < peerRank) {
    auto recvFuture =
        bootstrap_->recv(remotePayload, bytes, peerRank, /*tag=*/tag);
    int recvResult = waitFuture(std::move(recvFuture), "recv");
    if (recvResult != 0) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} recv from peer {} failed (error {})",
              myRank_,
              peerRank,
              recvResult));
    }
    auto sendFuture = bootstrap_->send(
        const_cast<void*>(localPayload), bytes, peerRank, /*tag=*/tag);
    int sendResult = waitFuture(std::move(sendFuture), "send");
    if (sendResult != 0) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} send to peer {} failed (error {})",
              myRank_,
              peerRank,
              sendResult));
    }
  } else {
    auto sendFuture = bootstrap_->send(
        const_cast<void*>(localPayload), bytes, peerRank, /*tag=*/tag);
    int sendResult = waitFuture(std::move(sendFuture), "send");
    if (sendResult != 0) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} send to peer {} failed (error {})",
              myRank_,
              peerRank,
              sendResult));
    }
    auto recvFuture =
        bootstrap_->recv(remotePayload, bytes, peerRank, /*tag=*/tag);
    int recvResult = waitFuture(std::move(recvFuture), "recv");
    if (recvResult != 0) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} recv from peer {} failed (error {})",
              myRank_,
              peerRank,
              recvResult));
    }
  }
}

} // namespace comms::prims
