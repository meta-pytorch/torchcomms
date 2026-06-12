// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/MultiPeerIbTransport.h"

#include <cerrno>
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
    std::shared_ptr<meta::comms::IBootstrap> bootstrap)
    : myRank_(myRank), nRanks_(nRanks), bootstrap_(std::move(bootstrap)) {
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

} // namespace comms::prims
