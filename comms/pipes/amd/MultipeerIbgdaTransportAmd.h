// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// AMD GPU (HIP/ROCm) MultipeerIbgdaTransport
// =============================================================================
//
// AMD equivalent of comms::pipes::MultipeerIbgdaTransport.
// Provides the same public API but uses:
//   - HIP runtime for GPU memory management (instead of CUDA)
//   - HSA runtime for mapping NIC UAR BlueFlame register to GPU
//   - Raw ibverbs + mlx5dv for QP/CQ creation and export
//   - Manual construction of pipes_gda_gpu_dev_verbs_qp in GPU memory
//
// Modeled after IbgdaTestFixture.h but refactored into a reusable transport
// class that supports N peers.
// =============================================================================

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <infiniband/verbs.h>

#include "PipesGdaShared.h"
#include "comms/common/bootstrap/IBootstrap.h"
#include "verbs/VerbsDev.h"

// Forward declaration
namespace pipes_gda {
template <typename NicBackend>
class P2pIbgdaTransportDeviceImpl;
struct Mlx5NicBackend;
using P2pIbgdaTransportDevice = P2pIbgdaTransportDeviceImpl<Mlx5NicBackend>;
} // namespace pipes_gda

namespace pipes_gda {

struct MultipeerIbgdaTransportAmdConfig {
  // HIP device index for GPU operations
  int hipDevice{0};

  // GID index for RoCE. Default = 3 (RoCEv2).
  int gidIndex{3};

  // Queue pair depth (number of outstanding WQEs per peer).
  uint32_t qpDepth{128};

  // InfiniBand timeout, retry, traffic class settings
  uint8_t timeout{20};
  uint8_t retryCount{7};
  uint8_t trafficClass{224};
  uint8_t serviceLevel{0};
  uint8_t minRnrTimer{12};
  uint8_t rnrRetry{7};
};

struct IbgdaTransportExchInfoAmd {
  uint32_t qpn{0};
  uint8_t gid[16]{};
  int gidIndex{0};
  uint16_t lid{0};
  enum ibv_mtu mtu { IBV_MTU_4096 };
};

constexpr int kMaxRanksAmd = 128;

struct IbgdaTransportExchInfoAllAmd {
  uint8_t gid[16]{};
  int gidIndex{0};
  uint16_t lid{0};
  enum ibv_mtu mtu { IBV_MTU_4096 };
  uint32_t qpnForRank[kMaxRanksAmd]{};
};

class MultipeerIbgdaTransportAmd {
 public:
  // Connect to ALL peers (nRanks - 1 QPs)
  MultipeerIbgdaTransportAmd(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultipeerIbgdaTransportAmdConfig& config);

  // Connect only to specified target ranks (filtered QPs).
  // Used for multi-node: only create QPs to remote peers, skip same-host.
  MultipeerIbgdaTransportAmd(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultipeerIbgdaTransportAmdConfig& config,
      const std::vector<int>& targetRanks);

  ~MultipeerIbgdaTransportAmd();

  MultipeerIbgdaTransportAmd(const MultipeerIbgdaTransportAmd&) = delete;
  MultipeerIbgdaTransportAmd& operator=(const MultipeerIbgdaTransportAmd&) =
      delete;

  // Collective: all ranks must call
  void exchange();

  // Get device transport for a specific peer
  P2pIbgdaTransportDevice* getP2pTransportDevice(int peerRank) const;

  // Get base pointer to device transport array
  P2pIbgdaTransportDevice* getDeviceTransportPtr() const;

  int numPeers() const;
  int myRank() const;
  int nRanks() const;

  // Register a GPU buffer for RDMA access
  IbgdaLocalBuffer registerBuffer(void* ptr, std::size_t size);

  // Deregister a buffer
  void deregisterBuffer(void* ptr);

  // Collective: exchange buffer info with all peers
  std::vector<IbgdaRemoteBuffer> exchangeBuffer(
      const IbgdaLocalBuffer& localBuf);

 private:
  // Initialization helpers
  void openIbDevice();
  bool createQpAndCq(int peerIndex);
  bool connectQp(int peerIndex, const IbgdaTransportExchInfoAmd& peerInfo);
  bool exportQpToGpu(int peerIndex);
  void cleanup();

  int rankToPeerIndex(int rank) const;
  int peerIndexToRank(int peerIndex) const;
  void initCommon();

  // Rank info
  const int myRank_;
  const int nRanks_;
  std::vector<int> targetRanks_; // Ranks to connect to (empty = all)
  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;
  MultipeerIbgdaTransportAmdConfig config_;

  // IB verbs resources
  ibv_context* ibvCtx_{nullptr};
  ibv_pd* ibvPd_{nullptr};
  union ibv_gid localGid_{};
  enum ibv_mtu localMtu_ { IBV_MTU_4096 };
  std::string gpuPciBusId_;

  // Per-peer QP/CQ resources
  struct PeerQpResources {
    ibv_cq* cq{nullptr};
    ibv_qp* qp{nullptr};
    uint32_t qpNum{0};
    pipes_gda_gpu_dev_verbs_qp* gpuQp{nullptr};
    void* gpuUarBf{nullptr};
    void* uarBfHostPtr{nullptr};
    size_t uarBfSize{0};
    struct mlx5dv_devx_uar* devxUar{
        nullptr}; // Per-QP UAR to avoid BF contention
    // Host pointers registered via hipHostRegister (must be unregistered)
    void* registeredSqBuf{nullptr};
    void* registeredCqBuf{nullptr};
    void* registeredSqDbrecPage{nullptr};
    void* registeredCqDbrecPage{nullptr}; // nullptr if same page as SQ
  };
  std::vector<PeerQpResources> peerResources_;

  // Sink buffer for atomic return values
  void* sinkBuffer_{nullptr};
  ibv_mr* sinkMr_{nullptr};

  // User-registered buffers
  std::unordered_map<void*, ibv_mr*> registeredBuffers_;

  // Device transports array (GPU memory)
  P2pIbgdaTransportDevice* peerTransportsGpu_{nullptr};
  std::size_t peerTransportSize_{0};

  // mlx5 UAR (allocated for the connection)
  void* uar_{nullptr};
};

} // namespace pipes_gda
