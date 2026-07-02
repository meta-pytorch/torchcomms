// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/moe_ep/cpp/Buffer.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <folly/futures/Future.h>
#include <torch/csrc/utils/pybind.h>

// `meta::comms::DeviceBuffer` is referenced by `MultiPeerNvlTransport.h`
// private members. Bring it into scope before that header (transitively
// pulled in by GpuMemHandler.h consumers).
#ifdef __HIP_PLATFORM_AMD__
#include "comms/prims/transport/amd/HipHostCompat.h"
#else
#include "comms/utils/CudaRAII.h"
#endif

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/collectives/moe_ep/cpp/intranode/Runtime.h"
#include "comms/prims/collectives/moe_ep/cpp/intranode/kernels/Combine.cuh"
#include "comms/prims/collectives/moe_ep/cpp/intranode/kernels/Dispatch.cuh"
#include "comms/prims/collectives/moe_ep/cpp/intranode/kernels/Layout.cuh"
#include "comms/prims/collectives/moe_ep/cpp/intranode/kernels/Notify.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/KernelConfigs.cuh"
#include "comms/prims/memory/GpuMemHandler.h"

namespace comms::prims::moe_ep {

namespace py = pybind11;

namespace {

void checkCuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
  }
}

/**
 * Trivial bootstrap that just remembers the rank and group size; collective
 * operations are no-ops (return 0). Sufficient for the local-rank-only path
 * inside `IntranodeRuntime` because `GpuMemHandler` falls back to a
 * single-rank fast path when nRanks == 1, and `MultiPeerNvlTransport`'s
 * `exchange()` does its own metadata gather.
 *
 * For multi-rank mode, a real `PreExchangedBootstrap` (per Q2/Q6 in the
 * design) gets plumbed through `Buffer::sync()` in a follow-up commit.
 * Until then, this stub is enough for single-rank smoke tests; multi-rank
 * intranode_dispatch will be exercised once D5's PreExchangedBootstrap
 * lands.
 */
class StubBootstrap : public meta::comms::IBootstrap {
 public:
  StubBootstrap(int rank, int nRanks) : rank_(rank), nRanks_(nRanks) {}

  folly::SemiFuture<int>
  allGather(void* /*buf*/, int /*len*/, int /*rank*/, int /*nranks*/) override {
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int> barrier(int /*rank*/, int /*nranks*/) override {
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int>
  send(void* /*buf*/, int /*len*/, int /*peer*/, int /*tag*/) override {
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int>
  recv(void* /*buf*/, int /*len*/, int /*peer*/, int /*tag*/) override {
    return folly::makeSemiFuture(0);
  }

 private:
  const int rank_;
  const int nRanks_;
};

} // namespace

Buffer::Buffer(
    int rank,
    int numRanks,
    std::int64_t numNvlBytes,
    std::int64_t numRdmaBytes,
    bool lowLatencyMode,
    bool explicitlyDestroy,
    bool enableShrink,
    bool useFabric)
    : rank_(rank),
      numRanks_(numRanks),
      numNvlBytes_(numNvlBytes),
      numRdmaBytes_(numRdmaBytes),
      lowLatencyMode_(lowLatencyMode),
      explicitlyDestroy_(explicitlyDestroy),
      enableShrink_(enableShrink),
      useFabric_(useFabric) {
  if (rank_ < 0 || rank_ >= numRanks_) {
    throw std::invalid_argument("Buffer: rank out of bounds");
  }
  if (numRanks_ <= 0 || numRanks_ > NUM_MAX_NVL_PEERS) {
    // Phase 1 is intranode-only; Phase 3 (D5) adds multi-NVL-peer handling.
    throw std::invalid_argument(
        "Buffer: numRanks must be in (0, NUM_MAX_NVL_PEERS] for Phase 1");
  }

  checkCuda(cudaGetDevice(&deviceId_), "Buffer: cudaGetDevice failed");

  // Allocate the local NVL data buffer EARLY (in ctor, not sync) so
  // `get_local_ipc_handle()` returns a real IPC handle BEFORE Python's
  // `dist.all_gather_object(local_ipc_handle)`. Without this, all peers
  // gather zero-filled placeholders and `cudaIpcOpenMemHandle` fails at
  // sync-time with "invalid argument".
  //
  // Layout:
  //   [0 .. numNvlBytes_)              data region (per_rank/per_expert
  //                                    prefix matrices, dispatch/combine
  //                                    chunked send/recv slots)
  //   [numNvlBytes_ .. +barrierBytes)  barrier signal slots — separate region
  //                                    so notify_dispatch's prefix-matrix
  //                                    writes don't clobber barrier counters
  //                                    (which would deadlock barrier_device).
  //
  // On AMD use the uncached allocator path so the buffer is BNXT
  // dma-buf-compatible; matches `GpuMemHandler::kCudaIpcUncached`.
  const std::size_t barrierBytes =
      static_cast<std::size_t>(kernels::NUM_MAX_FIFO_SLOTS) * sizeof(int);
  const std::size_t totalBytes =
      static_cast<std::size_t>(numNvlBytes_) + barrierBytes;
#ifdef __HIP_PLATFORM_AMD__
  checkCuda(
      hipExtMallocWithFlags(
          &localIpcBuffer_, totalBytes, hipDeviceMallocUncached),
      "Buffer: hipExtMallocWithFlags(uncached) failed");
#else
  checkCuda(
      cudaMalloc(&localIpcBuffer_, totalBytes),
      "Buffer: cudaMalloc(local NVL buffer) failed");
#endif
  // Defer `cudaIpcGetMemHandle` to `get_local_ipc_handle()` (lazy). On
  // ROCm/HIP, calling `hipIpcGetMemHandle` from the ctor — *before* PyTorch
  // does its first `dist.all_gather_object` — triggers a downstream
  // `hipErrorPeerAccessAlreadyEnabled` from PyTorch's caching allocator the
  // next time a CPU→GPU `Tensor.to(device)` runs. The handle is only ever
  // read by `get_local_ipc_handle()` so making it lazy is harmless.
  // Zero only the barrier region. Use the NULL (default) stream — async, so
  // the ctor returns immediately; the device runtime serializes against
  // subsequent kernel launches.
  if (barrierBytes > 0) {
    void* barrierPtr =
        static_cast<std::uint8_t*>(localIpcBuffer_) + numNvlBytes_;
    checkCuda(
        cudaMemsetAsync(barrierPtr, 0, barrierBytes, /*stream=*/nullptr),
        "Buffer: cudaMemsetAsync(barrier region) failed");
  }
}

Buffer::~Buffer() {
  // intranode_ destructor handles workspace cleanup. Close peer IPC handles
  // BEFORE freeing the local buffer (peer handles must be closed while their
  // backing allocations are still live across the IPC; the local buffer free
  // is independent).
  for (int i = 0; i < static_cast<int>(peerIpcBuffers_.size()); ++i) {
    if (i == rank_) {
      continue;
    }
    if (peerIpcBuffers_[i] != nullptr) {
      (void)cudaIpcCloseMemHandle(peerIpcBuffers_[i]);
    }
  }
  peerIpcBuffers_.clear();
  if (localIpcBuffer_ != nullptr) {
    (void)cudaFree(localIpcBuffer_);
    localIpcBuffer_ = nullptr;
  }
}

int Buffer::get_num_rdma_ranks() const noexcept {
  // Phase 1: intranode-only → exactly 1 RDMA rank (this node).
  return 1;
}

int Buffer::get_rdma_rank() const noexcept {
  return 0;
}

int Buffer::get_root_rdma_rank(bool /*global*/) const noexcept {
  return 0;
}

py::bytearray Buffer::get_local_ipc_handle() const {
  // Lazily compute the IPC handle on first access — see Buffer ctor for why
  // we don't do this eagerly. The handle is cached after the first call
  // because `cudaIpcGetMemHandle` is not idempotent across all HIP versions
  // and the Python wrapper calls this once anyway.
  if (!localIpcHandleReady_) {
    if (localIpcBuffer_ == nullptr) {
      throw std::runtime_error(
          "Buffer::get_local_ipc_handle: local buffer not allocated");
    }
    checkCuda(
        cudaIpcGetMemHandle(&localIpcHandle_, localIpcBuffer_),
        "Buffer::get_local_ipc_handle: cudaIpcGetMemHandle failed");
    localIpcHandleReady_ = true;
  }
  return py::bytearray(
      reinterpret_cast<const char*>(&localIpcHandle_), sizeof(localIpcHandle_));
}

py::bytearray Buffer::get_local_nvshmem_unique_id() const {
  // Pipes uses IBootstrap for QP exchange, not NVSHMEM unique IDs.
  // Return a fixed dummy bytearray; the test code only forwards it back
  // to `runtime.sync()` and we ignore it.
  static constexpr std::size_t kNvshmemUniqueIdSize = 128;
  return py::bytearray(nullptr, kNvshmemUniqueIdSize);
}

void Buffer::sync(
    const std::vector<int>& deviceIds,
    const std::vector<std::optional<py::bytearray>>& ipcHandles,
    const std::optional<py::bytearray>& /*rootUniqueId*/) {
  if (static_cast<int>(deviceIds.size()) != numRanks_) {
    throw std::runtime_error(
        "Buffer::sync: deviceIds length doesn't match numRanks");
  }
  if (deviceIds[rank_] != deviceId_) {
    throw std::runtime_error(
        "Buffer::sync: local device ID mismatch (Python vs C++)");
  }
  if (static_cast<int>(ipcHandles.size()) != numRanks_) {
    throw std::runtime_error(
        "Buffer::sync: ipcHandles length doesn't match numRanks");
  }

  // Open each peer's IPC handle and build the peer-pointer table. Entry
  // `rank_` is the local buffer; other entries come from
  // `cudaIpcOpenMemHandle(peer_i_handle)`. The opened pointers are owned by
  // this Buffer and closed in the destructor.
  peerIpcBuffers_.assign(numRanks_, nullptr);
  for (int i = 0; i < numRanks_; ++i) {
    if (i == rank_) {
      peerIpcBuffers_[i] = localIpcBuffer_;
      continue;
    }
    if (!ipcHandles[i].has_value()) {
      throw std::runtime_error(
          "Buffer::sync: missing IPC handle for peer " + std::to_string(i));
    }
    // Copy the bytearray contents into a `cudaIpcMemHandle_t` for the open
    // call. `bytearray` exposes its bytes via `PyBytes_AsString`-style API
    // through pybind's casting; using `std::string` round-trip preserves the
    // raw bytes including embedded nulls.
    const std::string handleBytes = std::string(*ipcHandles[i]);
    if (handleBytes.size() != sizeof(cudaIpcMemHandle_t)) {
      throw std::runtime_error(
          "Buffer::sync: peer " + std::to_string(i) +
          " IPC handle size mismatch (got " +
          std::to_string(handleBytes.size()) + ", expected " +
          std::to_string(sizeof(cudaIpcMemHandle_t)) + ")");
    }
    cudaIpcMemHandle_t peerHandle{};
    std::memcpy(&peerHandle, handleBytes.data(), sizeof(peerHandle));
    void* peerPtr = nullptr;
    checkCuda(
        cudaIpcOpenMemHandle(
            &peerPtr, peerHandle, cudaIpcMemLazyEnablePeerAccess),
        "Buffer::sync: cudaIpcOpenMemHandle failed");
    peerIpcBuffers_[i] = peerPtr;
  }

  // Construct intranode runtime via the pre-allocated-buffer ctor: skips
  // GpuMemHandler/transport entirely since Python already gathered IPC
  // handles for us.
  intranode_ = std::make_unique<IntranodeRuntime>(
      rank_,
      numRanks_,
      static_cast<std::size_t>(numNvlBytes_),
      localIpcBuffer_,
      localIpcHandle_,
      peerIpcBuffers_);

  // Full device sync after sync() ensures the host-to-device pointer-table
  // memcpys + barrier-region memset are visible to all subsequent kernels
  // (which may run on a stream different from the one used in setup).
  checkCuda(cudaDeviceSynchronize(), "Buffer::sync: cudaDeviceSynchronize");

  available_ = true;
}

void Buffer::destroy() {
  if (destroyed_) {
    return;
  }
  intranode_.reset();
  destroyed_ = true;
  available_ = false;
}

std::tuple<
    torch::Tensor,
    std::optional<torch::Tensor>,
    torch::Tensor,
    torch::Tensor,
    std::optional<EventHandle>>
Buffer::get_dispatch_layout(
    const torch::Tensor& topkIdx,
    int numExperts,
    const std::optional<EventHandle>& /*previousEvent*/,
    bool asyncFinish,
    bool /*allocateOnCommStream*/) {
  if (!available_) {
    throw std::runtime_error("Buffer::get_dispatch_layout: not synced yet");
  }

  TORCH_CHECK(topkIdx.dim() == 2, "topk_idx must be 2D");
  TORCH_CHECK(
      topkIdx.scalar_type() == torch::kInt64,
      "topk_idx must be torch.int64 (== topk_idx_t)");
  const int numTokens = topkIdx.size(0);
  const int numTopk = topkIdx.size(1);

  auto opts =
      torch::TensorOptions().dtype(torch::kInt32).device(topkIdx.device());
  auto numTokensPerRank = torch::empty({numRanks_}, opts);
  auto numTokensPerExpert = torch::empty({numExperts}, opts);
  auto isTokenInRank = torch::empty(
      {numTokens, numRanks_},
      torch::TensorOptions().dtype(torch::kBool).device(topkIdx.device()));

  // Phase 1 intranode → no per-RDMA-rank tensor.
  std::optional<torch::Tensor> numTokensPerRdmaRank = std::nullopt;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  kernels::get_dispatch_layout(
      topkIdx.data_ptr<std::int64_t>(),
      numTokensPerRank.data_ptr<int>(),
      /*num_tokens_per_rdma_rank=*/nullptr,
      numTokensPerExpert.data_ptr<int>(),
      isTokenInRank.data_ptr<bool>(),
      numTokens,
      numTopk,
      numRanks_,
      numExperts,
      stream);

  std::optional<EventHandle> event;
  if (asyncFinish) {
    event = EventHandle();
  }
  return std::make_tuple(
      numTokensPerRank,
      numTokensPerRdmaRank,
      numTokensPerExpert,
      isTokenInRank,
      event);
}

py::tuple Buffer::intranode_dispatch(
    const py::object& xObj,
    const py::object& handleObj,
    const std::optional<torch::Tensor>& numTokensPerRank,
    const std::optional<torch::Tensor>& isTokenInRank,
    const std::optional<torch::Tensor>& numTokensPerExpert,
    const std::optional<torch::Tensor>& topkIdx,
    const std::optional<torch::Tensor>& topkWeights,
    int expertAlignment,
    int numWorstTokens,
    const Config& config,
    const std::optional<EventHandle>& /*previousEvent*/,
    bool asyncFinish,
    bool /*allocateOnCommStream*/) {
  if (!available_) {
    throw std::runtime_error("Buffer::intranode_dispatch: not synced yet");
  }
  // Cached-dispatch path is selected when the caller passes a non-None handle
  // (a dict from a prior dispatch). The handle holds the cached prefix
  // matrices + recv metadata, so we can skip notify_dispatch's count exchange
  // and call cached_notify_dispatch for state refresh.
  const bool isCached =
      !handleObj.is_none() && py::isinstance<py::dict>(handleObj);
  if (!isCached &&
      (!numTokensPerRank.has_value() || !isTokenInRank.has_value() ||
       !numTokensPerExpert.has_value())) {
    throw std::runtime_error(
        "Buffer::intranode_dispatch: "
        "non-cached mode requires num_tokens_per_rank, is_token_in_rank, "
        "and num_tokens_per_expert");
  }
  if (isCached && !isTokenInRank.has_value()) {
    throw std::runtime_error(
        "Buffer::intranode_dispatch: cached mode requires is_token_in_rank "
        "(thread it through from the cached handle)");
  }

  // FP8 path: x is a (data, scales) tuple. Both BF16/FP32 (Tensor input) and
  // FP8 (tuple input) flow through the same kernel; scales are nullptr for
  // non-FP8. Use PyTuple_Check + try-cast because pybind11's `isinstance<>`
  // doesn't reliably recognize plain Python tuples or PyTorch tensors across
  // toolchain versions.
  torch::Tensor x;
  std::optional<torch::Tensor> xScales;
  bool isFp8 = false;
  if (PyTuple_Check(xObj.ptr())) {
    auto xTuple = xObj.cast<py::tuple>();
    TORCH_CHECK(
        xTuple.size() == 2, "x tuple must be (data, scales) of length 2");
    x = xTuple[0].cast<torch::Tensor>();
    xScales = xTuple[1].cast<torch::Tensor>();
    isFp8 = true;
    // FP8 scale path is deferred: the kernel derives num_scales from
    // scale_hidden_stride (Dispatch.cu), which collapses to 1 for contiguous
    // scales and silently under-fills the recv-scale buffer. Reject the
    // (data, scales) tuple until the FP8-enablement diff threads num_scales.
    TORCH_CHECK(
        !isFp8,
        "intranode_dispatch: FP8 (x=(data, scales)) is not supported yet; the "
        "scale path is deferred to a follow-up diff. Pass a bf16/fp16 tensor.");
  } else {
    // Tensor path — fall back on cast-or-throw to bypass unreliable
    // pybind11::isinstance<torch::Tensor>.
    x = xObj.cast<torch::Tensor>();
  }
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x must be 2D contiguous");
  TORCH_CHECK(
      (x.size(1) * x.element_size()) % sizeof(int4) == 0,
      "hidden * elem_size must be int4-aligned");
  TORCH_CHECK(
      isTokenInRank->scalar_type() == torch::kBool, "is_token_in_rank bool");
  if (!isCached) {
    TORCH_CHECK(
        numTokensPerRank->scalar_type() == torch::kInt32 &&
            numTokensPerRank->size(0) == numRanks_,
        "num_tokens_per_rank shape/dtype mismatch");
    TORCH_CHECK(
        numTokensPerExpert->scalar_type() == torch::kInt32 &&
            numTokensPerExpert->size(0) % numRanks_ == 0,
        "num_tokens_per_expert shape/dtype mismatch");
  }

  const int numTokens = static_cast<int>(x.size(0));
  const int hidden = static_cast<int>(x.size(1));
  // num_experts is unused on the cached path (no per-expert routing). Pull
  // it from numTokensPerExpert for non-cached, default to 0 for cached.
  const int numExperts =
      isCached ? 0 : static_cast<int>(numTokensPerExpert->size(0));
  const int numLocalExperts = isCached ? 0 : numExperts / numRanks_;

  // FP8 scale validation. x_scales is 2D, dim0 == num_tokens,
  // dim1 == num_scales (typically hidden / 128). Scales must be float32 or
  // int (containers for casted FP8 scale exponents).
  float* xScalesPtr = nullptr;
  int numScales = 0;
  int scaleTokenStride = 0;
  int scaleHiddenStride = 0;
  if (xScales.has_value()) {
    TORCH_CHECK(
        xScales->scalar_type() == torch::kFloat32 ||
            xScales->scalar_type() == torch::kInt,
        "x_scales must be float32 or int32");
    TORCH_CHECK(xScales->dim() == 2, "x_scales must be 2D");
    TORCH_CHECK(
        xScales->size(0) == numTokens,
        "x_scales.size(0) must equal num_tokens");
    numScales = static_cast<int>(xScales->size(1));
    xScalesPtr = static_cast<float*>(xScales->data_ptr());
    scaleTokenStride = static_cast<int>(xScales->stride(0));
    scaleHiddenStride = static_cast<int>(xScales->stride(1));
  }

  int numTopk = 0;
  std::int64_t* topkIdxPtr = nullptr;
  float* topkWeightsPtr = nullptr;
  if (topkIdx.has_value()) {
    TORCH_CHECK(topkWeights.has_value(), "topk_idx + topk_weights must agree");
    numTopk = static_cast<int>(topkIdx->size(1));
    TORCH_CHECK(numExperts > 0, "non-zero num_experts required for topk");
    TORCH_CHECK(
        topkIdx->dim() == 2 && topkIdx->is_contiguous() &&
            topkIdx->scalar_type() == torch::kInt64,
        "topk_idx must be 2D int64 contiguous");
    TORCH_CHECK(
        topkWeights->dim() == 2 && topkWeights->is_contiguous() &&
            topkWeights->scalar_type() == torch::kFloat32,
        "topk_weights must be 2D float32 contiguous");
    topkIdxPtr = topkIdx->data_ptr<std::int64_t>();
    topkWeightsPtr = topkWeights->data_ptr<float>();
  }

  TORCH_CHECK(config.num_sms % 2 == 0, "config.num_sms must be even");
  const int numChannels = config.num_sms / 2;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor rankPrefixMatrix;
  torch::Tensor channelPrefixMatrix;
  torch::Tensor recvSrcIdx;
  torch::Tensor recvChannelPrefixMatrix;
  torch::Tensor sendHead;
  int numRecvTokens = 0;
  std::vector<int> numRecvTokensPerExpertList;
  const int numMemsetInt = numChannels * numRanks_ * 4;

  if (isCached) {
    // Pull cached metadata from the handle dict (Python wrapper packs it for
    // us). cached_notify_dispatch refreshes the barrier + memsets per-channel
    // state (2 internal barriers, so move FIFO by 2). Skip the CPU sync —
    // numRecvTokens comes from the handle.
    auto handleDict = handleObj.cast<py::dict>();
    rankPrefixMatrix = handleDict["rank_prefix_matrix"].cast<torch::Tensor>();
    channelPrefixMatrix =
        handleDict["channel_prefix_matrix"].cast<torch::Tensor>();
    recvSrcIdx = handleDict["recv_src_idx"].cast<torch::Tensor>();
    recvChannelPrefixMatrix =
        handleDict["recv_channel_prefix_matrix"].cast<torch::Tensor>();
    sendHead = handleDict["send_head"].cast<torch::Tensor>();
    numRecvTokens = handleDict["num_recv_tokens"].cast<int>();

    kernels::cached_notify_dispatch(
        rankPrefixMatrix.data_ptr<int>(),
        numMemsetInt,
        intranode_->getPeerDataPtrsDevice(),
        intranode_->getBarrierSignalPtrsDevice(),
        rank_,
        numRanks_,
        stream,
        intranode_->head());
    intranode_->moveFifoSlots(2);
    // Cached test path doesn't use per-expert counts (no topk). Leave
    // numRecvTokensPerExpertList empty.
  } else {
    // Reset host counters before the notify_dispatch kernel writes them.
    *intranode_->getMoeRecvCounterHost() = -1;
    for (int i = 0; i < numLocalExperts; ++i) {
      intranode_->getMoeRecvExpertCounterHost()[i] = -1;
    }

    rankPrefixMatrix = torch::empty(
        {numRanks_, numRanks_},
        torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
    channelPrefixMatrix = torch::empty(
        {numRanks_, numChannels},
        torch::TensorOptions().dtype(torch::kInt32).device(x.device()));

    kernels::notify_dispatch(
        numTokensPerRank->data_ptr<int>(),
        intranode_->getMoeRecvCounterDevice(),
        numRanks_,
        numTokensPerExpert->data_ptr<int>(),
        intranode_->getMoeRecvExpertCounterDevice(),
        numExperts,
        numTokens,
        isTokenInRank->data_ptr<bool>(),
        channelPrefixMatrix.data_ptr<int>(),
        rankPrefixMatrix.data_ptr<int>(),
        numMemsetInt,
        expertAlignment,
        intranode_->getPeerDataPtrsDevice(),
        intranode_->getBarrierSignalPtrsDevice(),
        intranode_->head(),
        rank_,
        stream,
        numChannels);
    intranode_->moveFifoSlots(3);

    if (numWorstTokens > 0) {
      numRecvTokens = numWorstTokens;
      TORCH_CHECK(
          topkIdx.has_value(), "num_worst_tokens > 0 requires topk_idx");
    } else {
      // Block until notify_dispatch flushes the per-rank/per-expert counters.
      cudaStreamSynchronize(stream);
      numRecvTokens = *intranode_->getMoeRecvCounterHost();
      TORCH_CHECK(
          numRecvTokens >= 0, "notify_dispatch did not publish counter");

      numRecvTokensPerExpertList.resize(numLocalExperts);
      for (int i = 0; i < numLocalExperts; ++i) {
        numRecvTokensPerExpertList[i] =
            intranode_->getMoeRecvExpertCounterHost()[i];
      }
    }

    // Allocate fresh recv state — kernel writes recvSrcIdx,
    // recvChannelPrefixMatrix, and sendHead in non-cached mode.
    recvSrcIdx = torch::empty(
        {numRecvTokens},
        torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
    recvChannelPrefixMatrix = torch::empty(
        {numRanks_, numChannels},
        torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
    sendHead = torch::empty(
        {numTokens, numRanks_},
        torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
  }

  auto recvX = torch::empty({numRecvTokens, hidden}, x.options());
  std::optional<torch::Tensor> recvXScales;
  float* recvXScalesPtr = nullptr;
  if (xScales.has_value()) {
    recvXScales = torch::empty({numRecvTokens, numScales}, xScales->options());
    recvXScalesPtr = static_cast<float*>(recvXScales->data_ptr());
  }

  std::optional<torch::Tensor> recvTopkIdx;
  std::optional<torch::Tensor> recvTopkWeights;
  std::int64_t* recvTopkIdxPtr = nullptr;
  float* recvTopkWeightsPtr = nullptr;
  if (topkIdx.has_value()) {
    recvTopkIdx = torch::empty({numRecvTokens, numTopk}, topkIdx->options());
    recvTopkWeights =
        torch::empty({numRecvTokens, numTopk}, topkWeights->options());
    recvTopkIdxPtr = recvTopkIdx->data_ptr<std::int64_t>();
    recvTopkWeightsPtr = recvTopkWeights->data_ptr<float>();
  }

  const int hiddenInt4 =
      static_cast<int>(hidden * recvX.element_size() / sizeof(int4));

  // num_nvl_bytes must hold the exact per-rank NVL layout the dispatch kernel
  // writes (Dispatch.cu): R×R rank-prefix + per (channel, rank) 4 ring ints +
  // recv slots of data/src-idx/topk-idx/topk-weights/scales. Undersize ->
  // OOB peer-mapped writes, so fail loudly here instead.
  {
    const std::size_t cr = static_cast<std::size_t>(numChannels) * numRanks_;
    const std::size_t recv =
        static_cast<std::size_t>(config.num_max_nvl_chunked_recv_tokens);
    const std::size_t requiredNvlBytes =
        static_cast<std::size_t>(numRanks_) * numRanks_ * sizeof(int) +
        cr * 4UL * sizeof(int) +
        cr * recv * static_cast<std::size_t>(hidden) * recvX.element_size() +
        cr * recv * sizeof(int) +
        cr * recv * static_cast<std::size_t>(numTopk) * sizeof(std::int64_t) +
        cr * recv * static_cast<std::size_t>(numTopk) * sizeof(float) +
        cr * recv * static_cast<std::size_t>(numScales) * sizeof(float);
    TORCH_CHECK(
        requiredNvlBytes <= static_cast<std::size_t>(numNvlBytes_),
        "intranode_dispatch: num_nvl_bytes (",
        numNvlBytes_,
        ") is too small for the NVL layout; need >= ",
        requiredNvlBytes,
        " bytes. Size the Buffer with "
        "Config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks).");
  }

  kernels::intranode_dispatch(
      recvX.data_ptr(),
      recvXScalesPtr,
      recvTopkIdxPtr,
      recvTopkWeightsPtr,
      recvSrcIdx.data_ptr<int>(),
      recvChannelPrefixMatrix.data_ptr<int>(),
      sendHead.data_ptr<int>(),
      x.data_ptr(),
      xScalesPtr,
      topkIdxPtr,
      topkWeightsPtr,
      isTokenInRank->data_ptr<bool>(),
      channelPrefixMatrix.data_ptr<int>(),
      numTokens,
      numWorstTokens,
      hiddenInt4,
      numTopk,
      numExperts,
      scaleTokenStride,
      scaleHiddenStride,
      intranode_->getPeerDataPtrsDevice(),
      rank_,
      numRanks_,
      stream,
      config.num_sms,
      config.num_max_nvl_chunked_send_tokens,
      config.num_max_nvl_chunked_recv_tokens);

  // The Python wrapper expects:
  //   (recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
  //    num_recv_tokens_per_expert_list, handle, event)
  // where `handle` is opaque (we use a dict containing the cached buffers).
  // For FP8 (tuple input), recv_x is returned as a (data, scales) tuple;
  // we wrap recv_x + recv_x_scales into a tuple here.
  py::dict handle;
  handle["rank_prefix_matrix"] = rankPrefixMatrix;
  handle["channel_prefix_matrix"] = channelPrefixMatrix;
  handle["recv_channel_prefix_matrix"] = recvChannelPrefixMatrix;
  handle["recv_src_idx"] = recvSrcIdx;
  handle["send_head"] = sendHead;
  handle["num_recv_tokens"] = numRecvTokens;

  py::object recvXObj;
  if (isFp8) {
    recvXObj = py::make_tuple(recvX, *recvXScales);
  } else {
    recvXObj = py::cast(recvX);
  }

  std::optional<EventHandle> event;
  if (asyncFinish) {
    event = EventHandle();
  }

  return py::make_tuple(
      recvXObj,
      recvXScales.has_value() ? py::cast(*recvXScales) : py::none(),
      recvTopkIdx.has_value() ? py::cast(*recvTopkIdx) : py::none(),
      recvTopkWeights.has_value() ? py::cast(*recvTopkWeights) : py::none(),
      numRecvTokensPerExpertList,
      handle,
      event.has_value() ? py::cast(*event) : py::none());
}

std::tuple<
    torch::Tensor,
    std::optional<torch::Tensor>,
    std::optional<EventHandle>>
Buffer::intranode_combine(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& topkWeights,
    const std::optional<torch::Tensor>& bias0,
    const std::optional<torch::Tensor>& bias1,
    const py::object& handle,
    const Config& config,
    const std::optional<EventHandle>& /*previousEvent*/,
    bool asyncFinish,
    bool /*allocateOnCommStream*/) {
  if (!available_) {
    throw std::runtime_error("Buffer::intranode_combine: not synced yet");
  }
  TORCH_CHECK(
      x.dim() == 2 && x.is_contiguous() && x.scalar_type() == torch::kBFloat16,
      "Phase 1 combine requires bf16 contiguous 2D x");
  TORCH_CHECK(config.num_sms % 2 == 0, "config.num_sms must be even");

  if (!py::isinstance<py::dict>(handle)) {
    throw std::runtime_error(
        "Buffer::intranode_combine: handle must be the dict returned by "
        "intranode_dispatch");
  }
  auto handleDict = handle.cast<py::dict>();
  auto rankPrefixMatrix =
      handleDict["rank_prefix_matrix"].cast<torch::Tensor>();
  auto channelPrefixMatrix =
      handleDict["channel_prefix_matrix"].cast<torch::Tensor>();
  auto recvSrcIdx = handleDict["recv_src_idx"].cast<torch::Tensor>();
  auto sendHead = handleDict["send_head"].cast<torch::Tensor>();

  const int numTokens = static_cast<int>(x.size(0));
  const int hidden = static_cast<int>(x.size(1));
  const int numRecvTokens = static_cast<int>(sendHead.size(0));

  int numTopk = 0;
  float* topkWeightsPtr = nullptr;
  std::optional<torch::Tensor> recvTopkWeights;
  if (topkWeights.has_value()) {
    TORCH_CHECK(
        topkWeights->dim() == 2 && topkWeights->is_contiguous() &&
            topkWeights->scalar_type() == torch::kFloat32,
        "topk_weights must be 2D float32 contiguous");
    numTopk = static_cast<int>(topkWeights->size(1));
    topkWeightsPtr = topkWeights->data_ptr<float>();
    recvTopkWeights =
        torch::empty({numRecvTokens, numTopk}, topkWeights->options());
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto recvX = torch::empty({numRecvTokens, hidden}, x.options());

  // cached_notify_combine zeroes the per-channel buffers and fills in
  // placeholder `send_head` entries (`-last_head - 1`) for tokens this
  // rank didn't route to a given destination — so the combine kernel
  // can identify which buffer slots actually contain data.
  const int numChannels = config.num_sms / 2;
  // memset region: 2 * num_channels * num_ranks for head/tail per
  // (channel, rank).
  const int numMemsetInt = 2 * numChannels * numRanks_;
  kernels::cached_notify_combine(
      intranode_->getPeerDataPtrsDevice(),
      sendHead.data_ptr<int>(),
      numChannels,
      numRecvTokens,
      numMemsetInt,
      intranode_->getBarrierSignalPtrsDevice(),
      rank_,
      numRanks_,
      stream,
      intranode_->head());
  intranode_->moveFifoSlots(2);

  // num_nvl_bytes must hold the combine NVL layout (Combine.cu): per (channel,
  // rank) head+tail ints + recv slots of data/src-idx/topk-weights. A subset
  // of the dispatch layout; same fail-loud guard as dispatch.
  {
    const std::size_t cr = static_cast<std::size_t>(numChannels) * numRanks_;
    const std::size_t recv =
        static_cast<std::size_t>(config.num_max_nvl_chunked_recv_tokens);
    const std::size_t requiredNvlBytes = cr * 2UL * sizeof(int) +
        cr * recv * static_cast<std::size_t>(hidden) * recvX.element_size() +
        cr * recv * sizeof(int) +
        cr * recv * static_cast<std::size_t>(numTopk) * sizeof(float);
    TORCH_CHECK(
        requiredNvlBytes <= static_cast<std::size_t>(numNvlBytes_),
        "intranode_combine: num_nvl_bytes (",
        numNvlBytes_,
        ") is too small for the NVL layout; need >= ",
        requiredNvlBytes,
        " bytes. Size the Buffer with "
        "Config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks).");
  }

  kernels::intranode_combine(
      recvX.data_ptr(),
      recvTopkWeights.has_value() ? recvTopkWeights->data_ptr<float>()
                                  : nullptr,
      x.data_ptr(),
      topkWeightsPtr,
      bias0.has_value() ? bias0->data_ptr() : nullptr,
      bias1.has_value() ? bias1->data_ptr() : nullptr,
      recvSrcIdx.data_ptr<int>(),
      rankPrefixMatrix.data_ptr<int>(),
      channelPrefixMatrix.data_ptr<int>(),
      sendHead.data_ptr<int>(),
      numTokens,
      numRecvTokens,
      hidden,
      numTopk,
      intranode_->getPeerDataPtrsDevice(),
      rank_,
      numRanks_,
      stream,
      config.num_sms,
      config.num_max_nvl_chunked_send_tokens,
      config.num_max_nvl_chunked_recv_tokens);

  std::optional<EventHandle> event;
  if (asyncFinish) {
    event = EventHandle();
  }
  return std::make_tuple(recvX, recvTopkWeights, event);
}

void Buffer::notImplemented(const char* methodName) const {
  throw std::runtime_error(
      std::string("Buffer::") + methodName +
      " is not yet implemented in this commit (D3 work-in-progress)");
}

} // namespace comms::prims::moe_ep
