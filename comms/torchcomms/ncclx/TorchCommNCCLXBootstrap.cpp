// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <set>
#include <stdexcept>

#include <ATen/cuda/CUDAContext.h>
#include <fmt/core.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLXBootstrap.hpp"
#include "comms/torchcomms/utils/Logging.hpp"
#include "comms/torchcomms/utils/StoreManager.hpp"
#include "comms/torchcomms/utils/Utils.hpp"
#include "nccl.h" // @manual

namespace torch::comms {

// Initialize the static counter
int TorchCommNCCLXBootstrap::counter_ = 0;

namespace {

const std::string kUniqueidXchgMethodAuto = "auto";
const std::string kUniqueidXchgMethodTCPStore = "tcpstore";
const std::string kUniqueidXchgMethodDefault = kUniqueidXchgMethodAuto;

bool isFastInitEnable(const ncclx::Hints& hints) {
  std::string fastInitVal;
  if (hints.get("ncclx::fastInitMode", fastInitVal) == ncclSuccess &&
      std::stoi(fastInitVal) == NCCL_FAST_INIT_MODE_RING) {
    return true;
  }
  const char* env = std::getenv("NCCL_FASTINIT_MODE");
  if (env == nullptr) {
    return false;
  }
  return std::string(env) == "ring_hybrid";
}

} // namespace

TorchCommNCCLXBootstrap::TorchCommNCCLXBootstrap(
    c10::intrusive_ptr<c10d::Store> store,
    c10::Device device,
    std::shared_ptr<NcclxApi> nccl_api,
    std::shared_ptr<CudaApi> cuda_api,
    std::chrono::milliseconds timeout)
    : timeout_(timeout),
      store_(store),
      created_internal_store_(false),
      device_(device),
      nccl_api_(nccl_api),
      cuda_api_(cuda_api) {
  // Query rank and size using the utility function
  auto [rank, comm_size] = query_ranksize();
  rank_ = rank;
  comm_size_ = comm_size;

  const char* uniqueid_xchg_env =
      std::getenv("TORCHCOMM_NCCLX_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD");
  if (uniqueid_xchg_env == nullptr) {
    TC_LOG(INFO, nullptr)
        << "TORCHCOMM_NCCLX_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD not set, "
        << "defaulting to " << kUniqueidXchgMethodDefault;
    uniqueid_xchg_method_ = kUniqueidXchgMethodDefault;
  } else {
    uniqueid_xchg_method_ = uniqueid_xchg_env;
  }
  std::transform(
      uniqueid_xchg_method_.begin(),
      uniqueid_xchg_method_.end(),
      uniqueid_xchg_method_.begin(),
      [](unsigned char c) { return std::tolower(c); });

  if (device_.index() == -1) {
    int device_count{0};
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->getDeviceCount(&device_count),
        "Failed to get CUDA device count");
    if (device_count <= 0) {
      throw std::invalid_argument(
          "No CUDA devices found; please check your CUDA installation");
    }
    TC_LOG(INFO, nullptr) << "Found " << device_count << " CUDA devices";

    device_ = c10::Device(c10::kCUDA, rank_ % device_count);
    TC_LOG(INFO, nullptr)
        << "User did not provide device ID; using device cuda:"
        << static_cast<int>(device_.index());
  }

  CUDA_CHECK(
      cuda_api_,
      cuda_api_->setDevice(device_.index()),
      fmt::format("Failed to set device to {}", device_.index()));

  // Allocate CUDA memory for a single float32 value used in barrier operations
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->malloc(&barrier_buffer_, sizeof(float)),
      "Failed to allocate barrier buffer");
}

TorchCommNCCLXBootstrap::~TorchCommNCCLXBootstrap() noexcept {
  if (barrier_buffer_ != nullptr) {
    CUDA_CHECK_IGNORE(
        cuda_api_,
        cuda_api_->free(barrier_buffer_),
        "Failed to free barrier buffer");
    barrier_buffer_ = nullptr;
  }
}

std::string TorchCommNCCLXBootstrap::getNCCLXStoreKey() {
  std::string key = fmt::format("{}{}", getNCCLXStoreKeyPrefix(), counter_);
  counter_++;
  return key;
}

std::string TorchCommNCCLXBootstrap::getNCCLXStoreKeyPrefix() {
  return "ncclx_storekey_";
};

int TorchCommNCCLXBootstrap::getNCCLXStoreKeyCounter() {
  return counter_;
}

void TorchCommNCCLXBootstrap::createStore(std::string_view name) {
  if (store_ == nullptr) {
    bool is_tcp_store_enabled =
        std::getenv("MASTER_ADDR") && std::getenv("MASTER_PORT");
    if (uniqueid_xchg_method_ != kUniqueidXchgMethodAuto &&
        uniqueid_xchg_method_ != kUniqueidXchgMethodTCPStore) {
      throw std::runtime_error(
          "Invalid unique ID exchange method " + uniqueid_xchg_method_);
    }
    if (!is_tcp_store_enabled) {
      throw std::runtime_error("No way to exchange unique ID");
    }
    store_ = createPrefixStore(std::string(name), timeout_);
    created_internal_store_ = true;
  }
}

ncclUniqueId TorchCommNCCLXBootstrap::exchangeUniqueId() {
  ncclUniqueId uniqueId;

  auto key = getNCCLXStoreKey();
  if (rank_ == 0) {
    // Generate unique ID on rank 0
    ncclResult_t ncclErr = nccl_api_->getUniqueId(&uniqueId);
    if (ncclErr != ncclSuccess) {
      throw std::runtime_error(
          "Failed to get NCCL unique ID: " +
          std::string(nccl_api_->getErrorString(ncclErr)));
    }

    // Set the unique ID in the store
    std::vector<uint8_t> vec(
        reinterpret_cast<uint8_t*>(&uniqueId),
        reinterpret_cast<uint8_t*>(&uniqueId) + sizeof(uniqueId));
    store_->set(key, vec);
  } else {
    // Other ranks read the broadcast ID
    store_->wait({key}, timeout_);
    auto vec = store_->get(key);
    if (vec.size() != sizeof(ncclUniqueId)) {
      throw std::runtime_error("Invalid NCCL unique ID size");
    }
    uniqueId = *(reinterpret_cast<const ncclUniqueId*>(vec.data()));
  }

  return uniqueId;
}

void TorchCommNCCLXBootstrap::cleanupTCPStore(ncclComm_t nccl_comm) {
  if (created_internal_store_) {
    // Delete the internal store object and do a barrier to ensure that all
    // processes have deleted their store object too.  This way, when we
    // create the next torchcomm, we can use the same port to create a new store
    // object.
    store_.reset();

    auto stream = cuda_api_->getCurrentCUDAStream(device_.index());
    ncclResult_t result = nccl_api_->allReduce(
        barrier_buffer_,
        barrier_buffer_,
        1,
        ncclFloat32,
        ncclSum,
        nccl_comm,
        stream);
    if (result != ncclSuccess) {
      TC_LOG(ERROR) << "NCCL AllReduce failed: "
                    << nccl_api_->getErrorString(result);
    }

    CUDA_CHECK(
        cuda_api_,
        cuda_api_->streamSynchronize(stream),
        "Stream synchronization failed");
  }
}

// TorchComm-layer hint keys that are consumed by the backend init code
// (TorchCommNCCLX::init), not by ncclConfig.  Skip them here to avoid
// spurious "unsupported hint" warnings.
static const std::set<std::string> kTorchCommLayerHints = {
    std::string(kHintHighPriorityStream),
    std::string(kHintMaxEventPoolSize),
    std::string(kHintGarbageCollectIntervalMs),
    std::string(kHintEnableCudaGraphSupport),
    std::string(kHintGraphTimeoutCheckIntervalMs),
};

void populateNcclConfig(
    ncclConfig_t& config,
    const CommOptions& options,
    const std::string& name) {
  auto* hints = static_cast<ncclx::Hints*>(config.hints);
  constexpr std::string_view kNcclxPrefix = "ncclx::";

  // Iterate over the hints and set the corresponding fields.  Keys with
  // the "ncclx::" prefix are forwarded to the ncclx::Hints object.  All
  // other keys are matched against upstream NCCL config fields.  For
  // string arguments in the config struct, NCCLX uses a "const char*"
  // instead of a std::string.  The strings only need to be valid for the
  // duration of the ncclCommInitRankConfig call, so we use .c_str()
  // directly.
  for (const auto& [key, val] : options.hints) {
    // NCCLx-specific fields -- pass via ncclx::Hints
    if (key.compare(0, kNcclxPrefix.size(), kNcclxPrefix) == 0) {
      if (key == "ncclx::commDesc") {
        throw std::invalid_argument(
            "ncclx::commDesc must not be set in hints; "
            "it is derived internally");
      }
      if (key == "ncclx::splitGroupRanks") {
        throw std::invalid_argument(
            "ncclx::splitGroupRanks must not be set in hints; "
            "it is derived from the ranks parameter");
      }
      hints->set(key, val);
      TC_LOG(INFO, nullptr)
          << "[comm=" << name << "] Setting hint " << key << "=" << val;
    }
    // Upstream NCCL config fields -- set directly on the config struct
    else if (kTorchCommLayerHints.count(key)) {
      continue;
    } else if (key == "blocking") {
      config.blocking = std::stoi(val);
      TC_LOG(INFO, nullptr) << "[comm=" << name
                            << "] Setting config.blocking=" << config.blocking;
    } else if (key == "cgaClusterSize" || key == "cga_cluster_size") {
      config.cgaClusterSize = std::stoi(val);
      TC_LOG(INFO, nullptr)
          << "[comm=" << name
          << "] Setting config.cgaClusterSize=" << config.cgaClusterSize;
    } else if (key == "minCTAs" || key == "min_ctas") {
      config.minCTAs = std::stoi(val);
      TC_LOG(INFO, nullptr)
          << "[comm=" << name << "] Setting config.minCTAs=" << config.minCTAs;
    } else if (key == "maxCTAs" || key == "max_ctas") {
      config.maxCTAs = std::stoi(val);
      TC_LOG(INFO, nullptr)
          << "[comm=" << name << "] Setting config.maxCTAs=" << config.maxCTAs;
    } else if (key == "netName") {
      config.netName = val.c_str();
      TC_LOG(INFO, nullptr)
          << "[comm=" << name << "] Setting config.netName=" << config.netName;
    } else if (key == "splitShare" || key == "split_share") {
      config.splitShare = std::stoi(val);
      TC_LOG(INFO, nullptr)
          << "[comm=" << name
          << "] Setting config.splitShare=" << config.splitShare;
    } else if (key == "trafficClass" || key == "traffic_class") {
      config.trafficClass = std::stoi(val);
      TC_LOG(INFO, nullptr)
          << "[comm=" << name
          << "] Setting config.trafficClass=" << config.trafficClass;
    } else if (key == "commName") {
      config.commName = val.c_str();
      TC_LOG(INFO, nullptr) << "[comm=" << name
                            << "] Setting config.commName=" << config.commName;
    } else if (key == "collnetEnable" || key == "collnet_enable") {
      config.collnetEnable = std::stoi(val);
      TC_LOG(INFO, nullptr)
          << "[comm=" << name
          << "] Setting config.collnetEnable=" << config.collnetEnable;
    } else if (key == "CTAPolicy" || key == "cta_policy") {
      config.CTAPolicy = std::stoi(val);
      TC_LOG(INFO, nullptr)
          << "[comm=" << name
          << "] Setting config.CTAPolicy=" << config.CTAPolicy;
    } else if (key == "shrinkShare") {
      config.shrinkShare = std::stoi(val);
      TC_LOG(INFO, nullptr)
          << "[comm=" << name
          << "] Setting config.shrinkShare=" << config.shrinkShare;
    } else if (key == "nvlsCTAs" || key == "nvls_ctas") {
      config.nvlsCTAs = std::stoi(val);
      TC_LOG(INFO, nullptr) << "[comm=" << name
                            << "] Setting config.nvlsCTAs=" << config.nvlsCTAs;
    } else {
      TC_LOG(WARNING)
          << "NCCL hint '" << key
          << "' is not supported in this NCCL version, ignoring for comm '"
          << name << "'";
    }
  }
}

bool TorchCommNCCLXBootstrap::useFastInit(const ncclx::Hints& hints) {
  if (isFastInitEnable(hints)) {
    // Use raw dynamic_cast instead of c10::dynamic_intrusive_pointer_cast
    // because the latter has a refcount leak when the cast fails (the
    // by-value intrusive_ptr parameter is release()'d before the cast,
    // and if dynamic_cast returns nullptr, the original pointer is lost
    // without decrementing the refcount).
    bool isTcpStore = [this]() {
      if (store_ == nullptr) {
        return false;
      }
      if (auto* prefixStore = dynamic_cast<c10d::PrefixStore*>(store_.get())) {
        return dynamic_cast<c10d::TCPStore*>(
                   prefixStore->getUnderlyingNonPrefixStore().get()) != nullptr;
      }
      return dynamic_cast<c10d::TCPStore*>(store_.get()) != nullptr;
    }();
    if (!isTcpStore) {
      throw std::invalid_argument("TcpStore is required for fast init");
    }
    return isTcpStore;
  }
  return false;
}

ncclComm_t TorchCommNCCLXBootstrap::createNcclComm(
    const std::string& name,
    const CommOptions& options) {
  ncclUniqueId uniqueId;
  ncclComm_t nccl_comm = nullptr;

  // TODO: add logging on failures and successes
  // TODO: use scalable init
  // TODO: get the local rank
  createStore(name);

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  config.hints = &hints;
  populateNcclConfig(config, options, name);
  hints.set("ncclx::commDesc", name);

  if (useFastInit(hints)) {
    uniqueId = ncclUniqueId{};
  } else {
    uniqueId = exchangeUniqueId();
  }

  ncclResult_t ncclErr = nccl_api_->commInitRankConfig(
      &nccl_comm, comm_size_, uniqueId, rank_, &config);
  if (ncclErr != ncclSuccess || nccl_comm == nullptr) {
    throw std::runtime_error(
        "Failed to initialize NCCL communicator: " +
        std::string(nccl_api_->getErrorString(ncclErr)));
  }

  cleanupTCPStore(nccl_comm);

  return nccl_comm;
}

} // namespace torch::comms
