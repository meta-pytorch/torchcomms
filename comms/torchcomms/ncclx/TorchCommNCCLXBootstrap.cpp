// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchCommNCCLXBootstrap.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <dlfcn.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual
#include "comms/torchcomms/StoreManager.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/TorchCommUtils.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "nccl.h" // @manual

namespace torch {
namespace comms {

// Initialize the static counter
int TorchCommNCCLXBootstrap::counter_ = 0;

const std::string kUniqueidXchgMethodAuto = "auto";
const std::string kUniqueidXchgMethodTCPStore = "tcpstore";
const std::string kUniqueidXchgMethodDefault = kUniqueidXchgMethodAuto;

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
  auto ranksize = query_ranksize();
  rank_ = ranksize.first;
  comm_size_ = ranksize.second;

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
    int device_count;
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->getDeviceCount(&device_count),
        "Failed to get CUDA device count");

    device_ = c10::Device(c10::kCUDA, rank_ % device_count);
    TC_LOG(INFO, nullptr)
        << "User did not provide device ID; using device cuda:"
        << static_cast<int>(device_.index());
  }

  CUDA_CHECK(
      cuda_api_,
      cuda_api_->setDevice(device_.index()),
      "Failed to set device to " + std::to_string(device_.index()));

  // Allocate CUDA memory for a single float32 value used in barrier operations
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->malloc(&barrier_buffer_, sizeof(float)),
      "Failed to allocate barrier buffer");
}

TorchCommNCCLXBootstrap::~TorchCommNCCLXBootstrap() {
  if (barrier_buffer_ != nullptr) {
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->free(barrier_buffer_),
        "Failed to free barrier buffer");
    barrier_buffer_ = nullptr;
  }
}

std::string TorchCommNCCLXBootstrap::getNCCLXStoreKey() {
  std::string key = getNCCLXStoreKeyPrefix() + std::to_string(counter_);
  counter_++;
  return key;
}

std::string TorchCommNCCLXBootstrap::getNCCLXStoreKeyPrefix() {
  return "ncclx_storekey_";
};

int TorchCommNCCLXBootstrap::getNCCLXStoreKeyCounter() {
  return counter_;
}

ncclUniqueId TorchCommNCCLXBootstrap::exchangeUniqueIdStore() {
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
    auto vec = store_->get(key);
    if (vec.size() != sizeof(ncclUniqueId)) {
      throw std::runtime_error("Invalid NCCL unique ID size");
    }
    uniqueId = *(reinterpret_cast<const ncclUniqueId*>(vec.data()));
  }

  return uniqueId;
}

ncclUniqueId TorchCommNCCLXBootstrap::exchangeUniqueIdTCPStore(
    std::string_view name) {
  store_ = StoreManager::get().getStore(
      TorchCommNCCLX::kBackendName, name, timeout_);
  created_internal_store_ = true;

  return exchangeUniqueIdStore();
}

bool TorchCommNCCLXBootstrap::isTCPStoreEnabled() {
  return std::getenv("MASTER_ADDR") && std::getenv("MASTER_PORT");
}

ncclUniqueId TorchCommNCCLXBootstrap::exchangeUniqueId(std::string_view name) {
  if (store_ != nullptr) {
    return exchangeUniqueIdStore();
  }

  bool is_tcp_store_enabled = isTCPStoreEnabled();
  if (uniqueid_xchg_method_ != kUniqueidXchgMethodAuto &&
      uniqueid_xchg_method_ != kUniqueidXchgMethodTCPStore) {
    throw std::runtime_error(
        "Invalid unique ID exchange method " + uniqueid_xchg_method_);
  }
  if (!is_tcp_store_enabled) {
    throw std::runtime_error("No way to exchange unique ID");
  }
  return exchangeUniqueIdTCPStore(name);
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

// Helper function to populate NCCL config from hints
void populateNcclConfigFromHints(
    ncclConfig_t& config,
    const CommOptions& options,
    const std::string& name) {
  // Iterate over the hints and set the corresponding fields in the config.  For
  // string arguments, NCCLX uses a "const char*" instead of a std::string, so
  // it is hard to figure out the ownership structure.  Here, we create a copy
  // of the string and pass it to NCCLX, so that it is responsible for freeing
  // it.

  for (const auto& [key, val] : options.hints) {
    if (key == "blocking") {
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
      config.netName = strdup(val.c_str());
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
      config.commName = strdup(val.c_str());
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
    } else if (key == "ncclAllGatherAlgo") {
      config.ncclAllGatherAlgo = strdup(val.c_str());
      TC_LOG(INFO, nullptr)
          << "[comm=" << name
          << "] Setting config.ncclAllGatherAlgo=" << config.ncclAllGatherAlgo;
    } else {
      TC_LOG(WARNING)
          << "NCCL hint '" << key
          << "' is not supported in this NCCL version, ignoring for comm '"
          << name << "'";
    }
  }
}

ncclComm_t TorchCommNCCLXBootstrap::createNcclComm(
    const std::string& name,
    const CommOptions& options) {
  ncclUniqueId uniqueId;
  ncclComm_t nccl_comm = nullptr;

  uniqueId = exchangeUniqueId(name);

  // TODO: add logging on failures and successes
  // TODO: use scalable init
  // TODO: get the local rank
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.commDesc = strdup(name.c_str());

  // Populate NCCL config from user-provided hints
  populateNcclConfigFromHints(config, options, name);

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

} // namespace comms
} // namespace torch
