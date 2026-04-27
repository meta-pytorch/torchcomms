#include "comms/torchcomms/hccl/TorchCommHCCLBootstrap.hpp"
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual
#include <chrono>
#include <cstring>
#include <exception>
#include <thread>
#include "comms/torchcomms/StoreManager.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/TorchCommUtils.hpp"
#include "comms/torchcomms/hccl/TorchCommHCCL.hpp"

namespace torch::comms {

// Initialize the static counter
int TorchCommHCCLBootstrap::counter_ = 0;

const std::string kUniqueidXchgMethodAuto = "auto";
const std::string kUniqueidXchgMethodTCPStore = "tcpstore";
const std::string kUniqueidXchgMethodDefault = kUniqueidXchgMethodAuto;

TorchCommHCCLBootstrap::TorchCommHCCLBootstrap(
    c10::intrusive_ptr<c10d::Store> store,
    c10::Device device,
    std::shared_ptr<HcclApi> hccl_api,
    std::shared_ptr<NpuApi> npu_api,
    std::chrono::milliseconds timeout)
    : timeout_(timeout),
      store_(store),
      created_internal_store_(false),
      device_(device),
      hccl_api_(hccl_api),
      npu_api_(npu_api) {
  // Query rank and size using the utility function
  auto ranksize = query_ranksize();
  rank_ = ranksize.first;
  comm_size_ = ranksize.second;

  const char* uniqueid_xchg_env =
      std::getenv("TORCHCOMM_HCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD");
  if (uniqueid_xchg_env == nullptr) {
    TC_LOG(INFO)
        << "TORCHCOMM_HCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD not set, "
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
    NPU_CHECK(
        npu_api_,
        npu_api_->getDeviceCount(&device_count),
        "Failed to get NPU device count");

    device_ = c10::Device(c10::kPrivateUse1, rank_ % device_count);
    TC_LOG(INFO) << "User did not provide device ID; using device npu:"
                 << static_cast<int>(device_.index());
  }

  NPU_CHECK(
      npu_api_,
      npu_api_->setDevice(device_.index()),
      "Failed to set device to " + std::to_string(device_.index()));

  // Allocate NPU memory for a single float32 value used in barrier operations
  NPU_CHECK(
      npu_api_,
      npu_api_->malloc(&barrier_buffer_, sizeof(float)),
      "Failed to allocate barrier buffer");
}

TorchCommHCCLBootstrap::~TorchCommHCCLBootstrap() {
  if (barrier_buffer_ != nullptr) {
    try {
      NPU_CHECK(
          npu_api_,
          npu_api_->free(barrier_buffer_),
          "Failed to free barrier buffer");
    } catch (const std::exception& e) {
      TC_LOG(ERROR) << e.what();
    }
    barrier_buffer_ = nullptr;
  }
}

std::string TorchCommHCCLBootstrap::getHCCLStoreKey() {
  std::string key = getHCCLStoreKeyPrefix() + std::to_string(counter_);
  counter_++;
  return key;
}

std::string TorchCommHCCLBootstrap::getHCCLStoreKeyPrefix() {
  return "hccl_storekey_";
};

int TorchCommHCCLBootstrap::getHCCLStoreKeyCounter() {
  return counter_;
}

HcclRootInfo TorchCommHCCLBootstrap::exchangeUniqueIdStore() {
  HcclRootInfo uniqueId;

  auto key = getHCCLStoreKey();

  if (rank_ == 0) {
    // Generate unique ID on rank 0
    HcclResult hcclErr = hccl_api_->getUniqueId(&uniqueId);

    if (hcclErr != HCCL_SUCCESS) {
      throw std::runtime_error(
          "Failed to get HCCL unique ID: " +
          std::string(hccl_api_->getErrorString(hcclErr)));
    }

    // Set the unique ID in the store
    std::vector<uint8_t> vec(
        reinterpret_cast<uint8_t*>(&uniqueId),
        reinterpret_cast<uint8_t*>(&uniqueId) + sizeof(uniqueId));
    store_->set(key, vec);
  } else {
    // Other ranks read the broadcast ID
    auto vec = store_->get(key);

    if (vec.size() != sizeof(HcclRootInfo)) {
      throw std::runtime_error("Invalid HCCL unique ID size");
    }
    uniqueId = *(reinterpret_cast<const HcclRootInfo*>(vec.data()));
  }

  return uniqueId;
}

HcclRootInfo TorchCommHCCLBootstrap::exchangeUniqueIdTCPStore(
    std::string_view name) {
  store_ =
      StoreManager::get().getStore(TorchCommHCCL::kBackendName, name, timeout_);
  created_internal_store_ = true;

  return exchangeUniqueIdStore();
}

bool TorchCommHCCLBootstrap::isTCPStoreEnabled() {
  return std::getenv("MASTER_ADDR") && std::getenv("MASTER_PORT");
}

HcclRootInfo TorchCommHCCLBootstrap::exchangeUniqueId(std::string_view name) {
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

void TorchCommHCCLBootstrap::cleanupTCPStore(HcclComm hccl_comm) {
  if (created_internal_store_) {
    // Delete the internal store object and do a barrier to ensure that all
    // processes have deleted their store object too.  This way, when we
    // create the next torchcomm, we can use the same port to create a new store
    // object.
    store_.reset();

    auto stream = npu_api_->getCurrentNPUStream(device_.index());
    HcclResult result = hccl_api_->allReduce(
        barrier_buffer_,
        barrier_buffer_,
        1,
        HCCL_DATA_TYPE_FP32,
        HCCL_REDUCE_SUM,
        hccl_comm,
        stream);
    if (result != HCCL_SUCCESS) {
      TC_LOG(ERROR) << "HCCL AllReduce failed: "
                    << hccl_api_->getErrorString(result);
    }

    NPU_CHECK(
        npu_api_,
        npu_api_->streamSynchronize(stream),
        "Stream synchronization failed");
  }
}

// Helper function to populate HCCL config from hints
void populateHcclConfigFromHints(
    HcclCommConfig& config,
    const CommOptions& options,
    const std::string& name) {
  // Iterate over the hints and set the corresponding fields in the config
  for (const auto& [key, val] : options.hints) {
    if (key == "deterministic") {
      config.hcclDeterministic = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name << "] Setting config.hcclDeterministic="
                   << config.hcclDeterministic;
    } else if (key == "hcclBufferSize" || key == "hccl_buffer_size") {
      TC_LOG(INFO)
          << "[comm=" << name
          << "] HCCL hint 'hcclBufferSize' is recognized but may not be applicable";
    } else if (
        key == "blocking" || key == "cgaClusterSize" ||
        key == "cga_cluster_size" || key == "minCTAs" || key == "min_ctas" ||
        key == "maxCTAs" || key == "max_ctas" || key == "netName" ||
        key == "splitShare" || key == "split_share" || key == "trafficClass" ||
        key == "traffic_class" || key == "commName" || key == "collnetEnable" ||
        key == "collnet_enable" || key == "CTAPolicy" || key == "cta_policy" ||
        key == "shrinkShare" || key == "nvlsCTAs" || key == "nvls_ctas" ||
        key == "nChannelsPerNetPeer" || key == "n_channels_per_net_peer" ||
        key == "nvlinkCentricSched" || key == "nvlink_centric_sched") {
      TC_LOG(WARNING) << "HCCL hint '" << key
                      << "' is NCCL/XCCL-specific and not supported by HCCL, "
                         "ignoring for comm '"
                      << name << "'";
    } else {
      TC_LOG(WARNING)
          << "HCCL hint '" << key
          << "' is not supported in this HCCL version, ignoring for comm '"
          << name << "'";
    }
  }
}

HcclComm TorchCommHCCLBootstrap::createHcclComm(
    const std::string& name,
    const CommOptions& options) {
  HcclRootInfo uniqueId;
  HcclComm hccl_comm = nullptr;

  uniqueId = exchangeUniqueId(name);

  HcclCommConfig config;
  // Initialize config with HCCL defaults before applying hints
  HcclCommConfigInit(&config);

  // NOTE: root info must be identical across ranks.
  // We generate it on rank 0 and broadcast via store_ in exchangeUniqueId().
  // Do NOT call HcclGetRootInfo() on every rank here.
  HcclResult hcclErr = hccl_api_->commInitRankConfig(
      &hccl_comm, comm_size_, uniqueId, rank_, &config);

  if (hcclErr != HCCL_SUCCESS || hccl_comm == nullptr) {
    throw std::runtime_error(
        "Failed to initialize HCCL communicator: " +
        std::string(hccl_api_->getErrorString(hcclErr)));
  }

  cleanupTCPStore(hccl_comm);

  return hccl_comm;
}

} // namespace torch::comms
