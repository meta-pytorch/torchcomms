#include "comms/torchcomms/xccl/TorchCommXCCLBootstrap.hpp"
#include <ATen/xpu/XPUContext.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual
#include <exception>
#include "comms/torchcomms/StoreManager.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/TorchCommUtils.hpp"
#include "comms/torchcomms/xccl/TorchCommXCCL.hpp"

namespace torch {
namespace comms {

// Initialize the static counter
int TorchCommXCCLBootstrap::counter_ = 0;

const std::string kUniqueidXchgMethodAuto = "auto";
const std::string kUniqueidXchgMethodTCPStore = "tcpstore";
const std::string kUniqueidXchgMethodDefault = kUniqueidXchgMethodAuto;

TorchCommXCCLBootstrap::TorchCommXCCLBootstrap(
    c10::intrusive_ptr<c10d::Store> store,
    c10::Device device,
    std::shared_ptr<XcclApi> xccl_api,
    std::shared_ptr<XpuApi> xpu_api,
    std::chrono::milliseconds timeout)
    : timeout_(timeout),
      store_(store),
      created_internal_store_(false),
      device_(device),
      xccl_api_(xccl_api),
      xpu_api_(xpu_api) {
  // Query rank and size using the utility function
  auto ranksize = query_ranksize();
  rank_ = ranksize.first;
  comm_size_ = ranksize.second;

  const char* uniqueid_xchg_env =
      std::getenv("TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD");
  if (uniqueid_xchg_env == nullptr) {
    TC_LOG(INFO)
        << "TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD not set, "
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
    XPU_CHECK(
        xpu_api_,
        xpu_api_->getDeviceCount(&device_count),
        "Failed to get XPU device count");

    device_ = c10::Device(c10::kXPU, rank_ % device_count);
    TC_LOG(INFO) << "User did not provide device ID; using device xpu:"
                 << static_cast<int>(device_.index());
  }

  XPU_CHECK(
      xpu_api_,
      xpu_api_->setDevice(device_.index()),
      "Failed to set device to " + std::to_string(device_.index()));

  // Allocate XPU memory for a single float32 value used in barrier operations
  XPU_CHECK(
      xpu_api_,
      xpu_api_->malloc(&barrier_buffer_, sizeof(float)),
      "Failed to allocate barrier buffer");
}

TorchCommXCCLBootstrap::~TorchCommXCCLBootstrap() {
  if (barrier_buffer_ != nullptr) {
    try {
      XPU_CHECK(
          xpu_api_,
          xpu_api_->free(barrier_buffer_),
          "Failed to free barrier buffer");
    } catch (const std::exception& e) {
      TC_LOG(ERROR) << e.what();
    }
    barrier_buffer_ = nullptr;
  }
}

std::string TorchCommXCCLBootstrap::getXCCLStoreKey() {
  std::string key = getXCCLStoreKeyPrefix() + std::to_string(counter_);
  counter_++;
  return key;
}

std::string TorchCommXCCLBootstrap::getXCCLStoreKeyPrefix() {
  return "xccl_storekey_";
};

int TorchCommXCCLBootstrap::getXCCLStoreKeyCounter() {
  return counter_;
}

onecclUniqueId TorchCommXCCLBootstrap::exchangeUniqueIdStore() {
  onecclUniqueId uniqueId;

  auto key = getXCCLStoreKey();

  if (rank_ == 0) {
    // Generate unique ID on rank 0
    onecclResult_t xcclErr = xccl_api_->getUniqueId(&uniqueId);

    if (xcclErr != onecclSuccess) {
      throw std::runtime_error(
          "Failed to get XCCL unique ID: " +
          std::string(xccl_api_->getErrorString(xcclErr)));
    }

    // Set the unique ID in the store
    std::vector<uint8_t> vec(
        reinterpret_cast<uint8_t*>(&uniqueId),
        reinterpret_cast<uint8_t*>(&uniqueId) + sizeof(uniqueId));
    store_->set(key, vec);
  } else {
    // Other ranks read the broadcast ID
    auto vec = store_->get(key);

    if (vec.size() != sizeof(onecclUniqueId)) {
      throw std::runtime_error("Invalid XCCL unique ID size");
    }
    uniqueId = *(reinterpret_cast<const onecclUniqueId*>(vec.data()));
  }

  return uniqueId;
}

onecclUniqueId TorchCommXCCLBootstrap::exchangeUniqueIdTCPStore(
    std::string_view name) {
  store_ =
      StoreManager::get().getStore(TorchCommXCCL::kBackendName, name, timeout_);
  created_internal_store_ = true;

  return exchangeUniqueIdStore();
}

bool TorchCommXCCLBootstrap::isTCPStoreEnabled() {
  return std::getenv("MASTER_ADDR") && std::getenv("MASTER_PORT");
}

onecclUniqueId TorchCommXCCLBootstrap::exchangeUniqueId(std::string_view name) {
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

void TorchCommXCCLBootstrap::cleanupTCPStore(onecclComm_t xccl_comm) {
  if (created_internal_store_) {
    // Delete the internal store object and do a barrier to ensure that all
    // processes have deleted their store object too.  This way, when we
    // create the next torchcomm, we can use the same port to create a new store
    // object.
    store_.reset();

    auto stream = xpu_api_->getCurrentXPUStream(device_.index());
    onecclResult_t result = xccl_api_->allReduce(
        barrier_buffer_,
        barrier_buffer_,
        1,
        onecclFloat32,
        onecclSum,
        xccl_comm,
        stream);
    if (result != onecclSuccess) {
      TC_LOG(ERROR) << "XCCL AllReduce failed: "
                    << xccl_api_->getErrorString(result);
    }

    XPU_CHECK(
        xpu_api_,
        xpu_api_->streamSynchronize(stream),
        "Stream synchronization failed");
  }
}

// Helper function to populate XCCL config from hints
void populateXcclConfigFromHints(
    onecclConfig_t& config,
    const CommOptions& options,
    const std::string& name) {
  // Iterate over the hints and set the corresponding fields in the config.  For
  // string arguments, XCCL uses a "const char*" instead of a std::string, so
  // it is hard to figure out the ownership structure.  Here, we create a copy
  // of the string and pass it to XCCL, so that it is responsible for freeing
  // it.

  for (const auto& [key, val] : options.hints) {
    if (key == "blocking") {
      config.blocking = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.blocking=" << config.blocking;
    } else if (key == "cgaClusterSize" || key == "cga_cluster_size") {
      config.cgaClusterSize = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name << "] Setting config.cgaClusterSize="
                   << config.cgaClusterSize;
    } else if (key == "minCTAs" || key == "min_ctas") {
      config.minCTAs = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.minCTAs=" << config.minCTAs;
    } else if (key == "maxCTAs" || key == "max_ctas") {
      config.maxCTAs = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.maxCTAs=" << config.maxCTAs;
    } else if (key == "netName") {
      config.netName = strdup(val.c_str());
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.netName=" << config.netName;
    } else if (key == "splitShare" || key == "split_share") {
      config.splitShare = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.splitShare=" << config.splitShare;
    } else if (
        key == "trafficClass" || key == "traffic_class" || key == "commName" ||
        key == "collnetEnable" || key == "collnet_enable" ||
        key == "CTAPolicy" || key == "cta_policy" || key == "shrinkShare" ||
        key == "nvlsCTAs" || key == "nvls_ctas" ||
        key == "nChannelsPerNetPeer" || key == "n_channels_per_net_peer" ||
        key == "nvlinkCentricSched" || key == "nvlink_centric_sched") {
      TC_LOG(WARNING) << "XCCL hint '" << key
                      << "' is NCCL-specific and not supported by oneCCL, "
                         "ignoring for comm '"
                      << name << "'";
    } else {
      TC_LOG(WARNING)
          << "XCCL hint '" << key
          << "' is not supported in this XCCL version, ignoring for comm '"
          << name << "'";
    }
  }
}

onecclComm_t TorchCommXCCLBootstrap::createXcclComm(
    const std::string& name,
    const CommOptions& options) {
  onecclUniqueId uniqueId;
  onecclComm_t xccl_comm = nullptr;

  uniqueId = exchangeUniqueId(name);

  onecclConfig_t config = ONECCL_CONFIG_INITIALIZER;

  // Populate XCCL config from user-provided hints
  populateXcclConfigFromHints(config, options, name);

  // Set device for oneCCL before initializing communicator
  onecclResult_t xcclErr = xccl_api_->setDevice(device_.index());
  if (xcclErr != onecclSuccess) {
    throw std::runtime_error(
        "Failed to set oneCCL device: " +
        std::string(xccl_api_->getErrorString(xcclErr)));
  }

  xcclErr = xccl_api_->commInitRankConfig(
      &xccl_comm, comm_size_, uniqueId, rank_, &config);

  if (xcclErr != onecclSuccess || xccl_comm == nullptr) {
    throw std::runtime_error(
        "Failed to initialize XCCL communicator: " +
        std::string(xccl_api_->getErrorString(xcclErr)));
  }

  cleanupTCPStore(xccl_comm);

  return xccl_comm;
}

} // namespace comms
} // namespace torch
