// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rccl/TorchCommRCCLBootstrap.hpp"
#include <ATen/hip/HIPContext.h> // @manual
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual
#include "rccl.h" // @manual

#include "comms/torchcomms/StoreManager.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/TorchCommUtils.hpp"
#include "comms/torchcomms/rccl/TorchCommRCCL.hpp"

namespace torch {
namespace comms {

// Initialize the static counter
int TorchCommRCCLBootstrap::counter_ = 0;

const std::string kUniqueidXchgMethodAuto = "auto";
const std::string kUniqueidXchgMethodTCPStore = "tcpstore";
const std::string kUniqueidXchgMethodDefault = kUniqueidXchgMethodAuto;

TorchCommRCCLBootstrap::TorchCommRCCLBootstrap(
    c10::intrusive_ptr<c10d::Store> store,
    c10::Device device,
    std::shared_ptr<RcclApi> rccl_api,
    std::shared_ptr<HipApi> hip_api,
    std::chrono::milliseconds timeout)
    : timeout_(timeout),
      store_(store),
      created_internal_store_(false),
      device_(device),
      rccl_api_(rccl_api),
      hip_api_(hip_api) {
  // Query rank and size using the utility function
  auto ranksize = query_ranksize();
  rank_ = ranksize.first;
  comm_size_ = ranksize.second;

  const char* uniqueid_xchg_env =
      std::getenv("TORCHCOMM_RCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD");
  if (uniqueid_xchg_env == nullptr) {
    TC_LOG(INFO, nullptr)
        << "TORCHCOMM_RCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD not set, "
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
    HIP_CHECK(
        hip_api_,
        hip_api_->getDeviceCount(&device_count),
        "Failed to get CUDA device count");

    device_ = c10::Device(c10::kHIP, rank_ % device_count);
    TC_LOG(INFO, nullptr) << "User did not provide device ID; using device Hip:"
                          << device_.index();
  }

  HIP_CHECK(
      hip_api_,
      hip_api_->setDevice(device_.index()),
      "Failed to set device to " + std::to_string(device_.index()));

  // Allocate CUDA memory for a single float32 value used in barrier operations
  HIP_CHECK(
      hip_api_,
      hip_api_->malloc(&barrier_buffer_, sizeof(float)),
      "Failed to allocate barrier buffer");
}

TorchCommRCCLBootstrap::~TorchCommRCCLBootstrap() {
  if (barrier_buffer_ != nullptr) {
    HIP_CHECK_IGNORE(
        hip_api_,
        hip_api_->free(barrier_buffer_),
        "Failed to free barrier buffer");
    barrier_buffer_ = nullptr;
  }
}

std::string TorchCommRCCLBootstrap::getRCCLStoreKey() {
  std::string key = getRCCLStoreKeyPrefix() + std::to_string(counter_);
  counter_++;
  return key;
}

std::string TorchCommRCCLBootstrap::getRCCLStoreKeyPrefix() {
  return "rccl_storekey_";
};

int TorchCommRCCLBootstrap::getRCCLStoreKeyCounter() {
  return counter_;
}

ncclUniqueId TorchCommRCCLBootstrap::exchangeUniqueIdStore() {
  ncclUniqueId uniqueId;

  auto key = getRCCLStoreKey();
  if (rank_ == 0) {
    // Generate unique ID on rank 0
    ncclResult_t ncclErr = rccl_api_->getUniqueId(&uniqueId);
    if (ncclErr != ncclSuccess) {
      throw std::runtime_error(
          "Failed to get NCCL unique ID: " +
          std::string(rccl_api_->getErrorString(ncclErr)));
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

ncclUniqueId TorchCommRCCLBootstrap::exchangeUniqueIdTCPStore(
    std::string_view name) {
  store_ =
      StoreManager::get().getStore(TorchCommRCCL::kBackendName, name, timeout_);
  created_internal_store_ = true;

  return exchangeUniqueIdStore();
}

bool TorchCommRCCLBootstrap::isTCPStoreEnabled() {
  return std::getenv("MASTER_ADDR") && std::getenv("MASTER_PORT");
}

ncclUniqueId TorchCommRCCLBootstrap::exchangeUniqueId(std::string_view name) {
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

void TorchCommRCCLBootstrap::cleanupTCPStore(ncclComm_t nccl_comm) {
  if (created_internal_store_) {
    // Delete the internal store object and do a barrier to ensure that all
    // processes have deleted their store object too.  This way, when we
    // create the next torchcomm, we can use the same port to create a new store
    // object.
    store_.reset();

    auto stream =
        hip_api_->getCurrentHIPStreamMasqueradingAsCUDA(device_.index());
    ncclResult_t result = rccl_api_->allReduce(
        barrier_buffer_,
        barrier_buffer_,
        1,
        ncclFloat32,
        ncclSum,
        nccl_comm,
        stream);
    if (result != ncclSuccess) {
      TC_LOG(ERROR, nullptr)
          << "NCCL AllReduce failed: " << rccl_api_->getErrorString(result);
    }

    HIP_CHECK(
        hip_api_,
        hip_api_->streamSynchronize(stream),
        "Stream synchronization failed");
  }
}

ncclComm_t TorchCommRCCLBootstrap::createNcclComm(const std::string& name) {
  ncclUniqueId uniqueId;
  ncclComm_t nccl_comm = nullptr;

  uniqueId = exchangeUniqueId(name);

  // TODO: add logging on failures and successes
  // TODO: use scalable init
  // TODO: get the local rank
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
#ifdef NCCL_COMM_DESCRIPTION
  // The string only needs to be valid for the duration of the
  // commInitRankConfig call, so we use .c_str() directly.
  config.commDesc = name.c_str();
#endif
  ncclResult_t ncclErr = rccl_api_->commInitRankConfig(
      &nccl_comm, comm_size_, uniqueId, rank_, &config);
  if (ncclErr != ncclSuccess || nccl_comm == nullptr) {
    throw std::runtime_error(
        "Failed to initialize NCCL communicator: " +
        std::string(rccl_api_->getErrorString(ncclErr)));
  }

  cleanupTCPStore(nccl_comm);

  return nccl_comm;
}

} // namespace comms
} // namespace torch
