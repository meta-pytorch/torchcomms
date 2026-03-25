// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ncclx/meta/tests/NcclCommUtils.h"

#include <atomic>
#include <string>

#include "comms/testinfra/TestUtils.h"

namespace ncclx::test {

namespace {
std::atomic<int> commCounter{
    0}; // NOLINT(facebook-avoid-non-const-global-variables)

int getNextCommId() {
  return commCounter.fetch_add(1);
}
} // namespace

ncclComm_t createNcclComm(
    int globalRank,
    int numRanks,
    int devId,
    meta::comms::ITestBootstrap* bootstrap,
    bool isMock,
    ncclConfig_t* customConfig) {
  ncclComm_t comm = nullptr;
  ncclConfig_t config;
  if (customConfig) {
    config = *customConfig;
  } else {
    config = NCCL_CONFIG_INITIALIZER;
  }

  // Generate a unique commDesc per communicator to avoid stale TCPStore keys
  // when fast-init mode reuses the same TCPStore across multiple test cases.
  // Only override when the caller didn't explicitly set commDesc via
  // ncclConfig_t.commDesc or ncclConfig_t.hints.
  const auto commDesc =
      std::string(kNcclUtCommDesc) + "_" + std::to_string(getNextCommId());
  const bool hasCommDescInHints = [&] {
    if (config.hints == nullptr) {
      return false;
    }
    std::string val;
    const auto* hints = static_cast<const ncclx::Hints*>(config.hints);
    return hints->get("commDesc", val) == ncclSuccess;
  }();
  if (config.commDesc == nullptr && !hasCommDescInHints) {
    config.commDesc = commDesc.c_str();
  }

  ncclUniqueId id;
  if (globalRank == 0) {
    NCCLCHECK_TEST(ncclGetUniqueId(&id));
  }
  auto bcastRes =
      bootstrap->broadcast(&id, sizeof(id), 0, globalRank, numRanks);
  if (std::move(bcastRes).get() != 0) {
    LOG(FATAL) << "broadcast ncclUniqueId failed";
  }

  CUDACHECK_TEST(cudaSetDevice(devId));

  auto initRes =
      ncclCommInitRankConfig(&comm, numRanks, id, globalRank, &config);
  if (initRes != ncclInProgress) {
    if (isMock) {
      NCCLCHECKTHROW_TEST(initRes);
    } else {
      NCCLCHECK_TEST(initRes);
    }
  }

  return comm;
}

// NcclCommRAII

NcclCommRAII::NcclCommRAII(
    int globalRank,
    int numRanks,
    int localRank,
    meta::comms::ITestBootstrap* bootstrap,
    bool isMock,
    ncclConfig_t* config) {
  comm_ = createNcclComm(
      globalRank, numRanks, localRank, bootstrap, isMock, config);
}

NcclCommRAII::~NcclCommRAII() {
  NCCLCHECK_TEST(ncclCommDestroy(comm_));
}

ncclComm& NcclCommRAII::operator*() {
  return *comm_;
}

ncclComm_t NcclCommRAII::operator->() const {
  return comm_;
}

ncclComm_t NcclCommRAII::get() const {
  return comm_;
}

NcclCommRAII::operator ncclComm_t() const {
  return comm_;
}

// NcclCommSplitRAII

NcclCommSplitRAII::NcclCommSplitRAII(
    ncclComm_t parentComm,
    int color,
    int key,
    ncclConfig_t* config) {
  const ncclResult_t res =
      ncclCommSplit(parentComm, color, key, &comm_, config);
  NCCLCHECK_TEST(res);
}

NcclCommSplitRAII::~NcclCommSplitRAII() {
  NCCLCHECK_TEST(ncclCommDestroy(comm_));
}

ncclComm& NcclCommSplitRAII::operator*() {
  return *comm_;
}

ncclComm_t NcclCommSplitRAII::operator->() const {
  return comm_;
}

ncclComm_t NcclCommSplitRAII::get() const {
  return comm_;
}

NcclCommSplitRAII::operator ncclComm_t() const {
  return comm_;
}

} // namespace ncclx::test
