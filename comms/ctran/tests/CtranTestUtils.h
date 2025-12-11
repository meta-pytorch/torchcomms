// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h> // @manual
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <vector>
#include "caffe2/torch/csrc/distributed/c10d/TCPStore.hpp"
#include "comms/ctran/CtranComm.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

namespace ctran {

void logGpuMemoryStats(int gpu);

void commSetMyThreadLoggingName(std::string_view name);

commResult_t commMemAllocDisjoint(
    void** ptr,
    std::vector<size_t>& disjointSegmentSizes,
    std::vector<TestMemSegment>& segments,
    bool setRdmaSupport = true,
    std::optional<CUmemAllocationHandleType> handleType =
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
commResult_t commMemFreeDisjoint(
    void* ptr,
    std::vector<size_t>& disjointSegmentSizes);

void* commMemAlloc(
    size_t bufSize,
    MemAllocType memType,
    std::vector<TestMemSegment>& segments);
void commMemFree(void* buf, size_t bufSize, MemAllocType memType);

// Bootstrap initialization type
enum class InitEnvType { MPI, TCP_STORE };

inline bool checkTcpStoreEnv() {
  // Check if LOCAL_RANK, GLOBAL_RANK, WORLD_SIZE, MASTER_ADDR and MASTER_PORT
  // environment variable is set
  const char* localRankEnv = getenv("LOCAL_RANK");
  const char* globalRankEnv = getenv("GLOBAL_RANK");
  const char* worldSizeEnv = getenv("WORLD_SIZE");
  const char* localSizeEnv = getenv("LOCAL_SIZE");
  const char* masterAddrEnv = getenv("MASTER_ADDR");
  const char* masterPortEnv = getenv("MASTER_PORT");
  return (
      localRankEnv && globalRankEnv && worldSizeEnv && localSizeEnv &&
      masterAddrEnv && masterPortEnv);
}

// Detect which initialization environment to use
InitEnvType getInitEnvType();

class CtranEnvironmentBase : public ::testing::Environment {
 public:
  void SetUp() override;
  void TearDown() override;
};

// CtranDistTestFixture is a fixture for testing Ctran with multiple
// processes/ranks that supports both MPI and TCPStore bootstrap methods.
class CtranDistTestFixture : public ::testing::Test {
 public:
 protected:
  void SetUp() override;
  void TearDown() override;

  std::unique_ptr<CtranComm> makeCtranComm();

  // Rank information
  int globalRank{-1};
  int numRanks{-1};
  int localRank{-1};
  int numLocalRanks_{-1};
  bool enableNolocal{false};

  // CUDA stream for tests (RAII managed)
  std::optional<meta::comms::CudaStream> stream{std::nullopt};

 private:
  void setUpMpi();
  void setUpTcpStore();

  // TCP Store support
  std::unique_ptr<c10d::TCPStore> tcpStore_{nullptr};
  bool isTcpStoreServer() const;
  std::unique_ptr<c10d::TCPStore> createTcpStore(bool isServer);
  std::vector<std::string>
  exchangeInitUrls(const std::string& selfUrl, int numRanks, int selfRank);

  // Test counter for TCP Store key generation
  static std::atomic<int> testCount_;
};

} // namespace ctran
