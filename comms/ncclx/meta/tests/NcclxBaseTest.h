// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "comms/testinfra/DistTestBase.h"
#include "comms/testinfra/TestUtils.h"
#include "nccl.h" // @manual

// TODO: Long-term, migrate all tests from #ifdef compiler flags to NcclxEnvs
// params.
using NcclxEnvs = std::vector<std::pair<std::string, std::string>>;

class NcclxBaseTestFixture : public ::testing::Test,
                             protected meta::comms::DistBaseTest {
 public:
  ncclComm_t comm{nullptr};

 protected:
  // TODO: Migrate tests away from #ifdef compiler flags to NcclxEnvs params.
  // Keep compiler-flag overrides only for envs that must override dist setup
  // (e.g., TEST_ENABLE_FASTINIT to enforce TCPStore).
  void SetUp() override {
    NcclxEnvs envs;
#ifdef NCCL_COMM_STATE_DEBUG_TOPO_NOLOCAL
    envs.push_back({"NCCL_COMM_STATE_DEBUG_TOPO", "nolocal"});
#endif
#ifdef NCCL_COMM_STATE_DEBUG_TOPO_VNODE
    envs.push_back({"NCCL_COMM_STATE_DEBUG_TOPO", "vnode"});
#endif
#ifdef TEST_ENABLE_FASTINIT
    envs.push_back({"NCCL_FASTINIT_MODE", "ring_hybrid"});
#endif
#ifdef TEST_ENABLE_CTRAN
    envs.push_back({"NCCL_CTRAN_ENABLE", "1"});
    envs.push_back({"NCCL_CTRAN_IPC_REGCACHE_ENABLE_ASYNC_SOCKET", "1"});
#endif
#ifdef TEST_ENABLE_LOCAL_REGISTER
    envs.push_back({"NCCL_LOCAL_REGISTER", "1"});
#endif
#ifdef TEST_CUDA_GRAPH_MODE
    envs.push_back({"NCCL_CTRAN_ALLOW_CUDA_GRAPH", "1"});
#endif
    SetUp(envs);
  }

  void SetUp(const NcclxEnvs& envs);
  void TearDown() override;

  bool initEnvAtSetup{true};

 private:
  std::unordered_map<std::string, std::optional<std::string>> oldEnvs_;
};
