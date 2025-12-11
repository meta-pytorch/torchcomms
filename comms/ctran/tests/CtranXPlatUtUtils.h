// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include "caffe2/torch/csrc/distributed/c10d/TCPStore.hpp"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/mccl/bootstrap/Bootstrap.h"
#include "comms/mccl/bootstrap/CtranAdapter.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/commSpecs.h"

class TestCtranCommRAII {
 public:
  TestCtranCommRAII(std::unique_ptr<CtranComm> ctranComm)
      : ctranComm(std::move(ctranComm)) {}
  std::unique_ptr<CtranComm> ctranComm{nullptr};
  std::shared_ptr<mccl::bootstrap::Bootstrap> bootstrap_{nullptr};

  ~TestCtranCommRAII() {
    if (ctranComm) {
      ctranComm.reset();
    }
  }
};

std::unique_ptr<TestCtranCommRAII> createDummyCtranComm(int devId = 0);

static std::atomic<int> testCount = 0;
inline void incrTestCount() {
  testCount.fetch_add(1);
}

enum class TcpStorePhase { INIT, START, END };

inline std::string getTcpStoreKey(enum TcpStorePhase phase) {
  auto keyPrefix = std::string("commid_") + std::to_string(testCount);
  switch (phase) {
    case TcpStorePhase::INIT:
      return keyPrefix;
    case TcpStorePhase::START:
      return keyPrefix + "_start";
    case TcpStorePhase::END:
      return keyPrefix + "_end";
  }
}

inline std::unique_ptr<c10d::TCPStore> createTcpStore(bool isServer) {
  const char* masterAddrStr = getenv("MASTER_ADDR");
  const char* masterPortStr = getenv("MASTER_PORT");
  if (!masterAddrStr) {
    LOG(FATAL) << "MASTER_ADDR env variable is not set";
  }
  if (!masterPortStr) {
    LOG(FATAL) << "MASTER_PORT env variable is not set";
  }

  incrTestCount();
  auto key = fmt::format("test_tcpstore_init_{}", testCount);

  const std::string masterAddr(masterAddrStr);
  c10d::TCPStoreOptions opts;
  opts.port = std::stoi(masterPortStr);
  opts.waitWorkers = false;
  opts.useLibUV = true;
  opts.isServer = isServer;

  LOG(INFO) << "TCPStore "
            << (isServer ? "server starting on " : "client connecting to ")
            << masterAddr << ":" << opts.port << " ..." << " using key " << key;

  if (isServer) {
    auto server = std::make_unique<c10d::TCPStore>(masterAddr, opts);
    server->set(key, {1});
    LOG(INFO) << "TCPStore server started.";
    return server;
  }

  // TCPStore Client may start before fresh TCPStore Server has started
  // We need to retry until we connect to a fresh TCPStore Server
  while (true) {
    try {
      auto server = std::make_unique<c10d::TCPStore>(masterAddr, opts);
      if (server->check({key})) {
        LOG(INFO) << "TCPStore client started.";
        return server;
      }
    } catch (...) {
      LOG(INFO) << "Connected to stale TCPStore Server. "
                << "Waiting for fresh TCPStore Server to start.";
      std::this_thread::sleep_for(
          std::chrono::milliseconds{100}); // Sleep for 100ms
    }
  }
}

// Helper struct to hold bootstrap that needs to stay alive with the CtranComm
struct CtranCommWithBootstrap {
  std::shared_ptr<mccl::bootstrap::Bootstrap> bootstrap;
  std::unique_ptr<CtranComm> ctranComm;
};

inline CtranCommWithBootstrap createCtranCommWithBootstrap(
    int rank,
    int nRanks,
    uint64_t commId = 22,
    int commHash = -1,
    std::string_view commDesc = "ctran_comm_raii_comm_desc") {
  int cudaDev;
  CUDACHECK_TEST(cudaGetDevice(&cudaDev));

  COMMCHECK_TEST(ctran::utils::commCudaLibraryInit());

  std::unique_ptr<CtranComm> ctranComm = std::make_unique<CtranComm>(
      ::ctran::utils::createAbort(/*enabled=*/false));

  // Create and initialize bootstrap; needed for CTRAN backend initialization
  auto bootstrap = std::make_shared<mccl::bootstrap::Bootstrap>(
      NCCL_SOCKET_IFNAME,
      mccl::bootstrap::Options{
          .port = 0, .ifAddrPrefix = NCCL_SOCKET_IPADDR_PREFIX});

  const std::string selfUrl = bootstrap->semi_getInitUrl().get();
  std::vector<std::string> urls(nRanks);
  urls[rank] = selfUrl;

  // For single-rank case, just use our own URL
  // For multi-rank case, caller should use exchangeInitUrls to get all URLs
  if (nRanks == 1) {
    bootstrap->init(urls, rank, /*uuid=*/0);
  }

  ctranComm->bootstrap_ =
      std::make_unique<mccl::bootstrap::CtranAdapter>(bootstrap);

  ctranComm->logMetaData_.commId = commId;
  ctranComm->logMetaData_.commHash = commHash;
  ctranComm->logMetaData_.commDesc = std::string(commDesc);
  ctranComm->logMetaData_.rank = rank;
  ctranComm->logMetaData_.nRanks = nRanks;

  const int cudaArch = ctran::utils::getCudaArch(cudaDev).value_or(-1);
  const int64_t busId = ctran::utils::BusId::makeFrom(cudaDev).toInt64();

  std::vector<ncclx::RankTopology> rankTopologies{};
  std::vector<int> commRanksToWorldRanks{};
  ctranComm->statex_ = std::make_unique<ncclx::CommStateX>(
      rank,
      nRanks,
      cudaDev,
      cudaArch,
      busId,
      commHash,
      std::move(rankTopologies),
      std::move(commRanksToWorldRanks),
      std::string{commDesc});

  // For single-rank communicators (nRanks=1), use nolocal topology mode
  // which doesn't require bootstrap communication.
  if (nRanks == 1) {
    ctranComm->statex_->initRankTopologyNolocal();
  }

  FB_COMMCHECKTHROW(ctranInit(ctranComm.get()));

  return CtranCommWithBootstrap{
      .bootstrap = std::move(bootstrap),
      .ctranComm = std::move(ctranComm),
  };
}
