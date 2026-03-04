// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdlib>

#include <cuda_runtime.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Sleep.h>
#include <folly/logging/xlog.h>
#include <folly/stop_watch.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h> // @manual
#include <thrift/lib/cpp2/protocol/DebugProtocol.h>

#include "aiplatform/tw_platform/core/MastInfo.h"
#include "comms/analyzer/Analyzer.h"
#include "comms/analyzer/CommDumpPuller.h"
#include "comms/analyzer/if/gen-cpp2/CommsTracingService.h"
#include "comms/analyzer/if/gen-cpp2/CommsTracingService_types_custom_protocol.h"
#include "comms/mccl/integration_tests/CollectiveIntegrationTestMixin.h"
#include "comms/mccl/integration_tests/McclIntegrationTestUtil.h"
#include "comms/mccl/tests/CudaStream.h"
#include "comms/mccl/tests/CudaTestUtil.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/trainer/TrainerContext.h"
#include "ftar/DynMemGpuBuffer.h"
#include "meta/analyzer/NCCLXCommsTracingServiceUtil.h"
#include "servicerouter/client/cpp2/ServiceRouter.h"
#include "tupperware/common/client_factory/FreePortSocket.h"

using namespace facebook;
using namespace std::chrono_literals;
using namespace meta::comms::analyzer;
using namespace ncclx;

namespace {
#define NCCLCHECK(cmd)                                                  \
  do {                                                                  \
    ncclResult_t res = cmd;                                             \
    if (res != ncclSuccess) {                                           \
      XLOGF(FATAL, "NCCL error {} '{}'", res, ncclGetErrorString(res)); \
    }                                                                   \
  } while (0)

#define CUDACHECK(cmd)                                                  \
  do {                                                                  \
    cudaError_t err = cmd;                                              \
    if (err != cudaSuccess) {                                           \
      XLOGF(FATAL, "CUDA error {} '{}'", err, cudaGetErrorString(err)); \
    }                                                                   \
  } while (0)

using namespace ::testing;

// RAII for NCCL Communicator
class NcclComm {
 public:
  NcclComm(int worldSize, int globalRank, ncclUniqueId uniqueID) {
    NCCLCHECK(ncclCommInitRank(&comm_, worldSize, uniqueID, globalRank));
  }

  ~NcclComm() {
    NCCLCHECK(ncclCommDestroy(comm_));
  }

  // Movable, not copyable
  NcclComm(const NcclComm&) = delete;
  NcclComm& operator=(const NcclComm&) = delete;
  NcclComm(NcclComm&& other) noexcept = default;
  NcclComm& operator=(NcclComm&& other) = default;

  ncclComm_t raw() const {
    return comm_;
  }

 private:
  ncclComm_t comm_;
};

class NcclCommsTest : public mccl::CollectiveIntegrationTestMixin, public Test {
 public:
  void SetUp() override {
    int numRanks = 4;

    // Allocate ports for the comms tracing service in each rank
    // The port objects must be kept alive until the ranks bind to the ports,
    // otherwise some other process on the host may bind to the port before the
    // ranks.
    std::vector<uint16_t> commsTracingServicePorts;
    for (int i = 0; i < numRanks; ++i) {
      commsTracingServicePorts_.emplace_back();
      commsTracingServicePorts.push_back(
          commsTracingServicePorts_.back().getPort());
    }
    auto portStr = folly::join(",", commsTracingServicePorts);

    mccl::CollectiveIntegrationTestMixin::SetUp(
        mccl::CollectiveIntegrationTestMixin::Config{
            .numRanks = numRanks,
            .env =
                {
                    "NCCL_HPC_JOB_IDS=",
                    "NCCL_CTRAN_ENABLE=1",
                    "NCCL_CTRAN_REGISTER=async",
                    "NCCL_SENDRECV_ALGO=ctran",
                    "NCCL_CTRAN_ALGO_PROFILING_ENABLE=1",
                    "NCCL_CTRAN_ALGO_PROFILING_LOGGING=pipe:nccl_profiler_algo",
                    "NCCL_CTRAN_ALGO_PROFILING_OUTPUT=scuba",
                    "NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT=100",
                    "NCCL_CTRAN_EX_IB_QP_CONFIG=1048576,16,spray,128",
                    "NCCL_FIRST_COMM_AS_WORLD=1",
                    "NCCL_IB_ADAPTIVE_ROUTING=1",
                    "NCCL_IB_QPS_PER_CONNECTION=16",
                    "NCCL_IB_SPLIT_DATA_ON_QPS=0",
                    "NCCL_CTRAN_IB_VC_MODE=dqplb",
                    "NCCL_CTRAN_IB_QP_CONFIG_XRACK=1048576,16,spray,128",
                    "NCCL_CTRAN_IB_QP_CONFIG_XZONE=1048576,16,spray,128",
                    "NCCL_CTRAN_IB_QP_CONFIG_XDC=1048576,16,spray,128",
                    "NCCL_CTRAN_IB_MAX_QPS=16",
                    "NCCL_CTRAN_IB_QP_MAX_MSGS=128",
                    "NCCL_CTRAN_IB_QP_SCALING_THRESHOLD=524288",
                    "NCCL_BUFFSIZES_CONFIG_ENABLE=1",
                    "NCCL_COLLTRACE_SLOW_COLL_THRESHOLD_BY_PG=ANY:4000",
                    "NCCL_COLLTRACE=trace",
                    "NCCL_PROXYTRACE=trace",
                    "NCCL_ERROR_TRACE_ENABLE=1",
                    "NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED=1",
                    "NCCL_SCUBA_ENABLE_INCLUDE_BACKEND_TOPOLOGY=1",
                    "NCCL_COMM_ABORT_SCOPE=none",
                    "NCCL_COMM_DUMP_ENABLE_PROCESS_GLOBAL_ERRORS=1",
                    "NCCL_IB_ENABLE_REPORT_TO_PROCESS_GLOBAL_ERRORS=1",
                    "NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES=20",
                    "NCCL_LOGGER_MODE=async",
                    "NCCL_COMM_EVENT_LOGGING=pipe:nccl_structured_logging",
                    "NCCL_ERROR_EVENT_LOGGING=pipe:nccl_error_logging",
                    "NCCL_PXN_DISABLE=1",
                    "NCCL_MEMORY_EVENT_LOGGING=pipe:nccl_memory_logging",
                    "NCCL_SLOW_RANK_ENABLE=1",
                    "NCCL_SLOW_RANK_LOGGING=pipe:nccl_profiler_slow_rank",
                    "NCCL_CTRAN_TRANSPORT_PROFILER=1",
                    "NCCL_CONNECT_ROUND_MAX_PEERS=256",
                    "NCCL_IB_TIMEOUT=20",
                    "NCCL_MIN_NCHANNELS=4",
                    "NCCL_NET_OVERHEAD=2750",
                    "NCCL_CTRAN_IB_QP_CONFIG_ALGO=alltoall:1048576,16,dqplb,128,224",
                    "NCCL_DEBUG_SUBSYS=ALL",
                    "NCCL_DEBUG=INFO",
                    "NCCL_ASYNC_ERROR_HANDLING=3",
                    "NCCL_SET_THREAD_NAME=1",
                    // Slab Allocator & Lazy setup channel is not cuda graph
                    // compatible
                    "NCCL_MEM_USE_SLAB_ALLOCATOR=0",
                    "NCCL_RUNTIME_CONNECT=0",
                    "NCCL_LAZY_SETUP_CHANNELS=0",
                    // needed for bootstrapping
                    "NCCL_SOCKET_IFNAME=eth0",
                    "NCCL_CLIENT_SOCKET_IFNAME=eth0",
                    "NCCL_FASTINIT_MODE=none",
                    "NCCL_SOCKET_IPADDR_PREFIX=",
                    // enable commsDumpAll
                    "NCCL_COMMSMONITOR_ENABLE=1",
                    // enable analyzer
                    "NCCL_COMM_TRACING_SERVICE_ENABLE=1",
                    // log more tracing data
                    "NCCL_COLLTRACE_RECORD_MAX_ITERATIONS=10",
                    "NCCL_COLLTRACE_RECORD_MAX=10",
                    "NCCL_COLLTRACE_TRACE_CUDA_GRAPH=true",
                    fmt::format("NCCL_COMMS_TRACING_SERVICE_PORTS={}", portStr),
                },
        });
  }

 private:
  std::vector<facebook::tupperware::FreePortSocket> commsTracingServicePorts_;
};

template <typename Service>
std::unique_ptr<apache::thrift::Client<Service>> getClient(uint16_t port) {
  servicerouter::ClientParams clientParams;
  clientParams.setSingleHost("::1", port);
  clientParams.setProcessingTimeoutMs(std::chrono::milliseconds(120000));
  clientParams.setOverallTimeoutMs(std::chrono::milliseconds(120000));
  return servicerouter::cpp2::getClientFactory()
      .getSRClientUnique<apache::thrift::Client<Service>>(
          "", std::move(clientParams));
}

int64_t getCommServicePortNumberForRank(int rank) {
  char* envVarValue = getenv("NCCL_COMMS_TRACING_SERVICE_PORTS");
  XCHECK(envVarValue != nullptr)
      << "NCCL_COMMS_TRACING_SERVICE_PORTS env var is not set";
  std::vector<std::string_view> ports;
  folly::split(',', envVarValue, ports);
  XCHECK(rank < ports.size())
      << "NCCL_COMMS_TRACING_SERVICE_PORTS env var does not have enough ports";
  return folly::to<int64_t>(ports[rank]);
}

std::string getCommsTracingServicePortsMapStr() {
  char* envVarValue = getenv("NCCL_COMMS_TRACING_SERVICE_PORTS");
  XCHECK(envVarValue != nullptr)
      << "NCCL_COMMS_TRACING_SERVICE_PORTS env var is not set";
  std::vector<std::string_view> ports;
  folly::split(',', envVarValue, ports);
  std::vector<std::string> portParts;
  portParts.reserve(ports.size());
  for (auto rank = 0; rank < ports.size(); ++rank) {
    portParts.push_back(fmt::format("{}#::#{}", rank, ports[rank]));
  }
  return folly::join(",", portParts);
}

folly::coro::Task<PullOneJobResult> runAnalyzerAndGetResults() {
  // Lower thresholds to make the test run faster
  FLAGS_nccl_analyzer_stuck_threshold_coll_default_s = 1s;
  FLAGS_nccl_analyzer_stuck_threshold_coll_pp_s = 5s;
  FLAGS_nccl_analyzer_stuck_threshold_first_coll_s = 5s;
  // needed for analyzer to discover all the ports for the test ranks
  FLAGS_nccl_analyzer_comms_tracing_service_ports_rank_map =
      getCommsTracingServicePortsMapStr();

  auto cpuExecutor = folly::getGlobalCPUExecutor();
  std::vector<folly::Executor::KeepAlive<folly::EventBase>> ioEvbs{
      folly::getGlobalIOExecutor()->getEventBase()};

  folly::coro::Baton stopBaton;
  PullOneJobConfig config{
      .mast_job = "fake_job",
      .task_group_name = "",
      .comm_dump_input_dir = "",
      .parsed_dump = false,
      .command = "analyze",
      .loop = false,
      .only_tasks = 0,
      .comm_dump_output_dir = "",
      .analyze = true,
      .sleep_sec = 10,
      .report_to_scuba = false,
      .report_bad_hosts = false,
      .connection_type = "comms_tracing_service_thrift",
      .expected_verdict_file = "",
      .io_evbs = ioEvbs,
      .cpu_executor = cpuExecutor,
      .stopBaton = stopBaton,
  };

  co_return co_await pull_coro(config);
}

folly::coro::Task<void> runAnalyzerUntilResult(
    PullOneJobResult expectedResult,
    std::chrono::milliseconds timeout) {
  folly::stop_watch<std::chrono::milliseconds> timer;
  while (timer.elapsed() < timeout) {
    auto result = co_await runAnalyzerAndGetResults();
    if (result.badRanks == expectedResult.badRanks) {
      co_return;
    }
    co_await folly::coro::sleep(std::chrono::milliseconds(1000));
  }
  XLOG(FATAL) << "timed out waiting for expected result: "
              << folly::join(",", expectedResult.badRanks);
}

folly::coro::Task<bool> runAnalyzerUntilResultVerdict(
    PullOneJobResult expectedResult,
    std::chrono::milliseconds timeout) {
  folly::stop_watch<std::chrono::milliseconds> timer;
  while (timer.elapsed() < timeout) {
    auto result = co_await runAnalyzerAndGetResults();
    // Right now we only check whether the result verdicts contains all the
    // expected verdict. We can do more strict check in the future
    if (std::ranges::all_of(
            expectedResult.analyzerVerdictType, [&result](auto element) {
              return result.analyzerVerdictType.contains(element);
            })) {
      co_return true;
    }
    co_await folly::coro::sleep(std::chrono::milliseconds(1000));
  }
  co_return false;
}

template <typename Func>
folly::coro::Task<void>
waitUntilCommsForRank(int rank, Func func, std::chrono::milliseconds timeout) {
  folly::stop_watch<std::chrono::milliseconds> timer;
  while (timer.elapsed() < timeout) {
    auto analyzerPortForRank = getCommServicePortNumberForRank(rank);
    auto client = getClient<::comms::CommsTracingService>(analyzerPortForRank);
    ::comms::GetCommsRequest request;
    ::comms::GetCommsResponse response;
    client->sync_getComms(response, request);
    XLOG(DBG2) << "Comm state: " << apache::thrift::debugString(response);

    if (func(response)) {
      co_return;
    }
  }
  XLOG(FATAL) << "timed out waiting for comms predicate to be true, rank: "
              << rank;
}

} // namespace

TEST_F(NcclCommsTest, AnalyzerSuccess) {
  if (isTestDriverProcess()) {
    return;
  }

  int rank = getRank();
  int worldSize = getWorldSize();

  // Initialize CUDA state
  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  XLOG(INFO) << "CUDA device id: " << deviceId;
  mccl::cuda::CudaStream stream;

  // NCCL unique ID must be generated on one node and shared to all
  // others
  std::string uniqueIDKey{"uniqueID"};
  ncclUniqueId ncclUniqueID;
  if (rank == 0) {
    // Rank 0 creates the unique id
    NCCLCHECK(ncclGetUniqueId(&ncclUniqueID));
    mccl::McclIntegrationTestUtil::setKey(
        uniqueIDKey,
        std::string(ncclUniqueID.internal, NCCL_UNIQUE_ID_BYTES),
        std::nullopt);
  } else {
    // Everyone else waits for it
    auto value = mccl::McclIntegrationTestUtil::waitForKey(
        uniqueIDKey, [](const auto& versionAndValue) {
          return versionAndValue.has_value();
        });
    std::memcpy(
        ncclUniqueID.internal, value.value.data(), NCCL_UNIQUE_ID_BYTES);
  }

  // Initialize NCCL communicator
  NcclComm comm(worldSize, rank, ncclUniqueID);

  // Allocate memory on the GPU
  int size = 32;
  facebook::ftar::DynMemGpuBuffer sendBuff(size * sizeof(float));
  facebook::ftar::DynMemGpuBuffer recvBuff(size * sizeof(float));

  // Modify GPU memory
  mccl::McclIntegrationTestUtil::modifyGPUBuffer<float>(size, sendBuff.raw());

  // Perform all-reduce operation
  NCCLCHECK(ncclAllReduce(
      (const void*)sendBuff.raw(),
      (void*)recvBuff.raw(),
      size,
      ncclFloat,
      ncclSum,
      comm.raw(),
      stream.raw()));
  CUDACHECK(cudaStreamSynchronize(stream.raw()));

  // Validate the results of the all reduce
  mccl::McclIntegrationTestUtil::validateGPUBuffer<float>(
      size, recvBuff.raw(), worldSize);

  mccl::McclIntegrationTestUtil::setKey(
      fmt::format("done_with_collective_rank_{}", rank), "1", std::nullopt);

  // Rank 0 checks state of collectives
  if (rank == 0) {
    // Get comm dump state from all ranks
    for (int i = 0; i < worldSize; ++i) {
      // Wait for the rank to be done with collectives
      mccl::McclIntegrationTestUtil::waitForKey(
          fmt::format("done_with_collective_rank_{}", rank),
          [](const auto& versionAndValue) {
            return versionAndValue.has_value();
          });

      // Get the comm dump state for each rank
      auto analyzerPortForRank = getCommServicePortNumberForRank(i);
      auto client =
          getClient<::comms::CommsTracingService>(analyzerPortForRank);
      ::comms::GetCommsRequest request;
      ::comms::GetCommsResponse response;
      client->sync_getComms(response, request);
      XLOG(INFO) << "Comm state: " << apache::thrift::debugString(response);

      folly::coro::blockingWait(waitUntilCommsForRank(
          i,
          [](const auto& response) {
            const auto& commHashMap =
                *response.commsForRank()->ncclParsedEntryMap();
            if (commHashMap.size() != 1) {
              XLOG(INFO) << "Expecting 1 comm, got " << commHashMap.size();
              return false;
            }

            const auto& ncclParsedEntry = commHashMap.begin()->second;
            if (ncclParsedEntry.CT_currentColl().has_value()) {
              XLOG(INFO) << "Expecting no ongoing collectives";
              return false;
            }
            if (ncclParsedEntry.CT_pastColls()->size() != 1) {
              XLOG(INFO) << "Expecting 1 past coll, got "
                         << ncclParsedEntry.CT_pastColls()->size();
              return false;
            }
            return true;
          },
          std::chrono::milliseconds(10000)));
    }

    PullOneJobResult analyzerExpectedResult;
    folly::coro::blockingWait(runAnalyzerUntilResult(
        analyzerExpectedResult, std::chrono::milliseconds(10000)));

    // Notify other ranks to finish
    mccl::McclIntegrationTestUtil::setKey(
        "analyzer_check_0", "1", std::nullopt);
  }

  // All ranks wait for the collectives check to complete
  mccl::McclIntegrationTestUtil::waitForKey(
      "analyzer_check_0",
      [](const auto& versionAndValue) { return versionAndValue.has_value(); });
}

// Rank 0 doesn't join the second
TEST_F(NcclCommsTest, OneRankHangs) {
  if (isTestDriverProcess()) {
    return;
  }

  int rank = getRank();
  int worldSize = getWorldSize();

  // Initialize CUDA state
  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  XLOG(INFO) << "CUDA device id: " << deviceId;
  mccl::cuda::CudaStream stream;

  // NCCL unique ID must be generated on one node and shared to all
  // others
  std::string uniqueIDKey{"uniqueID"};
  ncclUniqueId ncclUniqueID;
  if (rank == 0) {
    // Rank 0 creates the unique id
    NCCLCHECK(ncclGetUniqueId(&ncclUniqueID));
    mccl::McclIntegrationTestUtil::setKey(
        uniqueIDKey,
        std::string(ncclUniqueID.internal, NCCL_UNIQUE_ID_BYTES),
        std::nullopt);
  } else {
    // Everyone else waits for it
    auto value = mccl::McclIntegrationTestUtil::waitForKey(
        uniqueIDKey, [](const auto& versionAndValue) {
          return versionAndValue.has_value();
        });
    std::memcpy(
        ncclUniqueID.internal, value.value.data(), NCCL_UNIQUE_ID_BYTES);
  }

  // Initialize NCCL communicator
  NcclComm comm(worldSize, rank, ncclUniqueID);

  // Allocate memory on the GPU
  int size = 32;
  facebook::ftar::DynMemGpuBuffer sendBuff(size * sizeof(float));
  facebook::ftar::DynMemGpuBuffer recvBuff(size * sizeof(float));

  for (int c = 0; c < 4; ++c) {
    XLOG(INFO) << "Iteration " << c;

    ncclxSetIteration(c);
    // Modify GPU memory
    mccl::McclIntegrationTestUtil::modifyGPUBuffer<float>(size, sendBuff.raw());

    // Rank 0 does not join the last collective
    if (c == 3 && rank == 0) {
      continue;
    }
    NCCLCHECK(ncclAllReduce(
        (const void*)sendBuff.raw(),
        (void*)recvBuff.raw(),
        size,
        ncclFloat,
        ncclSum,
        comm.raw(),
        stream.raw()));
    CUDACHECK(cudaStreamSynchronize(stream.raw()));

    // Validate the results of the all reduce
    mccl::McclIntegrationTestUtil::validateGPUBuffer<float>(
        size, recvBuff.raw(), worldSize);
  }

  // Rank 0 checks state of collectives
  if (rank == 0) {
    for (int i = 0; i < worldSize; ++i) {
      folly::coro::blockingWait(waitUntilCommsForRank(
          i,
          [&i](const auto& response) {
            const auto& commHashMap =
                *response.commsForRank()->ncclParsedEntryMap();
            if (commHashMap.size() != 1) {
              XLOG(INFO) << "Expecting 1 comm, got " << commHashMap.size();
              return false;
            }

            const auto& ncclParsedEntry = commHashMap.begin()->second;
            // Rank 0 is not in the collective
            if (i != 0) {
              if (!ncclParsedEntry.CT_currentColl().has_value()) {
                XLOG(INFO) << "Expecting current collective";
                return false;
              }
            }
            if (ncclParsedEntry.CT_pastColls()->size() != 3) {
              XLOG(INFO) << "Expecting 3 past colls, got "
                         << ncclParsedEntry.CT_pastColls()->size();
              return false;
            }
            return true;
          },
          std::chrono::milliseconds(10000)));
    }

    // Wait for analyzer to report rank 0 is missing
    PullOneJobResult analyzerExpectedResult;
    analyzerExpectedResult.badRanks.insert(0);
    folly::coro::blockingWait(runAnalyzerUntilResult(
        analyzerExpectedResult, std::chrono::milliseconds(20000)));

    // Make rank 0 join the collective to unblock the other ranks
    NCCLCHECK(ncclAllReduce(
        (const void*)sendBuff.raw(),
        (void*)recvBuff.raw(),
        size,
        ncclFloat,
        ncclSum,
        comm.raw(),
        stream.raw()));
    CUDACHECK(cudaStreamSynchronize(stream.raw()));

    // Validate the results of the all reduce
    mccl::McclIntegrationTestUtil::validateGPUBuffer<float>(
        size, recvBuff.raw(), worldSize);
  }
}

// Disabled due to we currently disabled CUDA Graph support for CollTrace due
// to difficulties in ensuring the correct behavior during capture
// (e.g. Graph + normal Launch). Will enable once added back support.
TEST_F(NcclCommsTest, DISABLED_OneRankHangsCudaGraph) {
  if (isTestDriverProcess()) {
    return;
  }

  int rank = getRank();
  int worldSize = getWorldSize();

  // Initialize CUDA state
  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  XLOG(INFO) << "CUDA device id: " << deviceId;
  mccl::cuda::CudaStream stream;

  // NCCL unique ID must be generated on one node and shared to all
  // others
  std::string uniqueIDKey{"uniqueID"};
  ncclUniqueId ncclUniqueID;
  if (rank == 0) {
    // Rank 0 creates the unique id
    NCCLCHECK(ncclGetUniqueId(&ncclUniqueID));
    mccl::McclIntegrationTestUtil::setKey(
        uniqueIDKey,
        std::string(ncclUniqueID.internal, NCCL_UNIQUE_ID_BYTES),
        std::nullopt);
  } else {
    // Everyone else waits for it
    auto value = mccl::McclIntegrationTestUtil::waitForKey(
        uniqueIDKey, [](const auto& versionAndValue) {
          return versionAndValue.has_value();
        });
    std::memcpy(
        ncclUniqueID.internal, value.value.data(), NCCL_UNIQUE_ID_BYTES);
  }

  // Initialize NCCL communicator
  NcclComm comm(worldSize, rank, ncclUniqueID);

  // Allocate memory on the GPU
  int size = 32;
  facebook::ftar::DynMemGpuBuffer sendBuff(size * sizeof(float));
  facebook::ftar::DynMemGpuBuffer recvBuff(size * sizeof(float));

  cudaGraph_t graph;
  cudaGraphExec_t instance;
  CUDACHECK(cudaStreamBeginCapture(stream.raw(), cudaStreamCaptureModeGlobal));
  NCCLCHECK(ncclAllReduce(
      (const void*)sendBuff.raw(),
      (void*)recvBuff.raw(),
      size,
      ncclFloat,
      ncclSum,
      comm.raw(),
      stream.raw()));
  CUDACHECK(cudaStreamEndCapture(stream.raw(), &graph));
  CUDACHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

  for (int c = 0; c < 4; ++c) {
    XLOG(INFO) << "Iteration " << c;

    ncclxSetIteration(c);
    // Modify GPU memory
    mccl::McclIntegrationTestUtil::modifyGPUBuffer<float>(size, sendBuff.raw());

    // Rank 0 does not join the last collective
    if (c == 3 && rank == 0) {
      continue;
    }
    CUDACHECK(cudaGraphLaunch(instance, stream.raw()));
    CUDACHECK(cudaStreamSynchronize(stream.raw()));

    // Validate the results of the all reduce
    mccl::McclIntegrationTestUtil::validateGPUBuffer<float>(
        size, recvBuff.raw(), worldSize);
  }

  // Rank 0 checks state of collectives
  if (rank == 0) {
    for (int i = 0; i < worldSize; ++i) {
      folly::coro::blockingWait(waitUntilCommsForRank(
          i,
          [&i](const auto& response) {
            const auto& commHashMap =
                *response.commsForRank()->ncclParsedEntryMap();
            if (commHashMap.size() != 1) {
              XLOG(INFO) << "Expecting 1 comm, got " << commHashMap.size();
              return false;
            }

            const auto& ncclParsedEntry = commHashMap.begin()->second;
            // Rank 0 is not in the collective
            if (i != 0) {
              if (!ncclParsedEntry.CT_currentColl().has_value()) {
                XLOG(INFO) << "Expecting current collective";
                return false;
              }
            }
            if (ncclParsedEntry.CT_pastColls()->size() != 3) {
              XLOG(INFO) << "Expecting 3 past colls, got "
                         << ncclParsedEntry.CT_pastColls()->size();
              return false;
            }
            return true;
          },
          std::chrono::milliseconds(10000)));
    }

    // Wait for analyzer to report rank 0 is missing
    PullOneJobResult analyzerExpectedResult;
    analyzerExpectedResult.badRanks.insert(0);
    folly::coro::blockingWait(runAnalyzerUntilResult(
        analyzerExpectedResult, std::chrono::milliseconds(20000)));

    // Make rank 0 join the collective to unblock the other ranks
    CUDACHECK(cudaGraphLaunch(instance, stream.raw()));
    CUDACHECK(cudaStreamSynchronize(stream.raw()));

    // Validate the results of the all reduce
    mccl::McclIntegrationTestUtil::validateGPUBuffer<float>(
        size, recvBuff.raw(), worldSize);
  }
  CUDACHECK(cudaGraphExecDestroy(instance));
  CUDACHECK(cudaGraphDestroy(graph));
}

TEST(NCCLXCommsTracingServiceHandlerTest, TestPortConflict) {
  EnvRAII enableTracing(NCCL_COMM_TRACING_SERVICE_ENABLE, true);
  EnvRAII servicePorts(NCCL_COMMS_TRACING_SERVICE_PORTS, {"27702"});
  SysEnvRAII rank("RANK", "0");
  SysEnvRAII localRank("LOCAL_RANK", "0");

  int port = 27702;
  folly::EventBase evb;
  auto server = folly::AsyncServerSocket::newSocket(&evb);
  folly::SocketAddress bindAddr("::", port, true);
  server->bind(bindAddr);
  server->setReusePortEnabled(true);
  server->listen(1);

  // this shouldn't fail, as both sockets use SO_REUSEPORT
  NCCLXCommsTracingServiceUtil::startService();

  // tear down both
  server.reset();
  NCCLXCommsTracingServiceUtil::stopService();

  // bind to the port again exclusively
  server = folly::AsyncServerSocket::newSocket(&evb);
  server->bind(bindAddr);
  server->listen(1);

  ASSERT_THAT(
      NCCLXCommsTracingServiceUtil::startService,
      ThrowsMessage<std::runtime_error>(HasSubstr(
          fmt::format(
              "failed to bind to async server socket: [::]:{}: Address already in use",
              port))));

  // now tell tracing service to warn only instead of crashing
  EnvRAII warningOnly(NCCL_COMM_TRACING_SERVICE_WARN_ON_PORT_CONFLICT, true);
  NCCLXCommsTracingServiceUtil::startService();
}

TEST(NCCLXCommsTracingServiceHandlerTest, TestDynamicPort) {
  EnvRAII enableTracing(NCCL_COMM_TRACING_SERVICE_ENABLE, true);
  EnvRAII servicePorts(NCCL_COMMS_TRACING_SERVICE_PORTS, {"0"});
  SysEnvRAII rank("RANK", "0");
  SysEnvRAII localRank("LOCAL_RANK", "0");

  // bind to a random free port
  NCCLXCommsTracingServiceUtil::startService();
  int actualPort = NCCLXCommsTracingServiceUtil::getPort();

  // ensure we can't bind to the port again
  folly::EventBase evb;
  auto actualPortServer = folly::AsyncServerSocket::newSocket(&evb);
  folly::SocketAddress actualPortAddr("::", actualPort, true);
  ASSERT_THAT(
      ([&actualPortServer, &actualPortAddr]() {
        actualPortServer->bind(actualPortAddr);
      }),
      ThrowsMessage<std::system_error>(HasSubstr("Address already in use")));

  // tear down
  NCCLXCommsTracingServiceUtil::stopService();
}

// FIXME: Currently we don't destory any resources. This is because otherwise
// we won't be able to finish the test due to cuda API being stuck. In the
// future we should try to release the memory and stream at destruction
TEST_F(NcclCommsTest, CollectiveMetadataMismatch) {
  if (isTestDriverProcess()) {
    return;
  }

  int rank = getRank();
  int worldSize = getWorldSize();

  // Initialize CUDA state
  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  XLOG(INFO) << "CUDA device id: " << deviceId;
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  // NCCL unique ID must be generated on one node and shared to all
  // others
  std::string uniqueIDKey{"uniqueID"};
  ncclUniqueId ncclUniqueID;
  if (rank == 0) {
    // Rank 0 creates the unique id
    NCCLCHECK(ncclGetUniqueId(&ncclUniqueID));
    mccl::McclIntegrationTestUtil::setKey(
        uniqueIDKey,
        std::string(ncclUniqueID.internal, NCCL_UNIQUE_ID_BYTES),
        std::nullopt);
  } else {
    // Everyone else waits for it
    auto value = mccl::McclIntegrationTestUtil::waitForKey(
        uniqueIDKey, [](const auto& versionAndValue) {
          return versionAndValue.has_value();
        });
    std::memcpy(
        ncclUniqueID.internal, value.value.data(), NCCL_UNIQUE_ID_BYTES);
  }

  // Initialize NCCL communicator. Could not use RAII because we need to abort
  // it.
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, worldSize, ncclUniqueID, rank));
  // auto abortGuard = folly::makeGuard([&comm]() { ncclCommAbort(comm); });

  // Allocate memory on the GPU
  int size = 32;

  void* sendBuff;
  void* recvBuff;
  cudaMalloc(&sendBuff, size * sizeof(float));
  cudaMalloc(&recvBuff, size * sizeof(float) * worldSize);

  // Intentionally use a different collective on rank 0
  if (rank == 0) {
    NCCLCHECK(ncclAllReduce(
        (const void*)sendBuff,
        (void*)recvBuff,
        size,
        ncclFloat,
        ncclSum,
        comm,
        stream));
  } else {
    NCCLCHECK(ncclAllGather(
        (const void*)sendBuff, (void*)recvBuff, size, ncclFloat, comm, stream));
  }

  // Rank 0 checks state of collectives
  if (rank == 0) {
    // Always make sure notify other ranks before exiting
    auto finishGuard = folly::makeGuard([&comm]() {
      mccl::McclIntegrationTestUtil::setKey(
          "test_return_baton", "Finished", std::nullopt);
    });
    // Wait for analyzer to report rank 0 is missing
    PullOneJobResult analyzerExpectedResult{
        .analyzerVerdictType = {
            AnalyzerVerdict::VerdictType::JOB_CONTAINS_COLL_METADATA_MISMATCH}};
    EXPECT_TRUE(
        folly::coro::blockingWait(runAnalyzerUntilResultVerdict(
            analyzerExpectedResult, std::chrono::milliseconds(20000))));

  } else {
    // Wait for analyzer from rank 0 from returning
    mccl::McclIntegrationTestUtil::waitForKey(
        "test_return_baton", [](const auto& versionAndValue) {
          return versionAndValue.has_value();
        });
  }
  XLOG(INFO) << "Test finished, start to abort NCCL comm";
}

// FIXME: Currently we don't destory any resources. This is because otherwise
// we won't be able to finish the test due to cuda API being stuck. In the
// future we should try to release the memory and stream at destruction
TEST_F(NcclCommsTest, CollectiveCircularDependency) {
  if (isTestDriverProcess()) {
    return;
  }

  int rank = getRank();
  int worldSize = getWorldSize();

  // Initialize CUDA state
  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  XLOG(INFO) << "CUDA device id: " << deviceId;

  // cudaGraph_t graph;
  cudaStream_t stream1;
  cudaStream_t stream2;
  CUDACHECK_TEST(cudaStreamCreate(&stream1));
  CUDACHECK_TEST(cudaStreamCreate(&stream2));

  // NCCL unique ID must be generated on one node and shared to all
  // others
  std::string uniqueIDKey1{"uniqueID1"};
  std::string uniqueIDKey2{"uniqueID2"};
  ncclUniqueId ncclUniqueID1, ncclUniqueID2;
  if (rank == 0) {
    // Rank 0 creates the unique id
    NCCLCHECK(ncclGetUniqueId(&ncclUniqueID1));
    mccl::McclIntegrationTestUtil::setKey(
        uniqueIDKey1,
        std::string(ncclUniqueID1.internal, NCCL_UNIQUE_ID_BYTES),
        std::nullopt);
    NCCLCHECK(ncclGetUniqueId(&ncclUniqueID2));
    mccl::McclIntegrationTestUtil::setKey(
        uniqueIDKey2,
        std::string(ncclUniqueID2.internal, NCCL_UNIQUE_ID_BYTES),
        std::nullopt);
  } else {
    // Everyone else waits for it
    auto value1 = mccl::McclIntegrationTestUtil::waitForKey(
        uniqueIDKey1, [](const auto& versionAndValue) {
          return versionAndValue.has_value();
        });
    std::memcpy(
        ncclUniqueID1.internal, value1.value.data(), NCCL_UNIQUE_ID_BYTES);
    auto value2 = mccl::McclIntegrationTestUtil::waitForKey(
        uniqueIDKey2, [](const auto& versionAndValue) {
          return versionAndValue.has_value();
        });
    std::memcpy(
        ncclUniqueID2.internal, value2.value.data(), NCCL_UNIQUE_ID_BYTES);
    ASSERT_NE(
        strncmp(
            ncclUniqueID1.internal,
            ncclUniqueID2.internal,
            NCCL_UNIQUE_ID_BYTES),
        0);
  }

  // Initialize NCCL communicator. Could not use RAII because we need to abort
  // it.
  ncclComm_t comm1;
  ncclComm_t comm2;
  NCCLCHECK(ncclCommInitRank(&comm1, worldSize, ncclUniqueID1, rank));
  NCCLCHECK(ncclCommInitRank(&comm2, worldSize, ncclUniqueID2, rank));
  // auto abortGuard = folly::makeGuard([&comm]() { ncclCommAbort(comm); });

  cudaEvent_t event;
  CUDACHECK(cudaEventCreate(&event));

  // Allocate memory on the GPU
  int size = 32;

  void* sendBuff1;
  void* recvBuff1;
  void* sendBuff2;
  void* recvBuff2;
  cudaMalloc(&sendBuff1, size * sizeof(float));
  cudaMalloc(&recvBuff1, size * sizeof(float) * worldSize);
  cudaMalloc(&sendBuff2, size * sizeof(float));
  cudaMalloc(&recvBuff2, size * sizeof(float) * worldSize);

  // Pre connect
  NCCLCHECK(ncclAllGather(
      (const void*)sendBuff1,
      (void*)recvBuff1,
      size,
      ncclFloat,
      comm1,
      stream1));

  NCCLCHECK(ncclAllReduce(
      (const void*)sendBuff2,
      (void*)recvBuff2,
      size,
      ncclFloat,
      ncclSum,
      comm2,
      stream2));

  // Intentionally use a different order collective on rank 0
  if (rank == 0) {
    NCCLCHECK(ncclAllGather(
        (const void*)sendBuff1,
        (void*)recvBuff1,
        size,
        ncclFloat,
        comm1,
        stream1));
    cudaEventRecord(event, stream1);
    cudaStreamWaitEvent(stream2, event);
    NCCLCHECK(ncclAllReduce(
        (const void*)sendBuff2,
        (void*)recvBuff2,
        size,
        ncclFloat,
        ncclSum,
        comm2,
        stream2));
  } else {
    NCCLCHECK(ncclAllReduce(
        (const void*)sendBuff2,
        (void*)recvBuff2,
        size,
        ncclFloat,
        ncclSum,
        comm2,
        stream2));
    cudaEventRecord(event, stream2);
    cudaStreamWaitEvent(stream1, event);
    NCCLCHECK(ncclAllGather(
        (const void*)sendBuff1,
        (void*)recvBuff1,
        size,
        ncclFloat,
        comm1,
        stream1));
  }

  // Rank 0 checks state of collectives
  if (rank == 0) {
    // Always make sure notify other ranks before exiting
    auto finishGuard = folly::makeGuard([]() {
      mccl::McclIntegrationTestUtil::setKey(
          "test_return_baton", "Finished", std::nullopt);
    });
    // Wait for analyzer to report rank 0 is missing
    PullOneJobResult analyzerExpectedResult{
        .analyzerVerdictType = {
            AnalyzerVerdict::VerdictType::JOB_CONTAINS_COLL_DEADLOCK}};
    EXPECT_TRUE(
        folly::coro::blockingWait(runAnalyzerUntilResultVerdict(
            analyzerExpectedResult, std::chrono::milliseconds(20000))));

  } else {
    // Wait for analyzer from rank 0 from returning
    mccl::McclIntegrationTestUtil::waitForKey(
        "test_return_baton", [](const auto& versionAndValue) {
          return versionAndValue.has_value();
        });
  }
  XLOG(INFO) << "Test finished, start to abort NCCL comm";
}
