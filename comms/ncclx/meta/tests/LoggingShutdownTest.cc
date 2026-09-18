// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Two rank processes initialize a communicator and tear it down in the mode
// selected by the test target, logging a sentinel at each step. The driver
// process outlives the ranks, so it observes what happens after each rank's
// last gtest assertion: it checks the rank exit statuses and scans the
// NCCL_DEBUG_FILE logs the ranks left behind.

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <folly/Random.h>
#include <folly/testing/TestUtil.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "nccl.h" // @manual

#include "comms/mccl/integration_tests/CollectiveIntegrationTestMixin.h"
#include "comms/testinfra/ShutdownLogScan.h"
#include "comms/utils/logger/SpdlogLogger.h"

namespace {

using meta::comms::testing::countLogsContaining;
using meta::comms::testing::describeCrashMarkers;
using meta::comms::testing::describeLogDir;
using meta::comms::testing::findCrashMarkers;
using meta::comms::testing::findTruncatedLogs;

constexpr std::string_view kLoggerName{"comms.ncclx"};

/*
 * Sentinels are logged at WARN so NCCL_DEBUG=INFO cannot gate them away, and
 * they are matched by the driver as plain substrings, so they must not appear
 * anywhere else in a rank's log.
 */
constexpr std::string_view kAfterDestroy{"LoggingShutdown phase=after_destroy"};
constexpr std::string_view kAfterAbort{"LoggingShutdown phase=after_abort"};
constexpr std::string_view kBeforeLeak{
    "LoggingShutdown phase=before_exit_without_destroy"};
constexpr std::string_view kAtExit{"LoggingShutdown phase=atexit"};
constexpr std::string_view kStaticDestruction{
    "LoggingShutdown phase=static_destruction"};

constexpr std::string_view kDestroyMode{"destroy"};
constexpr std::string_view kAbortMode{"abort"};
constexpr std::string_view kLeakMode{"leak"};

bool& workerBodyCompleted() {
  static bool completed = false;
  return completed;
}

void logSentinel(std::string_view sentinel) {
  auto& logger = ::meta::comms::logger::getSpdlogLogger(kLoggerName);
  COMMS_LOG_NAMED(kLoggerName, WARN, "{}", sentinel);
  // Without this an async sink could drop the record purely because the process
  // is exiting, which would be indistinguishable from a shutdown bug.
  logger.flush();
}

void logAtExitSentinel() {
  if (workerBodyCompleted()) {
    logSentinel(kAtExit);
  }
}

/*
 * Logging from a static destructor is the contract the comms spdlog facade is
 * built for -- its logger objects are deliberately leaked so threads that were
 * never joined can still log this late. This object is the last sentinel, so it
 * runs after the atexit handler registered by the test body.
 */
struct StaticDestructionSentinel {
  ~StaticDestructionSentinel() {
    if (workerBodyCompleted()) {
      logSentinel(kStaticDestruction);
    }
  }
};

const StaticDestructionSentinel staticDestructionSentinel{};

void storeBarrier(
    const std::shared_ptr<c10d::TCPStore>& store,
    std::string_view tag,
    int rank,
    int worldSize) {
  store->set(
      fmt::format("barrier_{}_{}", tag, rank), std::vector<uint8_t>{'1'});
  std::vector<std::string> keys;
  keys.reserve(worldSize);
  for (int peer = 0; peer < worldSize; ++peer) {
    keys.push_back(fmt::format("barrier_{}_{}", tag, peer));
  }
  store->wait(keys);
}

// Rank 0 creates the id; every rank picks it up under a mode-tagged key.
std::optional<ncclUniqueId> exchangeUniqueId(
    const std::shared_ptr<c10d::TCPStore>& store,
    std::string_view phase,
    int rank) {
  const auto key = fmt::format("nccl_unique_id_{}", phase);
  ncclUniqueId id{};
  std::vector<uint8_t> payload(sizeof(id) + 1, 0);
  if (rank == 0) {
    if (ncclGetUniqueId(&id) == ncclSuccess) {
      payload[0] = 1;
      std::memcpy(payload.data() + 1, &id, sizeof(id));
    }
    store->set(key, payload);
  } else {
    payload = store->get(key);
  }
  if (payload.size() != sizeof(id) + 1 || payload[0] != 1) {
    return std::nullopt;
  }
  std::memcpy(&id, payload.data() + 1, sizeof(id));
  return id;
}

// Tag the communicator so diagnostics identify the selected teardown mode.
ncclComm_t initComm(
    const ncclUniqueId& id,
    const char* commDesc,
    int rank,
    int worldSize) {
  ncclComm_t comm = nullptr;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.commDesc = commDesc;
  if (ncclCommInitRankConfig(&comm, worldSize, id, rank, &config) !=
      ncclSuccess) {
    return nullptr;
  }
  return comm;
}

} // namespace

class LoggingShutdownTest : public mccl::CollectiveIntegrationTestMixin,
                            public ::testing::Test {
 public:
  void SetUp() override {
    mccl::CollectiveIntegrationTestMixin::SetUp(
        mccl::CollectiveIntegrationTestMixin::Config{
            .numRanks = kNumRanks,
            // A rank that dies during shutdown must reach the driver's log scan
            // rather than aborting the run inside the mixin.
            .shouldExitOnFailure = false,
            .env =
                {
                    "NCCL_HPC_JOB_IDS=",
                    "NCCL_SOCKET_IFNAME=eth0",
                    "NCCL_CLIENT_SOCKET_IFNAME=eth0",
                    "NCCL_SOCKET_IPADDR_PREFIX=",
                    "NCCL_FASTINIT_MODE=none",
                    "NCCL_DEBUG=INFO",
                    fmt::format(
                        "NCCL_DEBUG_FILE={}",
                        (tmpDir_.path() / "logfile%p").string()),
                },
        });
  }

  static constexpr int kNumRanks{2};

  // The driver hands this path to the ranks via NCCL_DEBUG_FILE, so the ranks'
  // own copies of this member are unused.
  folly::test::TemporaryDirectory tmpDir_{
      fmt::format("LoggingShutdown{}", folly::Random::rand64())};

  // TemporaryDirectory hands back a boost path; the scanner takes a std one.
  std::filesystem::path logDir() const {
    return std::filesystem::path{tmpDir_.path().string()};
  }

  void checkRanksShutDownCleanly(std::string_view phaseSentinel) {
    ASSERT_TRUE(
        std::holds_alternative<
            mccl::CollectiveIntegrationTestMixin::TestDriverState>(state_));
    const auto& driverState =
        std::get<mccl::CollectiveIntegrationTestMixin::TestDriverState>(state_);

    std::vector<meta::comms::testing::LogCrashMarker> markers;
    ASSERT_NO_THROW(markers = findCrashMarkers(logDir()))
        << "failed to scan rank logs. " << describeLogDir(logDir());
    EXPECT_THAT(markers, ::testing::IsEmpty())
        << "crash markers in rank logs:" << describeCrashMarkers(markers);

    /*
     * Stop before sentinel checks when a rank failed in its test body. Those
     * sentinels intentionally identify only a successfully completed body.
     */
    ASSERT_THAT(driverState.workerExitCodes, ::testing::Each(::testing::Eq(0)))
        << "a rank did not exit cleanly. " << describeLogDir(logDir());

    std::vector<std::string> truncatedLogs;
    ASSERT_NO_THROW(truncatedLogs = findTruncatedLogs(logDir()))
        << "failed to inspect rank log termination. "
        << describeLogDir(logDir());
    EXPECT_THAT(truncatedLogs, ::testing::IsEmpty())
        << "rank log does not end in a newline, so the writer died mid-record "
           "or the final flush never completed. "
        << describeLogDir(logDir());

    // Every sentinel must appear in every rank's log: that is the positive
    // signal that logging still worked that late in the rank's life.
    for (const auto sentinel : {phaseSentinel, kAtExit, kStaticDestruction}) {
      int matchingLogCount{};
      ASSERT_NO_THROW(
          matchingLogCount = countLogsContaining(logDir(), sentinel))
          << "failed to scan for sentinel \"" << sentinel << "\". "
          << describeLogDir(logDir());
      EXPECT_EQ(kNumRanks, matchingLogCount)
          << "sentinel \"" << sentinel << "\" missing from a rank log. "
          << describeLogDir(logDir());
    }
  }
};

/*
 * One test only: the mixin spawns its ranks during the first test's SetUp, so a
 * second test in this binary would never launch ranks of its own. BUCK creates
 * one target per teardown mode so each mode receives fresh rank processes.
 */
TEST_F(LoggingShutdownTest, LogsSurviveSelectedTeardownPath) {
  const char* modeValue = std::getenv("NCCLX_LOGGING_SHUTDOWN_MODE");
  ASSERT_NE(nullptr, modeValue);
  const std::string_view mode{modeValue};
  ASSERT_THAT(
      mode,
      ::testing::AnyOf(
          ::testing::Eq(kDestroyMode),
          ::testing::Eq(kAbortMode),
          ::testing::Eq(kLeakMode)));

  const auto phaseSentinel = mode == kDestroyMode ? kAfterDestroy
      : mode == kAbortMode                        ? kAfterAbort
                                                  : kBeforeLeak;
  if (isTestDriverProcess()) {
    checkRanksShutDownCleanly(phaseSentinel);
    return;
  }

  const auto rank = getRank();
  const auto worldSize = getWorldSize();
  const auto store = getTCPStore();
  bool reachedBarrier = false;

  /*
   * A fatal assertion returns from this lambda, not the test, so a failed rank
   * still reaches the peer barrier before the test process exits.
   */
  const auto runRankBody = [&] {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(rank));

    const auto id = exchangeUniqueId(store, mode, rank);
    ASSERT_TRUE(id.has_value());
    const char* commDesc = mode == kDestroyMode ? "logging_shutdown_destroy"
        : mode == kAbortMode                    ? "logging_shutdown_abort"
                                                : "logging_shutdown_leak";
    auto* comm = initComm(*id, commDesc, rank, worldSize);
    ASSERT_NE(nullptr, comm);

    if (mode == kDestroyMode) {
      // ncclCommDestroy joins the communicator threads before logger teardown.
      ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
      logSentinel(kAfterDestroy);
    } else if (mode == kAbortMode) {
      ASSERT_EQ(ncclSuccess, ncclCommAbort(comm));
      logSentinel(kAfterAbort);
    } else {
      // Exit with a live communicator, as when a process terminates without a
      // clean shutdown. The barrier keeps both ranks at the same point so
      // neither observes the other's teardown as a peer failure.
      logSentinel(kBeforeLeak);
    }

    storeBarrier(store, "before_exit", rank, worldSize);
    reachedBarrier = true;

    ASSERT_EQ(0, std::atexit(logAtExitSentinel));
    workerBodyCompleted() = true;
  };
  runRankBody();
  if (!reachedBarrier) {
    storeBarrier(store, "before_exit", rank, worldSize);
  }
}
