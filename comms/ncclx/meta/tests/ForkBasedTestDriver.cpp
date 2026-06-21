// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ncclx/meta/tests/ForkBasedTestDriver.h"

#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <fcntl.h>
#include <cstdlib>
#include <cstring>

#include <c10/util/Exception.h>

#include <fmt/format.h>
#include <folly/FileUtil.h>
#include <folly/String.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace ncclx::test {

namespace {

constexpr std::string_view kWorkerProcessEnvVar{
    "NCCLX_FORK_TEST_WORKER_PROCESS"};
constexpr std::string_view kRankEnvVar{"RANK"};
constexpr std::string_view kLocalRankEnvVar{"LOCAL_RANK"};
constexpr std::string_view kWorldSizeEnvVar{"WORLD_SIZE"};
constexpr std::string_view kStorePortEnvVar{"FORK_TEST_STORE_PORT"};
constexpr std::string_view kStoreHostEnvVar{"FORK_TEST_STORE_HOST"};

// LLVM coverage env vars to propagate to workers
const std::vector<std::string_view> kCoverageEnvVars{
    "LLVM_PROFILE_FILE",
    "LLVM_COV",
    "LLVM_COVERAGE_ADDITIONAL_OBJECT_PATHS",
};

int32_t getIntFromEnvVar(std::string_view envVar) {
  auto* valueStr = std::getenv(envVar.data());
  XLOG_IF(FATAL, valueStr == nullptr) << envVar << " env var not set";
  auto valueTry = folly::tryTo<int32_t>(valueStr);
  XLOG_IF(FATAL, !valueTry.hasValue())
      << "Invalid " << envVar << " env var value: " << valueStr;
  return valueTry.value();
}

struct Cmdline {
  std::string binaryPath;
  std::vector<std::string> args;
};

Cmdline getCurrentProcessCmdline() {
  // Get binary path from /proc/self/exe (symlink to actual binary)
  char binaryPath[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", binaryPath, sizeof(binaryPath) - 1);
  XLOG_IF(FATAL, len < 0) << "Failed to read /proc/self/exe";
  binaryPath[len] = '\0';

  // Get command line args from /proc/self/cmdline
  std::string cmdline;
  bool ret = folly::readFile("/proc/self/cmdline", cmdline);
  XLOG_IF(FATAL, !ret) << "Failed to read /proc/self/cmdline";

  std::vector<std::string_view> parts;
  // Note: arguments are stored as null-byte-delimited sequence.
  folly::split(std::string("\x00", 1), cmdline, parts);

  Cmdline result;
  result.binaryPath = binaryPath;
  for (size_t i = 1; i < parts.size(); ++i) {
    if (!parts[i].empty()) {
      result.args.emplace_back(parts[i]);
    }
  }
  return result;
}

// Singleton state for test driver to ensure workers spawn only once per
// process. Uses Meyer's Singleton pattern.
struct TestDriverSingletonState {
  bool workersSpawned = false;
  // Raw pointer to avoid static destruction order issues.
  // Intentionally leaked - lives for process lifetime.
  ForkBasedTestDriver::TestDriverState* state = nullptr;
  std::shared_ptr<c10d::TCPStore> store;
};

TestDriverSingletonState& getTestDriverSingleton() {
  static TestDriverSingletonState instance;
  return instance;
}

// Singleton state for worker processes
struct WorkerSingletonState {
  std::shared_ptr<c10d::TCPStore> store;
  bool initialized = false;
};

WorkerSingletonState& getWorkerSingleton() {
  static WorkerSingletonState instance;
  return instance;
}

ForkBasedTestDriver::TestDriverState testDriverMain(
    const ForkBasedTestDriver::Config& config,
    std::shared_ptr<c10d::TCPStore>& storeOut) {
  XLOG(INFO) << "ForkBasedTestDriver: I am the test driver";

  // Create TCPStore server with OS-assigned port
  auto store = std::make_shared<c10d::TCPStore>(
      "127.0.0.1",
      c10d::TCPStoreOptions{
          .port = 0,
          .isServer = true,
          .numWorkers = config.numRanks + 1, // workers + driver
          .waitWorkers = false, // Workers haven't been forked yet
      });
  storeOut = store;

  auto storePort = store->getPort();
  XLOG(INFO) << "ForkBasedTestDriver: TCPStore running on port " << storePort;

  // Get current process command line and prepare for re-exec
  auto cmdline = getCurrentProcessCmdline();

  // Override --gtest_filter so workers only run the current test case
  auto* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
  XLOG_IF(FATAL, testInfo == nullptr) << "No current test info available";

  std::vector<std::string> filteredArgs;
  filteredArgs.reserve(cmdline.args.size());
  for (auto& arg : cmdline.args) {
    if (arg.find("--gtest_filter=") != 0) {
      filteredArgs.push_back(std::move(arg));
    }
  }
  filteredArgs.push_back(
      fmt::format(
          "--gtest_filter={}.{}",
          testInfo->test_suite_name(),
          testInfo->name()));
  cmdline.args = std::move(filteredArgs);
  XLOG(INFO) << "ForkBasedTestDriver: Worker gtest filter: "
             << cmdline.args.back();

  // Fork worker processes
  std::vector<pid_t> workerPids;
  workerPids.reserve(config.numRanks);

  for (int rank = 0; rank < config.numRanks; ++rank) {
    pid_t pid = fork();
    PCHECK(pid >= 0) << "fork() failed";

    if (pid == 0) {
      // Child process: set up env and exec

      // Auto-terminate if parent dies
      prctl(PR_SET_PDEATHSIG, SIGTERM);

      // Set required env vars
      setenv(kWorkerProcessEnvVar.data(), "1", 1);
      setenv(kRankEnvVar.data(), std::to_string(rank).c_str(), 1);
      setenv(kLocalRankEnvVar.data(), std::to_string(rank).c_str(), 1);
      setenv(
          kWorldSizeEnvVar.data(), std::to_string(config.numRanks).c_str(), 1);
      setenv(kStorePortEnvVar.data(), std::to_string(storePort).c_str(), 1);
      setenv(kStoreHostEnvVar.data(), "127.0.0.1", 1);

      // Unset vars that cause duplicate test result files
      unsetenv("TEST_RESULTS_OUTPUT_FILE");
      unsetenv("GTEST_OUTPUT");

      // Set user-provided env vars
      for (const auto& envVar : config.env) {
        auto pos = envVar.find('=');
        if (pos != std::string::npos) {
          auto key = envVar.substr(0, pos);
          auto value = envVar.substr(pos + 1);
          setenv(key.c_str(), value.c_str(), 1);
        }
      }

      // Propagate LLVM coverage env vars
      for (const auto& coverageEnvVar : kCoverageEnvVars) {
        if (auto* value = std::getenv(coverageEnvVar.data())) {
          setenv(coverageEnvVar.data(), value, 1);
        }
      }

      int devnull = ::open("/dev/null", O_WRONLY);
      if (devnull >= 0) {
        ::dup2(devnull, STDOUT_FILENO);
        ::close(devnull);
      }

      if (auto* existing = std::getenv("ASAN_OPTIONS")) {
        std::string opts(existing);
        if (opts.find("protect_shadow_gap=") == std::string::npos) {
          if (!opts.empty()) {
            opts += ":";
          }
          opts += "protect_shadow_gap=0";
          setenv("ASAN_OPTIONS", opts.c_str(), 1);
        }
      } else {
        setenv("ASAN_OPTIONS", "protect_shadow_gap=0", 1);
      }

      // Build argv for execv
      std::vector<char*> argv;
      argv.push_back(const_cast<char*>(cmdline.binaryPath.c_str()));
      for (auto& arg : cmdline.args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
      }
      argv.push_back(nullptr);

      execv(cmdline.binaryPath.c_str(), argv.data());
      // If execv returns, it failed
      XLOG(FATAL) << "execv failed: " << strerror(errno);
      _exit(1);
    }

    // Parent: record child PID
    workerPids.push_back(pid);
    XLOG(INFO) << "ForkBasedTestDriver: Launched worker rank " << rank
               << " with pid " << pid;
  }

  // Collect exit codes from all workers
  std::vector<int> workerExitCodes(config.numRanks, -1);

  for (int rank = 0; rank < config.numRanks; ++rank) {
    int status = 0;
    pid_t result = waitpid(workerPids[rank], &status, 0);

    if (result < 0) {
      XLOG(ERR) << "ForkBasedTestDriver: waitpid failed for rank " << rank
                << ": " << strerror(errno);
      workerExitCodes[rank] = -1;
      continue;
    }

    if (WIFEXITED(status)) {
      workerExitCodes[rank] = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      workerExitCodes[rank] = 128 + WTERMSIG(status);
    } else {
      workerExitCodes[rank] = -1;
    }

    if (workerExitCodes[rank] != 0) {
      if (config.shouldExitOnFailure) {
        XLOG(FATAL) << "ForkBasedTestDriver: Rank " << rank
                    << " exited with code: " << workerExitCodes[rank];
      } else {
        XLOG(INFO) << "ForkBasedTestDriver: Rank " << rank
                   << " exited with code: " << workerExitCodes[rank];
      }
    }
  }

  return ForkBasedTestDriver::TestDriverState{
      .workerExitCodes = std::move(workerExitCodes),
  };
}

ForkBasedTestDriver::WorkerState workerMain() {
  auto& singleton = getWorkerSingleton();
  if (!singleton.initialized) {
    auto port = getIntFromEnvVar(kStorePortEnvVar);
    auto* host = std::getenv(kStoreHostEnvVar.data());
    XLOG_IF(FATAL, host == nullptr) << "FORK_TEST_STORE_HOST not set";

    singleton.store = std::make_shared<c10d::TCPStore>(
        host,
        c10d::TCPStoreOptions{
            .port = static_cast<uint16_t>(port),
            .isServer = false,
        });
    singleton.initialized = true;
  }

  XLOG(INFO) << "ForkBasedTestDriver: I am worker rank "
             << ForkBasedTestDriver::getRank();

  return ForkBasedTestDriver::WorkerState{
      .store = singleton.store,
  };
}

} // namespace

void ForkBasedTestDriver::SetUp(const Config& config) {
  if (!isTestDriverProcess()) {
    state_ = workerMain();
    return;
  }

  // For the test driver: only spawn workers once, not per-test.
  auto& singleton = getTestDriverSingleton();
  if (!singleton.workersSpawned) {
    singleton.workersSpawned = true;
    singleton.state =
        new TestDriverState(testDriverMain(config, singleton.store));
  }
  state_ = *singleton.state;
}

void ForkBasedTestDriver::TearDown() {
  // No-op: workers are reaped in SetUp, store is cleaned up via shared_ptr
}

const ForkBasedTestDriver::TestDriverState&
ForkBasedTestDriver::getTestDriverState() {
  return std::get<TestDriverState>(state_);
}

/* static */
bool ForkBasedTestDriver::isTestDriverProcess() {
  if (auto* valueStr = std::getenv(kWorkerProcessEnvVar.data())) {
    return std::string(valueStr) != "1";
  }
  return true;
}

/* static */
int ForkBasedTestDriver::getRank() {
  return getIntFromEnvVar(kRankEnvVar);
}

/* static */
int ForkBasedTestDriver::getWorldSize() {
  return getIntFromEnvVar(kWorldSizeEnvVar);
}

/* static */
std::shared_ptr<c10d::TCPStore> ForkBasedTestDriver::getStore() {
  if (isTestDriverProcess()) {
    return getTestDriverSingleton().store;
  }
  return getWorkerSingleton().store;
}

/* static */
void ForkBasedTestDriver::setKey(
    const std::string& key,
    const std::string& value) {
  auto store = getStore();
  store->set(key, std::vector<uint8_t>(value.begin(), value.end()));
}

/* static */
std::string ForkBasedTestDriver::waitForKey(const std::string& key) {
  auto store = getStore();
  auto value = store->get(key);
  return std::string(value.begin(), value.end());
}

/* static */
std::optional<std::string> ForkBasedTestDriver::waitForKey(
    const std::string& key,
    std::chrono::milliseconds timeout) {
  auto store = getStore();
  try {
    store->wait({key}, timeout);
  } catch (const c10::DistStoreError&) {
    return std::nullopt;
  }
  // Keys are write-once in test usage, so get() after wait() is safe.
  auto value = store->get(key);
  return std::string(value.begin(), value.end());
}

/* static */
int ForkBasedTestDriver::getCudaDeviceId(int rank) {
  int numDevices{0};
  cudaGetDeviceCount(&numDevices);
  auto err = cudaGetDeviceCount(&numDevices);
  XLOG_IF(FATAL, err != cudaSuccess)
      << "cudaGetDeviceCount failed: " << cudaGetErrorString(err);

  XLOG_IF(FATAL, numDevices <= 0) << "No CUDA devices available";
  int deviceId = rank % numDevices;
  XLOG(INFO) << "Using CUDA device id: " << deviceId << " for rank: " << rank;
  cudaSetDevice(deviceId);
  err = cudaSetDevice(deviceId);
  XLOG_IF(FATAL, err != cudaSuccess)
      << "cudaSetDevice failed: " << cudaGetErrorString(err);

  return deviceId;
}

} // namespace ncclx::test
