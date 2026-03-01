// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "TimeoutTestHelpers.hpp"

#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <iostream>
#include <memory>

#include <fmt/format.h>
#include <folly/String.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>

namespace {

// RAII wrapper for a forked child process with captured stderr pipe
struct ChildProcess {
  pid_t pid = -1;
  int readFd = -1; // read end of stderr pipe (parent keeps)
  int writeFd = -1; // write end of stderr pipe (closed in parent after fork)
  bool waited = false;

  ChildProcess() = default;

  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;

  ChildProcess(ChildProcess&& other) noexcept
      : pid(other.pid),
        readFd(other.readFd),
        writeFd(other.writeFd),
        waited(other.waited) {
    other.pid = -1;
    other.readFd = -1;
    other.writeFd = -1;
    other.waited = true;
  }

  ~ChildProcess() {
    cleanup();
  }

  // Create pipe pair. Returns false on failure.
  bool setupPipe() {
    int fds[2];
    if (pipe(fds) == -1) {
      return false;
    }
    readFd = fds[0];
    writeFd = fds[1];
    return true;
  }

  // Called in forked child: redirect stderr to write end, close read end
  void pipeStderr() {
    if (close(readFd) == -1) {
      perror("close(pipe read end)");
      _exit(1);
    }
    readFd = -1;
    if (dup2(writeFd, STDERR_FILENO) == -1) {
      perror("dup2(pipe -> stderr)");
      _exit(1);
    }
    if (close(writeFd) == -1) {
      perror("close(pipe write end)");
      _exit(1);
    }
    writeFd = -1;
  }

  // Called in parent after fork: close write end, record pid
  void onForked(const pid_t p) {
    pid = p;
    if (writeFd != -1) {
      if (close(writeFd) == -1) {
        perror("close(pipe write end in parent)");
      }
      writeFd = -1;
    }
  }

  // Set environment variables for child process (called after fork in child)
  static void
  setEnv(const int rank, const int numChildren, const std::string& storePath) {
    if (unsetenv("OMPI_COMM_WORLD_RANK") == -1 ||
        unsetenv("OMPI_COMM_WORLD_SIZE") == -1 ||
        setenv("TORCHCOMM_RANK", std::to_string(rank).c_str(), 1) == -1 ||
        setenv("TORCHCOMM_SIZE", std::to_string(numChildren).c_str(), 1) ==
            -1 ||
        setenv("TORCHCOMM_STORE_PATH", storePath.c_str(), 1) == -1 ||
        setenv("MASTER_ADDR", "127.0.0.1", 1) == -1 ||
        setenv("MASTER_PORT", "29500", 1) == -1) {
      perror("setenv/unsetenv");
      _exit(1);
    }
  }

  // Wait for the child to exit within timeout. Returns true if child exited.
  bool waitForExit(int* status, const int timeoutSecs) {
    for (int elapsed = 0; elapsed < timeoutSecs * 10; elapsed++) {
      const pid_t result = waitpid(pid, status, WNOHANG);
      if (result == pid) {
        waited = true;
        return true;
      }
      if (result == -1) {
        waited = true;
        return false;
      }
      usleep(100000); // NOLINT(facebook-hte-BadCall-usleep)
    }
    // Timed out — kill the child
    if (kill(pid, SIGKILL) == -1 && errno != ESRCH) {
      perror("kill(SIGKILL)");
    }
    waitpid(pid, status, 0);
    waited = true;
    return false;
  }

  // Read all captured stderr from the pipe and close it
  std::string readStderr() {
    std::string result;
    if (readFd == -1) {
      return result;
    }
    char buf[4096];
    ssize_t n;
    while ((n = read(readFd, buf, sizeof(buf))) > 0) {
      result.append(buf, n);
    }
    close(readFd);
    readFd = -1;
    return result;
  }

 private:
  void cleanup() {
    if (!waited && pid > 0) {
      kill(pid, SIGKILL);
      int status;
      waitpid(pid, &status, 0);
    }
    if (readFd != -1) {
      close(readFd);
      readFd = -1;
    }
    if (writeFd != -1) {
      close(writeFd);
      writeFd = -1;
    }
    pid = -1;
    waited = true;
  }
};

// Verify child exit status matches expectation
void checkExitStatus(
    const int rank,
    const int status,
    const torch::comms::test::RankExpectation& expect) {
  bool matchesExitCode = true;
  bool matchesSignal = true;
  std::vector<std::string> expected;
  if (expect.exitCode.has_value()) {
    const auto cleanExit = WIFEXITED(status);
    const auto exitCode = cleanExit ? WEXITSTATUS(status) : -1;
    const auto expExitCode = expect.exitCode.value_or(-1);
    matchesExitCode = cleanExit && exitCode == expExitCode;
    if (!matchesExitCode) {
      expected.push_back(
          fmt::format(
              "expected exitCode={} but got {}", expExitCode, exitCode));
    }
  }
  if (expect.signal.has_value()) {
    const auto signaled = WIFSIGNALED(status);
    const auto termSignal = signaled ? WTERMSIG(status) : -1;
    const auto expSignal = expect.signal.value_or(-1);
    matchesSignal = signaled && termSignal == expSignal;
    if (!matchesSignal) {
      expected.push_back(
          fmt::format("expected signal={} but got {}", expSignal, termSignal));
    }
  }

  if (expect.exitCode.has_value() && expect.signal.has_value()) {
    EXPECT_TRUE(matchesExitCode || matchesSignal)
        << "Rank " << rank
        << " exit status mismatch: " << folly::join(" OR ", expected);
  } else if (expect.exitCode.has_value()) {
    EXPECT_TRUE(matchesExitCode)
        << "Rank " << rank
        << " exit status mismatch: " << folly::join(" OR ", expected);
  } else if (expect.signal.has_value()) {
    EXPECT_TRUE(matchesSignal)
        << "Rank " << rank
        << " exit status mismatch: " << folly::join(" OR ", expected);
  }
}

// Verify captured stderr matches log expectations
void checkLogExpectations(
    const int rank,
    const std::string& capturedStderr,
    const torch::comms::test::RankExpectation& expect) {
  for (const auto& pattern : expect.logMustContain) {
    EXPECT_NE(capturedStderr.find(pattern), std::string::npos)
        << "Rank " << rank << " stderr missing expected pattern: '" << pattern
        << "'\nstderr was:\n"
        << capturedStderr;
  }
  for (const auto& pattern : expect.logMustNotContain) {
    EXPECT_EQ(capturedStderr.find(pattern), std::string::npos)
        << "Rank " << rank << " stderr contains unexpected pattern: '"
        << pattern << "'\nstderr was:\n"
        << capturedStderr;
  }
}

} // namespace

namespace torch::comms::test {

std::string TimeoutTestHelper::execModeName(const ExecMode mode) {
  switch (mode) {
    case ExecMode::kEager:
      return "Eager";
    case ExecMode::kMultiGraphSequential:
      return "MultiGraphSeq";
    case ExecMode::kMultiGraphConcurrent:
      return "MultiGraphConc";
  }
  return "Unknown";
}

void TimeoutTestHelper::exec(
    const ExecMode mode,
    const std::vector<std::function<void()>>& ops) {
  switch (mode) {
    case ExecMode::kEager:
      for (const auto& op : ops) {
        op();
      }
      at::cuda::getCurrentCUDAStream().synchronize();
      break;
    case ExecMode::kMultiGraphSequential: {
      const auto stream = at::cuda::getStreamFromPool();
      at::cuda::CUDAStreamGuard guard(stream);
      const int numOps = static_cast<int>(ops.size());
      // Capture each op as a separate graph
      std::vector<std::unique_ptr<at::cuda::CUDAGraph>> graphs;
      graphs.reserve(numOps);
      for (const auto& op : ops) {
        auto& graph =
            graphs.emplace_back(std::make_unique<at::cuda::CUDAGraph>());
        graph->capture_begin();
        op();
        graph->capture_end();
      }
      // Replay all graphs sequentially on the same stream
      for (int i = 0; i < numOps; i++) {
        graphs[i]->replay();
      }
      stream.synchronize();
      break;
    }
    case ExecMode::kMultiGraphConcurrent: {
      const int numOps = static_cast<int>(ops.size());
      // Capture each op on its own stream
      std::vector<at::cuda::CUDAStream> streams;
      streams.reserve(numOps);
      std::vector<std::unique_ptr<at::cuda::CUDAGraph>> graphs;
      graphs.reserve(numOps);
      for (const auto& op : ops) {
        streams.push_back(at::cuda::getStreamFromPool());
        auto& graph =
            graphs.emplace_back(std::make_unique<at::cuda::CUDAGraph>());
        at::cuda::CUDAStreamGuard streamGuard(streams.back());
        graph->capture_begin();
        op();
        graph->capture_end();
      }
      // Replay all graphs on their respective streams
      for (int i = 0; i < numOps; i++) {
        at::cuda::CUDAStreamGuard streamGuard(streams[i]);
        graphs[i]->replay();
      }
      for (int i = 0; i < numOps; i++) {
        streams[i].synchronize();
      }
      break;
    }
  }
}

void TimeoutTestHelper::launch(
    const std::string& testName,
    const int numChildren,
    const std::function<void(int rank)>& childBody,
    const std::vector<RankExpectation>& expectations,
    const int childWaitTimeoutSecs) {
  const std::string storePath = "/tmp/torchcomm_timeout_test_" +
      std::to_string(getpid()) + "_" + testName;
  unlink(storePath.c_str());

  std::vector<ChildProcess> children;
  children.reserve(numChildren);

  // Fork child processes with redirected stderr for log capture
  for (int rank = 0; rank < numChildren; rank++) {
    ChildProcess child;
    if (!child.setupPipe()) {
      FAIL() << "pipe() failed for rank " << rank;
    }

    const pid_t pid = fork();
    if (pid == -1) {
      FAIL() << "fork() failed for rank " << rank;
      // child destructor cleans up pipe fds
    } else if (pid == 0) {
      child.pipeStderr();
      ChildProcess::setEnv(rank, numChildren, storePath);
      try {
        childBody(rank);
      } catch (const std::exception& ex) {
        std::cerr << "Child rank " << rank << " exception: " << ex.what()
                  << "\n";
        _exit(1);
      } catch (...) {
        std::cerr << "Child rank " << rank << " unknown exception\n";
        _exit(1);
      }
      _exit(0);
    } else {
      child.onForked(pid);
      children.emplace_back(std::move(child));
    }
  }

  // Wait for children and verify exit expectations
  for (int rank = 0; rank < numChildren; rank++) {
    int status = 0;
    const bool exited =
        children[rank].waitForExit(&status, childWaitTimeoutSecs);

    EXPECT_TRUE(exited) << "Rank " << rank
                        << " child did not exit within timeout";
    if (!exited) {
      continue;
    }

    const auto capturedStderr = children[rank].readStderr();
    const auto& expect = expectations[rank];
    checkExitStatus(rank, status, expect);
    checkLogExpectations(rank, capturedStderr, expect);
  }

  EXPECT_NE(unlink(storePath.c_str()), -1)
      << "unlink(" << storePath << ") failed";
}

} // namespace torch::comms::test
