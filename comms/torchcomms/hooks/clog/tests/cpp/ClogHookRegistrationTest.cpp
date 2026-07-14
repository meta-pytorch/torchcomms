// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Unit tests for ClogHook::registerWithComm's init-agnostic new_comm signature.
//
// Uses the CPU-only TorchCommFake backend (loaded dynamically via
// FAKE_TEST_BACKEND_LIB_PATH) so no GPU/MCCL is required. A comm created via
// new_comm is initialized; finalize() flips it back to uninitialized to
// exercise the dynamic-regime path (PAFT's inter-replica comm registers clog
// before its first reconfigure()).

#include <comms/torchcomms/TorchComm.hpp>
#include <comms/torchcomms/fake/TorchCommFake.hpp>
#include <comms/torchcomms/hooks/clog/ClogHook.hpp>

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace torch::comms {

namespace {
constexpr const char* kBackendName = "fake_test";
constexpr const char* kBackendEnvKey = "TORCHCOMMS_BACKEND_LIB_PATH_FAKE_TEST";

std::vector<std::string> readLines(const std::string& path) {
  std::vector<std::string> lines;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

size_t countContaining(
    const std::vector<std::string>& lines,
    const std::string& sub) {
  size_t n = 0;
  for (const auto& line : lines) {
    if (line.find(sub) != std::string::npos) {
      ++n;
    }
  }
  return n;
}
} // namespace

class ClogHookRegistrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* lib_path = std::getenv("FAKE_TEST_BACKEND_LIB_PATH");
    ASSERT_NE(lib_path, nullptr) << "FAKE_TEST_BACKEND_LIB_PATH not set";
    setenv(kBackendEnvKey, lib_path, 1);

    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    log_path_ = (std::filesystem::temp_directory_path() /
                 (std::string(info->test_suite_name()) + "_" +
                  std::string(info->name()) + ".clog"))
                    .string();
    std::filesystem::remove(log_path_);
  }

  void TearDown() override {
    unsetenv(kBackendEnvKey);
    std::filesystem::remove(log_path_);
  }

  std::string log_path_;
};

// An initialized comm (the WORLD-init path) writes the new_comm signature
// eagerly at registration, before any collective runs.
TEST_F(
    ClogHookRegistrationTest,
    SignatureWrittenAtRegistrationWhenInitialized) {
  auto comm = new_comm(kBackendName, at::Device(at::kCPU), "eager_comm", {});
  ASSERT_NE(comm, nullptr);

  auto hook =
      std::make_shared<ClogHook>(log_path_, std::vector<std::string>{"ALL"});
  hook->registerWithComm(comm);

  EXPECT_EQ(
      countContaining(readLines(log_path_), "new_comm|comm=eager_comm"), 1u);
}

// A comm registered before initialization (dynamic regime) must NOT throw and
// must defer the new_comm signature to the first pre-hook, which fires once a
// collective is issued.
TEST_F(
    ClogHookRegistrationTest,
    SignatureDeferredUntilFirstCollectiveWhenUninitialized) {
  auto comm = new_comm(kBackendName, at::Device(at::kCPU), "deferred_comm", {});
  ASSERT_NE(comm, nullptr);

  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm->getBackendImpl());
  ASSERT_NE(backend, nullptr);
  backend->finalize(); // isInitialized() -> false
  ASSERT_FALSE(backend->isInitialized());

  auto hook =
      std::make_shared<ClogHook>(log_path_, std::vector<std::string>{"ALL"});
  hook->registerWithComm(comm);

  // Deferred: no signature yet at registration time.
  EXPECT_EQ(
      countContaining(readLines(log_path_), "new_comm|comm=deferred_comm"), 0u);

  // First collective fires the pre-hook, which flushes the deferred signature.
  auto tensor = at::ones({2, 2}, at::kFloat);
  comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/false);

  EXPECT_EQ(
      countContaining(readLines(log_path_), "new_comm|comm=deferred_comm"), 1u);
}

// The deferred signature is emitted exactly once, even across many collectives.
TEST_F(ClogHookRegistrationTest, DeferredSignatureWrittenExactlyOnce) {
  auto comm = new_comm(kBackendName, at::Device(at::kCPU), "once_comm", {});
  ASSERT_NE(comm, nullptr);

  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm->getBackendImpl());
  ASSERT_NE(backend, nullptr);
  backend->finalize();

  auto hook =
      std::make_shared<ClogHook>(log_path_, std::vector<std::string>{"ALL"});
  hook->registerWithComm(comm);

  auto tensor = at::ones({2, 2}, at::kFloat);
  comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/false);
  comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/false);
  comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/false);

  EXPECT_EQ(
      countContaining(readLines(log_path_), "new_comm|comm=once_comm"), 1u);
}

} // namespace torch::comms
