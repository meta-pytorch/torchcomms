// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/registry/GpuBackendSelector.h"

#include <array>
#include <barrier>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include <gtest/gtest.h>

namespace uniflow {
namespace {

void expectBackend(
    const Result<std::string>& result,
    std::string_view expected) {
  ASSERT_TRUE(result.hasValue()) << result.error().toString();
  EXPECT_EQ(result.value(), expected);
}

class ScopedGpuBackendEnvironment {
 public:
  explicit ScopedGpuBackendEnvironment(const char* value) {
    if (const auto* original = std::getenv("UNIFLOW_GPU_BACKEND")) {
      original_ = original;
    }
    set(value);
  }

  ScopedGpuBackendEnvironment(const ScopedGpuBackendEnvironment&) = delete;
  ScopedGpuBackendEnvironment& operator=(const ScopedGpuBackendEnvironment&) =
      delete;

  ~ScopedGpuBackendEnvironment() {
    set(original_ ? original_->c_str() : nullptr);
  }

  void set(const char* value) {
    if (value) {
      setenv("UNIFLOW_GPU_BACKEND", value, 1);
    } else {
      unsetenv("UNIFLOW_GPU_BACKEND");
    }
  }

 private:
  std::optional<std::string> original_;
};

} // namespace

TEST(GpuBackendSelectorTest, Parse_ValidatesExplicitNamesWithoutSelecting) {
  for (const auto* backend : {"nvidia", "amd", "mtia"}) {
    expectBackend(parseGpuBackend(backend), backend);
    EXPECT_NO_THROW(validateGpuBackendRequest(std::string(backend)));
  }
  EXPECT_NO_THROW(validateGpuBackendRequest(std::nullopt));
  for (const auto* backend : {"", "cuda", "NVIDIA", "unknown"}) {
    auto result = parseGpuBackend(backend);
    ASSERT_TRUE(result.hasError());
    EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
    EXPECT_THROW(
        validateGpuBackendRequest(std::string(backend)), std::invalid_argument);
  }
}

TEST(GpuBackendSelectorTest, Resolve_UsesConfigurationWithoutClaiming) {
  GpuBackendSelector defaults("nvidia", std::nullopt);
  expectBackend(defaults.resolve(), "nvidia");
  EXPECT_FALSE(defaults.selectedBackend());

  GpuBackendSelector environment("nvidia", "amd");
  expectBackend(environment.resolve(), "amd");
  expectBackend(environment.resolve(std::string("mtia")), "mtia");
  EXPECT_FALSE(environment.selectedBackend());
}

TEST(GpuBackendSelectorTest, Resolve_ExplicitBypassesInvalidConfiguration) {
  GpuBackendSelector selector("invalid-default", "invalid-environment");

  expectBackend(selector.resolve(std::string("amd")), "amd");
  EXPECT_FALSE(selector.selectedBackend());
  expectBackend(selector.claim("amd", std::string("amd")), "amd");
}

TEST(GpuBackendSelectorTest, Resolve_ReadsEnvironmentUntilClaim) {
  ScopedGpuBackendEnvironment environment("amd");
  GpuBackendSelector selector("nvidia");

  expectBackend(selector.resolve(), "amd");
  environment.set("mtia");
  expectBackend(selector.resolve(), "mtia");
  environment.set(nullptr);
  expectBackend(selector.resolve(), "nvidia");
  EXPECT_FALSE(selector.selectedBackend());
}

TEST(GpuBackendSelectorTest, Resolve_InvalidConfigurationDoesNotClaim) {
  for (const auto* environment : {"", "unknown"}) {
    GpuBackendSelector selector("nvidia", environment);
    auto result = selector.resolve();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
    EXPECT_FALSE(selector.selectedBackend());
  }
  GpuBackendSelector selector("unknown", std::nullopt);
  EXPECT_FALSE(selector.resolve());
  EXPECT_FALSE(selector.selectedBackend());
  expectBackend(selector.resolve(std::string("mtia")), "mtia");
}

TEST(GpuBackendSelectorTest, Claim_PreviewIgnoresLaterEnvironmentChanges) {
  ScopedGpuBackendEnvironment environment("amd");
  GpuBackendSelector selector("nvidia");
  auto preview = selector.resolve();
  ASSERT_TRUE(preview);
  environment.set("invalid-environment");

  expectBackend(selector.claim(preview.value()), "amd");
  expectBackend(selector.resolve(), "amd");
  environment.set("nvidia");
  expectBackend(selector.resolve(), "amd");
  EXPECT_EQ(selector.selectedBackend(), "amd");
}

TEST(GpuBackendSelectorTest, Claim_SameVendorSucceedsAndExplicitConflictFails) {
  GpuBackendSelector selector("nvidia", std::nullopt);
  expectBackend(selector.claim("nvidia", std::string("nvidia")), "nvidia");
  expectBackend(selector.claim("nvidia", std::string("nvidia")), "nvidia");

  auto resolved = selector.resolve(std::string("amd"));
  auto claimed = selector.claim("amd", std::string("amd"));
  ASSERT_FALSE(resolved);
  ASSERT_FALSE(claimed);
  EXPECT_EQ(resolved.error().code(), ErrCode::InvalidArgument);
  EXPECT_EQ(claimed.error().code(), ErrCode::InvalidArgument);
  EXPECT_NE(claimed.error().message().find("nvidia"), std::string::npos);
  EXPECT_EQ(selector.selectedBackend(), "nvidia");
}

TEST(GpuBackendSelectorTest, Claim_RejectsMismatchedExplicitPreview) {
  GpuBackendSelector selector("nvidia", std::nullopt);
  auto result = selector.claim("amd", std::string("nvidia"));

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
  EXPECT_FALSE(selector.selectedBackend());
  expectBackend(selector.claim("nvidia", std::string("nvidia")), "nvidia");
}

TEST(GpuBackendSelectorTest, Claim_ValidatesNamesEvenAfterClaim) {
  GpuBackendSelector selector("nvidia", std::nullopt);
  for (const bool alreadyClaimed : {false, true}) {
    if (alreadyClaimed) {
      expectBackend(selector.claim("amd"), "amd");
    }
    auto candidate = selector.claim("unknown");
    auto request = selector.claim("amd", std::string("unknown"));
    ASSERT_FALSE(candidate);
    ASSERT_FALSE(request);
    EXPECT_EQ(candidate.error().code(), ErrCode::InvalidArgument);
    EXPECT_EQ(request.error().code(), ErrCode::InvalidArgument);
    EXPECT_EQ(selector.selectedBackend().has_value(), alreadyClaimed);
  }
}

TEST(GpuBackendSelectorTest, Claim_ImplicitReusesACompetingClaim) {
  GpuBackendSelector selector("nvidia", std::nullopt);
  auto preview = selector.resolve();
  ASSERT_TRUE(preview);
  expectBackend(selector.claim("amd", std::string("amd")), "amd");

  expectBackend(selector.claim(preview.value()), "amd");
  EXPECT_FALSE(selector.claim(preview.value(), std::string("nvidia")));
  expectBackend(selector.claim(preview.value(), std::string("amd")), "amd");
}

TEST(GpuBackendSelectorTest, Claim_ConcurrentImplicitPreviewsShareOneClaim) {
  GpuBackendSelector selector("nvidia", std::nullopt);
  std::barrier start(3);
  std::array<std::optional<Result<std::string>>, 2> results;
  std::array<std::jthread, 2> workers;
  const std::array<std::string, 2> previews{"nvidia", "amd"};
  for (size_t i = 0; i < workers.size(); ++i) {
    workers[i] = std::jthread([&, i] {
      start.arrive_and_wait();
      results[i] = selector.claim(previews[i]);
    });
  }
  start.arrive_and_wait();
  for (auto& worker : workers) {
    worker.join();
  }

  const auto selected = selector.selectedBackend();
  ASSERT_TRUE(selected);
  for (const auto& result : results) {
    ASSERT_TRUE(result);
    expectBackend(*result, *selected);
  }
}

TEST(GpuBackendSelectorTest, Claim_ConcurrentExplicitPreviewsHaveOneWinner) {
  GpuBackendSelector selector("nvidia", std::nullopt);
  std::barrier start(3);
  std::array<std::optional<Result<std::string>>, 2> results;
  std::array<std::jthread, 2> workers;
  const std::array<std::string, 2> requested{"nvidia", "amd"};
  for (size_t i = 0; i < workers.size(); ++i) {
    workers[i] = std::jthread([&, i] {
      start.arrive_and_wait();
      results[i] = selector.claim(requested[i], requested[i]);
    });
  }
  start.arrive_and_wait();
  for (auto& worker : workers) {
    worker.join();
  }

  ASSERT_TRUE(results[0]);
  ASSERT_TRUE(results[1]);
  const size_t winner = results[0]->hasValue() ? 0 : 1;
  const size_t loser = 1 - winner;
  expectBackend(*results[winner], requested[winner]);
  ASSERT_TRUE(results[loser]->hasError());
  EXPECT_EQ(results[loser]->error().code(), ErrCode::InvalidArgument);
  EXPECT_EQ(selector.selectedBackend(), requested[winner]);
}

} // namespace uniflow
