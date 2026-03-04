// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <folly/coro/BlockingWait.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/testing/TestUtil.h>
#include <thrift/lib/cpp2/util/ScopedServerInterfaceThread.h>

#include "aiplatform/tw_platform/core/SessionsCacheCommsTracingServiceThrift.h"
#include "comms/analyzer/Analyzer.h"
#include "comms/analyzer/CommDumpPuller.h"
#include "comms/analyzer/analysis/CudaErrorAnalyzer.h"
#include "comms/analyzer/analysis/IbErrorAnalyzer.h"
#include "comms/analyzer/tests/CommsTracingServiceTestHandler.h"

DECLARE_string(ip_to_hostname_map_file);

using namespace meta::comms::analyzer;
using namespace aiplatform::tw_platform::core;
using apache::thrift::ScopedServerInterfaceThread;

namespace {

// Test hostnames assigned to each rank.
std::string hostnameForRank(int64_t rank) {
  return fmt::format("rank{}.test.facebook.com", rank);
}

// Test peer IP assigned to each rank (used as the peer in IB errors reported
// by OTHER ranks).
std::string peerIpForRank(int64_t rank) {
  return fmt::format("fd00::{}", rank + 1);
}

// Peer string in the format expected by extractIpFromPeer(): "ip<port>".
std::string peerStringForRank(int64_t rank) {
  return fmt::format("{}<12345>", peerIpForRank(rank));
}

///////////////////////////////////////////////////////////////////////////////
// Helper builders
///////////////////////////////////////////////////////////////////////////////

::comms::IbCompletionError makeIbError(
    int64_t timestampMs,
    const std::string& peer,
    const std::string& statusStr = "FLUSH_ERR",
    const std::string& hcaName = "mlx5_0",
    int64_t vendorErr = 0x81) {
  ::comms::IbCompletionError error;
  error.timestampMs() = timestampMs;
  error.peer() = peer;
  error.statusStr() = statusStr;
  error.hcaName() = hcaName;
  error.vendorErr() = vendorErr;
  error.status() = 1;
  error.opcodeStr() = "RDMA_WRITE";
  error.opcode() = 0;
  error.reqSize() = 4096;
  error.reqType() = "RC";
  error.localGid() = "fe80::1";
  error.remoteGid() = "fe80::2";
  return error;
}

::comms::CudaError makeCudaError(
    int64_t timestampMs,
    const std::string& errorString = "CUDA_ERROR_ECC_UNCORRECTABLE",
    int32_t errorCode = 214,
    const std::string& scaleupDomain = "",
    const std::string& localHostname = "") {
  ::comms::CudaError error;
  error.timestampMs() = timestampMs;
  error.errorString() = errorString;
  error.errorCode() = errorCode;
  error.scaleupDomain() = scaleupDomain;
  error.localHostname() = localHostname;
  return error;
}

// Build a GetCommsResponse for a single rank with one communicator and
// optional IB / CUDA errors.
::comms::GetCommsResponse makeResponse(
    int64_t rank,
    int64_t nRanks,
    std::vector<::comms::IbCompletionError> ibErrors = {},
    std::vector<::comms::CudaError> cudaErrors = {}) {
  ::comms::GetCommsResponse response;
  response.globalRank() = rank;

  // Minimal communicator entry so the analyzer sees this rank as connected.
  ::comms::NCCLParsedEntry entry;
  entry.commHash() = "0x1";
  entry.rank() = rank;
  entry.nRanks() = nRanks;

  ::comms::CommsForRank commsForRank;
  commsForRank.ncclParsedEntryMap()["0x1"] = std::move(entry);
  response.commsForRank() = std::move(commsForRank);

  auto nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
  response.currentTimeNs() = nowNs;
  response.jobStartTimeNs() = nowNs - 100'000'000'000; // 100 s ago
  response.step() = 1;
  response.stepStartTimeNs() = nowNs - 1'000'000'000; // 1 s ago

  if (!ibErrors.empty()) {
    response.ibErrors() = std::move(ibErrors);
  }
  if (!cudaErrors.empty()) {
    response.cudaErrors() = std::move(cudaErrors);
  }

  return response;
}

// Populate a TrainerSession from a GetCommsResponse, replicating the data
// flow in CommsTracingServiceThriftFetcher::fetch().
void populateSession(
    TrainerSession& session,
    const ::comms::GetCommsResponse& resp) {
  session.rank = *resp.globalRank();
  session.connectionStatus.setSuccess();

  session.lastStatusResponse.status.setSuccess();
  session.lastRankResponse.status.setSuccess();

  session.lastDump.status.setSuccess();
  session.lastDump.response.nccl_parsed_state() =
      *resp.commsForRank()->ncclParsedEntryMap();

  if (resp.ibErrors().has_value()) {
    session.lastDump.ibErrors = *resp.ibErrors();
  }
  if (resp.cudaErrors().has_value()) {
    session.lastDump.cudaErrors = *resp.cudaErrors();
  }
}

///////////////////////////////////////////////////////////////////////////////
// Fixture
///////////////////////////////////////////////////////////////////////////////

class AnalyzerThriftIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    savedEnableIb_ = FLAGS_nccl_analyzer_enable_ib_error_analyzer;
    savedEnableCuda_ = FLAGS_nccl_analyzer_enable_cuda_error_analyzer;
    savedIpToHostnameMapFile_ = FLAGS_ip_to_hostname_map_file;
    executor_ = std::make_shared<folly::CPUThreadPoolExecutor>(4);
  }

  void TearDown() override {
    FLAGS_nccl_analyzer_enable_ib_error_analyzer = savedEnableIb_;
    FLAGS_nccl_analyzer_enable_cuda_error_analyzer = savedEnableCuda_;
    FLAGS_ip_to_hostname_map_file = savedIpToHostnameMapFile_;
    analysis::IbErrorAnalyzer::resetCache();
    servers_.clear();
  }

  // Start a ScopedServerInterfaceThread for each configured response,
  // call getComms via a direct (non-ServiceRouter) Thrift client, and
  // return the responses received over the wire.
  std::vector<::comms::GetCommsResponse> startAndFetch(
      const std::vector<::comms::GetCommsResponse>& configuredResponses) {
    std::vector<::comms::GetCommsResponse> received;
    for (const auto& resp : configuredResponses) {
      auto handler =
          std::make_shared<test::CommsTracingServiceTestHandler>(resp);
      auto server = std::make_unique<ScopedServerInterfaceThread>(handler);
      auto client = server->newClient<
          apache::thrift::Client<::comms::CommsTracingService>>();
      received.push_back(
          folly::coro::blockingWait(
              client->co_getComms(::comms::GetCommsRequest{})));
      servers_.push_back(std::move(server));
    }
    return received;
  }

  // Write a JSON map file mapping peer IPs to hostnames for the given
  // number of ranks, and point the ip_to_hostname_map_file flag at it.
  void writeIpToHostnameMapFile(int nRanks) {
    auto path = tmpDir_.path().string() + "/ip_map.json";
    std::ofstream f(path);
    f << "{";
    for (int i = 0; i < nRanks; ++i) {
      if (i > 0) {
        f << ",";
      }
      f << fmt::format(R"("{}": "{}")", peerIpForRank(i), hostnameForRank(i));
    }
    f << "}";
    f.close();
    FLAGS_ip_to_hostname_map_file = path;
  }

  // Populate a SessionsCache from the Thrift responses and run the
  // analyzer. Returns verdict types, bad ranks, and the verdict itself.
  struct AnalyzerResult {
    PullOneJobResult pullResult;
    AnalyzerVerdict verdict;
  };

  AnalyzerResult analyzeResponses(
      const std::vector<::comms::GetCommsResponse>& responses) {
    auto cpuExec = folly::getKeepAliveToken(executor_.get());
    SessionsCacheCommsTracingServiceThrift cache(
        "test_job", "torchrun", std::move(cpuExec));

    for (const auto& resp : responses) {
      auto rank = *resp.globalRank();
      cache.sessions.emplace_back(hostnameForRank(rank), 0);
      populateSession(cache.sessions.back(), resp);
    }
    cache.isValid = true;
    cache.jobStartTimeSec = 1;

    AnalyzerHistory history;
    auto verdict = folly::coro::blockingWait(
        Analyzer::analyze(&cache, history, std::nullopt));
    verdict.print();

    PullOneJobResult result;
    for (const auto& [ts, _] : verdict.brokenRanks) {
      result.badRanks.insert(ts->rank);
    }
    for (const auto& vt : verdict.verdictType) {
      result.analyzerVerdictType.insert(vt);
    }
    return {std::move(result), std::move(verdict)};
  }

 private:
  std::vector<std::unique_ptr<ScopedServerInterfaceThread>> servers_;
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
  folly::test::TemporaryDirectory tmpDir_{"analyzer_thrift_test"};
  bool savedEnableIb_{};
  bool savedEnableCuda_{};
  std::string savedIpToHostnameMapFile_;
};

///////////////////////////////////////////////////////////////////////////////
// Test cases
///////////////////////////////////////////////////////////////////////////////

TEST_F(AnalyzerThriftIntegrationTest, IbErrorProducesCorrectVerdict) {
  FLAGS_nccl_analyzer_enable_ib_error_analyzer = true;
  constexpr int kNumRanks = 2;

  // Rank 0 reports an IB error with rank 1 as peer.
  writeIpToHostnameMapFile(kNumRanks);
  auto responses = startAndFetch({
      makeResponse(0, kNumRanks, {makeIbError(1000, peerStringForRank(1))}),
      makeResponse(1, kNumRanks),
  });

  auto [result, verdict] = analyzeResponses(responses);

  EXPECT_TRUE(result.analyzerVerdictType.count(
      AnalyzerVerdict::VerdictType::IB_COMPLETION_ERROR));
  EXPECT_TRUE(result.badRanks.count(0));

  // Validate hostnames in the IB error result.
  ASSERT_TRUE(verdict.ibErrorResult.has_value());
  EXPECT_EQ(
      verdict.ibErrorResult->earliestErrorReportingHostname,
      hostnameForRank(0));
  EXPECT_EQ(
      verdict.ibErrorResult->earliestErrorPeerHostname, hostnameForRank(1));
}

TEST_F(AnalyzerThriftIntegrationTest, CudaErrorProducesCorrectVerdict) {
  FLAGS_nccl_analyzer_enable_cuda_error_analyzer = true;
  constexpr int kNumRanks = 2;

  auto responses = startAndFetch({
      makeResponse(0, kNumRanks, {}, {makeCudaError(1000)}),
      makeResponse(1, kNumRanks),
  });

  auto [result, verdict] = analyzeResponses(responses);

  EXPECT_TRUE(result.analyzerVerdictType.count(
      AnalyzerVerdict::VerdictType::CUDA_ERROR));
  EXPECT_FALSE(result.analyzerVerdictType.count(
      AnalyzerVerdict::VerdictType::CUDA_NVLINK_UNCORRECTABLE_ERROR));
  EXPECT_TRUE(result.badRanks.count(0));
}

TEST_F(
    AnalyzerThriftIntegrationTest,
    NvlinkUncorrectableErrorProducesCorrectVerdict) {
  FLAGS_nccl_analyzer_enable_cuda_error_analyzer = true;
  constexpr int kNumRanks = 2;

  auto responses = startAndFetch({
      makeResponse(
          0,
          kNumRanks,
          {},
          {makeCudaError(
              1000,
              "CUDA_ERROR_NVLINK_UNCORRECTABLE",
              /*errorCode=*/72,
              "scaleup_domain_0",
              hostnameForRank(0))}),
      makeResponse(1, kNumRanks),
  });

  auto [result, verdict] = analyzeResponses(responses);

  EXPECT_TRUE(result.analyzerVerdictType.count(
      AnalyzerVerdict::VerdictType::CUDA_NVLINK_UNCORRECTABLE_ERROR));
  EXPECT_FALSE(result.analyzerVerdictType.count(
      AnalyzerVerdict::VerdictType::CUDA_ERROR));
  EXPECT_TRUE(result.badRanks.count(0));

  ASSERT_TRUE(verdict.cudaErrorResult.has_value());
  EXPECT_TRUE(verdict.cudaErrorResult->hasNvlinkUncorrectableError);
  EXPECT_EQ(verdict.cudaErrorResult->earliestNvlinkErrorRank, 0);
  EXPECT_EQ(
      verdict.cudaErrorResult->earliestErrorScaleupDomain, "scaleup_domain_0");
  EXPECT_EQ(
      verdict.cudaErrorResult->earliestErrorLocalHostname, hostnameForRank(0));
}

TEST_F(
    AnalyzerThriftIntegrationTest,
    NvlinkErrorMultiRankTracksEarliestScaleupDomain) {
  FLAGS_nccl_analyzer_enable_cuda_error_analyzer = true;
  constexpr int kNumRanks = 2;

  auto responses = startAndFetch({
      // Rank 0: NVLink error at t=2000 (later), scaleup_domain_a
      makeResponse(
          0,
          kNumRanks,
          {},
          {makeCudaError(
              2000,
              "CUDA_ERROR_NVLINK_UNCORRECTABLE",
              72,
              "scaleup_domain_a",
              hostnameForRank(0))}),
      // Rank 1: NVLink error at t=1000 (earlier), scaleup_domain_b
      makeResponse(
          1,
          kNumRanks,
          {},
          {makeCudaError(
              1000,
              "CUDA_ERROR_NVLINK_UNCORRECTABLE",
              72,
              "scaleup_domain_b",
              hostnameForRank(1))}),
  });

  auto [result, verdict] = analyzeResponses(responses);

  EXPECT_TRUE(result.analyzerVerdictType.count(
      AnalyzerVerdict::VerdictType::CUDA_NVLINK_UNCORRECTABLE_ERROR));

  ASSERT_TRUE(verdict.cudaErrorResult.has_value());
  // Earliest NVLink error is rank 1
  EXPECT_EQ(verdict.cudaErrorResult->earliestNvlinkErrorRank, 1);
  // Metadata should come from rank 1 (the earliest NVLink error)
  EXPECT_EQ(
      verdict.cudaErrorResult->earliestErrorScaleupDomain, "scaleup_domain_b");
  EXPECT_EQ(
      verdict.cudaErrorResult->earliestErrorLocalHostname, hostnameForRank(1));
}

TEST_F(AnalyzerThriftIntegrationTest, NoErrorsProducesNoErrorVerdict) {
  FLAGS_nccl_analyzer_enable_ib_error_analyzer = true;
  FLAGS_nccl_analyzer_enable_cuda_error_analyzer = true;
  constexpr int kNumRanks = 2;

  auto responses = startAndFetch({
      makeResponse(0, kNumRanks),
      makeResponse(1, kNumRanks),
  });

  auto [result, verdict] = analyzeResponses(responses);

  EXPECT_FALSE(result.analyzerVerdictType.count(
      AnalyzerVerdict::VerdictType::IB_COMPLETION_ERROR));
  EXPECT_FALSE(result.analyzerVerdictType.count(
      AnalyzerVerdict::VerdictType::CUDA_ERROR));
  EXPECT_FALSE(result.analyzerVerdictType.count(
      AnalyzerVerdict::VerdictType::CUDA_NVLINK_UNCORRECTABLE_ERROR));
}

TEST_F(AnalyzerThriftIntegrationTest, MultiRankIbErrorsEarliestWins) {
  FLAGS_nccl_analyzer_enable_ib_error_analyzer = true;
  constexpr int kNumRanks = 2;

  // Rank 0 sees peer rank 1, rank 1 sees peer rank 0.
  writeIpToHostnameMapFile(kNumRanks);
  auto responses = startAndFetch({
      // Rank 0: IB error at t=2000 (later), peer is rank 1
      makeResponse(0, kNumRanks, {makeIbError(2000, peerStringForRank(1))}),
      // Rank 1: IB error at t=1000 (earlier), peer is rank 0
      makeResponse(1, kNumRanks, {makeIbError(1000, peerStringForRank(0))}),
  });

  auto [result, verdict] = analyzeResponses(responses);

  EXPECT_TRUE(result.analyzerVerdictType.count(
      AnalyzerVerdict::VerdictType::IB_COMPLETION_ERROR));
  // Only the earliest error rank (1) should be in badRanks.
  EXPECT_TRUE(result.badRanks.count(1));

  // Validate hostnames: reporting rank is 1, peer is rank 0.
  ASSERT_TRUE(verdict.ibErrorResult.has_value());
  EXPECT_EQ(
      verdict.ibErrorResult->earliestErrorReportingHostname,
      hostnameForRank(1));
  EXPECT_EQ(
      verdict.ibErrorResult->earliestErrorPeerHostname, hostnameForRank(0));
}

} // namespace
