// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/testing/TestUtil.h>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "graph.h"
#include "nccl.h"

class topoTest : public ::testing::Test {
 public:
  topoTest() = default;

 protected:
  void SetUp() override {
    setenv("NCCL_DEBUG", "INFO", 0);
    setenv("NCCL_DEBUG_LOGGING_ASYNC", "0", 1);
    CUDACHECK_TEST(cudaSetDevice(0));
    NCCLCHECK_TEST(ncclCommInitAll(&mockComm, 1, nullptr));
  }
  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(mockComm));
  }

  ncclComm_t mockComm{};
};

TEST_F(topoTest, defaultTopoXmlNotFound) {
  folly::test::TemporaryFile dumpXmlFile;

  testing::internal::CaptureStdout();
  auto res = ncclTopoGetSystem(mockComm, nullptr, dumpXmlFile.path().c_str());
  const auto logContent = testing::internal::GetCapturedStdout();

  EXPECT_EQ(res, ncclSuccess);
  // Verify no WARN emitted by ncclTopoGetSystem
  EXPECT_THAT(logContent, ::testing::Not(::testing::HasSubstr("NCCL WARN")));

  // print the dump xml file
  std::ifstream dumpXml(dumpXmlFile.path().c_str());
  std::string dumpXmlStr(
      (std::istreambuf_iterator<char>(dumpXml)),
      std::istreambuf_iterator<char>());
  LOG(INFO) << "dumpXmlStr: " << dumpXmlStr;
}

TEST_F(topoTest, userTopoXmlFileNotFound) {
  // Starting NCCLX 2.29, we started fully relying on the Nvidia PARAM
  // infrastructure for the Nvidia-provided control variables.
#if NCCL_VERSION_CODE >= 22900
  SysEnvRAII topoFileEnv("NCCL_TOPO_FILE", "/tmp/nccl_topo_not_exist.xml");
#else
  EnvRAII<std::string> topoFileEnv(
      NCCL_TOPO_FILE, "/tmp/nccl_topo_not_exist.xml");
#endif
  folly::test::TemporaryFile dumpXmlFile;

  testing::internal::CaptureStdout();
  auto res = ncclTopoGetSystem(mockComm, nullptr, dumpXmlFile.path().c_str());
  const auto logContent = testing::internal::GetCapturedStdout();

  EXPECT_EQ(res, ncclSuccess);
  // Verify log about missing topo file
  EXPECT_THAT(
      logContent, ::testing::HasSubstr("Could not open XML topology file"));
}
