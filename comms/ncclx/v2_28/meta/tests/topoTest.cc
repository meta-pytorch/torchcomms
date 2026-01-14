// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/testing/TestUtil.h>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/testinfra/TestUtils.h"
#include "graph.h"
#include "nccl.h"

class topoTest : public ::testing::Test {
 public:
  topoTest() = default;

 protected:
  void SetUp() override {
    setenv("NCCL_DEBUG", "WARN", 0);
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
  auto res = ncclTopoGetSystem(mockComm, nullptr, dumpXmlFile.path().c_str());
  EXPECT_EQ(res, ncclSuccess);
  auto lastErr = std::string(ncclGetLastError(mockComm));
  EXPECT_TRUE(lastErr.empty());
  // print the dump xml file
  std::ifstream dumpXml(dumpXmlFile.path().c_str());
  std::string dumpXmlStr(
      (std::istreambuf_iterator<char>(dumpXml)),
      std::istreambuf_iterator<char>());
  LOG(INFO) << "dumpXmlStr: " << dumpXmlStr;
}

TEST_F(topoTest, userTopoXmlFileNotFound) {
  EnvRAII<std::string> topoFileEnv(
      NCCL_TOPO_FILE, "/tmp/nccl_topo_not_exist.xml");
  folly::test::TemporaryFile dumpXmlFile;
  // return ncclSuccess, but with warning/error message and record last error
  auto res = ncclTopoGetSystem(mockComm, nullptr, dumpXmlFile.path().c_str());
  EXPECT_EQ(res, ncclSuccess);
  auto lastErr = std::string(ncclGetLastError(mockComm));
  LOG(INFO) << "ncclGetLastError: " << lastErr;
  EXPECT_GT(lastErr.size(), 0);
}
