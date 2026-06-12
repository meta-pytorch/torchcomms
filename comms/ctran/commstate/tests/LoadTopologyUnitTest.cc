// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <unistd.h>
#include <fstream>

#include "comms/ctran/commstate/Topology.h"

namespace {

void expectHostFromGethostname(
    const std::optional<ctran::commstate::TopologyResult>& topo) {
  ASSERT_TRUE(topo);
  char hostname[256] = {};
  ASSERT_EQ(gethostname(hostname, sizeof(hostname) - 1), 0);
  EXPECT_EQ(std::string(topo->rankTopology.host), std::string(hostname));
  EXPECT_EQ(topo->networkTopo, "");
}

} // namespace

TEST(TopologyTest, LoadTopologySuccess) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=rtptest021.nha1.facebook.com" << std::endl;
  file << "DEVICE_BACKEND_NETWORK_TOPOLOGY=/nha1.1D//rtsw098.c084.f00.nha1"
       << std::endl;
  file << "DEVICE_RACK_SERIAL=50022944" << std::endl;

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_TRUE(topo);
  EXPECT_EQ(topo->rankTopology.rank, 0);
  const std::string host(topo->rankTopology.host);
  EXPECT_EQ(host, "rtptest021.nha1.facebook.com");
  EXPECT_EQ(topo->networkTopo, "/nha1.1D//rtsw098.c084.f00.nha1");
}

TEST(TopologyTest, LoadTopologyFallsBackToHostnameWhenFileIsEmpty) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file.close();

  auto topo = ctran::commstate::loadTopology(0, filepath);
  expectHostFromGethostname(topo);
}

TEST(TopologyTest, LoadTopologyFallsBackToHostnameWhenHostMissing) {
  const std::string filepath = "/tmp/mocked_fbwhoami.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=" << std::endl;
  file.close();

  auto topo = ctran::commstate::loadTopology(0, filepath);
  expectHostFromGethostname(topo);
}

TEST(TopologyTest, LoadTopologyTruncatesLongHostNameSafely) {
  const std::string filepath = "/tmp/ut-topology.txt";
  const std::string longHost(ncclx::kMaxNameLen + 16, 'a');
  std::ofstream file(filepath);
  file << "DEVICE_NAME=" << longHost << std::endl;
  file.close();

  auto topo = ctran::commstate::loadTopology(0, filepath);
  ASSERT_TRUE(topo);
  const std::string host(topo->rankTopology.host);
  EXPECT_EQ(host.size(), ncclx::kMaxNameLen - 1);
  EXPECT_EQ(host, longHost.substr(0, ncclx::kMaxNameLen - 1));
  EXPECT_EQ(topo->networkTopo, "");
}

TEST(TopologyTest, LoadTopologyFallsBackToHostnameWhenFileMissing) {
  const std::string filepath = "/tmp/missing_fbwhoami.txt";
  unlink(filepath.c_str());

  auto topo = ctran::commstate::loadTopology(0, filepath);
  expectHostFromGethostname(topo);
}

TEST(TopologyTest, RailBasedTopology) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=testhost.rail.facebook.com" << std::endl;
  file << "DEVICE_BACKEND_NETWORK_TOPOLOGY=/snb1.z081/snb1.z081.u015/"
       << std::endl;
  file << "DEVICE_RACK_SERIAL=12345" << std::endl;

  auto topo = ctran::commstate::loadTopology(1, filepath);
  EXPECT_TRUE(topo);
  EXPECT_EQ(topo->rankTopology.rank, 1);

  const std::string host(topo->rankTopology.host);
  const std::string dc(topo->rankTopology.dc);
  const std::string zone(topo->rankTopology.zone);

  EXPECT_EQ(host, "testhost.rail.facebook.com");
  EXPECT_EQ(dc, "");
  EXPECT_EQ(zone, "snb1.z081");
  EXPECT_STREQ(topo->rankTopology.rackSerial, "12345");
  EXPECT_EQ(topo->networkTopo, "/snb1.z081/snb1.z081.u015/");
}

TEST(TopologyTest, InvalidTopologyFormat) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=testhost.facebook.com" << std::endl;
  file << "DEVICE_BACKEND_NETWORK_TOPOLOGY=part1/part2/part3" << std::endl;

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_FALSE(
      topo); // Should fail due to incorrect format (needs exactly 4 parts)
}

TEST(TopologyTest, BothRtswAndScalingUnit) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=testhost.facebook.com" << std::endl;
  file << "DEVICE_BACKEND_NETWORK_TOPOLOGY=dc/zone/scaling_unit/rtsw_name"
       << std::endl;

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_TRUE(topo);
  EXPECT_EQ(topo->networkTopo, "dc/zone/scaling_unit/rtsw_name");
}

TEST(TopologyTest, GB300DsfTopologyZoneOnly) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=twshared1766.40.uco1.facebook.com" << std::endl;
  file << "DEVICE_BACKEND_NETWORK_TOPOLOGY=uco1/uco1.z086//" << std::endl;
  file << "DEVICE_RACK_SERIAL=12278553" << std::endl;

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_TRUE(topo);
  const std::string host(topo->rankTopology.host);
  const std::string dc(topo->rankTopology.dc);
  const std::string zone(topo->rankTopology.zone);
  EXPECT_EQ(host, "twshared1766.40.uco1.facebook.com");
  EXPECT_EQ(dc, "uco1");
  EXPECT_EQ(zone, "uco1.z086");
  EXPECT_STREQ(topo->rankTopology.rackSerial, "12278553");
  EXPECT_EQ(topo->networkTopo, "uco1/uco1.z086//");
}

TEST(TopologyTest, EmptyTopologyValue) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=testhost.facebook.com" << std::endl;
  file << "DEVICE_BACKEND_NETWORK_TOPOLOGY=" << std::endl;

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_TRUE(topo); // Should succeed, empty topology is allowed when no
                     // backend network

  const std::string host(topo->rankTopology.host);
  EXPECT_EQ(host, "testhost.facebook.com");
  EXPECT_EQ(topo->networkTopo, "");
}

TEST(TopologyTest, EmptyRackSerial) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=rtptest021.nha1.facebook.com" << std::endl;
  file << "DEVICE_BACKEND_NETWORK_TOPOLOGY=/nha1.1D//rtsw098.c084.f00.nha1"
       << std::endl;
  file << "DEVICE_RACK_SERIAL=" << std::endl;

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_TRUE(topo);
  EXPECT_EQ(topo->rankTopology.rank, 0);
  const std::string host(topo->rankTopology.host);
  EXPECT_EQ(host, "rtptest021.nha1.facebook.com");
  EXPECT_STREQ(topo->rankTopology.rackSerial, "");
}

TEST(TopologyTest, NonNumericRackSerial) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=testhost.facebook.com" << std::endl;
  file << "DEVICE_BACKEND_NETWORK_TOPOLOGY=/zone//rtsw001" << std::endl;
  file << "DEVICE_RACK_SERIAL=not_a_number" << std::endl;

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_TRUE(topo);
  EXPECT_STREQ(topo->rankTopology.rackSerial, "not_a_number");
}
