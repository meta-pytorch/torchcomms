// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <fstream>

#include "comms/ctran/commstate/Topology.h"

TEST(TopologyTest, LoadTopologySuccess) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=rtptest021.nha1.facebook.com" << std::endl;
  file << "DEVICE_BACKEND_NETWORK_TOPOLOGY=/nha1.1D//rtsw098.c084.f00.nha1"
       << std::endl;
  file << "DEVICE_RACK_SERIAL=50022944" << std::endl;

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_TRUE(topo);
  EXPECT_EQ(topo->rank, 0);
  const std::string host(topo->host);
  const std::string rtsw(topo->rtsw);
  EXPECT_EQ(host, "rtptest021.nha1.facebook.com");
  EXPECT_EQ(rtsw, "rtsw098.c084.f00.nha1");
}

TEST(TopologyTest, LoadTopologyFailure) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_FALSE(topo);
}

TEST(TopologyTest, EmptyHostName) {
  const std::string filepath = "/tmp/mocked_fbwhoami.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=";

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_FALSE(topo);
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
  EXPECT_EQ(topo->rank, 1);

  const std::string host(topo->host);
  const std::string dc(topo->dc);
  const std::string zone(topo->zone);
  const std::string su(topo->su);
  const std::string rtsw(topo->rtsw);

  EXPECT_EQ(host, "testhost.rail.facebook.com");
  EXPECT_EQ(dc, "");
  EXPECT_EQ(zone, "snb1.z081");
  EXPECT_EQ(su, "snb1.z081.u015");
  EXPECT_EQ(rtsw, "");
  EXPECT_STREQ(topo->rackSerial, "12345");
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
  const std::string su(topo->su);
  const std::string rtsw(topo->rtsw);
  EXPECT_EQ(su, "scaling_unit");
  EXPECT_EQ(rtsw, "rtsw_name");
}

TEST(TopologyTest, GB300DsfTopologyZoneOnly) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=twshared1766.40.uco1.facebook.com" << std::endl;
  file << "DEVICE_BACKEND_NETWORK_TOPOLOGY=uco1/uco1.z086//" << std::endl;
  file << "DEVICE_RACK_SERIAL=12278553" << std::endl;

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_TRUE(topo);
  const std::string host(topo->host);
  const std::string dc(topo->dc);
  const std::string zone(topo->zone);
  const std::string su(topo->su);
  const std::string rtsw(topo->rtsw);
  EXPECT_EQ(host, "twshared1766.40.uco1.facebook.com");
  EXPECT_EQ(dc, "uco1");
  EXPECT_EQ(zone, "uco1.z086");
  EXPECT_EQ(su, "");
  EXPECT_EQ(rtsw, "");
  EXPECT_STREQ(topo->rackSerial, "12278553");
}

TEST(TopologyTest, EmptyTopologyValue) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=testhost.facebook.com" << std::endl;
  file << "DEVICE_BACKEND_NETWORK_TOPOLOGY=" << std::endl;

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_TRUE(topo); // Should succeed, empty topology is allowed when no
                     // backend network

  const std::string host(topo->host);
  EXPECT_EQ(host, "testhost.facebook.com");
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
  EXPECT_EQ(topo->rank, 0);
  const std::string host(topo->host);
  const std::string rtsw(topo->rtsw);
  EXPECT_EQ(host, "rtptest021.nha1.facebook.com");
  EXPECT_EQ(rtsw, "rtsw098.c084.f00.nha1");
  EXPECT_STREQ(topo->rackSerial, "");
}

TEST(TopologyTest, NonNumericRackSerial) {
  const std::string filepath = "/tmp/ut-topology.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=testhost.facebook.com" << std::endl;
  file << "DEVICE_BACKEND_NETWORK_TOPOLOGY=/zone//rtsw001" << std::endl;
  file << "DEVICE_RACK_SERIAL=not_a_number" << std::endl;

  auto topo = ctran::commstate::loadTopology(0, filepath);
  EXPECT_TRUE(topo);
  EXPECT_STREQ(topo->rackSerial, "not_a_number");
}
