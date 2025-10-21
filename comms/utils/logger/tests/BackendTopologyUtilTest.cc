// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/testing/TestUtil.h>
#include <gtest/gtest.h>

#include "comms/utils/logger/BackendTopologyUtil.h"

TEST(BackendTopologyUtilTest, GetBackendTopology) {
  // See https://www.internalfb.com/phabricator/paste/view/P1561091182
  // Taken from twshared43957.01.pci2 /etc/fbwhoami
  const char* contents = R"(DEVICE_UPDATED_AT=2024-08-27T15:54:24-07:00
DEVICE_UPDATED_AT_UNIX=1724799264
DEVICE_ANR_LAST_RUN_UNIX=1724799264
DEVICE_ID=325003645
DEVICE_ASSET_TAG=ES79879
DEVICE_NAME=twshared43957.01.pci2.facebook.com
DEVICE_PARENT_NAME=
DEVICE_PARENT_ASSET_TAG=EP40684
DEVICE_PARENT_ASSET_ID=325003806
DEVICE_PARENT_FBPN=01-100428
DEVICE_PARENT_MODEL_ID=346566
DEVICE_PARENT_MODEL_MAKE=Quanta
DEVICE_PARENT_MODEL_NAME=GRANDTETON_CHASSIS_80GB_HBM3_ROCE
DEVICE_RAM=2048
SHARD_SEED=71698335
DEVICE_HOSTNAME_SCHEME=twshared
DEVICE_HOSTNAME_SCHEME_TYPE=
DEVICE_SERIAL_NUMBER=QTWF0TS24250104
DEVICE_RACK_SERIAL=10849739
DEVICE_SWITCH_SERIAL=FJ25424200387
DEVICE_COUNTRY=US
DEVICE_REGION=iowa
DEVICE_CITY=Altoona
DEVICE_DATACENTER=pci2
DEVICE_CLUSTER=01
DEVICE_CLUSTER_RACK=
DEVICE_CLUSTER_RACK_STATE=RACK_IN_USE
DEVICE_NETWORK_ID=pci2.0001.0006.0031
DEVICE_TC_NETWORK_ID=/193/
DEVICE_AI_ZONE=pci2.z085
DEVICE_BACKEND_NETWORK_TOPOLOGY=pci2/pci2.2A.z085//rtsw049.c085.f00.pci2
DEVICE_NETWORK_NAME=
DEVICE_POD=006
DEVICE_SUITE=2A
DEVICE_ROW=12B
DEVICE_RACK=15
DEVICE_RACK_SIDE=A
DEVICE_RACK_POS=08
DEVICE_RACK_SUB_POS=02
DEVICE_RACK_SUB_POS_SLOT=1
DEVICE_PHYSICAL_LOC=US:iowa:pci2:2A:12B:15:A:08:02
DEVICE_LOGICAL_LOC=US:iowa:pci2.0001.0006.0031
DEVICE_SUBNET=
DEVICE_SERVICE_SUBNET=2803:6082:8014:7801:0000:0000:0000:0000/64
DEVICE_SVC1_SUBNET=
DEVICE_SVC0_1_SUBNET=2803:6082:8014:7826:0000:0000:0000:0000/64
DEVICE_SVC0_2_SUBNET=2803:6082:8014:784b:0000:0000:0000:0000/64
DEVICE_SVC0_3_SUBNET=2803:6082:8014:78ba:0000:0000:0000:0000/64
DEVICE_PRIMARY_IP=
DEVICE_SUBNET_IPV6=2401:db00:142c:051e:0000:0000:0000:0000/64
DEVICE_PRIMARY_IPV6=2401:db00:142c:051e:face:0000:0001:0000
DEVICE_CLUSTER_SUBNET=
DEVICE_CLUSTER_SUBNET_IPV6=
DEVICE_STATE=IN_USE
DEVICE_STATUS=INSTALLED
DEVICE_MAINTENANCE_STATUS=NONE
DEVICE_EVALUATION_STATUS=EVAL_MASS_PRODUCTION
DEVICE_FBPN=01-100427
HARDWARE_PROGRAM_ID=6338
DEVICE_TYPE=SERVER
DEVICE_DEPARTMENT=ops
SERVER_TYPE=TYPE_XX_GPU_TC
RACK_TYPE=RACK_GPU_TC
CPU_ARCHITECTURE=sapphirerapids
MODEL_ID=346565
MODEL_MAKE=Quanta
MODEL_NAME=GRANDTETON_SPR_T20_80GB_HBM3_ROCE
CLUSTER_PROTOCOL_TOPOLOGY=L3BGPV6DYNAMIC
CLUSTER_TYPE=SERVICE_GENERIC_NON_MEMCACHE
CLUSTER_ASN=0
POD_ASN=65406
POD_GEOVIP_SECURITY_MODE_1=2401:db00:14ff:b088:5000:0000:0000:0000/68
POD_FABRIC_RSW_ALIAS=
FBNET_CHASSIS_MODEL=
FBNET_LINECARD_MODEL=
RSW_UPLINK_CAPACITY=8X200G,1X400G
PLANNED_POWER_WATTS=9250.0
REGION_DATACENTER_PREFIX=pci
REGION_ROUTABLE_CLUSTER=pci1.03
SHARED_FATE_ZONE=sfz_ia1
CLUSTER_STATE=CLUSTER_IN_USE
CLUSTER_MAINTENANCE_STATE=CLUSTER_MAINT_NONE
FBNET_DEVICE_CAPABILITIES=
FBNET_CLUSTER_CAPABILITIES=
RACK_DESCRIPTION=TLA,T20,US,277V,GRANDTETON_CHASSIS_80GB_HBM3_ROCE,SAMSUNG-E1.S-4TB-PM9D3,MELLANOX-400G-NIC,WEDGE400C-48VDC-F,DELTA-V3-POWER,PANASONIC/ARETSYN-V3-BBU-SHELF,OPEN-RACK-V3,QUANTA
EXPECTED_DATA_HDD_COUNT=0
EXPECTED_DATA_SSD_COUNT=8
DEVICE_OOB_ASSET_TAG=ES79879
DEVICE_OOB_IP=
DEVICE_OOB_IPV6=2401:db00:142c:051e:face:0000:007d:0000
OPENRACK_TYPE=OpenRack
OPENRACK_BATTERY_BACKED=NO
POWER_FAILURE_DOMAIN=FB__PCI2__SRFD_03
DEVICE_NICS_ETH0_IPV6=2401:db00:142c:051e:face:0000:0001:0000
DEVICE_NICS_ETH0_MAC=9C:63:C0:73:86:22
DEVICE_NICS_ETH1_IPV6=2401:db00:142c:051e:face:0000:0026:0000
DEVICE_NICS_ETH1_MAC=9C:63:C0:73:86:23
DEVICE_NICS_ETH2_IPV6=2401:db00:142c:051e:face:0000:004b:0000
DEVICE_NICS_ETH2_MAC=9C:63:C0:75:B2:C2
DEVICE_NICS_ETH3_IPV6=2401:db00:142c:051e:face:0000:00ba:0000
DEVICE_NICS_ETH3_MAC=9C:63:C0:75:B2:C3
DEVICE_NICS_BETH0_IPV6=2401:db00:142a:0604:bace:0000:0129:0000
DEVICE_NICS_BETH0_MAC=9C:63:C0:2E:24:AA
DEVICE_NICS_BETH1_IPV6=2401:db00:142a:0607:bace:0000:0173:0000
DEVICE_NICS_BETH1_MAC=9C:63:C0:43:BD:74
DEVICE_NICS_BETH2_IPV6=2401:db00:142a:0600:bace:0000:01bd:0000
DEVICE_NICS_BETH2_MAC=9C:63:C0:43:B9:64
DEVICE_NICS_BETH3_IPV6=2401:db00:142a:0606:bace:0000:0207:0000
DEVICE_NICS_BETH3_MAC=9C:63:C0:43:BE:BC
DEVICE_NICS_BETH4_IPV6=2401:db00:142a:0605:bace:0000:029b:0000
DEVICE_NICS_BETH4_MAC=9C:63:C0:2D:E0:12
DEVICE_NICS_BETH5_IPV6=2401:db00:142a:0603:bace:0000:030a:0000
DEVICE_NICS_BETH5_MAC=9C:63:C0:43:A0:B6
DEVICE_NICS_BETH6_IPV6=2401:db00:142a:0602:bace:0000:039e:0000
DEVICE_NICS_BETH6_MAC=9C:63:C0:43:C4:68
DEVICE_NICS_BETH7_IPV6=2401:db00:142a:0601:bace:0000:0033:0000
DEVICE_NICS_BETH7_MAC=9C:63:C0:2D:FA:02
DEVICE_NICS_SVC0_IPV6=2803:6082:8014:7801:0000:0000:0000:0001
DEVICE_NICS_SVC0_1_IPV6=2803:6082:8014:7826:0000:0000:0000:0001
DEVICE_NICS_SVC0_2_IPV6=2803:6082:8014:784b:0000:0000:0000:0001
DEVICE_NICS_SVC0_3_IPV6=2803:6082:8014:78ba:0000:0000:0000:0001
DEVICE_NICS_ENUM=ETH0,ETH1,ETH2,ETH3,BETH0,BETH1,BETH2,BETH3,BETH4,BETH5,BETH6,BETH7,SVC0,SVC0_1,SVC0_2,SVC0_3
SHARED_NIC_FACTOR=1
DEVICE_LOGICAL_SERVER_TYPE=T20
DEVICE_LOGICAL_SERVER_SUBTYPE=T20_GRAND_TETON_HBM3_ROCE)";
  folly::test::TemporaryFile tmpFile;
  std::ofstream file(tmpFile.path().string());
  file << contents;
  file.close();

  auto topology =
      BackendTopologyUtil::getBackendTopology(tmpFile.path().string());
  ASSERT_TRUE(topology.has_value());
  ASSERT_EQ(topology->sfz, "sfz_ia1");
  ASSERT_EQ(topology->region, "pci");
  ASSERT_EQ(topology->dc, "pci2");
  ASSERT_EQ(topology->zone, "pci2.2A.z085");
  ASSERT_EQ(topology->rtsw, "rtsw049.c085.f00.pci2");
  ASSERT_EQ(topology->host, "twshared43957.01.pci2.facebook.com");
  std::vector<std::string> expectedFullScopes{
      "sfz_ia1",
      "pci",
      "pci2",
      "pci2.2A.z085",
      "rtsw049.c085.f00.pci2",
      "twshared43957.01.pci2.facebook.com"};
  ASSERT_EQ(topology->fullScopes, expectedFullScopes);
}

TEST(BackendTopologyUtilTest, EmptyFile) {
  const char* contents = "";
  folly::test::TemporaryFile tmpFile;
  std::ofstream file(tmpFile.path().string());
  file << contents;
  file.close();

  auto topology =
      BackendTopologyUtil::getBackendTopology(tmpFile.path().string());
  ASSERT_FALSE(topology.has_value());
}

TEST(BackendTopologyUtilTest, SFZMissing) {
  const char* contents = R"(REGION_DATACENTER_PREFIX=pci
DEVICE_DATACENTER=pci2
DEVICE_NAME=twshared43957.01.pci2.facebook.com
DEVICE_BACKEND_NETWORK_TOPOLOGY=pci2/pci2.2A.z085//rtsw049.c085.f00.pci2)";
  folly::test::TemporaryFile tmpFile;
  std::ofstream file(tmpFile.path().string());
  file << contents;
  file.close();

  auto topology =
      BackendTopologyUtil::getBackendTopology(tmpFile.path().string());
  ASSERT_FALSE(topology.has_value());
}

TEST(BackendTopologyUtilTest, RegionMissing) {
  const char* contents = R"(SHARED_FATE_ZONE=sfz_ia1
DEVICE_DATACENTER=pci2
DEVICE_NAME=twshared43957.01.pci2.facebook.com
DEVICE_BACKEND_NETWORK_TOPOLOGY=pci2/pci2.2A.z085//rtsw049.c085.f00.pci2)";
  folly::test::TemporaryFile tmpFile;
  std::ofstream file(tmpFile.path().string());
  file << contents;
  file.close();

  auto topology =
      BackendTopologyUtil::getBackendTopology(tmpFile.path().string());
  ASSERT_FALSE(topology.has_value());
}

TEST(BackendTopologyUtilTest, DCMissing) {
  const char* contents = R"(SHARED_FATE_ZONE=sfz_ia1
REGION_DATACENTER_PREFIX=pci
DEVICE_NAME=twshared43957.01.pci2.facebook.com
DEVICE_BACKEND_NETWORK_TOPOLOGY=pci2/pci2.2A.z085//rtsw049.c085.f00.pci2)";
  folly::test::TemporaryFile tmpFile;
  std::ofstream file(tmpFile.path().string());
  file << contents;
  file.close();

  auto topology =
      BackendTopologyUtil::getBackendTopology(tmpFile.path().string());
  ASSERT_FALSE(topology.has_value());
}

TEST(BackendTopologyUtilTest, HostMissing) {
  const char* contents = R"(SHARED_FATE_ZONE=sfz_ia1
REGION_DATACENTER_PREFIX=pci
DEVICE_DATACENTER=pci2
DEVICE_BACKEND_NETWORK_TOPOLOGY=pci2/invalid)";
  folly::test::TemporaryFile tmpFile;
  std::ofstream file(tmpFile.path().string());
  file << contents;
  file.close();

  auto topology =
      BackendTopologyUtil::getBackendTopology(tmpFile.path().string());
  ASSERT_FALSE(topology.has_value());
}

TEST(BackendTopologyUtilTest, BackendNetworkTopologyMissing) {
  const char* contents = R"(SHARED_FATE_ZONE=sfz_ia1
REGION_DATACENTER_PREFIX=pci
DEVICE_DATACENTER=pci2,
DEVICE_NAME=twshared43957.01.pci2.facebook.com)";
  folly::test::TemporaryFile tmpFile;
  std::ofstream file(tmpFile.path().string());
  file << contents;
  file.close();

  auto topology =
      BackendTopologyUtil::getBackendTopology(tmpFile.path().string());
  ASSERT_FALSE(topology.has_value());
}

TEST(BackendTopologyUtilTest, BackendNetworkTopologyInvalidValue) {
  const char* contents = R"(SHARED_FATE_ZONE=sfz_ia1
REGION_DATACENTER_PREFIX=pci
DEVICE_DATACENTER=pci2
DEVICE_NAME=twshared43957.01.pci2.facebook.com
DEVICE_BACKEND_NETWORK_TOPOLOGY=pci2/invalid)";
  folly::test::TemporaryFile tmpFile;
  std::ofstream file(tmpFile.path().string());
  file << contents;
  file.close();

  auto topology =
      BackendTopologyUtil::getBackendTopology(tmpFile.path().string());
  ASSERT_FALSE(topology.has_value());
}
