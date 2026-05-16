// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/DeviceRackSerial.h"
#include "transport.h"

#include <gtest/gtest.h>
#include <fstream>

using ncclx::isSameRackSerial;
using ncclx::loadRackSerial;

TEST(DeviceRackSerialTest, LoadNumericSerial) {
  const std::string filepath = "/tmp/ut-rack-serial.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=testhost.facebook.com" << std::endl;
  file << "DEVICE_RACK_SERIAL=50022944" << std::endl;
  file.close();

  char buf[kMaxRackSerialLen + 1] = {};
  EXPECT_TRUE(loadRackSerial(filepath, buf, sizeof(buf)));
  EXPECT_STREQ(buf, "50022944");
}

TEST(DeviceRackSerialTest, LoadAlphanumericSerial) {
  const std::string filepath = "/tmp/ut-rack-serial.txt";
  std::ofstream file(filepath);
  file << "DEVICE_RACK_SERIAL=C1507842765072" << std::endl;
  file.close();

  char buf[kMaxRackSerialLen + 1] = {};
  EXPECT_TRUE(loadRackSerial(filepath, buf, sizeof(buf)));
  EXPECT_STREQ(buf, "C1507842765072");
}

TEST(DeviceRackSerialTest, MissingSerial) {
  const std::string filepath = "/tmp/ut-rack-serial.txt";
  std::ofstream file(filepath);
  file << "DEVICE_NAME=testhost.facebook.com" << std::endl;
  file.close();

  char buf[kMaxRackSerialLen + 1] = {};
  EXPECT_FALSE(loadRackSerial(filepath, buf, sizeof(buf)));
  EXPECT_STREQ(buf, "");
}

TEST(DeviceRackSerialTest, EmptySerial) {
  const std::string filepath = "/tmp/ut-rack-serial.txt";
  std::ofstream file(filepath);
  file << "DEVICE_RACK_SERIAL=" << std::endl;
  file.close();

  char buf[kMaxRackSerialLen + 1] = {};
  EXPECT_FALSE(loadRackSerial(filepath, buf, sizeof(buf)));
}

TEST(DeviceRackSerialTest, SerialExceedsMaxLen) {
  const std::string filepath = "/tmp/ut-rack-serial.txt";
  std::ofstream file(filepath);
  std::string longSerial(kMaxRackSerialLen + 1, 'A');
  file << "DEVICE_RACK_SERIAL=" << longSerial << std::endl;
  file.close();

  char buf[kMaxRackSerialLen + 1] = {};
  EXPECT_FALSE(loadRackSerial(filepath, buf, sizeof(buf)));
}

TEST(DeviceRackSerialTest, FileNotFound) {
  char buf[kMaxRackSerialLen + 1] = {};
  EXPECT_FALSE(loadRackSerial("/tmp/nonexistent_file_12345", buf, sizeof(buf)));
}

TEST(DeviceRackSerialTest, IsSameRackSerialBothNonEmpty) {
  EXPECT_TRUE(isSameRackSerial("50022944", "50022944"));
  EXPECT_FALSE(isSameRackSerial("50022944", "50022945"));
  EXPECT_TRUE(isSameRackSerial("C1507842765072", "C1507842765072"));
  EXPECT_FALSE(isSameRackSerial("C1507842765072", "C9999999999999"));
}

TEST(DeviceRackSerialTest, IsSameRackSerialEmptyReturnsFalse) {
  EXPECT_FALSE(isSameRackSerial("", "50022944"));
  EXPECT_FALSE(isSameRackSerial("50022944", ""));
  EXPECT_FALSE(isSameRackSerial("", ""));
}
