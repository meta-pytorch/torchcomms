// Copyright (c) Meta Platforms, Inc. and affiliates.

/*

g++ -std=c++20 -fPIC \
  -I . \
  -I "$CONDA_PREFIX/include" \
  comms/utils/logger/tests/ScubaOtelTest.cc \
  comms/utils/logger/ScubaOtel.cc \
  -L "$CONDA_PREFIX/lib" \
  -l:libgtest.a -l:libgtest_main.a \
  -l:libfolly.a -l:libfmt.a -l:libglog.a -l:libgflags.a \
  -l:libdouble-conversion.a \
  -l:libopentelemetry_exporter_otlp_http_log.a \
  -l:libopentelemetry_otlp_recordable.a \
  -l:libopentelemetry_exporter_otlp_http_client.a \
  -l:libopentelemetry_http_client_curl.a \
  -l:libopentelemetry_proto.a \
  -l:libopentelemetry_logs.a \
  -l:libopentelemetry_trace.a \
  -l:libopentelemetry_resources.a \
  -l:libopentelemetry_common.a \
  -l:libprotobuf.a \
  -l:libcurl.a \
  -l:libssh2.a \
  -l:libnghttp2.a \
  -l:libpsl.a \
  -l:libzstd.a \
  -l:libz.a \
  -l:libssl.a -l:libcrypto.a \
  -l:libboost_context.a \
  -l:libevent.a \
  -lpthread -ldl \
  -o /tmp/ScubaOtelTest

/tmp/ScubaOtelTest

*/

#include <gtest/gtest.h>
#include <folly/json.h>
#include <folly/dynamic.h>

#include "comms/utils/logger/ScubaOtel.h"

class ScubaOtelTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    initLoggerProvider();
  }
};

TEST_F(ScubaOtelTest, ConstructorPrefixesDataset) {
  // Should not throw; constructs with "fair_" + dataset.
  EXPECT_NO_THROW(ScubaOtel otel("test_dataset"));
}

TEST_F(ScubaOtelTest, AddSampleReturnsOne) {
  ScubaOtel otel("test_dataset");

  std::unordered_map<std::string, std::string> normalMap = {
      {"key1", "value1"}, {"key2", "value2"}};
  std::unordered_map<std::string, int64_t> intMap = {
      {"int_key1", 42}, {"int_key2", -100}};
  std::unordered_map<std::string, double> doubleMap = {
      {"double_key1", 3.14}, {"double_key2", 0.0}};

  auto result = otel.addSample("test_dataset", normalMap, intMap, doubleMap);
  EXPECT_EQ(result, 1);
}

TEST_F(ScubaOtelTest, AddSampleEmptyMaps) {
  ScubaOtel otel("test_dataset");

  std::unordered_map<std::string, std::string> normalMap;
  std::unordered_map<std::string, int64_t> intMap;
  std::unordered_map<std::string, double> doubleMap;

  auto result = otel.addSample("test_dataset", normalMap, intMap, doubleMap);
  EXPECT_EQ(result, 1);
}

TEST_F(ScubaOtelTest, AddRawDataValidJson) {
  ScubaOtel otel("test_dataset");

  folly::dynamic message = folly::dynamic::object("normal",
      folly::dynamic::object("host", "myhost")("env", "prod"))("int",
      folly::dynamic::object("rank", 0)("world_size", 8))("double",
      folly::dynamic::object("latency", 1.5)("throughput", 100.0))("normvector",
      folly::dynamic::object("tags_list",
          folly::dynamic::array("tag1", "tag2")))("tags",
      folly::dynamic::object("labels",
          folly::dynamic::array("label1", "label2")));

  std::string jsonStr = folly::toJson(message);
  auto result = otel.addRawData("test_dataset", jsonStr, folly::none);
  EXPECT_EQ(result, 1);
}

TEST_F(ScubaOtelTest, AddRawDataMinimalJson) {
  ScubaOtel otel("test_dataset");

  folly::dynamic message = folly::dynamic::object("normal",
      folly::dynamic::object())("int", folly::dynamic::object())("double",
      folly::dynamic::object())("normvector",
      folly::dynamic::object())("tags", folly::dynamic::object());

  std::string jsonStr = folly::toJson(message);
  auto result = otel.addRawData("test_dataset", jsonStr, folly::none);
  EXPECT_EQ(result, 1);
}

TEST_F(ScubaOtelTest, AddRawDataInvalidJson) {
  ScubaOtel otel("test_dataset");

  // Malformed JSON should be handled gracefully (caught exception, return 0).
  auto result = otel.addRawData("test_dataset", "not valid json{{{", folly::none);
  EXPECT_EQ(result, 0);
}

TEST_F(ScubaOtelTest, AddRawDataMissingFields) {
  ScubaOtel otel("test_dataset");

  // JSON missing expected fields should throw internally and be caught (return 0).
  folly::dynamic message = folly::dynamic::object("unexpected_field", 42);
  std::string jsonStr = folly::toJson(message);

  auto result = otel.addRawData("test_dataset", jsonStr, folly::none);
  EXPECT_EQ(result, 0);
}

TEST_F(ScubaOtelTest, AddRawDataWithTimeout) {
  ScubaOtel otel("test_dataset");

  folly::dynamic message = folly::dynamic::object("normal",
      folly::dynamic::object("key", "val"))("int",
      folly::dynamic::object("count", 10))("double",
      folly::dynamic::object("rate", 0.5))("normvector",
      folly::dynamic::object())("tags", folly::dynamic::object());

  std::string jsonStr = folly::toJson(message);
  auto timeout = folly::Optional<std::chrono::milliseconds>(
      std::chrono::milliseconds(5000));
  auto result = otel.addRawData("test_dataset", jsonStr, timeout);
  EXPECT_EQ(result, 1);
}

TEST_F(ScubaOtelTest, AddRawDataEmptyNormvector) {
  ScubaOtel otel("test_dataset");

  folly::dynamic message = folly::dynamic::object("normal",
      folly::dynamic::object("key", "val"))("int",
      folly::dynamic::object())("double", folly::dynamic::object())("normvector",
      folly::dynamic::object("empty_vec",
          folly::dynamic::array()))("tags",
      folly::dynamic::object());

  std::string jsonStr = folly::toJson(message);
  auto result = otel.addRawData("test_dataset", jsonStr, folly::none);
  EXPECT_EQ(result, 1);
}

TEST_F(ScubaOtelTest, AddSampleLargeValues) {
  ScubaOtel otel("test_dataset");

  std::unordered_map<std::string, std::string> normalMap;
  std::unordered_map<std::string, int64_t> intMap = {
      {"max_val", std::numeric_limits<int64_t>::max()},
      {"min_val", std::numeric_limits<int64_t>::min()}};
  std::unordered_map<std::string, double> doubleMap = {
      {"inf_val", std::numeric_limits<double>::infinity()},
      {"neg_inf", -std::numeric_limits<double>::infinity()}};

  auto result = otel.addSample("test_dataset", normalMap, intMap, doubleMap);
  EXPECT_EQ(result, 1);
}

TEST_F(ScubaOtelTest, MultipleDatasets) {
  // Verify multiple ScubaOtel instances with different datasets can coexist.
  ScubaOtel otel1("dataset_a");
  ScubaOtel otel2("dataset_b");

  std::unordered_map<std::string, std::string> normalMap = {{"k", "v"}};
  std::unordered_map<std::string, int64_t> intMap;
  std::unordered_map<std::string, double> doubleMap;

  EXPECT_EQ(otel1.addSample("dataset_a", normalMap, intMap, doubleMap), 1);
  EXPECT_EQ(otel2.addSample("dataset_b", normalMap, intMap, doubleMap), 1);
}
