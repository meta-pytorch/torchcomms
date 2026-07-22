// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/LoggingFormat.h"

#include <sstream>
#include <string>

#include <folly/Singleton.h>
#include <folly/init/Init.h>
#include <folly/json/dynamic.h>
#include <folly/json/json.h>
#include <folly/portability/GTest.h>
#include <folly/testing/TestUtil.h>

#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/DataTableWrapper.h"
#include "comms/utils/logger/EventMgr.h"
#include "comms/utils/logger/tests/MockScubaTable.h"

using meta::comms::logger::logErrorToScuba;

namespace {

// Drives logErrorToScuba() end-to-end into a mocked nccl_structured_logging
// table so the flushed sample JSON can be inspected. The DataTableAllTables
// singleton is replaced with MockScubaTable instances that capture the
// serialized sample instead of writing it to a pipe/scuba sink.
class LogErrorToScubaTest : public ::testing::Test {
 public:
  void SetUp() override {
    savedLogFilePrefix_ = NCCL_SCUBA_LOG_FILE_PREFIX;
    savedCommEventLogging_ = NCCL_COMM_EVENT_LOGGING;

    // The nccl_structured_logging table is created only when its cvar names a
    // "type:name" sink; point it at a pipe backed by a throwaway temp dir so
    // the DataTable is constructed (and its writer thread runs) while the mock
    // still captures every sample.
    NCCL_SCUBA_LOG_FILE_PREFIX = tmpDir_.path().string();
    NCCL_COMM_EVENT_LOGGING = "pipe:nccl_structured_logging";

    folly::Singleton<const DataTableAllTables, DataTableAllTablesTag>::
        make_mock([]() {
          return new DataTableAllTables(
              createAllMockTables(/*callParent=*/false));
        });
    DataTableWrapper::init();
  }

  void TearDown() override {
    DataTableWrapper::shutdown();
    NCCL_SCUBA_LOG_FILE_PREFIX = savedLogFilePrefix_;
    NCCL_COMM_EVENT_LOGGING = savedCommEventLogging_;
  }

  // Flush the async writer thread and return the parsed JSON of the single
  // sample logErrorToScuba() emitted.
  folly::dynamic getLoggedSample() {
    DataTableWrapper::shutdown();
    std::istringstream iss(getMessages(LoggerEventType::CommEventType));
    std::string line;
    std::getline(iss, line);
    return folly::parseJson(line);
  }

 private:
  folly::test::TemporaryDirectory tmpDir_;
  std::string savedLogFilePrefix_;
  std::string savedCommEventLogging_;
};

// A non-zero code is recorded as "code:name", alongside the message and stack.
TEST_F(LogErrorToScubaTest, RecordsErrorCodeMessageAndStack) {
  logErrorToScuba("msg", 5, "ncclSystemError", {"f1", "f2"});

  const auto json = getLoggedSample();

  EXPECT_EQ(json["normal"]["error_code"].asString(), "5:ncclSystemError");
  EXPECT_EQ(json["normal"]["exception_message"].asString(), "msg");
  const auto expectedStack = folly::dynamic::array("f1", "f2");
  EXPECT_EQ(json["normvector"]["stack_trace"], expectedStack);
}

// An empty error name still records the code with a trailing colon.
TEST_F(LogErrorToScubaTest, ErrorCodeWithEmptyName) {
  logErrorToScuba("msg", 5, "", {});

  const auto json = getLoggedSample();

  EXPECT_EQ(json["normal"]["error_code"].asString(), "5:");
}

// A zero code omits the error_code column entirely.
TEST_F(LogErrorToScubaTest, NoErrorCodeWhenCodeIsZero) {
  logErrorToScuba("msg", 0, "ncclSuccess", {});

  const auto json = getLoggedSample();

  EXPECT_EQ(json["normal"].count("error_code"), 0);
}

} // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
