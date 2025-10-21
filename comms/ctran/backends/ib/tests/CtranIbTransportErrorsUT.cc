// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdlib.h>
#include <chrono>
#include <climits>
#include <iostream>
#include <memory>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/dynamic.h>
#include <folly/init/Init.h>
#include <folly/json/json.h>

#include "comm.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/ibutils.h"
#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "nccl.h"

using namespace ::testing;
using namespace ctran::ibvwrap;

// Environment is set up once for all tests.
class MPIEnvironment : public DistEnvironmentBase {
 public:
  void SetUp() override {
    DistEnvironmentBase::SetUp();
    setenv("NCCL_HEALTH_WATCHER_ENABLE", "False", 1);
    setenv("NCCL_IB_ENABLE_REPORT_TO_PROCESS_GLOBAL_ERRORS", "True", 1);
    setenv("NCCL_COMM_DUMP_ENABLE_PROCESS_GLOBAL_ERRORS", "True", 1);
    setenv("NCCL_IB_ASYNC_EVENT_LOOP", "ctran", 1);
    setenv("NCCL_ERROR_TRACE_ENABLE", "True", 0);
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
  }
};

/*
 * Mock Ctran Verbs Wrapper module for testing
 */
class MockIbVerbsWrapper : public IVerbsWrapper {
 public:
  MockIbVerbsWrapper() : IVerbsWrapper() {
    ON_CALL(*this, ibv_ack_async_event)
        .WillByDefault(
            [this](ibverbx::ibv_async_event* event) { return commSuccess; });
  }
  ~MockIbVerbsWrapper() override = default;
  MOCK_METHOD(
      int,
      ibv_poll_async_fd,
      (struct pollfd * fds, nfds_t nfds, int timeout),
      (override));

  MOCK_METHOD(
      commResult_t,
      ibv_get_async_event,
      (ibverbx::ibv_context * context, ibverbx::ibv_async_event* event),
      (override));

  MOCK_METHOD(
      commResult_t,
      ibv_ack_async_event,
      (ibverbx::ibv_async_event * event),
      (override));
};

// CtranIbTest is instantiated per test.
class CtranIbTest : public NcclxBaseTest {
 public:
  CtranIbTest() = default;
  void SetUp() override {
    NcclxBaseTest::SetUp();
    this->commDeprecated = createNcclComm(
        globalRank, numRanks, localRank, false, nullptr, server.get());
    this->comm = this->commDeprecated->ctranComm_.get();
    this->ctrlMgr = std::make_unique<CtranCtrlManager>();
    CtranIbSingleton& s = CtranIbSingleton::getInstance();
    installedVerbs = s.verbsUtils->setVerbs<MockIbVerbsWrapper>();
  }

  void TearDown() override {
    CtranIbSingleton::getInstance().verbsUtils->setVerbs<VerbsWrapper>();
    ncclCommSetAsyncError(this->commDeprecated, ncclSuccess);
    finalizeNcclComm(globalRank, server.get());
    NCCLCHECK_TEST(ncclCommDestroy(this->commDeprecated));
    this->ctrlMgr.reset();
    NcclxBaseTest::TearDown();
  }

  void printTestDesc(const std::string& testName, const std::string& testDesc) {
    if (this->globalRank == 0) {
      std::cout << testName << " numRanks " << this->numRanks << "."
                << std::endl
                << testDesc << std::endl;
    }
  }

  MockIbVerbsWrapper* installedVerbs;

 protected:
  CtranComm* comm{nullptr};
  // TODO: remove this once refatoring is finished
  // !!! DO NOT USE !!!
  ncclComm_t commDeprecated{nullptr};
  std::unique_ptr<CtranCtrlManager> ctrlMgr{nullptr};
};

static std::unordered_map<std::string, std::string> pullNcclCommDump(
    ncclComm_t comm) {
  std::unordered_map<std::string, std::string> dump;
  ncclResult_t res = ncclCommDump(comm, dump);
  EXPECT_EQ(res, ncclSuccess);
  // Check if all the values can be parsed as json entries
  for (const auto& [key, val] : dump) {
    folly::dynamic js;
    EXPECT_NO_THROW(js = folly::parseJson(val));
    std::cout << key << ": " << js << std::endl;
  }
  return dump;
}

/* function extracts the difference in timestamps between
 * badNics and
 * the nth occurrence of the error stack trace containing the errMsg
 * where n = occurrence
 */
static int64_t extractDeltaTsBadNicStackTrace(
    const std::unordered_map<std::string, std::string>& dump,
    const std::string& errMsg,
    const int occurrence) {
  int64_t timestampMsTrace = 0, timestampMsBadNic = 0;
  for (const auto& [key, val] : dump) {
    if (key == "processGlobalErrors") {
      auto js = folly::parseJson(val);

      // extract the timestamp from badNics
      auto& badNics = js["badNics"];
      // Iterate over the interface name in 'badNics'
      for (auto& nicEntry : badNics.items()) {
        // Iterate over the port numbers
        for (auto& portEntry : nicEntry.second.items()) {
          timestampMsBadNic = portEntry.second["timestampMs"].asInt();
        }
      }

      // extract the timestamp for the specified event
      auto& stackTraces = js["errorAndStackTraces"];
      int count = 0;
      // Iterate over the array
      for (const auto& item : stackTraces) {
        std::string errorMessage = item["errorMessage"].asString();
        if (errorMessage.find(errMsg) != std::string::npos) {
          if (count++ == (occurrence - 1)) {
            timestampMsTrace = item["timestampMs"].asInt();
            break;
          }
        }
      }
      break;
    }
  }
  return timestampMsBadNic - timestampMsTrace;
}

/* No Errors. */
TEST_F(CtranIbTest, Baseline) {
  this->printTestDesc(
      "Baseline",
      "Expect CTranIbAsyncEventHandler to run without internal error.");
  CtranIbSingleton& s = CtranIbSingleton::getInstance();
  int asyncFdCounter = 0;

  // For testing, stop and restart the async event handler
  s.stopIbAsyncEventHandler();
  const int waitCount = 10;

  EXPECT_CALL(*installedVerbs, ibv_poll_async_fd(_, 1, _))
      .Times(AtLeast(waitCount))
      .WillRepeatedly(
          [&asyncFdCounter](struct pollfd* fds, nfds_t nfds, int timeout) {
            asyncFdCounter++;
            return 0;
          });
  EXPECT_CALL(*installedVerbs, ibv_get_async_event(_, _)).Times(0);

  s.startIbAsyncEventHandler(this->comm->statex_->cudaDev());
  try {
    auto ctranIb1 = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    auto ctranIb2 = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    // ensures ibv_poll_async_fd runs at least waitCount times
    for (int i = 0; i < 10; i++) {
      if (asyncFdCounter >= waitCount) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    EXPECT_GE(asyncFdCounter, waitCount);
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
  s.stopIbAsyncEventHandler();
}

/* ibv_poll_async_fd() returns with negative value (error). Thread
 * should exit but should not crash the process.
 */
TEST_F(CtranIbTest, PollError) {
  this->printTestDesc(
      "PollError", "Expect CTranIbAsyncEventHandler to exit gracefully.");
  int asyncFdCounter = 0;
  CtranIbSingleton& s = CtranIbSingleton::getInstance();

  // CtranIbSingleton is instantiated the first time CtranIb is instantiated
  // but freed when the process dies and so remains active throughout the test
  s.stopIbAsyncEventHandler();

  EXPECT_CALL(*installedVerbs, ibv_poll_async_fd(_, 1, _))
      .Times(1)
      .WillOnce(
          [&asyncFdCounter](struct pollfd* fds, nfds_t nfds, int timeout) {
            asyncFdCounter++;
            return -1;
          });
  // revents will be 0 and fail the check for POLLIN which will cause
  // the thread to break
  EXPECT_CALL(*installedVerbs, ibv_get_async_event(_, _)).Times(0);

  s.startIbAsyncEventHandler(this->comm->statex_->cudaDev());
  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    // ensures ibv_poll_async_fd runs at least once
    for (int i = 0; i < 10; i++) {
      if (asyncFdCounter >= 1) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
  pullNcclCommDump(this->commDeprecated);
  s.stopIbAsyncEventHandler();
}

/* ibv_poll_async_fd() returns with positive value meaning that
 *  it found an async event. It should call ibv_get_async_event()
 *  and return successfully
 */
TEST_F(CtranIbTest, AsyncEventFound) {
  this->printTestDesc(
      "AsyncEventFound", "Expect an async event and then return to polling.");
  CtranIbSingleton& s = CtranIbSingleton::getInstance();
  int asyncFdCounter = 0;

  // For testing, stop and restart the async event handler
  s.stopIbAsyncEventHandler();

  EXPECT_CALL(*installedVerbs, ibv_poll_async_fd(_, 1, _))
      .Times(1)
      .WillOnce(
          [&asyncFdCounter](struct pollfd* fds, nfds_t nfds, int timeout) {
            fds->revents = POLLIN;
            asyncFdCounter++;
            return 2;
          });

  EXPECT_CALL(*installedVerbs, ibv_get_async_event(_, _))
      .Times(1)
      .WillOnce(
          [&asyncFdCounter](
              ibverbx::ibv_context* context, ibverbx::ibv_async_event* event) {
            event->event_type = ibverbx::IBV_EVENT_QP_FATAL;
            return commSuccess;
          });

  EXPECT_CALL(*installedVerbs, ibv_ack_async_event(_)).Times(1);

  s.startIbAsyncEventHandler(this->comm->statex_->cudaDev());
  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    // ensures ibv_poll_async_fd runs at least waitCount times
    const int waitCount = 10;
    for (int i = 0; i < 10; i++) {
      if (asyncFdCounter >= waitCount) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }

  // IBV_EVENT_QP_FATAL will trigger a ncclSystemError
  ncclResult_t asyncError;
  NCCLCHECK_TEST(ncclCommGetAsyncError(this->commDeprecated, &asyncError));
  EXPECT_EQ(asyncError, ncclSystemError);

  pullNcclCommDump(this->commDeprecated);
  s.stopIbAsyncEventHandler();
}

/* This test includes 2 link flaps: The first one is
 * less than the timeout and the second exceeds the timeout
 * and will cause a failure.
 */
TEST_F(CtranIbTest, AsyncEventLinkFlap) {
  const int linkDownTimeout = 1; // set to 1s to make test run faster
  EnvRAII env1(NCCL_IB_LINK_DOWN_TIMEOUT, linkDownTimeout);
  this->printTestDesc(
      "AsyncEventLinkFlap", "Expect failure after second link down.");
  CtranIbSingleton& s = CtranIbSingleton::getInstance();
  int asyncFdCounter = 0;

  // For testing, stop and restart the async event handler
  s.stopIbAsyncEventHandler();

  EXPECT_CALL(*installedVerbs, ibv_poll_async_fd)
      .Times(AtLeast(3)) // down, up, down
      .WillOnce(
          [&asyncFdCounter](struct pollfd* fds, nfds_t nfds, int timeout) {
            fds->revents = POLLIN;
            asyncFdCounter++;
            return 5;
          })
      .WillOnce(
          [&asyncFdCounter](struct pollfd* fds, nfds_t nfds, int timeout) {
            fds->revents = POLLIN;
            asyncFdCounter++;
            // link up event 500ms later
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return 6;
          })
      .WillOnce(
          [&asyncFdCounter](struct pollfd* fds, nfds_t nfds, int timeout) {
            fds->revents = POLLIN;
            asyncFdCounter++;
            // link down event 500ms later
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return 7;
          })
      .WillRepeatedly( // wait for timeout
          [&asyncFdCounter](struct pollfd* fds, nfds_t nfds, int timeout) {
            fds->revents = POLLIN;
            asyncFdCounter++;
            return 0;
          });

  EXPECT_CALL(*installedVerbs, ibv_get_async_event(_, _))
      .Times(3)
      .WillOnce(
          [&asyncFdCounter](
              ibverbx::ibv_context* context, ibverbx::ibv_async_event* event) {
            event->event_type = ibverbx::IBV_EVENT_PORT_ERR; // link down
            return commSuccess;
          })
      .WillOnce(
          [&asyncFdCounter](
              ibverbx::ibv_context* context, ibverbx::ibv_async_event* event) {
            event->event_type = ibverbx::IBV_EVENT_PORT_ACTIVE; // link back up
            return commSuccess;
          })
      .WillOnce(
          [&asyncFdCounter](
              ibverbx::ibv_context* context, ibverbx::ibv_async_event* event) {
            event->event_type = ibverbx::IBV_EVENT_PORT_ERR; // link back down
            return commSuccess;
          });

  EXPECT_CALL(*installedVerbs, ibv_ack_async_event(_)).Times(3);

  s.startIbAsyncEventHandler(this->comm->statex_->cudaDev());
  ncclResult_t asyncError;
  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    // need to wait for timeout; total time > down + up + timeout (2s in this
    // test)
    for (int i = 0; i < 10; i++) {
      ncclCommGetAsyncError(this->commDeprecated, &asyncError);
      if (asyncError != ncclSuccess) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }

  ncclCommGetAsyncError(this->commDeprecated, &asyncError);
  EXPECT_EQ(asyncError, ncclSystemError);

  auto dump = pullNcclCommDump(this->commDeprecated);
  s.stopIbAsyncEventHandler();

  // there are two link down events in this test. Only the second
  // one is long enough to trigger a timeout.
  auto deltaTimestampMs =
      extractDeltaTsBadNicStackTrace(dump, "Got async event: port error", 2);

  // check that timeout is as configured
  std::cout << "Actual timeout was " << deltaTimestampMs << "ms" << std::endl;
  // Add margin to prevent false failures
  EXPECT_GE(deltaTimestampMs, linkDownTimeout * 1000 - 10);
  EXPECT_LT(deltaTimestampMs, linkDownTimeout * 2000);
}

/* Test that setting timeout to zero means no timeout.
 * This test includes 2 link flaps: The first one is
 * 500ms and the second one is hard down.
 */
TEST_F(CtranIbTest, LinkFlapZeroTimeout) {
  EnvRAII env1(NCCL_IB_LINK_DOWN_TIMEOUT, 0);
  this->printTestDesc(
      "LinkFlapZeroTimeout", "Expect stack trace but no failure.");
  CtranIbSingleton& s = CtranIbSingleton::getInstance();
  int asyncFdCounter = 0;

  // For testing, stop and restart the async event handler
  s.stopIbAsyncEventHandler();

  EXPECT_CALL(*installedVerbs, ibv_poll_async_fd)
      .Times(AtLeast(4)) // down, up, down
      .WillOnce(
          [&asyncFdCounter](struct pollfd* fds, nfds_t nfds, int timeout) {
            fds->revents = POLLIN;
            asyncFdCounter++;
            return 7;
          })
      .WillOnce(
          [&asyncFdCounter](struct pollfd* fds, nfds_t nfds, int timeout) {
            fds->revents = POLLIN;
            asyncFdCounter++;
            // link up event 500ms later
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return 8;
          })
      .WillOnce(
          [&asyncFdCounter](struct pollfd* fds, nfds_t nfds, int timeout) {
            fds->revents = POLLIN;
            asyncFdCounter++;
            // link down event 500ms later
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return 9;
          })
      .WillRepeatedly( // wait for timeout
          [&asyncFdCounter](struct pollfd* fds, nfds_t nfds, int timeout) {
            fds->revents = POLLIN;
            asyncFdCounter++;
            return 0;
          });

  EXPECT_CALL(*installedVerbs, ibv_get_async_event(_, _))
      .Times(3)
      .WillOnce(
          [&asyncFdCounter](
              ibverbx::ibv_context* context, ibverbx::ibv_async_event* event) {
            event->event_type = ibverbx::IBV_EVENT_PORT_ERR; // link down
            return commSuccess;
          })
      .WillOnce(
          [&asyncFdCounter](
              ibverbx::ibv_context* context, ibverbx::ibv_async_event* event) {
            event->event_type = ibverbx::IBV_EVENT_PORT_ACTIVE; // link back up
            return commSuccess;
          })
      .WillOnce(
          [&asyncFdCounter](
              ibverbx::ibv_context* context, ibverbx::ibv_async_event* event) {
            event->event_type = ibverbx::IBV_EVENT_PORT_ERR; // link back down
            return commSuccess;
          });

  EXPECT_CALL(*installedVerbs, ibv_ack_async_event(_)).Times(3);

  s.startIbAsyncEventHandler(this->comm->statex_->cudaDev());

  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    // need to wait for timeout; total time > down + up + timeout (2s in this
    // test)
    for (int i = 0; i < 10; i++) {
      if (asyncFdCounter >= 4) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }

  ncclResult_t asyncStatus;
  ncclCommGetAsyncError(this->commDeprecated, &asyncStatus);
  EXPECT_EQ(asyncStatus, ncclSuccess);

  auto dump = pullNcclCommDump(this->commDeprecated);
  s.stopIbAsyncEventHandler();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
