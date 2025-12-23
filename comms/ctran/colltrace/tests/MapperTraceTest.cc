// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/colltrace/MapperTrace.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "comms/utils/colltrace/CollRecord.h"

using namespace testing;
using namespace ncclx::colltrace;

// Mock CollRecord for testing
class MockCollRecord : public meta::comms::colltrace::ICollRecord {
 public:
  MOCK_METHOD(uint64_t, getCollId, (), (const, noexcept, override));
  MOCK_METHOD(folly::dynamic, toDynamic, (), (const, noexcept, override));
};

class MapperTraceTest : public Test {
 protected:
  void SetUp() override {
    ncclCvarInit();
    mapperTrace = std::make_unique<MapperTrace>();
  }

  std::unique_ptr<MapperTrace> mapperTrace;
};

/* Start of serialization tests */

TEST_F(MapperTraceTest, SerializeEvents) {
  // Test serialization of all event types
  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  folly::dynamic testDynamic = folly::dynamic::object("collId", 123);
  EXPECT_CALL(*mockColl, toDynamic()).WillRepeatedly(Return(testDynamic));

  // Test CollStart serialization
  CollStart collStart{mockColl};
  std::string serialized = collStart.serialize();
  EXPECT_THAT(serialized, HasSubstr("CollStart"));
  EXPECT_THAT(serialized, HasSubstr("coll"));

  // Test CollEnd serialization
  CollEnd collEnd{};
  serialized = collEnd.serialize();
  EXPECT_THAT(serialized, HasSubstr("CollEnd"));

  // Test CopyStart serialization
  CtranMapperRequest dummyReq;
  CopyStart copyStart{
      .sourceBuffer = reinterpret_cast<void*>(0x1000),
      .destBuffer = reinterpret_cast<void*>(0x2000),
      .length = 100,
      .stream = reinterpret_cast<cudaStream_t>(0x3000),
      .req = &dummyReq,
  };
  serialized = copyStart.serialize();
  EXPECT_THAT(serialized, HasSubstr("CopyStart"));
  EXPECT_THAT(serialized, HasSubstr("sourceBuffer"));
  EXPECT_THAT(serialized, HasSubstr("destBuffer"));
  EXPECT_THAT(serialized, HasSubstr("length"));

  // Test SendCtrlStart serialization
  SendCtrlStart sendCtrlStart{
      .buffer = reinterpret_cast<void*>(0x4000),
      .mapperHandle = reinterpret_cast<mapperHandle_t>(0x5000),
      .peerRank = 1,
      .req = &dummyReq,
  };
  serialized = sendCtrlStart.serialize();
  EXPECT_THAT(serialized, HasSubstr("SendCtrlStart"));
  EXPECT_THAT(serialized, HasSubstr("buffer"));
  EXPECT_THAT(serialized, HasSubstr("peerRank"));

  // Test RecvNotified serialization
  RecvNotified recvNotified{2};
  serialized = recvNotified.serialize();
  EXPECT_THAT(serialized, HasSubstr("RecvNotified"));
  EXPECT_THAT(serialized, HasSubstr("peerRank"));

  // Test MapperRequestEnd serialization
  MapperRequestEnd requestEnd{&dummyReq};
  serialized = requestEnd.serialize();
  EXPECT_THAT(serialized, HasSubstr("MapperRequestEnd"));
  EXPECT_THAT(serialized, HasSubstr("reqAddr"));
}

TEST_F(MapperTraceTest, SerializeWithQuotes) {
  // Test serialization with quoted parameter
  CollEnd collEnd{};
  std::string unquoted = collEnd.serialize(false);
  std::string quoted = collEnd.serialize(true);

  // Both should contain the type, but quoted version should have quotes around
  // string values
  EXPECT_THAT(unquoted, HasSubstr("CollEnd"));
  EXPECT_THAT(quoted, HasSubstr("CollEnd"));
}

TEST_F(MapperTraceTest, RecvCtrlStartSerialization) {
  // Test RecvCtrlStart serialization with valid pointers
  CtranMapperRequest dummyReq;
  remoteBuffer_t recvBuffer = reinterpret_cast<remoteBuffer_t>(0x1000);
  CtranMapperRemoteAccessKey accessKey;

  RecvCtrlStart recvCtrlStart{
      .recvBufferPtr = &recvBuffer,
      .accessKeyPtr = &accessKey,
      .peerRank = 2,
      .req = &dummyReq,
  };

  std::string serialized = recvCtrlStart.serialize();
  EXPECT_THAT(serialized, HasSubstr("RecvCtrlStart"));
  EXPECT_THAT(serialized, HasSubstr("recvBufferPtr"));
  EXPECT_THAT(serialized, HasSubstr("accessKeyPtr"));
  EXPECT_THAT(serialized, HasSubstr("peerRank"));
}

TEST_F(MapperTraceTest, PutStartSerialization) {
  // Test PutStart serialization
  CtranMapperRequest dummyReq;
  CtranMapperRemoteAccessKey accessKey;

  PutStart putStart{
      .sendBuffer = reinterpret_cast<void*>(0x1000),
      .remoteBuffer = reinterpret_cast<remoteBuffer_t>(0x2000),
      .length = 500,
      .peerRank = 3,
      .sourceHandle = reinterpret_cast<mapperHandle_t>(0x3000),
      .remoteAccessKey = accessKey,
      .req = &dummyReq,
  };

  std::string serialized = putStart.serialize();
  EXPECT_THAT(serialized, HasSubstr("remoteBuffer"));
  EXPECT_THAT(serialized, HasSubstr("length"));
  EXPECT_THAT(serialized, HasSubstr("peerRank"));
  EXPECT_THAT(serialized, HasSubstr("sourceHandle"));
}

TEST_F(MapperTraceTest, MapperRequestEventInfoSerialization) {
  // Test MapperRequestEventInfo serialization
  RecvNotified recvNotified{5};
  MapperRequestEventInfo eventInfo{
      .event = recvNotified,
      .seqNum = 42,
  };

  std::string serialized = eventInfo.serialize();
  EXPECT_THAT(serialized, HasSubstr("seqNum"));
  EXPECT_THAT(serialized, HasSubstr("event"));
  EXPECT_THAT(serialized, HasSubstr("RecvNotified"));
}

/* End of serialization tests */

TEST_F(MapperTraceTest, InitialState) {
  // Test initial state of MapperTrace
  auto dump = mapperTrace->dump();
  EXPECT_EQ(dump.currentColl, nullptr);
  EXPECT_TRUE(dump.recvNotifiedByPeer.empty());
  EXPECT_TRUE(dump.putFinishedByPeer.empty());
  EXPECT_TRUE(dump.unfinishedRequests.empty());
}

TEST_F(MapperTraceTest, RecordCollStartAndEnd) {
  // Create a mock CollRecord
  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  // Record CollStart event
  CollStart collStart{mockColl};
  mapperTrace->recordMapperEvent(collStart);

  // Verify CollStart was recorded
  auto dump = mapperTrace->dump();
  EXPECT_EQ(dump.currentColl, mockColl);

  // Record CollEnd event
  CollEnd collEnd{};
  mapperTrace->recordMapperEvent(collEnd);

  // Verify CollEnd cleared the state
  dump = mapperTrace->dump();
  EXPECT_EQ(dump.currentColl, nullptr);
  EXPECT_TRUE(dump.recvNotifiedByPeer.empty());
  EXPECT_TRUE(dump.putFinishedByPeer.empty());
  EXPECT_TRUE(dump.unfinishedRequests.empty());
}

TEST_F(MapperTraceTest, RecordRecvNotified) {
  // Create a mock CollRecord
  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  // Record CollStart event
  CollStart collStart{mockColl};
  mapperTrace->recordMapperEvent(collStart);

  // Record RecvNotified events
  RecvNotified recvNotified1{1};
  mapperTrace->recordMapperEvent(recvNotified1);

  RecvNotified recvNotified2{2};
  mapperTrace->recordMapperEvent(recvNotified2);

  RecvNotified recvNotified1Again{1};
  mapperTrace->recordMapperEvent(recvNotified1Again);

  // Verify RecvNotified events were recorded
  auto dump = mapperTrace->dump();
  EXPECT_EQ(dump.recvNotifiedByPeer.size(), 2);
  EXPECT_EQ(dump.recvNotifiedByPeer[1], 2);
  EXPECT_EQ(dump.recvNotifiedByPeer[2], 1);

  mapperTrace->recordMapperEvent(CollEnd{});
}

TEST_F(MapperTraceTest, RecordMapperRequestStartAndEnd) {
  // Create a mock CollRecord
  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  // Record CollStart event
  CollStart collStart{mockColl};
  mapperTrace->recordMapperEvent(collStart);

  // Create a dummy CtranMapperRequest
  CtranMapperRequest dummyReq;

  // Record PutStart event
  PutStart putStart{
      .sendBuffer = nullptr,
      .remoteBuffer = nullptr,
      .length = 100,
      .peerRank = 2,
      .sourceHandle = nullptr,
      .remoteAccessKey = {},
      .req = &dummyReq,
  };
  mapperTrace->recordMapperEvent(putStart);

  // Verify PutStart was recorded
  auto dump = mapperTrace->dump();
  EXPECT_EQ(dump.unfinishedRequests.size(), 1);

  // Record MapperRequestEnd event
  MapperRequestEnd requestEnd{&dummyReq};
  mapperTrace->recordMapperEvent(requestEnd);

  // Verify MapperRequestEnd was recorded
  dump = mapperTrace->dump();
  EXPECT_TRUE(dump.unfinishedRequests.empty());
  EXPECT_EQ(dump.putFinishedByPeer.size(), 1);
  EXPECT_EQ(dump.putFinishedByPeer[2], 1);

  mapperTrace->recordMapperEvent(CollEnd{});
}

TEST_F(MapperTraceTest, RegisterBeforeCollEndCallback) {
  bool callbackCalled = false;

  // Create a mock CollRecord
  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  // Record CollStart event
  CollStart collStart{mockColl};
  mapperTrace->recordMapperEvent(collStart);

  // Register callback
  mapperTrace->registerBeforeCollEndCallback(
      [&callbackCalled]() { callbackCalled = true; });

  // Record CollEnd event which should trigger the callback
  CollEnd collEnd{};
  mapperTrace->recordMapperEvent(collEnd);

  // Verify callback was called
  EXPECT_TRUE(callbackCalled);
}

TEST_F(MapperTraceTest, RecordCopyStart) {
  // Create a mock CollRecord
  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  // Record CollStart event
  CollStart collStart{mockColl};
  mapperTrace->recordMapperEvent(collStart);

  // Create a dummy CtranMapperRequest
  CtranMapperRequest dummyReq;

  // Record CopyStart event
  CopyStart copyStart{
      .sourceBuffer = nullptr,
      .destBuffer = nullptr,
      .length = 100,
      .stream = nullptr,
      .req = &dummyReq,
  };
  mapperTrace->recordMapperEvent(copyStart);

  // Verify CopyStart was recorded
  auto dump = mapperTrace->dump();
  EXPECT_EQ(dump.unfinishedRequests.size(), 1);

  // Record MapperRequestEnd event
  MapperRequestEnd requestEnd{&dummyReq};
  mapperTrace->recordMapperEvent(requestEnd);

  // Verify MapperRequestEnd was recorded
  dump = mapperTrace->dump();
  EXPECT_TRUE(dump.unfinishedRequests.empty());

  mapperTrace->recordMapperEvent(CollEnd{});
}

TEST_F(MapperTraceTest, RecordSendCtrlStart) {
  // Create a mock CollRecord
  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  // Record CollStart event
  CollStart collStart{mockColl};
  mapperTrace->recordMapperEvent(collStart);

  // Create a dummy CtranMapperRequest
  CtranMapperRequest dummyReq;

  // Record SendCtrlStart event
  SendCtrlStart sendCtrlStart{
      .buffer = nullptr,
      .mapperHandle = nullptr,
      .peerRank = 3,
      .req = &dummyReq,
  };
  mapperTrace->recordMapperEvent(sendCtrlStart);

  // Verify SendCtrlStart was recorded
  auto dump = mapperTrace->dump();
  EXPECT_EQ(dump.unfinishedRequests.size(), 1);

  // Record MapperRequestEnd event
  MapperRequestEnd requestEnd{&dummyReq};
  mapperTrace->recordMapperEvent(requestEnd);

  // Verify MapperRequestEnd was recorded
  dump = mapperTrace->dump();
  EXPECT_TRUE(dump.unfinishedRequests.empty());

  mapperTrace->recordMapperEvent(CollEnd{});
}

TEST_F(MapperTraceTest, RecordRecvCtrlStart) {
  // Create a mock CollRecord
  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  // Record CollStart event
  CollStart collStart{mockColl};
  mapperTrace->recordMapperEvent(collStart);

  // Create a dummy CtranMapperRequest and buffer pointers
  CtranMapperRequest dummyReq;
  remoteBuffer_t recvBuffer = nullptr;
  CtranMapperRemoteAccessKey accessKey;

  // Record RecvCtrlStart event
  RecvCtrlStart recvCtrlStart{
      .recvBufferPtr = &recvBuffer,
      .accessKeyPtr = &accessKey,
      .peerRank = 4,
      .req = &dummyReq,
  };
  mapperTrace->recordMapperEvent(recvCtrlStart);

  // Verify RecvCtrlStart was recorded
  auto dump = mapperTrace->dump();
  EXPECT_EQ(dump.unfinishedRequests.size(), 1);

  // Record MapperRequestEnd event
  MapperRequestEnd requestEnd{&dummyReq};
  mapperTrace->recordMapperEvent(requestEnd);

  // Verify MapperRequestEnd was recorded
  dump = mapperTrace->dump();
  EXPECT_TRUE(dump.unfinishedRequests.empty());

  mapperTrace->recordMapperEvent(CollEnd{});
}

TEST_F(MapperTraceTest, RecordSyncCtrlEvents) {
  // Create a mock CollRecord
  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  // Record CollStart event
  CollStart collStart{mockColl};
  mapperTrace->recordMapperEvent(collStart);

  // Create dummy CtranMapperRequests
  CtranMapperRequest sendReq;
  CtranMapperRequest recvReq;

  // Record SendSyncCtrlStart event
  SendSyncCtrlStart sendSyncCtrlStart{
      .peerRank = 5,
      .req = &sendReq,
  };
  mapperTrace->recordMapperEvent(sendSyncCtrlStart);

  // Record RecvSyncCtrlStart event
  RecvSyncCtrlStart recvSyncCtrlStart{
      .peerRank = 6,
      .req = &recvReq,
  };
  mapperTrace->recordMapperEvent(recvSyncCtrlStart);

  // Verify both events were recorded
  auto dump = mapperTrace->dump();
  EXPECT_EQ(dump.unfinishedRequests.size(), 2);

  // Record MapperRequestEnd events
  MapperRequestEnd sendRequestEnd{&sendReq};
  mapperTrace->recordMapperEvent(sendRequestEnd);

  MapperRequestEnd recvRequestEnd{&recvReq};
  mapperTrace->recordMapperEvent(recvRequestEnd);

  // Verify MapperRequestEnd events were recorded
  dump = mapperTrace->dump();
  EXPECT_TRUE(dump.unfinishedRequests.empty());

  mapperTrace->recordMapperEvent(CollEnd{});
}

TEST_F(MapperTraceTest, MultipleCollStartEvents) {
  // Test behavior when multiple CollStart events are recorded without CollEnd
  auto mockColl1 = std::make_shared<MockCollRecord>();
  auto mockColl2 = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl1, getCollId()).WillRepeatedly(Return(123));
  EXPECT_CALL(*mockColl2, getCollId()).WillRepeatedly(Return(456));

  // Record first CollStart
  CollStart collStart1{mockColl1};
  mapperTrace->recordMapperEvent(collStart1);

  auto dump = mapperTrace->dump();
  EXPECT_EQ(dump.currentColl, mockColl1);

  // Record second CollStart without CollEnd - should be ignored/warned
  CollStart collStart2{mockColl2};
  mapperTrace->recordMapperEvent(collStart2);

  dump = mapperTrace->dump();
  EXPECT_EQ(dump.currentColl, mockColl1); // Should still be the first one

  mapperTrace->recordMapperEvent(CollEnd{});
}

TEST_F(MapperTraceTest, NullCollStartEvent) {
  // Test CollStart with null collective
  CollStart collStart{nullptr};
  mapperTrace->recordMapperEvent(collStart);

  auto dump = mapperTrace->dump();
  EXPECT_EQ(dump.currentColl, nullptr);

  mapperTrace->recordMapperEvent(CollEnd{});
}

TEST_F(MapperTraceTest, MapperRequestEndWithoutStart) {
  // Test MapperRequestEnd without corresponding start event
  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  CollStart collStart{mockColl};
  mapperTrace->recordMapperEvent(collStart);

  CtranMapperRequest dummyReq;
  MapperRequestEnd requestEnd{&dummyReq};
  mapperTrace->recordMapperEvent(requestEnd);

  // Should handle gracefully without crashing
  auto dump = mapperTrace->dump();
  EXPECT_TRUE(dump.unfinishedRequests.empty());

  mapperTrace->recordMapperEvent(CollEnd{});
}

TEST_F(MapperTraceTest, EventHistorySequenceNumbers) {
  // Test that sequence numbers are assigned correctly
  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  CollStart collStart{mockColl};
  mapperTrace->recordMapperEvent(collStart);

  CtranMapperRequest req1, req2;

  // Record multiple events
  CopyStart copyStart{
      .sourceBuffer = nullptr,
      .destBuffer = nullptr,
      .length = 100,
      .stream = nullptr,
      .req = &req1,
  };
  mapperTrace->recordMapperEvent(copyStart);

  SendCtrlStart sendCtrlStart{
      .buffer = nullptr,
      .mapperHandle = nullptr,
      .peerRank = 1,
      .req = &req2,
  };
  mapperTrace->recordMapperEvent(sendCtrlStart);

  auto dump = mapperTrace->dump();
  EXPECT_EQ(dump.unfinishedRequests.size(), 2);

  // Check that sequence numbers are different
  std::set<uint64_t> seqNums;
  for (const auto& eventInfo : dump.unfinishedRequests) {
    seqNums.insert(eventInfo.seqNum);
  }
  EXPECT_EQ(seqNums.size(), 2); // Should have 2 unique sequence numbers

  mapperTrace->recordMapperEvent(CollEnd{});
}

TEST_F(MapperTraceTest, PutStartCountsCorrectly) {
  // Test that PutStart events are counted correctly when they finish
  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  CollStart collStart{mockColl};
  mapperTrace->recordMapperEvent(collStart);

  CtranMapperRequest req1, req2;

  // Record multiple PutStart events for different peers
  PutStart putStart1{
      .sendBuffer = nullptr,
      .remoteBuffer = nullptr,
      .length = 100,
      .peerRank = 1,
      .sourceHandle = nullptr,
      .remoteAccessKey = {},
      .req = &req1,
  };
  mapperTrace->recordMapperEvent(putStart1);

  PutStart putStart2{
      .sendBuffer = nullptr,
      .remoteBuffer = nullptr,
      .length = 200,
      .peerRank = 1, // Same peer
      .sourceHandle = nullptr,
      .remoteAccessKey = {},
      .req = &req2,
  };
  mapperTrace->recordMapperEvent(putStart2);

  // End both requests
  MapperRequestEnd requestEnd1{&req1};
  mapperTrace->recordMapperEvent(requestEnd1);

  MapperRequestEnd requestEnd2{&req2};
  mapperTrace->recordMapperEvent(requestEnd2);

  auto dump = mapperTrace->dump();
  EXPECT_EQ(dump.putFinishedByPeer[1], 2); // Both puts finished for peer 1

  mapperTrace->recordMapperEvent(CollEnd{});
}

TEST_F(MapperTraceTest, ThreadLocalShouldMapperTrace) {
  // Test the thread_local variable behavior
  EXPECT_FALSE(shouldMapperTraceCurrentThread);

  auto mockColl = std::make_shared<MockCollRecord>();
  EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

  // CollStart should set shouldMapperTraceCurrentThread to true
  CollStart collStart{mockColl};
  mapperTrace->recordMapperEvent(collStart);
  EXPECT_TRUE(shouldMapperTraceCurrentThread);

  // CollEnd should set shouldMapperTraceCurrentThread to false
  CollEnd collEnd{};
  mapperTrace->recordMapperEvent(collEnd);
  EXPECT_FALSE(shouldMapperTraceCurrentThread);
}

TEST_F(MapperTraceTest, TwoThreadsLogging) {
  auto normalLoggingThread = std::thread([&]() {
    // Test the thread_local variable behavior
    EXPECT_FALSE(shouldMapperTraceCurrentThread);

    auto mockColl = std::make_shared<MockCollRecord>();
    EXPECT_CALL(*mockColl, getCollId()).WillRepeatedly(Return(123));

    for (int i = 0; i < 1000; i++) {
      // CollStart should set shouldMapperTraceCurrentThread to true
      CollStart collStart{mockColl};
      mapperTrace->recordMapperEvent(collStart);
      EXPECT_TRUE(shouldMapperTraceCurrentThread);

      CtranMapperRequest req1;
      // Record multiple PutStart events for different peers
      PutStart putStart1{
          .sendBuffer = nullptr,
          .remoteBuffer = nullptr,
          .length = 100,
          .peerRank = 1,
          .sourceHandle = nullptr,
          .remoteAccessKey = {},
          .req = &req1,
      };
      mapperTrace->recordMapperEvent(putStart1);

      // CollEnd should set shouldMapperTraceCurrentThread to false
      CollEnd collEnd{};
      mapperTrace->recordMapperEvent(collEnd);
      EXPECT_FALSE(shouldMapperTraceCurrentThread);
    }
  });

  auto abnormalLoggingThread = std::thread([&]() {
    // Test the thread_local variable behavior
    EXPECT_FALSE(shouldMapperTraceCurrentThread);

    // Loop sufficiently long to ensure that we should encounter a race
    for (int i = 0; i < 1000; i++) {
      CtranMapperRequest req1;
      // Record multiple PutStart events for different peers
      PutStart putStart1{
          .sendBuffer = nullptr,
          .remoteBuffer = nullptr,
          .length = 100,
          .peerRank = 1,
          .sourceHandle = nullptr,
          .remoteAccessKey = {},
          .req = &req1,
      };
      mapperTrace->recordMapperEvent(putStart1);
    }
  });

  // As long as the program doesn't crash, we can assume that there is no race
  normalLoggingThread.join();
  abnormalLoggingThread.join();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
