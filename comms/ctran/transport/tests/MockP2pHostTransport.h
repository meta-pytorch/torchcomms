// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <gmock/gmock.h>

#include "comms/ctran/transport/IP2pHostTransport.h"

namespace ctran::transport {

// GMock implementation of IP2pHostTransport for unit-testing algorithms
// that call into transports without requiring a real IB-backed comm.
class MockP2pHostTransport : public IP2pHostTransport {
 public:
  MOCK_METHOD(int, peerRank, (), (const, override));
  MOCK_METHOD(HostTransportMode, mode, (), (const, override));
  MOCK_METHOD(int, pipelineDepth, (), (const, override));
  MOCK_METHOD(size_t, chunkSize, (), (const, override));

  MOCK_METHOD(commResult_t, progress, (), (override));

  MOCK_METHOD(
      commResult_t,
      iSendCtrlMsg,
      (ControlMsgType type, const void* payload, size_t len, CtrlRequest* out),
      (override));
  MOCK_METHOD(
      commResult_t,
      iRecvCtrlMsg,
      (void* payload, size_t len, CtrlRequest* out),
      (override));

  MOCK_METHOD(int, computeTotalChunks, (size_t totalSize), (const, override));
  MOCK_METHOD(
      size_t,
      computeChunkOffset,
      (int chunkIdx, size_t totalSize),
      (const, override));
  MOCK_METHOD(
      size_t,
      computeChunkLen,
      (int chunkIdx, size_t totalSize),
      (const, override));

  MOCK_METHOD(bool, isReadyForSend, (int vcIdx, int stagingSlot), (override));
  MOCK_METHOD(bool, isReadyForRecv, (int vcIdx, int stagingSlot), (override));

  MOCK_METHOD(
      commResult_t,
      iSendChunk,
      (const SendChunkArgs& args),
      (override));
  MOCK_METHOD(
      commResult_t,
      iRecvChunk,
      (const RecvChunkArgs& args),
      (override));

  MOCK_METHOD(
      commResult_t,
      testChunkDone,
      (const ChunkRequest& req, bool* done),
      (override));

  MOCK_METHOD(
      commResult_t,
      testCtrlMsgDone,
      (CtrlRequest & req, bool* done),
      (override));
  MOCK_METHOD(commResult_t, waitCtrlMsgDone, (CtrlRequest & req), (override));

  // Per-transport lock (no-op for unit tests).
  void lock() override {}
  void unlock() override {}
};

} // namespace ctran::transport
