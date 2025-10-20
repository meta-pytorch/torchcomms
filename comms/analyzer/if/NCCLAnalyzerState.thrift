// Copyright (c) Meta Platforms, Inc. and affiliates.

namespace cpp2 facebook.comms.analyzer

include "thrift/annotation/thrift.thrift"

// Numerical rank identifier within a communicator.
typedef i64 CommRank
// Numerical rank identifier across all GPUs in training job.
typedef i64 GlobalRank

struct AnalyzerState {
  // Info about which hosts/ranks the analyzer has flagged as possibly having issues
  1: list<BrokenHostInfo> brokenHostInfos;

  // Info about which communicators have stuck collectives
  2: list<StuckCommunicatorInfo> stuckCommunicatorInfos;
}

struct BrokenHostInfo {
  // A particular host that the Analyzer has pointed out as faulty
  1: string hostname;
  // The ranks (identified by global rank number) that the Analyzer has pointed out as faulty for the hostname above.
  2: set<GlobalRank> globalRanks;
}

struct StuckCommunicatorInfo {
  1: string commHash;
  2: string commDesc; // i.e. process group desc + ":" + process group UID
  3: string reason; // extra info providing context about the problem
  4: string opName;
  5: i64 opCount;
  6: bool isCtranColl;

  // A list of info about every rank in the stuck collective
  7: list<StuckRankInfo> stuckRankInfos;
}

struct StuckRankInfo {
  1: GlobalRank globalRank;
  2: CommRank commRank;
  3: CollKernStatus collKernStatus;

  4: RankTimingInfo rankTimingInfo;
}

@thrift.DeprecatedUnvalidatedAnnotations{items = {"hash": "1"}}
enum CollKernStatus {
  UNKNOWN = 0,
  WAIT_START = 1,
  ACTIVE = 2,
  PENDING = 3,
  PAST = 4,
}

struct RankTimingInfo {
  // Timestamp the collective started running (not when it was enqueued)
  1: i64 startTs;
  // finish time - started time based on cudaEventElapsedTime
  2: i64 executionTimeUs;
  // collective N start ts - collective (N - 1) end ts
  3: i64 interCollTimeUs;
  // startTs - enqueueTs
  4: i64 queueingTimeUs;
  // time the collective was enqueued
  5: i64 enqueueTs;
}

service AnalyzerService {
}
