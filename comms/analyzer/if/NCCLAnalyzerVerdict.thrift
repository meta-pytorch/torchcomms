// Copyright (c) Meta Platforms, Inc. and affiliates.

include "thrift/annotation/cpp.thrift"
include "thrift/annotation/thrift.thrift"

@thrift.AllowLegacyMissingUris
package;

cpp_include "<unordered_set>"

namespace cpp2 facebook.comms.analyzer

typedef i64 CommRank
typedef i64 GlobalRank

struct BrokenRankInfo {
  1: string hostname;
  2: GlobalRank globalRank;
}

@thrift.DeprecatedUnvalidatedAnnotations{items = {"hash": "1"}}
enum BrokenRankType {
  UNKNOWN_BROKEN_RANK_TYPE = 0,
  NO_CONNECTION_VIA_HTTP = 1,
  FLAKY_OR_SLOW_CONNECTION_VIA_HTTP = 2,
  NO_REPLY_ON_STATUS = 3,
  NO_REPLY_ON_NCCL_DUMP = 4,
  EMPTY_NCCL_DUMP = 5,
  STUCK_OUTSIDE_NCCL = 6,
  STUCK_INSIDE_NCCL = 7,
  STUCK_INSIDE_NCCL_BASED_ON_MAPPER = 8,
  STUCK_INSIDE_NCCL_BASED_ON_PT = 9,
  STUCK_INSIDE_NCCL_BASED_ON_CROSS_COLL = 10,
  NCCL_LOCAL_FAILURE_TYPE = 11,
  NCCL_REMOTE_FAILURE_TYPE = 12,
  STUCK_INSIDE_NCCL_BASED_ON_INTERSECT_COLL = 13,
  RANK_UNRESPONSIVE_OVER_HTTP_TOO_LONG = 14,
  IB_COMPLETION_ERROR_TYPE = 15,
  CUDA_ERROR_TYPE = 16,
}

@thrift.DeprecatedUnvalidatedAnnotations{items = {"hash": "1"}}
enum VerdictType {
  UNKNOWN_VERDICT_TYPE = 0,
  ALL_GOOD = 1,
  JOB_DEAD_OR_DOES_NOT_EXIST = 2,
  JOB_STARTING_OR_STOPPING = 3,
  TOO_MANY_RANKS_FAILED = 4,
  SOME_RANKS_FAILED = 5,
  JOB_STUCK_OUTSIDE_NCCL = 6,
  JOB_STUCK_IN_NCCL_INITIALIZATION = 7,
  JOB_STUCK_IN_NCCL_WITH_ACTIVE_RANKS = 8,
  JOB_STUCK_IN_NCCL_WITH_INACTIVE_RANKS = 9,
  JOB_STUCK_IN_NCCL_BASED_ON_MAPPER = 10,
  JOB_STUCK_IN_NCCL_BASED_ON_PT = 11,
  PT_NETWORK_ROOT_ANALYSIS_DISAGREEMENT = 12,
  JOB_STUCK_IN_NCCL_BASED_ON_CROSS_COLL = 13,
  JOB_FAILED_AT_INIT = 14,
  JOB_COMM_DUMP_MISSING = 15,
  JOB_STUCK_IN_NCCL_RANK_NOT_JOIN = 16,
  JOB_STUCK_IN_NCCL_BASED_ON_INTERSECT_COLL = 17,
  SOME_RANKS_FISHY = 18,
  JOB_STUCK_IN_NCCL_UNKNOWN_BAD_RANKS = 19,
  CHECKSUM_MISMATCH = 20,
  RANKS_UNRESPONSIVE_OVER_HTTP_TOO_LONG = 21,
  NETWORK_PERF_SLOWNESS = 22,
  JOB_CONTAINS_COLL_DEADLOCK = 23,
  JOB_CONTAINS_COLL_METADATA_MISMATCH = 24,
  IB_COMPLETION_ERROR = 25,
  CUDA_ERROR = 26,
  CUDA_NVLINK_UNCORRECTABLE_ERROR = 27,
}

struct SlowRankLeastCommsVerdict {
  // Hosts that are causing the slowdown for the job identified
  // by the least comms heuristic.
  //
  // Verify that the top slowest ranks are on these hosts, and the
  // largest set of ranks are at least 10x larger (magnitude).
  @cpp.Type{template = "std::unordered_set"}
  1: set<string> hostnames;
  2: double magnitude = 10.0;
}

struct SlowRankOutlierCommsVerdict {
  // Hosts that are causing the slowdown for the job identified
  // by the outlier comms heuristic.
  //
  // Verify that the top slowest ranks are on these hosts, and these hosts have been
  // identified as outliers (by frequency count) at least 10x more often (magnitude)
  // then the non-anomolous ranks.
  @cpp.Type{template = "std::unordered_set"}
  1: set<string> hostnames;
  2: double magnitude = 10.0;
}

@thrift.ReserveIds{ids = [1]}
struct Verdict {
  2: list<BrokenRankInfo> brokenRankInfos;
  3: list<VerdictType> verdictList;
  4: optional SlowRankLeastCommsVerdict slowRankLeastCommsVerdict;
  5: optional SlowRankOutlierCommsVerdict slowRankOutlierCommsVerdict;
}
