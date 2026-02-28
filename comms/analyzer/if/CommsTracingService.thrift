// Copyright (c) Meta Platforms, Inc. and affiliates.

namespace cpp2 comms
namespace py comms.CommsTracingService
namespace py3 comms

cpp_include "<unordered_map>"
cpp_include "<unordered_set>"

include "thrift/annotation/cpp.thrift"
include "thrift/annotation/thrift.thrift"

@thrift.AllowLegacyMissingUris
package;

typedef i64 GlobalRank
typedef string CommHash
typedef i64 CommRank
typedef i64 Checksum

struct NCCLEntries {
  // 1: list<PyTorchRawEntry> entries;
  2: map<string, NCCLCommRawEntry> nccl_comm_state;
  @cpp.Type{template = "std::unordered_map"}
  3: map<string, NCCLParsedEntry> nccl_parsed_state;
  4: i64 completed_count;
  5: double completed_max_duration_ms;
  6: i64 completed_min_time_ns;
  7: i64 completed_max_time_ns;
  8: i64 started_count;
  9: i64 started_min_time_ns;
  10: i64 started_max_time_ns;
  11: i64 scheduled_count;
  12: i64 scheduled_min_time_ns;
  13: i64 scheduled_max_time_ns;

  14: i64 nccl_dump_start_time_ns;
  15: i64 nccl_dump_end_time_ns;
  16: GlobalInfo global_info;
}

struct PyTorchRawEntry {
  1: double duration_ms;
  2: i64 pg_id;
  3: string process_group_name; // string but int inside
  4: string profiling_name;
  5: i64 record_id;
  6: i64 seq_id;
  7: string state; // can be enum but bad luck
  8: i64 time_created_ns;
  9: i64 time_discovered_completed_ns;
  10: i64 time_discovered_started_ns;
}

struct CT_Coll_struct {
  // Unique operation on this communicator
  1: i64 opCount;
  2: string opName;
  /////// Timing //////////////////////////////////////////////////////////////
  //
  // See https://docs.google.com/document/d/1hVWsgAy-nFeeoU3thxIbTRgVeO1VzGw1HgRxl5HtIzw/edit?tab=t.0#heading=h.70k39tx7ljma
  // for details on timing.
  //
  // Timestamp the collective started running (not when it was enqueued)
  3: i64 startTs;
  // finish time - started time
  // based on cudaEventElapsedTime
  4: double latencyUs;
  // finish time - started time
  // based on cudaEventElapsedTime
  5: i64 ExecutionTimeUs;
  // collective N start ts - collective (N - 1) end ts
  6: i64 InterCollTimeUs;
  // startTs - enqueueTs
  7: i64 QueueingTimeUs;
  // time the collective was enqueued
  8: i64 enqueueTs;
  /////////////////////////////////////////////////////////////////////////////
  9: string algoName;
  10: string algorithm;
  11: string codepath;
  12: string dataType;
  13: string pattern;
  14: string protocol;
  15: string redOp;
  16: i64 root;
  17: i64 nChannels;
  18: i64 nThreads;
  19: optional Checksum checksum;
  // Step number for this collective
  20: i64 iteration;
  // Colltrace internal id
  21: i64 collId;
  // Size of the collective in bytes / sizeof(dataType)
  22: optional i64 count;
}

struct CT_Coll_list {
  1: list<CT_Coll_struct> ct_coll_list;
}

struct PT_Step {
  1: i64 step;
  2: i64 ts;
}

struct PT_Op_struct {
  1: string commHash;
  2: i64 opCount;
  3: string coll;
  4: i64 proxyOpId;
  5: i64 channelId;
  6: i64 rank;
  7: i64 remoteRank;
  8: string opType;
  9: string status;
  10: i64 startTs;
  11: i64 doneTs;
  12: optional PT_Step POSTED;
  13: optional PT_Step RECEIVED;
  14: optional PT_Step TRANSMITTED;
  15: optional PT_Step DONE;
  16: optional PT_Step REM_FIFO_WAIT;
}
struct PT_Coll_list {
  1: list<PT_Coll_struct> pt_coll_list;
}

struct PT_Coll_struct {
  1: string commHash;
  2: i64 opCount;
  3: string coll;
  4: i64 nProxyOps;
}

struct PT_Op_list {
  1: list<PT_Op_struct> pt_op_list;
}

struct MT_Event {
  1: string type;
  2: optional CommRank peerRank;
}

struct MT_Request {
  3: i64 seq_num;
  4: i64 timestamp; // Time in microsecond
  5: MT_Event event;
}

struct MT_Request_list {
  1: list<MT_Request> mt_req_list;
}

struct MT_Comm_map {
  1: map<CommRank, i64> mt_comm_map;
}

// NOTE: Keep in sync with dumpProcessGlobalErrors in commDump.cc
struct NicError {
  1: i64 timestampMs;
  2: string errorMessage;
}

@cpp.Type{template = "std::unordered_map"}
typedef map<string, NicError> PortNicErrorMap

// NOTE: Keep in sync with dumpProcessGlobalErrors in commDump.cc
struct ErrorAndStackTrace {
  1: i64 timestampMs;
  2: string errorMessage;
  3: list<string> stackTrace;
}

// NOTE: Keep in sync with dumpProcessGlobalErrors in commDump.cc
struct ProcessGlobalErrors {
  // Map of device name -> port as a string -> error message
  @cpp.Type{template = "std::unordered_map"}
  1: map<string, PortNicErrorMap> badNics;
  2: list<ErrorAndStackTrace> errorAndStackTraces;
}

struct IbCompletionError {
  1: i64 timestampMs;
  2: string peer;
  3: string statusStr;
  4: i32 status;
  5: string opcodeStr;
  6: i32 opcode;
  7: i32 reqSize;
  8: i64 vendorErr;
  9: string reqType;
  10: string localGid;
  11: string remoteGid;
  12: string hcaName;
  13: string scaleupDomain;
  14: string localHostname;
}

struct CudaError {
  1: i64 timestampMs;
  2: string errorString;
  3: i32 errorCode;
}

// NOTE: Keep in sync with commDump.cc
// The field names must exactly match the json keys.
// The values themselves are serialized json.
@thrift.ReserveIds{ids = [19, 20, 21]}
struct NCCLCommRawEntry {
  1: string CT_currentColl;
  2: string CT_pastColls;
  3: string CT_pendingColls;
  4: string PT_activeColls;
  5: string PT_activeOps;
  6: string PT_pastColls;
  7: string commHash;
  8: string localRank;
  9: string localRanks;
  10: string nNodes;
  11: string nRanks;
  12: string node;
  13: string rank;
  14: string commDesc;
  15: string MT_currentColl;
  16: string MT_unfinishedRequests;
  17: string MT_recvNotifiedByPeer;
  18: string MT_putFinishedByPeer;
  22: string stage;
  // serialized json of ProcessGlobalErrors
  23: string processGlobalErrors;
  24: string NetworkPerfMonitor;
  25: string ncclTopoInfo;
}

struct NCCLTreeNodeInfo {
  1: i64 parentNode;
  // We should have at max three nodes in this list. In the list, -1 means no
  // child node.
  2: list<i64> childrenNodes;
}

struct NCCLTopologyInfo {
  // For checking the correctness of treeInfo and rings. We should have a total
  // of nChannels of NCCLTreeNodeInfo for treeInfo and list<i64> for rings.
  1: i64 nChannels;
  // Information for who this local rank is connected in trees for each channel
  // We will report this for every rank and it is required to gather
  // the information from all the ranks to form the overall tree topology.
  2: list<NCCLTreeNodeInfo> treeInfos;
  // All the rings nccl uses in the current communicator. Currently only rank 0
  // will report this.
  3: optional list<list<i64>> rings;
}

// Reserving 24 for NetworkPerfMonitor, need to get the struct for it.
@thrift.ReserveIds{ids = [19, 20, 21, 24]}
struct NCCLParsedEntry {
  1: optional CT_Coll_struct CT_currentColl;
  2: list<CT_Coll_struct> CT_pastColls;
  3: list<CT_Coll_struct> CT_pendingColls;
  4: list<PT_Coll_struct> PT_activeColls;
  5: list<PT_Coll_struct> PT_pastColls;
  6: list<PT_Op_struct> PT_activeOps;
  7: string commHash;
  8: i64 localRank;
  9: i64 localRanks;
  10: i64 nNodes;
  11: i64 nRanks;
  12: i64 node;
  13: i64 rank;
  14: string commDesc;
  15: optional CT_Coll_struct MT_currentColl;
  16: list<MT_Request> MT_unfinishedRequests;
  17: map<CommRank, i64> MT_recvNotifiedByPeer;
  18: map<CommRank, i64> MT_putFinishedByPeer;
  22: string stage;
  23: ProcessGlobalErrors processGlobalErrors;
  25: NCCLTopologyInfo ncclTopoInfo;
}

struct GlobalInfo {
  1: optional NetworkPerfMonitor NetworkPerfMonitor;
}

struct NetworkPerfMonitor {
  1: double avgBw;
  2: list<CommNetworkPerfInfo> commAvgBw;
}

struct CommNetworkPerfInfo {
  1: string commHash;
  2: double avgBw;
  3: optional string commDesc;
}

struct CommsForRank {
  // TODO: Replace with types that are not vendor specific
  @cpp.Type{template = "std::unordered_map"}
  3: map<string, NCCLParsedEntry> ncclParsedEntryMap;
}

struct GetCommsRequest {}

struct GetCommsResponse {
  1: GlobalRank globalRank;
  2: CommsForRank commsForRank;
  // Current time locally on this rank
  3: i64 currentTimeNs;
  4: i64 jobStartTimeNs;
  // For training, this is the current step.
  // For inference, this is 0
  5: i64 step;
  6: i64 stepStartTimeNs;
  7: optional list<IbCompletionError> ibErrors;
  8: optional list<CudaError> cudaErrors;
}

// Implementors of this service expose tracing information about communications
// from *CCL libraries. Callers of this service can use this information to track
// hung collectives and other issues.
service CommsTracingService {
  GetCommsResponse getComms(1: GetCommsRequest request);
}
