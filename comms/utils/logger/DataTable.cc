// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/DataTable.h"

#include <atomic>
#include <optional>

#include <fmt/format.h>
#include <folly/File.h>
#include <folly/FileUtil.h>
#include <folly/system/ThreadName.h>

#include "comms/utils/EnvUtils.h"
#include "comms/utils/RankUtils.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/BackendTopologyUtil.h"
#include "comms/utils/logger/ScubaFileUtils.h"

namespace {

struct JobFields {
  std::string jobName;
  int64_t jobVersion{0};
  int64_t jobAttempt{0};
  int64_t jobQuorumRestartId{0};
  std::string jobIdStr;
  // Used to identify replica ID for PAFT
  std::string jobTaskGroupName;
  std::string twJobName;
  std::string twGangId;
  int64_t twGangMemberId{};
  int64_t twTaskId{};
};

JobFields getJobFields() {
  JobFields jobFields;
  if (NCCL_HPC_JOB_IDS.size() >= 3) {
    jobFields.jobName =
        meta::comms::getStrEnv(NCCL_HPC_JOB_IDS[0]).value_or("");
    jobFields.jobVersion =
        RankUtils::getInt64FromEnv(NCCL_HPC_JOB_IDS[1].c_str()).value_or(0);
    jobFields.jobAttempt =
        RankUtils::getInt64FromEnv(NCCL_HPC_JOB_IDS[2].c_str()).value_or(0);
    jobFields.jobQuorumRestartId =
        RankUtils::getInt64FromEnv("TW_RESTART_ID").value_or(-1);
    jobFields.jobIdStr = fmt::format(
        "{}:{}:{}:{}",
        jobFields.jobName,
        jobFields.jobVersion,
        jobFields.jobAttempt,
        jobFields.jobQuorumRestartId);
  }
  if (auto taskGroupName = getenv("MAST_HPC_TASK_GROUP_NAME");
      taskGroupName != nullptr) {
    jobFields.jobTaskGroupName = taskGroupName;
  }
  if (auto twJobName = getenv("TW_JOB_NAME"); twJobName != nullptr) {
    jobFields.twJobName = twJobName;
  }
  if (auto twGangId = getenv("TW_GANG_ID"); twGangId != nullptr) {
    jobFields.twGangId = twGangId;
  }
  jobFields.twGangMemberId =
      RankUtils::getInt64FromEnv("TW_GANG_MEMBER_ID").value_or(-1);
  jobFields.twTaskId = RankUtils::getInt64FromEnv("TW_TASK_ID").value_or(-1);
  return jobFields;
}

// Fields that never change upon initialization, and should be included
// with every sample.
struct CommonFields {
  std::string hostname;
  int64_t globalRank{-1};
  int64_t worldSize{-1};
  JobFields jobFields;
  std::string fastInitMode;
  std::optional<BackendTopologyUtil::Topology> backendTopology;
};

static CommonFields kCommonFields;
std::once_flag kCommonFieldsOnceFlag;

std::string getHostname() {
  char hostname[64];
  bzero(hostname, sizeof(hostname));
  // To make sure string is null terminated when hostname exceeds
  // the buffer size pass buffer size - 1
  if (gethostname(hostname, sizeof(hostname) - 1) == 0) {
    return hostname;
  }
  // return empty string on error
  return "";
}

void setCommonFields() {
  kCommonFields.hostname = getHostname();
  kCommonFields.globalRank = RankUtils::getGlobalRank().value_or(-1);
  kCommonFields.worldSize = RankUtils::getWorldSize().value_or(-1);
  kCommonFields.jobFields = getJobFields();

  auto fastInitMode = getenv("NCCL_FASTINIT_MODE");
  if (fastInitMode != nullptr) {
    kCommonFields.fastInitMode = std::string(fastInitMode);
  }
  if (NCCL_SCUBA_ENABLE_INCLUDE_BACKEND_TOPOLOGY) {
    kCommonFields.backendTopology =
        BackendTopologyUtil::getBackendTopology("/etc/fbwhoami");
  }
}

// Every log gets a monotonically increasing sequence number.
// This allows us to identify the last sample from this rank
// (scuba query: take the MAX of this column) to see what the rank
// was doing last.
std::atomic<int64_t> kSampleSequenceNumber{0};

void addCommonFieldsToSample(NcclScubaSample& sample) {
  std::call_once(kCommonFieldsOnceFlag, setCommonFields);
  // Start of Lite Scuba Sample Fields
  sample.addInt("sequenceNumber", kSampleSequenceNumber++);
  sample.addNormal("hostname", kCommonFields.hostname);
  sample.addInt("globalRank", kCommonFields.globalRank);
  sample.addNormal("jobName", kCommonFields.jobFields.jobName);
  sample.addInt("jobVersion", kCommonFields.jobFields.jobVersion);
  sample.addInt("jobAttempt", kCommonFields.jobFields.jobAttempt);
  sample.addNormal(
      "jobTaskGroupName", kCommonFields.jobFields.jobTaskGroupName);
  sample.addInt(
      "jobQuorumRestartId", kCommonFields.jobFields.jobQuorumRestartId);
  sample.addNormal("twJob", kCommonFields.jobFields.twJobName);
  sample.addNormal("twGangId", kCommonFields.jobFields.twGangId);
  sample.addInt("twGangMemberId", kCommonFields.jobFields.twGangMemberId);
  sample.addInt("twTaskId", kCommonFields.jobFields.twTaskId);

  if (sample.getLogType() == NcclScubaSample::ScubaLogType::LITE) {
    return;
  }

  // Start of Regular Scuba Sample Fields
  sample.addInt("worldSize", kCommonFields.worldSize);
  sample.addNormal("jobIdStr", kCommonFields.jobFields.jobIdStr);
  sample.addNormal("fastinit_mode", kCommonFields.fastInitMode);
  // TODO: should we add this to each sample?
  if (kCommonFields.backendTopology.has_value()) {
    sample.addNormal(
        "backend_topology_sfz", kCommonFields.backendTopology->sfz);
    sample.addNormal(
        "backend_topology_region", kCommonFields.backendTopology->region);
    sample.addNormal("backend_topology_dc", kCommonFields.backendTopology->dc);
    sample.addNormal(
        "backend_topology_ai_zone", kCommonFields.backendTopology->zone);
    sample.addNormal(
        "backend_topology_rtsw", kCommonFields.backendTopology->rtsw);
    sample.addNormal(
        "backend_topology_scaleup_domain",
        kCommonFields.backendTopology->scaleUp.domain);
    sample.addNormal(
        "backend_topology_scaleup_rack",
        kCommonFields.backendTopology->scaleUp.rack);
    sample.addNormal(
        "backend_topology_scaleup_unit",
        kCommonFields.backendTopology->scaleUp.unit);
    sample.addNormVector(
        "backend_topology_full_scopes_stack",
        kCommonFields.backendTopology->fullScopes);
  }
}
} // namespace

// We cannot log to scuba directly from conda. Instead, we log to a file
// and then a separate process scans the logs and uploads to scuba.
DataTable::DataTable(const std::string& tableType, const std::string& tableName)
    : tableName_(tableName) {
  if (tableType == "pipe") {
    auto fileName =
        comms::logger::getScubaFileName(NCCL_SCUBA_LOG_FILE_PREFIX, tableName);
    file_ = comms::logger::createScubaFile(fileName);
  } else if (tableType == "scuba") {
    sink_ = std::make_unique<DataSink>(tableName);
  }
  thread_ = std::thread([this, tableName] {
    folly::setThreadName("scuba_logger_" + tableName);
    loggingFunc();
  });
}

void DataTable::shutdown() {
  // use thread joinable check as proxy for whether scuba table is active.
  if (thread_.joinable()) {
    state_.lock()->stopTriggered = true;
    cv_.notify_one();
    thread_.join();
    if (file_.has_value()) {
      file_->close();
    }
  }
}

DataTable::~DataTable() {
  shutdown();
}

void DataTable::addSample(NcclScubaSample sample) {
  state_.lock()->samples.emplace_back(std::move(sample));
  cv_.notify_one();
}

// Wait until there are messages, or until shutdown is triggered.
// Returns State
DataTable::State DataTable::waitAndGetAllMessages() {
  auto locked = state_.lock();
  cv_.wait(locked.as_lock(), [&locked] {
    return !locked->samples.empty() || locked->stopTriggered;
  });
  State state;
  std::swap(state.samples, locked->samples);
  state.stopTriggered = locked->stopTriggered;
  return state;
}

void DataTable::loggingFunc() {
  if (!file_.has_value() && !sink_) {
    return;
  }
  while (true) {
    auto state = waitAndGetAllMessages();

    // Log all scuba-samples to the file. We do populate common fields and
    // perform serialization here to keep the work to bare minimum when
    // sample is being submitted
    for (auto& sample : state.samples) {
      addCommonFieldsToSample(sample);
      auto message = sample.toJson();
      writeMessage(message);
    }

    if (state.stopTriggered) {
      break;
    }
  }
}

void DataTable::writeMessage(const std::string& message) {
  if (file_.has_value()) {
    folly::writeFull(file_->fd(), message.data(), message.size());
    folly::writeFull(file_->fd(), "\n", 1);
  } else if (sink_) {
    sink_->addRawData(tableName_, message, folly::none);
  }
}
