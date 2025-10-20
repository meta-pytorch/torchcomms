// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/DataTableWrapper.h"

#include <chrono>

#include <folly/Singleton.h>
#include <folly/String.h>
#include <folly/logging/xlog.h>

#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars

namespace {
constexpr std::string_view k_nccl_memory_logging = "nccl_memory_logging";
constexpr std::string_view k_nccl_profiler_slow_rank =
    "nccl_profiler_slow_rank";
constexpr std::string_view k_nccl_profiler_algo = "nccl_profiler_algo";
constexpr std::string_view k_nccl_structured_logging =
    "nccl_structured_logging";
constexpr std::string_view k_nccl_slow_coll = "nccl_slow_coll";
constexpr std::string_view k_nccl_collective_stats = "nccl_collective_stats";
} // namespace

namespace {

DataTableAllTables createAllTables() {
  DataTableAllTables allTables;
  for (const auto& [tableId, tableTypeName] : getAllTableNames()) {
    // Ignore empty table names (user doens't want it to be populated)
    if (tableTypeName.empty()) {
      continue;
    }

    // Parse table type and name
    std::vector<std::string> tokens;
    folly::split(':', tableTypeName, tokens, true /* ignoreEmpty */);
    if (tokens.size() != 2) {
      XLOG(ERR) << "Invalid table type and name: " << tableTypeName
                << ". Ignoring";
      continue;
    }
    auto& tableType = tokens.at(0);
    auto& tableName = tokens.at(1);
    if (tableType != "pipe" && tableType != "scuba") {
      XLOG(ERR) << "Invalid table type: " << tableType
                << " for table: " << tableName << ". Ignoring";
      continue;
    }

    // Add more tables here. We can refer the table by both the fixed tableId
    // (e.g. nccl_structured_log), or by the user provided name via CVAR.
    auto table = std::make_shared<DataTable>(tableType, tableName);
    allTables.tables[std::string(tableId)] = table;
    allTables.tables[tableName] = table;
  }
  return allTables;
}

// Initialize a singleton containing all the tables
static folly::Singleton<const DataTableAllTables, DataTableAllTablesTag>
    kDataTableAllTables([]() {
      return new DataTableAllTables(createAllTables());
    });

} // namespace

std::vector<std::pair<std::string_view, std::string_view>> getAllTableNames() {
  std::vector<std::pair<std::string_view, std::string_view>> tableNames{
      {k_nccl_memory_logging, NCCL_MEMORY_EVENT_LOGGING},
      {k_nccl_profiler_slow_rank, NCCL_SLOW_RANK_LOGGING},
      {k_nccl_profiler_algo, NCCL_CTRAN_ALGO_PROFILING_LOGGING},
      {k_nccl_structured_logging, NCCL_COMM_EVENT_LOGGING},
      {k_nccl_slow_coll, NCCL_SLOW_COLL_LOGGING},
      {k_nccl_collective_stats, NCCL_COLL_STATS_LOGGING},
  };
  return tableNames;
}

DataTableWrapper::DataTableWrapper(const std::string& tableName)
    : tableName_(tableName) {}

void DataTableWrapper::addSample(NcclScubaSample sample) {
  // Lazy initialize the table pointer on first use
  std::call_once(tableInitFlag_, [this]() {
    auto allTables = kDataTableAllTables.try_get();
    table_ = folly::get_default(allTables->tables, tableName_, nullptr);

    if (!table_) {
      XLOG(WARNING) << "Could not find table name: " << tableName_ << std::endl;
    }
  });

  if (table_ != nullptr) {
    // Add timestamp when sample is published
    const auto timestamp = std::chrono::system_clock::now().time_since_epoch();
    sample.addInt(
        "time",
        std::chrono::duration_cast<std::chrono::seconds>(timestamp).count());
    sample.addNormal(
        "timestamp",
        std::to_string(
            std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp)
                .count()));
    table_->addSample(std::move(sample));
  }
}

std::shared_ptr<DataTable> DataTableWrapper::getTable() {
  return table_;
}

DEFINE_scuba_table(nccl_coll_signature);
DEFINE_scuba_table(nccl_coll_timestamp);
DEFINE_scuba_table(nccl_commdump);
DEFINE_scuba_table(nccl_error_logging);
DEFINE_scuba_table(nccl_memory_logging);
DEFINE_scuba_table(nccl_profiler_slow_rank);
DEFINE_scuba_table(nccl_profiler_algo);
DEFINE_scuba_table(nccl_structured_logging);
DEFINE_scuba_table(nccl_slow_coll);
DEFINE_scuba_table(nccl_collective_stats);

void DataTableWrapper::init() {
  INIT_scuba_table(nccl_coll_signature);
  INIT_scuba_table(nccl_coll_timestamp);
  INIT_scuba_table(nccl_commdump);
  INIT_scuba_table(nccl_error_logging);
  INIT_scuba_table(nccl_memory_logging);
  INIT_scuba_table(nccl_profiler_slow_rank);
  INIT_scuba_table(nccl_profiler_algo);
  INIT_scuba_table(nccl_structured_logging);
  INIT_scuba_table(nccl_slow_coll);
  INIT_scuba_table(nccl_collective_stats);
}

void DataTableWrapper::shutdown() {
  SHUTDOWN_scuba_table(nccl_coll_signature);
  SHUTDOWN_scuba_table(nccl_coll_timestamp);
  SHUTDOWN_scuba_table(nccl_commdump);
  SHUTDOWN_scuba_table(nccl_error_logging);
  SHUTDOWN_scuba_table(nccl_memory_logging);
  SHUTDOWN_scuba_table(nccl_profiler_slow_rank);
  SHUTDOWN_scuba_table(nccl_profiler_algo);
  SHUTDOWN_scuba_table(nccl_structured_logging);
  SHUTDOWN_scuba_table(nccl_slow_coll);
  SHUTDOWN_scuba_table(nccl_collective_stats);
}
