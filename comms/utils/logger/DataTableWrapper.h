// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "comms/utils/logger/DataTable.h"
#include "comms/utils/logger/NcclScubaSample.h"

class DataTableWrapper {
 public:
  explicit DataTableWrapper(const std::string& tableName);

  void addSample(NcclScubaSample sample);

  static void init();
  static void shutdown();

  std::shared_ptr<DataTable> getTable();

 private:
  std::string tableName_;
  std::shared_ptr<DataTable> table_;
  std::once_flag tableInitFlag_;
};

// Struct is immutable after creation, so no need to wrap in Synchronized
struct DataTableAllTables {
  std::unordered_map<std::string, std::shared_ptr<DataTable>> tables;
};

struct DataTableAllTablesTag {};

std::vector<std::pair<std::string_view, std::string_view>> getAllTableNames();

// Convenience macros to make it easier to log. Example usage:
//
// header file:
// DECLARE_scuba_table(nccl_structured_logging);
//
// cc file:
// DEFINE_scuba_table(nccl_structured_logging);
//
// call sites:
// NcclScubaSample sample;
// ... // add fields to sample
// SCUBA_nccl_structured_logging.addSample(sample);
#define DECLARE_scuba_table(tablename) \
  extern std::unique_ptr<DataTableWrapper> SCUBA_##tablename##_ptr

#define DEFINE_scuba_table(tablename) \
  std::unique_ptr<DataTableWrapper> SCUBA_##tablename##_ptr = nullptr

#define INIT_scuba_table(tablename) \
  SCUBA_##tablename##_ptr = std::make_unique<DataTableWrapper>(#tablename)

#define SHUTDOWN_scuba_table(tablename)                               \
  do {                                                                \
    if (SCUBA_##tablename##_ptr && SCUBA_##tablename##_ptr->table_) { \
      SCUBA_##tablename##_ptr->table_->shutdown();                    \
    }                                                                 \
  } while (0)

// See: https://www.internalfb.com/intern/wiki/?fbid=1914217522335057
//
// Tables must exist here:
// https://www.internalfb.com/code/configerator/[master]/source/datainfra/logarithm/transport/logarithm_conda_custom_transport.cinc
DECLARE_scuba_table(nccl_coll_signature);
DECLARE_scuba_table(nccl_coll_timestamp);
DECLARE_scuba_table(nccl_commdump);
DECLARE_scuba_table(nccl_error_logging);
DECLARE_scuba_table(nccl_memory_logging);
DECLARE_scuba_table(nccl_profiler_slow_rank);
DECLARE_scuba_table(nccl_profiler_algo);
DECLARE_scuba_table(nccl_structured_logging);
DECLARE_scuba_table(nccl_slow_coll);
DECLARE_scuba_table(nccl_collective_stats);

#define SCUBA_nccl_structured_logging (*SCUBA_nccl_structured_logging_ptr)
#define SCUBA_nccl_profiler_algo (*SCUBA_nccl_profiler_algo_ptr)
#define SCUBA_nccl_profiler_slow_rank (*SCUBA_nccl_profiler_slow_rank_ptr)
#define SCUBA_nccl_memory_logging (*SCUBA_nccl_memory_logging_ptr)
#define SCUBA_nccl_error_logging (*SCUBA_nccl_error_logging_ptr)
#define SCUBA_nccl_coll_signature (*SCUBA_nccl_coll_signature_ptr)
#define SCUBA_nccl_coll_timestamp (*SCUBA_nccl_coll_timestamp_ptr)
#define SCUBA_nccl_commdump (*SCUBA_nccl_commdump_ptr)
#define SCUBA_nccl_slow_coll (*SCUBA_nccl_slow_coll_ptr)
#define SCUBA_nccl_collective_stats (*SCUBA_nccl_collective_stats_ptr)
