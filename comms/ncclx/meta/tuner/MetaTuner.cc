// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/tuner/MetaTuner.h"

#include <cstdint>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "comm.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "debug.h"
#include "meta/tuner/CsvConfigParse.h"
#include "meta/tuner/Int64Range.h"

#ifdef NCCLX_TUNER_WITH_FOLLY_JSON
#include <folly/json/dynamic.h>
#include <folly/json/json.h>

#include "meta/tuner/JsonConfigParse.h"
#endif

namespace ncclx::tuner {

std::string TuningConfig::toString() const {
  return fmt::format(
      "collType={}, bytesPerRank={}, algorithm={}, protocol={}, "
      "nChannels={}, nNodes={}, nLocalRanks={}, numPipeOps={}, regBuff={}, "
      "chunkSize={}",
      static_cast<int>(collType),
      bytesPerRank.toString(),
      algorithm,
      protocol,
      nChannels,
      nNodes.toString(),
      nLocalRanks.toString(),
      numPipeOps,
      regBuff,
      chunkSize);
}

namespace {

// Version-portable aliases for the init callback's versioned parameter types.
// The init function-pointer field is declared with _v6_t params in NCCLX v2.30
// (tuner API v6) and _v5_t params in v2.29 (tuner API v5); a free function
// assigned to it must match the version's struct exactly. v2.29 does not even
// define the _v6_t names, so they must never appear unguarded.
#ifdef NCCLX_TUNER_HAS_GETCHUNKSIZE
using MetaNvlDomainInfo = ncclNvlDomainInfo_v6_t;
using MetaTunerConstants = ncclTunerConstants_v6_t;
#else
using MetaNvlDomainInfo = ncclNvlDomainInfo_v5_t;
using MetaTunerConstants = ncclTunerConstants_v5_t;
#endif

// Heap context allocated per comm in metaTunerInit and freed in
// metaTunerFinalize. Holds the parsed rule table and the NCCL logger callback.
struct MetaTunerContext {
  std::vector<TuningConfig> configs;
  ncclDebugLogger_t logFunction{nullptr};
  int nNodes{-1};
  int nLocalRanks{-1};
};

// Emits a single line through the NCCL logger callback (if provided). The
// message is formatted with fmt and passed via a "%s" format to keep the
// variadic C callback type-safe. The file/line identify the call site (see the
// NCCLX_TUNER_LOG macro), not this function.
void logLine(
    const ncclDebugLogger_t logFunction,
    const ncclDebugLogLevel level,
    const char* const file,
    const int line,
    const std::string& message) {
  if (logFunction != nullptr) {
    logFunction(level, NCCL_TUNING, file, line, "%s", message.c_str());
  }
}

// Logs through the NCCL logger, capturing the CALL SITE's file/line (like NCCL
// core's WARN/INFO macros) so messages point at the real source, not this file.
#define NCCLX_TUNER_LOG(logFunction, level, message) \
  logLine((logFunction), (level), __FILE__, __LINE__, (message))

// Hand-written, zero-dependency CSV parser. ALWAYS compiled (fbcode and OSS).
// Skips blank lines and lines beginning with '#'. When ignoreErrors is false
// (strict, the default), any file-level or per-rule problem is logged at ERROR
// and returns ncclInvalidUsage so comm init fails. When ignoreErrors is true,
// a missing file logs a WARN and returns ncclSuccess (no override), and a bad
// rule logs an ERROR and is skipped while the remaining valid rules still load.
ncclResult_t loadConfigCsv(
    MetaTunerContext& context,
    const std::string& path,
    const bool ignoreErrors) {
  std::ifstream file(path);
  if (!file.is_open()) {
    NCCLX_TUNER_LOG(
        context.logFunction,
        ignoreErrors ? NCCL_LOG_WARN : NCCL_LOG_ERROR,
        fmt::format("NCCLX TUNER: config file '{}' could not be opened", path));
    return ignoreErrors ? ncclSuccess : ncclInvalidUsage;
  }

  std::string line;
  while (std::getline(file, line)) {
    const std::string_view trimmedLine = trim(line);
    if (trimmedLine.empty() || trimmedLine.front() == '#') {
      continue;
    }

    const std::vector<std::string_view> fields = splitCsvLine(trimmedLine);
    if (fields.size() < kMinCsvFields) {
      NCCLX_TUNER_LOG(
          context.logFunction,
          NCCL_LOG_ERROR,
          fmt::format(
              "NCCLX TUNER: {} malformed CSV line (too few columns) '{}'",
              ignoreErrors ? "skipping" : "rejecting",
              trimmedLine));
      if (!ignoreErrors) {
        return ncclInvalidUsage;
      }
      continue;
    }

    const std::optional<TuningConfig> config = buildConfig(fields);
    if (!config.has_value()) {
      NCCLX_TUNER_LOG(
          context.logFunction,
          NCCL_LOG_ERROR,
          fmt::format(
              "NCCLX TUNER: {} CSV rule with an invalid collective/algorithm/protocol, interval, or numeric value '{}'",
              ignoreErrors ? "skipping" : "rejecting",
              trimmedLine));
      if (!ignoreErrors) {
        return ncclInvalidUsage;
      }
      continue;
    }
    context.configs.push_back(*config);
  }

  NCCLX_TUNER_LOG(
      context.logFunction,
      NCCL_LOG_INFO,
      fmt::format(
          "NCCLX TUNER: loaded {} tuning rule(s) from '{}'",
          context.configs.size(),
          path));
  return ncclSuccess;
}

#ifdef NCCLX_TUNER_WITH_FOLLY_JSON
// JSON parser, only compiled when folly is available. Parses a top-level
// object with a "rules" array. Each rule is split into two nested objects:
//   "filter" -- match conditions: collective, bytesPerRank, nNodes,
//               nLocalRanks, numPipeOps, regBuff. Only "collective" is
//               required; an omitted filter field means wildcard/any
//               (bytesPerRank/nNodes/nLocalRanks -> *, numPipeOps/regBuff ->
//               -1).
//   "config" -- overrides applied on match: algorithm, protocol, channels,
//               chunkSize. "algorithm"/"protocol" are required; "channels" and
//               "chunkSize" are optional and omitting them means "no override"
//               (channels defaults to -1 = keep NCCL default, chunkSize to 0).
// The flat (un-nested) form is no longer accepted. The two objects are
// flattened back into the CSV column order so JSON and an equivalent CSV always
// parse to identical TuningConfig values.
ncclResult_t loadConfigJson(
    MetaTunerContext& context,
    const std::string& path,
    const bool ignoreErrors) {
  std::ifstream file(path);
  if (!file.is_open()) {
    NCCLX_TUNER_LOG(
        context.logFunction,
        ignoreErrors ? NCCL_LOG_WARN : NCCL_LOG_ERROR,
        fmt::format("NCCLX TUNER: config file '{}' could not be opened", path));
    return ignoreErrors ? ncclSuccess : ncclInvalidUsage;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();

  // Parse in one shot and copy-initialize `parsed` from the IIFE rather than
  // default-constructing then move-assigning: this keeps the dynamic's lifetime
  // unambiguous for static analysis (avoids a USE_AFTER_LIFETIME false positive
  // on the get_ptr below) and still distinguishes a parse error from a missing
  // "rules" array.
  const std::optional<folly::dynamic> parsed =
      [&]() -> std::optional<folly::dynamic> {
    try {
      return folly::parseJson(buffer.str());
    } catch (const std::exception& exception) {
      NCCLX_TUNER_LOG(
          context.logFunction,
          ignoreErrors ? NCCL_LOG_WARN : NCCL_LOG_ERROR,
          fmt::format(
              "NCCLX TUNER: failed to parse JSON '{}': {}",
              path,
              exception.what()));
      return std::nullopt;
    }
  }();
  if (!parsed.has_value()) {
    return ignoreErrors ? ncclSuccess : ncclInvalidUsage;
  }

  const auto* rules = parsed->get_ptr("rules");
  if (rules == nullptr || !rules->isArray()) {
    NCCLX_TUNER_LOG(
        context.logFunction,
        ignoreErrors ? NCCL_LOG_WARN : NCCL_LOG_ERROR,
        fmt::format("NCCLX TUNER: JSON '{}' has no 'rules' array", path));
    return ignoreErrors ? ncclSuccess : ncclInvalidUsage;
  }

  context.configs.reserve(context.configs.size() + rules->size());
  for (const auto& rule : *rules) {
    const std::optional<TuningConfig> config = parseJsonRule(rule);
    if (!config.has_value()) {
      NCCLX_TUNER_LOG(
          context.logFunction,
          NCCL_LOG_ERROR,
          fmt::format(
              "NCCLX TUNER: {} JSON rule (missing filter/config, unknown collective/algorithm/protocol, invalid interval/numeric, or bad value type): {}",
              ignoreErrors ? "skipping" : "rejecting",
              folly::toJson(rule)));
      if (!ignoreErrors) {
        return ncclInvalidUsage;
      }
      continue;
    }
    context.configs.push_back(*config);
  }

  NCCLX_TUNER_LOG(
      context.logFunction,
      NCCL_LOG_INFO,
      fmt::format(
          "NCCLX TUNER: loaded {} tuning rule(s) from JSON '{}'",
          context.configs.size(),
          path));
  return ncclSuccess;
}
#else
ncclResult_t loadConfigJson(
    MetaTunerContext& context,
    const std::string& path,
    const bool ignoreErrors) {
  NCCLX_TUNER_LOG(
      context.logFunction,
      ignoreErrors ? NCCL_LOG_WARN : NCCL_LOG_ERROR,
      fmt::format(
          "NCCLX TUNER: JSON config '{}' is unsupported in this build; use CSV",
          path));
  return ignoreErrors ? ncclSuccess : ncclInvalidUsage;
}
#endif

bool hasJsonExtension(const std::string& path) {
  constexpr std::string_view kJsonExt = ".json";
  if (path.size() < kJsonExt.size()) {
    return false;
  }
  return path.compare(
             path.size() - kJsonExt.size(), kJsonExt.size(), kJsonExt) == 0;
}

ncclResult_t loadConfig(
    MetaTunerContext& context,
    const std::string& path,
    const bool ignoreErrors) {
  if (hasJsonExtension(path)) {
    return loadConfigJson(context, path, ignoreErrors);
  }
  return loadConfigCsv(context, path, ignoreErrors);
}

// Returns true when every populated attribute of config matches the incoming
// collective. Fields set to -1 act as wildcards. The algorithm/protocol of the
// rule are NOT part of this match (they are the override target for
// getCollInfo); they ARE matched separately in getChunkSize.
inline bool matchesCollective(
    const TuningConfig& config,
    const ncclFunc_t collType,
    const size_t nBytes,
    const int numPipeOps,
    const int nNodes,
    const int nLocalRanks,
    const int regBuff,
    const ncclDebugLogger_t logFunction) {
  // bytesPerRank matches nBytes / nRanks, where nRanks = nNodes * nLocalRanks
  // (both from MetaTunerContext). Guard a zero/negative product so the division
  // is always safe. A wildcard bytesPerRank matches any size.
  int64_t nRanks =
      static_cast<int64_t>(nNodes) * static_cast<int64_t>(nLocalRanks);
  if (nRanks <= 0) {
    nRanks = 1;
  }
  const int64_t perRank = static_cast<int64_t>(nBytes) / nRanks;
  auto matched = config.collType == collType &&
      config.bytesPerRank.matches(perRank) && config.nNodes.matches(nNodes) &&
      config.nLocalRanks.matches(nLocalRanks) &&
      (config.numPipeOps == -1 || config.numPipeOps == numPipeOps) &&
      (config.regBuff == -1 || config.regBuff == regBuff);
  if (!matched && NCCLX_META_TUNER_LOG_MISMATCH) {
    NCCLX_TUNER_LOG(
        logFunction,
        NCCL_LOG_INFO,
        fmt::format(
            "NCCLX TUNER: rule did not match collective. rule: [{}]. actual: "
            "collType={}, nBytes={}, bytesPerRank={}, nNodes={}, "
            "nLocalRanks={}, numPipeOps={}, regBuff={}",
            config.toString(),
            static_cast<int>(collType),
            nBytes,
            perRank,
            nNodes,
            nLocalRanks,
            numPipeOps,
            regBuff));
  }
  return matched;
}

ncclResult_t metaTunerInit(
    void** ctx,
    uint64_t /* commId */,
    size_t nRanks,
    size_t nNodes,
    ncclDebugLogger_t logFunction,
    MetaNvlDomainInfo* /* nvlDomainInfo */,
    MetaTunerConstants* /* constants */) {
  auto context = std::make_unique<MetaTunerContext>();
  context->logFunction = logFunction;
  // nNodes / nLocalRanks are physical node and per-node rank counts, always
  // well within int range, so the size_t -> int conversion at this boundary is
  // safe.
  context->nNodes = static_cast<int>(nNodes);
  context->nLocalRanks =
      nNodes > 0 ? static_cast<int>(nRanks / nNodes) : static_cast<int>(nRanks);

  // An empty/unset config path simply disables the tuner (zero regression);
  // that is never an error. A SET-but-broken config fails comm init unless
  // NCCLX_TUNER_IGNORE_CONFIG_ERRORS is true (strict = ignore inverted).
  const std::string& path = NCCLX_TUNER_CONFIG_FILE;
  if (!path.empty()) {
    const bool ignoreErrors = NCCLX_TUNER_IGNORE_CONFIG_ERRORS;
    const ncclResult_t result = loadConfig(*context, path, ignoreErrors);
    if (result != ncclSuccess) {
      // Return before *ctx is set: the unique_ptr frees the context and the
      // NCCLCHECK on tuner->init in core init.cc fails communicator init.
      return result;
    }
  }

  NCCLX_TUNER_LOG(
      logFunction,
      NCCL_LOG_INFO,
      fmt::format(
          "NCCLX TUNER: initialized for {} node(s), {} rank(s), {} rule(s)",
          nNodes,
          nRanks,
          context->configs.size()));

  *ctx = context.release();
  return ncclSuccess;
}

ncclResult_t metaTunerGetCollInfo(
    void* ctx,
    ncclFunc_t collType,
    size_t nBytes,
    int numPipeOps,
    float** collCostTable,
    int numAlgo,
    int numProto,
    int regBuff,
    int* nChannels) {
  auto* context = static_cast<MetaTunerContext*>(ctx);
  if (context == nullptr) {
    return ncclInternalError;
  }

  auto* table = reinterpret_cast<float (*)[NCCL_NUM_PROTOCOLS]>(collCostTable);
  for (const auto& config : context->configs) {
    if (!matchesCollective(
            config,
            collType,
            nBytes,
            numPipeOps,
            context->nNodes,
            context->nLocalRanks,
            regBuff,
            context->logFunction)) {
      continue;
    }

    if (config.algorithm >= numAlgo || config.protocol >= numProto) {
      continue;
    }

    // Never force an algo/proto combo that core marked unsupported.
    if (table[config.algorithm][config.protocol] == NCCL_ALGO_PROTO_IGNORE) {
      continue;
    }

    table[config.algorithm][config.protocol] = 0.0F;
    if (config.nChannels != -1) {
      *nChannels = config.nChannels;
    }

    NCCLX_TUNER_LOG(
        context->logFunction,
        NCCL_LOG_INFO,
        fmt::format(
            "NCCLX TUNER: matched rule for collType={} nBytes={} (bytesPerRank={} nNodes={} nLocalRanks={}) -> algo={} proto={} channels={}",
            static_cast<int>(collType),
            nBytes,
            config.bytesPerRank.toString(),
            config.nNodes.toString(),
            config.nLocalRanks.toString(),
            config.algorithm,
            config.protocol,
            config.nChannels));
    // First matching rule wins (row order = priority).
    return ncclSuccess;
  }

  return ncclSuccess;
}

ncclResult_t metaTunerFinalize(void* ctx) {
  delete static_cast<MetaTunerContext*>(ctx);
  return ncclSuccess;
}

#ifdef NCCLX_TUNER_HAS_GETCHUNKSIZE
ncclResult_t metaTunerGetChunkSize(
    void* ctx,
    ncclFunc_t collType,
    size_t nBytes,
    int algo,
    int proto,
    int /* nChannels */,
    size_t* chunkSize) {
  auto* context = static_cast<MetaTunerContext*>(ctx);
  if (context == nullptr) {
    return ncclInternalError;
  }

  for (const auto& config : context->configs) {
    if (config.chunkSize == 0) {
      continue;
    }
    if (config.algorithm != algo || config.protocol != proto) {
      continue;
    }
    if (!matchesCollective(
            config,
            collType,
            nBytes,
            /* numPipeOps */ -1,
            context->nNodes,
            context->nLocalRanks,
            /* regBuff */ -1,
            context->logFunction)) {
      continue;
    }

    // Core clamps the result to bufferMaxChunkSize; we do not clamp here.
    *chunkSize = config.chunkSize;
    NCCLX_TUNER_LOG(
        context->logFunction,
        NCCL_LOG_INFO,
        fmt::format(
            "NCCLX TUNER: chunkSize override collType={} nBytes={} algo={} proto={} -> {} (clamped by core to bufferMaxChunkSize)",
            static_cast<int>(collType),
            nBytes,
            algo,
            proto,
            config.chunkSize));
    // First matching rule wins (row order = priority).
    return ncclSuccess;
  }

  return ncclSuccess;
}
#endif

} // namespace

bool metaTunerEnabled() {
  return !NCCLX_TUNER_CONFIG_FILE.empty();
}

const ncclTuner_t kMetaTuner = {
    /* .name = */ "NCCLXBuiltinTuner",
    /* .init = */ metaTunerInit,
    /* .getCollInfo = */ metaTunerGetCollInfo,
    /* .finalize = */ metaTunerFinalize,
#ifdef NCCLX_TUNER_HAS_GETCHUNKSIZE
    /* .getChunkSize = */ metaTunerGetChunkSize,
#endif
};

bool tryLoadMetaTuner(struct ncclComm* comm) {
  if (!metaTunerEnabled()) {
    return false;
  }
  comm->tuner = const_cast<ncclTuner_t*>(&kMetaTuner);
  INFO(
      NCCL_INIT | NCCL_TUNING,
      "NCCLX TUNER: using built-in tuner (NCCLX_TUNER_CONFIG_FILE set)");
  return true;
}

} // namespace ncclx::tuner
