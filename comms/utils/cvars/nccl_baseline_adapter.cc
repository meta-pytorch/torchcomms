// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>

#include <cstring>
#include <string>

#include <cuda_runtime.h>

#include <folly/String.h>
#include <folly/logging/xlog.h>

#include "comms/utils/cvars/nccl_baseline_adapter.h"
#include "comms/utils/cvars/nccl_cvars.h"

#define NCCL_ADAPTER_INFO(fmt, ...) \
  XLOGF(INFO, "NcclBaselineAdapter INFO CVAR: " fmt, __VA_ARGS__);

namespace nccl_baseline_adapter {
void ncclLoadParam(
    char const* env,
    int64_t deftVal,
    int64_t uninitialized,
    int64_t* cache) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  if (__atomic_load_n(cache, __ATOMIC_RELAXED) != uninitialized) {
    // If the value is already initialized, return immediately.
    return;
  }

  auto it_int64 = ncclx::env_int64_values.find(env);
  int64_t int64Value;
  if (it_int64 != ncclx::env_int64_values.end()) {
    int64Value = *it_int64->second;
    NCCL_ADAPTER_INFO(
        "{} set by int64_t CVAR map to {}.", env, (long long)int64Value);
    __atomic_store_n(cache, int64Value, __ATOMIC_RELAXED);
    return;
  }

  auto it_int = ncclx::env_int_values.find(env);
  int64_t intValue;
  if (it_int != ncclx::env_int_values.end()) {
    intValue = *it_int->second;
    NCCL_ADAPTER_INFO(
        "{} set by integer CVAR map to {}.", env, (long long)intValue);
    __atomic_store_n(cache, static_cast<int64_t>(intValue), __ATOMIC_RELAXED);
    return;
  }

  auto it_bool = ncclx::env_bool_values.find(env);
  bool boolValue;
  if (it_bool != ncclx::env_bool_values.end()) {
    boolValue = *it_bool->second;
    NCCL_ADAPTER_INFO(
        "{} set by bool CVAR map to {}.", env, (long long)boolValue);
    __atomic_store_n(cache, static_cast<int64_t>(boolValue), __ATOMIC_RELAXED);
    return;
  }

  // We first try to get the value from the string values map.
  auto it_str = ncclx::env_string_values.find(env);
  const char* strValue = nullptr;
  if (it_str == ncclx::env_string_values.end()) {
    // If the value is not found in the string values map either,
    // then we'll use the default value.
    NCCL_ADAPTER_INFO(
        "No value found for {} in either CVAR map, using default {}.",
        env,
        (long long)deftVal);
    __atomic_store_n(cache, deftVal, __ATOMIC_RELAXED);
    return;
  }

  const std::basic_string<char>* s = it_str->second;
  strValue = s->c_str();

  int64_t value = deftVal;
  if (strValue && strlen(strValue) > 0) {
    errno = 0;
    char* endptr;
    value = strtoll(strValue, &endptr, 0);
    if (errno || endptr == strValue) {
      value = deftVal;
      XLOG(WARN) << "NcclBaselineAdapter WARN CVAR: Invalid value \""
                 << strValue << "\" for CVAR " << env << ", using default \""
                 << deftVal << "\".";
    } else {
      NCCL_ADAPTER_INFO(
          "{} set by string CVAR map to {}.", env, (long long)value);
    }
  }

  __atomic_store_n(cache, value, __ATOMIC_RELAXED);
}

const char* ncclGetEnv(const char* name) {
  // Note: we omit the initEnv() call here (which is present in the baseline
  // NCCL implementation) because calling initEnv() breaks a large number of
  // unit tests. This is because the unit tests temporarily set values for
  // various CVARS, which are then overwritten by the initEnv() call.

  // First try to get the value from the string values map.
  auto it = ncclx::env_string_values.find(name);
  if (it != ncclx::env_string_values.end()) {
    if (it->second->empty()) {
      return nullptr;
    }
    return it->second->c_str();
  }

  // Use a static thread-local map to store converted values
  // to avoid converting the same numeric value multiple times.
  static thread_local std::unordered_map<std::string, std::string>
      converted_values;
  auto it_conv = converted_values.find(name);
  if (it_conv != converted_values.end()) {
    return it_conv->second.c_str();
  }

  // Try the int64_t map and convert to string.
  auto it_int64 = ncclx::env_int64_values.find(name);
  if (it_int64 != ncclx::env_int64_values.end()) {
    converted_values[name] = std::to_string(*it_int64->second);
    return converted_values[name].c_str();
  }

  // Try the bool map and convert to string.
  auto it_bool = ncclx::env_bool_values.find(name);
  if (it_bool != ncclx::env_bool_values.end()) {
    converted_values[name] = std::to_string(*it_bool->second);
    return converted_values[name].c_str();
  }

  // Try the int map and convert to string.
  auto it_int = ncclx::env_int_values.find(name);
  if (it_int != ncclx::env_int_values.end()) {
    converted_values[name] = std::to_string(*it_int->second);
    return converted_values[name].c_str();
  }

  throw std::runtime_error(
      "Undefined NCCL environment variable: \"" + std::string(name) + "\"");
}
} // namespace nccl_baseline_adapter
