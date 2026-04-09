// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// Compile-time log level gate (spdlog best practice for HPC).
// Levels below this threshold compile to (void)0 — zero cost.
// Override at build time: -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE
#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#endif

#include <spdlog/spdlog.h>

namespace uniflow::logging {

// Returns the "uniflow" named async logger. Thread-safe, lazy-initialized.
// First call creates a non-blocking async logger (overrun_oldest policy)
// with a background thread pool. Subsequent calls return the cached pointer.
spdlog::logger* getLogger();

} // namespace uniflow::logging

// ============================================================================
// Logging macros — the ONLY way to log in uniflow code.
//
// These wrap spdlog's SPDLOG_LOGGER_* macros which provide:
//   1. Compile-time level gating via SPDLOG_ACTIVE_LEVEL
//   2. Source location (__FILE__, __LINE__) in log messages
//   3. Zero-cost when compiled out — no argument evaluation
//
// NEVER call getLogger()->info(...) directly — it bypasses compile-time gating.
// ============================================================================
#define UNIFLOW_LOG_TRACE(...) \
  SPDLOG_LOGGER_TRACE(::uniflow::logging::getLogger(), __VA_ARGS__)
#define UNIFLOW_LOG_DEBUG(...) \
  SPDLOG_LOGGER_DEBUG(::uniflow::logging::getLogger(), __VA_ARGS__)
#define UNIFLOW_LOG_INFO(...) \
  SPDLOG_LOGGER_INFO(::uniflow::logging::getLogger(), __VA_ARGS__)
#define UNIFLOW_LOG_WARN(...) \
  SPDLOG_LOGGER_WARN(::uniflow::logging::getLogger(), __VA_ARGS__)
#define UNIFLOW_LOG_ERROR(...) \
  SPDLOG_LOGGER_ERROR(::uniflow::logging::getLogger(), __VA_ARGS__)
#define UNIFLOW_LOG_CRITICAL(...) \
  SPDLOG_LOGGER_CRITICAL(::uniflow::logging::getLogger(), __VA_ARGS__)
