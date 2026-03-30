// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <memory>

#include "comms/ctran/utils/ReporterType.h"
#include "comms/ctran/utils/IErrorReporter.h"

class CtranComm;

namespace ctran {

// Factory function type for creating custom error reporters.
// Callers (e.g., MCCL) can register a factory to inject their own reporter
// without ctran needing to depend on caller-specific libraries.
using ErrorReporterFactory =
    std::function<std::unique_ptr<IErrorReporter>(CtranComm*)>;

// Register a factory for the given reporter type. Must be called before
// ctranInit() so that the error reporter can be created during construction.
void registerErrorReporterFactory(
    ReporterType type,
    ErrorReporterFactory factory);

// Create an error reporter for the given type. Falls back to
// NcclxErrorReporter if no factory is registered for the type.
std::unique_ptr<IErrorReporter> createErrorReporter(
    ReporterType type,
    CtranComm* comm);

} // namespace ctran
