// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string_view>

//
// This file defines Logging APIs modelled atop `folly/log.h`. Atop it intends
// to add some additional features like GPU index prefixing and conditional
// logging per sub-system
//
// Use this API specifically for logging in the context of GPU code. Otherwise,
// prefer using `folly/log.h` directly for performance reasons.
//

namespace ctran::logging {

// AKA Ctran file path. This needs to be changed in refactoring
constexpr std::string_view kCtranCategory = "comms.ncclx.v2_25.comms.ctran";
/**
 * buck2 copies header files from their original location in the source tree
 * and places them under buck-out/ with a path like
 * buck-out/v2/gen/cell/hash/dirA/dirB/__rule_name__/buck-headers/dirA/dirB/Foo.h
 * For nccl it is even worse -- for some weird reason, the buck-out path is
 * something like
 * buck-out/v2/gen/fbcode/hash/comms/ncclx/__nccl2.25-internal__/buck-headers/comms/ctran/...
 * So there is no way we can make it consistent with other ctran source files.
 * Currently, XLOG will strip the last part after buck-headers, so the category
 * would be comms.ctran..., which is what we are going to use for header files.
 */
constexpr std::string_view kCtranHeaderCategory = "comms.ctran";

/**
 * Initialize logging for Ctran. By default it only initializes once globlally
 * and no-op for future calls on the process.
 *
 * @param alwaysInit If true, always initialize logging, for testing purpose.
 */
void initCtranLogging(bool alwaysInit = false);

}; // namespace ctran::logging
