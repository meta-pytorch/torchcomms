/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <errno.h>
#include <stdlib.h>
#include <mutex>

#include "checks.h"
#include "debug.h"
#include "tuner.h"
#include "plugin.h"
#include "meta/tuner/MetaTuner.h"

extern ncclTuner_t* getNcclTuner_v2(void* lib);
extern ncclTuner_t* getNcclTuner_v3(void* lib);
extern ncclTuner_t* getNcclTuner_v4(void* lib);
extern ncclTuner_t* getNcclTuner_v5(void* lib);
extern ncclTuner_t* getNcclTuner_v6(void* lib);

static std::mutex tunerPluginMutex;
static int tunerPluginRefCount;
static void* tunerPluginLib = nullptr;
static ncclTuner_t* tunerSymbol = nullptr;

enum {
  tunerPluginLoadFailed = -1,
  tunerPluginLoadReady = 0,
  tunerPluginLoadSuccess = 1,
};

#define MAX_PLUGIN_LOAD 4

static int status = tunerPluginLoadReady;

ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm) {
  const char* tunerName;
  // Initialize to nullptr by default if plugin tuner cannot be loaded.
  comm->tuner = nullptr;
  // [META] The built-in CSV/JSON tuner takes precedence over NCCL_TUNER_PLUGIN
  // when NCCLX_TUNER_CONFIG_FILE is set; see meta/tuner/MetaTuner.h. Warn rather
  // than silently overriding, because the dlopen path below never runs and would
  // otherwise not even log that NCCL_TUNER_PLUGIN was seen.
  if (ncclx::tuner::tryLoadMetaTuner(comm)) {
    const char* overriddenPlugin = ncclGetEnv("NCCL_TUNER_PLUGIN");
    if (overriddenPlugin != nullptr && strcasecmp(overriddenPlugin, "none") != 0) {
      WARN("NCCLX_TUNER_CONFIG_FILE is set, so the built-in tuner is used and NCCL_TUNER_PLUGIN=%s is ignored. "
           "Unset NCCLX_TUNER_CONFIG_FILE to load the external plugin.",
           overriddenPlugin);
    }
    return ncclSuccess;
  }
  if (tunerPluginLoadFailed == status) {
    return ncclSuccess;
  }

  std::lock_guard<std::mutex> lock(tunerPluginMutex);
  if (tunerPluginLoadFailed == status) {
    goto exit;
  }

  if (tunerPluginLoadSuccess == status) {
    comm->tuner = tunerSymbol;
    ++tunerPluginRefCount;
    goto exit;
  }

  if ((tunerName = ncclGetEnv("NCCL_TUNER_PLUGIN")) != nullptr) {
    INFO(NCCL_ENV | NCCL_TUNING, "NCCL_TUNER_PLUGIN set by environment to %s", tunerName);
    if (strcasecmp(tunerName, "none") == 0) goto fail;
  }
  tunerPluginLib = ncclOpenTunerPluginLib(tunerName);
  if (nullptr == tunerPluginLib) {
    tunerPluginLib = ncclGetNetPluginLib(ncclPluginTypeTuner);
    if (nullptr == tunerPluginLib) {
      goto fail;
    }
    tunerName = nullptr;
  } else if (ncclPluginLibPaths[ncclPluginTypeTuner]) {
    tunerName = ncclPluginLibPaths[ncclPluginTypeTuner];
  }

  tunerSymbol = getNcclTuner_v6(tunerPluginLib);
  if (tunerSymbol == NULL) {
    tunerSymbol = getNcclTuner_v5(tunerPluginLib);
  }
  if (tunerSymbol == NULL) {
    tunerSymbol = getNcclTuner_v4(tunerPluginLib);
  }
  if (tunerSymbol == NULL) {
    tunerSymbol = getNcclTuner_v3(tunerPluginLib);
  }
  if (tunerSymbol == NULL) {
    tunerSymbol = getNcclTuner_v2(tunerPluginLib);
  }
  if (tunerSymbol == NULL) {
    if (tunerName) INFO(NCCL_INIT | NCCL_TUNING, "External tuner plugin %s is unsupported", tunerName);
    goto fail;
  }
  if (tunerName) INFO(NCCL_INIT | NCCL_TUNING, "Successfully loaded external tuner plugin %s", tunerName);

  comm->tuner = tunerSymbol;
  ++tunerPluginRefCount;
  status = tunerPluginLoadSuccess;
  comm->tunerPluginLoaded = 1;

exit:
  return ncclSuccess;
fail:
  if (tunerPluginLib) NCCLCHECK(ncclClosePluginLib(tunerPluginLib, ncclPluginTypeTuner));
  tunerPluginLib = nullptr;
  status = tunerPluginLoadFailed;
  goto exit;
}

ncclResult_t ncclTunerPluginUnload(struct ncclComm* comm) {
  std::lock_guard<std::mutex> lock(tunerPluginMutex);
  if (comm->tunerPluginLoaded && 0 == (--tunerPluginRefCount)) {
    INFO(NCCL_DESTROY | NCCL_TUNING, "TUNER/Plugin: Closing tuner: '%s'", tunerSymbol->name);
    NCCLCHECK(ncclClosePluginLib(tunerPluginLib, ncclPluginTypeTuner));
    tunerPluginLib = nullptr;
    tunerSymbol = nullptr;
    comm->tuner = nullptr;
    status = tunerPluginLoadReady;
    comm->tunerPluginLoaded = 0;
  }
  return ncclSuccess;
}
