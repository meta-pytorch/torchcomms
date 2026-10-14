/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Shader-side 16-byte reads of peer mappings, the way the relay kernels read
// peer inputs. `addresses` is a device array of `count` pointers; each site's
// 16 bytes land in out[2*i], out[2*i+1]. Never synchronizes `stream`.
hipError_t hipipcRelayProbeRead(
    const unsigned char* const* addresses,
    int count,
    unsigned long long* out,
    hipStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif
