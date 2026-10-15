/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__)
#define HIPIPC_RELAY_EXPORT __attribute__((visibility("default")))
#else
#define HIPIPC_RELAY_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

HIPIPC_RELAY_EXPORT uint32_t hipipc_relay_abi_version(void);
HIPIPC_RELAY_EXPORT int hipipc_relay_device_info(
    int device,
    char* bus,
    size_t buslen,
    char* arch,
    size_t archlen);
HIPIPC_RELAY_EXPORT size_t hipipc_relay_descriptor_size(void);
HIPIPC_RELAY_EXPORT int hipipc_relay_create(
    int device,
    int rank,
    const int* helpers,
    int helper_count,
    void* input,
    void* output,
    size_t capacity,
    void** request);
HIPIPC_RELAY_EXPORT int
hipipc_relay_export(void* request, void* descriptor, size_t length);
HIPIPC_RELAY_EXPORT int
hipipc_relay_import(void* request, const void* peer_descriptor, size_t length);
HIPIPC_RELAY_EXPORT int hipipc_relay_probe(void* request, void* stream);
HIPIPC_RELAY_EXPORT int
hipipc_relay_execute(void* request, size_t bytes, void* stream);
HIPIPC_RELAY_EXPORT int hipipc_relay_drain(void* request, void* stream);
HIPIPC_RELAY_EXPORT int hipipc_relay_close_imports(void* request);
HIPIPC_RELAY_EXPORT int hipipc_relay_destroy(void* request);
HIPIPC_RELAY_EXPORT const char* hipipc_relay_last_error(void);

#ifdef __cplusplus
} // extern "C"
#endif

#undef HIPIPC_RELAY_EXPORT
