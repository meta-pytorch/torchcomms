// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

// Sentinels shared between the stub libibverbs and the delegation test. Values
// only, no verb declarations: the stub writes its signatures out by hand and
// the test resolves them by hand, which is what keeps the check independent of
// IbverbxSymbols.def.

constexpr int kStubQueryDeviceRet = 4242;
constexpr int kStubQueryDeviceMaxQp = 0x5EED;
constexpr uint32_t kStubVendorId = 0xFACEu;

// The return is this plus the obj_type the caller passed, so the test can prove
// the second argument survived the forwarder.
constexpr int kStubInitObjRetBase = 7000;

constexpr int kStubQueryMlx5DeviceRet = 1717;
constexpr uint64_t kStubCompMask = 0xABCDu;
