// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// A stub libibverbs for the shim to forward INTO: the stand-in "real provider"
// behind IB_INJECTION_REAL_IBVERBS_SO / IB_INJECTION_REAL_MLX5_SO.
//
// Not to be confused with FakeProvider in IbInjectionEngineTest.cc, which is a
// different thing on the other side of the shim. That one is a linked,
// in-process C++ class handing out ibv_context / ibv_cq / ibv_qp objects with a
// mutable ops vtable, so the engine has something to register and the shims
// have something to patch. This one is a dlopen-able DSO exporting versioned C
// symbols, because what it stands in for is the library the forwarders call
// through to. Neither can do the other's job.
//
// HAND-WRITTEN ON PURPOSE. Everything else about the shim is driven by
// IbverbxSymbols.def -- the forwarders, the version script and the
// symbol-coverage expectations all come from that one table, which means a
// wrong entry produces a wrong forwarder AND a matching wrong expectation, and
// the symbol tests stay green. Writing these three signatures out by hand is
// what makes the delegation test an independent check rather than the table
// agreeing with itself.
//
// Each verb returns a distinct sentinel and echoes an argument back through an
// out-param, so the test can tell "the forwarder reached me" from "the
// forwarder reached me with the arguments the caller passed".
//
// Three verbs, one per lookup style the shim implements:
//   ibv_query_device     dlvsym on the ibv handle       (realSym)
//   mlx5dv_init_obj      dlvsym on the mlx5 handle      (realMlx5Sym w/
//   version) mlx5dv_query_device  plain dlsym on the mlx5 handle (realMlx5Sym
//   w/ nullptr)

#include <cstdint>

#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/ib_injection/tests/IbInjectionStubLibibverbs.h"

using ibverbx::ibv_context;
using ibverbx::ibv_device_attr;
using ibverbx::mlx5dv_context;
using ibverbx::mlx5dv_obj;

extern "C" {

// Writes the sentinel into a field the caller owns, so a forwarder that dropped
// or reordered its arguments cannot pass.
int ibv_query_device(ibv_context* context, ibv_device_attr* attr) {
  if (attr == nullptr) {
    return -1;
  }
  attr->max_qp = kStubQueryDeviceMaxQp;
  attr->vendor_id = (context == nullptr) ? 0u : kStubVendorId;
  return kStubQueryDeviceRet;
}

// Echoes obj_type into the return so the test sees the second argument arrive,
// not merely that something called through.
int mlx5dv_init_obj(mlx5dv_obj* obj, uint64_t obj_type) {
  if (obj == nullptr) {
    return -1;
  }
  return kStubInitObjRetBase + static_cast<int>(obj_type);
}

int mlx5dv_query_device(ibv_context* context, mlx5dv_context* attrs) {
  if (attrs == nullptr) {
    return -1;
  }
  attrs->comp_mask = (context == nullptr) ? 0u : kStubCompMask;
  return kStubQueryMlx5DeviceRet;
}

} // extern "C"
