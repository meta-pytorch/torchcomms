// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// NIC Backend Selector for pipes-gda
// =============================================================================
//
// This is the ONLY file in the project that uses #ifdef NIC_* to select the
// active NIC backend. All other files use the ActiveNicBackend type alias.
//
// To add a new NIC:
//   1. Create a new backend header (e.g., IonicNicBackend.h)
//   2. Add a new #elif branch below
//   3. Add the NIC_* define to NicConfig.h validation
//   4. Add a new BUCK target with the appropriate compiler_flags
// =============================================================================

#pragma once

#include "nic/NicConfig.h" // @manual

#if defined(NIC_BNXT)
#include "nic/BnxtNicBackend.h" // @manual
#elif defined(NIC_IONIC)
#include "nic/IonicNicBackend.h" // @manual
#else
#include "nic/Mlx5NicBackend.h" // @manual
#endif

namespace pipes_gda {

#if defined(NIC_BNXT)
using ActiveNicBackend = BnxtNicBackend;
#elif defined(NIC_IONIC)
using ActiveNicBackend = IonicNicBackend;
#else
using ActiveNicBackend = Mlx5NicBackend;
#endif

} // namespace pipes_gda
