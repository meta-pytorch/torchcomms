// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// NIC Selection Configuration
// =============================================================================
//
// Compile-time NIC selection for pipes-gda. Exactly one NIC type must be
// defined via BUCK compiler_flags (e.g., -DNIC_MLX5 or -DNIC_BNXT).
//
// If no NIC is specified, NIC_MLX5 is used as the default for backward
// compatibility.
//
// NIC-specific vendor prefix, vendor ID, byte order, and WQE/CQ logic are
// provided by the NIC backend classes (Mlx5NicBackend, BnxtNicBackend, etc.)
// selected at compile time via NicSelector.h.
// =============================================================================

#pragma once

// Default to mlx5 if no NIC is specified
#if !defined(NIC_MLX5) && !defined(NIC_BNXT) && !defined(NIC_IONIC)
#define NIC_MLX5
#endif

// Validate: exactly one NIC must be selected
#if (defined(NIC_MLX5) + defined(NIC_BNXT) + defined(NIC_IONIC)) > 1
#error "Only one NIC type may be selected (NIC_MLX5, NIC_BNXT, NIC_IONIC)"
#endif
