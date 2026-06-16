// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace uniflow {

/// Returns the NUMA node for the page at `addr` via get_mempolicy(2), or
/// -1 if the node cannot be determined. Intended for host (DRAM) addresses
/// after pages have been faulted (e.g. post ibv_reg_mr for pinned MRs).
int detectHostNumaNode(const void* addr);

} // namespace uniflow
