// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace comms::prims {

// Single source of truth for the Data-Direct mode, shared by NIC discovery
// (GpuNicDiscovery) and the IB transport config
// (MultipeerIbTransportConfig::enableDataDirect). One knob drives two coupled
// decisions: which NIC candidates discovery surfaces (and tags isDataDirect),
// and whether registerBuffer takes the Data-Direct (BAR1 PCIe) registration
// path on a DD-tagged NIC. Mirrors NCCL_IB_DATA_DIRECT (0/1/2).
enum class DataDirectMode {
  // 0: never discover or use Data-Direct.
  Disabled = 0,
  // 1 (NCCL default): discovery REPLACES a DD-capable NIC with its Data-Direct
  // variant -- each physical NIC appears exactly once (non-DD NICs are
  // unaffected) -- and the transport registers MRs through the Data-Direct
  // (BAR1) path on it.
  Only = 1,
  // 2: discovery exposes BOTH the regular and the Data-Direct variant of a
  // DD-capable NIC -- the SAME physical NIC over its two PCIe routes (Grace
  // C2C and BAR1), not two different NICs. In NCCL this drives one NIC over
  // both rails for more aggregate NIC<->HBM concurrency (each rail is a
  // separate net device; the same buffer is registered on both). NOTE: this
  // transport does NOT exploit that -- it selects one variant per NIC slot
  // (the DD variant sorts first via getBestAffinityNics), so Both currently
  // behaves identically to Only. It is kept only for NCCL_IB_DATA_DIRECT
  // parity; true dual-rail would register each buffer on both routes and
  // spread QP groups across them.
  Both = 2,
};

} // namespace comms::prims
