// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace comms::pipes {

// Maximum number of IBGDA NICs per GPU supported by the multi-NIC transport.
// One "NIC" corresponds to one ibverbs path: a mlx5_X device + ibv_pd + QPs.
//
// Hardware mapping (per comms/tcp_devmem/HW.md):
//   - H100 (Grand Teton): 1 NIC per GPU → numNics=1
//   - GB200 (Catalina):   2 NICs per GPU (2 ConnectX-7 chips) → numNics=2
//   - GB300 (Clemente):   2 NICs per GPU (1 dual-port ConnectX-8 exposed as
//                         2 mlx5_X) → numNics=2
//
// kMaxIbgdaNics caps the static array sizes used in:
//   - IbgdaLocalBuffer.lkeys[kMaxIbgdaNics]
//   - IbgdaRemoteBuffer.rkeys[kMaxIbgdaNics]
//   - LocalBufferRegistration.lkeys[kMaxIbgdaNics]
//   - RemoteBufferRegistration.rkeys[kMaxIbgdaNics]
//   - P2pIbgdaTransportDevice.sinkLkeys_[kMaxIbgdaNics]
//   - IbgdaTransportExchInfoAll.nicInfo[kMaxIbgdaNics]
//   - CachedMr.mrs[kMaxIbgdaNics]
//
// Increase this constant only if a future platform supports > 2 NICs per GPU.
// At that point, also audit the wire format (IbgdaTransportExchInfoAll size
// scales with kMaxIbgdaNics × kMaxQpsPerPeer × kMaxRanksForAllGather).
constexpr int kMaxIbgdaNics = 2;

} // namespace comms::pipes
