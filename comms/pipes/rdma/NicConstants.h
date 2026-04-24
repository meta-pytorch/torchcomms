// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace comms::pipes {

/**
 * Maximum number of NICs (RDMA rails) per GPU supported by the multi-NIC
 * transport stack. One "NIC" corresponds to one ibverbs path: a mlx5_X
 * device + ibv_pd + QPs.
 *
 * Hardware mapping (per comms/tcp_devmem/HW.md):
 *   - H100 (Grand Teton): 1 NIC per GPU → numNics=1
 *   - GB200 (Catalina):   2 NICs per GPU (2 ConnectX-7 chips) → numNics=2
 *   - GB300 (Clemente):   2 NICs per GPU (1 dual-port ConnectX-8 exposed
 *                         as 2 mlx5_X) → numNics=2
 *
 * `kMaxNicsPerGpu` caps the static array sizes used in the IBGDA backend:
 *   - NetworkLKeys / NetworkRKeys (per-NIC key array wrappers)
 *   - IbgdaLocalBuffer.lkeys / IbgdaRemoteBuffer.rkeys
 *   - LocalBufferRegistration.lkeys / RemoteBufferRegistration.rkeys
 *   - P2pIbgdaTransportDevice.sinkLkeys_
 *   - IbgdaTransportExchInfoAll.nicInfo
 *   - MultipeerIbgdaTransport per-NIC IB resources (ibvCtxs_, ibvPds_, ...)
 *   - CachedMr.mrs
 *
 * Increase this constant only if a future platform supports > 2 NICs per
 * GPU. At that point, also audit the IBGDA wire format
 * (IbgdaTransportExchInfoAll size scales with kMaxNicsPerGpu ×
 * kMaxQpsPerPeer × kMaxRanksForAllGather).
 *
 * Lives in its own minimal header (no doca/ibverbs deps) so lightweight
 * device-side headers like IbgdaBuffer.h can include it without dragging
 * in NicDiscovery's heavyweight transitive deps. The C-side mirror for
 * the NCCLx ABI is `NCCLX_MAX_NICS_PER_GPU` in nccl.h, kept in lockstep
 * via static_assert at the bridge layer.
 */
constexpr int kMaxNicsPerGpu = 2;

} // namespace comms::pipes
