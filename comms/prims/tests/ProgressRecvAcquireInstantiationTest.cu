// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// The split acquire/release recv path reaches progress_recv_ready() through a
// call site that no NVIDIA-side target instantiated for every protocol: only
// ReduceScatterDirectIbV2.cu pulls it in, and only for protocol::Simple. When
// the progress_recv_ready() overloads gained channelLayout/chunk parameters,
// the acquire call site kept passing the old argument list and nothing in the
// package caught it until a downstream collective failed to compile.
//
// These explicit instantiations force the bodies to be compiled for every
// transport/protocol pair the seam claims to support, so signature drift fails
// here instead of in whichever collective happens to instantiate it next.

#include "comms/prims/tests/ProgressRecvAcquireInstantiationTest.cuh"

#include "comms/prims/transport/P2pIbTransportDevice.cuh"
#include "comms/prims/transport/P2pIbTransportProgressImpl.cuh"

namespace comms::prims::detail {

template __device__ IbgdaSendRecvProgressStatus
progress_recv_acquire_once<P2pIbrcTransportDevice, protocol::Simple>(
    P2pIbrcTransportDevice&,
    ThreadGroup&,
    std::size_t,
    std::size_t,
    const Timeout&,
    RecvChunkAcquisition&);

template __device__ IbgdaSendRecvProgressStatus
progress_recv_acquire_once<P2pIbrcTransportDevice, protocol::LL>(
    P2pIbrcTransportDevice&,
    ThreadGroup&,
    std::size_t,
    std::size_t,
    const Timeout&,
    RecvChunkAcquisition&);

template __device__ IbgdaSendRecvProgressStatus
progress_recv_acquire_once<P2pIbgdaTransportDevice, protocol::Simple>(
    P2pIbgdaTransportDevice&,
    ThreadGroup&,
    std::size_t,
    std::size_t,
    const Timeout&,
    RecvChunkAcquisition&);

template __device__ IbgdaSendRecvProgressStatus
progress_recv_acquire_once<P2pIbgdaTransportDevice, protocol::LL>(
    P2pIbgdaTransportDevice&,
    ThreadGroup&,
    std::size_t,
    std::size_t,
    const Timeout&,
    RecvChunkAcquisition&);

template __device__ void
progress_recv_release_once<P2pIbrcTransportDevice, protocol::Simple>(
    P2pIbrcTransportDevice&,
    ThreadGroup&,
    const RecvChunkAcquisition&);

template __device__ void
progress_recv_release_once<P2pIbrcTransportDevice, protocol::LL>(
    P2pIbrcTransportDevice&,
    ThreadGroup&,
    const RecvChunkAcquisition&);

template __device__ void
progress_recv_release_once<P2pIbgdaTransportDevice, protocol::Simple>(
    P2pIbgdaTransportDevice&,
    ThreadGroup&,
    const RecvChunkAcquisition&);

template __device__ void
progress_recv_release_once<P2pIbgdaTransportDevice, protocol::LL>(
    P2pIbgdaTransportDevice&,
    ThreadGroup&,
    const RecvChunkAcquisition&);

} // namespace comms::prims::detail

namespace comms::prims::test {

namespace {

// Covers the dispatching wrappers as well: the backend-agnostic facade and both
// concrete transports, which is how every collective actually reaches the seam.
__global__ void acquireReleaseInstantiationProbe(
    P2pIbTransportDevice transport,
    P2pIbrcTransportDevice ibrc,
    P2pIbgdaTransportDevice ibgda,
    std::size_t nbytes,
    std::size_t maxSignalBytes) {
  ThreadGroup group = make_block_group();
  const Timeout timeout;
  detail::RecvChunkAcquisition view{};

  if (transport.progress_recv_acquire_once(
          group, nbytes, maxSignalBytes, timeout, view) ==
      IbgdaSendRecvProgressStatus::Progressed) {
    transport.progress_recv_release_once(group, view);
  }
  if (ibrc.progress_recv_acquire_once(
          group, nbytes, maxSignalBytes, timeout, view) ==
      IbgdaSendRecvProgressStatus::Progressed) {
    ibrc.progress_recv_release_once(group, view);
  }
  if (ibgda.progress_recv_acquire_once(
          group, nbytes, maxSignalBytes, timeout, view) ==
      IbgdaSendRecvProgressStatus::Progressed) {
    ibgda.progress_recv_release_once(group, view);
  }
}

} // namespace

bool progress_recv_acquire_instantiations_linked() {
  return reinterpret_cast<const void*>(&acquireReleaseInstantiationProbe) !=
      nullptr;
}

} // namespace comms::prims::test
