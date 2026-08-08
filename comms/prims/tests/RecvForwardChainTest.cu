// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <cstdint>

#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims::test {

// Test-local accumulating CopyOp. Mirrors the reduce contract of MCCL's
// `IbReduceCopy` WITHOUT depending on MCCL (Prims must not depend on
// comms/mccl): `recv` accumulates staging into dst (`dst += staging`,
// byteOffset ignored — the transport advances `dst` per sub-chunk); `forward`
// writes `fwd_staging = staging + local_input[byteOffset]` and leaves `dst`
// unused (the transport advances staging/fwd_staging per sub-chunk but passes
// `local_input` un-advanced). Element type T. Used by the send-then-reduce-
// forward regression to exercise the resumable REDUCE forward path.
template <typename T>
struct AccumCopy {
  template <typename... Args>
  __device__ __forceinline__ static std::size_t send(
      char* staging,
      const char* src,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      Args...) {
    for (std::size_t i = group.thread_id_in_group; i < nbytes;
         i += group.group_size) {
      staging[i] = src[i];
    }
    return nbytes;
  }

  template <typename... Args>
  __device__ __forceinline__ static void recv(
      char* dst,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      Args...) {
    T* d = reinterpret_cast<T*>(dst);
    const T* s = reinterpret_cast<const T*>(staging);
    const std::size_t n = nbytes / sizeof(T);
    for (std::size_t i = group.thread_id_in_group; i < n;
         i += group.group_size) {
      d[i] += s[i];
    }
  }

  template <typename... Args>
  __device__ __forceinline__ static void forward(
      char* /*dst*/,
      char* fwd_staging,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t byte_offset,
      const char* local_input,
      Args...) {
    T* f = reinterpret_cast<T*>(fwd_staging);
    const T* s = reinterpret_cast<const T*>(staging);
    const T* l = reinterpret_cast<const T*>(local_input + byte_offset);
    const std::size_t n = nbytes / sizeof(T);
    for (std::size_t i = group.thread_id_in_group; i < n;
         i += group.group_size) {
      f[i] = s[i] + l[i];
    }
  }
};

// Chain kernel: rank 0 sends, intermediates recv_forward, last rank receives.
__global__ void recv_forward_chain_kernel(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    bool use_dst) {
  auto group = make_block_group();
  const auto num_blocks = gridDim.x;

  const std::size_t per_block = (nbytes / num_blocks) & ~15ULL;
  const std::size_t my_off = group.group_id * per_block;
  const std::size_t my_bytes =
      (group.group_id == num_blocks - 1) ? (nbytes - my_off) : per_block;

  const int prev_rank = (my_rank - 1 + world_size) % world_size;
  const int next_rank = (my_rank + 1) % world_size;

  if (my_rank == 0) {
    // First rank: send to next
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    next.send(group, send_buf + my_off, my_bytes);
  } else if (my_rank == world_size - 1) {
    // Last rank: receive from prev
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    prev.recv(group, recv_buf + my_off, my_bytes);
  } else {
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    char* dst = use_dst ? (recv_buf + my_off) : nullptr;
    prev.forward(group, dst, next, my_bytes);
  }
}

void launch_recv_forward_chain(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream) {
  recv_forward_chain_kernel<<<num_blocks, 128, 0, stream>>>(
      transports,
      send_buf,
      recv_buf,
      nbytes,
      my_rank,
      world_size,
      /*use_dst=*/true);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "recv_forward_chain kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
}

void launch_recv_forward_chain_no_dst(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream) {
  recv_forward_chain_kernel<<<num_blocks, 128, 0, stream>>>(
      transports,
      send_buf,
      recv_buf,
      nbytes,
      my_rank,
      world_size,
      /*use_dst=*/false);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "recv_forward_chain_no_dst kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
}

// Same chain topology as recv_forward_chain_kernel, but intermediates drive the
// RESUMABLE forward (init_forward_progress / progress_forward_once) to
// completion instead of the blocking forward. The endpoints stay blocking
// (send/recv): they interoperate with the resumable forward on the same
// group_id slots because the wire protocol (DATA_READY/SLOT_FREE/NIC_DONE per
// chunk) is identical. With per-block bytes > one pipeline slot the forward
// runs multiple chunks, exercising the fwd cursor advancing between puts.
__global__ void recv_forward_chain_progress_kernel(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    bool use_dst) {
  auto group = make_block_group();
  const auto num_blocks = gridDim.x;

  const std::size_t per_block = (nbytes / num_blocks) & ~15ULL;
  const std::size_t my_off = group.group_id * per_block;
  const std::size_t my_bytes =
      (group.group_id == num_blocks - 1) ? (nbytes - my_off) : per_block;

  const int prev_rank = (my_rank - 1 + world_size) % world_size;
  const int next_rank = (my_rank + 1) % world_size;

  if (my_rank == 0) {
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    next.send(group, send_buf + my_off, my_bytes);
  } else if (my_rank == world_size - 1) {
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    prev.recv(group, recv_buf + my_off, my_bytes);
  } else {
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    char* dst = use_dst ? (recv_buf + my_off) : nullptr;
    prev.init_forward_progress(group, next, my_bytes);
    IbgdaSendRecvProgressStatus st = IbgdaSendRecvProgressStatus::Waiting;
    do {
      st = prev.progress_forward_once(group, dst, next, my_bytes);
    } while (st != IbgdaSendRecvProgressStatus::Done);
  }
}

cudaError_t launch_recv_forward_chain_progress(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream) {
  recv_forward_chain_progress_kernel<<<num_blocks, 128, 0, stream>>>(
      transports,
      send_buf,
      recv_buf,
      nbytes,
      my_rank,
      world_size,
      /*use_dst=*/true);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "recv_forward_chain_progress kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
  // Propagate (not just print) the launch status: cudaGetLastError()
  // above already consumed it, so a caller's later cudaStreamSynchronize
  // would report success and the test would pass silently on a failed
  // launch.
  return err;
}

cudaError_t launch_recv_forward_chain_progress_no_dst(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream) {
  recv_forward_chain_progress_kernel<<<num_blocks, 128, 0, stream>>>(
      transports,
      send_buf,
      recv_buf,
      nbytes,
      my_rank,
      world_size,
      /*use_dst=*/false);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "recv_forward_chain_progress_no_dst kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
  // Propagate (not just print) the launch status: cudaGetLastError()
  // above already consumed it, so a caller's later cudaStreamSynchronize
  // would report success and the test would pass silently on a failed
  // launch.
  return err;
}

// Guards resumable-forward slot completion when a forward REUSES a send slot an
// ordinary resumable send just released: rank 1 issues a resumable SEND (op A)
// then a reduce-FORWARD (op B) to the SAME `next` transport, so op B's send
// reuses op A's send slot and crosses a pipeline-window boundary (nbytes == one
// window) -- the per-round "seed send then forward on the same slot" pattern
// the step-halved ring uses. rank 0 feeds op B's recv; rank 2 receives op A
// then op B. After the run: rank 2's recv_a == rank 1's send_buf; recv_b ==
// rank 0's send_buf + rank 1's local_buf (reduced). Requires world_size >= 3;
// single block (nbytes sized to one window by the caller).
__global__ void send_then_reduce_forward_kernel(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    const char* local_buf,
    char* recv_a,
    char* recv_b,
    std::size_t nbytes,
    int my_rank,
    int world_size) {
  auto group = make_block_group();
  const int prev_rank = (my_rank - 1 + world_size) % world_size;
  const int next_rank = (my_rank + 1) % world_size;

  if (my_rank == 0) {
    // Feed rank 1's reduce-forward recv (op B) with a blocking send.
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    next.send(group, send_buf, nbytes);
  } else if (my_rank == 1) {
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    // op A: ordinary resumable send (window 0) on next's send slot.
    next.init_send_progress(group, nbytes);
    IbgdaSendRecvProgressStatus st = IbgdaSendRecvProgressStatus::Waiting;
    do {
      st = next.progress_send_once(group, send_buf, nbytes);
    } while (st != IbgdaSendRecvProgressStatus::Done);
    // op B: reduce-forward (window 1) REUSING next's send slot -> crosses the
    // pipeline-window boundary.
    prev.init_forward_progress(group, next, nbytes);
    st = IbgdaSendRecvProgressStatus::Waiting;
    do {
      st = prev.progress_forward_once<AccumCopy<float>>(
          group,
          /*dst=*/nullptr,
          next,
          nbytes,
          /*max_signal_bytes=*/0,
          comms::prims::Timeout(),
          /*local_input=*/local_buf);
    } while (st != IbgdaSendRecvProgressStatus::Done);
  } else if (my_rank == 2) {
    // Receive op A then op B (blocking) on the same prev recv slot.
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    prev.recv(group, recv_a, nbytes);
    prev.recv(group, recv_b, nbytes);
  }
  // ranks >= 3: idle.
}

cudaError_t launch_send_then_reduce_forward(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    const char* local_buf,
    char* recv_a,
    char* recv_b,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream) {
  send_then_reduce_forward_kernel<<<num_blocks, 128, 0, stream>>>(
      transports,
      send_buf,
      local_buf,
      recv_a,
      recv_b,
      nbytes,
      my_rank,
      world_size);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "send_then_reduce_forward kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
  // Propagate (not just print) the launch status: cudaGetLastError()
  // above already consumed it, so a caller's later cudaStreamSynchronize
  // would report success and the test would pass silently on a failed
  // launch.
  return err;
}

} // namespace comms::prims::test
