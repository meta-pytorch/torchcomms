// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <gtest/gtest.h>
#include <chrono>
#include <optional>

#include <folly/init/Init.h>

#include "comms/common/CudaWrap.h"
#include "comms/common/fault_tolerance/Abort.h"
#include "comms/prims/collectives/SendRecvTile.cuh"
#include "comms/prims/collectives/SendRecvTileCompressed.cuh"
#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/transport/MultiPeerTransport.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::prims::tests {

namespace {

// Payload byte `i` of the buffer owned by `rank`.
//
// A SplitMix64-style finalizer over (rank, i), NOT `(rank * 7 + i) % 256`. The
// old pattern had period 256, and almost every tile and signal-chunk boundary
// in this suite is a multiple of 256 -- so replaying or swapping an aligned
// tile or chunk still satisfied a byte-by-byte oracle. Mixing the high index
// bits in means any misplacement of a whole tile now fails verification.
//
// `generation` is the launch counter within a test. Without it every round of a
// wrap-around test sends byte-identical data, so a slot that is recycled
// wrongly and replays the PREVIOUS round's contents still satisfies the oracle
// -- which is precisely the bug these tests exist to catch. Mixing it in means
// round N and round N+1 never agree on a single byte.
char pattern_byte(int rank, std::size_t i, int generation) {
  uint64_t x = static_cast<uint64_t>(rank) * 0x9e3779b97f4a7c15ULL +
      static_cast<uint64_t>(generation) * 0xd1b54a32d192ed03ULL + i;
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return static_cast<char>(x & 0xff);
}

std::vector<char> make_pattern(int rank, std::size_t bytes, int generation) {
  std::vector<char> v(bytes);
  for (std::size_t i = 0; i < bytes; i++) {
    v[i] = pattern_byte(rank, i, generation);
  }
  return v;
}

// One comparison over the whole buffer instead of a per-byte EXPECT_EQ loop:
// same coverage, one clear failure, and no millions of gtest assertions on a
// multi-megabyte payload. Reports the first differing offset itself.
void expect_pattern(
    const std::vector<char>& actual,
    int peer,
    int myRank,
    int generation) {
  const std::vector<char> expected =
      make_pattern(peer, actual.size(), generation);
  if (actual == expected) {
    return;
  }
  std::size_t at = 0;
  while (at < actual.size() && actual[at] == expected[at]) {
    ++at;
  }
  ADD_FAILURE() << "Rank " << myRank << ": payload from peer " << peer
                << " (generation " << generation << ") differs at byte " << at
                << " of " << actual.size() << " (expected "
                << static_cast<int>(expected[at]) << ", got "
                << static_cast<int>(actual[at]) << ")";
}

} // namespace

class SendRecvTileTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
    // Every case here needs at least one clean pair; `partner()` is
    // `globalRank ^ 1`, which only pairs for an even rank count. Guarding once
    // here rather than repeating the same prologue in every test body. The
    // ring cases need >= 3 and check that themselves.
    if (skip_unless_paired()) {
      GTEST_SKIP() << "Requires an even rank count >= 2 (got " << numRanks
                   << ")";
    }
  }

  // ---- shared plumbing -----------------------------------------------------
  // The four run_* helpers below differ in geometry and which kernel they
  // launch; everything else used to be copy-pasted between them, which is how
  // the compressed helper ended up without a `max_signal_bytes` knob the plain
  // one had. Keep the differences in the helpers and the sameness here.

  // 30s is well clear of the slowest case (the 8-iteration wrap-around runs)
  // while still failing fast on a genuine hang instead of hitting the harness
  // timeout with no diagnostic.
  //
  // A `make_abort()` factory is not possible -- `Abort` is neither copyable nor
  // movable, so it has to be constructed in the caller's scope. Sharing the
  // timeout is the part that actually mattered: it was the same magic 30000
  // written out in five places.
  static constexpr std::chrono::milliseconds kAbortTimeout{30000};

  // A cluster launch needs the grid to be a multiple of the cluster dimension,
  // and num_blocks need not be (IbLarge uses 14), so fall back to a plain
  // launch rather than letting cudaLaunchKernelExC reject the grid. Checking
  // the launch status matters as much as the fallback: discarding it turned
  // that rejection into a silently all-zero output buffer.
  static void launch_and_sync(
      const void* kernel,
      int num_blocks,
      void** kernel_args,
      bool allow_cluster = true) {
    std::optional<dim3> clus;
    if (allow_cluster && num_blocks % comms::common::kDefaultClusterSize == 0) {
      clus = dim3(comms::common::kDefaultClusterSize, 1, 1);
    }
    CUDACHECK_TEST(
        comms::common::launchKernel(
            const_cast<void*>(kernel),
            dim3(num_blocks),
            dim3(512),
            kernel_args,
            nullptr,
            clus));
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  // Copy a device buffer back and compare it against `peer`'s pattern.
  void expect_device_pattern(
      const DeviceBuffer& buf,
      std::size_t bytes,
      int peer,
      int generation) {
    std::vector<char> host(bytes);
    CUDACHECK_TEST(
        cudaMemcpy(host.data(), buf.get(), bytes, cudaMemcpyDeviceToHost));
    expect_pattern(host, peer, globalRank, generation);
  }

  // Monotonic across EVERY launch in a test, including across a
  // plain -> compressed -> plain sequence on one transport, so no two rounds
  // anywhere in a test ever put the same bytes on the wire.
  int launch_generation_ = 0;

  // The compressed kernel symbol is private now; the supported launch shapes
  // are reachable only through `pick_sendrecv_tile_compressed_kernel()`. Going
  // through the picker here is deliberate -- it is the API consumers have, so
  // the suite should exercise it rather than a back door. blockDim.x = 512
  // (NumWarps = 16) at MinBlocksPerSM = 2.
  void* compressed_kernel() {
    void* k = pick_sendrecv_tile_compressed_kernel(
        /*threads_per_block=*/512, /*min_blocks_per_sm=*/2);
    EXPECT_NE(k, nullptr)
        << "pick_sendrecv_tile_compressed_kernel(512, 2) returned nullptr; "
           "that pair must be instantiated in SendRecvTileCompressed.cu";
    return k;
  }

  // Ranks are paired (0<->1, 2<->3, ...). Even ranks send to their odd
  // partner; odd ranks receive. Returns the partner rank.
  //
  // `globalRank ^ 1` only pairs cleanly for an even rank count: with an odd
  // one the highest rank would compute a partner equal to `numRanks`, which is
  // not a valid rank and would be handed to `get_device_handle()` as a peer.
  // The BUCK config runs ppn=8 so this is latent, but the tests guard on it
  // (see `skip_unless_paired`) rather than leaving the assumption implicit.
  int partner() const {
    return globalRank ^ 1;
  }

  // Every test needs at least one pair, and the pairing above needs an even
  // rank count. One place to express both.
  bool skip_unless_paired() const {
    return numRanks < 2 || (numRanks % 2) != 0;
  }
  bool is_sender() const {
    return (globalRank % 2) == 0;
  }

  std::unique_ptr<MultiPeerTransport> create_and_exchange(
      bool ib_only,
      std::size_t data_buffer_size = 8 * 1024 * 1024,
      IbBackendMode ib_mode = IbBackendMode::kIbgda) {
    MultiPeerTransportConfig config{
        .nvlConfig =
            {
                .pipelineDepth = 4,
                .maxNumChannels = 32,
                .perChannelSize = data_buffer_size / 32,
            },
        .ibConfig =
            {
                .cudaDevice = localRank,
                // perBlockSlot = perChannelSize / pipelineDepth must hold at
                // least one worst-case ANS chunk, i.e. the padded compressed
                // size of kAnsMaxUncompBytes (256 KiB), which is slightly
                // larger than 256 KiB. Below that,
                // max_safe_chunk_size_for_slot() finds room for zero chunks,
                // falls back to returning MaxUncompBytes, and the transport
                // traps on chunkStride > perBlockSlot — a __trap() whose
                // printf is swallowed, so it surfaces only as a bare
                // "unspecified launch failure". 1 MiB gives a 512 KiB slot.
                .perChannelSize = 1024 * 1024,
                .max_num_channels = 32,
                .pipelineDepth = 2,
                .maxGroups = 32,
            },
        .ibMode = ib_mode,
        .topoConfig =
            {
                .p2pDisable = ib_only,
            },
    };
    auto bootstrap = std::make_shared<MpiBootstrap>();
    auto transport = std::make_unique<MultiPeerTransport>(
        globalRank, numRanks, localRank, bootstrap, config);
    transport->exchange();
    return transport;
  }

  void run_sendrecv_tile_test(
      std::size_t bytes,
      int num_blocks,
      bool ib_only = false,
      std::size_t max_signal_bytes = 0,
      IbBackendMode ib_mode = IbBackendMode::kIbgda,
      // Back-to-back launches on ONE transport. >1 exercises the state that
      // persists across invocations and that a single launch can never reach:
      // the per-channel staging cursors wrap the pipeline ring, so SLOT_FREE
      // credit return and local NIC-completion retirement actually run. With
      // perChannelSize/pipelineDepth = 1 MiB / 2 there are two slots per
      // channel, and each launch advances a block by bytes/num_blocks, so a
      // handful of iterations is enough to wrap. Mirrors `iterations` in
      // AllToAllvTileTest.
      int iterations = 1,
      // When non-null, run on the CALLER's transport instead of building one.
      // The alternating wrap test needs plain and compressed launches to share
      // a single handle -- with a transport each they exercise three
      // independent sets of staging cursors and prove nothing about one
      // protocol inheriting the other's state.
      MultiPeerTransport* shared_transport = nullptr) {
    std::unique_ptr<MultiPeerTransport> owned_transport;
    if (shared_transport == nullptr) {
      owned_transport = create_and_exchange(ib_only, 8 * 1024 * 1024, ib_mode);
      shared_transport = owned_transport.get();
    }
    auto handle = shared_transport->get_device_handle({partner()});

    const int peer = partner();
    const bool send = is_sender();

    DeviceBuffer buf(bytes);

    // Filled per ROUND below rather than once here: each round uses a fresh
    // generation, so a recycled slot replaying the previous round is caught.
    CUDACHECK_TEST(cudaMemset(buf.get(), 0, bytes));

    SendRecvTileArgs args{
        .handle = handle,
        .is_send = send,
        .is_recv = !send,
        .send_peer = peer,
        .recv_peer = peer,
        .send_data = send ? static_cast<char*>(buf.get()) : nullptr,
        .send_count = send ? bytes : 0,
        .recv_data = send ? nullptr : static_cast<char*>(buf.get()),
        .recv_count = send ? 0 : bytes,
        .max_signal_bytes = max_signal_bytes,
    };

    comms::fault_tolerance::Abort abort{/*enabled=*/true};
    abort.setDefaultTimeout(kAbortTimeout);
    AbortDevice abortDevice = abort.getDeviceHandle();

    MPI_Barrier(MPI_COMM_WORLD);

    // A cluster launch requires the grid to be a multiple of the cluster
    // dimension, and num_blocks need not be (IbLarge uses 14). Fall back to a
    // plain launch rather than letting cudaLaunchKernelExC reject the grid.
    // Checking the launch status matters as much as the fallback: discarding
    // it turned that rejection into a silently all-zero output buffer.
    void* kernel_args[] = {&args, &abortDevice};
    for (int iter = 0; iter < iterations; iter++) {
      const int generation = launch_generation_++;
      if (send) {
        const std::vector<char> h_buf =
            make_pattern(globalRank, bytes, generation);
        CUDACHECK_TEST(
            cudaMemcpy(buf.get(), h_buf.data(), bytes, cudaMemcpyHostToDevice));
      } else {
        // Re-zero the destination each round so a later iteration cannot pass
        // on bytes an earlier one left behind.
        CUDACHECK_TEST(cudaMemset(buf.get(), 0, bytes));
      }
      MPI_Barrier(MPI_COMM_WORLD);
      launch_and_sync((void*)sendrecv_tile_kernel, num_blocks, kernel_args);

      MPI_Barrier(MPI_COMM_WORLD);

      if (!send) {
        SCOPED_TRACE("iteration " + std::to_string(iter));
        expect_device_pattern(buf, bytes, peer, generation);
      }
    }
  }

  // Bidirectional: every rank simultaneously sends to and receives from
  // its partner. The grid is split in half inside the kernel (send half +
  // recv half). `num_blocks` must be even.
  void run_sendrecv_tile_twoway_test(
      std::size_t bytes,
      int num_blocks,
      bool ib_only = false,
      std::size_t max_signal_bytes = 0,
      IbBackendMode ib_mode = IbBackendMode::kIbgda,
      bool asymmetric_zero_count = false) {
    // The kernel splits the grid in half by role, so an odd count would give
    // this rank a different per-direction active-block count from its peer.
    // Per the cross-rank contract that is not a clean failure -- it is dropped
    // data or a hang -- so fail fast here instead.
    ASSERT_EQ(num_blocks % 2, 0) << "bidirectional launches need an even grid";

    auto transport = create_and_exchange(ib_only, 8 * 1024 * 1024, ib_mode);
    auto handle = transport->get_device_handle({partner()});

    const int peer = partner();

    DeviceBuffer send_buf(bytes);
    DeviceBuffer recv_buf(bytes);

    // Send buffer carries a pattern keyed on this rank; recv buffer is
    // zeroed and must end up matching the peer's send pattern.
    const int generation = launch_generation_++;
    const std::vector<char> h_send =
        make_pattern(globalRank, bytes, generation);
    CUDACHECK_TEST(cudaMemcpy(
        send_buf.get(), h_send.data(), bytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recv_buf.get(), 0, bytes));

    SendRecvTileArgs args{
        .handle = handle,
        // Both flags stay true even when a count is zeroed: that is the whole
        // point -- the grid split must follow the flags, not the counts.
        .is_send = true,
        .is_recv = true,
        .send_peer = peer,
        .recv_peer = peer,
        .send_data = static_cast<char*>(send_buf.get()),
        .send_count = (asymmetric_zero_count && is_sender()) ? 0 : bytes,
        .recv_data = static_cast<char*>(recv_buf.get()),
        .recv_count = (asymmetric_zero_count && !is_sender()) ? 0 : bytes,
        .max_signal_bytes = max_signal_bytes,
    };

    comms::fault_tolerance::Abort abort{/*enabled=*/true};
    abort.setDefaultTimeout(kAbortTimeout);
    AbortDevice abortDevice = abort.getDeviceHandle();

    MPI_Barrier(MPI_COMM_WORLD);

    // A cluster launch requires the grid to be a multiple of the cluster
    // dimension, and num_blocks need not be (IbLarge uses 14). Fall back to a
    // plain launch rather than letting cudaLaunchKernelExC reject the grid.
    // Checking the launch status matters as much as the fallback: discarding
    // it turned that rejection into a silently all-zero output buffer.
    void* kernel_args[] = {&args, &abortDevice};
    launch_and_sync((void*)sendrecv_tile_kernel, num_blocks, kernel_args);

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<char> h_recv(bytes);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(), recv_buf.get(), bytes, cudaMemcpyDeviceToHost));
    if (asymmetric_zero_count && !is_sender()) {
      // Nothing was sent to us; the buffer must be untouched, which also
      // catches a rank that ran the recv direction anyway.
      EXPECT_EQ(h_recv, std::vector<char>(bytes, 0))
          << "recv_count == 0 but the receive buffer was written";
    } else {
      expect_pattern(h_recv, peer, globalRank, generation);
    }
  }

  // Compressed mixed-mode one-way: launches the ANS-compressed kernel with
  // `plain_block_fraction` so a subset of blocks fall back to plain Memcpy
  // while the rest ANS-compress, then checks the received buffer round-trips
  // byte-exact. IB-only (compression applies only to the IBGDA path). The
  // staging buffer (8 MiB) / active_blocks must be 512-aligned so plain and
  // ANS blocks agree on their per-block slot; power-of-two `num_blocks` that
  // divide 8 MiB into >=512-byte slots satisfy this.

  // Bytes rank `r` sends to rank `r + 1` on the ring. Deliberately different
  // per rank, so a flow's send_count and its recv_count are never the same
  // number -- which is the whole point of this helper.
  static std::size_t ring_flow_bytes(int r) {
    return static_cast<std::size_t>(r + 1) * 256 * 1024;
  }

  // 3+ rank ring: rank i sends to i+1 and receives from i-1, with DIFFERENT
  // byte counts in each direction.
  //
  // Every other case in this file is pairwise (`partner() == globalRank ^ 1`)
  // with both directions using the same peer and the same count, so a receive
  // path that mistakenly read `send_peer` or `send_count` reads an identical
  // value and still passes. Here `send_peer != recv_peer` and
  // `send_count != recv_count` for every rank, so each field has to be used
  // for what it names.
  //
  // Counts stay consistent per flow -- rank i's `send_count` is
  // `ring_flow_bytes(i)` and its `recv_count` is `ring_flow_bytes(i - 1)`,
  // which is exactly what the upstream neighbour sends -- so the cross-rank
  // contract holds while the two directions differ locally.
  void run_sendrecv_tile_ring_test(int num_blocks, bool ib_only = true) {
    ASSERT_EQ(num_blocks % 2, 0) << "bidirectional launches need an even grid";

    const int n = numRanks;
    const int send_peer = (globalRank + 1) % n;
    const int recv_peer = (globalRank - 1 + n) % n;
    const std::size_t send_bytes = ring_flow_bytes(globalRank);
    const std::size_t recv_bytes = ring_flow_bytes(recv_peer);
    ASSERT_NE(send_bytes, recv_bytes)
        << "the ring must give this rank different counts per direction";
    ASSERT_NE(send_peer, recv_peer) << "ring needs >= 3 ranks";

    auto transport = create_and_exchange(ib_only);
    auto handle = transport->get_device_handle({send_peer, recv_peer});

    DeviceBuffer send_buf(send_bytes);
    DeviceBuffer recv_buf(recv_bytes);
    const int generation = launch_generation_++;
    const std::vector<char> h_send =
        make_pattern(globalRank, send_bytes, generation);
    CUDACHECK_TEST(cudaMemcpy(
        send_buf.get(), h_send.data(), send_bytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recv_buf.get(), 0, recv_bytes));

    SendRecvTileArgs args{
        .handle = handle,
        .is_send = true,
        .is_recv = true,
        .send_peer = send_peer,
        .recv_peer = recv_peer,
        .send_data = static_cast<char*>(send_buf.get()),
        .send_count = send_bytes,
        .recv_data = static_cast<char*>(recv_buf.get()),
        .recv_count = recv_bytes,
        .max_signal_bytes = 0,
    };

    comms::fault_tolerance::Abort abort{/*enabled=*/true};
    abort.setDefaultTimeout(kAbortTimeout);
    AbortDevice abortDevice = abort.getDeviceHandle();

    MPI_Barrier(MPI_COMM_WORLD);

    void* kernel_args[] = {&args, &abortDevice};
    launch_and_sync((void*)sendrecv_tile_kernel, num_blocks, kernel_args);

    MPI_Barrier(MPI_COMM_WORLD);

    // Must match the UPSTREAM neighbour's pattern at the UPSTREAM neighbour's
    // byte count -- reading `send_peer`/`send_count` here would give the wrong
    // rank and the wrong length.
    expect_device_pattern(recv_buf, recv_bytes, recv_peer, generation);
  }

  void run_sendrecv_tile_compressed_test(
      std::size_t bytes,
      int num_blocks,
      float plain_block_fraction,
      std::size_t max_signal_bytes = 0,
      // See the note on the plain helper: >1 wraps the pipeline ring so slot
      // recycling actually runs. Also the only way the ANS stats counters get
      // read across more than one invocation.
      int iterations = 1,
      // See the plain helper: lets the alternating wrap test drive both
      // protocols over one transport.
      MultiPeerTransport* shared_transport = nullptr) {
    std::unique_ptr<MultiPeerTransport> owned_transport;
    if (shared_transport == nullptr) {
      owned_transport = create_and_exchange(/*ib_only=*/true);
      shared_transport = owned_transport.get();
    }
    auto handle = shared_transport->get_device_handle({partner()});

    const int peer = partner();
    const bool send = is_sender();

    DeviceBuffer buf(bytes);
    // Per-block 16-byte-aligned scratch for the compressed send path (keyed
    // on global blockIdx.x); sized one max-uncompressed-chunk slice per block,
    // derived from the compressor rather than a hard-coded constant.
    constexpr std::size_t kAnsChunkBytes = AnsCompressor::kMaxUncompBytes;
    DeviceBuffer aligned_aux_buf(
        static_cast<std::size_t>(num_blocks) * kAnsChunkBytes);

    // Filled per ROUND below; see the plain helper.
    CUDACHECK_TEST(cudaMemset(buf.get(), 0, bytes));

    SendRecvTileArgs args{
        .handle = handle,
        .is_send = send,
        .is_recv = !send,
        .send_peer = peer,
        .recv_peer = peer,
        .send_data = send ? static_cast<char*>(buf.get()) : nullptr,
        .send_count = send ? bytes : 0,
        .recv_data = send ? nullptr : static_cast<char*>(buf.get()),
        .recv_count = send ? 0 : bytes,
        .max_signal_bytes = max_signal_bytes,
        .aligned_aux_buf = static_cast<char*>(aligned_aux_buf.get()),
        .plain_block_fraction = plain_block_fraction,
    };

    comms::fault_tolerance::Abort abort{/*enabled=*/true};
    abort.setDefaultTimeout(kAbortTimeout);
    AbortDevice abortDevice = abort.getDeviceHandle();

    // Clear the running ANS counters so what we read after the launch belongs
    // to this launch only.
    (void)fetch_and_reset_sendrecv_ans_compress_stats(nullptr);

    MPI_Barrier(MPI_COMM_WORLD);

    // blockDim.x = 512 -> NumWarps = 16 (an instantiated compressed kernel).
    void* kernel_args[] = {&args, &abortDevice};
    int last_generation = 0;
    for (int iter = 0; iter < iterations; iter++) {
      const int generation = launch_generation_++;
      last_generation = generation;
      if (send) {
        const std::vector<char> h_buf =
            make_pattern(globalRank, bytes, generation);
        CUDACHECK_TEST(
            cudaMemcpy(buf.get(), h_buf.data(), bytes, cudaMemcpyHostToDevice));
      } else {
        CUDACHECK_TEST(cudaMemset(buf.get(), 0, bytes));
      }
      MPI_Barrier(MPI_COMM_WORLD);
      launch_and_sync(
          compressed_kernel(),
          num_blocks,
          kernel_args,
          /*allow_cluster=*/false);
      MPI_Barrier(MPI_COMM_WORLD);
      if (!send && iter + 1 < iterations) {
        // Verify every round, not just the last: a slot-recycling bug that
        // corrupts round N would otherwise be overwritten by round N+1.
        SCOPED_TRACE("iteration " + std::to_string(iter));
        expect_device_pattern(buf, bytes, peer, generation);
      }
    }

    // A byte-exact round trip does NOT prove the compressed path ran: the
    // plain Memcpy fallback is byte-exact too, so a regression that routed
    // every block through it (an activation-threshold or plain_block_fraction
    // bug) would leave these tests green while the coverage they exist for was
    // gone. Check the counters as well.
    const SendRecvAnsCompressStats stats =
        fetch_and_reset_sendrecv_ans_compress_stats(nullptr);
    if (send && plain_block_fraction < 1.0f) {
      EXPECT_GT(stats.uncompressed_bytes, 0ULL)
          << "no bytes went through the ANS compressor -- the compressed path "
             "did not run (plain_block_fraction="
          << plain_block_fraction << ")";
      EXPECT_GT(stats.compressed_bytes, 0ULL);
      // The payload is a hash, i.e. incompressible, so do not assert a ratio;
      // assert only that the compressor saw the data it was supposed to.
    }
    if (send && plain_block_fraction >= 1.0f) {
      EXPECT_EQ(stats.uncompressed_bytes, 0ULL)
          << "plain_block_fraction=1 must bypass the compressor entirely";
    }

    if (!send) {
      expect_device_pattern(buf, bytes, peer, last_generation);
    }
  }

  // Compressed mixed-mode two-way: every rank simultaneously sends to and
  // receives from its partner via the ANS-compressed kernel, with a fraction
  // of each direction's blocks falling back to plain Memcpy. The grid is
  // split half-send / half-recv, so per-direction active blocks =
  // num_blocks/2; the same `plain_block_fraction` on both peers keeps a
  // plain-sent tile plain-received.
  void run_sendrecv_tile_compressed_twoway_test(
      std::size_t bytes,
      int num_blocks,
      float plain_block_fraction) {
    // The kernel splits the grid in half by role, so an odd count would give
    // this rank a different per-direction active-block count from its peer.
    // Per the cross-rank contract that is not a clean failure -- it is dropped
    // data or a hang -- so fail fast here instead.
    ASSERT_EQ(num_blocks % 2, 0) << "bidirectional launches need an even grid";

    auto transport = create_and_exchange(/*ib_only=*/true);
    auto handle = transport->get_device_handle({partner()});

    const int peer = partner();

    DeviceBuffer send_buf(bytes);
    DeviceBuffer recv_buf(bytes);
    constexpr std::size_t kAnsChunkBytes = AnsCompressor::kMaxUncompBytes;
    DeviceBuffer aligned_aux_buf(
        static_cast<std::size_t>(num_blocks) * kAnsChunkBytes);

    const int generation = launch_generation_++;
    const std::vector<char> h_send =
        make_pattern(globalRank, bytes, generation);
    CUDACHECK_TEST(cudaMemcpy(
        send_buf.get(), h_send.data(), bytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recv_buf.get(), 0, bytes));

    SendRecvTileArgs args{
        .handle = handle,
        .is_send = true,
        .is_recv = true,
        .send_peer = peer,
        .recv_peer = peer,
        .send_data = static_cast<char*>(send_buf.get()),
        .send_count = bytes,
        .recv_data = static_cast<char*>(recv_buf.get()),
        .recv_count = bytes,
        .max_signal_bytes = 0,
        .aligned_aux_buf = static_cast<char*>(aligned_aux_buf.get()),
        .plain_block_fraction = plain_block_fraction,
    };

    comms::fault_tolerance::Abort abort{/*enabled=*/true};
    abort.setDefaultTimeout(kAbortTimeout);
    AbortDevice abortDevice = abort.getDeviceHandle();

    MPI_Barrier(MPI_COMM_WORLD);

    void* kernel_args[] = {&args, &abortDevice};
    launch_and_sync(
        compressed_kernel(),
        num_blocks,
        kernel_args,
        /*allow_cluster=*/false);

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<char> h_recv(bytes);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(), recv_buf.get(), bytes, cudaMemcpyDeviceToHost));
    expect_pattern(h_recv, peer, globalRank, generation);
  }
};

// ============================================================================
// NVL tests
// ============================================================================

TEST_F(SendRecvTileTestFixture, NvlSmall) {
  run_sendrecv_tile_test(4096, 4);
}

TEST_F(SendRecvTileTestFixture, NvlMedium) {
  run_sendrecv_tile_test(256 * 1024, 8);
}

TEST_F(SendRecvTileTestFixture, NvlLarge) {
  run_sendrecv_tile_test(4 * 1024 * 1024, 16);
}

TEST_F(SendRecvTileTestFixture, NvlWithSignalBytes) {
  // 16 KiB, NOT the NVL per-channel slot size. The slot here is
  // (data_buffer_size / maxNumChannels) / pipelineDepth
  // = (8 MiB / 32) / 4 = 64 KiB, and P2pNvlTransportDevice only takes the
  // custom-chunk path when `0 < maxSignalBytes < perChannelSlot` -- so passing
  // exactly 64 KiB fell through to the same default path as passing 0 and this
  // test exercised nothing. Anything strictly below the slot does.
  run_sendrecv_tile_test(1024 * 1024, 8, /*ib_only=*/false, 16 * 1024);
}

// ============================================================================
// IB tests (p2pDisable forces the peer onto IBGDA)
// ============================================================================

// The plain path dispatches on the IB backend rather than reaching for
// get_ibgda(), so the same collective has to be correct on both backends.
// Parameterized rather than IBGDA-only: an IBGDA-only run passes just as
// happily when the plain path hard-routes to IBGDA, which is exactly the bug
// the dispatch replaced. Compressed stays unparameterized below -- it is
// genuinely IBGDA-bound (nvcompdx staging + the variable-size wire protocol)
// and now says so through require_ibgda().
class SendRecvTileIbBackendFixture
    : public SendRecvTileTestFixture,
      public ::testing::WithParamInterface<IbBackendMode> {
 protected:
  IbBackendMode ibMode() const {
    return GetParam();
  }
};

INSTANTIATE_TEST_SUITE_P(
    IbBackends,
    SendRecvTileIbBackendFixture,
    ::testing::Values(IbBackendMode::kIbgda, IbBackendMode::kIbrc),
    [](const ::testing::TestParamInfo<IbBackendMode>& info) {
      return info.param == IbBackendMode::kIbgda ? "IBGDA" : "IBRC";
    });

TEST_P(SendRecvTileIbBackendFixture, IbSmall) {
  run_sendrecv_tile_test(4096, 4, /*ib_only=*/true, 0, ibMode());
}

TEST_P(SendRecvTileIbBackendFixture, IbMedium) {
  run_sendrecv_tile_test(256 * 1024, 8, /*ib_only=*/true, 0, ibMode());
}

TEST_P(SendRecvTileIbBackendFixture, IbLarge) {
  run_sendrecv_tile_test(4 * 1024 * 1024, 14, /*ib_only=*/true, 0, ibMode());
}

// ============================================================================
// Two-way (bidirectional) tests: each rank sends to and receives from its
// partner at the same time; the grid is split half-send / half-recv.
// ============================================================================

// Both direction flags set on both ranks, but one DIRECTION OF THE PAIR
// carries no bytes: the even rank sends 0 and receives N, the odd rank sends N
// and receives 0. (Zeroing the same field on both ranks would just be an
// invalid flow -- nobody sends, both wait, and the test hangs on its own
// premise rather than on a kernel defect. Asked for by @boyda.)
//
// The grid split is keyed on the FLAGS, not the counts. A rank that collapsed
// to "one direction, all blocks" because a count happened to be zero would use
// `gridDim.x` blocks for that direction while its peer uses `gridDim.x / 2`,
// and the mismatched per-direction active-block count shows up as dropped data
// or a hang, not a clean error.
TEST_F(SendRecvTileTestFixture, NvlTwoWayAsymmetricZeroCount) {
  run_sendrecv_tile_twoway_test(
      /*bytes=*/256 * 1024,
      /*num_blocks=*/8,
      /*ib_only=*/false,
      /*max_signal_bytes=*/0,
      IbBackendMode::kIbgda,
      /*asymmetric_zero_count=*/true);
}

TEST_F(SendRecvTileTestFixture, NvlTwoWaySmall) {
  run_sendrecv_tile_twoway_test(4096, 8);
}

TEST_F(SendRecvTileTestFixture, NvlTwoWayLarge) {
  run_sendrecv_tile_twoway_test(4 * 1024 * 1024, 16);
}

TEST_P(SendRecvTileIbBackendFixture, IbTwoWaySmall) {
  run_sendrecv_tile_twoway_test(4096, 8, /*ib_only=*/true, 0, ibMode());
}

TEST_P(SendRecvTileIbBackendFixture, IbTwoWayLarge) {
  run_sendrecv_tile_twoway_test(
      4 * 1024 * 1024, 16, /*ib_only=*/true, 0, ibMode());
}

// ============================================================================
// Compressed mixed-mode tests: the ANS-compressed kernel with a runtime
// fraction of blocks falling back to plain Memcpy. Transfers are >= the 4 MiB
// ANS activation threshold so compression actually engages. Fractions 0.0
// (all-ANS) and 1.0 (all-plain) exercise the endpoints; 0.5 is the true mix.
// ============================================================================

// Boundary case for the `max_signal_bytes` clamp added in D112007059.
//
// 384 KiB is the value that used to trap. The transport accepts a caller hint
// when `0 < value < perBlockSlot`, and here perBlockSlot is
// `ibConfig.perChannelSize / pipelineDepth` = 1 MiB / 2 = 512 KiB -- so 384 KiB
// passes that check. But the binding constraint is the WORST-CASE EXPANDED
// layout: 384 KiB spans two 256 KiB ANS chunks, whose
// `worst_case_chunk_stride` is ~669 KiB, well past the 512 KiB slot. Before the
// clamp that reached `chunkStride > perBlockSlot` and trapped, with the
// diagnostic printf swallowed by `__trap()` so it surfaced as a bare
// "unspecified launch failure". The clamp now reduces the hint to the largest
// chunk that does fit, so this must simply succeed.
//
// 256 KiB is included as the control: it is also below the slot but its
// expanded form fits, so it is honoured rather than clamped. Both must produce
// byte-exact output.
// ============================================================================
// 3+ rank ring: send_peer != recv_peer AND send_count != recv_count.
//
// The pairwise fixture used everywhere else cannot express this: with one
// partner and one byte count, a receive path that read `send_peer` or
// `send_count` by mistake reads the same value and passes anyway. Requested by
// @boyda.
// ============================================================================
TEST_F(SendRecvTileTestFixture, IbRingDistinctPeersAndCounts) {
  if (numRanks < 3) {
    GTEST_SKIP() << "Ring requires >= 3 ranks";
  }
  run_sendrecv_tile_ring_test(/*num_blocks=*/8, /*ib_only=*/true);
}

TEST_F(SendRecvTileTestFixture, NvlRingDistinctPeersAndCounts) {
  if (numRanks < 3) {
    GTEST_SKIP() << "Ring requires >= 3 ranks";
  }
  run_sendrecv_tile_ring_test(/*num_blocks=*/8, /*ib_only=*/false);
}

// ============================================================================
// Ring wrap-around: repeated launches on ONE transport.
//
// Every other test in this file builds a fresh transport and launches once, so
// a block touches at most the initial slots and never gives one back. That
// leaves SLOT_FREE credit return, local NIC-completion retirement and the
// persistent per-channel cursors unexercised -- despite the Summary claiming
// otherwise. The benchmark (D108504937) does hammer these paths (one transport,
// 105 launches per size), but it measures time and verifies nothing, so silent
// corruption in recycling would pass there unnoticed. Verification lives here.
//
// The cursors are transport state that survives a kernel launch, so wrapping
// needs more LAUNCHES, not bigger payloads. IB staging is
// perChannelSize / pipelineDepth = 1 MiB / 2 = two 512 KiB slots per channel;
// at 4 MiB over 16 blocks each launch advances a block 256 KiB, so 8 rounds
// consume 2 MiB per block and wrap the ring roughly four times. Each round is
// byte-verified, so a round that recycles a slot wrongly fails on content
// rather than being overwritten by the next one.
// ============================================================================

TEST_F(SendRecvTileTestFixture, IbPlainRingWrapAround) {
  run_sendrecv_tile_test(
      4 * 1024 * 1024,
      16,
      /*ib_only=*/true,
      /*max_signal_bytes=*/0,
      IbBackendMode::kIbgda,
      /*iterations=*/8);
}

TEST_F(SendRecvTileTestFixture, IbCompressedRingWrapAround) {
  run_sendrecv_tile_compressed_test(
      4 * 1024 * 1024,
      16,
      /*plain_block_fraction=*/0.0f,
      /*max_signal_bytes=*/0,
      /*iterations=*/8);
}

// plain -> compressed -> plain on ONE transport.
//
// The two paths use different wire protocols over the same per-channel staging,
// so this is the case where a cursor or category left in the wrong state by one
// protocol is picked up by the other. That only means anything if all three
// phases share a handle: an earlier version of this test let each helper build
// its own transport, which gave the three phases three independent sets of
// staging cursors and therefore tested nothing about protocol hand-off. The
// transport is created here and threaded into every phase.
//
// Each phase wraps the ring on its own, and every round is byte-verified
// against its own generation, so a slot recycled wrongly cannot pass by
// replaying the previous round.
TEST_F(SendRecvTileTestFixture, IbAlternatingPlainCompressedReusesTransport) {
  auto transport = create_and_exchange(/*ib_only=*/true);

  run_sendrecv_tile_test(
      4 * 1024 * 1024,
      16,
      /*ib_only=*/true,
      /*max_signal_bytes=*/0,
      IbBackendMode::kIbgda,
      /*iterations=*/4,
      transport.get());
  run_sendrecv_tile_compressed_test(
      4 * 1024 * 1024,
      16,
      /*plain_block_fraction=*/0.0f,
      /*max_signal_bytes=*/0,
      /*iterations=*/4,
      transport.get());
  run_sendrecv_tile_test(
      4 * 1024 * 1024,
      16,
      /*ib_only=*/true,
      /*max_signal_bytes=*/0,
      IbBackendMode::kIbgda,
      /*iterations=*/4,
      transport.get());
}

// The picker is the public contract for which launch shapes exist, so test the
// contract and not just the happy path. Before it existed, an unsupported pair
// compiled at the call site and failed at DEVICE LINK with an undefined symbol;
// the point of the indirection is that it now fails here, host-side, with a
// null return the caller can act on.
//
// Host-only: no GPU, no peers, no transport -- it is a pure lookup.
TEST_F(
    SendRecvTileTestFixture,
    CompressedKernelPickerMapsOnlyInstantiatedPairs) {
  // Every pair explicitly instantiated in SendRecvTileCompressed.cu.
  for (const int threads : {32, 64, 128, 256, 512, 1024}) {
    EXPECT_NE(pick_sendrecv_tile_compressed_kernel(threads, 2), nullptr)
        << "threads_per_block=" << threads << " at min_blocks_per_sm=2";
  }
  // The one extra full-occupancy variant.
  EXPECT_NE(pick_sendrecv_tile_compressed_kernel(256, 8), nullptr);

  // Not instantiated -- each of these used to link-fail instead.
  EXPECT_EQ(pick_sendrecv_tile_compressed_kernel(256, 4), nullptr)
      << "<AnsCompressor, 8, 4> is not emitted; the picker must say so";
  EXPECT_EQ(pick_sendrecv_tile_compressed_kernel(512, 8), nullptr);
  // 64 warps cannot exist: CUDA caps a block at 1024 threads, so
  // __launch_bounds__(2048, ...) has no valid launch.
  EXPECT_EQ(pick_sendrecv_tile_compressed_kernel(2048, 2), nullptr);
  // Not a multiple of the warp size.
  EXPECT_EQ(pick_sendrecv_tile_compressed_kernel(100, 2), nullptr);
  EXPECT_EQ(pick_sendrecv_tile_compressed_kernel(0, 2), nullptr);
}

TEST_F(SendRecvTileTestFixture, IbCompressedClampsUnsafeSignalBytes) {
  run_sendrecv_tile_compressed_test(
      4 * 1024 * 1024,
      16,
      /*plain_block_fraction=*/0.0f,
      /*max_signal_bytes=*/384 * 1024);
}

TEST_F(SendRecvTileTestFixture, IbCompressedHonoursSafeSignalBytes) {
  run_sendrecv_tile_compressed_test(
      4 * 1024 * 1024,
      16,
      /*plain_block_fraction=*/0.0f,
      /*max_signal_bytes=*/256 * 1024);
}

TEST_F(SendRecvTileTestFixture, IbCompressedMixedAllAns) {
  run_sendrecv_tile_compressed_test(4 * 1024 * 1024, 16, /*frac=*/0.0f);
}

TEST_F(SendRecvTileTestFixture, IbCompressedMixedHalf) {
  run_sendrecv_tile_compressed_test(4 * 1024 * 1024, 16, /*frac=*/0.5f);
}

TEST_F(SendRecvTileTestFixture, IbCompressedMixedAllPlain) {
  run_sendrecv_tile_compressed_test(4 * 1024 * 1024, 16, /*frac=*/1.0f);
}

TEST_F(SendRecvTileTestFixture, IbCompressedTwoWayMixedAllAns) {
  run_sendrecv_tile_compressed_twoway_test(4 * 1024 * 1024, 16, /*frac=*/0.0f);
}

TEST_F(SendRecvTileTestFixture, IbCompressedTwoWayMixedHalf) {
  run_sendrecv_tile_compressed_twoway_test(4 * 1024 * 1024, 16, /*frac=*/0.5f);
}

TEST_F(SendRecvTileTestFixture, IbCompressedTwoWayMixedAllPlain) {
  run_sendrecv_tile_compressed_twoway_test(4 * 1024 * 1024, 16, /*frac=*/1.0f);
}

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
