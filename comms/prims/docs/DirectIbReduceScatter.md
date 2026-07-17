# Direct IB Reduce-Scatter

Direct IB reduce-scatter is a PRIMS collective kernel for reducing equal-sized
rank shards directly over IB and writing only the local output shard. The
current surface is a standalone PRIMS launcher and benchmark; the follow-up MCCL
integration will route `ncclReduceScatter` through CTran to this same kernel for
explicit opt-in benchmarking with `nccl-tests-suite`.

## Data Flow

Each rank starts with `num_ranks` input chunks and owns one output chunk. For
rank `r`, the output is:

```text
output_r = reduce(input_0[r], input_1[r], ..., input_(num_ranks - 1)[r])
```

The kernel maps one CTA to one logical IB channel. Inside each CTA, threads are
split into a receive/reduce group and a send group. The receive group first
copies this rank's local contribution for the owned shard into the output tile.
It then receives the peer contributions for that same shard through IB and
reduces each contribution into the output tile. The send group concurrently
sends this rank's contribution for peer-owned shards to those peers.

The channel schedule is staggered by channel ID so all channels do not target
the same peer in the same step. This keeps the direct all-to-all exchange
simple while avoiding a single hot peer at each step.

## Implementation

The supported launcher is:

```cpp
launch_direct_reduce_scatter_ib(params);
```

The current kernel uses:

- 512 threads per CTA.
- 128 send threads and 384 receive/reduce threads.
- One CTA per IB channel.
- `TileReduceStaged` for receive-side reduction.

The receive group reduces incoming staged IB data into the output tile using a
register/tile-staged reduce policy. Shared-memory async staging is a follow-up
optimization layer on top of this baseline.

## Transport Setup

The standalone benchmark creates a `MultipeerIbgdaTransport` with fixed-channel
IB staging:

- `max_num_channels = 8`
- `perChannelSize = 512KB`
- `pipelineDepth = 4`
- `qpsPerConnection = 1`

Here `perChannelSize` is the total staging window for one channel, so the
transport chunk size is `perChannelSize / pipelineDepth = 128KB`.

Each rank exchanges transport metadata, materializes every non-self peer, and
passes the peer device handles to the launch params. The future CTran/MCCL hook
should move that transport ownership to communicator lifetime so standard
`ncclReduceScatter` callers do not need benchmark-local setup code.

## Current Scope

The current PRIMS surface is intentionally narrow:

- `float`
- `sum`
- out-of-place benchmark comparison
- IB transport
- `num_ranks <= kDirectReduceScatterIbMaxRanks`

The benchmark compares the direct IB kernel against NCCL Ring and forced NCCL
PAT on the same 2-node x 4-GPU GB300 setup. The standalone benchmark remains in
this diff until the MCCL / `nccl-tests-suite` path exercises the same kernel.
