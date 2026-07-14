# PAT transport connect honors `NCCL_NETDEVS_POLICY`: Baseline Modifications

## Background

`ncclTransportPatConnect` (the eager binomial-tree connect path in
`src/transport/generic.cc`, and its lazy Meta counterpart `ncclx::transportPatConnect`
in `meta/transport/transportConnect.cc`) originally called
`ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0)` for both the
ReduceScatter and AllGather directions of each binomial-tree mask. Passing a
non-NULL `graph` argument routes NIC selection through the "graph-based" branch
of `ncclTopoGetNetDev` (`src/graph/search.cc:1325-1338`):

```cpp
if (graph) {
  int channel = channelId % graph->nChannels;
  int ngpus   = comm->topo->nodes[GPU].count;
  int index   = graph->intra[channel*ngpus] == rank ? 0 : 1;
  if (graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
    netId = graph->inter[channel*2 + index];
  }
  ...
}
```

That branch reads a shared per-channel entry NIC from
`graph->inter[channel*2 + index]`. For collectives like RING / TREE this is
correct — each channel is a chain of GPUs that must agree on the entry NIC.
**PAT is different:** each `(mask, direction)` opens a set of independent
rank-to-rank connections (`prevPeer` / `nextPeer`), with no cross-rank
consistency requirement. Every rank should be free to use its own local NIC.

The consequence of routing PAT through the graph branch was two-fold:

1. On a multi-rail node (e.g., GB300 4 GPUs × 4 bonded vNICs), the TREE graph
   rotates the entry NIC across channels for aggregate-bandwidth optimization,
   so all ranks on channel `c` land on the same NIC — which is local to only
   one of them. `NCCL_NETDEVS_POLICY=MAX:1` had no effect: it caps the graph's
   *candidate pool* but does not force per-rank local pinning inside the graph
   branch.
2. On an MNNVL clique that spans multiple hosts (multiple `systemId`s in one
   fused topology), `graph->inter[]` can hold a netId whose `systemId` is not
   the local rank's `systemId`. The runtime `ncclTopoIdToNetDev` strips the
   `systemId` and returns the local rail index, so the plugin correctly opens
   the local NIC — but the paired `ncclTopoCheckGdr` call in
   `src/transport/net.cc:308` uses the full composite `netId` and looks up
   `PATH_DIS` (0/0.0) in the topology, returning `useGdr=0`. Result: every
   channel whose graph slot pointed at a cross-`systemId` NIC silently
   degraded to a pinned-host-bounce path even though the actual local NIC is
   GDR-capable.

Passing `NULL` for the graph argument makes `ncclTopoGetNetDev` fall through
to `ncclTopoGetLocalNet` (the `graph == NULL` branch at
`src/graph/search.cc:1339+`). That branch honors `netsPerGpu` (which
`NCCL_NETDEVS_POLICY=MAX:1` collapses to 1 on this hardware) and pins each
rank to its own local NIC — which is what PAT's rank-to-rank connections
actually want.

## Versions Affected

v2.29, v2.30

## Baseline Files Modified

Exactly ONE baseline file per version: `src/transport/generic.cc`. Two lines
inside `ncclTransportPatConnect` (the eager, non-lazy path) — one flush after
the RS-direction `ncclTransportP2pConnect` loop, one after the AG-direction
loop:

```cpp
ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) {
  ...
  for (int mask=1; mask<comm->nRanks; mask<<=1) {
    ...
    for (int c = 0; c < comm->nChannels; c++) {
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, nullptr, 0), ret, fail);  // [META]: pass nullptr, see meta/baseline_modification_docs/pat_transport_netdevs_policy.md
    for (int c = 0; c < comm->nChannels; c++) {
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, nullptr, 0), ret, fail);  // [META]: pass nullptr, see meta/baseline_modification_docs/pat_transport_netdevs_policy.md
  }
  ...
}
```

The parallel change in the Meta lazy path (`meta/transport/transportConnect.cc`,
function `ncclx::transportPatConnect`, two `ncclTransportP2pSetup` calls) is
Meta-owned code and does not carry a `[META]` tag — the whole file is a Meta
overlay. Both callers now consistently pass `nullptr`.

All OTHER PAT call sites are UNCHANGED:
- `ncclTransportP2pConnect` inside PAT still passes `connIndex=0` (the
  collective slot). The graph argument was only ever consumed by the *setup*
  step, not the *connect* bookkeeping step.
- `net.cc`'s `sendSetup`/`recvSetup` and the plugin-side QP creation are
  unchanged; they consume `req.netDev` (a local index) which is what the
  `graph == NULL` branch now produces.

## Why in baseline

There is no clean overlay point. The `ncclTransportPatConnect` symbol is
declared in the baseline (`src/include/transport.h`) and is what
`ncclCollPreconnect` (`src/group.cc:190-193`, baseline) dispatches to on the
`NCCL_ALGO_PAT` case. Replacing the whole function via a Meta wrapper would
mean duplicating the entire binomial-tree loop and keeping it in sync with
upstream on every rebase. A two-line edit at the exact `P2pSetup` call sites
is the smallest possible footprint and follows the pattern in
[`builtin_tuner.md`](./builtin_tuner.md) of narrow, tagged baseline hooks.

The lazy Meta path already runs entirely inside `meta/transport/`, so it
gets the same fix without any baseline touch.

## Related upstream discussion

We're proposing this upstream (see the draft email in the diff summary /
task) because the root cause — PAT reusing the TREE graph even though its
rank-to-rank connections don't need graph-consistent NICs — affects all
users of PAT on multi-rail / MNNVL topologies with `NCCL_PXN_DISABLE=1`,
not just Meta. If upstream accepts the change, the baseline hook can be
removed on the next rebase.
