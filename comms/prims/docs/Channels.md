# PRIMS Transport Channels

This document describes the channel ownership model used by PRIMS transports.
A channel is a fixed logical lane between two peer ranks. Device collectives
map a `ThreadGroup::group_id` to a channel id, and the transport maps that
channel id to persistent protocol state and a staging-buffer slice.

## Common Model

The host transport owns channel resources. A device transport only receives
pointers and scalar options that let a kernel index those resources. A channel
is a fixed logical lane id. The transport maps that lane id to:

- a fixed staging-buffer slice
- one channel-owned protocol-state record for each peer

The transport physically owns the backing memory, but the channel semantically
owns its protocol state. For NVLink, that state is `NvlChannelState`: the
channel's byte cursors and synchronization signals. The channel protocol is the
only code path that should mutate those fields.

## NVLink

`MultiPeerNvlTransport` owns:

- staging buffers, one per peer and direction
- `NvlChannelState[max_num_channels]`, one array per peer
- IPC mappings to each peer's staging buffers and channel-state arrays

`P2pNvlTransportDevice` stores two channel-state pointers:

```cpp
NvlChannelState* local_channels_;
NvlChannelState* remote_channels_;
```

For channel `c`, `local_channels_[c]` is this rank's local protocol state and
`remote_channels_[c]` is an IPC pointer to the remote rank's local state for
the same channel id. The sender waits on its local `slot_free`, writes data into
the remote staging slice, then signals the remote channel's `data_ready`. The
receiver waits on its local `data_ready`, reads its local staging slice, then
signals the remote channel's `slot_free`.

`NvlChannelState` contains the channel-owned state for one local lane:

```cpp
struct NvlChannelState {
  int64_t send_cursor;   // local send progress for this channel
  int64_t recv_cursor;   // local recv progress for this channel
  SignalState data_ready; // remote sender writes; local recv waits
  SignalState slot_free;  // remote receiver writes; local send waits
};
```

The same channel id exists on both ranks. Each rank owns its own
`NvlChannelState`; the peer can signal that state through its
`remote_channels_` pointer.

NVLink uses fixed channel geometry. `max_num_channels` and `perChannelSize` are
selected at host transport construction and remain stable across kernel
launches. This keeps the staging-buffer slice for channel `c` stable even when
a later kernel uses fewer active groups.

## IB

IB transports still use per-call active group geometry today. Their staging
slice width can depend on the active block count for that call, so users must
preserve the transport's synchronization contract when changing active blocks
between calls. NVLink avoids that dynamic-layout hazard by making channels
fixed at transport initialization.

## Bidirectional CTAs

A bidirectional CTA can use one channel for both directions. The current NVLink
benchmark creates two half-block multiwarp groups:

```cpp
auto group = make_multiwarp_group(blockDim.x / 2);
auto [role, sub] = group.partition_interleaved(2);
```

For a 512-thread CTA, this creates two 256-thread groups. Interleaved
partitioning maps the send half and recv half back to the same `sub.group_id`,
so one CTA owns one bidirectional channel.
