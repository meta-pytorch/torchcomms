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

## User Model

Users choose how many channels a transport can use when constructing the host
transport. Kernels normally select a channel implicitly: `ThreadGroup::group_id`
is the channel id. Current send/recv APIs do not expose an arbitrary explicit
channel id to device callers; explicit channel selection would need an API that
also defines ownership and duplicate-use rules.

Multiple channels may connect the same pair of ranks. Each channel is an
independent logical lane with its own staging slice and protocol state. A
kernel may use fewer channels than the host allocated, but it must not launch
more channel-using groups than the transport's configured channel count.

Channels are duplex. The same channel id can carry traffic in both directions
between two ranks, but each protocol defines the concurrent ownership unit. For
NVLink, the channel state contains both send and recv cursors. For IB, the
ownership unit is `(peer, channel_id, direction)`.

Channel resources are created by the host transport before device kernels use
them. After peers exchange IPC/RDMA addresses, device transports hold pointers
into those resources. The resources remain valid until the host transport is
destroyed; device transports must not outlive the host transport that produced
them.

Current channel initialization is eager: the host allocates the configured
channel count up front. Lazy channel initialization is future work and would
need to preserve stable channel ids, staging layout, and peer-visible protocol
state before any kernel indexes a channel.

## NVLink

`MultiPeerNvlTransport` owns:

- staging buffers, one per peer and direction
- `NvlChannelState[maxNumChannels]`, one array per peer
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

NVLink uses fixed channel geometry. `maxNumChannels` and `perChannelSize` are
selected at host transport construction and remain stable across kernel
launches. This keeps the staging-buffer slice for channel `c` stable even when
a later kernel uses fewer active groups.

## IB

IB transports are moving to the same fixed-channel model. In this model, an IB
channel is selected by `ThreadGroup::group_id`, but the concurrent ownership
unit is:

```text
(peer, channel_id, direction)
```

`direction` is `Send` or `Recv`. One group may actively use the Send side of a
channel while another group uses the Recv side of the same channel. Concurrent
duplicate use of the same `(peer, channel_id, direction)` is caller error and
the transport does not serialize it.

Unlike NVLink, an IB channel owns both protocol state and NIC posting
bookkeeping. The local channel carries:

- send and recv progress slots
- local `DATA_READY`, `SLOT_FREE`, and `NIC_DONE` wait/completion buffers. The
  `NIC_DONE` wait view and completion-target view are separate because IBRC
  waits from the GPU while the CPU proxy completes through a host alias.
- send-side and recv-side QP selection/flush state

The remote channel is not a pointer to the peer's full local channel. It is a
set of RDMA targets this rank can write: peer `DATA_READY`, peer `SLOT_FREE`,
and peer recv staging.

QP resources are selected by `(channel_id, direction, NIC, qp_index)`.
`qpsPerConnection` means QPs per `(channel_id, direction, NIC)`. IBGDA
companion QPs are per `(channel_id, direction, NIC)` and are not multiplied by
`qpsPerConnection`.

Public raw put/signal APIs default to the Send direction. Send/recv/forward
internals use explicit directions: data puts and `DATA_READY` use Send, while
recv-side `SLOT_FREE` uses Recv.
