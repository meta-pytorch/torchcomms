# socket.cc eth0 enhancements: Baseline Modifications

## Background

Meta production jobs configure NCCL's control-plane sockets through four cvars. Three of them have
no upstream equivalent and are implemented as patches on baseline socket code:

- `NCCL_SOCKET_IPADDR_PREFIX` — selects interfaces whose numeric address starts with a given
  prefix. 606+ external references; set to `2401` across GB300.
- `NCCL_CLIENT_SOCKET_IFNAME` — binds the client side of a connection to a named interface via
  `SO_BINDTODEVICE`. 358+ external references.
- `NCCL_SOCKET_TOS_CONFIG` — sets `IP_TOS` (IPv4) or `IPV6_TCLASS` (IPv6) on the socket.

The fourth, `NCCL_SOCKET_IFNAME`, is upstream's own and needs no patch: baseline
`ncclFindInterfaces` reads it via `ncclGetEnv` and v2_30 and v2_31 are byte-identical there.

## Versions Affected

v2_29, v2_30, v2_31

## Baseline Files Modified

Three files. Note the Feature Owners Map historically listed only `src/misc/socket.cc`; the row
actually spans all three, and `NCCL_SOCKET_IPADDR_PREFIX` lives in the two that were unlisted.

### 1. `src/include/socket.h`

- `#include <string>` for the helper's return type.
- `ncclSocketConnect` gains a trailing `const char* localIfName = nullptr`. The default keeps all
  nine existing call sites source-compatible; none passes the argument today.
- Declares `std::string ncclSocketToIPv6String(union ncclSocketAddress* addr)`.

### 2. `src/misc/socket.cc`

- `#include "comms/utils/cvars/nccl_cvars.h"`.
- Defines `ncclSocketToIPv6String`, a numeric-host rendering via `getnameinfo` with
  `NI_NUMERICHOST`, used to match the prefix.
- `ncclSocketConnect` takes `localIfName`, defaults it from `NCCL_CLIENT_SOCKET_IFNAME`, and binds
  with `SO_BINDTODEVICE`.
- `ncclSocketInit` applies `NCCL_SOCKET_TOS_CONFIG` when it is not `-1`.

### 3. `src/os/linux.cc`

- `#include "comms/utils/cvars/nccl_cvars.h"` explicitly. v2_30 got this transitively; relying on
  that is the same header-hygiene hazard D117198384 fixed for `cpuset.h`.
- `ncclOsFindInterfaces` skips interfaces whose rendered address does not match
  `NCCL_SOCKET_IPADDR_PREFIX`.

## Deliberate divergence from v2_30

The 2.31 port is not a byte-for-byte copy of v2_30. Four things were left behind on purpose,
because they are not this row and are stale relative to upstream. Pristine 2.30's `socket.cc` is
byte-identical to pristine 2.31's, so upstream already carried all of the below in 2.30; these are
Meta patches that won earlier rebase conflicts, not 2.31 regressions.

1. **The topology/PCI block** deleted from v2_30's `os/linux.cc` (`ncclOsGetPciPath`,
   `ncclOsTopoGetStrFromSys`, `memcpylower`, ~108 lines). It is coupled to `include/os.h`,
   `os/windows.cc` and `graph/xml.cc`, whose Meta patch keeps a private static `getPciPath`. It
   belongs to the topology row, and porting it here breaks the build because 2.31 still declares
   and calls those symbols.
2. **Removal of `ncclSocketDefaultMagic`, `ncclSocketStateBadHandshake`, and `ncclSocketAccept`'s
   `retry` parameter.** All three are used by other 2.31 files (`ras/`, `transport/net_socket.cc`,
   `transport/net_ib/connect.cc`, `proxy.cc`, `os/windows.cc`).
3. **`socketFinalizeAccept` hard-failing on socket-type mismatch.** Upstream logs, discards the
   peer connection and keeps listening; v2_30 returns `ncclInternalError`.
4. **The dropped `virbr` exclusion.** v2_30's `ncclFindInterfaces` filters `"^docker,lo"` where
   upstream filters `"^docker,lo,virbr"`, so v2_30 can select a libvirt bridge as the bootstrap
   interface. v2_30 flags this itself with `FIXME[max7255]: we dropped virbr for some reason`.
   2.31 keeps upstream's exclusion.

## Hardening applied on top of the v2_30 behaviour

These were pre-existing issues in v2_30, fixed rather than carried forward:

- `ncclSocketToIPv6String` zero-initializes its buffer and returns empty when `getnameinfo` fails,
  instead of constructing a `std::string` over uninitialized stack memory. It also passes the
  family-correct `salen` rather than the union size.
- The `SO_BINDTODEVICE` `ifreq` is zero-initialized and explicitly NUL-terminated, and `setsockopt`
  receives `&ifr` with `sizeof(ifr)` so pointer and length agree. The interface name is
  user-supplied via a cvar.
- The prefix render and its `INFO` line are gated behind a non-empty `NCCL_SOCKET_IPADDR_PREFIX`.
  v2_30 ran both unconditionally, costing a `getnameinfo` and a log line per interface on every
  job, including the majority that never set the prefix.

## Open items

- `FIXME[max7255]` in `socketWait` ties `ncclParamPollTimeOut` to `NCCL_FASTINIT_MODE`. The ncclx
  patch it refers to was dropped during the 2.30 rebase and only the comment survived;
  `SOCKET_POLL_TIMEOUT_MSEC` still defaults to `0`, so the poll is skipped and the loop spins.
  Owned by the Fast Init row.
- These cvars use the NCCLX cvar framework. If NCCLX converges on upstream 2.31's env plugin ABI
  (`ncclEnv_v2_t::getEnv`), the two `nccl_cvars.h` includes in baseline files here are what would
  go away.

## Revert checklist

Remove the three cvar uses and the helper, restore `ncclSocketConnect` to a single parameter, and
drop the two `nccl_cvars.h` includes. Nothing else in `v2_31/src` depends on them.
