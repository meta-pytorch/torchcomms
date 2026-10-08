# Socket interface and traffic-class options

## Background

NCCLX extends the baseline socket implementation with three controls:

- `NCCL_SOCKET_IPADDR_PREFIX` limits interface discovery to numeric IP addresses with the configured prefix.
- `NCCL_CLIENT_SOCKET_IFNAME` binds outgoing sockets to a named Linux interface.
- `NCCL_SOCKET_TOS_CONFIG` sets `IP_TOS` for IPv4 sockets or `IPV6_TCLASS` for IPv6 sockets.

`NCCL_SOCKET_IFNAME` is provided by upstream NCCL and is not part of this modification.

## Versions affected

The controls exist in the maintained NCCLX baseline trees and must be carried when a new baseline version is introduced. The 2.32 implementation deliberately differs from the older implementation in order to preserve options across connection retries and to keep platform-specific APIs out of common socket code.

## 2.32 implementation

### Persistent socket configuration

`src/include/socket.h` stores the selected client interface, whether interface binding is active, and the configured traffic class in `ncclSocket`. `ncclSocketMove` already transfers the complete structure, so these values follow socket ownership changes.

`src/misc/socket.cc` snapshots the two CVAR values during `ncclSocketInit`. Interface binding is activated by `ncclSocketConnect`, immediately before TLS setup and before the first operating-system `connect` call. The optional `localIfName` argument can override the CVAR for an individual connection.

`src/include/os.h` declares `ncclOsSocketSetNetworkOptions`. When no descriptor exists, each platform calls it from `ncclOsSocketResetFd` while configuring the initial descriptor from the cached settings. On Linux, a retry instead reads the live interface binding and traffic class from the existing descriptor and copies them to the replacement, so direct descriptor updates are preserved as well. Windows reapplies the cached supported settings. The replacement is installed only after all setup succeeds, so a failure preserves the original descriptor instead of exposing a partially configured socket.

### Linux behavior

`src/os/linux.cc` implements the three controls. `SO_BINDTODEVICE` receives the NUL-terminated interface name, and traffic-class configuration uses the address-family-specific socket option.

Interface discovery renders an address only when `NCCL_SOCKET_IPADDR_PREFIX` is nonempty. A render failure or a nonmatching prefix excludes that address. IPv4 prefixes must end at an octet boundary unless they include the trailing dot; IPv6 retains textual-prefix matching so partial hextet prefixes remain supported.

### Windows behavior

`src/os/windows.cc` applies the supported traffic-class option without depending on Linux declarations. `NCCL_CLIENT_SOCKET_IFNAME` returns `ncclInvalidUsage` on Windows because Windows has no name-based `SO_BINDTODEVICE` equivalent. This makes unsupported configuration explicit while preserving the Windows build.

## Initialization dependency

These controls use the shared NCCLX CVAR framework. The 2.32 integration must call `ncclCvarInit` before baseline socket initialization so the socket snapshots initialized values rather than zero-initialized globals.

## Regression coverage

`meta/tests/SocketTosTest.cpp` covers:

- applying TOS or traffic class to the initial descriptor and a reset descriptor;
- positive and negative IP-address-prefix filtering;
- rejecting IPv4 prefixes that end in the middle of an octet;
- applying `NCCL_CLIENT_SOCKET_IFNAME` and preserving it after a descriptor reset;

`meta/tests/SocketRetryTest.cc` separately covers descriptor-derived retry preservation and retaining the original descriptor when replacement setup fails.

The 2.32-only assertions are version-gated so the shared source continues to build for older baselines.

The per-rule meta-test generator enables `socket-set-tos-test_v2_32` and supplies the Linux-only compiler flag needed by the 2.32 assertions, so these checks remain part of the checked-in build graph without a duplicate standalone rule.

## Revert checklist

Remove the three fields from `ncclSocket`, restore the single-argument `ncclSocketConnect` declaration, remove `ncclOsSocketSetNetworkOptions` and its reset calls, remove the address-prefix filter and the three version-gated tests, and re-add 2.32 to the socket test's `excluded_versions`.
