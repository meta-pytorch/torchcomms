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

`src/include/os.h` declares `ncclOsSocketSetNetworkOptions`. Every platform calls it from `ncclOsSocketResetFd` while configuring a replacement descriptor. The replacement is installed only after all setup succeeds, so a failure preserves the original descriptor instead of exposing a partially configured socket. Consequently, a retry-created descriptor receives the same interface binding and traffic class as the original descriptor.

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
- preserving the original descriptor when replacement setup fails.

The 2.32-only assertions are version-gated so the shared source continues to build for older baselines.

The shared meta-test generator still excludes 2.32. `meta/tests/BUCK` therefore owns a dedicated `socket-set-tos-test_v2_32` target so these assertions remain part of the checked-in build graph.

## Revert checklist

Remove the three fields from `ncclSocket`, restore the single-argument `ncclSocketConnect` declaration, remove `ncclOsSocketSetNetworkOptions` and its reset calls, remove the address-prefix filter and the three version-gated tests, and delete the dedicated 2.32 socket-test target.
