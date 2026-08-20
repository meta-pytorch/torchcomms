# OS Platform Macro Default: Baseline Modifications

## Background

`src/include/os.h` selects the platform implementation header from a
preprocessor macro:

```c
#if defined(NCCL_OS_WINDOWS)
#include "os/windows.h"
#elif defined(NCCL_OS_LINUX)
#include "os/linux.h"
#endif
```

Those headers are where `ncclSocketDescriptor` and `ncclAffinity` are declared.
If neither macro is defined, neither header is included, and every declaration in
`os.h` that names one of those types fails to compile — along with
`socket.h`, which embeds `ncclSocketDescriptor` in `struct ncclSocket`, and
`alloc.h`, whose `page_size` local becomes unused because neither the POSIX nor
the Windows allocation branch is compiled.

Upstream expects the build system to supply the macro. The Buck build does, via
`-DNCCL_OS_LINUX` in `NCCL_COMPILER_FLAGS` (`def_build.bzl`) — but `compiler_flags`
are private to the target that declares them. Any target that consumes only the
NCCLX **header** set inherits the headers without the flag:

- `ncclx-private-headers` / `nccl<ver>-internal`
- everything reached through them, notably `comms/testinfra/TestUtils.h`, which
  pulls in `bootstrap.h` -> `comm.h` -> `p2p.h` -> `core.h` -> `alloc.h` -> `os.h`

The result is that the library itself builds, while every generated
`*_v2_31` meta test that includes `TestUtils.h` fails to compile, on types that
look unrelated to anything the test does.

## Versions Affected

v2_31. `v2_29` and `v2_30` already carry an equivalent block; only the 2.31
import lacks it.

## Baseline Files Modified

### `src/include/os.h` — self-defaulting platform macro

**Change**: Before the platform-header selection, default the macro from the
compiler's own platform predefines when the build has not supplied one.

```c
#if !defined(NCCL_OS_LINUX) && !defined(NCCL_OS_WINDOWS)
#if defined(_WIN32) || defined(_WIN64)
#define NCCL_OS_WINDOWS 1
#else
#define NCCL_OS_LINUX 1
#endif
#endif
```

**Why in baseline**: the failure is in the header, so it has to be fixed in the
header. Propagating `-DNCCL_OS_LINUX` from Buck instead (via
`exported_preprocessor_flags`) would fix the Buck build only, and leave the
CMake and OSS Makefile builds — and any out-of-tree consumer of the exported
headers — with the same latent break.

**Difference from v2_29 / v2_30**: those versions `#define NCCL_OS_LINUX` with no
value. 2.31 introduced two bare `#if NCCL_OS_LINUX` / `#if NCCL_OS_WINDOWS`
tests (`src/include/socket.h`, and `os.h` itself) which require the macro to
expand to something. The valueless spelling produces
`error: expected value in expression` there, so this version defines it to `1`.
`1` is also what `-D<name>` yields, so a build that does pass the flag is
unaffected and the guard prevents any redefinition warning.

**Architecture note**: `_WIN32` / `_WIN64` are OS predefines, not architecture
predefines — MSVC defines them on ARM64 as well. aarch64 Linux (GB200 / GB300)
defines neither and correctly falls through to `NCCL_OS_LINUX`.

## Revert Checklist

To remove the platform-macro default from the baseline:

1. `src/include/os.h`: delete the `#if !defined(NCCL_OS_LINUX) && !defined(NCCL_OS_WINDOWS)` block.
2. Ensure every target that consumes the NCCLX header set defines
   `NCCL_OS_LINUX` (or `NCCL_OS_WINDOWS`) itself — for Buck, promote
   `-DNCCL_OS_LINUX` out of `compiler_flags` into an exported/propagated
   preprocessor flag in `def_build.bzl`.
