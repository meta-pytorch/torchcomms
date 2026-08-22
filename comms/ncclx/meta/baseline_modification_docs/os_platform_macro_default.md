# OS Platform Macro Default: Baseline Modifications

## Background

Several NCCL headers branch on `NCCL_OS_LINUX` / `NCCL_OS_WINDOWS`:

- `src/include/os.h` selects the platform implementation header (`os/linux.h` or
  `os/windows.h`), which is where `ncclSocketDescriptor` and `ncclAffinity` are
  declared.
- `src/nccl.h.in` gates the `ncclResetDebugInit` / `pncclResetDebugInit`
  declarations behind `#ifdef NCCL_OS_LINUX`.
- `src/include/socket.h` and `os.h` itself contain bare `#if NCCL_OS_LINUX` /
  `#if NCCL_OS_WINDOWS` tests.

If neither macro is defined, none of those resolve the way the code expects.
`os.h` includes neither platform header, so every declaration naming
`ncclSocketDescriptor` or `ncclAffinity` fails to compile, `socket.h` fails on
`struct ncclSocket`, and `alloc.h`'s `page_size` local becomes unused because
neither allocation branch is compiled.

Upstream expects the build system to supply the macro, and every in-tree build
does:

- Buck: `-DNCCL_OS_LINUX` in `NCCL_COMPILER_FLAGS` (`def_build.bzl`).
- CMake: `add_compile_definitions(NCCL_OS_LINUX)` at `CMakeLists.txt:81`,
  project-wide.
- Makefile: `CXXFLAGS += -DNCCL_OS_LINUX` at `makefiles/common.mk:31`, and
  `NVCUFLAGS += -DNCCL_OS_LINUX` at `:110`.

The gap is not a build that forgets the flag; it is a **consumer that gets the
headers without it**. In Buck, `compiler_flags` are private to the target that
declares them, so anything consuming only the NCCLX header set inherits the
headers and not the define:

- `ncclx-private-headers` / `nccl<ver>-internal`
- everything reached through them, notably `comms/testinfra/TestUtils.h`, which
  pulls in `bootstrap.h` -> `comm.h` -> `p2p.h` -> `core.h` -> `alloc.h` -> `os.h`
- any out-of-tree consumer of the installed/exported headers

The result is that the library itself builds while every generated `*_v2_31` meta
test that includes `TestUtils.h` fails to compile, on types that look unrelated
to anything the test does.

## Versions Affected

v2_31. `v2_29` and `v2_30` carry an equivalent block; only the 2.31 import lacks
it.

## Baseline Files Modified

### `src/nccl.h.in` — self-defaulting platform macro

**Change**: ahead of everything else in the header, default the macro from the
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

**Why in `nccl.h.in` and not `os.h`**: `os.h` includes `nccl.h` as its first
include, so by the time `os.h` is reached `nccl.h` has already been fully
processed behind its `NCCL_H_` guard and will never be reprocessed. A definition
placed in `os.h` therefore cannot affect `nccl.h`'s own `#ifdef NCCL_OS_LINUX`
gate around `ncclResetDebugInit` / `pncclResetDebugInit`, which would keep
evaluating false for exactly the header-set consumers this fixes. Defining it in
`nccl.h.in` covers the whole exported header set at once — `os.h`, `core.h`,
`alloc.h` and `socket.h` all include `nccl.h` first.

**Why in the baseline at all**: the failure is in the headers, so it has to be
fixed in a header. Propagating the Buck flag instead (via
`exported_preprocessor_flags`) would fix the Buck header-set consumers only, and
leave every out-of-tree consumer of the exported headers with the same latent
break.

**Difference from v2_29 / v2_30**: those versions `#define NCCL_OS_LINUX` with no
value. 2.31 introduced two bare `#if NCCL_OS_LINUX` / `#if NCCL_OS_WINDOWS` tests
(`src/include/socket.h`, and `os.h` itself) which require the macro to expand to
something. The valueless spelling produces `error: expected value in expression`
there, so this version defines it to `1`. `1` is also what `-D<name>` yields, so a
build that does pass the flag is unaffected and the guard prevents any
redefinition warning.

**Architecture note**: `_WIN32` / `_WIN64` are OS predefines, not architecture
predefines — MSVC defines them on ARM64 as well. aarch64 Linux (GB200 / GB300)
defines neither and correctly falls through to `NCCL_OS_LINUX`.

## Revert Checklist

To remove the platform-macro default from the baseline:

1. `src/nccl.h.in`: delete the `#if !defined(NCCL_OS_LINUX) && !defined(NCCL_OS_WINDOWS)` block.
2. Ensure every consumer of the NCCLX header set defines `NCCL_OS_LINUX` (or
   `NCCL_OS_WINDOWS`) itself — for Buck, promote `-DNCCL_OS_LINUX` out of
   `compiler_flags` into an exported/propagated preprocessor flag in
   `def_build.bzl`, and document the requirement for out-of-tree consumers.
