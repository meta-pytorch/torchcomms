# HIPIPC Relay Resources

The data-only `hipipc_relay` package supplies a native TP2 BF16 relay library,
its canonical kernel archive, and four required headers. Importing it does not
load HIP, Torch, RCCL, or any native code. The typed binding compiles separately.

The installed API is `__version__`, `ABI_VERSION`, `get_include()` and
`get_library_dir()`. The library directory contains the shared object and
archive; the include directory preserves the canonical `comms/` header layout.
The six native payloads are the library, archive, workspace header, its required
record header, argument header and kernel-launch header. SDK libraries are not
bundled; the record header is needed because the workspace header includes it.

## Fresh ROCm 7.14.1 Build

The USER builds the wheel after the reviewed native stack is committed:

```bash
fbpython -B /data/users/srinathb/fbsource/fbcode/comms/rcclx/develop/meta/relay/hipipc/packaging/build_rocm714_wheel.py
```

The script locates its own checkout and derives the version from its full source
HEAD. Keep source fixed throughout the build. Prerequisites are Linux x86-64,
Python 3.12 with venv/ensurepip, Sapling, readelf, patchelf, compatible host C++
headers/libraries, HTTPS access to AMD/PyPI, and several GB of disk space.
No OS packages or existing serving environments are modified.

Seven SHA256-pinned downloads supply ROCm 7.14.1 core/devel/libraries/gfx950,
the ROCm SDK CLI-provider sdist, setuptools 81.0.0 and packaging 26.2.
Only downloads are reused; every run creates its own SDK venv. Older verified
downloads remain usable; legacy SDK venv/readiness/lock state is ignored.
Installation uses local verified bytes, no index, no dependency resolution, no pip cache,
and no build isolation for the SDK CLI-provider sdist.

Options: `--cache-dir` selects dedicated storage; `--offline` requires all seven
verified downloads already cached; `--output-dir` must name a nonexistent
directory; `--base-version` defaults to `0.1.0`. Existing output directories,
including empty directories and dangling symlinks, are refused before SDK work.
Default outputs and retained command logs live in fresh per-build directories.

Compilation pins C++17/O3/gfx950/PIC/hidden visibility and libstdc++ C++11 ABI 1.
The canonical kernel is rebuilt, archived with SDK llvm-ar, and linked with the
host/probe objects and amdhip64. RPATH is removed. A public-header consumer is
compiled and linked but never executed. Packaging performs no GPU tests.

## Buck Bundle And Offline Packaging

The native bundle target is
`fbcode//comms/rcclx/develop/meta/relay/hipipc:hipipc_relay`; its output is a
directory, not a shared-library filename. Buck's selected ROCm toolchain is not
the pinned 7.14.1 script toolchain. Use the measured build identity when packaging.

The low-level packager accepts six required flags: `--native-dir`, `--output-dir`,
`--version`, `--source-commit`, `--rocm-version`, `--architecture gfx950`.
Directories must be absolute; the source commit is full 40-hex. The version is
`X.Y.Z+abi1.<source-prefix>` with a matching 10-40 lowercase-hex prefix.
It neither compiles nor installs anything. Existing wheel files are never replaced.

## ABI And Validation

The default filename is `hipipc_relay-0.1.0+abi1.<commit10>-py3-none-linux_x86_64.whl`.
The header's `kAbiVersion = 1`, exported `hipipc_relay_abi_version()` and installed
`ABI_VERSION = 1` identify the single native ABI. A changed native payload needs
a new USER-built immutable version; do not relabel an older artifact.

Packaging checks materialized payloads, regular archive magic, ELF64/x86-64,
SONAME, a visible defined ABI getter, forbidden runtime dependencies/registered
references, and any RPATH/RUNPATH. `manifest.json` records build-time
payload hashes and source/build identity for audit only, not runtime verification.
Standard wheel metadata and CSV RECORD hash every member.

CPU fixture tests cover packaging contracts and builder refusals; they do not
establish SDK build compatibility, native correctness, or GPU acceptance.
After artifact handback, record the wheel path/SHA256 and validate the matched
native/binding ABI separately. This tooling does not install or publish the wheel.

The ROCm 7.1.4 builder requires clean source and packaging inputs. The manifest
records payload hashes and `source_commit`; the preserved `source/` snapshot
supports source-input comparisons. Rebuild after any payload change.

Native TP2 GPU modes remain disabled in CI until an exclusive eight-gfx950 pool
is provisioned. Run them manually with `HIPIPC_TEST_EXCLUSIVE_GFX950_POOL` set to
the eight assigned device ordinals; CPU and packaging tests remain enabled.
