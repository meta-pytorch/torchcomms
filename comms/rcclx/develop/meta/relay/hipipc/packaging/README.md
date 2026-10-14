# Standalone HIPIPC Relay Artifact

The optional leaf wheel contains the prebuilt shared library and a structured
hash manifest. Its package imports without loading HIP and declares no Python
runtime dependencies. No native sources, uploads or runtime replacements.

## Package A Prebuilt Library

In the public torchcomms repository, canonical device code remains under
`comms/rcclx/develop/meta/relay`. Standalone host/probe sources and isolated build
targets are under `comms/rcclx/develop/meta/relay/hipipc`.

From the public repository root, package an already-built library with:

```bash
python3 comms/rcclx/develop/meta/relay/hipipc/packaging/build_hipipc_relay_wheel.py \
  --library /absolute/resolved/libhipipc_relay.so \
  --output-dir /absolute/scratch \
  --version 0.1.0+abi1.source
```

Both artifact paths must be absolute. The builder does not compile the library.

### Meta-Internal Build

The following Buck commands require a Meta-internal checkout and toolchain. From
its fbsource repository root, build the library and use its resolved output path
as the builder's `--library` argument:

```bash
buck2 build @fbcode//mode/opt-amd-gpu -m rocm_latest \
  fbcode//comms/rcclx/develop/meta/relay/hipipc:hipipc_relay \
  --show-full-json-output

python3 fbcode/comms/rcclx/develop/meta/relay/hipipc/packaging/build_hipipc_relay_wheel.py \
  --library /absolute/resolved/libhipipc_relay.so \
  --output-dir /absolute/scratch \
  --version 0.1.0+abi1.source
```

The selected HIP compiler is a build-time dependency only. Isolated genrules
avoid GPU-library macros that inject a mandatory Meta lazify runtime. The leaf
requires `HIP_UNCACHED_MEMORY`: gfx950 protocol flags and helper rings must use
uncached allocation, not a fine-grained fallback. Verify
compiler/runtime compatibility on the intended gfx950 environment; a compiled
device image or ABI-version call alone is not GPU correctness evidence.

The builder requires an exact `libhipipc_relay.so` SONAME, ELF64 x86-64, and twelve
defined, externally visible function exports. It rejects RCCL/NCCL/Torch/lazify
dependencies, registered RCCLX API references and build-tree runtime paths.
Verification uses `readelf` without loading the library or initializing a GPU.

## Contents And Serving

The `py3-none-linux_x86_64` wheel holds the `hipipc_relay` package, its data library,
and `METADATA`, `WHEEL`, and CSV/SHA256 `RECORD`. Existing files are never
silently overwritten. The stdlib builder's declared-dependency CPU tests run in
a Meta-internal checkout with:

```bash
buck2 test fbcode//comms/rcclx/develop/meta/relay/hipipc/packaging/tests:wheel_builder_test
```

Install the versioned artifact only in the opted-in image. The Python adapter
resolves its library through `importlib.resources`, or an explicit absolute
`SGLANG_HIPIPC_RELAY_LIBRARY` override. Override selection takes precedence;
serving never discovers a checkout or Buck output.

The native library needs the existing HIP major ABI `libamdhip64.so.7` and
standard C/C++ libraries. Torch's already-loaded compatible HIP runtime supplies
that dependency. No separate Meta binder, embedded SDK search path, torchcomms,
Torch C++ linkage, RCCLX registered symbols or librccl swap is required.
Validate the exact installed artifact, topology, shader probes, full-payload
arithmetic and quiescent teardown before enabling a serving recipe.
