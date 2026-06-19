# Built-in Meta Tuner: Baseline Modifications

## Background

The built-in Meta tuner is a CSV/JSON-driven NCCL tuner compiled directly into
`libnccl.so` (not a separate `dlopen` plugin `.so`). It overrides NCCL core
`algorithm × protocol` selection, `nChannels`, and (on tuner API v6) `chunkSize`
from a runtime config file pointed to by the `NCCLX_TUNER_CONFIG_FILE` cvar. Its
implementation lives entirely in `meta/tuner/` (see
`meta/design_docs/builtin_csv_json_tuner.md`). It needs exactly one baseline hook
because NCCL discovers tuners only via `dlopen` + `ncclDlsym` on an external
`.so` — a tuner compiled into `libnccl.so` is never auto-discovered, so it must
be explicitly wired into `comm->tuner` from the tuner-load path.

## Versions Affected

v2.29, v2.30

## Baseline Files Modified

Exactly ONE baseline file per version: `src/plugin/tuner.cc`. The change is two
parts in `ncclTunerPluginLoad`: an `#include` and a `[META]`-tagged early
fallback that wires the built-in tuner when the cvar is set.

```cpp
#include "meta/tuner/MetaTuner.h"
```

```cpp
ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm) {
  const char* tunerName;
  // Initialize to nullptr by default if plugin tuner cannot be loaded.
  comm->tuner = nullptr;
  // [META] built-in CSV/JSON tuner takes precedence when set; see meta/tuner/MetaTuner.h
  if (ncclx::tuner::tryLoadMetaTuner(comm)) {
    return ncclSuccess;
  }
  // ... unchanged external NCCL_TUNER_PLUGIN dlopen path ...
}
```

Net footprint is tiny: ~5 lines (1 include + 1 `[META]` comment + a 3-line
`if`-block). When the cvar is empty, `tryLoadMetaTuner` returns `false` and the
function proceeds exactly as upstream (zero regression).

All OTHER tuner call sites are UNCHANGED, because they already guard on
`comm->tuner != NULL`:

| Call site | File | Status |
|-----------|------|--------|
| `init` | `src/init.cc` | unchanged (guarded) |
| `getCollInfo` | `src/enqueue.cc` | unchanged (guarded) |
| `getChunkSize` | `src/enqueue.cc` (v2.30 only) | unchanged (guarded) |
| `finalize` | `src/init.cc` | unchanged (guarded) |
| `ncclTunerPluginUnload` | `src/plugin/tuner.cc` | unchanged (gated on `tunerPluginLoaded`) |

## Why in baseline

NCCL's tuner discovery is `dlopen`-only: `ncclTunerPluginLoad` opens an external
`.so` and resolves the tuner symbol via `ncclDlsym`. A tuner compiled into
`libnccl.so` has no `.so` to open and no symbol to discover, so the address of
the statically linked `kMetaTuner` struct must be assigned to `comm->tuner`
explicitly from this function — there is no other entry point. The hook runs
before the external-plugin path so the built-in tuner takes precedence when the
cvar is set.

The fallback intentionally leaves `comm->tunerPluginLoaded == 0`. Because
`ncclTunerPluginUnload` gates its `dlclose` on that flag, Unload naturally skips
`dlclose` on the static struct — no Unload change is needed. `finalize` still
runs (it only checks `comm->tuner != NULL`) and frees the per-comm heap context.

## Not baseline files

The build glob (`def_build.bzl` / `src/Makefile`) and all `meta/tuner/` sources
are Meta-only additions, not baseline NCCL files. `getChunkSize` support is
v2.30-only (tuner API v6); v2.29 uses tuner API v5 and has no `getChunkSize`
hook, gated in the shared sources by `NCCLX_TUNER_HAS_GETCHUNKSIZE`.

# Design details

The detailed design document for the NCCLX built-in tuner lives next to the tuner
sources. See it for the motivation, architecture and data flow, key design
decisions, version handling, config-format reference, and testing strategy.
[`comms/ncclx/meta/tuner/docs/design.md`](../tuner/docs/design.md)

The user-facing quick reference is
[`comms/ncclx/meta/tuner/README.md`](../tuner/README.md).
