# CVARs row: param export visibility on v2_31

## Status: open. Port drafted in D117238276, not landed.

The CVARs row needs at most one baseline change on v2_31: restoring
`NCCL_PARAM_COMPILER_EXPORT_SYMBOL`. Everything else the row would have carried is already supplied
by upstream 2.31 or by D116807651. This document records the analysis so the 2.32 port does not
repeat it, and states the one question that decides whether the change is required.

## Scope of the row

The Feature Owners Map lists two files (`src/misc/param.cc`, `src/param/param_registry.cc`) and
says "only the two registry hooks port". Both halves are wrong. The row touches **eight** baseline
files, all with zero upstream drift between pristine 2.30 and 2.31:

`src/misc/param.cc`, `src/param/param.cc`, `src/param/param_registry.cc`, `src/param/ncclparam.cc`,
`src/param/c_api.cc`, `src/include/param/param.h`, `src/include/param/utils.h`,
`src/include/param/param_registry.h`.

Their combined raw delta against v2_31 is roughly 300 lines, but symbol-level comparison shows the
base already provides `NCCL_PARAM_IF_CONSTEXPR`, `ncclParamEnvPluginGet`,
`ncclParamIsCacheDisabled`, `srcDefault`/`srcEnvPlugin`, the `misc/param.cc` env-file loader, and
`ncclParamBind` plus the ENV-deprecation `INFO`. The remainder is formatting and signature drift,
including upstream's two-argument `ncclParamEnvPluginGet(key, env_init)` superseding v2_30's
one-argument form. Do not port that drift.

The single symbol absent from v2_31 is `NCCL_PARAM_COMPILER_EXPORT_SYMBOL`: 10 sites in v2_30, 0 in
v2_31.

## Why the attribute is not decorative

`#define NCCL_PARAM_COMPILER_EXPORT_SYMBOL __attribute__((visibility("default")))`.

The fbcode Buck build compiles the library with `-fvisibility=hidden`
(`nccl_build_config.bzl:135`, `v2_31/def_build.bzl:535`) and applies **no** linker version script
(no reference in `nccl_build_config.bzl`, `def_build.bzl`, or `comms/ncclx/BUCK`). So for the
Buck-built `libncclx.so`, a per-symbol attribute is the only export mechanism that exists.

Upstream does not need the attribute because its Makefile links with
`-Wl,--version-script=libnccl.map` (`v2_31/src/Makefile:171`), whose `global: nccl*;` prefix the
symbols match. v2_29/v2_30 instead link Meta's `src/version.script`
(`v2_30/src/Makefile:314`). **v2_31 has no `version.script` at all.** So the Buck path has neither
of the two mechanisms upstream assumes, which is why v2_30 added the attribute.

Measured on the Buck-built `libncclx.so.2.31`:

| | exported `ncclParam*` | total dynamic symbols |
| :---- | :---- | :---- |
| without the attribute (current v2_31) | 14 | 135884 |
| with D117238276 applied | 35 | 135905 |

The 14 are entirely the `NCCL_API`-decorated public C API from `c_api.cc`, which is unaffected
either way. The +21 delta is the 3 accessors plus 18 from `DEFINE_NCCL_PARAM` (9 invocations in
v2_31, two symbols each). Without the attribute, `ncclParamRegistryInstance`,
`ncclParamEnvPluginGet` and `ncclParamIsCacheDisabled` do not appear in `nm -D` at all.

## What upstream says the symbols are for

This matters because the mechanism is **symbol interposition**, not by-name calls, and upstream
states the intent in its own comments:

`include/param/param_registry.h`:

> Returns a process-wide RegistryState so map and mutex share identity across DSOs.

> The underlying state (RegistryState) is held behind a C-linkage accessor
> (ncclParamRegistryInstance) so that all DSOs in the process share a single map and mutex, even
> when NCCL is statically linked into multiple libraries.

> Parameters defined through the DEFINE_NCCL_PARAM macro are automatically registered here at
> program init (before main()). **This also works for DEFINE_NCCL_PARAM in external .so files,
> where the parameter is registered when the .so is loaded and initialized, or at dlopen().**

`param/param.cc`:

> **Exported** helper for `ncclParam<T>::loadValue()` **so plugins can resolve a single symbol**
> instead of requiring `ncclInitEnv` + `ncclEnvPluginGetEnv` to be exported.

External `.so` files registering params, and plugins resolving `ncclParamEnvPluginGet`, are
described as intended behaviour rather than hypotheticals.

## Why code search cannot settle this

A full-repo search found zero references to the three symbols outside `comms/ncclx/v2_*` and
`third-party/nccl` — including `comms/utils/cvars/`, `comms/ncclx/meta/`, and the `nccl4py`
bindings, with no `dlsym` call site naming them.

**That result is consistent with either answer and must not be read as "unused."** Interposition
produces no by-name references by construction: the linker resolves duplicate definitions across
DSOs to one, and nothing in source mentions it. Any future port that re-runs this search will get
the same empty result for the same reason.

Note also that `third-party/rccl/develop` matches on these names but is **not** a consumer. It is a
vendored AMD tree (imported in D116250835) carrying its own `utils.h`, `param.cc`,
`param_registry.cc` and `param.h`, linking into `librccl.so`.

## The open question

**Does a Meta process end up with more than one copy of the ncclx param registry?**

With one copy, hidden visibility is harmless: the accessor call resolves within the DSO. With two
or more, each gets a private registry, so a param registered in one is invisible to a lookup in
another, and the `"PARAM: Duplicate registration for key"` warning that would reveal it does not
fire either. The failure is silent and would not surface at build time.

The trigger condition exists in our tree. `//comms/torchcomms/ncclx:_comms_ncclx` is a
`cpp_python_extension` that statically links `//comms/ncclx:nccl` through `ncclx-api` and
`ncclx-global-api`, and it loads alongside the `//comms/torchcomms:_comms` extension in one Python
process. Separately, v2_30's `src/version.script` names `_comms_ncclx` explicitly as a cross-`.so`
caller for the Scuba error loggers, so cross-`.so` resolution from this extension is already an
established fact rather than a theory.

What is not established is whether the *param registry specifically* ends up duplicated there. The
way to settle it is `nm -D` on a built `_comms_ncclx.so`, checking whether it defines its own
`ncclParamRegistryInstance`.

A related consumer to keep in view: if the Meta tuner becomes a real plugin `.so` (the plan for the
tuner row) and declares params via `DEFINE_NCCL_PARAM`, it lands squarely in the "external .so"
case upstream describes.

## Decision guidance

- If the registry is duplicated: land D117238276. Absent it, v2_31 silently fragments the registry
  where v2_30 did not.
- If it is not duplicated: abandon D117238276 and keep this document. The attribute would then be
  genuinely inert, and the standing guidance to avoid porting applies.

Either way, do not justify the decision as "no consumers found." That reasoning is unsound for an
interposition-based symbol and would mislead the next port.

## Related, and NOT resolved by this decision

The OSS build path diverges independently of the attribute. v2_31 dropped Meta's `version.script`
in favour of upstream's `libnccl.map`, which also drops these entries:

```
*ctran*; *prims*; *cxa*;
*ErrorToScuba*;   /* called cross-.so from _comms_ncclx via CERR (CtranMapper.h) */
*getDevMemType*; *getCudaDevFromPtr*; *StreamCaptureModeGuard*;
```

and narrows `*nccl*` to `nccl*`, so mangled C++ symbols (which begin `_Z`) no longer match. The
`*cxa*` entry is load-bearing for folly's exception tracer, which resolves it via
`dlsym(RTLD_NEXT, "cxa...")` (see D64442615). `logCommErrorToScuba` is defined in
`comms/utils/logger/LoggingFormat.cc:315` and called from `comms/ctran/utils/CtranLogUtils.h:39`
and `comms/utils/logger/LogUtils.h:60`.

This affects only the Makefile/CMake path, since Buck applies no version script. Whether that path
is used for the 2.31 release is unconfirmed. `ErrorToScuba` belongs to the Scuba / Structured
Logging row, not to CVARs.

One stale entry noted in passing: `v2_30/src/libnccl.map:8` exports `ncclParamNoCacheSet*`, but no
such symbol exists in either tree.

## Also do not port

`src/misc/param.cc` in v2_30 carries a private copy of `loggerLevelToSpdlogLevel`. D116807651
deleted the v2_31 equivalent and routed the call to the shared
`comms/utils/logger/SpdlogLogger.h`. A raw v2_30-vs-v2_31 diff shows it as missing from 2.31;
porting it back reintroduces an ambiguous overload.

## Forward pointer: the two cvar systems

Upstream 2.31 supplies a native path that the NCCLX cvar framework currently duplicates.
`ncclParamEnvPluginGet` (`param/param.cc:47`) calls `ncclEnvPluginGetEnv`, which dispatches to an
env plugin implementing `ncclEnv_v2_t`, whose entire interface is
`getEnv(const char* name) -> const char*`. `include/param/param.h:151-156` invokes this inside the
param load chain, gated per-param by `NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT`.

`comms/utils/cvars/nccl_baseline_adapter.h` exposes `ncclGetEnvImpl(const char* name)`, the same
signature, but intercepts baseline env reads rather than registering a plugin. Serving cvar values
as an env plugin would let upstream call NCCLX natively and remove that interception. Not attempted
for 2.31; recorded here as the likely 2.32 direction.
