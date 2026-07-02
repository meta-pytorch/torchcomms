# NCCLX Built-in CSV/JSON Tuner

## Motivation

NCCL's algorithm/protocol selection comes from a built-in cost model. Overriding
it normally requires an external `NCCL_TUNER_PLUGIN` `.so` that NCCL discovers
via `dlopen`. Shipping and packaging a separate plugin `.so` is awkward for L4x
tuning workflows: every tuning iteration that only edits a cost table should not
force a binary/conda rebuild, and a `dlopen` plugin adds packaging surface.

The built-in tuner solves this by compiling a CSV/JSON-driven tuner directly into
`libnccl.so`. It overrides core `algorithm × protocol` selection, `nChannels`,
and (on tuner API v6 only) `chunkSize`, keyed by `(collective, per-rank size
range, topology)`. The topology key is `(nNodes,
nLocalRanks)` where `nLocalRanks = nRanks / nNodes` is ranks per node. The override table lives in a runtime
config file (kept in the L4x job's fbpkg, decoupled from the binary), so a tuning
loop is just "edit the file, repackage the fbpkg, restart" — `libnccl.so` is
never recompiled.

## User-Facing API

Set the cvar/env var `NCCLX_TUNER_CONFIG_FILE` to the absolute path of a CSV or
JSON config file. The format is auto-detected by extension (`.json` → JSON,
otherwise CSV):

```bash
export NCCLX_TUNER_CONFIG_FILE=/packaged/path/algo_override.csv
# Do NOT set NCCL_TUNER_PLUGIN.
```

Precedence and lifecycle:

- When `NCCLX_TUNER_CONFIG_FILE` is set, the built-in tuner takes **precedence**:
  it is wired into `comm->tuner` and the external `NCCL_TUNER_PLUGIN` `dlopen`
  path is skipped entirely.
- When the cvar is empty/unset, no tuner is wired in and NCCL's default
  cost-model selection is used (zero regression). An external
  `NCCL_TUNER_PLUGIN`, if present, still loads as usual.
- The config file is read once per communicator at init; the cvar is
  process-global, so every comm reads the same table.
- A **set-but-malformed** config fails comm init by default. Set
  `NCCLX_TUNER_IGNORE_CONFIG_ERRORS=1` to instead log and skip the bad
  rule(s) / ignore the bad file and continue. See "Per-comm-init parsing,
  process-global table" for the full policy.

Observability — run with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING` to see the
`NCCLX TUNER:` log lines: the wiring message, the per-rule match (selected
algo/proto/channels), and any chunkSize override.

## Architecture

### Data Flow

```
ncclTunerPluginLoad(comm)                          // src/plugin/tuner.cc [META]
  comm->tuner = nullptr
  ncclx::tuner::tryLoadMetaTuner(comm)
    metaTunerEnabled()  ->  NCCLX_TUNER_CONFIG_FILE non-empty?
      yes -> comm->tuner = &ncclx::tuner::kMetaTuner   (static struct, NOT dlopen)
              leave comm->tunerPluginLoaded == 0       (Unload skips dlclose)
              return true  -> caller skips external NCCL_TUNER_PLUGIN path
      no  -> return false  -> fall through to external plugin dlopen

comm init (src/init.cc, guarded by comm->tuner != NULL)
  -> kMetaTuner.init(&ctx, ...)                    // metaTunerInit
       allocate MetaTunerContext (heap)
       loadConfig(NCCLX_TUNER_CONFIG_FILE)
         .json -> loadConfigJson (folly)  else -> loadConfigCsv (zero-dep)
       *ctx = context

per collective (src/enqueue.cc, guarded by comm->tuner != NULL)
  -> kMetaTuner.getCollInfo(ctx, collType, nBytes, ...) // metaTunerGetCollInfo
       AND-match each rule on collType/bytesPerRank/nNodes/nLocalRanks
       (bytesPerRank/nNodes/nLocalRanks are Int64Range matchers with
        `*` = wildcard; bytesPerRank matches nBytes/nRanks, nRanks =
        nNodes*nLocalRanks)
       first match: costTable[algo][proto] = 0.0 (skip if NCCL_ALGO_PROTO_IGNORE)
                    if nChannels != -1, override *nChannels

  -> kMetaTuner.getChunkSize(ctx, collType, nBytes, algo, proto, ...) // v6 only
       AND-match rule with chunkSize != 0 AND algorithm==algo AND protocol==proto
       first match: *chunkSize = rule.chunkSize  (core clamps to bufferMaxChunkSize)

comm finalize (src/init.cc, guarded by comm->tuner != NULL)
  -> kMetaTuner.finalize(ctx)                      // metaTunerFinalize: delete ctx
```

### File Organization

All tuner logic lives in `meta/tuner/`, shared across NCCLX versions:

| File | Role |
|------|------|
| `meta/tuner/MetaTuner.h` | `TuningConfig` rule struct; `metaTunerEnabled()`, `tryLoadMetaTuner(comm)`, `extern const ncclTuner_t kMetaTuner` |
| `meta/tuner/MetaTuner.cc` | CSV + JSON parsers, the tuner callbacks (`init`/`getCollInfo`/`finalize`/`getChunkSize`), and `kMetaTuner` definition |
| `meta/tuner/Int64Range.{h,cc}` | Self-contained `Int64Range` interval matcher (`bytesPerRank`/`nNodes`/`nLocalRanks`) and `parseInt64Range`; no NCCL/folly dependency |
| `meta/tuner/tests/MetaTunerTest.cpp` | Unit tests (CSV/JSON parsing, match semantics, version gating) |
| `meta/tuner/tests/Int64RangeTest.cpp` | `int64_range_ut` (an `ncclx_meta_unittest` that links the nccl-internal lib) for `Int64Range` parse/match edge cases; no GPU |
| `meta/tuner/tests/example_tuner_config.{csv,json}` | Example configs |

The only baseline change is a single `[META]`-tagged hook in
`src/plugin/tuner.cc`. Details are in
`meta/baseline_modification_docs/builtin_tuner.md`.

### Key Design Decisions

**1. Compiled-in, not a dlopen plugin (explicit wiring required)**

NCCL discovers tuners only via `dlopen` + `ncclDlsym` on an external `.so`. A
tuner compiled into `libnccl.so` is therefore not auto-discovered. We explicitly
take the address of the static `kMetaTuner` struct and assign it to
`comm->tuner` from `ncclTunerPluginLoad`. We do NOT set `NCCL_TUNER_PLUGIN`. The
fallback leaves `comm->tunerPluginLoaded == 0` so `ncclTunerPluginUnload` skips
`dlclose` on the static struct; `finalize` still runs (it only guards on
`comm->tuner != NULL`) and frees the per-comm heap context.

**2. Built-in takes precedence over external plugin**

`tryLoadMetaTuner` runs first in `ncclTunerPluginLoad`. If the cvar is set it
wires the built-in tuner and returns early, so the external-plugin `dlopen` path
is never reached. This keeps semantics simple: one knob (`NCCLX_TUNER_CONFIG_FILE`)
fully determines whether the built-in tuner is active.

**3. Per-comm-init parsing, process-global table**

`metaTunerInit` allocates a heap `MetaTunerContext` per comm and parses the file
into a `std::vector<TuningConfig>`. The cvar is process-global so every comm
parses the same content.

An **empty/unset** `NCCLX_TUNER_CONFIG_FILE` means the tuner is disabled — no
parse happens, the context stays empty, and init succeeds (zero regression).
This is never an error.

A **set-but-malformed** config, by default, **fails communicator init**:
`loadConfig` returns `ncclInvalidUsage` and `metaTunerInit` propagates it
*before* publishing `*ctx`, so the `unique_ptr` frees the context and the
existing `NCCLCHECK(comm->tuner->init(...))` in core `init.cc` fails comm init.
Fatal conditions: the file cannot be opened; `folly::parseJson` failure;
missing/non-array `rules`; a `.json` path in a no-folly build; a rule missing
its `filter`/`config`; an unknown collective/algorithm/protocol; an invalid
interval; a present-but-invalid numeric; or a bad JSON value type (whose
`asInt()` would otherwise throw — caught per-rule so it can never escape).

Setting `NCCLX_TUNER_IGNORE_CONFIG_ERRORS=1` downgrades this to the legacy
"log-and-continue" behavior: file-level problems log a WARN and yield an empty
(no-override) table; a bad rule logs an ERROR and is skipped while the valid
rules still load; init returns success. `strict = !NCCLX_TUNER_IGNORE_CONFIG_ERRORS`
is threaded from `metaTunerInit` through `loadConfig` into both `loadConfigCsv`
and `loadConfigJson` (the folly path and the no-folly stub).

**4. First match wins (row order = priority)**

Both `getCollInfo` and `getChunkSize` iterate rules in file order and return on
the first AND-match. A wildcard field matches any value: `*` for the
`bytesPerRank`/`nNodes`/`nLocalRanks` (`Int64Range`) fields.
`getCollInfo` never forces an `algo×proto` combo that core marked
`NCCL_ALGO_PROTO_IGNORE`.

**5. chunkSize via the v6 `getChunkSize` hook (v2.30 only)**

`getChunkSize` receives the already-selected `algo`/`proto`, so it is the only
knob that can set chunk size per `algo×proto` and per-comm. A rule contributes a
chunkSize override only when its `chunkSize != 0` AND its `algorithm`/`protocol`
equal the hook's arguments. Core clamps the result to `bufferMaxChunkSize`, so
lowering always works but raising above the buffer ceiling is silently clamped —
enlarge `NCCL_BUFFSIZE` (or per-comm `ncclxConfig.ncclBuffSize`) to make a larger
chunk effective.

**6. Folly-JSON isolation**

CSV is the always-available, zero-dependency baseline parser (compiled in fbcode
and OSS). JSON parsing uses `folly::parseJson` and is gated behind the compile
macro `NCCLX_TUNER_WITH_FOLLY_JSON`. In a build without folly (e.g. the OSS
Makefile, or a future no-folly conda), `loadConfigJson` compiles to a stub that
rejects `.json`: by default this fails comm init (with
`NCCLX_TUNER_IGNORE_CONFIG_ERRORS=1` it logs a warning and applies no overrides).
The CSV path and the public tuner API are unchanged. Dropping folly later does
not force an API/ABI churn.

## Config Formats

A rule matches a collective when every populated field AND-matches. The first
matching rule wins, so rule order encodes priority.

### Interval grammar (`bytesPerRank`, `nNodes`, `nLocalRanks`)

`bytesPerRank`, `nNodes`, and `nLocalRanks` are `Int64Range` matchers backed by
`meta/tuner/Int64Range.{h,cc}` (a self-contained `int64_t` interval type with no
NCCL/folly dependency, so it covers GB-scale byte values and is unit-tested by
the `int64_range_ut` `ncclx_meta_unittest` target). They share one value grammar
(leading/trailing whitespace tolerated; `[` `]` inclusive, `(` `)` exclusive; an
empty bound = unbounded on that side):

| Form | Meaning |
|---|---|
| `*` | wildcard — matches any |
| `N` (e.g. `5`, `1048576`) | exact, `== N` |
| `[a,b]` | `a <= n <= b` |
| `(a,b)` | `a < n < b` |
| `(a,b]` / `[a,b)` | half-open |
| `(1,)` | `n > 1` (upper unbounded) |
| `[2,)` | `n >= 2` |
| `(,4]` | `n <= 4` (lower unbounded) |
| `(,4)` | `n < 4` |

`parseInt64Range` returns `nullopt` on any parse error (empty, non-numeric bound,
unbalanced bracket, missing comma, or `lo > hi` such as `(2,1]`). A rule with a
bad interval is a config error: by default it is logged at **ERROR** (naming the
offending value / line) and **fails comm init**. With
`NCCLX_TUNER_IGNORE_CONFIG_ERRORS=1` it is logged and skipped while the remaining
valid rules still load (see "Per-comm-init parsing").

### Per-rank size filter (`bytesPerRank`)

`bytesPerRank` is the single size field. It matches the **per-rank shard**
`nBytes / nRanks`, where `nRanks = nNodes * nLocalRanks` (both captured in
`metaTunerInit` from the comm, so `nRanks` always reflects the real topology even
when a rule leaves `nNodes` / `nLocalRanks` as the `*` wildcard). A guard sets
`nRanks = 1` if the product is non-positive, so the division is always safe, and
a wildcard `bytesPerRank` matches any size.

Why per-rank: for a fixed per-rank workload, the LL128→Simple crossover is roughly
**constant in per-rank bytes** but scales linearly in *total* bytes with rank
count. Keying on `bytesPerRank` lets **one** rule span every rank count,
collapsing the per-topology rule explosion (e.g. the GB200 config goes from 20
total-byte rules to ~4 per-rank rules; see the `nccl-tuner-from-sweep` skill's
`gen --per-rank`).

Best suited to **AllGather / ReduceScatter**: NCCL passes those collectives
`nBytes = nRanks × count × eltsize` (the rank-scaled total), so `nBytes / nRanks`
is exactly the per-rank shard. For **AllReduce** `nBytes` is already the full
buffer (not scaled by ranks), so `bytesPerRank` is `buffer / nRanks` — author
AllReduce rules with that in mind.

Combining fields: every populated filter is **AND-matched**, so `bytesPerRank`
intersects with any other set field. Setting only `bytesPerRank` (the common
case) matches purely on per-rank size across all topologies; additionally pinning
`nNodes` / `nLocalRanks` narrows the rule to a specific topology.

### CSV (zero-dependency, always available)

```
collective,bytesPerRank,algorithm,protocol,channels,nNodes,nLocalRanks,chunkSize
```

- `collective`: `allreduce` / `broadcast` / `reduce` / `allgather` / `reducescatter`
- `bytesPerRank`: per-rank-size Int64Range (`nBytes / nRanks`, interval / exact /
  `*` wildcard); see "Per-rank size filter"
- `algorithm`: `tree` / `ring` / `collnet_direct` / `collnet_chain` / `nvls` / `nvls_tree` / `pat`
- `protocol`: `ll` / `ll128` / `simple`
- `channels`: `-1` keeps the NCCL default; any other value overrides `*nChannels`
- `nNodes`: number of nodes (Int64Range)
- `nLocalRanks`: ranks per node (`nRanks / nNodes`) (Int64Range)
- `chunkSize` (bytes): `0` means no override
- `chunkSize` column is optional; at least the first 7 columns (through
  `nLocalRanks`) are required.
- The CSV split is **bracket-aware**: it tracks `()`/`[]` nesting and only splits
  on commas at depth 0, so an interval like `[0,1048576]` or `(1,)` stays a
  single column
- Lines beginning with `#` and blank lines are ignored

Example (small allreduce forced to ring + ll128):

```
allreduce,[0,1048576],ring,ll128,-1,*,*
```

### JSON (folly-enabled build only)

Each rule is split into two nested objects: `filter` holds the match conditions
and `config` holds the overrides applied on a match.

```json
{
  "rules": [
    {
      "filter": { "collective": "allgather", "bytesPerRank": "[0,2097152]", "nNodes": 2, "nLocalRanks": 8 },
      "config": { "algorithm": "tree", "protocol": "simple", "channels": 4 }
    }
  ]
}
```

This rule is topology-specific: it only matches an allgather on a 2-node
communicator with 8 ranks per node (`nNodes` / `nLocalRanks` filters), and also
overrides `channels`. Rules that should apply regardless of topology simply omit
`nNodes` / `nLocalRanks` from `filter` (an omitted filter field is the `*`
wildcard).

- `filter` holds the match conditions: `collective`, `bytesPerRank`, `nNodes`,
  `nLocalRanks`. Only `collective` is required;
  omitting any other filter field means wildcard / any.
- `config` holds the overrides: `algorithm`, `protocol`, `channels`,
  `chunkSize`. `algorithm` and `protocol` are required; `channels` and
  `chunkSize` are optional and omitting them means "no override".
- `bytesPerRank` / `nNodes` / `nLocalRanks` may be a JSON integer (`N`
  exact), the string `"*"` (wildcard), OR an interval string (`"[0,1048576]"`,
  `"(1,)"`); a
  bad interval string, an unknown enum, or a bad value type (`"channels": [4]`
  / `"abc"`) is a config error, handled per the policy in "Per-comm-init
  parsing" (default: fail comm init; with the ignore cvar: log + skip the rule),
  mirroring CSV. Bad value types are caught per-rule so a `folly::TypeError`
  never escapes `loadConfigJson`.
- JSON and an equivalent CSV parse to identical `TuningConfig` values; the
  `rules` array order = priority. The flat (un-nested) form is no longer
  accepted.
- In a build without folly, a `.json` path is a config error: by default it
  fails comm init; with `NCCLX_TUNER_IGNORE_CONFIG_ERRORS=1` it is rejected with
  a warning and no overrides are applied — convert to CSV first.

See `meta/tuner/tests/example_tuner_config.csv` and `.json` for fuller examples,
and `meta/tuner/README.md` for the user-facing reference.

## Versions

The tuner is wired into both NCCLX versions; the shared `meta/tuner/` sources are
identical, with version differences handled by compile macros:

| Version | Tuner API | `getChunkSize` | Macro |
|---------|-----------|----------------|-------|
| v2.29 | v5 | not available | — |
| v2.30 | v6 | available | `NCCLX_TUNER_HAS_GETCHUNKSIZE` |

- The `init` callback's versioned parameter types (`ncclNvlDomainInfo_*`,
  `ncclTunerConstants_*`) differ between v5 and v6; `MetaTuner.cc` selects the
  right aliases under `NCCLX_TUNER_HAS_GETCHUNKSIZE`, since v2.29 does not even
  define the `_v6_t` names.
- The `kMetaTuner` struct omits the `getChunkSize` field on v2.29 (it is the last
  member, present only on v6).
- `NCCLX_TUNER_WITH_FOLLY_JSON` is defined in the buck build (folly present) and
  undefined in the OSS Makefile.

## Testing

**Unit tests** (`meta/tuner/tests/MetaTunerTest.cpp`, no GPU, OSS-safe). Set
`NCCLX_TUNER_CONFIG_FILE` via `setenv` then re-read with `ncclCvarInit()`:

- CSV parsing: multiple rows incl. comments / blank lines / omitted columns →
  verify rule count and fields (`*` / `-1` wildcard placement).
- JSON parsing & equivalence (folly build only): a `rules` array parses to the
  same `TuningConfig` vector as the equivalent CSV; JSON in a no-folly build
  yields a clear error and empty table.
- `getCollInfo` semantics: size-range match zeroes `table[algo][proto]`; topology
  filter (`nNodes`/`nLocalRanks`); interval matching (open-ended / endpoint
  inclusivity / wildcard) on `bytesPerRank`/`nNodes`/`nLocalRanks`; bracket-aware CSV
  split; bad-interval rule skipped; row-order priority; `nChannels` override;
  `NCCL_ALGO_PROTO_IGNORE` protection; missing/empty file leaves outputs
  untouched.
- `Int64Range` parse/match edge cases are covered standalone in
  `meta/tuner/tests/Int64RangeTest.cpp` (`int64_range_ut`, an `ncclx_meta_unittest` linking the nccl-internal lib; no GPU): wildcard / exact / closed / open / half-open / open-ended
  intervals, endpoint inclusivity, 64-bit GB-scale values, whitespace tolerance,
  and invalid inputs (incl. `lo > hi`) returning `nullopt`.
- `getChunkSize` (v6 only): a matching row with `chunkSize != 0` sets
  `*chunkSize`; rows differing only in `(algo, proto)` override only when those
  equal the hook's arguments.
- Gate: empty cvar → `metaTunerEnabled() == false`.

**Integration / E2E** (GPU):

- Zero-regression gate: without the cvar, existing collective dist tests match
  baseline and `comm->tuner == NULL`.
- Effect verification: a config forcing a collective/size range to a specific
  algo/proto; `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING` logs confirm the
  `NCCLX TUNER:` wiring + match lines and the actually-selected algo/proto.
- Precedence: with both `NCCL_TUNER_PLUGIN` and the cvar set, the built-in tuner
  wins.
