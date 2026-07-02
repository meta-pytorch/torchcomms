# NCCLX Built-in CSV/JSON Tuner

`meta/tuner` is a tuner that is compiled directly into `libnccl.so` (no separate
plugin `.so`). It overrides NCCL core algorithm/protocol selection, the number
of channels, and the per-collective chunk size based on a runtime config file.

## Enabling

Set the cvar/env var `NCCLX_TUNER_CONFIG_FILE` to the absolute path of a CSV or
JSON file:

```bash
export NCCLX_TUNER_CONFIG_FILE=/path/to/algo_override.csv
```

Precedence rules:

- When `NCCLX_TUNER_CONFIG_FILE` is set, the built-in CSV/JSON tuner takes
  precedence and the external `NCCL_TUNER_PLUGIN` dlopen path is skipped.
- If `NCCLX_TUNER_CONFIG_FILE` is empty/unset, no tuner is wired in and NCCL's
  default cost-model selection is used (zero regression).
- The config file is read once per communicator at init; one table is shared by
  all comms in the process.

## Config-error policy

By default, a **set-but-malformed** config **fails communicator init**. Any of
the following is fatal: the file cannot be opened; malformed JSON; a missing or
non-array `rules`; a `.json` path in a no-folly build; a rule missing its
`filter`/`config` object; an unknown collective/algorithm/protocol token; an
invalid interval; a present-but-invalid numeric value; or a bad JSON value type
(e.g. `"channels": [4]` or `"channels": "abc"`). The offending file/rule is
logged at ERROR (saying why), and init returns `ncclInvalidUsage`.

An empty/unset `NCCLX_TUNER_CONFIG_FILE` is **not** an error — it simply
disables the tuner (zero regression).

Set `NCCLX_TUNER_IGNORE_CONFIG_ERRORS=1` to downgrade errors: file-level
problems log a WARN and yield an empty (no-override) table; a bad rule logs an
ERROR and is skipped while the remaining valid rules still load. Init then
returns success.

## Config formats

A rule matches a collective when every populated field AND-matches. The first
matching rule wins, so rule order encodes priority.

### Interval grammar (`bytesPerRank`, `nNodes`, `nLocalRanks`)

These three fields are `Int64Range` matchers and share a single value grammar
(leading/trailing whitespace tolerated; `[` `]` inclusive, `(` `)` exclusive; an
empty bound means unbounded on that side):

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

An invalid interval (empty, non-numeric bound, unbalanced bracket, missing comma,
or `lo > hi` such as `(2,1]`) is a config error: by default it fails comm init
(see "Config-error policy"). With `NCCLX_TUNER_IGNORE_CONFIG_ERRORS=1` the
single rule is logged at ERROR and skipped while remaining valid rules load.

### CSV (zero-dependency, always available)

```
collective,bytesPerRank,algorithm,protocol,channels,nNodes,nLocalRanks,numPipeOps,regBuff,chunkSize
```

- `collective`: `allreduce` / `broadcast` / `reduce` / `allgather` /
  `reducescatter`
- `bytesPerRank`: per-rank-size Int64Range (`nBytes / nRanks`, interval / exact /
  `*` wildcard); see "Per-rank size filter"
- `algorithm`: `tree` / `ring` / `collnet_direct` / `collnet_chain` / `nvls` /
  `nvls_tree` / `pat`
- `protocol`: `ll` / `ll128` / `simple`
- `channels`: `-1` keeps the NCCL default; any other value overrides it
- `nNodes`: number of nodes (Int64Range)
- `nLocalRanks`: ranks per node (`nRanks / nNodes`) (Int64Range)
- `numPipeOps` / `regBuff`: exact int, `-1` is a wildcard
- `numPipeOps`, `regBuff`, `chunkSize` columns are optional (in that order)
- The CSV split is bracket-aware: commas inside `()` or `[]` are part of an
  interval and do NOT separate columns (so `[0,1048576]` is one field)
- Lines beginning with `#` and blank lines are ignored

### Per-rank size filter (`bytesPerRank`)

`bytesPerRank` matches the **per-rank shard** `nBytes / nRanks`
(`nRanks = nNodes * nLocalRanks`, taken from the comm — so it is correct even when
`nNodes` / `nLocalRanks` are left wildcard). Because the LL128→Simple crossover is
~constant in per-rank bytes (but scales with rank count in total bytes), one
`bytesPerRank` rule can span every rank count — collapsing a per-topology rule
explosion into a few rules.

- Best suited to **allgather / reducescatter**: NCCL passes those the rank-scaled
  total `nBytes = nRanks × count × eltsize`, so `nBytes / nRanks` is exactly the
  per-rank shard. For **allreduce**, `nBytes` is the full buffer (not rank-scaled),
  so `bytesPerRank` is `buffer / nRanks` — keep that in mind when writing allreduce
  rules.
- All populated fields are AND-matched: setting only `bytesPerRank` matches on
  per-rank size across all topologies; pinning `nNodes` / `nLocalRanks` too narrows
  it to a specific topology.
- `gen --per-rank` in the `nccl-tuner-from-sweep` skill emits these rules.

Example (per-rank): force ring + ll128 for any allgather whose per-rank shard is
8–16 MiB, on any topology (`nNodes` / `nLocalRanks` left wildcard):

```
allgather,[8388608,16777216],ring,ll128,-1,*,*
```

Example (2 nodes, 8 ranks/node, 1–256 MiB per-rank allreduce forced to ring +
ll128):

```
allreduce,[1048576,268435456],ring,ll128,-1,2,8
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

- `filter` — match conditions: `collective`, `bytesPerRank`, `nNodes`,
  `nLocalRanks`, `numPipeOps`, `regBuff`. Only `collective` is required; omitting
  any other filter field means wildcard / any.
- `config` — overrides: `algorithm`, `protocol`, `channels`, `chunkSize`.
  `algorithm` and `protocol` are required; `channels` and `chunkSize` are
  optional and omitting them means "no override".
- `bytesPerRank` / `nNodes` / `nLocalRanks` may be a JSON integer (`N` exact),
  the string `"*"` (wildcard), OR an interval string (`"[0,1048576]"`, `"(1,)"`).
- JSON and an equivalent CSV still parse to identical `TuningConfig` values;
  `rules` array order = priority. The flat (un-nested) form is no longer
  accepted.
- In a build without folly (`NCCLX_TUNER_WITH_FOLLY_JSON` undefined, e.g. the
  OSS Makefile), a `.json` path is rejected with a clear warning and no
  overrides are applied. Author the config as CSV instead.

## chunkSize and the buffer ceiling

`chunkSize` (bytes) overrides the chunk size NCCL computes for the selected
`algorithm`/`protocol`. `0` (or an omitted column) means no override.

NCCL core clamps the tuner's chunk size to `bufferMaxChunkSize`. Lowering the
chunk size always works; raising it above the buffer ceiling is silently
clamped. To make a larger chunk size effective, also enlarge the buffer via
`NCCL_BUFFSIZE` (or the per-comm `ncclxConfig.ncclBuffSize`).

## Observability

Run with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING` to see:

- `NCCLX TUNER: using built-in tuner` (tuner wired in)
- per-rule match logs and the selected algo/proto/channels/chunkSize
