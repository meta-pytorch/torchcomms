# Comms wheel bump drafts

The `Create comms wheel bump draft` Conveyor node runs after `Comms wheel build`
passes. It consumes the build's manifest and grafted fbsource revision, then
creates one draft on a landed `master` commit with:

- Both `_COMMS_SRC_HASH` entries in `genai/msl_infra/dev/wheels/packages.py`.
- All members of farm's `[tool.env-checks.stamp-groups].comms`.
- Farm's `uv.lock`, resolved with `UV_FROZEN=0 uv lock --upgrade-package …`.

The script checks source ancestry, architecture coverage, registry, source hash,
and a shared stamp. `cca4py` and `nccl4py` inherit their owner wheel's stamp and
source hash; their API version prefixes come from the existing farm pins unless
the manifest supplies explicit companion versions. Resolution and environment
checks must confirm those versions exist.

The lock audit rejects changes to unrelated packages, dependency edges, markers,
source registries, or other metadata. The wheel configuration tests, lint,
environment pin checks before and after (same warnings), and farm lock freshness
must pass before submission. Deterministic and nightly validation run separately.

## Local dry run

Start on a clean landed fbsource commit and keep output outside the checkout:

```bash
with-proxy fbpython fbcode/comms/wheel_bump/bump.py \
  --fbsource-root "$PWD" --fbsource-hash <40-hex-source-revision> \
  --manifest /tmp/wheel_manifest.json --output-dir /tmp/comms-wheel-bump \
  --dry-run
```

This runs the relock and checks, writes the complete `bump.patch`, commit message,
and provenance, and restores the three original files. A failed preparation also
restores them. It does not commit or submit.

The workflow commits the three files together and calls `jf submit --draft`
for that commit alone, without stack dependencies. After submission, it finds
the preceding draft carrying this automation's marker and bot author, rechecks
that it has never been published, and retires it. Cleanup failures are logged
without failing a successfully submitted run.

## Rollout

Land the fbsource workflow and scripts before the Configerator change to
`source/hpc_comms/conveyors/comms_cicd/ncclx_v2_30_latest.conveyor_config.cconf`.
The build node uses `SkycastleNodeV3` so its manifest becomes a Conveyor artifact.
The wheel launcher tags must allow the `shortwave.skycastle_node` scheduler
identity, and the draft node uses the same bot identity as the feedstock draft
workflow (`1993598427904740`, overridable with `service_user_id`). Verify these
grants with an end-to-end release before enabling the node in production.
