# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from comms.wheel_bump import bump, retire_draft

REVISION = "a" * 40
STAMP = "2026.10.21.1234"
BASES = {
    "meta-ncclx-cu13": "2.30.3",
    "torchcomms": "0.3.0",
    "torchcomms-mccl": "0.0.1",
    "cca4py": "0.1.0",
    "nccl4py": "0.2.0",
    "commsutils": "0.1.0",
}


def fixture() -> tuple[dict, dict[str, str]]:
    current = {name: f"{base}.2026.10.20.1+bbbbbbbbbb" for name, base in BASES.items()}
    wheels = {
        owner: {
            variant: f"{BASES[name]}.{STAMP}+{REVISION[:10]}"
            for variant in bump.VARIANTS
        }
        for name, owner in bump.OWNERS.items()
        if name not in {"cca4py", "nccl4py"}
    }
    return {
        "state": "passed",
        "registry": "msl-shared",
        "number": 1234,
        "web_url": "https://buildkite.com/msl/msl-wheel-adhoc/builds/1234",
        "segment": STAMP,
        "wheels": wheels,
    }, current


class ManifestTest(unittest.TestCase):
    def test_complete_cohort_derives_companion_versions(self) -> None:
        manifest, current = fixture()
        pins = bump.derive_pins(manifest, REVISION, current)
        self.assertEqual(
            pins,
            {name: f"{base}.{STAMP}+{REVISION[:10]}" for name, base in BASES.items()},
        )

    def test_rejects_failed_partial_foreign_and_mixed_builds(self) -> None:
        manifest, current = fixture()
        cases = []
        for field, value in [
            ("state", "failed"),
            ("state", "dry-run"),
            ("registry", "farm"),
            ("segment", ""),
            ("number", True),
            ("web_url", "https://example.com"),
        ]:
            cases.append(dict(manifest, **{field: value}))
        partial = copy.deepcopy(manifest)
        del partial["wheels"]["commsutils"]["gbunified"]
        cases.append(partial)
        for version in [
            "0.1.0.2026.10.20.1234+aaaaaaaaaa",
            "0.1.0.2026.10.21.1234+bbbbbbbbbb",
        ]:
            mixed = copy.deepcopy(manifest)
            mixed["wheels"]["commsutils"]["h100"] = version
            cases.append(mixed)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                bump.derive_pins(case, REVISION, current)

    def test_explicit_companions_must_match_the_source_and_stamp(self) -> None:
        manifest, current = fixture()
        manifest["wheels"]["cca4py"] = {
            variant: f"0.2.0.{STAMP}+{REVISION[:10]}" for variant in bump.VARIANTS
        }
        self.assertEqual(
            bump.derive_pins(manifest, REVISION, current)["cca4py"],
            f"0.2.0.{STAMP}+{REVISION[:10]}",
        )
        manifest["wheels"]["cca4py"]["h100"] = "0.2.0.2026.10.20.1+bbbbbbbbbb"
        with self.assertRaises(ValueError):
            bump.derive_pins(manifest, REVISION, current)

    def test_invalid_source_hash(self) -> None:
        manifest, current = fixture()
        for revision in ["a" * 10, "z" * 40, REVISION + "\n", "-r master"]:
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                bump.derive_pins(manifest, revision, current)


class SourceTest(unittest.TestCase):
    def test_updates_only_hash_dictionary_and_comms_dependency_lines(self) -> None:
        manifest, current = fixture()
        source = (
            '_COMMS_SRC_HASH: dict[str, str] = {\n    # preserve this comment\n    "gbunified": "'
            + "b" * 40
            + '",\n    "h100": "'
            + "b" * 40
            + '",\n}\n_OTHER_HASH = "'
            + "b" * 40
            + '"\n'
        )
        requirements = ",\n".join(
            f'"{name}=={version}"' for name, version in current.items()
        )
        members = ", ".join(f'"{name}"' for name in current)
        project = f'[project]\ndependencies = [\n{requirements},\n"torch==2.15.0",\n]\n[tool.env-checks.stamp-groups]\ncomms = [{members}]\n'
        pins = bump.derive_pins(manifest, REVISION, current)
        source_new, project_new = bump.update_sources(source, project, REVISION, pins)
        self.assertEqual(
            bump.source_hashes(source_new), dict.fromkeys(bump.VARIANTS, REVISION)
        )
        self.assertIn("# preserve this comment", source_new)
        self.assertIn('_OTHER_HASH = "' + "b" * 40 + '"', source_new)
        self.assertEqual(bump.comms_pins(bump.tomllib.loads(project_new)), pins)
        self.assertIn('"torch==2.15.0"', project_new)

    def test_rejects_source_downgrade_or_divergent_branch(self) -> None:
        _, current = fixture()
        with (
            mock.patch.object(bump.subprocess, "run"),
            mock.patch.object(
                bump.subprocess, "check_output", side_effect=["b" * 40, "c" * 40]
            ),
        ):
            with self.assertRaisesRegex(ValueError, "downgrade"):
                bump.check_revision(
                    Path("/repo"),
                    REVISION,
                    dict.fromkeys(bump.VARIANTS, "b" * 40),
                    current,
                )


def lock(
    version: str, *, unrelated: str = "1.0", extra_edge: str = "", arch: str = "aarch64"
) -> str:
    return f'''version = 1
[[package]]
name = "commsutils"
version = "{version}"
source = {{ registry = "https://registry" }}
dependencies = [{{name = "provider"}}{extra_edge}]
wheels = [{{url = "https://registry/commsutils-{version}-{arch}.whl"}}, {{url = "https://registry/commsutils-{version}-x86_64.whl"}}]
[[package]]
name = "provider"
version = "{unrelated}"
source = {{ registry = "https://registry" }}
[[package]]
name = "farm"
version = "0.1.1"
dependencies = [{{name = "commsutils"}}]
[package.metadata]
requires-dist = [{{name = "commsutils", specifier = "=={version}"}}]
'''


class LockTest(unittest.TestCase):
    def test_allows_version_artifact_and_requirement_updates(self) -> None:
        bump.audit_lock(
            lock("0.1.0.old"), lock("0.1.0.new"), {"commsutils": "0.1.0.new"}
        )
        self.assertEqual(
            bump.tomllib.loads(lock("0.1.0.new"))["package"][0]["version"], "0.1.0.new"
        )

    def test_rejects_unrelated_movement_new_edges_and_missing_arch(self) -> None:
        before = lock("0.1.0.old")
        for after in [
            lock("0.1.0.new", unrelated="2.0"),
            lock("0.1.0.new", extra_edge=', {name="extra"}'),
            lock("0.1.0.new", arch="x86_64"),
            lock("0.1.0.wrong"),
            lock("0.1.0.new").replace('https://registry"', 'https://foreign"'),
        ]:
            with self.subTest(after=after), self.assertRaises(ValueError):
                bump.audit_lock(before, after, {"commsutils": "0.1.0.new"})


class DraftTest(unittest.TestCase):
    def test_published_or_foreign_diffs_are_never_retired(self) -> None:
        diff = {
            "status": "Unpublished",
            "author_fbid": "123",
            "published_epoch": None,
            "summary": bump.AUTOMATION,
        }
        self.assertTrue(retire_draft.can_retire(diff, "123", bump.AUTOMATION))
        for overrides in [
            {"status": "Needs Review"},
            {"published_epoch": 100},
            {"author_fbid": "456"},
            {"summary": "unrelated automation"},
        ]:
            with self.subTest(overrides=overrides):
                self.assertFalse(
                    retire_draft.can_retire(
                        dict(diff, **overrides), "123", bump.AUTOMATION
                    )
                )


class PreparationTest(unittest.TestCase):
    def test_dry_run_writes_combined_patch_and_restores_all_three_files(self) -> None:
        manifest, current = fixture()
        pins = bump.derive_pins(manifest, REVISION, current)
        source = f'_COMMS_SRC_HASH: dict[str, str] = {{"h100": "{"b" * 40}", "gbunified": "{"b" * 40}"}}\n'
        dependencies = ", ".join(
            f'"{name}=={version}"' for name, version in current.items()
        )
        members = ", ".join(f'"{name}"' for name in current)
        project = f"[project]\ndependencies = [{dependencies}]\n[tool.env-checks.stamp-groups]\ncomms = [{members}]\n"
        original = {
            bump.PACKAGES: source,
            bump.PROJECT: project,
            bump.LOCK: cohort_lock(current),
        }
        with tempfile.TemporaryDirectory() as directory:
            root, output = make_checkout(Path(directory), original)

            def run(command, **kwargs):
                if command[:2] == ["uv", "lock"]:
                    self.assertEqual(kwargs["env"]["UV_FROZEN"], "0")
                    self.assertEqual(set(command[3::2]), set(pins))
                    (root / bump.LOCK).write_text(cohort_lock(pins))
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            def check_output(command, **kwargs):
                if command[1] == "status":
                    return ""
                return "public" if "{phase}" in command else "b" * 40

            with (
                mock.patch.object(bump.subprocess, "run", side_effect=run),
                mock.patch.object(
                    bump.subprocess, "check_output", side_effect=check_output
                ),
                mock.patch.object(bump, "check_farm_lock"),
            ):
                bump.prepare(root, manifest, REVISION, output, dry_run=True)
            for path, content in original.items():
                self.assertEqual((root / path).read_text(), content)
                self.assertIn(path, (output / "bump.patch").read_text())
            self.assertIn(
                "This diff: Update both `_COMMS_SRC_HASH`",
                (output / "commit_message.txt").read_text(),
            )
            self.assertEqual(
                bump.json.loads((output / "provenance.json").read_text())[
                    "fbsource_hash"
                ],
                REVISION,
            )


def cohort_lock(pins: dict[str, str]) -> str:
    return "version = 1\n" + "".join(
        f'''[[package]]
name = "{name}"
version = "{version}"
source = {{registry = "https://registry"}}
wheels = [{{url = "https://registry/{name}-{version}-aarch64.whl"}}, {{url = "https://registry/{name}-{version}-x86_64.whl"}}]
'''
        for name, version in sorted(pins.items())
    )


def make_checkout(directory: Path, original: dict[str, str]) -> tuple[Path, Path]:
    root = directory / "repo"
    for path, content in original.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return root, directory / "output"
