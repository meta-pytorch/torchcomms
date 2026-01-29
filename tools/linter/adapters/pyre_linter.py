import argparse
import contextlib
import json
import logging
import os
import re
import site
import subprocess
import sys
from enum import Enum
from typing import Iterator, List, NamedTuple, Optional, Set, TypedDict

logger: logging.Logger = logging.getLogger(__name__)


def _get_packages_from_requirements() -> List[str]:
    """Parse package names from requirements.txt."""
    packages = []
    requirements_path = "requirements.txt"
    if not os.path.exists(requirements_path):
        return packages

    with open(requirements_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^([a-zA-Z0-9_-]+)", line)
            if match:
                packages.append(match.group(1))
    return packages


@contextlib.contextmanager
def _temporary_package_symlinks(
    packages: Optional[List[str]] = None,
) -> Iterator[None]:
    """Context manager that creates temporary symlinks for editable installs.

    Pyre doesn't understand editable installs that use import finders (like pytorch
    dev installs). This creates symlinks in site-packages so pyre can find them,
    then removes them when done.
    """
    if packages is None:
        packages = _get_packages_from_requirements()

    site_packages = site.getsitepackages()
    if not site_packages:
        yield
        return

    site_pkg_dir = site_packages[0]
    created_symlinks: List[str] = []

    try:
        for package_name in packages:
            site_pkg_path = os.path.join(site_pkg_dir, package_name)

            if os.path.exists(site_pkg_path) and not os.path.islink(site_pkg_path):
                continue

            try:
                package = __import__(package_name)
                if not hasattr(package, "__file__") or not package.__file__:
                    continue

                pkg_dir = os.path.dirname(package.__file__)

                if not pkg_dir.startswith(site_pkg_dir):
                    if os.path.islink(site_pkg_path):
                        current_target = os.readlink(site_pkg_path)
                        if current_target == pkg_dir:
                            continue
                        os.remove(site_pkg_path)

                    try:
                        os.symlink(pkg_dir, site_pkg_path)
                        created_symlinks.append(site_pkg_path)
                        logger.debug(
                            f"Created symlink for {package_name}: "
                            f"{site_pkg_path} -> {pkg_dir}"
                        )
                    except OSError as e:
                        logger.warning(
                            f"Failed to create symlink for {package_name}: {e}"
                        )
            except ImportError:
                pass

        yield
    finally:
        for symlink_path in created_symlinks:
            try:
                os.remove(symlink_path)
                logger.debug(f"Removed symlink: {symlink_path}")
            except OSError as e:
                logger.warning(f"Failed to remove symlink {symlink_path}: {e}")


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: Optional[str]
    line: Optional[int]
    char: Optional[int]
    code: str
    severity: LintSeverity
    name: str
    original: Optional[str]
    replacement: Optional[str]
    description: Optional[str]


class PyreResult(TypedDict):
    line: int
    column: int
    stop_line: int
    stop_column: int
    path: str
    code: int
    name: str
    description: str
    concise_description: str


def run_pyre() -> List[PyreResult]:
    with _temporary_package_symlinks():
        cmd = ["pyre", "--output", "json", "check"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.stdout:
            return json.loads(proc.stdout)
    return []


def check_pyre(
    filenames: Set[str],
) -> List[LintMessage]:
    try:
        results = run_pyre()

        normalized_filenames = set()
        for f in filenames:
            normalized_filenames.add(f)
            normalized_filenames.add(os.path.basename(f))
            if os.path.isabs(f):
                try:
                    normalized_filenames.add(os.path.relpath(f))
                except ValueError:
                    pass

        return [
            LintMessage(
                path=result["path"],
                line=result["line"],
                char=result["column"],
                code="PYRE",
                severity=LintSeverity.ERROR,
                name=result["name"],
                description=result["description"],
                original=None,
                replacement=None,
            )
            for result in results
            if result["path"] in normalized_filenames
        ]
    except Exception as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code="PYRE",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checks files with pyre",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(processName)s:%(levelname)s> %(message)s",
        level=(
            logging.NOTSET
            if args.verbose
            else logging.DEBUG
            if len(args.filenames) < 1000
            else logging.INFO
        ),
        stream=sys.stderr,
    )

    lint_messages = check_pyre(set(args.filenames))

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
