import os
import re

import pytest


def pytest_collection_modifyitems(config, items):
    pattern = os.environ.get("DISABLED_TESTS")
    if not pattern:
        return

    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise pytest.UsageError(f"Invalid DISABLED_TESTS regex {pattern!r}: {e}")

    skip_marker = pytest.mark.skip(reason=f"disabled via DISABLED_TESTS={pattern!r}")
    for item in items:
        if regex.search(item.nodeid):
            item.add_marker(skip_marker)
