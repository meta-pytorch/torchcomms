# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""The one machine-checked backend contract: the public a2a entry signatures.

The ``comms.dsl.collectives`` facade forwards its keyword arguments uniformly to whichever
backend ``backend=`` selects, so a backend entry must accept every kwarg the facade forwards
-- otherwise ``all_to_all(backend=..., that_param=...)`` would break silently with no other
test catching it. Today CuTe is the only registered backend; when a second DSL (Triton) is
re-registered this test grows a cross-backend parameter-name parity assertion again (the
entries' parameter names must match so the facade can forward uniformly).

Needs the cutlass DSL to import the CuTe substrate, so it runs on a GPU host.
"""

import importlib
import inspect
import unittest

_BACKENDS = ("cute",)
# Public a2a entries the facade forwards uniformly, and the kwargs it forwards to each.
_A2A_ENTRY_KWARGS = {
    "all_to_all": ("produce", "consume", "rows", "config", "variant"),
    "all_to_all_zc": ("primitive", "config"),
}


class A2AEntryContractTest(unittest.TestCase):
    def test_backend_entries_accept_forwarded_kwargs(self) -> None:
        # Every kwarg the facade forwards must be a parameter of each backend's entry.
        for b in _BACKENDS:
            mod = importlib.import_module(f"comms.dsl.{b}.a2a.host")
            for entry, kwargs in _A2A_ENTRY_KWARGS.items():
                params = tuple(inspect.signature(getattr(mod, entry)).parameters)
                for kw in kwargs:
                    self.assertIn(
                        kw,
                        params,
                        f"{b}.{entry} is missing facade-forwarded kwarg {kw!r}: {params}",
                    )
