# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Tests for the Triton-side wrapper in ``hrdw_ring_buffer.triton``.

Covers:
- ``find_hrdw_ring_bitcode()`` path resolution (env-var override + miss).
- ``_rewrite_kernel_source`` AST transform: signature injection, body
  prefix/suffix placement, decorator stripping.
- ``wrap_kernel_inline(...)`` decorator: prefix/suffix tag requirement,
  rejects non-JITFunction targets with a useful message, composes with
  ``@triton.autotune`` stacked above.
- GPU end-to-end: launch an instrumented kernel and drain the ring.
"""

import os
import tempfile
import unittest

import torch
from hrdw_ring_buffer.triton import find_hrdw_ring_bitcode


class FindBitcodeTest(unittest.TestCase):
    """Path resolution for ``find_hrdw_ring_bitcode``."""

    def setUp(self) -> None:
        self._saved_env = os.environ.get("HRDW_RING_BUFFER_LIB_DIR")

    def tearDown(self) -> None:
        if self._saved_env is None:
            os.environ.pop("HRDW_RING_BUFFER_LIB_DIR", None)
        else:
            os.environ["HRDW_RING_BUFFER_LIB_DIR"] = self._saved_env

    def test_env_var_hit(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bc = os.path.join(d, "libhrdw_ring_buffer.bc")
            open(bc, "wb").close()
            os.environ["HRDW_RING_BUFFER_LIB_DIR"] = d
            self.assertEqual(find_hrdw_ring_bitcode(), bc)

    def test_env_var_set_but_file_missing_falls_through(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            os.environ["HRDW_RING_BUFFER_LIB_DIR"] = d
            # Result depends on whether the package/__file__ neighbour has
            # a bitcode (typical buck-run environment doesn't). We just
            # assert it didn't accidentally return the empty env-dir path.
            result = find_hrdw_ring_bitcode()
            self.assertNotEqual(result, os.path.join(d, "libhrdw_ring_buffer.bc"))


@unittest.skipUnless(torch.utils._triton.has_triton(), "triton not available")
class RewriteKernelSourceTest(unittest.TestCase):
    """Pure-Python AST rewrite — no GPU, no JIT compilation."""

    def _make_jit(self):  # pyre-fixme[3]
        import triton  # noqa: F401
        import triton.language as tl

        @triton.jit
        def kernel(x_ptr: torch.Tensor, BLOCK: tl.constexpr) -> None:
            offs = tl.arange(0, BLOCK)
            tl.store(x_ptr + offs, offs)

        return kernel

    def test_injects_signature_args_and_decorator_stripped(self) -> None:
        from hrdw_ring_buffer.triton import _HRDW_RING_ARGS, _rewrite_kernel_source

        kernel = self._make_jit()
        new_src = _rewrite_kernel_source(kernel.fn, _HRDW_RING_ARGS, [], [])
        # The 4 ring args appear in the signature.
        for name in _HRDW_RING_ARGS:
            self.assertIn(name, new_src)
        # The @triton.jit decorator was stripped (re-applied externally).
        self.assertNotIn("@triton.jit", new_src)
        # Original body is preserved.
        self.assertIn("tl.store", new_src)

    def test_body_prefix_emit_comes_before_original(self) -> None:
        from hrdw_ring_buffer.triton import (
            _HRDW_RING_ARGS,
            _make_hrdw_emit_stmt,
            _rewrite_kernel_source,
        )

        kernel = self._make_jit()
        new_src = _rewrite_kernel_source(
            kernel.fn, _HRDW_RING_ARGS, [_make_hrdw_emit_stmt(0xDEAD)], []
        )
        emit_pos = new_src.find("_device_hrdw_ring_write_extern")
        body_pos = new_src.find("tl.store")
        self.assertGreater(emit_pos, 0)
        self.assertLess(emit_pos, body_pos)
        # The tag is baked into the rewritten source as a literal.
        self.assertIn("57005", new_src)  # 0xDEAD

    def test_body_suffix_emit_comes_after_original(self) -> None:
        from hrdw_ring_buffer.triton import (
            _HRDW_RING_ARGS,
            _make_hrdw_emit_stmt,
            _rewrite_kernel_source,
        )

        kernel = self._make_jit()
        new_src = _rewrite_kernel_source(
            kernel.fn, _HRDW_RING_ARGS, [], [_make_hrdw_emit_stmt(0xBEEF)]
        )
        emit_pos = new_src.find("_device_hrdw_ring_write_extern")
        body_pos = new_src.find("tl.store")
        self.assertGreater(body_pos, 0)
        self.assertGreater(emit_pos, body_pos)


@unittest.skipUnless(torch.utils._triton.has_triton(), "triton not available")
class WrapKernelInlineTest(unittest.TestCase):
    """Decorator-level behaviour for ``wrap_kernel_inline``."""

    def _make_jit(self):  # pyre-fixme[3]
        import triton  # noqa: F401
        import triton.language as tl

        @triton.jit
        def kernel(x_ptr: torch.Tensor, BLOCK: tl.constexpr) -> None:
            offs = tl.arange(0, BLOCK)
            tl.store(x_ptr + offs, offs)

        return kernel

    def test_requires_at_least_one_tag(self) -> None:
        from hrdw_ring_buffer.triton import wrap_kernel_inline

        with self.assertRaises(ValueError):
            wrap_kernel_inline(ring_provider=lambda: (0, 0, 0, 0))

    def test_rejects_non_jitfunction(self) -> None:
        from hrdw_ring_buffer.triton import wrap_kernel_inline

        with self.assertRaisesRegex(
            TypeError, "must wrap a @triton.jit kernel directly"
        ):
            wrap_kernel_inline(prefix_tag=1, ring_provider=lambda: (0, 0, 0, 0))(
                # pyre-ignore[6]: intentionally passing a non-JITFunction to
                # exercise the TypeError path.
                lambda x: x
            )

    def test_proxy_forwards_jit_attributes(self) -> None:
        """The launcher proxy should expose the underlying JITFunction's
        attributes so outer decorators (autotune, heuristics, custom user
        wrappers) can introspect it as if it were a plain JITFunction."""
        from hrdw_ring_buffer.triton import wrap_kernel_inline

        decorated = wrap_kernel_inline(
            prefix_tag=1, ring_provider=lambda: (0, 0, 0, 0)
        )(self._make_jit())
        # `.fn` is the Python source function — autotune / heuristics rely
        # on it being accessible. The launcher proxies via __getattr__.
        self.assertTrue(callable(decorated.fn))
        # The rewritten kernel name matches the original.
        self.assertEqual(decorated.fn.__name__, "kernel")

    def test_composes_with_triton_autotune(self) -> None:
        """``@triton.autotune`` stacked above ``@wrap_kernel_inline`` should
        wrap the launcher proxy without complaining — autotune calls into our
        launcher via its `.fn` attribute, which is what the proxy forwards."""
        import triton  # noqa: F401
        import triton.language as tl
        from hrdw_ring_buffer.triton import wrap_kernel_inline

        @triton.autotune(
            configs=[triton.Config({"BLOCK": 16}, num_warps=1)],
            key=[],
        )
        @wrap_kernel_inline(prefix_tag=1, ring_provider=lambda: (0, 0, 0, 0))
        @triton.jit
        def kernel(x_ptr: torch.Tensor, BLOCK: tl.constexpr) -> None:
            offs = tl.arange(0, BLOCK)
            tl.store(x_ptr + offs, offs)

        # Autotune's wrapper should construct without error and expose the
        # standard Autotuner surface (fn pointing to our launcher's
        # rewritten kernel).
        self.assertTrue(hasattr(kernel, "fn"))

    def test_forwards_triton_jit_kwargs(self) -> None:
        """Kwargs the user originally passed to `@triton.jit(...)` (e.g.
        `do_not_specialize`, `noinline`) should survive the rewrite and be
        re-applied to the new JITFunction."""
        import triton  # noqa: F401
        import triton.language as tl
        from hrdw_ring_buffer.triton import wrap_kernel_inline

        @wrap_kernel_inline(prefix_tag=1, ring_provider=lambda: (0, 0, 0, 0))
        @triton.jit(do_not_specialize=["x_ptr"], noinline=True)
        def kernel(x_ptr: torch.Tensor, BLOCK: tl.constexpr) -> None:
            offs = tl.arange(0, BLOCK)
            tl.store(x_ptr + offs, offs)

        rewritten = kernel._rewritten  # the inner JITFunction we re-jit-ed
        # The two kwargs flow through verbatim.
        self.assertEqual(rewritten.do_not_specialize, ["x_ptr"])
        self.assertTrue(rewritten.noinline)


@unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
@unittest.skipUnless(torch.utils._triton.has_triton(), "triton not available")
class InlineEmitEndToEndTest(unittest.TestCase):
    """Launch an instrumented Triton kernel on the GPU and drain the ring."""

    def setUp(self) -> None:
        torch.cuda.set_device(0)

    def test_emit_prefix_and_suffix(self) -> None:
        import triton  # noqa: F401
        import triton.language as tl
        from hrdw_ring_buffer import RingBuffer
        from hrdw_ring_buffer.triton import wrap_kernel_inline

        ring: RingBuffer[int] = RingBuffer(256)
        self.assertTrue(ring.valid)
        handle = ring.device_handle()

        @wrap_kernel_inline(
            prefix_tag=0xA1,
            suffix_tag=0xB2,
            ring_provider=lambda: tuple(handle),
        )
        @triton.jit
        def noop_kernel(x_ptr: torch.Tensor, BLOCK: tl.constexpr) -> None:
            offs = tl.arange(0, BLOCK)
            tl.store(x_ptr + offs, offs)

        x = torch.zeros(16, dtype=torch.int32, device="cuda")
        noop_kernel[(1,)](x, BLOCK=16)
        stream = torch.cuda.current_stream().cuda_stream
        entries, result = ring.poll(stream)
        tags = [e.data for e in entries]
        self.assertGreaterEqual(result.entries_read, 2)
        self.assertIn(0xA1, tags)
        self.assertIn(0xB2, tags)

    def test_zero_handle_short_circuits_emit(self) -> None:
        import triton  # noqa: F401
        import triton.language as tl
        from hrdw_ring_buffer import RingBuffer
        from hrdw_ring_buffer.triton import wrap_kernel_inline

        # Ring is real (so the bitcode can be linked) but the provider
        # returns zeros — the `if _hrdw_ring_ptr != 0` short-circuit in
        # the rewritten body should skip the extern call entirely.
        ring: RingBuffer[int] = RingBuffer(256)
        self.assertTrue(ring.valid)

        @wrap_kernel_inline(prefix_tag=0xC3, ring_provider=lambda: (0, 0, 0, 0))
        @triton.jit
        def noop_kernel(x_ptr: torch.Tensor, BLOCK: tl.constexpr) -> None:
            offs = tl.arange(0, BLOCK)
            tl.store(x_ptr + offs, offs)

        x = torch.zeros(16, dtype=torch.int32, device="cuda")
        noop_kernel[(1,)](x, BLOCK=16)
        stream = torch.cuda.current_stream().cuda_stream
        _, result = ring.poll(stream)
        self.assertEqual(result.entries_read, 0)

    def test_autotune_above_forwards_caller_and_config_args(self) -> None:
        """End-to-end check that the autotune-above-wrap path doesn't drop or
        modify any launch args. The autotuner picks a config (which contributes
        `BLOCK` as a constexpr kwarg), the caller passes `n` positionally, and
        our launcher injects the 4 `_hrdw_*` kwargs. All three must arrive at
        the rewritten kernel intact for the test array to be written correctly
        AND for the prefix tag to land in the ring.
        """
        import triton  # noqa: F401
        import triton.language as tl
        from hrdw_ring_buffer import RingBuffer
        from hrdw_ring_buffer.triton import wrap_kernel_inline

        ring: RingBuffer[int] = RingBuffer(256)
        self.assertTrue(ring.valid)
        handle = ring.device_handle()

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK": 16}, num_warps=1),
                triton.Config({"BLOCK": 32}, num_warps=2),
            ],
            key=["n"],
        )
        @wrap_kernel_inline(
            prefix_tag=0xD4,
            ring_provider=lambda: tuple(handle),
        )
        @triton.jit
        def autotuned_kernel(x_ptr: torch.Tensor, n: int, BLOCK: tl.constexpr) -> None:
            offs = tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(x_ptr + offs, offs, mask=mask)

        x = torch.zeros(16, dtype=torch.int32, device="cuda")
        autotuned_kernel[(1,)](x, n=16)
        stream = torch.cuda.current_stream().cuda_stream
        # Kernel ran correctly (caller arg `n` + autotune-picked `BLOCK`
        # both forwarded to the rewritten kernel).
        expected = torch.arange(16, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(x, expected))
        # And the prefix tag landed in the ring (our launcher's ring-arg
        # injection survived the autotune → launcher hop).
        entries, _ = ring.poll(stream)
        self.assertIn(0xD4, [e.data for e in entries])
