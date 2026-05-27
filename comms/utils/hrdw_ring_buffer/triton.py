# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
"""
HRDWRingBuffer Triton Integration

Provides device-side ring-buffer writes callable from Triton kernels via
LLVM bitcode linking, plus a composable kernel-rewriting decorator that
auto-injects ring handle args + emit calls into a Triton kernel.

Two integration points:

1. Manual: pass `extern_libs={"libhrdw_ring_buffer": find_hrdw_ring_bitcode()}`
   at launch and call `hrdw_ring_write(...)` directly from your kernel body.
   Composes trivially with other bitcode libs — just merge dicts.

2. Decorator: `@wrap_kernel_inline(prefix_tag=..., suffix_tag=..., ring_provider=...)`
   rewrites the kernel at decoration time to add implicit `_hrdw_ring_*`
   args + a guarded emit call at the start and/or end of the body, and
   auto-injects the ring handle values at launch via the provided
   `ring_provider` callable. Composes with other Triton decorators
   (`@triton.autotune`, `@triton.heuristics`, custom user wrappers) in
   BOTH orders — when stacked outside, we descend the `.fn` chain to
   find the underlying JITFunction and splice our launcher back in.

Usage (manual)::

    from hrdw_ring_buffer.triton import find_hrdw_ring_bitcode, hrdw_ring_write

    @triton.jit
    def my_kernel(ring_ptr, write_idx_ptr, mask, shift, ...):
        if tl.program_id(0) == 0:
            hrdw_ring_write(ring_ptr, write_idx_ptr, mask, shift, tag)

    my_kernel[grid](
        *ring.device_handle(), ...,
        extern_libs={"libhrdw_ring_buffer": find_hrdw_ring_bitcode()},
    )

Usage (composable)::

    from hrdw_ring_buffer.triton import wrap_kernel_inline

    def _ring_provider():
        # Return (ring_ptr, write_idx_ptr, mask, shift) or (0, 0, 0, 0)
        # to disable emission for this launch.
        return ring.device_handle() if enabled else (0, 0, 0, 0)

    @wrap_kernel_inline(prefix_tag=42, suffix_tag=43, ring_provider=_ring_provider)
    @triton.jit
    def my_kernel(...):
        ...

    my_kernel[grid](...)  # ring handle injected automatically
"""

import importlib
import os
from typing import Any, Callable

from torch.utils._triton import has_triton


__all__ = [
    "find_hrdw_ring_bitcode",
]


def find_hrdw_ring_bitcode() -> str | None:
    """Find the path to the HRDWRingBuffer bitcode (libhrdw_ring_buffer.bc).

    Looks in (in order):
      1. ``$HRDW_RING_BUFFER_LIB_DIR``
      2. ``__path__`` of the ``hrdw_ring_buffer`` package (conda env install).
      3. ``__file__``-relative (development / buck builds).
    """
    lib_filename = "libhrdw_ring_buffer.bc"

    user_lib_dir = os.environ.get("HRDW_RING_BUFFER_LIB_DIR")
    if user_lib_dir is not None:
        lib_path = os.path.join(user_lib_dir, lib_filename)
        if os.path.exists(lib_path):
            return lib_path

    # `__package__` is `None` when this module is loaded outside a
    # package context (script execution, certain importers).
    # `import_module(None)` would raise TypeError — guard explicitly.
    if __package__:
        try:
            mod = importlib.import_module(__package__)
            for base in getattr(mod, "__path__", []):
                lib_path = os.path.join(base, lib_filename)
                if os.path.exists(lib_path):
                    return lib_path
        except ImportError:
            pass

    try:
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(pkg_dir, lib_filename)
        if os.path.exists(lib_path):
            return lib_path
    except NameError:
        pass

    return None


if has_triton():  # noqa: C901
    import ast
    import hashlib
    import importlib.util
    import inspect
    import tempfile
    import textwrap

    import triton  # noqa: F811
    import triton.language as tl
    from triton.language import core

    JITFunction = triton.JITFunction

    @core.extern
    def _device_hrdw_ring_write_extern(
        ring_ptr: Any,
        write_index_ptr: Any,
        mask: Any,
        shift: Any,
        data: Any,
        _semantic: Any = None,
    ) -> Any:
        """Low-level extern wrapper for device_hrdw_ring_write."""
        return core.extern_elementwise(
            "",
            "",
            [ring_ptr, write_index_ptr, mask, shift, data],
            {
                (
                    core.dtype("int64"),  # ring_ptr
                    core.dtype("int64"),  # write_index_ptr
                    core.dtype("int32"),  # mask
                    core.dtype("int32"),  # shift
                    core.dtype("int64"),  # data
                ): ("device_hrdw_ring_write", core.dtype("int32")),
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def hrdw_ring_write(
        ring_ptr,
        write_index_ptr,
        mask,
        shift,
        data,
    ):
        """Write one (timestamp, data) entry into an HRDWRingBuffer.

        The ring atomically claims a slot and stores {globaltimer, epoch, data}.
        No-op when ring_ptr is null.

        Args:
            ring_ptr: device ring pointer (int64, from RingBuffer.device_handle())
            write_index_ptr: write index pointer (int64)
            mask: ring index mask (int32)
            shift: ring index shift (int32)
            data: uint64 tag to store (int64)
        """
        _device_hrdw_ring_write_extern(
            ring_ptr.to(tl.int64),
            write_index_ptr.to(tl.int64),
            mask.to(tl.int32),
            shift.to(tl.int32),
            data.to(tl.int64),
        )

    # -----------------------------------------------------------------
    # Kernel rewriting
    # -----------------------------------------------------------------

    # Standard 4-arg HRDW ring handle injected into the kernel signature.
    _HRDW_RING_ARGS = (
        "_hrdw_ring_ptr",
        "_hrdw_write_idx_ptr",
        "_hrdw_mask",
        "_hrdw_shift",
    )

    def _rewrite_kernel_source(
        orig_fn: Any,
        injected_args: tuple[str, ...],
        body_prefix: list[ast.stmt],
        body_suffix: list[ast.stmt],
    ) -> str:
        """Apply the HRDW instrumentation AST transform; return the new source.

        Pure transform — does not write files or JIT. Exposed at module
        scope so tests can assert on the rewritten source without paying
        the JIT compilation cost (or needing a GPU).
        """
        src = textwrap.dedent(inspect.getsource(orig_fn))
        tree = ast.parse(src)
        func_def = tree.body[0]
        if not isinstance(func_def, ast.FunctionDef):
            raise TypeError(f"Expected a function definition; got {type(func_def)}")
        # Strip @triton.jit and any other decorators on the source — we
        # re-apply @triton.jit after the rewrite.
        func_def.decorator_list = []

        # Append injected args at the end with default=0. Mirror the
        # defaults so existing user defaults (e.g.
        # `USE_X: tl.constexpr = False`) stay bound to their original
        # parameters — args.defaults aligns to the LAST len(defaults)
        # entries in args.args.
        for name in injected_args:
            func_def.args.args.append(ast.arg(arg=name, annotation=None))
            func_def.args.defaults.append(ast.Constant(value=0))

        func_def.body = list(body_prefix) + list(func_def.body) + list(body_suffix)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    class _RingArgsLauncher:
        """Tiny proxy that wraps a rewritten Triton kernel.

        Behaves like a ``triton.JITFunction`` for the purposes of other
        decorators (autotune, heuristics, custom user wrappers): every
        attribute except ``__getitem__`` is forwarded to the underlying
        rewritten ``JITFunction`` via ``__getattr__``. ``__getitem__``
        wraps the launch to auto-inject the four ring args (from
        ``ring_provider()``) and the bitcode ``extern_libs`` entry.
        """

        def __init__(
            self,
            rewritten: JITFunction,
            ring_provider: Callable[[], tuple[int, int, int, int]],
            bitcode_path: str | None,
        ) -> None:
            self._rewritten = rewritten
            self._ring_provider = ring_provider
            self._extern_libs: dict[str, str] = (
                {"libhrdw_ring_buffer": bitcode_path} if bitcode_path else {}
            )

        def __getitem__(self, grid: Any) -> Callable[..., Any]:
            launcher = self._rewritten[grid]
            ring_provider = self._ring_provider
            base_libs = self._extern_libs
            arg_names = _HRDW_RING_ARGS

            def call(*args: Any, **kwargs: Any) -> Any:
                vals = ring_provider()
                if len(vals) != len(arg_names):
                    raise ValueError(
                        f"ring_provider returned {len(vals)} values, "
                        f"expected {len(arg_names)} for {arg_names}"
                    )
                # Decorator binding wins; callers shouldn't pass these by name.
                kwargs.update(zip(arg_names, vals))
                # extern_libs: merge additively; caller-passed entries win
                # on key conflict so manual overrides still work.
                kwargs["extern_libs"] = {**base_libs, **kwargs.get("extern_libs", {})}
                return launcher(*args, **kwargs)

            return call

        def __getattr__(self, name: str) -> Any:
            return getattr(self._rewritten, name)

    def _materialize_and_load(new_src: str, orig_fn: Any) -> Any:
        """Write rewritten source to a per-kernel cache file and import it.

        Triton's @jit requires the wrapped function to live in a real .py
        file (``inspect.getsource`` is called at JIT time). Per-rewrite
        SHA-digested filename keeps the cache stable across decorations
        and identical across processes.
        """
        cache_root = os.environ.get(
            "HRDW_RING_BUFFER_TRITON_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "hrdw_ring_buffer_triton_jit_cache"),
        )
        os.makedirs(cache_root, exist_ok=True)
        src_digest = hashlib.sha256(new_src.encode()).hexdigest()[:16]
        # Strip all underscores from the module path. Two kernels in
        # differently-named modules can therefore collapse to the same
        # `safe_mod`; the SHA digest in the suffix disambiguates whenever
        # the rewritten source differs, which is the only case we care
        # about for cache correctness.
        safe_mod = orig_fn.__module__.replace("_", "")
        module_name = f"hrdwrb_{safe_mod}_{orig_fn.__name__}_{src_digest}"
        src_path = os.path.join(cache_root, f"{module_name}.py")
        # Multiple workers on the same host (e.g. 4 trainer ranks per node)
        # decorate the same kernel at module-import time, racing on this
        # file. `open(..., "w")` truncates immediately, so a concurrent
        # exec_module can read an empty file and silently produce a module
        # with no kernel def — surfacing later as AttributeError on the
        # getattr below. Skip if already present (sha-digested name → same
        # content) and otherwise write via tmp + atomic rename.
        if not os.path.exists(src_path):
            tmp_path = f"{src_path}.tmp.{os.getpid()}"
            with open(tmp_path, "w") as fh:
                fh.write("import triton.language as tl  # noqa: F401\n")
                fh.write(
                    "from hrdw_ring_buffer.triton import "
                    "_device_hrdw_ring_write_extern  # noqa: F401\n\n"
                )
                fh.write(new_src)
            os.replace(tmp_path, src_path)

        spec = importlib.util.spec_from_file_location(module_name, src_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to spec_from_file_location {src_path}")
        loader = spec.loader  # narrow Optional for pyre
        new_mod = importlib.util.module_from_spec(spec)
        # Seed the new module's globals with the original kernel's globals so
        # any free names in the body (constants, other helpers) resolve.
        # Don't clobber the new module's dunders (__name__, __spec__, ...) —
        # the importlib loader's _check_name_wrapper would fail.
        for k, v in orig_fn.__globals__.items():
            if not (k.startswith("__") and k.endswith("__")):
                new_mod.__dict__.setdefault(k, v)
        loader.exec_module(new_mod)
        return getattr(new_mod, orig_fn.__name__)

    def _make_hrdw_emit_stmt(tag: int) -> ast.stmt:
        """AST for: gated single-write into the HRDW ring with `tag`.

        Fires exactly once per launch (gated on `program_id(0..2) == 0`),
        only when the ring is enabled (`_hrdw_ring_ptr != 0`). The
        ``tl.full([], v, dtype)`` casts turn Python-int kernel args into
        Triton scalar tensors (the `hrdw_ring_write` wrapper uses `.to()`
        which fails on ints).
        """
        code = textwrap.dedent(
            f"""
            if (tl.program_id(0) == 0) & (tl.program_id(1) == 0) & (tl.program_id(2) == 0):
                if _hrdw_ring_ptr != 0:
                    _device_hrdw_ring_write_extern(
                        tl.full([], _hrdw_ring_ptr, tl.int64),
                        tl.full([], _hrdw_write_idx_ptr, tl.int64),
                        tl.full([], _hrdw_mask, tl.int32),
                        tl.full([], _hrdw_shift, tl.int32),
                        tl.full([], {int(tag)}, tl.int64),
                    )
            """
        )
        return ast.parse(code).body[0]

    def wrap_kernel_inline(
        *,
        prefix_tag: int | None = None,
        suffix_tag: int | None = None,
        ring_provider: Callable[[], tuple[int, int, int, int]],
    ) -> Callable[[JITFunction], "_RingArgsLauncher"]:
        """Decorator: rewrite a Triton kernel to emit HRDW ring entries.

        Must be applied immediately above ``@triton.jit``. The returned
        launcher proxies to the rewritten ``JITFunction`` for all attribute
        access, so any other Triton decorator (``@triton.autotune``,
        ``@triton.heuristics``, custom user wrappers) can stack ABOVE this
        one — they receive a normal JITFunction-shaped object and pass
        launch args through to our launcher unchanged.

            @triton.autotune(configs=[...], key=[...])
            @wrap_kernel_inline(prefix_tag=42, suffix_tag=43, ring_provider=...)
            @triton.jit
            def my_kernel(...): ...

        Overhead (GB200, measured on `_kernel_fused_swiglu`): ~300 ns/launch
        when telemetry is enabled (one extern call by thread 0 of the first
        program, plus a per-thread predicate check). When the ring provider
        returns zeros, the `_hrdw_ring_ptr != 0` short-circuit elides the
        extern; only the predicate evaluation remains.

        Args:
            prefix_tag: If not None, emit an entry with this tag at the
                start of the kernel body.
            suffix_tag: If not None, emit an entry with this tag at the
                end of the kernel body.
            ring_provider: Callable invoked at each launch; returns
                ``(ring_ptr, write_idx_ptr, mask, shift)`` as 4 int64s
                (typically from ``RingBuffer.device_handle()``). Return
                ``(0, 0, 0, 0)`` to disable emission for that launch.
        """
        if prefix_tag is None and suffix_tag is None:
            raise ValueError(
                "wrap_kernel_inline requires at least one of prefix_tag or suffix_tag"
            )

        def decorator(jit_fn: JITFunction) -> "_RingArgsLauncher":
            if not isinstance(jit_fn, JITFunction):
                raise TypeError(
                    "@wrap_kernel_inline must wrap a @triton.jit kernel directly; "
                    f"got {type(jit_fn).__name__}. Put other decorators "
                    "(@triton.autotune, @triton.heuristics, etc.) ABOVE "
                    "@wrap_kernel_inline, not below."
                )
            body_prefix = (
                [_make_hrdw_emit_stmt(prefix_tag)] if prefix_tag is not None else []
            )
            body_suffix = (
                [_make_hrdw_emit_stmt(suffix_tag)] if suffix_tag is not None else []
            )
            new_src = _rewrite_kernel_source(
                jit_fn.fn, _HRDW_RING_ARGS, body_prefix, body_suffix
            )
            new_fn = _materialize_and_load(new_src, jit_fn.fn)
            # Forward whatever the user originally passed to `@triton.jit(...)`
            # (`do_not_specialize`, `noinline`, `launch_metadata`, etc.) by
            # iterating over `JITFunction.__init__`'s signature and pulling
            # each matching attr off the original `jit_fn`. Robust to Triton
            # adding/removing options across releases — we forward whatever
            # the current JITFunction class declares as an init param.
            jit_kwargs: dict[str, Any] = {}
            for name in inspect.signature(JITFunction.__init__).parameters:
                if name in ("self", "fn"):
                    continue
                if hasattr(jit_fn, name):
                    jit_kwargs[name] = getattr(jit_fn, name)
            rewritten = triton.jit(new_fn, **jit_kwargs)
            return _RingArgsLauncher(rewritten, ring_provider, find_hrdw_ring_bitcode())

        return decorator

    __all__ += [
        "hrdw_ring_write",
        "wrap_kernel_inline",
    ]
