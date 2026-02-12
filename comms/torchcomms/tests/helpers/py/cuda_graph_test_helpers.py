# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import contextlib
import os
import re
import tempfile
import unittest
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, TYPE_CHECKING

import pydot
import torch
import torchcomms

# pyre-fixme[5]: Global annotation for skip decorator.
skip_unless_ncclx = unittest.skipIf(
    os.getenv("TEST_BACKEND") != "ncclx", "Skipping NCCLX-only tests"
)

# pyre-fixme[21]: Could not find name `ProfilerActivity` in `torch.profiler`.
from torch.profiler import profile, ProfilerActivity
from torchcomms.tests.integration.py.TorchCommTestHelpers import get_rank_and_size

if TYPE_CHECKING:
    from typing import Self

# Type alias for a single substep that can run on a stream
_Substep = (
    torch.cuda.CUDAGraph
    | tuple[torch.cuda.CUDAGraph | Callable[[], None], torch.cuda.Stream]
)

# Type alias for pipeline steps
PipelineStep = _Substep | Callable[[], None] | list[_Substep]


@dataclass
class CudaGraphNode:
    id: str
    type: str
    label: str
    kernel_name: str | None


@dataclass
class CudaGraphInfo:
    nodes: list[CudaGraphNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def nodes_of_type(self, node_type: str) -> list[CudaGraphNode]:
        return [n for n in self.nodes if n.type == node_type]

    @property
    def num_kernel_nodes(self) -> int:
        return len(self.nodes_of_type("KERNEL"))

    def kernels_with_name(self, substring: str) -> list[CudaGraphNode]:
        """Return KERNEL nodes whose kernel_name contains the given substring."""
        return [
            n
            for n in self.nodes
            if n.type == "KERNEL" and n.kernel_name and substring in n.kernel_name
        ]

    def has_path(self, src_id: str, dst_id: str) -> bool:
        """Check if there is a directed path from src_id to dst_id."""
        adjacency: dict[str, list[str]] = defaultdict(list)
        for s, d in self.edges:
            adjacency[s].append(d)
        visited: set[str] = set()
        stack = [src_id]
        while stack:
            node = stack.pop()
            if node == dst_id:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency[node])
        return False

    def are_sequential(self, node_a: CudaGraphNode, node_b: CudaGraphNode) -> bool:
        """Check if a and b are ordered (a path exists in either direction)."""
        return self.has_path(node_a.id, node_b.id) or self.has_path(
            node_b.id, node_a.id
        )

    def are_parallel(self, node_a: CudaGraphNode, node_b: CudaGraphNode) -> bool:
        """Check if a and b can execute concurrently (no path in either direction)."""
        return not self.are_sequential(node_a, node_b)


def _collect_dot_elements(
    graph: pydot.Dot | pydot.Subgraph,
) -> tuple[list[pydot.Node], list[pydot.Edge]]:
    """Recursively collect all nodes and edges from a graph and its subgraphs."""
    nodes = list(graph.get_nodes())
    edges = list(graph.get_edges())
    for subgraph in graph.get_subgraphs():
        sub_nodes, sub_edges = _collect_dot_elements(subgraph)
        nodes.extend(sub_nodes)
        edges.extend(sub_edges)
    return nodes, edges


def _wait(work: object | None) -> None:
    """Wait on an async collective work handle, no-op if sync (None)."""
    if work is not None:
        work.wait()  # pyre-ignore[16]


def _parse_cuda_graph_dot(dot_content: str) -> CudaGraphInfo:
    """Parse DOT output from cudaGraphDebugDotPrint into structured CudaGraphInfo."""
    dot_content = dot_content.strip()
    if not dot_content:
        return CudaGraphInfo()
    graphs = pydot.graph_from_dot_data(dot_content)
    if not graphs:
        return CudaGraphInfo()

    dot_nodes, dot_edges = _collect_dot_elements(graphs[0])

    nodes: list[CudaGraphNode] = []
    for dot_node in dot_nodes:
        node_id = dot_node.get_name().strip('"')
        label = (dot_node.get("label") or "").strip('"')
        if not label:
            continue

        # Node type: first all-caps identifier (3+ chars) in the label.
        # Works for both record labels ("{KERNEL\n|...") and plain
        # labels ("0 (topoId: 6)\nEVENT_WAIT\n...").
        type_match = re.search(r"\b([A-Z][A-Z_]{2,})\b", label)
        node_type = type_match.group(1) if type_match else "UNKNOWN"

        # For KERNEL nodes, extract function name before launch config
        kernel_name: str | None = None
        if node_type == "KERNEL":
            name_match = re.search(r"(\S+?)(?:\\<\\<\\<|<<<)", label)
            if name_match:
                kernel_name = name_match.group(1)

        nodes.append(
            CudaGraphNode(
                id=node_id, type=node_type, label=label, kernel_name=kernel_name
            )
        )

    edges: list[tuple[str, str]] = [
        (e.get_source().strip('"'), e.get_destination().strip('"')) for e in dot_edges
    ]

    return CudaGraphInfo(nodes=nodes, edges=edges)


def analyze_cuda_graph(
    graph: torch.cuda.CUDAGraph, svg_path: str | None = None
) -> CudaGraphInfo:
    """Dump a CUDA graph to DOT format and parse it into structured CudaGraphInfo.

    Requires the graph to have been created with keep_graph=True so that
    the underlying cudaGraph_t is available for debug_dump.

    Args:
        graph: The CUDA graph to analyze.
        svg_path: If provided, render the graph to this SVG file path.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        graph.debug_dump(tmp_path)
        with open(tmp_path) as f:
            dot_content = f.read()
    finally:
        os.unlink(tmp_path)

    if not dot_content.strip():
        raise RuntimeError("debug_dump produced empty DOT output")

    if svg_path is not None:
        graphs = pydot.graph_from_dot_data(dot_content)
        if graphs:
            graphs[0].write_svg(svg_path)

    return _parse_cuda_graph_dot(dot_content)


# Number of tensors per graph in create_capture:
#   input (10x10), intermediate (1,), output (size,)
TENSORS_PER_CAPTURE = 3


def create_capture(
    input_idx: int,
    intermediate_idx: int,
    output_idx: int,
    comm0_idx: int = 0,
    comm1_idx: int = 0,
) -> "Callable[[GraphTestBuilder], None]":
    """Create a capture that mixes sync/async ops with intra-graph stream dependencies.

    Pattern (two local streams with an explicit dependency):
      stream0: all_reduce(sync, comm0) on inputs[input_idx]
               → sum → inputs[intermediate_idx]
      stream1 (waits stream0): all_gather(async, comm1)
               inputs[intermediate_idx] → inputs[output_idx]
    """

    def capture(b: "GraphTestBuilder") -> None:
        stream0 = torch.cuda.Stream()
        stream1 = torch.cuda.Stream()
        initial = torch.cuda.current_stream()

        stream0.wait_stream(initial)
        with torch.cuda.stream(stream0):
            # Sync collective
            b.comms[comm0_idx].all_reduce(
                b.inputs[input_idx], torchcomms.ReduceOp.SUM, async_op=False
            )
            torch.sum(
                b.inputs[input_idx].flatten(),
                dim=0,
                keepdim=True,
                out=b.inputs[intermediate_idx],
            )

        # Stream dependency: stream1 waits for stream0 to finish
        stream1.wait_stream(stream0)
        with torch.cuda.stream(stream1):
            # Async collective
            _wait(
                b.comms[comm1_idx].all_gather_single(
                    b.inputs[output_idx],
                    b.inputs[intermediate_idx],
                    async_op=True,
                )
            )

        initial.wait_stream(stream1)

    return capture


class GraphTestBuilder:
    """
    Builder for CUDA graph tests that handles the common pattern:
    1. Create comms and streams
    2. Capture operations into graphs
    3. For each replay: reset inputs, run pipeline, assert results

    Usage:
        # Serial replay (default):
        GraphTestBuilder(self)
            .add_capture(lambda b: b.comms[0].all_reduce(b.inputs[0], ...))
            .run_serial(
                inputs=lambda b: [torch.ones(10, 10, device=device)],
                expected=lambda b: [inp * b.comms[0].get_size() for inp in b.inputs],
            )

        # Concurrent replay on different streams:
        GraphTestBuilder(self)
            .with_streams(2)
            .add_capture(lambda b: ..., stream=0)
            .add_capture(lambda b: ..., stream=1)
            .run_concurrent(
                inputs=lambda b: [...],
                expected=lambda b: [...],
            )

        # Custom schedule:
        GraphTestBuilder(self)
            .with_streams(2)
            .add_capture(lambda b: ..., stream=0)
            .add_capture(lambda b: ..., stream=1)
            .run_custom_schedule(
                lambda b: [(b.graphs[0], b.streams[0]), callback, ...],
                inputs=lambda b: [...],
                expected=lambda b: [...],
            )

    Both inputs and expected can be either a list of tensors or a callable that
    takes the builder (for when you need comm.get_rank() or comm.get_size()).
    """

    def __init__(self, test_case: "CudaGraphTestBase") -> None:
        self.test_case = test_case
        self._num_comms = 1
        self._num_streams = 0

        # Populated during run
        self.comms: list[torchcomms.TorchComm] = []
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self.streams: list[torch.cuda.Stream] = []
        self.inputs: list[torch.Tensor] = []
        self.expected: list[torch.Tensor] = []
        self.graph: torch.cuda.CUDAGraph | None = None  # Current graph being captured
        self.graph_infos: list[CudaGraphInfo] = []

        self._captures: list[tuple[int | None, Callable[[GraphTestBuilder], None]]] = []
        self._existing_comms: list[torchcomms.TorchComm] | None = None

    def with_existing_comms(self, comms: list[torchcomms.TorchComm]) -> "Self":
        """Use pre-existing comms instead of creating new ones."""
        self._existing_comms = comms
        return self

    def with_comms(self, n: int) -> "Self":
        self._num_comms = n
        return self

    def with_streams(self, n: int) -> "Self":
        self._num_streams = n
        return self

    def add_capture(
        self,
        fn: Callable[["GraphTestBuilder"], None],
        stream: int | None = None,
    ) -> "Self":
        """
        Add a capture function for a new graph.

        The function receives the builder with b.graph set to the current graph.
        The torch.cuda.graph() context is handled automatically.

        Args:
            fn: Capture function that records operations (accesses b.graph, b.inputs, etc.)
            stream: Stream index to capture on (None = default stream)
        """
        self._captures.append((stream, fn))
        return self

    def _run(
        self,
        inputs: list[torch.Tensor] | Callable[["GraphTestBuilder"], list[torch.Tensor]],
        expected: list[torch.Tensor]
        | Callable[["GraphTestBuilder"], list[torch.Tensor]],
        pipeline_fn: Callable[["GraphTestBuilder"], list[PipelineStep]],
        graph_assertions: Callable[["GraphTestBuilder"], None] | None = None,
    ) -> None:
        """Internal method that executes the test with the given pipeline."""
        assert len(self._captures) > 0, "add_capture() must be called at least once"

        self.streams = [torch.cuda.Stream() for _ in range(self._num_streams)]

        comms_ctx = (
            contextlib.nullcontext(self._existing_comms)
            if self._existing_comms is not None
            else self.test_case.create_comms(self._num_comms)
        )

        with comms_ctx as comms:
            self.comms = list(comms)
            if callable(inputs):
                self.inputs = inputs(self)
            else:
                self.inputs = inputs
            if callable(expected):
                self.expected = expected(self)
            else:
                self.expected = expected
            originals = [inp.clone() for inp in self.inputs]

            with self.test_case.create_graphs(len(self._captures)) as graphs:
                self.graphs = list(graphs)

                # Capture phase
                for i, (stream_idx, capture_fn) in enumerate(self._captures):
                    self.graph = self.graphs[i]
                    if stream_idx is not None:
                        with torch.cuda.graph(
                            self.graph, stream=self.streams[stream_idx]
                        ):
                            capture_fn(self)
                    else:
                        with torch.cuda.graph(self.graph):
                            capture_fn(self)
                    # keep_graph=True skips auto-instantiation in capture_end
                    self.graphs[i].instantiate()
                self.graph = None

                # Build a test name for output files (SVGs, profiles)
                rank, _ = get_rank_and_size()
                test_name = self.test_case._testMethodName
                subtest = getattr(self.test_case, "_subtest", None)
                if subtest is not None:
                    params = "_".join(f"{k}={v}" for k, v in subtest.params.items())
                    test_name = f"{test_name}_{params}"

                # Analyze captured graphs, optionally saving SVGs
                svg_dir = os.environ.get("CUDA_GRAPH_SVG_DIR")
                if svg_dir:
                    os.makedirs(svg_dir, exist_ok=True)
                self.graph_infos = [
                    analyze_cuda_graph(
                        g,
                        svg_path=os.path.join(
                            svg_dir,
                            f"{test_name}_graph{i}_rank{rank}.svg",
                        )
                        if svg_dir
                        else None,
                    )
                    for i, g in enumerate(self.graphs)
                ]

                if graph_assertions is not None:
                    graph_assertions(self)

                # Replay phase, optionally under torch profiler
                profile_dir = os.environ.get("TORCH_PROFILE_DIR")
                if profile_dir:
                    os.makedirs(profile_dir, exist_ok=True)
                profile_path = (
                    os.path.join(profile_dir, f"{test_name}_rank{rank}.json")
                    if profile_dir
                    else None
                )
                profile_ctx = (
                    profile(
                        activities=[
                            # pyre-fixme[16]: Module `torch.profiler` has no
                            # attribute `ProfilerActivity`.
                            ProfilerActivity.CPU,
                            # pyre-fixme[16]: Module `torch.profiler` has no
                            # attribute `ProfilerActivity`.
                            ProfilerActivity.CUDA,
                        ],
                    )
                    if profile_path
                    else contextlib.nullcontext()
                )

                with profile_ctx as prof:
                    for _ in range(self.test_case.NUM_REPLAYS):
                        for inp, orig in zip(self.inputs, originals):
                            inp.copy_(orig)
                        # Ensure copies on default stream complete before graph
                        # replays on side streams read the input tensors.
                        torch.cuda.synchronize()

                        steps = pipeline_fn(self)
                        self.test_case.run_graph_pipeline(steps)

                        for inp, exp in zip(self.inputs, self.expected):
                            torch.testing.assert_close(inp, exp)

                if prof is not None and profile_path is not None:
                    prof.export_chrome_trace(profile_path)

    def run_serial(
        self,
        inputs: list[torch.Tensor] | Callable[["GraphTestBuilder"], list[torch.Tensor]],
        expected: list[torch.Tensor]
        | Callable[["GraphTestBuilder"], list[torch.Tensor]],
        graph_assertions: Callable[["GraphTestBuilder"], None] | None = None,
    ) -> None:
        """
        Execute the test, replaying graphs serially on the default stream.

        Args:
            inputs: Input tensors, or a callable that returns them (for rank-dependent inputs)
            expected: Expected output tensors, or a callable that returns them
            graph_assertions: Optional callback to assert properties of captured graph infos
        """
        self._run(inputs, expected, lambda b: list(b.graphs), graph_assertions)

    def run_concurrent(
        self,
        inputs: list[torch.Tensor] | Callable[["GraphTestBuilder"], list[torch.Tensor]],
        expected: list[torch.Tensor]
        | Callable[["GraphTestBuilder"], list[torch.Tensor]],
        graph_assertions: Callable[["GraphTestBuilder"], None] | None = None,
    ) -> None:
        """
        Execute the test, replaying all graphs concurrently on their respective streams.

        Requires that add_capture() was called with stream= for each graph, and that
        with_streams() was called with enough streams.

        Args:
            inputs: Input tensors, or a callable that returns them (for rank-dependent inputs)
            expected: Expected output tensors, or a callable that returns them
            graph_assertions: Optional callback to assert properties of captured graph infos
        """

        def concurrent_pipeline(b: "GraphTestBuilder") -> list[PipelineStep]:
            return [[(g, s) for g, s in zip(b.graphs, b.streams)]]

        self._run(inputs, expected, concurrent_pipeline, graph_assertions)

    def run_custom_schedule(
        self,
        pipeline: Callable[["GraphTestBuilder"], list[PipelineStep]],
        inputs: list[torch.Tensor] | Callable[["GraphTestBuilder"], list[torch.Tensor]],
        expected: list[torch.Tensor]
        | Callable[["GraphTestBuilder"], list[torch.Tensor]],
        graph_assertions: Callable[["GraphTestBuilder"], None] | None = None,
    ) -> None:
        """
        Execute the test with a custom replay pipeline.

        Args:
            pipeline: Function that returns the pipeline steps for replay
            inputs: Input tensors, or a callable that returns them (for rank-dependent inputs)
            expected: Expected output tensors, or a callable that returns them
            graph_assertions: Optional callback to assert properties of captured graph infos
        """
        self._run(inputs, expected, pipeline, graph_assertions)


class CudaGraphTestBase(unittest.TestCase):
    """Base class with shared infrastructure for CUDA graph tests."""

    NUM_REPLAYS = 3
    NUM_OPS = 5
    NUM_GRAPHS = 3

    def setUp(self) -> None:
        self.backend = os.environ.get("TEST_BACKEND", "")
        rank, _ = get_rank_and_size()
        device_count = torch.cuda.device_count()
        self.device = torch.device("cuda", rank % device_count)
        torch.cuda.set_device(self.device)

    def tearDown(self) -> None:
        torch.accelerator.synchronize()

    _comm_creation_counter: int = 0

    @contextmanager
    def create_comms(
        self, num_comms: int
    ) -> Generator[list[torchcomms.TorchComm], None, None]:
        """Context manager that creates and finalizes comms."""
        self._comm_creation_counter += 1
        comms = [
            torchcomms.new_comm(
                self.backend,
                self.device,
                # prevents collision between subtests
                name=f"{self._testMethodName}_creation{self._comm_creation_counter}_comm{i}",
            )
            for i in range(num_comms)
        ]
        try:
            yield comms
        finally:
            for comm in comms:
                comm.finalize()

    @contextmanager
    def create_graphs(
        self, num_graphs: int
    ) -> Generator[list[torch.cuda.CUDAGraph], None, None]:
        """Context manager that creates and resets CUDA graphs."""
        graphs = [torch.cuda.CUDAGraph(keep_graph=True) for _ in range(num_graphs)]
        try:
            yield graphs
        finally:
            for graph in graphs:
                graph.reset()

    def run_graph_pipeline(
        self,
        steps: list[PipelineStep],
    ) -> None:
        """
        Run a pipeline of graphs and non-graphable callbacks with proper synchronization.

        Each step can be:
        - A CUDAGraph: replays the graph on the default stream
        - A (graph, stream) tuple: replays the graph on the specified stream
        - A (callable, stream) tuple: runs the callable on the specified stream
        - A callable: runs non-graphable code (after synchronizing with previous step)
        - A list of the above substeps: runs them concurrently (fork-join pattern)

        Events are used to chain dependencies between steps.
        Bare callables (not in a tuple) trigger a full synchronization.

        Example:
            self.run_graph_pipeline([
                graph0,                                    # Run on default stream
                [(graph1, stream1), (graph2, stream2)],    # Run concurrently
                [(graph1, stream1), (nongraph_fn, stream2)],  # Graph + callable concurrent
                lambda: inp2.copy_(inp1),                  # Non-graphable: full sync first
                (graph3, stream3),                         # Run on stream3
            ])
        """
        prev_events: list[torch.cuda.Event] = []
        default_stream: torch.cuda.Stream = torch.cuda.current_stream()

        def run_substep(substep: _Substep) -> torch.cuda.Event:
            event = torch.cuda.Event()
            if isinstance(substep, torch.cuda.CUDAGraph):
                for prev_event in prev_events:
                    default_stream.wait_event(prev_event)
                substep.replay()
                event.record(default_stream)
            else:
                item, stream = substep
                with torch.cuda.stream(stream):
                    for prev_event in prev_events:
                        stream.wait_event(prev_event)
                    if isinstance(item, torch.cuda.CUDAGraph):
                        item.replay()
                    else:
                        item()
                    event.record(stream)
            return event

        for step in steps:
            if isinstance(step, list):
                # Concurrent group - fork: run all in parallel, join: collect events
                prev_events = [run_substep(substep) for substep in step]
            elif callable(step):
                # Non-graphable callback - need full sync before running
                torch.cuda.synchronize()
                step()
                prev_events = []
            else:
                # Single substep (graph or tuple)
                prev_events = [run_substep(step)]

        torch.cuda.synchronize()
