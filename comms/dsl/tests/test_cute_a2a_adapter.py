# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# GPU adapter test: imports the untyped cutlass DSL and constructs adapters over untyped cute
# handles that pyre cannot model, so strict typing adds no value here.

"""GPU tests for the comms/dsl CuTe all_to_all tuning adapter + backend contract.

The CuTe twin of the GPU half of ``test_a2a_adapter``. The CuTe adapter imports the cutlass
DSL, so the pure-CPU adapter surface has no CuTe analogue; the cutlass-free config/key/lookup
layer is covered by ``test_cute_tuned_roundtrip``. This suite covers the parts that need a GPU:

* **Adapter round trip** (2-rank GPU) -- ``make_inputs`` / ``run_candidate`` / ``run_baseline``
  / ``check_correctness`` plus the shared-key rule (adapter key == runtime
  ``make_a2a_key``), proving a tuned CuTe config drives the CuTe kernel correctly.

The fused CuTe staging schedules' produce/consume hook + ``rows>0`` transpose support -- and the
TMA / zero-copy paths that reject those transforms -- are covered bit-exact by
``test_a2a_cute_variants``.

Gold is ``dist.all_to_all_single``. Skipped unless GPUs are present. The cutlass-importing
adapter / host imports are deferred into the test bodies so collection stays clean without a
GPU.
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# cutlass-free; safe to import at module scope (see test_cute_tuned_roundtrip).
from comms.dsl.cute.a2a.tuning import make_a2a_key
from comms.dsl.tests._dist_harness import _find_free_port


def _gpu_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    from comms.dsl.cute.a2a.adapter import CuteA2AInputSpec, CuteA2ATuningAdapter

    a = CuteA2ATuningAdapter()
    spec = CuteA2AInputSpec(numel=8192, dtype=torch.bfloat16)
    inputs = a.make_inputs(spec, rank=rank, world_size=world_size, device=device)

    # shared-key rule: the adapter key must equal the runtime key.
    rt_key = make_a2a_key(inputs.input, inputs.transport, rows=spec.rows)
    assert a.make_key(spec, world_size) == rt_key, "adapter/runtime key mismatch"

    group = dist.group.WORLD
    assert group is not None, "distributed not initialized"
    cfg = a.enumerate_candidate_configs(spec, rt_key)[0]
    cand = a.run_candidate(inputs, cfg, group)
    ref = a.run_baseline(inputs, "nccl", group)
    torch.cuda.synchronize()
    a.check_correctness(cand, ref)

    dist.barrier()
    dist.destroy_process_group()


class A2ACuteAdapterGpuTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs >=2 GPUs")
    def test_adapter_roundtrip(self) -> None:
        ws = min(4, torch.cuda.device_count())
        mp.spawn(_gpu_worker, args=(ws, _find_free_port()), nprocs=ws, join=True)
