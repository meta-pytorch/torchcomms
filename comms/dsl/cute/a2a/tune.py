# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""CLI entrypoint: tune the comms/dsl CuTe all_to_all via the comm tuner.

The CuTe twin of ``triton/a2a/tune.py`` -- the body of the tuning-job entrypoint that the
MAST/conda launcher submits and the ``comm_tuning`` engine re-execs per candidate. The
default ``A2A(backend="cute")`` (identity copy hook) tunes the plain CuTe all_to_all:

    buck2 run @fbcode//mode/opt //comms/dsl:tune_a2a_cute -- --mode parent --output-dir /tmp/t ...

Run modes (parent/child/select) and ``--max-sizes`` / ``--max-candidates`` are parsed by
``run_tuning_cli`` via the adapter.
"""

from comms.dsl.collectives import A2A


def main() -> None:
    A2A(backend="cute").autotune()


if __name__ == "__main__":
    main()
