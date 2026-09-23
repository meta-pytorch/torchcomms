#!@PYTHON_EXECUTABLE@

import sys
import numpy
import argparse

use_mpi = True
try:
    import mpi4py  # noqa: F401
    from mpi4py import MPI  # noqa: F401
    from mpi4py.MPI import Exception as MPIException  # noqa: F401
except ImportError:
    use_mpi = False
    MPIException = RuntimeError
    pass

import timemory  # noqa: E402
from timemory.profiler import profile  # noqa: E402
from timemory.tools import function_wrappers  # noqa: E402
import libex_python_bindings as ex_bindings  # noqa: E402

if use_mpi:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    comm = None
    rank = 0
    size = 1

if size > 2:
    raise RuntimeError("This example only supports two MPI procs")


def run_profile(nitr=100, nsize=1000000):
    return ex_bindings.run(nitr, nsize)


def run_mpi(nitr=100, nsize=1000000):
    if use_mpi is False:
        _sum = 0.0
        for i in range(nitr):
            data = numpy.arange(nsize, dtype="i")
            _val = numpy.sum(data)
            _sum += 1.0 / _val
            data = numpy.arange(nsize, dtype=numpy.float64)
            _val = numpy.sum(data)
            _sum += 1.0 / _val

    msgs = set()
    for i in range(nitr):
        # passing MPI datatypes explicitly
        try:
            if rank == 0:
                data = numpy.arange(nsize, dtype="i")
                comm.Send([data, MPI.INT], dest=1, tag=77)
            elif rank == 1:
                data = numpy.empty(nsize, dtype="i")
                comm.Recv([data, MPI.INT], source=0, tag=77)
        except MPIException as e:
            msgs.add(f"{e}")

        # automatic MPI datatype discovery
        try:
            if rank == 0 and size == 2:
                data = numpy.empty(nsize, dtype=numpy.float64)
                comm.Recv(data, source=1, tag=13)
            elif rank == 1:
                data = numpy.arange(nsize, dtype=numpy.float64)
                comm.Send(data, dest=0, tag=13)
        except MPIException as e:
            msgs.add(f"{e}")

    for i, itr in enumerate(msgs):
        sys.stderr.write("{}: {}\n".format(i, itr))


def main(args):
    # start function wrappers (MPI, OpenMP, etc. if available)
    with function_wrappers(*args.profile, nccl=False):
        return run_profile(args.iterations, args.size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iterations",
        default=100,
        type=int,
        help="Iterations",
    )
    parser.add_argument(
        "-n",
        "--size",
        default=1000000,
        type=int,
        help="Array size",
    )
    parser.add_argument(
        "-c",
        "--components",
        default=[
            "wall_clock",
            "peak_rss",
            "cpu_clock",
            "cpu_util",
            "thread_cpu_clock",
            "thread_cpu_util",
        ],
        type=str,
        help="Additional components",
        nargs="*",
    )
    parser.add_argument(
        "-p",
        "--profile",
        default=["mpi", "openmp", "malloc"],
        choices=("mpi", "openmp", "malloc", "nccl"),
        type=str,
        help="Profiling library wrappers to activate",
        nargs="*",
    )

    args = parser.parse_args()
    timemory.enable_signal_detection()
    timemory.settings.width = 8
    timemory.settings.precision = 2
    timemory.settings.scientific = True
    timemory.settings.plot_output = True
    timemory.settings.dart_output = False
    timemory.timemory_init([__file__])

    @function_wrappers(*args.profile, nccl=False)
    def runner(nitr, nsize):
        run_mpi(nitr, nsize)

    runner(args.iterations, args.size)

    with profile(args.components):
        ans = main(args)

    print("Success! Answer = {}. Finalizing...".format(ans))
    timemory.finalize()
    print("Python Finished")
