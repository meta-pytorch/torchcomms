#!@PYTHON_EXECUTABLE@

"""
Display help:

    @PYTHON_EXECUTABLE@ ./@FILENAME@ -h
    @PYTHON_EXECUTABLE@ ./@FILENAME@ timemory-config -h

Usage:

    @PYTHON_EXECUTABLE@ ./@FILENAME@
    @PYTHON_EXECUTABLE@ ./@FILENAME@ -n 10
    @PYTHON_EXECUTABLE@ ./@FILENAME@ -f text
    @PYTHON_EXECUTABLE@ ./@FILENAME@ -f json_flat
    @PYTHON_EXECUTABLE@ ./@FILENAME@ -f json_tree
    @PYTHON_EXECUTABLE@ ./@FILENAME@ timemory-config --enabled=n
    @PYTHON_EXECUTABLE@ ./@FILENAME@ timemory-config --auto-output=y --cout-output=n

"""

import os
import json
import argparse
import timemory
from timemory.profiler import profile
from timemory.bundle import auto_timer, auto_tuple, marker


def get_config(items=["wall_clock", "cpu_clock", "user_global_bundle"]):
    return [getattr(timemory.component, x) for x in items]


def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


@profile(["wall_clock", "peak_rss", "global_bundle"])
def run_profile(n):
    """Run full profiler"""
    return fib(int(n))


@auto_timer()
def run_auto_timer(n):
    """Decorator and context manager for high-level pre-defined collection"""
    fib(n)
    with auto_timer(key="auto_timer_ctx_manager"):
        fib(n)


@marker(["wall_clock", "peak_rss", "global_bundle"])
def run_marker(n):
    """Decorator and context manager for high-level custom collection"""
    fib(n)
    with auto_tuple(get_config(), key="auto_tuple_ctx_manager"):
        fib(n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--nfib", type=int, default=10, help="Fibonacci depth"
    )
    parser.add_argument(
        "-o", "--output-name", type=str, default=None, help="Output filename"
    )
    parser.add_argument(
        "-t",
        "--types",
        type=str,
        nargs="*",
        default=[],
        help="Component types to report",
    )
    parser.add_argument(
        "-f",
        "--formats",
        type=str,
        nargs="*",
        choices=("text", "json_flat", "json_tree"),
        default=["json_tree"],
        help=(
            'Data reporting format. In a "flat" json, all entries are in a 1D array'
            'and labels are modified with indentation to reflect hierarchy. In a "tree" '
            "json, entries are arranged in a hierarchical and the labels are unmodified."
        ),
    )

    # disable output in timemory_finalize by default
    # can override with command-line:
    #   '... timemory-config --auto-output=y ...'
    timemory.settings.auto_output = False
    # disable console output by default
    timemory.settings.cout_output = False

    # demonstrate how to add command-line support
    # The subparser stuff is not strictly necessary: if a
    # subparser is not provided, the default behavior is
    # to do this internally. If you want to disable the
    # subparser behavior, pass 'subparser=None' or
    # subparser=False
    subparser = parser.add_subparsers()
    timemory.add_arguments(
        parser,
        subparser=subparser.add_parser(
            "timemory-config",
            description="Configure settings for timemory",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        ),
    )

    args = parser.parse_args()
    n = args.nfib

    run_profile(n)
    run_auto_timer(n)
    run_marker(n)

    # write the output
    def write(output_name, data, ext):
        if not data:
            return
        if output_name is not None:
            fname = os.path.join(
                timemory.settings.output_path, "{}.{}".format(output_name, ext)
            )
            with open(fname, "w") as f:
                print("[timemory]> Outputting {}...".format(fname))
                f.write(data)
        else:
            print("\n{}".format(data))

    # Get the results as text table
    if "text" in args.formats:
        data = timemory.get_text(components=args.types)
        write(args.output_name, data, "txt")

    # Get the results as a JSON where all entries are
    # in a 1D JSON array and the labels (i.e. prefix)
    # are modified with indentation to reflect hierarchy
    if "json_flat" in args.formats:
        data = timemory.get(hierarchy=False, components=args.types)
        write(
            args.output_name,
            "{}".format(json.dumps(data, indent=2, sort_keys=True)),
            "json",
        )

    # Get the results as a JSON where all entries are
    # in a hierarchical JSON and the labels (i.e. prefix)
    # are unmodified. This data layout includes inclusive
    # and exclusive values
    if "json_tree" in args.formats:
        data = timemory.get(hierarchy=True, components=args.types)
        write(
            args.output_name,
            "{}".format(json.dumps(data, indent=2, sort_keys=False)),
            "tree.json",
        )

    print("")

    # Generate output (potentially)
    timemory.finalize()
