#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import hatchet as ht

if __name__ == "__main__":
    report_choices = ("category", "process", "threads", "track_ids", "profile")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--metric", default=None, type=str, help="Metric to display"
    )
    parser.add_argument(
        "--print-dataframe", action="store_true", help="Print the dataframe"
    )
    parser.add_argument(
        "--quiet-tree", action="store_true", help="Do not print the tree"
    )
    parser.add_argument(
        "-t", "--thread", default=0, type=int, help="Thread ID to print"
    )
    parser.add_argument(
        "-r", "--rank", default=0, type=int, help="Process rank ID to print"
    )
    parser.add_argument(
        "-p", "--precision", default=6, type=int, help="Data value precision"
    )
    parser.add_argument(
        "-d", "--max-depth", default=None, type=int, help="Max call-stack depth"
    )
    parser.add_argument(
        "-c",
        "--context",
        default="file",
        type=str,
        help="Context column printed after function name",
    )
    parser.add_argument(
        "-I",
        "--include",
        default=[],
        type=str,
        nargs="+",
        help="Restrict data to the given set of categories",
    )
    parser.add_argument(
        "-E",
        "--exclude",
        default=[],
        type=str,
        nargs="+",
        help="Exclude data from the given set of categories",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        type=int,
        help="Increase verbosity while processing",
    )
    parser.add_argument(
        "--report",
        default=["profile"],
        nargs="*",
        type=str,
        choices=("all", "none") + report_choices,
        help="Increase verbosity while processing",
    )
    parser.add_argument(
        "--squash",
        action="store_true",
        help="Squash the graphframe before printing tree",
    )
    parser.add_argument(
        "--use-attributes",
        action="store_true",
        help="Read the data and then use the injected attributes to reread the data",
    )
    args, argv = parser.parse_known_args()

    _report = [] if "none" in args.report else args.report
    if "all" in _report:
        _report = list(report_choices)

    files = argv[:]
    if not files:
        this_dir = os.path.dirname(__file__)
        data_path = os.path.join(
            this_dir, "../../../hatchet/tests/data/perfetto/*.proto"
        )
        files = glob.glob(data_path)
        if not files:
            sys.stderr.write(
                f"Provide data files. Data files not found in: {data_path}\n"
            )
            sys.exit(1)

    print("")

    # Use hatchet's ``from_perfetto`` API for the protobuf input files.
    # The result is stored into Hatchet's GraphFrame.
    gf = ht.GraphFrame.from_omnitrace(
        files,
        include_category=[] if args.use_attributes else args.include,
        exclude_category=[] if args.use_attributes else args.exclude,
        max_depth=args.max_depth,
        verbose=args.verbose,
        report=_report,
    )

    if args.use_attributes:
        old_gf = gf.deepcopy()
        print(f"old categories: {old_gf.selected_categories()}")
        print(f"old df categories: {old_gf.available_categories()}")

        gf = gf.read(
            include_category=args.include, exclude_category=args.exclude, verbose=0
        )
        print(f"new categories: {gf.selected_categories()}")
        print(f"new df categories: {gf.available_categories()}")

        if old_gf.selected_categories() != gf.selected_categories():
            raise RuntimeError("categories not retained")
    else:
        old_gf = None

    if args.squash:
        gf = gf.squash()
        if old_gf:
            old_gf = old_gf.squash()

    # Printout the DataFrame component of the GraphFrame.
    if args.print_dataframe:
        if old_gf:
            print(f"\nold dataframe:\n{old_gf.dataframe.to_string()}\n")
        print(f"\ndataframe:\n{gf.dataframe.to_string()}\n")

    if not args.quiet_tree:
        if old_gf:
            print(
                "\nold tree (len={}):\n{}".format(
                    len(old_gf.dataframe),
                    old_gf.tree(
                        args.metric,
                        # context_column=args.context,
                        # rank=args.rank,
                        # thread=args.thread,
                        precision=args.precision,
                    ),
                )
            )

        print(
            "\ntree (len={}):\n{}".format(
                len(gf.dataframe),
                gf.tree(
                    args.metric,
                    context_column=args.context,
                    rank=args.rank,
                    thread=args.thread,
                    precision=args.precision,
                ),
            )
        )
