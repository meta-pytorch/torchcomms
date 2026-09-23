#!@PYTHON_EXECUTABLE@

import os
import sys
import argparse
import traceback


"""
A) Run a process and report timemory's timing and memory resuls

    Usage:
        timem ls -l

B) Run a process and pass a JSON serialization to a custom
   'timemory_json_handler.py' script so that the data can be processed

    Usage:
        timem --enable-json-handler --timem-quiet sleep 3
"""


# ----------------------------------------------------------------------------------------#
#   Allow some options
#
def handle_arguments():
    """
    Processes options for timem and removes known
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable-json-handler",
        required=False,
        action="store_true",
        dest="enable_handler",
        help="Enable:\n\t{}\n\t{} to handle timing info".format(
            "import timemory_json_handler",
            "timemory_json_handler.receive(timemory.json())",
        ),
    )
    parser.add_argument(
        "--timem-quiet",
        required=False,
        action="store_true",
        dest="quiet",
        help="Suppress reporting to stdout",
    )
    parser.add_argument(
        "--debug",
        required=False,
        action="store_true",
        dest="debug",
        help="Enable debug output",
    )

    parser.set_defaults(enable_handler=False)
    parser.set_defaults(quiet=False)
    parser.set_defaults(debug=False)

    args, left = parser.parse_known_args()
    # replace sys.argv with unknown args only
    sys.argv = sys.argv[:1] + left

    return args


# ----------------------------------------------------------------------------------------#
#   Set the environment
#
def set_environ(field, default_val):
    if os.environ.get(field) is None:
        os.environ[field] = "{}".format(default_val)


# ----------------------------------------------------------------------------------------#
#   Main execution
#
if __name__ == "__main__":
    ret = 0
    try:
        # ----------------------------------------------------------------------#
        #   default output formatting
        #
        set_environ("TIMEMORY_PRECISION", 6)
        set_environ("TIMEMORY_WIDTH", 12)
        set_environ("TIMEMORY_SCIENTIFIC", "OFF")
        set_environ("TIMEMORY_FILE_OUTPUT", "OFF")
        set_environ("TIMEMORY_BANNER", "OFF")
        set_environ("TIMEMORY_AUTO_OUTPUT", "OFF")
        set_environ("TIMEMORY_FILE_OUTPUT", "OFF")
        set_environ("TIMEMORY_TEXT_OUTPUT", "OFF")
        set_environ("TIMEMORY_SKIP_FINALIZE", "ON")

        # ----------------------------------------------------------------------#
        #   parse arguments
        #
        args = handle_arguments()

        # ----------------------------------------------------------------------#
        #   import and setup
        #
        import timemory
        import subprocess as sp

        # grab a manager handle
        timemory_manager = timemory.manager()
        timemory_manager.set_write_metadata(-1)

        # rusage records child processes
        timemory.set_rusage_children()

        components = []
        for c in [
            "wall_clock",
            "user_clock",
            "sys_clock",
            "cpu_clock",
            "cpu_util",
            "peak_rss",
            "num_minor_page_faults",
            "num_major_page_faults",
            "voluntary_context_switch",
            "priority_context_switch",
        ]:
            components.append(getattr(timemory.component, c))

        # run the command
        exe_name = "[pytimem]"
        if len(sys.argv) > 1:
            exe_name = "[{}]".format(sys.argv[1])

        comp = timemory.component_tuple(components, "|")
        if len(sys.argv) > 1:
            comp.start()
            p = sp.Popen(sys.argv[1:])
            ret = p.wait()
            comp.stop()
        else:
            comp.start()
            comp.stop()

        report = (
            "{}".format(comp).strip().strip(">>>").strip().strip("|").strip()
        )
        report = report.replace(": ", "", 1)
        suffix_report = report.split(",")
        _sorted = []

        def _insert(key):
            for itr in suffix_report:
                if key in itr:
                    _sorted.append(itr)
                    suffix_report.remove(itr)
                    return

        _insert(" wall")
        _insert(" sys")
        _insert(" user")
        _insert(" cpu")
        _insert(" cpu_util")
        _insert(" peak")
        _insert(" prio_")
        _insert(" vol_")
        _insert(" major_")
        _insert(" minor_")
        for itr in suffix_report:
            _sorted.append(itr)
        suffix_report = _sorted

        strings = [f"\n{exe_name}> Measurement totals:"]
        for itr in suffix_report:
            if "laps" not in itr:
                data = itr.strip().split()
                if len(data) > 1:
                    entry = "{:>16} {:<16}".format(data[0], " ".join(data[1:]))
                else:
                    entry = "{:>16}".format(" ".join(data))
                strings.append("    {}".format(entry))

        for s in strings:
            print("{}".format(s))

        # generate report
        # timemory.report()

        # ----------------------------------------------------------------------#
        #   handler
        #
        if args.enable_handler:
            try:
                import timemory_json_handler

                timemory_json_handler.receive(
                    sys.argv[1:], timemory_manager.json()
                )

            except Exception as e:
                sys.stderr.write(f"{e}\n")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback)

    except Exception as e:
        sys.stderr.write(f"{e}\n")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.exit(ret)
