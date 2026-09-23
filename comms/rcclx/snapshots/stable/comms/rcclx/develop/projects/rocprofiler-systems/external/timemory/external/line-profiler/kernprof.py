#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Script to conveniently run profilers on code in a variety of circumstances.
"""

import functools
import optparse
import os
import sys
import timemory

PY3 = sys.version_info[0] == 3

# Guard the import of cProfile such that 3.x people
# without lsprof can still use this script.
try:
    from cProfile import Profile
except ImportError:
    try:
        from lsprof import Profile
    except ImportError:
        from profile import Profile


# Python 3.x compatibility utils: execfile
# ========================================
try:
    execfile
except NameError:
    # Python 3.x doesn't have 'execfile' builtin
    import builtins

    exec_ = getattr(builtins, "exec")

    def execfile(filename, globals=None, locals=None):
        with open(filename, "rb") as f:
            exec_(compile(f.read(), filename, "exec"), globals, locals)


# =====================================


CO_GENERATOR = 0x0020


def is_generator(f):
    """Return True if a function is a generator."""
    isgen = (f.__code__.co_flags & CO_GENERATOR) != 0
    return isgen


class ContextualProfile(Profile):
    """A subclass of Profile that adds a context manager for Python
    2.5 with: statements and a decorator.
    """

    def __init__(self, *args, **kwds):
        super(ContextualProfile, self).__init__(*args, **kwds)
        self.enable_count = 0

    def enable_by_count(self, subcalls=True, builtins=True):
        """Enable the profiler if it hasn't been enabled before."""
        if self.enable_count == 0:
            self.enable(subcalls=subcalls, builtins=builtins)
        self.enable_count += 1

    def disable_by_count(self):
        """Disable the profiler if the number of disable requests matches the
        number of enable requests.
        """
        if self.enable_count > 0:
            self.enable_count -= 1
            if self.enable_count == 0:
                self.disable()

    def __call__(self, func):
        """Decorate a function to start the profiler on function entry and stop
        it on function exit.
        """
        # FIXME: refactor this into a utility function so that both it and
        # line_profiler can use it.
        if is_generator(func):
            wrapper = self.wrap_generator(func)
        else:
            wrapper = self.wrap_function(func)
        return wrapper

    # FIXME: refactor this stuff so that both LineProfiler and
    # ContextualProfile can use the same implementation.
    def wrap_generator(self, func):
        """Wrap a generator to profile it."""

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            g = func(*args, **kwds)
            # The first iterate will not be a .send()
            self.enable_by_count()
            try:
                item = next(g)
            except StopIteration:
                return
            finally:
                self.disable_by_count()
            input = yield item
            # But any following one might be.
            while True:
                self.enable_by_count()
                try:
                    item = g.send(input)
                except StopIteration:
                    return
                finally:
                    self.disable_by_count()
                input = yield item

        return wrapper

    def wrap_function(self, func):
        """Wrap a function to profile it."""

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            self.enable_by_count()
            try:
                result = func(*args, **kwds)
            finally:
                self.disable_by_count()
            return result

        return wrapper

    def __enter__(self):
        self.enable_by_count()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable_by_count()


def find_script(script_name):
    """Find the script.

    If the input is not a file, then $PATH will be searched.
    """
    if os.path.isfile(script_name):
        return script_name
    path = os.getenv("PATH", os.defpath).split(os.pathsep)
    for dir in path:
        if dir == "":
            continue
        fn = os.path.join(dir, script_name)
        if os.path.isfile(fn):
            return fn

    sys.stderr.write("Could not find script %s\n" % script_name)
    raise SystemExit(1)


def parse_args(args):
    usage = "%prog [-s setupfile] [-o output_file_path] [-l] [-b] [-v] [-c component] [--] scriptfile [arg] ..."
    component_choices = timemory.component.get_available_types(arithmetic=True)
    component_choices = [x for x in component_choices
        if not x.startswith("data_tracker") and "gotcha" not in x]
    version = timemory.version
    parser = optparse.OptionParser(usage=usage, version="%prog {}".format(version))
    parser.allow_interspersed_args = False
    parser.add_option(
        "-l",
        "--line-by-line",
        action="store_true",
        help="Use the line-by-line profiler from the line_profiler module "
        "instead of Profile. Implies --builtin.",
    )
    parser.add_option(
        "-b",
        "--builtin",
        action="store_true",
        help="Put 'profile' in the builtins. Use 'profile.enable()' and "
        "'profile.disable()' in your code to turn it on and off, or "
        "'@profile' to decorate a single function, or 'with profile:' "
        "to profile a single section of code.",
    )
    parser.add_option("-o", "--outfile", default=None, help="Save stats to <outfile>")
    parser.add_option(
        "-s", "--setup", default=None, help="Code to execute before the code to profile"
    )
    parser.add_option(
        "-v",
        "--view",
        action="store_true",
        help="View the results of the profile in addition to saving it.",
    )
    parser.add_option(
        "-c",
        "--component",
        default="wall_clock",
        choices=component_choices,
        help="Set the performance analysis component: {}".format(
            ", ".join([f"'{x}'" for x in component_choices])),
    )
    parser.add_option(
        "-p",
        "--num-preallocated",
        type="int",
        default=0,
        help=(
            "Set the number of pre-allocated component entries for each function. "
            "Enable this option for better performance at the cost of using more memory"
        ),
    )

    if not sys.argv[1:]:
        parser.print_usage()
        sys.exit(2)

    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv

    #
    options = None
    args = None
    if "--" in sys.argv:
        _idx = sys.argv.index("--")
        _argv = sys.argv[_idx + 1 :]
        options, argv = parse_args(sys.argv[1:_idx])
        args = _argv[:]
    else:
        _argv = sys.argv[1:]
        if "-h" in _argv or "--help" in _argv:
            options, args = parse_args(["--help"])
        else:
            options, args = parse_args([])
        args = _argv[:]

    try:
        from ..libs.libpytimemory import initialize

        if os.path.isfile(args[0]):
            initialize(args)
    except ImportError:
        pass

    if not options.outfile:
        if options.line_by_line:
            extension = "lprof"
        else:
            extension = "prof"
        options.outfile = "%s.%s" % (os.path.basename(args[0]), extension)

    sys.argv[:] = args
    if options.setup is not None:
        # Run some setup code outside of the profiler. This is good for large
        # imports.
        setup_file = find_script(options.setup)
        __file__ = setup_file
        __name__ = "__main__"
        # Make sure the script's directory is on sys.path instead of just
        # kernprof.py's.
        sys.path.insert(0, os.path.dirname(setup_file))
        ns = locals()
        execfile(setup_file, ns, ns)

    from timemory.component import get_generator
    from ._line_profiler import set_component_generator
    from ._line_profiler import set_component_prealloc
    from . import line_profiler

    generator = get_generator(options.component)
    if generator is None:
        msg = "Error! timemory.component.get_generator produced no generator for {}. ".format(
            options.component
        )
        msg += (
            "Either the component is not available or there is no component by that ID"
        )
        raise RuntimeError(msg)
    set_component_generator(generator)
    set_component_prealloc(options.num_preallocated)

    prof = line_profiler.LineProfiler()
    options.builtin = True
    if options.builtin:
        if PY3:
            import builtins
        else:
            import __builtin__ as builtins
        builtins.__dict__["profile"] = prof
        builtins.__dict__["line_profile"] = prof

    script_file = find_script(sys.argv[0])
    __file__ = script_file
    __name__ = "__main__"
    # Make sure the script's directory is on sys.path instead of just
    # kernprof.py's.
    sys.path.insert(0, os.path.dirname(script_file))

    try:
        try:
            execfile_ = execfile
            ns = locals()
            if options.builtin:
                execfile(script_file, ns, ns)
            else:
                prof.runctx("execfile_(%r, globals())" % (script_file,), ns, ns)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            timemory.finalize()
    finally:
        prof.dump_stats(options.outfile)
        print("Wrote profile results to %s" % options.outfile)
        if options.view:
            prof.print_stats()
        del prof


if __name__ == "__main__":
    sys.exit(main(sys.argv))
