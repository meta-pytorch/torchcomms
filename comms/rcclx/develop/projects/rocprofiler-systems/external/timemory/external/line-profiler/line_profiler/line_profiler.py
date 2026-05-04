#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__version__ = '3.0.2'

try:
    import cPickle as pickle
except ImportError:
    import pickle

import functools
import inspect
import linecache
import optparse
import os
import sys

from ._line_profiler import LineProfiler as CLineProfiler

# Python 2/3 compatibility utils
# ===========================================================
PY3 = sys.version_info[0] == 3
PY35 = PY3 and sys.version_info[1] >= 5

# exec (from https://bitbucket.org/gutworth/six/):
if PY3:
    import builtins
    exec_ = getattr(builtins, "exec")
    del builtins
else:
    def exec_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")

if PY35:
    import inspect

    def is_coroutine(f):
        return inspect.iscoroutinefunction(f)
else:
    def is_coroutine(f):
        return False


# ============================================================
CO_GENERATOR = 0x0020


def is_generator(f):
    """ Return True if a function is a generator.
    """
    isgen = (f.__code__.co_flags & CO_GENERATOR) != 0
    return isgen


class LineProfiler(CLineProfiler):
    """ A profiler that records a metric on individual lines.
    """

    def __call__(self, func):
        """ Decorate a function to start the profiler on function entry and stop
        it on function exit.
        """
        self.add_function(func)
        if is_coroutine(func):
            wrapper = self.wrap_coroutine(func)
        elif is_generator(func):
            wrapper = self.wrap_generator(func)
        else:
            wrapper = self.wrap_function(func)
        return wrapper

    def wrap_generator(self, func):
        """ Wrap a generator to profile it.
        """
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
            input = (yield item)
            # But any following one might be.
            while True:
                self.enable_by_count()
                try:
                    item = g.send(input)
                except StopIteration:
                    return
                finally:
                    self.disable_by_count()
                input = (yield item)
        return wrapper

    def wrap_function(self, func):
        """ Wrap a function to profile it.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            self.enable_by_count()
            try:
                result = func(*args, **kwds)
            finally:
                self.disable_by_count()
            return result
        return wrapper

    if PY35:
        from . import line_profiler_py35
        wrap_coroutine = line_profiler_py35.wrap_coroutine

    def dump_stats(self, filename):
        """ Dump a representation of the data to a file as a pickled LineStats
        object from `get_stats()`.
        """
        lstats = self.get_stats()
        with open(filename, 'wb') as f:
            pickle.dump(lstats, f, pickle.HIGHEST_PROTOCOL)

    def print_stats(self, stream=None, output_unit=None, stripzeros=False):
        """ Show the gathered statistics.
        """
        lstats = self.get_stats()
        show_text(lstats.metrics, lstats.unit, lstats.display_unit,
                  output_unit=output_unit, stream=stream, stripzeros=stripzeros)

    def run(self, cmd):
        """ Profile a single executable statment in the main namespace.
        """
        import __main__
        main_dict = __main__.__dict__
        return self.runctx(cmd, main_dict, main_dict)

    def runctx(self, cmd, globals, locals):
        """ Profile a single executable statement in the given namespaces.
        """
        self.enable_by_count()
        try:
            exec_(cmd, globals, locals)
        finally:
            self.disable_by_count()
        return self

    def runcall(self, func, *args, **kw):
        """ Profile a single function call.
        """
        self.enable_by_count()
        try:
            return func(*args, **kw)
        finally:
            self.disable_by_count()

    def add_module(self, mod):
        """ Add all the functions in a module and its classes.
        """
        from inspect import isclass, isfunction

        nfuncsadded = 0
        for item in mod.__dict__.values():
            if isclass(item):
                for k, v in item.__dict__.items():
                    if isfunction(v):
                        self.add_function(v)
                        nfuncsadded += 1
            elif isfunction(item):
                self.add_function(item)
                nfuncsadded += 1

        return nfuncsadded


def show_func(filename, start_lineno, func_name, metrics, unit,
              display_unit, output_unit=None, stream=None, stripzeros=False):
    """ Show results for a single function.
    """
    if stream is None:
        stream = sys.stdout

    template = '%6s %9s %12s %8s %8s  %-s'
    d = {}
    total_metric = 0.0
    linenos = []
    for lineno, nhits, metric in metrics:
        total_metric += metric
        linenos.append(lineno)

    if stripzeros and total_metric == 0:
        return

    if output_unit is None:
        output_unit = unit
    scalar = unit / output_unit

    stream.write("Total metric: %g %s\n" % (total_metric, display_unit))
    if os.path.exists(filename) or filename.startswith("<ipython-input-"):
        stream.write("File: %s\n" % filename)
        stream.write("Function: %s at line %s\n" % (func_name, start_lineno))
        if os.path.exists(filename):
            # Clear the cache to ensure that we get up-to-date results.
            linecache.clearcache()
        all_lines = linecache.getlines(filename)
        sublines = inspect.getblock(all_lines[start_lineno-1:])
    else:
        stream.write("\n")
        stream.write("Could not find file %s\n" % filename)
        stream.write(
            "Are you sure you are running this program from the same directory\n")
        stream.write("that you ran the profiler from?\n")
        stream.write("Continuing without the function's contents.\n")
        # Fake empty lines so we can see the metrics, if not the code.
        nlines = max(linenos) - min(min(linenos), start_lineno) + 1
        sublines = [''] * nlines
    for lineno, nhits, metric in metrics:
        perc = 0.0 if not total_metric > 0 else (100 * metric / total_metric)
        d[lineno] = (nhits,
                     '%5.1f' % (metric * scalar),
                     '%5.1f' % (float(metric) * scalar / nhits),
                     '%5.1f' % (perc))
    linenos = range(start_lineno, start_lineno + len(sublines))
    empty = ('', '', '', '')
    header = template % ('Line #', 'Hits', 'Metric', 'Per Hit', '% Metric',
                         'Line Contents')
    stream.write("\n")
    stream.write(header)
    stream.write("\n")
    stream.write('=' * len(header))
    stream.write("\n")
    for lineno, line in zip(linenos, sublines):
        nhits, metric, per_hit, percent = d.get(lineno, empty)
        txt = template % (lineno, nhits, metric, per_hit, percent,
                          line.rstrip('\n').rstrip('\r'))
        stream.write(txt)
        stream.write("\n")
    stream.write("\n")


def show_text(stats, unit, display_unit, output_unit=None, stream=None, stripzeros=False):
    """ Show text for the given metrics.
    """
    if stream is None:
        stream = sys.stdout

    if output_unit is not None:
        stream.write('Metric unit: %g (%s)\n\n' % (output_unit, display_unit))
    else:
        stream.write('Metric unit: %g %s\n\n' % (unit, display_unit))

    for (fn, lineno, name), metrics in sorted(stats.items()):
        show_func(fn, lineno, name, stats[fn, lineno, name], unit, display_unit,
                  output_unit=output_unit, stream=stream, stripzeros=stripzeros)


def load_ipython_extension(ip):
    """ API for IPython to recognize this module as an IPython extension.
    """
    try:
        from .line_profiler_ipython import LineProfilerMagics
        ip.register_magics(LineProfilerMagics)
    except Exception as e:
        print("line_profiler ipython extension import failed")


def load_stats(filename):
    """ Utility function to load a pickled LineStats object from a given
    filename.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def main():
    usage = "usage: python -m timemory.line_profiler <script>"

    parser = optparse.OptionParser(usage=usage,
                                   version=__version__)

    options, args = parser.parse_args()
    if len(args) != 1:
        parser.error("Must provide a filename.")
    lstats = load_stats(args[0])
    show_text(lstats.metrics, lstats.unit, lstats.display_unit)


if __name__ == '__main__':
    main()
