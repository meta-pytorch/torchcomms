#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2018, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import threading
from functools import wraps

from ..libs.libpytimemory.profiler import (
    profiler_function as _profiler_function,
)
from ..libs.libpytimemory.profiler import config as _profiler_config
from ..libs.libpytimemory.profiler import profiler_init as _profiler_init
from ..libs.libpytimemory.profiler import profiler_finalize as _profiler_fini
from ..libs.libpytimemory.profiler import profiler_bundle as _profiler_bundle
from ..libs.libpytimemory import settings


__all__ = ["profile", "config", "Profiler", "FakeProfiler", "Config"]


#
def _default_functor():
    return True


#
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


config = _profiler_config
Config = _profiler_config


#
class Profiler:
    """Provides decorators and context-manager for the timemory profilers"""

    global _default_functor

    # static variable
    _conditional_functor = _default_functor

    # ---------------------------------------------------------------------------------- #
    #
    @staticmethod
    def condition(functor):
        """Assign a function evaluating whether to enable the profiler"""
        Profiler._conditional_functor = functor

    # ---------------------------------------------------------------------------------- #
    #
    @staticmethod
    def is_enabled():
        """Checks whether the profiler is enabled"""

        try:
            return Profiler._conditional_functor()
        except Exception:
            pass
        return False

    # ---------------------------------------------------------------------------------- #
    #
    def __init__(self, components=[], flat=False, timeline=False, **kwargs):
        """
        Arguments:
            - components [list, str, or function]  : list of timemory components
            - flat [bool]                          : enable flat profiling
            - timeline [bool]                      : enable timeline profiling
        """

        self.flat = flat
        self.timeline = timeline
        self.components = components
        self._original_function = (
            sys.getprofile() if sys.getprofile() != _profiler_function else None
        )
        self._unset = 0
        self._use = (
            not _profiler_config._is_running and Profiler.is_enabled() is True
        )
        self.debug = kwargs["debug"] if "debug" in kwargs else False

    # ---------------------------------------------------------------------------------- #
    #
    def __del__(self):
        """Make sure the profiler stops"""

        self.stop()
        sys.setprofile(self._original_function)

    # ---------------------------------------------------------------------------------- #
    #
    def configure(self):
        """Initialize, configure the bundle, store original profiler function"""

        _profiler_init()
        _trace = settings.trace_components
        _profl = settings.profiler_components
        _components = _trace if _profl is None else _profl
        if len(_components) == 0:
            _components = settings.global_components

        # support function and list
        import inspect

        if inspect.isfunction(self.components):
            self.components = self.components()

        input_components = []
        if isinstance(self.components, list):
            for itr in self.components:
                if callable(itr):
                    input_components.extend(itr())
                else:
                    input_components.append(itr)
        else:
            input_components = ["{}".format(self.components)]

        # factor in global and local settings
        components = input_components + _components.split(",")
        components = list(set([v.lower() for v in components]))
        components = [f"{v}" for v in components]
        if "" in components:
            components.remove("")

        # update global setting
        if _profl is None or len(_profl) == 0:
            settings.profiler_components = ",".join(components)

        # configure
        if settings.debug or self.debug:
            sys.stderr.write(
                "configuring Profiler with components (type: {}): {}\n".format(
                    type(components).__name__, components
                )
            )
        try:
            _profiler_bundle.configure(
                components,
                settings.flat_profile or self.flat,
                settings.timeline_profile or self.timeline,
            )
        except Exception as e:
            sys.stderr.write(f"Profiler configuration failed: {e}\n")

        # store original
        if settings.debug or self.debug:
            sys.stderr.write("setting profile function...\n")
        if sys.getprofile() != _profiler_function:
            self._original_function = sys.getprofile()

        if settings.debug or self.debug:
            sys.stderr.write("Profiler configured...\n")

    # ---------------------------------------------------------------------------------- #
    #
    def update(self):
        """Updates whether the profiler is already running based on whether the tracer
        is not already running, is enabled, and the function is not already set
        """

        self._use = (
            not _profiler_config._is_running
            and Profiler.is_enabled() is True
            and sys.getprofile() == self._original_function
        )

    # ---------------------------------------------------------------------------------- #
    #
    def start(self):
        """Start the profiler explicitly"""

        self.update()
        if self._use:
            if settings.debug or self.debug:
                sys.stderr.write("Profiler starting...\n")
            self.configure()
            sys.setprofile(_profiler_function)
            threading.setprofile(_profiler_function)
            if settings.debug or self.debug:
                sys.stderr.write("Profiler started...\n")

        self._unset = self._unset + 1
        return self._unset

    # ---------------------------------------------------------------------------------- #
    #
    def stop(self):
        """Stop the profiler explicitly"""

        self._unset = self._unset - 1
        if self._unset == 0:
            if settings.debug or self.debug:
                sys.stderr.write("Profiler stopping...\n")
            sys.setprofile(self._original_function)
            _profiler_fini()
            if settings.debug or self.debug:
                sys.stderr.write("Profiler stopped...\n")

        return self._unset

    # ---------------------------------------------------------------------------------- #
    #
    def __call__(self, func):
        """Decorator"""

        @wraps(func)
        def function_wrapper(*args, **kwargs):
            # store whether this tracer started
            self.start()
            # execute the wrapped function
            result = func(*args, **kwargs)
            # unset the profiler if this wrapper set it
            self.stop()
            # return the result of the wrapped function
            return result

        return function_wrapper

    # ---------------------------------------------------------------------------------- #
    #
    def __enter__(self, *args, **kwargs):
        """Context manager start function"""

        self.start()

    # ---------------------------------------------------------------------------------- #
    #
    def __exit__(self, exec_type, exec_value, exec_tb):
        """Context manager stop function"""

        self.stop()

        if (
            exec_type is not None
            and exec_value is not None
            and exec_tb is not None
        ):
            import traceback

            traceback.print_exception(exec_type, exec_value, exec_tb, limit=5)

    # ---------------------------------------------------------------------------------- #
    #
    def run(self, cmd):
        """Execute and profile a command"""

        import __main__

        dict = __main__.__dict__
        if isinstance(cmd, str):
            return self.runctx(cmd, dict, dict)
        else:
            return self.runctx(" ".join(cmd), dict, dict)

    # ---------------------------------------------------------------------------------- #
    #
    def runctx(self, cmd, globals, locals):
        """Profile a context"""

        try:
            self.start()
            exec_(cmd, globals, locals)
        finally:
            self.stop()

        return self

    # ---------------------------------------------------------------------------------- #
    #
    def runcall(self, func, *args, **kw):
        """Profile a single function call"""

        try:
            self.start()
            return func(*args, **kw)
        finally:
            self.stop()


profile = Profiler


class FakeProfiler:
    """Provides dummy decorators and context-manager for the timemory profiler"""

    # ---------------------------------------------------------------------------------- #
    #
    @staticmethod
    def condition(functor):
        pass

    # ---------------------------------------------------------------------------------- #
    #
    @staticmethod
    def is_enabled():
        return False

    # ---------------------------------------------------------------------------------- #
    #
    def __init__(self, *args, **kwargs):
        """
        Arguments:
            - components [list of strings]  : list of timemory components
            - flat [bool]                   : enable flat profiling
            - timeline [bool]               : enable timeline profiling
        """
        pass

    # ---------------------------------------------------------------------------------- #
    #
    def __call__(self, func):
        """Decorator"""

        @wraps(func)
        def function_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return function_wrapper

    # ---------------------------------------------------------------------------------- #
    #
    def __enter__(self, *args, **kwargs):
        """Context manager begin"""
        pass

    # ---------------------------------------------------------------------------------- #
    #
    def __exit__(self, exec_type, exec_value, exec_tb):
        """Context manager end"""

        import traceback

        if (
            exec_type is not None
            and exec_value is not None
            and exec_tb is not None
        ):
            traceback.print_exception(exec_type, exec_value, exec_tb, limit=5)
