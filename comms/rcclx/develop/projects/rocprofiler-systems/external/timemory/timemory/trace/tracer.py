#!@PYTHON_EXECUTABLE@
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
from functools import wraps

from ..libs.libpytimemory.trace import tracer_function as _tracer_function
from ..libs.libpytimemory.trace import config as _tracer_config
from ..libs.libpytimemory.trace import tracer_init as _tracer_init
from ..libs.libpytimemory.trace import tracer_finalize as _tracer_fini
from ..libs.libpytimemory.trace import trace_bundle as _tracer_bundle
from ..libs.libpytimemory import settings


__all__ = ["trace", "config", "Tracer", "FakeTracer", "Config"]


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


config = _tracer_config
Config = _tracer_config


class Tracer:
    """Provides decorators and context-manager for the timemory tracers"""

    global _default_functor

    # static variable
    _conditional_functor = _default_functor

    # ---------------------------------------------------------------------------------- #
    #
    @staticmethod
    def condition(functor):
        """Assign a function evaluating whether to enable the tracer"""
        Tracer._conditional_functor = functor

    # ---------------------------------------------------------------------------------- #
    #
    @staticmethod
    def is_enabled():
        """Checks whether the tracer is enabled"""

        try:
            return Tracer._conditional_functor()
        except Exception:
            pass
        return False

    # ---------------------------------------------------------------------------------- #
    #
    def __init__(self, components=[], **kwargs):
        """
        Arguments:
            - components [list, str, or function]  : list of timemory components
        """

        self.components = components
        self._original_function = (
            sys.gettrace() if sys.gettrace() != _tracer_function else None
        )
        self._unset = 0
        self._use = (
            not _tracer_config._is_running and Tracer.is_enabled() is True
        )
        self.debug = kwargs["debug"] if "debug" in kwargs else False

    # ---------------------------------------------------------------------------------- #
    #
    def __del__(self):
        """Make sure the tracer stops"""

        self.stop()
        sys.settrace(self._original_function)

    # ---------------------------------------------------------------------------------- #
    #
    def configure(self):
        """Initialize, configure the bundle, store original tracer function"""

        _tracer_init()
        _trace = settings.trace_components
        _profl = settings.profiler_components
        _components = _profl if _trace is None else _trace
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
        if _trace is None or len(_trace) == 0:
            settings.trace_components = ",".join(components)

        # configure
        if settings.debug or self.debug:
            sys.stderr.write(
                "configuring Tracer with components (type: {}): {}\n".format(
                    type(components).__name__, components
                )
            )
        try:
            _tracer_bundle.configure(components, True, False)
        except Exception as e:
            sys.stderr.write(f"Tracer configuration failed: {e}\n")

        # store original
        if settings.debug or self.debug:
            sys.stderr.write("setting trace function...\n")
        if sys.gettrace() != _tracer_function:
            self._original_function = sys.gettrace()

        if settings.debug or self.debug:
            sys.stderr.write("Tracer configured...\n")

    # ---------------------------------------------------------------------------------- #
    #
    def update(self):
        """Updates whether the tracer is already running based on whether the tracer
        is not already running, is enabled, and the function is not already set
        """

        self._use = (
            not _tracer_config._is_running
            and Tracer.is_enabled() is True
            and sys.gettrace() == self._original_function
        )

    # ---------------------------------------------------------------------------------- #
    #
    def start(self):
        """Start the tracer"""

        self.update()
        if self._use:
            if settings.debug or self.debug:
                sys.stderr.write("Tracer starting...\n")
            self.configure()
            sys.settrace(_tracer_function)
            if settings.debug or self.debug:
                sys.stderr.write("Tracer started...\n")

        self._unset = self._unset + 1
        return self._unset

    # ---------------------------------------------------------------------------------- #
    #
    def stop(self):
        """Stop the tracer"""

        self._unset = self._unset - 1
        if self._unset == 0:
            if settings.debug or self.debug:
                sys.stderr.write("Tracer stopping...\n")
            sys.settrace(self._original_function)
            _tracer_fini()
            if settings.debug or self.debug:
                sys.stderr.write("Tracer stopped...\n")

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
            # unset the tracer if this wrapper set it
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
        """Execute and trace a command"""

        import __main__

        dict = __main__.__dict__
        if isinstance(cmd, str):
            return self.runctx(cmd, dict, dict)
        else:
            return self.runctx(" ".join(cmd), dict, dict)

    # ---------------------------------------------------------------------------------- #
    #
    def runctx(self, cmd, _globals, _locals):
        """Trace a context"""

        try:
            self.start()
            exec_(cmd, _globals, _locals)
        finally:
            self.stop()

        return self

    # ---------------------------------------------------------------------------------- #
    #
    def runcall(self, func, *args, **kw):
        """Trace a single function call"""

        try:
            self.start()
            return func(*args, **kw)
        finally:
            self.stop()


trace = Tracer


class FakeTracer:
    """Provides dummy decorators and context-manager for the timemory tracer"""

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
