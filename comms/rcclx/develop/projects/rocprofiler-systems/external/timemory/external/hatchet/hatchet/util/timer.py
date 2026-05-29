# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from functools import wraps


class Timer(object):
    """Simple phase timer with a context manager."""

    def __init__(self):
        self._phase = None
        self._start_time = None
        self._times = OrderedDict()

    def start_phase(self, phase):
        now = datetime.now()
        delta = None

        if self._phase:
            delta = now - self._start_time
            self._times[self._phase] = delta

        self._phase = phase
        self._start_time = now
        self._times[self._phase] = None
        return delta

    def end_phase(self):
        assert self._phase and self._start_time

        now = datetime.now()
        delta = now - self._start_time
        if self._times.get(self._phase):
            self._times[self._phase] = self._times.get(self._phase) + delta
        else:
            self._times[self._phase] = delta

        self._phase = None
        self._start_time = None

    def to_string(self, header="Times:\n"):
        _times = [[x, y] for x, y in self._times.items() if y is not None]
        _fmt = "    %-{}s %.2fs\n".format(
            min([73, max([20] + [len(x) + 1 for x, y in _times])])
        )
        out = StringIO()
        out.write(header)
        for phase, delta in _times:
            out.write(_fmt % ("{}:".format(phase), delta.total_seconds()))
        return out.getvalue()

    def __str__(self):
        return self.to_string()

    def __iadd__(self, rhs):
        for phase, delta in rhs._times.items():
            if self._times.get(phase):
                self._times[phase] = self._times.get(phase) + delta
            else:
                self._times[phase] = delta
        return self

    @contextmanager
    def phase(self, name):
        _timer = Timer()
        _timer.start_phase(name)
        yield
        _timer.end_phase()
        self += _timer

    def decorator(self, name):
        return TimerDecorator(self, name)


class TimerDecorator(object):
    """Wrapper around Timer to provide a decorator"""

    def __init__(self, timer, name):
        self._timer = timer
        self._name = name

    def __call__(self, func):
        """Decorator"""

        @wraps(func)
        def function_wrapper(*args, **kwargs):
            _timer = Timer()
            _timer.start_phase(self._name)
            result = func(*args, **kwargs)
            _timer.end_phase()
            self._timer += _timer
            return result

        return function_wrapper
