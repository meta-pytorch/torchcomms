from .python25 cimport PyFrameObject, PyObject, PyStringObject
from libc.stdlib cimport free

def _default_component():
    return None

_component_generator = _default_component
_component_display_units = "sec"
_component_num_prealloc = 0

def set_component_prealloc(_num_prealloc):
    global _component_num_prealloc
    _component_num_prealloc = _num_prealloc

def set_component_generator(_gen):
    global _component_generator
    global _component_display_units
    _component_generator = _gen
    try:
        _component_display_units = _gen().display_unit()
    except:
        pass

def get_component():
    global _component_generator
    return _component_generator()

cdef extern from "frameobject.h":
    ctypedef int (*Py_tracefunc)(object self, PyFrameObject *py_frame, int what, PyObject *arg)

cdef extern from "Python.h":
    ctypedef long long PY_LONG_LONG
    cdef bint PyCFunction_Check(object obj)

    cdef void PyEval_SetProfile(Py_tracefunc func, object arg)
    cdef void PyEval_SetTrace(Py_tracefunc func, object arg)

    ctypedef object (*PyCFunction)(object self, object args)

    ctypedef struct PyMethodDef:
        char *ml_name
        PyCFunction ml_meth
        int ml_flags
        char *ml_doc

    ctypedef struct PyCFunctionObject:
        PyMethodDef *m_ml
        PyObject *m_self
        PyObject *m_module

    # They're actually #defines, but whatever.
    cdef int PyTrace_CALL
    cdef int PyTrace_EXCEPTION
    cdef int PyTrace_LINE
    cdef int PyTrace_RETURN
    cdef int PyTrace_C_CALL
    cdef int PyTrace_C_EXCEPTION
    cdef int PyTrace_C_RETURN

cdef extern from "unset_trace.h":
    void unset_trace()

def label(code):
    """ Return a (filename, first_lineno, func_name) tuple for a given code
    object.

    This is the same labelling as used by the cProfile module in Python 2.5.
    """
    if isinstance(code, str):
        return ('~', 0, code)    # built-in functions ('~' sorts at the end)
    else:
        return (code.co_filename, code.co_firstlineno, code.co_name)


cdef class LineMetrics:
    """ The timing for a single line.
    """
    cdef public object code
    cdef public int lineno
    cdef public object comp

    def __init__(self, object code, int lineno, object comp = get_component()):
        self.code = code
        self.lineno = lineno
        self.comp = comp

    def valid(self):
        return (self.comp.laps() > 0)

    def start(self):
        self.comp.start()

    def stop(self):
        self.comp.stop()

    def astuple(self):
        """ Convert to a tuple of (lineno, hits, total).
        """
        return (self.lineno, self.comp.laps(), self.comp.get())

    def __repr__(self):
        return '<LineMetrics for %r\n  lineno: %r\n  hits: %r\n  total: %r>' % (self.code, self.lineno, self.comp.laps(), self.comp.get())


# Note: this is a regular Python class to allow easy pickling.
class LineStats(object):
    """ Object to encapsulate line-profile statistics.

    Attributes
    ----------
    metrics : dict
        Mapping from (filename, first_lineno, function_name) of the profiled
        function to a list of (lineno, nhits, total) tuples for each
        profiled line. total is an integer in the native units of the
        timer.
    unit : float
        The number of seconds per timer unit.
    """
    def __init__(self, metrics, unit, display_unit):
        self.metrics = metrics
        self.unit = unit
        self.display_unit = display_unit


cdef class LineProfiler:
    """ Time the execution of lines of Python code.
    """
    cdef public long last_index
    cdef public object last_tool
    cdef public object last_code
    #
    cdef public list functions
    cdef public dict code_map
    cdef public long enable_count
    cdef public double unit

    def __init__(self, *functions):
        global _component_display_units

        self.functions = []
        self.code_map = {}
        self.last_index = 0
        self.last_tool = None
        self.last_code = None
        self.unit = get_component().unit()
        self.display_unit = _component_display_units
        self.enable_count = 0
        for func in functions:
            self.add_function(func)

    def add_function(self, func):
        global _component_num_prealloc

        """ Record line profiling information for the given Python function.
        """
        try:
            code = func.__code__
        except AttributeError:
            import warnings
            warnings.warn("Could not extract a code object for the object %r" % (func,))
            return
        if code not in self.code_map:
            self.code_map[code] = {}
            self.functions.append(func)
        for i in range(_component_num_prealloc):
            self.code_map[code][i] = LineMetrics(code, i, get_component())
        if _component_num_prealloc > 0:
            print("Number of pre-allocated components for {}: {}".format(
                func.__name__, _component_num_prealloc))

    def enable_by_count(self):
        """ Enable the profiler if it hasn't been enabled before.
        """
        if self.enable_count == 0:
            self.enable()
        self.enable_count += 1

    def disable_by_count(self):
        """ Disable the profiler if the number of disable requests matches the
        number of enable requests.
        """
        if self.enable_count > 0:
            self.enable_count -= 1
            if self.enable_count == 0:
                self.disable()

    def __enter__(self):
        self.enable_by_count()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable_by_count()

    def enable(self):
        PyEval_SetTrace(python_trace_callback, self)

    def disable(self):
        unset_trace()

    def cleanup(self):
        # cleanup any lines pre-allocated but without entries
        for code in self.code_map:
            rmkeys = []
            for f, s in self.code_map[code].items():
                if not s.valid():
                    rmkeys.append(f)
            for i in rmkeys:
                del self.code_map[code][i]

    def get_stats(self):
        """ Return a LineStats object containing the metrics.
        """
        self.cleanup()
        stats = {}
        for code in self.code_map:
            entries = self.code_map[code].values()
            key = label(code)
            stats[key] = [e.astuple() for e in entries]
            stats[key].sort()
        return LineStats(stats, self.unit, self.display_unit)


cdef int python_trace_callback(object self_, PyFrameObject *py_frame, int what,
    PyObject *arg):
    """ The PyEval_SetTrace() callback.
    """
    cdef LineProfiler self
    cdef long nline
    cdef object code
    cdef LineMetrics comp

    self = <LineProfiler>self_
    code_map = self.code_map
    if self.last_tool is not None:
        self.last_tool.stop()

    if what == PyTrace_LINE:
        code = <object>py_frame.f_code
        if code in code_map:
            # the current line
            nline = py_frame.f_lineno
            if nline not in code_map[code]:
                comp = LineMetrics(code, nline, get_component())
                code_map[code][nline] = comp
            else:
                comp = code_map[code][nline]
            self.last_tool = comp
            self.last_code = code
            self.last_index = nline
            comp.start()

    return 0


