// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#if !defined(TIMEMORY_PYTRACE_SOURCE)
#    define TIMEMORY_PYTRACE_SOURCE
#endif

#include "libpytimemory-component-bundle.hpp"
#include "timemory/library.h"

#include <cctype>
#include <cstdint>
#include <locale>
#include <unordered_set>

#if !defined(TIMEMORY_PYTHON_VERSION)
#    define TIMEMORY_PYTHON_VERSION                                                      \
        ((10000 * PY_MAJOR_VERSION) + (100 * PY_MINOR_VERSION) + PY_MICRO_VERSION)
#endif

using namespace tim::component;

namespace pytrace
{
//
template <typename KeyT, typename MappedT>
using uomap_t = std::unordered_map<KeyT, MappedT>;
//
using string_t            = std::string;
using frame_object_t      = PyFrameObject;
using tracer_t            = tim::lightweight_tuple<user_trace_bundle>;
using tracer_line_map_t   = uomap_t<size_t, tracer_t>;
using tracer_code_map_t   = uomap_t<string_t, tracer_line_map_t>;
using tracer_iterator_t   = typename tracer_line_map_t::iterator;
using function_vec_t      = std::vector<tracer_iterator_t>;
using function_code_map_t = std::unordered_map<frame_object_t*, function_vec_t>;
using strset_t            = std::unordered_set<string_t>;
using strvec_t            = std::vector<string_t>;
using decor_line_map_t    = uomap_t<string_t, std::set<size_t>>;
using file_line_map_t     = uomap_t<string_t, strvec_t>;
//
struct config
{
    bool     is_running        = false;
    bool     include_internal  = false;
    bool     include_args      = false;
    bool     include_line      = true;
    bool     include_filename  = true;
    bool     full_filepath     = false;
    int32_t  max_stack_depth   = std::numeric_limits<uint16_t>::max();
    int32_t  base_stack_depth  = -1;
    string_t base_module_path  = "";
    strset_t include_functions = {};
    strset_t include_filenames = {};
    strset_t exclude_functions = { "^(FILE|FUNC|LINE)$",
                                   "^get_fcode$",
                                   "^_(_exit__|handle_fromlist|shutdown|get_sep)$",
                                   "^is(function|class)$",
                                   "^basename$",
                                   "^<.*>$" };
    strset_t exclude_filenames = {
        "(__init__|__main__|functools|encoder|decoder|_pylab_helpers|threading).py$",
        "^<.*>$"
    };
    tracer_code_map_t   records      = {};
    function_code_map_t functions    = {};
    tim::scope::config  tracer_scope = tim::scope::config{ true, false, false };
    int32_t verbose = tim::settings::verbose() + ((tim::settings::debug()) ? 16 : 0);
};
//
inline config&
get_config()
{
    static auto*              _instance    = new config{};
    static thread_local auto* _tl_instance = [&]() {
        static std::atomic<uint32_t> _count{ 0 };
        auto                         _cnt = _count++;
        if(_cnt == 0)
            return _instance;

        auto* _tmp              = new config{};
        _tmp->is_running        = _instance->is_running;
        _tmp->include_internal  = _instance->include_internal;
        _tmp->include_args      = _instance->include_args;
        _tmp->include_line      = _instance->include_line;
        _tmp->include_filename  = _instance->include_filename;
        _tmp->full_filepath     = _instance->full_filepath;
        _tmp->max_stack_depth   = _instance->max_stack_depth;
        _tmp->base_module_path  = _instance->base_module_path;
        _tmp->include_functions = _instance->include_functions;
        _tmp->include_filenames = _instance->include_filenames;
        _tmp->exclude_functions = _instance->exclude_functions;
        _tmp->exclude_filenames = _instance->exclude_filenames;
        _tmp->verbose           = _instance->verbose;
        return _tmp;
    }();
    return *_tl_instance;
}
//
int
get_frame_lineno(frame_object_t* frame)
{
#if TIMEMORY_PYTHON_VERSION >= 31100
    return PyFrame_GetLineNumber(frame);
#else
    return frame->f_lineno;
#endif
}
//
int
get_frame_lasti(frame_object_t* frame)
{
#if TIMEMORY_PYTHON_VERSION >= 31100
    return PyFrame_GetLasti(frame);
#else
    return frame->f_lasti;
#endif
}
//
auto
get_frame_code(frame_object_t* frame)
{
#if TIMEMORY_PYTHON_VERSION >= 31100
    return PyFrame_GetCode(frame);
#else
    return frame->f_code;
#endif
}
//
auto
get_frame_back(frame_object_t* frame)
{
#if TIMEMORY_PYTHON_VERSION >= 31100
    return PyFrame_GetBack(frame);
#else
    return frame->f_back;
#endif
}
//
int32_t
get_depth(frame_object_t* frame)
{
    auto* frame_back = get_frame_back(frame);
    return (frame_back) ? (get_depth(frame_back) + 1) : 0;
}
//
py::function
tracer_ignore_function(py::object, const char*, py::object)
{
    return py::cpp_function{ &tracer_ignore_function };
}
//
py::function
tracer_function(py::object pframe, const char* swhat, py::object arg)
{
    //
    static thread_local tracer_iterator_t _last     = {};
    static thread_local bool              _has_last = false;
    //
    if(_has_last)
    {
        _last->second.stop();
        _has_last = false;
    }

    struct trace_tl_data
    {
        using pushed_funcs_t = std::unordered_map<string_t, std::unordered_set<string_t>>;

        config           configuration = get_config();
        bool             disable       = false;
        file_line_map_t  file_lines    = {};
        decor_line_map_t decor_lines   = {};
        strset_t         file_lskip    = {};
        pushed_funcs_t   pushed_funcs  = {};
    };

    static thread_local auto _tl_data      = trace_tl_data{};
    auto&                    _config       = _tl_data.configuration;
    auto&                    _disable      = _tl_data.disable;
    auto&                    _file_lines   = _tl_data.file_lines;
    auto&                    _decor_lines  = _tl_data.decor_lines;
    auto&                    _file_lskip   = _tl_data.file_lskip;
    auto&                    _pushed_funcs = _tl_data.pushed_funcs;
    const auto&              _mpath        = _config.base_module_path;
    const auto&              _only_funcs   = _config.include_functions;
    const auto&              _only_files   = _config.include_filenames;
    const auto&              _skip_funcs   = _config.exclude_functions;
    const auto&              _skip_files   = _config.exclude_filenames;

    if(!tim::settings::enabled() || pframe.is_none() || pframe.ptr() == nullptr ||
       _disable)
        return py::cpp_function{ &tracer_ignore_function };

    if(user_trace_bundle::bundle_size() == 0)
    {
        if(_config.verbose > 1)
            TIMEMORY_PRINT_HERE("%s", "Tracer bundle is empty");
        return py::cpp_function{ &tracer_ignore_function };
    }

    int what = (strcmp(swhat, "line") == 0)
                   ? PyTrace_LINE
                   : (strcmp(swhat, "call") == 0)
                         ? PyTrace_CALL
                         : (strcmp(swhat, "return") == 0) ? PyTrace_RETURN : -1;

    // only support PyTrace_{LINE,CALL,RETURN}
    if(what < 0)
    {
        if(_config.verbose > 2)
            TIMEMORY_PRINT_HERE("%s :: %s", "Ignoring what != {LINE,CALL,RETURN}", swhat);
        return py::cpp_function{ &tracer_function };
    }

    //
    auto* frame = reinterpret_cast<frame_object_t*>(pframe.ptr());
    auto  _code = get_frame_code(frame);
    auto  _line = get_frame_lineno(frame);
    auto& _file = _code->co_filename;
    auto& _name = _code->co_name;

    // get the depth of the frame
    auto _fdepth = get_depth(frame);

    if(_config.base_stack_depth < 0)
        _config.base_stack_depth = _fdepth;

    bool    _iscall = (what == PyTrace_CALL);
    int32_t _sdepth = _fdepth - _config.base_stack_depth - 3;
    // if frame exceeds max stack-depth
    if(_iscall && _sdepth > _config.max_stack_depth)
    {
        if(_config.verbose > 1)
            TIMEMORY_PRINT_HERE("skipping %i > %i", (int) _sdepth,
                                (int) _config.max_stack_depth);
        return py::cpp_function{ &tracer_ignore_function };
    }

    // get the function name
    auto _get_funcname = [&]() -> string_t { return py::cast<string_t>(_name); };

    // get the filename
    auto _get_filename = [&]() -> string_t { return py::cast<string_t>(_file); };

    // get the basename of the filename
    static auto _get_basename = [](const string_t& _fullpath) {
        if(_fullpath.find('/') != string_t::npos)
            return _fullpath.substr(_fullpath.find_last_of('/') + 1);
        return _fullpath;
    };

    static auto _sanitize_source_line = [](string_t& itr) {
        for(auto c : { '\n', '\r', '\t' })
        {
            size_t pos = 0;
            while((pos = itr.find(c)) != string_t::npos)
                itr = itr.replace(pos, 1, "");
        }
        return itr;
    };

    static size_t _maxw      = 0;
    auto          _get_lines = [&](const auto& _fullpath) -> strvec_t& {
        auto litr = _file_lines.find(_fullpath);
        if(litr != _file_lines.end())
            return litr->second;
        _disable              = true;
        auto        linecache = py::module::import("linecache");
        static bool _once     = false;
        if(!_once)
            _once = (linecache.attr("clearcache")(), true);
        auto _lines = linecache.attr("getlines")(_fullpath).template cast<strvec_t>();
        _disable = false;
        for(size_t i = 0; i < _lines.size(); ++i)
        {
            auto& itr = _lines.at(i);
            _sanitize_source_line(itr);
            _maxw      = std::max<size_t>(_maxw, itr.length() + 1);
            auto _apos = itr.find_first_not_of(" \t");
            if(_apos != std::string::npos && i + 1 < _lines.size())
            {
                if(itr.at(_apos) == '@')
                {
                    auto& _ditr = _lines.at(i + 1);
                    if(_ditr.find("def ") == _apos)
                        _decor_lines[_fullpath].insert(i);
                }
            }
        }
        litr = _file_lines.insert({ _fullpath, _lines }).first;
        return litr->second;
    };

    // get the arguments
    auto _get_args = [&]() {
        _disable = true;
        tim::scope::destructor _dtor{ [&_disable]() { _disable= false; } };
        auto                   inspect = py::module::import("inspect");
        try
        {
            return py::cast<string_t>(
                inspect.attr("formatargvalues")(*inspect.attr("getargvalues")(pframe)));
        } catch(py::error_already_set& _exc)
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(_config.verbose > 1, "Error! %s",
                                            _exc.what());
            if(!_exc.matches(PyExc_AttributeError))
                throw;
        }
        return std::string{};
    };

    // get the final label
    auto _get_label = [&](auto _func, const auto& _filename, const auto& _fullpath,
                          const auto& _flines, auto _fline) {
        // append the arguments
        if(_config.include_args)
            _func.append(_get_args());
        // append the filename
        if(_config.include_filename)
        {
            _func.append("][");
            _func.append((_config.full_filepath) ? _fullpath : _filename);
            // append the line number
            if(_config.include_line)
            {
                _func.append(":");
                auto              _w = log10(_flines.size()) + 1;
                std::stringstream _sline{};
                _sline.fill('0');
                _sline << std::setw(_w) << _fline;
                _func.append(_sline.str());
            }
        }
        else if(_config.include_line)
        {
            _func.append(TIMEMORY_JOIN("", ':', _fline));
        }
        return _func;
    };

    static auto _find_matching = [](const strset_t& _expr, const std::string& _name) {
        const auto _rconstants =
            std::regex_constants::egrep | std::regex_constants::optimize;
        for(const auto& itr : _expr)
        {
            if(std::regex_search(_name, std::regex(itr, _rconstants)))
                return true;
        }
        return false;
    };

    auto _func = _get_funcname();

    if(!_only_funcs.empty() && !_find_matching(_only_funcs, _func))
    {
        if(_config.verbose > 1)
            TIMEMORY_PRINT_HERE("Skipping non-included function: %s", _func.c_str());
        return py::cpp_function{ (_iscall) ? &tracer_ignore_function : &tracer_function };
    }

    if(_find_matching(_skip_funcs, _func))
    {
        if(_config.verbose > 1)
            TIMEMORY_PRINT_HERE("Skipping designated function: '%s'", _func.c_str());
        auto _manager = tim::manager::instance();
        if(!_manager || _manager->is_finalized() || _func == "_shutdown")
        {
            if(_config.verbose > 1)
                TIMEMORY_PRINT_HERE("Shutdown detected: %s", _func.c_str());
            _disable       = true;
            auto sys       = py::module::import("sys");
            auto threading = py::module::import("threading");
            sys.attr("settrace")(py::none{});
            threading.attr("settrace")(py::none{});
            _disable = false;
        }
        return py::cpp_function{ (_iscall) ? &tracer_ignore_function : &tracer_function };
    }

    auto _full = _get_filename();
    auto _base = _get_basename(_full);

    if(!_config.include_internal &&
       strncmp(_full.c_str(), _mpath.c_str(), _mpath.length()) == 0)
    {
        if(_config.verbose > 2)
            TIMEMORY_PRINT_HERE("Skipping internal function: %s", _func.c_str());
        return py::cpp_function{ &tracer_ignore_function };
    }

    if(!_only_files.empty() && !_find_matching(_only_files, _full))
    {
        if(_config.verbose > 2)
            TIMEMORY_PRINT_HERE("Skipping non-included file: %s", _full.c_str());
        return py::cpp_function{ &tracer_function };
    }

    if(_find_matching(_skip_files, _full))
    {
        if(_config.verbose > 2)
            TIMEMORY_PRINT_HERE("Skipping non-included file: %s", _full.c_str());
        return py::cpp_function{ &tracer_function };
    }

    strvec_t* _plines = nullptr;
    try
    {
        if(_file_lskip.count(_full) == 0)
            _plines = &_get_lines(_full);
    } catch(std::exception& e)
    {
        if(_config.verbose > -1)
            TIMEMORY_PRINT_HERE(
                "Exception thrown when retrieving lines for file '%s'. Functions "
                "in this file will not be traced:\n%s",
                _full.c_str(), e.what());
        _file_lskip.insert(_full);
        return py::cpp_function{ (_iscall) ? &tracer_ignore_function : &tracer_function };
    }

    if(!_plines)
    {
        if(_config.verbose > 3)
            TIMEMORY_PRINT_HERE("No source code lines for '%s'. Returning",
                                _full.c_str());
        return py::cpp_function{ &tracer_function };
    }

    auto& _flines = *_plines;

    //----------------------------------------------------------------------------------//
    //
    auto _get_trace_lines = [&]() -> tracer_line_map_t& {
        auto titr = _config.records.find(_full);
        if(titr != _config.records.end())
            return titr->second;
        auto _tvec = tracer_line_map_t{};
        _tvec.reserve(_flines.size());
        for(size_t i = 0; i < _flines.size(); ++i)
        {
            auto&             itr     = _flines.at(i);
            auto              _flabel = _get_label(_func, _base, _full, _flines, i + 1);
            auto              _prefix = TIMEMORY_JOIN("", '[', _flabel, ']');
            std::stringstream _slabel;
            _slabel << std::setw(_maxw) << std::left << itr;
            int64_t _rem = tim::settings::max_width();
            // three spaces for '>>>'
            _rem -= _maxw + _prefix.length() + 3;
            // space for 'XY|' for thread
            _rem -= (tim::settings::collapse_threads()) ? 3 : 0;
            // space for 'XY|' for process
            _rem -= (tim::settings::collapse_processes()) ? 3 : 0;
            // make sure not less than zero
            _rem = std::max<int64_t>(_rem, 0);
            _slabel << std::setw(_rem) << std::right << "" << ' ' << _prefix;
            auto _label = _slabel.str();
            TIMEMORY_DEBUG_PRINT_HERE("%8s | %s%s | %s", swhat, _func.c_str(),
                                      _get_args().c_str(), _label.c_str());
            // create a tracer for the line
            _tvec.emplace(i, tracer_t{ _label, _config.tracer_scope });
        }
        titr = _config.records.emplace(_full, _tvec).first;
        return titr->second;
    };

    // the tracers for the fullpath
    auto& _tlines = _get_trace_lines();

    //----------------------------------------------------------------------------------//
    // the first time a frame is encountered, use the inspect module to process
    // the source lines. Essentially, this function finds all the source code
    // lines in the frame, generates an label via the source code, and then
    // "pushes" that label into the storage. Then we are free to call start/stop
    // repeatedly and only when the pop is applied does the storage instance get
    // updated. NOTE: this means the statistics are not correct.
    //
    auto _push_tracer = [&](auto object) {
        _disable     = true;
        auto inspect = py::module::import("inspect");
        try
        {
            py::object srclines = py::none{};
            try
            {
                srclines = inspect.attr("getsourcelines")(object);
            } catch(std::exception& e)
            {
                if(_config.verbose > 2)
                    std::cerr << e.what() << std::endl;
            }
            if(!srclines.is_none())
            {
                auto _get_docstring = [](const string_t& _str, size_t _pos) {
                    auto _q1 = _str.find("'''", _pos);
                    auto _q2 = _str.find("\"\"\"", _pos);
                    return std::min<size_t>(_q1, _q2);
                };

                auto pysrclist = srclines.cast<py::list>()[0].cast<py::list>();
                auto _srclines = strvec_t{};
                // auto _skip_docstring = false;
                for(auto itr : pysrclist)
                {
                    auto sline = itr.cast<std::string>();
                    _sanitize_source_line(sline);
                    _srclines.emplace_back(sline);
                }
                //
                if(_config.verbose > 3)
                {
                    std::cout << "\nSource lines:\n";
                    for(const auto& itr : _srclines)
                        std::cout << "    " << itr << '\n';
                    std::cout << std::endl;
                }
                //
                bool _in_docstring = false;
                auto ibeg          = (_line == 0) ? _line : _line - 1;
                auto iend = std::min<size_t>(_tlines.size(), ibeg + pysrclist.size());
                for(size_t i = ibeg; i < iend; ++i)
                {
                    auto& _tracer = _tlines.at(i);
                    for(auto& sitr : _srclines)
                    {
                        if(sitr.empty())
                            continue;  // skip empty lines
                        auto _docspos = _get_docstring(sitr, 0);
                        if(_docspos != std::string::npos)
                        {
                            _docspos = _get_docstring(sitr, _docspos + 4);
                            if(_docspos != std::string::npos)
                            {
                                continue;  // one-line docstring
                            }
                            else
                            {
                                // multiline docstring
                                auto _old_in_docstring = _in_docstring;
                                _in_docstring          = !_in_docstring;
                                if(_old_in_docstring)
                                    continue;  // end of docstring
                            }
                        }
                        if(_in_docstring)
                        {
                            continue;
                        }
                        if(sitr.find_first_of('#') < sitr.find_first_not_of(" \t#"))
                            continue;  // skip comments
                        if(_tracer.key().find(sitr) != std::string::npos)
                        {
                            _tracer.push();
                            break;
                        }
                    }
                }
            }
        } catch(py::cast_error& e)
        {
            std::cerr << e.what() << std::endl;
        }
        _disable = false;
    };

    //----------------------------------------------------------------------------------//
    //
    if(_pushed_funcs[_full].find(_func) == _pushed_funcs[_full].end())
    {
        _pushed_funcs[_full].emplace(_func);
        _push_tracer(pframe);
    }

    //----------------------------------------------------------------------------------//
    // start function
    //
    auto _tracer_call = [&]() {
        // auto itr = _tlines.find(_line - 1);
        // if(itr == _tlines.end())
        //     return;
        // auto& _entry = _config.functions[frame];
        // if(_entry.empty())
        // {
        //     // if(_decor_lines[_full].count(_line - 1) > 0)
        //     //     ++itr;
        //     itr->second.push();
        //     itr->second.start();
        // }
        // _entry.emplace_back(itr);
    };

    //----------------------------------------------------------------------------------//
    // stop function
    //
    auto _tracer_return = [&]() {
        // auto fitr = _config.functions.find(frame);
        // if(fitr == _config.functions.end() || fitr->second.empty())
        //     return;
        // auto itr = fitr->second.back();
        // fitr->second.pop_back();
        // if(fitr->second.empty())
        //     itr->second.stop();
    };

    //----------------------------------------------------------------------------------//
    // line function
    //
    auto _tracer_line = [&]() {
        auto itr = _tlines.find(_line - 1);
        if(itr == _tlines.end())
            return;
        itr->second.start();
        _has_last = true;
        _last     = itr;
    };

    //----------------------------------------------------------------------------------//
    // process what
    //
    switch(what)
    {
        case PyTrace_LINE: _tracer_line(); break;
        case PyTrace_CALL: _tracer_call(); break;
        case PyTrace_RETURN: _tracer_return(); break;
        default: break;
    }

    // don't do anything with arg
    tim::consume_parameters(arg);

    if(_config.verbose > 3)
        TIMEMORY_PRINT_HERE("Returning trace function for %s of '%s'", swhat,
                            _func.c_str());

    return py::cpp_function{ &tracer_function };
}
//
py::module
generate(py::module& _pymod)
{
    py::module _trace =
        _pymod.def_submodule("trace", "Python tracing functions and "
                                      "C/C++/Fortran-compatible library "
                                      "functions (subject to throttling)");

    // py::class_<PyCodeObject>  _code_object(_pymod, "code_object", "PyCodeObject");
    // py::class_<PyFrameObject> _frame_object(_pymod, "frame_object", "PyFrameObject");

    // tim::consume_parameters(_code_object, _frame_object);

    static auto _set_scope = [](bool _flat, bool _timeline) {
        get_config().tracer_scope = tim::scope::config{ _flat, _timeline };
    };

    pycomponent_bundle::generate<user_trace_bundle>(
        _trace, "trace_bundle", "User-bundle for Python tracing interface", _set_scope);

    _trace.def("init", &timemory_trace_init, "Initialize Tracing",
               py::arg("args") = "wall_clock", py::arg("read_command_line") = false,
               py::arg("cmd") = "");
    _trace.def("finalize", &timemory_trace_finalize, "Finalize Tracing");
    _trace.def("is_throttled", &timemory_is_throttled, "Check if key is throttled",
               py::arg("key"));
    _trace.def("push", &timemory_push_trace, "Push Trace", py::arg("key"));
    _trace.def("push", &timemory_push_trace_hash, "Push Trace", py::arg("key"));
    _trace.def("pop", &timemory_pop_trace, "Pop Trace", py::arg("key"));
    _trace.def("pop", &timemory_pop_trace_hash, "Pop Trace", py::arg("key"));

    auto _init = []() {
        auto _verbose = get_config().verbose;
        TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose > 1, "%s", "Initializing trace");
        try
        {
            auto _file = py::module::import("timemory").attr("__file__").cast<string_t>();
            if(_file.find('/') != string_t::npos)
                _file = _file.substr(0, _file.find_last_of('/'));
            get_config().base_module_path = _file;
        } catch(py::cast_error& e)
        {
            std::cerr << "[trace_init]> " << e.what() << std::endl;
        }

        if(get_config().is_running)
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose > 1, "%s", "Trace already running");
            return;
        }
        TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose < 2 && _verbose > 0, "%s",
                                        "Initializing trace");
        TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose > 0, "%s",
                                        "Resetting trace state for initialization");
        get_config().records.clear();
        get_config().functions.clear();
        get_config().is_running = true;
    };

    auto _fini = []() {
        auto _verbose = get_config().verbose;
        TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose > 2, "%s", "Finalizing trace");
        if(!get_config().is_running)
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose > 2, "%s",
                                            "Trace already finalized");
            return;
        }
        TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose > 0 && _verbose < 3, "%s",
                                        "Finalizing trace");
        get_config().is_running = false;
        TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose > 1, "%s",
                                        "Popping records from call-stack");
        for(auto& ritr : get_config().records)
        {
            for(auto& itr : ritr.second)
                itr.second.pop();
        }
        TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose > 1, "%s", "Destroying records");
        get_config().records.clear();
        get_config().functions.clear();
    };

    _trace.def("tracer_function", &tracer_function, "Tracing function");
    _trace.def("tracer_init", _init, "Initialize the tracer");
    _trace.def("tracer_finalize", _fini, "Finalize the tracer");

    py::class_<config> _pyconfig(_trace, "config", "Tracer configuration");

#define CONFIGURATION_PROPERTY(NAME, TYPE, DOC, ...)                                     \
    _pyconfig.def_property_static(                                                       \
        NAME, [](py::object) { return __VA_ARGS__; },                                    \
        [](py::object, TYPE val) { __VA_ARGS__ = val; }, DOC);

    CONFIGURATION_PROPERTY("_is_running", bool, "Tracer is currently running",
                           get_config().is_running)
    CONFIGURATION_PROPERTY("include_internal", bool, "Include functions within timemory",
                           get_config().include_internal)
    CONFIGURATION_PROPERTY("include_args", bool, "Encode the function arguments",
                           get_config().include_args)
    CONFIGURATION_PROPERTY("include_line", bool, "Encode the function line number",
                           get_config().include_line)
    CONFIGURATION_PROPERTY("include_filename", bool,
                           "Encode the function filename (see also: full_filepath)",
                           get_config().include_filename)
    CONFIGURATION_PROPERTY("full_filepath", bool,
                           "Display the full filepath (instead of file basename)",
                           get_config().full_filepath)
    CONFIGURATION_PROPERTY("verbosity", int32_t, "Verbosity of the logging",
                           get_config().verbose)

    static auto _get_strset = [](const strset_t& _targ) {
        auto _out = py::list{};
        for(auto itr : _targ)
            _out.append(itr);
        return _out;
    };

    static auto _set_strset = [](py::list _inp, strset_t& _targ) {
        for(auto itr : _inp)
            _targ.insert(itr.cast<std::string>());
    };

#define CONFIGURATION_PROPERTY_LAMBDA(NAME, DOC, GET, SET)                               \
    _pyconfig.def_property_static(NAME, GET, SET, DOC);
#define CONFIGURATION_STRSET(NAME, DOC, ...)                                             \
    {                                                                                    \
        auto GET = [](py::object) { return _get_strset(__VA_ARGS__); };                  \
        auto SET = [](py::object, py::list val) { _set_strset(val, __VA_ARGS__); };      \
        CONFIGURATION_PROPERTY_LAMBDA(NAME, DOC, GET, SET)                               \
    }

    CONFIGURATION_STRSET("only_functions", "Function regexes to collect exclusively",
                         get_config().include_functions)
    CONFIGURATION_STRSET("only_filenames", "Filename regexes to collect exclusively",
                         get_config().include_filenames)
    CONFIGURATION_STRSET("skip_functions", "Function regexes to filter out of collection",
                         get_config().exclude_functions)
    CONFIGURATION_STRSET("skip_filenames", "Filename regexes to filter out of collection",
                         get_config().exclude_filenames)

    tim::operation::init<user_trace_bundle>(
        tim::operation::mode_constant<tim::operation::init_mode::global>{});

    return _trace;
}
}  // namespace pytrace
//
//======================================================================================//
