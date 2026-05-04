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

#if !defined(TIMEMORY_PYAPI_SOURCE)
#    define TIMEMORY_PYAPI_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/backends/cuda.hpp"
#include "timemory/backends/papi.hpp"

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"
#else
#    include "timemory/backends/types/cupti.hpp"
#endif

#include <pybind11/cast.h>

//======================================================================================//
//
namespace pyapi
{
//
py::module
generate_papi(py::module& _pymod);
//
py::module
generate_cupti(py::module& _pymod);
//
py::module
generate_cuda(py::module& _pymod);
//
py::module
generate(py::module& _api)
{
    generate_papi(_api);
    generate_cuda(_api);
    generate_cupti(_api);
    return _api;
}
//
py::module
generate_papi(py::module& _pymod)
{
    py::module _papi = _pymod.def_submodule("papi", "PAPI Python interface");

#define PYTIMEMORY_PAPI_BINDING(NAME, HELP) _papi.def(#NAME, &tim::papi::NAME, HELP)
#define PYTIMEMORY_PAPI_BINDING_1(NAME, HELP)                                            \
    _papi.def(                                                                           \
        #NAME, [](int _v) { return tim::papi::NAME(_v); }, HELP);                        \
    _papi.def(                                                                           \
        #NAME, [](const std::string& _v) { return tim::papi::NAME(_v); }, HELP)
#define PYTIMEMORY_PAPI_BINDING_2(NAME, HELP)                                            \
    _papi.def(                                                                           \
        #NAME, [](int _c, int _v) { return tim::papi::NAME(_c, _v); }, HELP);            \
    _papi.def(                                                                           \
        #NAME, [](int _c, const std::string& _v) { return tim::papi::NAME(_c, _v); },    \
        HELP)

    PYTIMEMORY_PAPI_BINDING(working, "Return status of PAPI");
    PYTIMEMORY_PAPI_BINDING(set_debug, "Set the debug level");
    PYTIMEMORY_PAPI_BINDING(register_thread, "Register a thread");
    PYTIMEMORY_PAPI_BINDING(get_event_code_name, "Get name of an event code");
    PYTIMEMORY_PAPI_BINDING(detach, "Detach from a process or thread");
    PYTIMEMORY_PAPI_BINDING(shutdown, "Shutdown PAPI");
    PYTIMEMORY_PAPI_BINDING(print_hw_info, "Print the hardware info");
    PYTIMEMORY_PAPI_BINDING(enable_multiplexing, "Enable multiplexing on an event set");
    PYTIMEMORY_PAPI_BINDING(destroy_event_set, "Destroy an event set");
    PYTIMEMORY_PAPI_BINDING(start, "Start an event set");
    PYTIMEMORY_PAPI_BINDING(reset, "Reset hardware events for an event set");
    PYTIMEMORY_PAPI_BINDING(assign_event_set_component,
                            "Assign a component index to an existing but empty eventset");
    PYTIMEMORY_PAPI_BINDING_1(query_event, "Query availability of an event");
    PYTIMEMORY_PAPI_BINDING_2(add_event, "Add hardware events to an event set");
    PYTIMEMORY_PAPI_BINDING_2(remove_event, "Remove hardware events from an event set");

#undef PYTIMEMORY_PAPI_BINDING

    _papi.def(
        "init", []() { tim::papi::init(); },
        "Initialize library and multiplexing (if settings.papi_multiplexing is True)");

    _papi.def(
        "init_threading", []() { tim::papi::details::init_threading(); },
        "Initialize threading support");

    _papi.def(
        "init_multiplexing", []() { tim::papi::details::init_multiplexing(); },
        "Initialize multiplexing support");

    _papi.def(
        "init_library", []() { tim::papi::details::init_library(); },
        "Initialize PAPI library");

    _papi.def(
        "attach",
        [](int event_set, unsigned long tid) {
            return tim::papi::attach(event_set, tid);
        },
        "Attach specified event set to a specific process or thread id",
        py::arg("event_set"), py::arg("tid"));

    _papi.def(
        "get_event_code", [](std::string s) { return tim::papi::get_event_code(s); },
        "Get the hardware event code");

    _papi.def(
        "create_event_set",
        [](bool enable_multiplex) {
            auto evtset = new int{};
            tim::papi::create_event_set(evtset, enable_multiplex);
            return evtset;
        },
        "Create an event set", py::arg("enable_multiplex") = true);

    _papi.def(
        "stop",
        [](int evtset, int evtsz) {
            auto vec = new std::vector<long long>(evtsz, 0);
            tim::papi::stop(evtset, vec->data());
            return vec;
        },
        "Stop an event set", py::arg("event_set"), py::arg("event_set_size"));

    _papi.def(
        "read",
        [](int evtset, int evtsz) {
            auto vec = new std::vector<long long>(evtsz, 0);
            tim::papi::read(evtset, vec->data());
            return vec;
        },
        "Read an event set", py::arg("event_set"), py::arg("event_set_size"));

    _papi.def(
        "write",
        [](int evtset, std::vector<long long> values) {
            tim::papi::write(evtset, values.data());
        },
        "Write values to an event set", py::arg("event_set"), py::arg("values"));

    _papi.def(
        "accum",
        [](int evtset, int evtsz) {
            auto vec = new std::vector<long long>(evtsz, 0);
            tim::papi::accum(evtset, vec->data());
            return vec;
        },
        "Accumulate and reset hardware events for an event set", py::arg("event_set"),
        py::arg("event_set_size"));

    _papi.def(
        "add_events",
        [](int evtset, std::vector<int> evts) {
            tim::papi::add_events(evtset, evts.data(), evts.size());
        },
        "Add hardware events to an event set", py::arg("event_set"), py::arg("events"));

    _papi.def(
        "remove_events",
        [](int evtset, std::vector<int> evts) {
            tim::papi::remove_events(evtset, evts.data(), evts.size());
        },
        "Remove hardware events from an event set", py::arg("event_set"),
        py::arg("events"));

    return _papi;
}
//
py::module
generate_cupti(py::module& _pymod)
{
    py::module _cupti = _pymod.def_submodule("cupti", "cupti query");

    auto get_available_events = [](int device) {
#if defined(TIMEMORY_USE_CUPTI)
        CUdevice cu_device;
        TIMEMORY_CUDA_DRIVER_API_CALL(cuInit(0));
        TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGet(&cu_device, device));
        return tim::cupti::available_events(cu_device);
#else
        tim::consume_parameters(device);
        return py::list{};
#endif
    };

    auto get_available_metrics = [](int device) {
#if defined(TIMEMORY_USE_CUPTI)
        CUdevice cu_device;
        TIMEMORY_CUDA_DRIVER_API_CALL(cuInit(0));
        TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGet(&cu_device, device));
        auto     ret = tim::cupti::available_metrics(cu_device);
        py::list l;
        for(const auto& itr : ret)
            l.append(itr);
        return l;
#else
        tim::consume_parameters(device);
        return py::list{};
#endif
    };

    _cupti.def("available_events", get_available_events,
               "Return the available CUPTI events", py::arg("device") = 0);
    _cupti.def("available_metrics", get_available_metrics,
               "Return the available CUPTI metric", py::arg("device") = 0);

    return _cupti;
}
//
py::module
generate_cuda(py::module& _pymod)
{
    py::module _cuda = _pymod.def_submodule("cuda", "cuda");
    /*
    py::class_<tim::cuda::stream_t> _stream(_cuda, "Stream");

    auto _create_stream = [](bool default_stream) -> tim::cuda::stream_t* {
        auto _s = new tim::cuda::stream_t{};
        if(!default_stream)
        {
            auto _ret = tim::cuda::stream_create(*_s);
            if(!_ret)
            {
                delete _s;
                _s = nullptr;
            }
        }
        return _s;
    };

    _stream.def(py::init(_create_stream), "Creates a CUDA stream",
                py::arg("default_stream") = false);
    _cuda.def("stream", _create_stream, "Create a CUDA stream");
    _cuda.def(
        "default_stream", []() { return new tim::cuda::stream_t{}; },
        "Get the default CUDA stream");
    */
    return _cuda;
}
//
}  // namespace pyapi
//
//======================================================================================//
