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

#if !defined(TIMEMORY_PYHARDWARE_COUNTERS_SOURCE)
#    define TIMEMORY_PYHARDWARE_COUNTERS_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/backends/hardware_counters.hpp"
#include "timemory/backends/papi.hpp"

#include <pybind11/pytypes.h>

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"
#else
#    include "timemory/backends/types/cupti.hpp"
#endif

struct hwcounter_presets
{};

//======================================================================================//
//
namespace pyhardware_counters
{
py::module
generate(py::module& _pymod)
{
    using iface_t      = tim::hardware_counters::api::type;
    using iface_enum_t = py::enum_<iface_t>;
    using info_t       = tim::hardware_counters::info;

    py::module _hw = _pymod.def_submodule("hardware_counters",
                                          "Hardware counter identifiers and info");

    iface_enum_t _iface(_hw, "api");
    _iface.value("papi", tim::hardware_counters::api::papi)
        .value("cuda", tim::hardware_counters::api::cupti)
        .value("unknown", tim::hardware_counters::api::cupti)
        .export_values();

    auto _init = [](std::string _sym, iface_t _if) -> info_t* {
        for(const auto& itr : tim::hardware_counters::get_info())
        {
            if((_sym == itr.symbol() && _if == itr.iface()) ||
               (_sym == itr.symbol() && _if == tim::hardware_counters::api::unknown))
                return new info_t(itr);
        }
        if(_if == tim::hardware_counters::api::papi)
            return new info_t(tim::papi::get_hwcounter_info(_sym));

        return nullptr;
    };

    py::class_<info_t> _info(_hw, "Info");
    _info.def(py::init(_init), "Get a hardware counter identifier");
    _info.def("api", py::overload_cast<>(&info_t::iface, py::const_),
              "Library API used for accessing hardware counter");
    _info.def("symbol", py::overload_cast<>(&info_t::symbol, py::const_),
              "Symbol for the hardware counter");
    _info.def("short_description",
              py::overload_cast<>(&info_t::short_description, py::const_),
              "Short description of the hardware counter");
    _info.def("long_description",
              py::overload_cast<>(&info_t::long_description, py::const_),
              "Long description of the hardware counter");
    _info.def("available", py::overload_cast<>(&info_t::available, py::const_),
              "Whether the hardware counter is available or not");

    py::class_<hwcounter_presets> _preset(_hw, "preset",
                                          "Known hardware counter identifiers and info");

    auto _generate_presets = [&_preset]() {
    // tim::cupti::device_t device;
#if defined(TIMEMORY_USE_CUPTI)
    // TIMEMORY_CUDA_DRIVER_API_CALL(cuInit(0));
    // TIMEMORY_CUDA_DRIVER_API_CALL(
    //    cuDeviceGet(&device, tim::settings::cupti_device()));
#endif
        // auto _cupti_events  = tim::cupti::available_events_info(device);
        // auto _cupti_metrics = tim::cupti::available_metrics_info(device);
        auto _papi_events = tim::papi::available_events_info();

        auto _process_counters = [](auto& _events, int32_t _offset) {
            for(auto& itr : _events)
                itr.offset() += _offset;
            return static_cast<int32_t>(_events.size());
        };

        int32_t _offset = 0;
        _offset += _process_counters(_papi_events, _offset);
        // _offset += _process_counters(_cupti_events, _offset);
        // _offset += _process_counters(_cupti_metrics, _offset);

        // for(auto&& fitr : { _papi_events, _cupti_events, _cupti_metrics })
        for(auto&& fitr : { _papi_events })
            for(auto&& itr : fitr)
            {
                tim::hardware_counters::get_info().emplace_back(std::move(itr));
            }

        for(const auto& itr : tim::hardware_counters::get_info())
        {
            // std::string _help =
            //   "Built-in identifier for " + itr.symbol() + " : " +
            //   itr.long_description();
            if(tim::settings::debug() && tim::settings::verbose() > 1)
                TIMEMORY_PRINT_HERE("%s", itr.python_symbol().c_str());
            _preset.def_property_readonly_static(
                itr.python_symbol().c_str(),
                [itr](py::object) { return new info_t(itr); });
        }
    };

    _generate_presets();
    // _hw.def("generate_presets", _generate_presets, "Get attribute");
    return _hw;
}
//
}  // namespace pyhardware_counters
//
//======================================================================================//
