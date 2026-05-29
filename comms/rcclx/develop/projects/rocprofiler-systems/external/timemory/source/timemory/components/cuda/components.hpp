//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/**
 * \file timemory/components/cuda/components.hpp
 * \brief Implementation of the cuda component(s)
 */

#pragma once

#include "timemory/backends/device.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/cuda/backends.hpp"
#include "timemory/components/cuda/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"

#include <memory>

#if defined(TIMEMORY_PYBIND11_SOURCE)
#    include "pybind11/cast.h"
#    include "pybind11/pybind11.h"
#    include "pybind11/stl.h"
#endif

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
// this component extracts the time spent in GPU kernels
//
/// \struct tim::component::cuda_event
/// \brief Records the time interval between two points in a CUDA stream. Less accurate
/// than 'cupti_activity' for kernel timing but does not require linking to the CUDA
/// driver.
///
struct cuda_event : public base<cuda_event, float>
{
    struct marker
    {
        bool          valid   = true;
        bool          synced  = false;
        bool          running = false;
        cuda::event_t first   = cuda::event_t{};
        cuda::event_t second  = cuda::event_t{};

        marker() { valid = (cuda::event_create(first) && cuda::event_create(second)); }
        ~marker() = default;

        void start(cuda::stream_t& stream)
        {
            if(!valid || running)
                return;
            synced  = false;
            running = true;
            cuda::event_record(first, stream);
        }

        void stop(cuda::stream_t& stream)
        {
            if(!valid || !running)
                return;
            cuda::event_record(second, stream);
            running = false;
        }

        float sync()
        {
            if(!valid)
                return 0.0;
            if(!synced)
                cuda::event_sync(second);
            synced = true;
            return cuda::event_elapsed_time(first, second) * units::msec;
        }
    };

    using ratio_t       = std::milli;
    using value_type    = float;
    using base_type     = base<cuda_event, value_type>;
    using marker_list_t = std::vector<marker>;

    static std::string label() { return "cuda_event"; }
    static std::string description()
    {
        return "Records the time interval between two points in a CUDA stream. Less "
               "accurate than 'cupti_activity' for kernel timing";
    }
    static value_type record() { return 0.0f; }

    static uint64_t& get_batched_marker_size()
    {
        static uint64_t _instance = settings::cuda_event_batch_size();
        return _instance;
    }

    struct explicit_streams_only
    {};

public:
    TIMEMORY_DEFAULT_OBJECT(cuda_event)

    explicit cuda_event(cuda::stream_t _stream)
    : m_stream(_stream)
    {}

    float get() const noexcept
    {
        return load() / static_cast<float>(base_type::get_unit());
    }
    float get_display() const noexcept { return get(); }

    void store(explicit_streams_only, bool _v) { m_explicit_only = _v; }

    void start()
    {
        if(!m_explicit_only || m_stream != cuda::default_stream_v)
        {
            m_global_synced = false;
            m_global.start(m_stream);
        }
    }

    void stop()
    {
        for(uint64_t i = 0; i < m_num_markers; ++i)
            m_markers[i].stop(m_stream);
        if(m_current_marker == 0 && m_num_markers == 0)
            m_global.stop(m_stream);
        sync();
    }

    void sync()
    {
        if(m_current_marker == 0 && m_num_markers == 0)
        {
            if(!m_global_synced)
            {
                float tmp       = m_global.sync();
                m_global_synced = true;
                accum += tmp;
                value = tmp;
            }
        }
        else if(m_current_marker > m_synced_markers)
        {
            float tmp = 0.0;
            for(uint64_t i = m_synced_markers; i < m_num_markers; ++i, ++m_synced_markers)
                tmp += m_markers[i].sync();
            m_markers_synced = true;
            accum += tmp;
            value = tmp;
        }
    }

    void set_stream(cuda::stream_t _stream) { m_stream = _stream; }
    auto get_stream() { return m_stream; }

    void mark_begin()
    {
        m_markers_synced = false;
        m_current_marker = m_num_markers++;
        if(m_current_marker >= m_markers.size())
            append_marker_list(std::max<uint64_t>(m_marker_batch_size, 1));
        m_markers[m_current_marker].start(m_stream);
    }

    void mark_end() { m_markers[m_current_marker].stop(m_stream); }

    void mark_begin(cuda::stream_t _stream)
    {
        m_markers_synced = false;
        m_current_marker = m_num_markers++;
        if(m_current_marker >= m_markers.size())
            append_marker_list(std::max<uint64_t>(m_marker_batch_size, 1));
        m_markers[m_current_marker].start(_stream);
    }

    void mark_end(cuda::stream_t _stream) { m_markers[m_current_marker].stop(_stream); }

protected:
    void append_marker_list(const uint64_t nsize)
    {
        m_markers.reserve(m_markers.size() + nsize);
        for(uint64_t i = 0; i < nsize; ++i)
            m_markers.emplace_back(marker{});
    }

private:
    bool           m_global_synced     = false;
    bool           m_markers_synced    = false;
    bool           m_explicit_only     = false;
    uint64_t       m_synced_markers    = 0;
    uint64_t       m_current_marker    = 0;
    uint64_t       m_num_markers       = 0;
    uint64_t       m_marker_batch_size = get_batched_marker_size();
    cuda::stream_t m_stream            = cuda::default_stream_v;
    marker         m_global            = {};
    marker_list_t  m_markers           = {};

public:
#if defined(TIMEMORY_PYBIND11_SOURCE)
    //
    /// this is called by python api
    ///
    ///     Use this to add customizations to the python module. The instance
    ///     of the component is within in a variadic wrapper which is used
    ///     elsewhere to ensure that calling mark_begin(...) on a component
    ///     without that member function is not invalid
    ///
    template <template <typename...> class BundleT>
    static void configure(project::python,
                          pybind11::class_<BundleT<cuda_event>>& _pyclass)
    {
        auto _sync = [](BundleT<cuda_event>* obj) {
            obj->template get<cuda_event>()->sync();
        };
        _pyclass.def("sync", _sync, "Synchronize the event (blocking)");
    }
#endif
};
//
//======================================================================================//
//
// controls the CUDA profiler
//
/// \struct tim::component::cuda_profiler
/// \brief Control switch for a CUDA profiler running on the application. Only the
/// first call to `start()` and the last call to `stop()` actually toggle the
/// state of the external CUDA profiler when component instances are nested.
///
struct cuda_profiler
: public base<cuda_profiler, void>
, private policy::instance_tracker<cuda_profiler>
{
    using value_type   = void;
    using this_type    = cuda_profiler;
    using base_type    = base<this_type, value_type>;
    using tracker_type = policy::instance_tracker<cuda_profiler>;

    static std::string label() { return "cuda_profiler"; }
    static std::string description()
    {
        return "Control switch for a CUDA profiler running on the application";
    }

    enum class mode : short
    {
        nvp,
        csv
    };

    using config_type      = std::tuple<std::string, std::string, mode>;
    using initializer_type = std::function<config_type()>;

    static initializer_type& get_initializer()
    {
        static initializer_type _instance = []() {
            return config_type("cuda_profiler.inp", "cuda_profiler.out", mode::nvp);
        };
        return _instance;
    }

    static void global_init()
    {
#if defined(TIMEMORY_USE_CUDA)
        cudaProfilerStop();
#endif
    }

    static void global_finalize()
    {
#if defined(TIMEMORY_USE_CUDA)
        cudaProfilerStop();
#endif
    }

    static void configure()
    {
        auto _config = get_initializer()();
        configure(std::get<0>(_config), std::get<1>(_config), std::get<2>(_config));
    }

    static void configure(const std::string& _infile, const std::string& _outfile,
                          mode _mode)
    {
        static std::atomic<int32_t> _once;
        if(_once++ > 0)
            return;
#if defined(TIMEMORY_USE_CUDA) && (CUDA_VERSION < 11000)
        cudaProfilerInitialize(_infile.c_str(), _outfile.c_str(),
                               (_mode == mode::nvp) ? cudaKeyValuePair : cudaCSV);
#else
        consume_parameters(_infile, _outfile, _mode);
#endif
    }

    cuda_profiler() { configure(); }

    void start()
    {
#if defined(TIMEMORY_USE_CUDA)
        tracker_type::start();
        if(m_tot == 0)
            cudaProfilerStart();
#endif
    }

    void stop()
    {
#if defined(TIMEMORY_USE_CUDA)
        tracker_type::stop();
        if(m_tot == 0)
            cudaProfilerStop();
#endif
    }

public:
#if defined(TIMEMORY_PYBIND11_SOURCE)
    //
    /// this is called by python api
    ///
    ///     args --> pybind11::args --> pybind11::tuple
    ///     kwargs --> pybind11::kwargs --> pybind11::dict
    ///
    static void configure(project::python, pybind11::args _args, pybind11::kwargs _kwargs)
    {
        auto _config = get_initializer()();
        if(_args.size() > 0)
            std::get<0>(_config) = _args[0].cast<std::string>();
        if(_args.size() > 1)
            std::get<1>(_config) = _args[1].cast<std::string>();
        if(_args.size() > 2)
        {
            auto _m = _args[2].cast<std::string>();
            if(_m == "csv")
                std::get<2>(_config) = mode::csv;
        }
        //
        if(_kwargs)
        {
            for(auto itr : _kwargs)
            {
                if(itr.first.cast<std::string>().find("in") == 0)
                    std::get<0>(_config) = itr.second.cast<std::string>();
                else if(itr.first.cast<std::string>().find("out") == 0)
                    std::get<1>(_config) = itr.second.cast<std::string>();
                else
                {
                    auto _m = itr.second.cast<std::string>();
                    if(_m == "csv")
                        std::get<2>(_config) = mode::csv;
                }
            }
        }
        configure(std::get<0>(_config), std::get<1>(_config), std::get<2>(_config));
    }
#endif
};
//
//======================================================================================//
// adds NVTX markers
//
/// \struct tim::component::nvtx_marker
/// \brief Inserts NVTX markers with the current timemory prefix. The default color
/// scheme is a round-robin of red, blue, green, yellow, purple, cyan, pink, and
/// light_green. These colors
///
struct nvtx_marker : public base<nvtx_marker, void>
{
    using value_type = void;
    using this_type  = nvtx_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "nvtx_marker"; }
    static std::string description()
    {
        return "Generates high-level region markers for CUDA profilers";
    }
    static value_type record() {}

    static bool& use_device_sync()
    {
        static bool _instance = settings::nvtx_marker_device_sync();
        return _instance;
    }

    static void thread_init() { nvtx::name_thread(threading::get_id()); }

    nvtx_marker() = default;

    /// construct with an specific color
    explicit nvtx_marker(const nvtx::color::color_t& _color)
    : m_color(_color)
    {}

    /// construct with an specific CUDA stream
    explicit nvtx_marker(cuda::stream_t _stream)
    : m_stream(_stream)
    {}

    /// construct with an specific color and CUDA stream
    nvtx_marker(const nvtx::color::color_t& _color, cuda::stream_t _stream)
    : m_color(_color)
    , m_stream(_stream)

    {}

#if defined(TIMEMORY_PYBIND11_SOURCE)
    // explicit nvtx_marker(pybind11::object _stream)
    //: nvtx_marker(_stream.cast<cuda::stream_t>())
    //{}

    // nvtx_marker(const nvtx::color::color_t& _color, pybind11::object _stream)
    //: nvtx_marker(_color, _stream.cast<cuda::stream_t>())
    //{}
#endif

    /// start an nvtx range. Equivalent to `nvtxRangeStartEx`
    void start() { m_range_id = nvtx::range_start(get_attribute()); }

    /// stop the nvtx range. Equivalent to `nvtxRangeEnd`. Depending on
    /// `settings::nvtx_marker_device_sync()` this will either call
    /// `cudaDeviceSynchronize()` or `cudaStreamSynchronize(m_stream)` before stopping the
    /// range.
    void stop()
    {
        if(m_device_sync)
        {
            cuda::device_sync();
        }
        else
        {
            cuda::stream_sync(m_stream);
        }
        nvtx::range_stop(m_range_id);
    }

    /// asynchronously add a marker. Equivalent to `nvtxMarkA`
    void mark_begin()
    {
        nvtx::mark(TIMEMORY_JOIN("", m_prefix, "_begin_t", threading::get_id()));
    }

    /// asynchronously add a marker. Equivalent to `nvtxMarkA`
    void mark_end()
    {
        nvtx::mark(TIMEMORY_JOIN("", m_prefix, "_end_t", threading::get_id()));
    }

    /// asynchronously add a marker for a specific stream. Equivalent to `nvtxMarkA`
    void mark_begin(cuda::stream_t _stream)
    {
        nvtx::mark(TIMEMORY_JOIN("", m_prefix, "_begin_t", threading::get_id(), "_s",
                                 get_stream_id(_stream)));
    }

    /// asynchronously add a marker for a specific stream. Equivalent to `nvtxMarkA`
    void mark_end(cuda::stream_t _stream)
    {
        nvtx::mark(TIMEMORY_JOIN("", m_prefix, "_end_t", threading::get_id(), "_s",
                                 get_stream_id(_stream)));
    }

#if defined(TIMEMORY_PYBIND11_SOURCE)
    // void mark_begin(pybind11::object obj) { mark_begin(obj.cast<cuda::stream_t>()); }
    // void mark_end(pybind11::object obj) { mark_begin(obj.cast<cuda::stream_t>()); }
#endif

    /// set the current CUDA stream
    void set_stream(cuda::stream_t _stream) { m_stream = _stream; }
    /// set the current color
    void set_color(nvtx::color::color_t _color) { m_color = _color; }
    void set_prefix(const char* _prefix) { m_prefix = _prefix; }

    auto get_range_id() { return m_range_id; }
    auto get_stream() { return m_stream; }
    auto get_color() { return m_color; }

private:
    static int32_t get_stream_id(cuda::stream_t _stream)
    {
        using pair_t    = std::pair<cuda::stream_t, int32_t>;
        using map_t     = std::map<cuda::stream_t, int32_t>;
        using map_ptr_t = std::unique_ptr<map_t>;

        static thread_local map_ptr_t _instance = std::make_unique<map_t>();
        if(_instance->find(_stream) == _instance->end())
            _instance->insert(pair_t(_stream, _instance->size()));
        return _instance->find(_stream)->second;
    }

private:
    bool                     m_has_attribute = false;
    bool                     m_device_sync   = use_device_sync();
    nvtx::color::color_t     m_color         = 0;
    nvtx::event_attributes_t m_attribute     = {};
    nvtx::range_id_t         m_range_id      = 0;
    cuda::stream_t           m_stream        = 0;
    const char*              m_prefix        = nullptr;

private:
    nvtx::event_attributes_t& get_attribute()
    {
        if(!m_has_attribute)
        {
            m_has_attribute = true;
            if(settings::debug())
            {
                std::stringstream ss;
                ss << "[nvtx_marker]> Creating NVTX marker with label: \"" << m_prefix
                   << "\" and color " << std::hex << m_color << "...";
                std::cout << ss.str() << std::endl;
            }
            m_attribute = nvtx::init_marker(m_prefix, m_color);
        }
        return m_attribute;
    }

public:
#if defined(TIMEMORY_PYBIND11_SOURCE)
    //
    /// this is called by python api
    ///
    ///     Use this to add customizations to the python module. The instance
    ///     of the component is within in a variadic wrapper which is used
    ///     elsewhere to ensure that calling mark_begin(...) on a component
    ///     without that member function is not invalid
    ///
    template <template <typename...> class BundleT>
    static void configure(project::python,
                          pybind11::class_<BundleT<nvtx_marker>>& _pyclass)
    {
        _pyclass.def_property_static(
            "use_device_sync", [](pybind11::object) { return use_device_sync(); },
            [](pybind11::object, bool v) { use_device_sync() = v; },
            "Configure CudaEvent to use cudaSynchronize() vs. cudaStreamSychronize(...)");

        // add nvtx colors
        pybind11::enum_<nvtx::color::color_idx> _pyattr(_pyclass, "color", "NVTX colors");
        _pyattr.value("red", nvtx::color::red_idx)
            .value("blue", nvtx::color::blue_idx)
            .value("green", nvtx::color::green_idx)
            .value("yellow", nvtx::color::yellow_idx)
            .value("purple", nvtx::color::purple_idx)
            .value("cyan", nvtx::color::cyan_idx)
            .value("pink", nvtx::color::pink_idx)
            .value("light_green", nvtx::color::light_green_idx);
        _pyattr.export_values();

        auto _set_color = [](BundleT<nvtx_marker>* obj, nvtx::color::color_t arg) {
            obj->template get<nvtx_marker>()->set_color(arg);
        };
        auto _get_color = [](BundleT<nvtx_marker>* obj) {
            return obj->template get<nvtx_marker>()->get_color();
        };
        _pyclass.def("set_color", _set_color, "Set the color");
        _pyclass.def("get_color", _get_color, "Return the color");
    }
#endif
};
//
//======================================================================================//
//
//
//======================================================================================//
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
