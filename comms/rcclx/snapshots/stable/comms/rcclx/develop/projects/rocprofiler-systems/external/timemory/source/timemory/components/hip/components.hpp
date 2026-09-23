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
 * \file timemory/components/hip/components.hpp
 * \brief Implementation of the hip component(s)
 */

#pragma once

#include "timemory/backends/device.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/hip/backends.hpp"
#include "timemory/components/hip/types.hpp"
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
/// \struct tim::component::hip_event
/// \brief Records the time interval between two points in a HIP stream. Less accurate
/// than 'cupti_activity' for kernel timing but does not require linking to the HIP
/// driver.
///
struct hip_event : public base<hip_event, float>
{
    struct marker
    {
        bool         valid   = true;
        bool         synced  = false;
        bool         running = false;
        hip::event_t first   = hip::event_t{};
        hip::event_t second  = hip::event_t{};

        marker() { valid = (hip::event_create(first) && hip::event_create(second)); }
        ~marker() = default;

        void start(hip::stream_t& stream)
        {
            if(!valid || running)
                return;
            synced  = false;
            running = true;
            hip::event_record(first, stream);
        }

        void stop(hip::stream_t& stream)
        {
            if(!valid || !running)
                return;
            hip::event_record(second, stream);
            running = false;
        }

        float sync()
        {
            if(!valid)
                return 0.0;
            if(!synced)
                hip::event_sync(second);
            synced = true;
            return hip::event_elapsed_time(first, second) * units::msec;
        }
    };

    using value_type    = float;
    using base_type     = base<hip_event, value_type>;
    using marker_list_t = std::vector<marker>;

    static std::string label() { return "hip_event"; }
    static std::string description()
    {
        return "Records the time interval between two points in a HIP stream. Less "
               "accurate than 'roctracer' for kernel timing";
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
    TIMEMORY_DEFAULT_OBJECT(hip_event)

    explicit hip_event(hip::stream_t _stream)
    : m_stream(_stream)
    {}

    float get_display() const { return get(); }

    float get() const { return load() / static_cast<float>(base_type::get_unit()); }

    void store(explicit_streams_only, bool _v) { m_explicit_only = _v; }

    void start()
    {
        if(!m_explicit_only || m_stream != hip::default_stream_v)
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

    void set_stream(hip::stream_t _stream) { m_stream = _stream; }
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

    void mark_begin(hip::stream_t _stream)
    {
        m_markers_synced = false;
        m_current_marker = m_num_markers++;
        if(m_current_marker >= m_markers.size())
            append_marker_list(std::max<uint64_t>(m_marker_batch_size, 1));
        m_markers[m_current_marker].start(_stream);
    }

    void mark_end(hip::stream_t _stream) { m_markers[m_current_marker].stop(_stream); }

protected:
    void append_marker_list(const uint64_t nsize)
    {
        m_markers.reserve(m_markers.size() + nsize);
        for(uint64_t i = 0; i < nsize; ++i)
            m_markers.emplace_back(marker{});
    }

private:
    bool          m_global_synced     = false;
    bool          m_markers_synced    = false;
    bool          m_explicit_only     = false;
    uint64_t      m_synced_markers    = 0;
    uint64_t      m_current_marker    = 0;
    uint64_t      m_num_markers       = 0;
    uint64_t      m_marker_batch_size = get_batched_marker_size();
    hip::stream_t m_stream            = hip::default_stream_v;
    marker        m_global            = {};
    marker_list_t m_markers           = {};

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
    static void configure(project::python, pybind11::class_<BundleT<hip_event>>& _pyclass)
    {
        auto _sync = [](BundleT<hip_event>* obj) {
            obj->template get<hip_event>()->sync();
        };
        _pyclass.def("sync", _sync, "Synchronize the event (blocking)");
    }
#endif
};
//
//======================================================================================//
// adds ROCTX markers
//
/// \struct tim::component::roctx_marker
/// \brief Inserts ROCTX markers with the current timemory prefix.
///
struct roctx_marker : public base<roctx_marker, void>
{
    using value_type = void;
    using this_type  = roctx_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "roctx_marker"; }
    static std::string description()
    {
        return "Generates high-level region markers for HIP profilers";
    }
    static value_type record() {}

    static bool& use_device_sync()
    {
        static bool _instance = settings::nvtx_marker_device_sync();
        return _instance;
    }

    TIMEMORY_DEFAULT_OBJECT(roctx_marker)

    /// construct with an specific HIP stream
    explicit roctx_marker(hip::stream_t _stream)
    : m_stream(_stream)
    {}

    /// start an roctx range. Equivalent to `roctxRangeStartEx`
    void start() { m_range_id = roctx::range_start(m_prefix); }

    /// stop the roctx range. Equivalent to `roctxRangeEnd`. Depending on
    /// `settings::roctx_marker_device_sync()` this will either call
    /// `hipDeviceSynchronize()` or `hipStreamSynchronize(m_stream)` before stopping the
    /// range.
    void stop()
    {
        if(use_device_sync())
        {
            hip::device_sync();
        }
        else
        {
            hip::stream_sync(m_stream);
        }
        roctx::range_stop(m_range_id);
    }

    /// asynchronously add a marker. Equivalent to `roctxMarkA`
    void mark_begin()
    {
        roctx::mark(TIMEMORY_JOIN("", m_prefix, "_begin_t", threading::get_id()));
    }

    /// asynchronously add a marker. Equivalent to `roctxMarkA`
    void mark_end()
    {
        roctx::mark(TIMEMORY_JOIN("", m_prefix, "_end_t", threading::get_id()));
    }

    /// asynchronously add a marker for a specific stream. Equivalent to `roctxMarkA`
    void mark_begin(hip::stream_t _stream)
    {
        roctx::mark(TIMEMORY_JOIN("", m_prefix, "_begin_t", threading::get_id(), "_s",
                                  get_stream_id(_stream)));
    }

    /// asynchronously add a marker for a specific stream. Equivalent to `roctxMarkA`
    void mark_end(hip::stream_t _stream)
    {
        roctx::mark(TIMEMORY_JOIN("", m_prefix, "_end_t", threading::get_id(), "_s",
                                  get_stream_id(_stream)));
    }

    /// set the current HIP stream
    void set_stream(hip::stream_t _stream) { m_stream = _stream; }
    /// set the label
    void set_prefix(const char* _prefix) { m_prefix = _prefix; }

    auto get_range_id() { return m_range_id; }
    auto get_stream() { return m_stream; }

private:
    static int32_t get_stream_id(hip::stream_t _stream)
    {
        using pair_t    = std::pair<hip::stream_t, int32_t>;
        using map_t     = std::map<hip::stream_t, int32_t>;
        using map_ptr_t = std::unique_ptr<map_t>;

        static thread_local map_ptr_t _instance = std::make_unique<map_t>();
        if(_instance->find(_stream) == _instance->end())
            _instance->insert(pair_t(_stream, _instance->size()));
        return _instance->find(_stream)->second;
    }

private:
    roctx::range_id_t m_range_id = 0;
    hip::stream_t     m_stream   = 0;
    const char*       m_prefix   = nullptr;

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
                          pybind11::class_<BundleT<roctx_marker>>& _pyclass)
    {
        _pyclass.def_property_static(
            "use_device_sync", [](pybind11::object) { return use_device_sync(); },
            [](pybind11::object, bool v) { use_device_sync() = v; },
            "Configure CudaEvent to use hipSynchronize() vs. hipStreamSychronize(...)");
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
