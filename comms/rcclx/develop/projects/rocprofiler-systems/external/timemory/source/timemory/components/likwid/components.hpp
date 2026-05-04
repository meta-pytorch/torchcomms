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
 * \file timemory/components/likwid/components.hpp
 * \brief Implementation of the likwid component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/likwid/backends.hpp"
#include "timemory/components/likwid/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//======================================================================================//
//
// perfmon marker (CPU)
//
//======================================================================================//
//
/// \struct tim::component::likwid_marker
/// \brief Provides likwid perfmon marker forwarding. Requires \def LIKWID_PERFMON to
/// be defined before including <likwid-marker.h>
struct likwid_marker : public base<likwid_marker, void>
{
    // timemory component api
    using value_type = void;
    using this_type  = likwid_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "likwid_marker"; }
    static std::string description() { return "LIKWID perfmon (CPU) marker forwarding"; }
    static value_type  record() {}

    static void global_init()
    {
#if defined(TIMEMORY_USE_LIKWID_PERFMON)
        likwid_markerInit();
#endif
    }

    static void thread_init()
    {
#if defined(TIMEMORY_USE_LIKWID_PERFMON)
        likwid_markerThreadInit();
#endif
    }

    static void global_finalize()
    {
#if defined(TIMEMORY_USE_LIKWID_PERFMON)
        likwid_markerClose();
#endif
    }

    static void next()
    {
#if defined(TIMEMORY_USE_LIKWID_PERFMON)
        likwid_markerNextGroup();
#endif
    }

    TIMEMORY_DEFAULT_OBJECT(likwid_marker)

    void start()
    {
#if defined(TIMEMORY_USE_LIKWID_PERFMON)
        likwid_markerStartRegion(m_prefix);
#endif
    }

    void stop()
    {
#if defined(TIMEMORY_USE_LIKWID_PERFMON)
        likwid_markerStopRegion(m_prefix);
#endif
    }

    void reset_region()
    {
#if defined(TIMEMORY_USE_LIKWID_PERFMON)
        likwid_markerResetRegion(m_prefix);
#endif
    }

    void register_marker()
    {
#if defined(TIMEMORY_USE_LIKWID_PERFMON)
        likwid_markerRegisterRegion(m_prefix);
#endif
    }

    likwid_data get() const  // NOLINT
    {
        likwid_data _data{};
#if defined(TIMEMORY_USE_LIKWID_PERFMON)
        likwid_markerGetRegion(m_prefix, &_data.nevents, _data.events.data(), &_data.time,
                               &_data.count);
        _data.events.resize(_data.nevents);
#endif
        return _data;
    }

    void set_prefix(const char* _prefix)
    {
        m_prefix = _prefix;
        register_marker();
    }

private:
    //----------------------------------------------------------------------------------//
    //
    // Member Variables
    //
    //----------------------------------------------------------------------------------//
    const char* m_prefix = nullptr;
};
//
//======================================================================================//
//
// nvmon marker (GPU)
//
//======================================================================================//
//
/// \struct tim::component::likwid_nvmarker
/// \brief Provides likwid nvmon marker forwarding. Requires \def LIKWID_NVMON to
/// be defined before including <likwid-marker.h>
struct likwid_nvmarker : public base<likwid_nvmarker, void>
{
    // timemory component api
    using value_type = void;
    using this_type  = likwid_nvmarker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "likwid_nvmarker"; }
    static std::string description() { return "LIKWID nvmon (GPU) marker forwarding"; }
    static value_type  record() {}

    static void global_init()
    {
#if defined(TIMEMORY_USE_LIKWID_NVMON)
        likwid_gpuMarkerInit();
#endif
    }

    static void global_finalize()
    {
#if defined(TIMEMORY_USE_LIKWID_NVMON)
        likwid_gpuMarkerClose();
#endif
    }

    TIMEMORY_DEFAULT_OBJECT(likwid_nvmarker)

    void start()
    {
#if defined(TIMEMORY_USE_LIKWID_NVMON)
        likwid_gpuMarkerStartRegion(m_prefix);
#endif
    }

    void stop()
    {
#if defined(TIMEMORY_USE_LIKWID_NVMON)
        likwid_gpuMarkerStopRegion(m_prefix);
#endif
    }

    void reset()
    {
#if defined(TIMEMORY_USE_LIKWID_NVMON)
        likwid_gpuMarkerResetRegion(m_prefix);
#endif
    }

    void register_marker()
    {
#if defined(TIMEMORY_USE_LIKWID_NVMON)
        likwid_gpuMarkerRegisterRegion(m_prefix);
#endif
    }

    /*likwid_nvdata get() const
    {
        likwid_nvdata _data{};
#if defined(TIMEMORY_USE_LIKWID_NVMON)
        likwid_gpuMarkerGetRegion(m_prefix, &_data.ndevices, &_data.nevents,
                                  &_data.events[0]), &_data.time[0],
                                  &_data.count[0]);
        _data.time.resize(_data.ndevices);
        _data.count.resize(_data.ndevices);
        _data.events.resize(_data.ndevices * _data.nevents);
#endif
        return _data;
     }*/

    void set_prefix(const char* _prefix)
    {
        m_prefix = _prefix;
        register_marker();
    }

private:
    //----------------------------------------------------------------------------------//
    //
    // Member Variables
    //
    //----------------------------------------------------------------------------------//
    const char* m_prefix = nullptr;
};
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
