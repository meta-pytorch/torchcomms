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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#ifndef TIMEMORY_COMPONENTS_PERFETTO_PERFETTO_CPP_
#define TIMEMORY_COMPONENTS_PERFETTO_PERFETTO_CPP_

#include "timemory/components/perfetto/types.hpp"

#if !defined(TIMEMORY_PERFETTO_COMPONENT_HEADER_MODE) ||                                 \
    TIMEMORY_PERFETTO_COMPONENT_HEADER_MODE < 1
#    include "timemory/components/perfetto/perfetto.hpp"
#    define TIMEMORY_COMPONENT_PERFETTO_INLINE
#else
#    define TIMEMORY_COMPONENT_PERFETTO_INLINE inline
#endif

#include "timemory/defines.h"
#include "timemory/manager.hpp"
#include "timemory/operations/types/decode.hpp"
#include "timemory/operations/types/file_output_message.hpp"
#include "timemory/operations/types/storage_initializer.hpp"
#include "timemory/settings.hpp"

#include <string>

namespace tim
{
namespace component
{
TIMEMORY_COMPONENT_PERFETTO_INLINE
perfetto_trace::initializer_t&
perfetto_trace::get_initializer()
{
    static initializer_t _v = []() {
#if defined(TIMEMORY_USE_PERFETTO)
        TIMEMORY_DEBUG_PRINT_HERE("Initializing %s", label().c_str());
        if(get_config().session)
            return;

        auto& args     = get_tracing_init_args();
        auto& _session = get_config().session;
        auto  shmem_size_hint =
            tim::get_env<size_t>("TIMEMORY_PERFETTO_SHMEM_SIZE_HINT_KB", 40960);
        auto buffer_size =
            tim::get_env<size_t>("TIMEMORY_PERFETTO_BUFFER_SIZE_KB", 1024000);

        ::perfetto::TraceConfig                   cfg{};
        ::perfetto::protos::gen::TrackEventConfig track_event_cfg{};

        cfg.add_buffers()->set_size_kb(buffer_size);
        auto* ds_cfg = cfg.add_data_sources()->mutable_config();
        ds_cfg->set_name("track_event");
        ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

        args.shmem_size_hint_kb = shmem_size_hint;

        if(get_config().in_process)
            args.backends |= ::perfetto::kInProcessBackend;

        if(get_config().system_backend)
            args.backends |= ::perfetto::kSystemBackend;

        ::perfetto::Tracing::Initialize(args);
        ::perfetto::TrackEvent::Register();

        _session = ::perfetto::Tracing::NewTrace();
        _session->Setup(cfg);
        _session->StartBlocking();

        /*
        ::perfetto::ProcessTrack _process_track        =
        ::perfetto::ProcessTrack::Current();
        ::perfetto::protos::gen::TrackDescriptor _desc = _process_track.Serialize();
        auto& _cmdline = settings::instance()->get_command_line();
        if(!_cmdline.empty())
        {
            auto _name = _cmdline.at(0);
            auto _pos  = _name.find_last_of('/');
            if(_pos != std::string::npos && _pos + 1 < _name.size())
                _name = _name.substr(_pos + 1);
            _desc.mutable_process()->set_process_name(_name);
        }
        else
        {
            std::string _category =
                ::tim::trait::perfetto_category<TIMEMORY_PERFETTO_API>::value;
            _desc.mutable_process()->set_process_name(TIMEMORY_JOIN("-", label(),
        _category));
        }
        ::perfetto::TrackEvent::SetTrackDescriptor(_process_track, _desc);
        */
#endif
    };
    return _v;
}

TIMEMORY_COMPONENT_PERFETTO_INLINE
perfetto_trace::finalizer_t&
perfetto_trace::get_finalizer()
{
    static finalizer_t _v = []() {
#if defined(TIMEMORY_USE_PERFETTO)
        TIMEMORY_DEBUG_PRINT_HERE("Finalizing %s", label().c_str());
        dmp::barrier();

        // Make sure the last event is closed for this example.
        ::perfetto::TrackEvent::Flush();

        auto& _session = get_config().session;
        if(!_session)
            return;

        // Stop tracing and read the trace data.
        _session->StopBlocking();
        std::vector<char> trace_data(_session->ReadTraceBlocking());

        auto        _rank     = dmp::rank();
        std::string _label    = label();
        std::string _category = trait::perfetto_category<TIMEMORY_PERFETTO_API>::value;
        std::string _fname    = settings::compose_output_filename(
            TIMEMORY_JOIN('_', _label, _category), "proto");
        // output to a unique filename per rank if DMP is initialized
        if(settings::verbose() >= 0)
            operation::file_output_message<perfetto_trace>{}(_fname,
                                                             std::string{ _label });

        manager::instance()->add_file_output("binary", _label, _fname);

        // Write the result into a file.
        // Note: To save memory with longer traces, you can tell Perfetto to write
        // directly into a file by passing a file descriptor into Setup() above.
        std::ofstream ofs{};
        ofs.open(_fname.c_str(), std::ios::out | std::ios::binary);
        ofs.write(&trace_data[0], trace_data.size());
        ofs.close();

        dmp::barrier();
        if(_rank == 0)
            puts("");
#endif
    };
    return _v;
}

TIMEMORY_COMPONENT_PERFETTO_INLINE
std::string
perfetto_trace::label()
{
    return "perfetto_trace";
}

TIMEMORY_COMPONENT_PERFETTO_INLINE
std::string
perfetto_trace::description()
{
    return "Provides Perfetto Tracing SDK: system profiling, app tracing and trace "
           "analysis";
}

TIMEMORY_COMPONENT_PERFETTO_INLINE
perfetto_trace::config&
perfetto_trace::get_config()
{
    static auto _config = perfetto_trace::config{};
    return _config;
}

TIMEMORY_COMPONENT_PERFETTO_INLINE
void
perfetto_trace::global_init()
{
    if(get_initializer())
        get_initializer()();
}

TIMEMORY_COMPONENT_PERFETTO_INLINE
void
perfetto_trace::global_finalize()
{
    if(get_finalizer())
        get_finalizer()();
}

TIMEMORY_COMPONENT_PERFETTO_INLINE
perfetto_trace::TracingInitArgs&
perfetto_trace::get_tracing_init_args()
{
    static auto _args = perfetto_trace::TracingInitArgs{};
    return _args;
}

TIMEMORY_COMPONENT_PERFETTO_INLINE
void
perfetto_trace::start()
{
    if(m_prefix)
        backend::perfetto::trace_event_start(
            policy::perfetto_category<TIMEMORY_PERFETTO_API>{}, m_prefix);
}

TIMEMORY_COMPONENT_PERFETTO_INLINE
void
perfetto_trace::start(const char* _label)
{
    backend::perfetto::trace_event_start(
        policy::perfetto_category<TIMEMORY_PERFETTO_API>{}, _label);
}

TIMEMORY_COMPONENT_PERFETTO_INLINE
void
perfetto_trace::stop()
{
    backend::perfetto::trace_event_stop(
        policy::perfetto_category<TIMEMORY_PERFETTO_API>{});
}

TIMEMORY_COMPONENT_PERFETTO_INLINE
void
perfetto_trace::set_prefix(const char* _prefix)
{
    if(_prefix && m_prefix != _prefix)
    {
        m_prefix = _prefix;
    }
}

}  // namespace component
}  // namespace tim

#endif  // TIMEMORY_COMPONENTS_PERFETTO_PERFETTO_CPP_
